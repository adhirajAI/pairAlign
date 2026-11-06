#!/usr/bin/env python3
"""Generate anchor/positive/negative tokenizations for consistency probes.

Negative construction is automatic by default:

1. PHN content-negative mode
   Used only when every audio file has a matching .PHN file. A negative
   segment is sampled from another file such that its phone-overlap units
   do not intersect the anchor's phone-overlap units.

2. Temporal-rearrangement mode
   Used when PHN files are unavailable. The anchor waveform is divided into
   contiguous chunks and permuted with a derangement, so every chunk moves
   to a different temporal position. No external-content negative is used.

The output CSV contains token sequences and edit-distance diagnostics for:
    anchor vs positive  -> augmentation/content consistency
    anchor vs negative  -> content or temporal-order sensitivity
    positive vs negative
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import warnings
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from lightning.pytorch import seed_everything
from tqdm.auto import tqdm

import torchaudio
import torchaudio.functional as AF
import torchaudio.transforms as T
import yaml

from train import AudioTokenizer, AudioTokenizerARPrior, AudioTokenizerAR_FINAL


# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------


def path_to_file_id(path: str) -> str:
    return os.path.splitext(os.path.basename(str(path)))[0]


def phn_path_from_audio(audio_path: str) -> str:
    """Return the canonical PHN path used in error messages."""
    return os.path.splitext(audio_path)[0] + ".PHN"


def find_existing_phn_path(audio_path: str) -> Optional[str]:
    """Find a PHN sidecar while tolerating common case variants."""
    stem = os.path.splitext(audio_path)[0]
    for extension in (".PHN", ".phn", ".Phn", ".pHN"):
        candidate = stem + extension
        if os.path.isfile(candidate):
            return candidate
    return None


def resolve_config_path(root_dir: str, config_arg: str) -> str:
    if os.path.isabs(config_arg):
        path = config_arg
    else:
        path = os.path.join(root_dir, "config", config_arg)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Config not found: {path}")
    return path


def resolve_dataset_csv(root_dir: str, config_path: str, csv_arg: str) -> str:
    candidates = [
        csv_arg,
        os.path.join(root_dir, csv_arg),
        os.path.join(os.path.dirname(config_path), csv_arg),
    ]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return os.path.abspath(candidate)
    raise FileNotFoundError(
        "Dataset CSV not found. Tried:\n  " + "\n  ".join(candidates)
    )


def read_audio_paths(csv_path: str, path_col: str = "path") -> List[str]:
    paths: List[str] = []
    with open(csv_path, "r", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or path_col not in reader.fieldnames:
            raise ValueError(
                f"CSV must contain column `{path_col}`. Found: {reader.fieldnames}"
            )
        for row in reader:
            raw = str(row[path_col]).strip()
            if not raw:
                continue
            path = os.path.expanduser(raw)
            if not os.path.isabs(path):
                path = os.path.join(os.path.dirname(csv_path), path)
            path = os.path.abspath(path)
            if not os.path.isfile(path):
                warnings.warn(f"Skipping missing audio file: {path}")
                continue
            paths.append(path)

    if not paths:
        raise ValueError(f"No valid audio paths found in {csv_path}")
    return paths


def load_audio_mono(path: str) -> Tuple[torch.Tensor, int]:
    wav, sr = torchaudio.load(path)
    wav = wav.float()
    if wav.ndim != 2:
        raise ValueError(f"Expected waveform [channels, time], got {wav.shape}: {path}")
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    return wav, int(sr)


def ensure_sample_rate(
    wav: torch.Tensor,
    sample_rate: int,
    target_sample_rate: int,
) -> torch.Tensor:
    if sample_rate == target_sample_rate:
        return wav
    return AF.resample(wav, orig_freq=sample_rate, new_freq=target_sample_rate)


def deterministic_crop_start(
    num_samples: int,
    target_len: int,
    seed: int,
    item_index: int,
    split: str,
) -> int:
    if num_samples <= target_len:
        return 0

    # A deterministic random crop for train and a center crop for validation/test.
    if split == "train":
        rng = random.Random(seed + 1_000_003 * item_index)
        return rng.randint(0, num_samples - target_len)
    return (num_samples - target_len) // 2


def pad_or_trim(wav: torch.Tensor, target_len: int) -> Tuple[torch.Tensor, int]:
    valid_len = min(int(wav.size(-1)), target_len)
    if wav.size(-1) < target_len:
        wav = F.pad(wav, (0, target_len - wav.size(-1)))
    elif wav.size(-1) > target_len:
        wav = wav[..., :target_len]
    return wav.contiguous(), valid_len


def serialize_tokens(tokens: Sequence[int]) -> str:
    return " ".join(str(int(token)) for token in tokens)


def normalise_token_batch(output: Any) -> List[List[int]]:
    """Convert common model.tokenize outputs into List[List[int]]."""

    def to_int_list(item: Any) -> List[int]:
        if isinstance(item, torch.Tensor):
            return [int(x) for x in item.detach().cpu().reshape(-1).tolist()]
        if isinstance(item, (list, tuple)):
            return [int(x) for x in item]
        raise TypeError(f"Unsupported token sequence type: {type(item)}")

    if isinstance(output, torch.Tensor):
        output = output.detach().cpu()
        if output.ndim == 1:
            return [to_int_list(output)]
        return [to_int_list(output[index]) for index in range(output.size(0))]

    if isinstance(output, (list, tuple)):
        return [to_int_list(item) for item in output]

    if isinstance(output, dict):
        for key in ("tokens", "token_ids", "sequences", "seqs"):
            if key in output:
                return normalise_token_batch(output[key])

    raise TypeError(f"Unsupported tokenize() output type: {type(output)}")


# -----------------------------------------------------------------------------
# Token edit-distance diagnostics
# -----------------------------------------------------------------------------


try:
    from rapidfuzz.distance import Levenshtein as RapidLevenshtein

    def edit_distance(a: Sequence[int], b: Sequence[int]) -> int:
        return int(RapidLevenshtein.distance(a, b))

except Exception:

    def edit_distance(a: Sequence[int], b: Sequence[int]) -> int:
        n, m = len(a), len(b)
        if n == 0:
            return m
        if m == 0:
            return n
        if m > n:
            a, b, n, m = b, a, m, n

        previous = list(range(m + 1))
        current = [0] * (m + 1)
        for i in range(1, n + 1):
            current[0] = i
            for j in range(1, m + 1):
                substitution_cost = 0 if a[i - 1] == b[j - 1] else 1
                current[j] = min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + substitution_cost,
                )
            previous, current = current, previous
        return previous[m]


def normalized_edit_distance(a: Sequence[int], b: Sequence[int]) -> float:
    return edit_distance(a, b) / max(len(a), len(b), 1)


def safe_ratio(numerator: float, denominator: float) -> Optional[float]:
    return float(numerator) / float(denominator) if denominator != 0 else None


# -----------------------------------------------------------------------------
# PHN parsing and segment metadata
# -----------------------------------------------------------------------------


SILENCE_PHONES_DEFAULT = {"h#", "pau", "epi"}


@dataclass(frozen=True)
class SegmentMeta:
    dataset_index: int
    path: str
    file_id: str
    original_sample_rate: int
    original_num_frames: int
    resampled_num_samples: int
    crop_start: int
    crop_end: int
    valid_samples: int
    phones: Tuple[str, ...]
    overlap_units: frozenset


def read_phn_file(phn_path: str) -> List[Tuple[int, int, str]]:
    intervals: List[Tuple[int, int, str]] = []
    with open(phn_path, "r") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 3:
                raise ValueError(
                    f"Malformed PHN line {line_number} in {phn_path}: {line!r}"
                )
            start, end, phone = parts
            intervals.append((int(start), int(end), phone))
    return intervals


def phone_overlap_units(
    phones: Sequence[str],
    unit: str,
    ngram_size: int,
) -> Set[Any]:
    """Return units whose intersection must be empty for a valid PHN negative.

    unit='phone':
        Literal strict mode. Anchor and negative may not share any individual
        phone label. This can be too restrictive for long speech segments.

    unit='ngram':
        Sequence-overlap mode. Anchor and negative may share individual phones,
        but may not share any contiguous phone n-gram of length ngram_size.
        This is usually the practical interpretation of "no overlap in phone
        sequence" for 3-second speech segments.
    """
    if unit == "phone":
        return set(phones)
    if unit == "ngram":
        if ngram_size <= 0:
            raise ValueError("phone_ngram must be positive")
        if len(phones) < ngram_size:
            # Preserve strictness for short sequences instead of returning an
            # empty set, which would allow every candidate.
            return {tuple(phones)} if phones else set()
        return {
            tuple(phones[i : i + ngram_size])
            for i in range(len(phones) - ngram_size + 1)
        }
    raise ValueError(f"Unknown phone overlap unit: {unit}")


def phones_for_crop(
    audio_path: str,
    original_sample_rate: int,
    target_sample_rate: int,
    crop_start_target: int,
    crop_end_target: int,
    silence_phones: Set[str],
) -> Tuple[str, ...]:
    phn_path = find_existing_phn_path(audio_path)
    if phn_path is None:
        raise FileNotFoundError(f"Missing PHN sidecar for: {audio_path}")
    phones: List[str] = []

    for start_original, end_original, phone in read_phn_file(phn_path):
        start_target = int(round(start_original * target_sample_rate / original_sample_rate))
        end_target = int(round(end_original * target_sample_rate / original_sample_rate))

        overlaps_crop = (
            end_target > crop_start_target and start_target < crop_end_target
        )
        if overlaps_crop and phone not in silence_phones:
            phones.append(phone)

    return tuple(phones)


def build_segment_metadata(
    paths: Sequence[str],
    target_sample_rate: int,
    context_sec: float,
    seed: int,
    split: str,
    include_phn: bool,
    silence_phones: Set[str],
    phone_overlap_unit: str,
    phone_ngram: int,
) -> List[SegmentMeta]:
    target_len = int(round(context_sec * target_sample_rate))
    metadata: List[SegmentMeta] = []

    for index, path in enumerate(
        tqdm(paths, desc="Reading audio/PHN metadata", unit="file")
    ):
        info = torchaudio.info(path)
        original_sr = int(info.sample_rate)
        original_frames = int(info.num_frames)
        resampled_samples = int(round(original_frames * target_sample_rate / original_sr))

        crop_start = deterministic_crop_start(
            num_samples=resampled_samples,
            target_len=target_len,
            seed=seed,
            item_index=index,
            split=split,
        )
        crop_end = crop_start + target_len
        valid_samples = max(0, min(target_len, resampled_samples - crop_start))

        phones: Tuple[str, ...] = ()
        units: Set[Any] = set()
        if include_phn:
            phones = phones_for_crop(
                audio_path=path,
                original_sample_rate=original_sr,
                target_sample_rate=target_sample_rate,
                crop_start_target=crop_start,
                crop_end_target=crop_end,
                silence_phones=silence_phones,
            )
            units = phone_overlap_units(
                phones=phones,
                unit=phone_overlap_unit,
                ngram_size=phone_ngram,
            )

        metadata.append(
            SegmentMeta(
                dataset_index=index,
                path=path,
                file_id=path_to_file_id(path),
                original_sample_rate=original_sr,
                original_num_frames=original_frames,
                resampled_num_samples=resampled_samples,
                crop_start=crop_start,
                crop_end=crop_end,
                valid_samples=valid_samples,
                phones=phones,
                overlap_units=frozenset(units),
            )
        )

    return metadata


def build_phn_negative_map(
    metadata: Sequence[SegmentMeta],
    seed: int,
    no_candidate_policy: str,
) -> Tuple[Dict[int, int], List[int], Dict[str, Any]]:
    """Map each anchor index to a phone-disjoint negative index."""
    eligible = [meta.dataset_index for meta in metadata if meta.overlap_units]
    eligible_set = set(eligible)

    unit_to_indices: Dict[Any, Set[int]] = defaultdict(set)
    for meta in metadata:
        for unit in meta.overlap_units:
            unit_to_indices[unit].add(meta.dataset_index)

    negative_map: Dict[int, int] = {}
    skipped: List[int] = []
    overlap_fallback_count = 0

    for anchor_index in tqdm(
        eligible, desc="Sampling phone-disjoint negatives", unit="anchor"
    ):
        anchor = metadata[anchor_index]
        forbidden = {anchor_index}
        for unit in anchor.overlap_units:
            forbidden.update(unit_to_indices[unit])

        candidates = sorted(eligible_set.difference(forbidden))

        if not candidates:
            if no_candidate_policy == "error":
                raise RuntimeError(
                    "No phone-disjoint negative exists for anchor "
                    f"{anchor.path}. Consider --phone_overlap_unit ngram, a larger "
                    "--phone_ngram, or --no_phn_negative_policy skip."
                )

            if no_candidate_policy == "skip":
                skipped.append(anchor_index)
                continue

            if no_candidate_policy == "min_overlap":
                other_indices = [i for i in eligible if i != anchor_index]
                if not other_indices:
                    skipped.append(anchor_index)
                    continue

                # Explicit fallback: choose the candidate with the fewest shared
                # overlap units. This is recorded in the summary because it is
                # not a strict zero-overlap negative.
                min_shared = min(
                    len(anchor.overlap_units.intersection(metadata[i].overlap_units))
                    for i in other_indices
                )
                candidates = [
                    i
                    for i in other_indices
                    if len(anchor.overlap_units.intersection(metadata[i].overlap_units))
                    == min_shared
                ]
                overlap_fallback_count += 1
            else:
                raise ValueError(
                    f"Unknown no_candidate_policy: {no_candidate_policy}"
                )

        rng = random.Random(seed + 2_000_003 * anchor_index)
        negative_map[anchor_index] = candidates[rng.randrange(len(candidates))]

    diagnostics = {
        "eligible_phone_anchors": len(eligible),
        "strict_or_fallback_negative_pairs": len(negative_map),
        "skipped_no_candidate": len(skipped),
        "minimum_overlap_fallback_count": overlap_fallback_count,
    }
    return negative_map, skipped, diagnostics


# -----------------------------------------------------------------------------
# Positive augmentation and temporal rearrangement
# -----------------------------------------------------------------------------


def build_positive_augmenter(
    add_noise: bool,
    sample_rate: int,
    dist_root: Optional[str],
    use_time_compression: bool = False,
) -> Tuple[Callable[[torch.Tensor, random.Random], torch.Tensor], str]:
    from utils.dataset import (
        AudioAugmenter,
        AugConfig,
        _collect_audio_files_from_dirs,
    )

    noise_files: List[str] = []
    rir_files: List[str] = []
    if dist_root and os.path.isdir(dist_root):
        noise_files = _collect_audio_files_from_dirs(
            [
                os.path.join(dist_root, directory)
                for directory in (
                    "noise_16k",
                    "pointsource_noises",
                    "real_rirs_isotropic_noises",
                )
            ],
            exts=(".wav", ".flac"),
        )
        rir_files = _collect_audio_files_from_dirs(
            [os.path.join(dist_root, "rir_16k")],
            exts=(".wav", ".flac"),
        )

    config = AugConfig()
    if not add_noise:
        config = AugConfig(
            p_gain=0.0,
            p_noise=0.0,
            p_reverb=0.0,
            p_real_rir=0.0,
            p_bandpass=0.0,
            p_time_mask=0.0,
            max_ops_per_sample=0,
        )

    augmenter = AudioAugmenter(
        sr=sample_rate,
        cfg=config,
        noise_files=noise_files,
        rir_files=rir_files,
        use_time_compression=use_time_compression,
    )
    return augmenter, "utils.dataset.AudioAugmenter"


def sattolo_derangement(num_items: int, rng: random.Random) -> List[int]:
    """Return a one-cycle derangement; no position keeps its original chunk."""
    if num_items < 2:
        raise ValueError("A derangement requires at least two chunks")

    order = list(range(num_items))
    for i in range(num_items - 1, 0, -1):
        j = rng.randrange(i)  # j is in [0, i-1], never i
        order[i], order[j] = order[j], order[i]

    if any(new_chunk == position for position, new_chunk in enumerate(order)):
        raise RuntimeError(f"Sattolo implementation produced fixed point: {order}")
    return order


def rearrange_waveform_chunks(
    wav: torch.Tensor,
    valid_len: int,
    num_chunks: int,
    rng: random.Random,
) -> Tuple[torch.Tensor, List[int]]:
    """Permute contiguous valid-audio chunks while leaving padding at the end.

    Tensor shape is [1, time]. The returned chunk order is zero-based.
    """
    valid_len = min(max(int(valid_len), 0), int(wav.size(-1)))
    if valid_len < 2:
        return wav.clone(), [0]

    effective_chunks = min(max(int(num_chunks), 2), valid_len)
    order = sattolo_derangement(effective_chunks, rng)

    # tensor_split preserves all samples even when valid_len is not divisible by
    # effective_chunks. Chunk lengths may differ by at most one sample.
    valid_audio = wav[..., :valid_len]
    chunks = list(torch.tensor_split(valid_audio, effective_chunks, dim=-1))
    rearranged_valid = torch.cat([chunks[index] for index in order], dim=-1)

    output = wav.clone()
    output[..., :valid_len] = rearranged_valid
    return output.contiguous(), order


# -----------------------------------------------------------------------------
# Dataset and collate
# -----------------------------------------------------------------------------


class PairwiseNegativeDataset(Dataset):
    def __init__(
        self,
        metadata: Sequence[SegmentMeta],
        anchor_indices: Sequence[int],
        negative_mode: str,
        negative_map: Optional[Dict[int, int]],
        target_sample_rate: int,
        context_sec: float,
        positive_augmenter: Callable[[torch.Tensor, random.Random], torch.Tensor],
        seed: int,
        rearrange_chunks: int,
    ):
        self.metadata = list(metadata)
        self.anchor_indices = list(anchor_indices)
        self.negative_mode = negative_mode
        self.negative_map = negative_map or {}
        self.target_sample_rate = int(target_sample_rate)
        self.target_len = int(round(context_sec * target_sample_rate))
        self.positive_augmenter = positive_augmenter
        self.seed = int(seed)
        self.rearrange_chunks = int(rearrange_chunks)

    def __len__(self) -> int:
        return len(self.anchor_indices)

    def _load_crop(self, meta: SegmentMeta) -> Tuple[torch.Tensor, int]:
        wav, sr = load_audio_mono(meta.path)
        wav = ensure_sample_rate(wav, sr, self.target_sample_rate)
        wav = wav[..., meta.crop_start : meta.crop_end]
        wav, valid_len = pad_or_trim(wav, self.target_len)
        return wav, valid_len

    def __getitem__(self, item: int) -> Dict[str, Any]:
        anchor_index = self.anchor_indices[item]
        anchor_meta = self.metadata[anchor_index]
        anchor_wav, valid_len = self._load_crop(anchor_meta)

        positive_rng = random.Random(self.seed + 3_000_017 * anchor_index)
        positive_wav = self.positive_augmenter(anchor_wav.clone(), positive_rng)
        positive_wav, _ = pad_or_trim(positive_wav, self.target_len)

        negative_meta: Optional[SegmentMeta] = None
        rearrangement_order: List[int] = []

        if self.negative_mode == "phn":
            negative_index = self.negative_map[anchor_index]
            negative_meta = self.metadata[negative_index]
            negative_wav, _ = self._load_crop(negative_meta)
            negative_type = "content_negative_phn"
        elif self.negative_mode == "rearrange":
            rearrange_rng = random.Random(self.seed + 4_000_037 * anchor_index)
            negative_wav, rearrangement_order = rearrange_waveform_chunks(
                wav=anchor_wav,
                valid_len=valid_len,
                num_chunks=self.rearrange_chunks,
                rng=rearrange_rng,
            )
            negative_type = "ordering_negative_rearranged"
        else:
            raise ValueError(f"Unknown negative mode: {self.negative_mode}")

        shared_overlap_units = 0
        if negative_meta is not None:
            shared_overlap_units = len(
                anchor_meta.overlap_units.intersection(negative_meta.overlap_units)
            )

        return {
            "anchor_wav": anchor_wav,
            "positive_wav": positive_wav,
            "negative_wav": negative_wav,
            "file_id": anchor_meta.file_id,
            "anchor_path": anchor_meta.path,
            "negative_file_id": negative_meta.file_id if negative_meta else "",
            "negative_path": negative_meta.path if negative_meta else "",
            "negative_type": negative_type,
            "anchor_phones": " ".join(anchor_meta.phones),
            "negative_phones": " ".join(negative_meta.phones) if negative_meta else "",
            "shared_phone_overlap_units": shared_overlap_units,
            # Store one-based positions in CSV for easier human inspection.
            "rearrangement_order": " ".join(str(index + 1) for index in rearrangement_order),
            "anchor_valid_samples": valid_len,
        }


def pair_negative_collate(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "anchor_wav": torch.stack([item["anchor_wav"] for item in batch], dim=0),
        "positive_wav": torch.stack([item["positive_wav"] for item in batch], dim=0),
        "negative_wav": torch.stack([item["negative_wav"] for item in batch], dim=0),
        "metadata": [
            {key: value for key, value in item.items() if not key.endswith("_wav")}
            for item in batch
        ],
    }


# -----------------------------------------------------------------------------
# Feature extraction and model loading
# -----------------------------------------------------------------------------


def build_batch_featextract(
    sample_rate: int,
    feats_type: str,
    rf_size: int,
    stride: int,
    n_mels: int,
) -> Optional[torch.nn.Module]:
    if feats_type == "melspectrogram":
        return torch.nn.Sequential(
            T.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=rf_size,
                hop_length=stride,
                n_mels=n_mels,
                center=True,
            ),
            T.AmplitudeToDB(),
        )
    if feats_type == "spectrogram":
        return torch.nn.Sequential(
            T.Spectrogram(
                n_fft=rf_size,
                hop_length=stride,
                center=True,
                power=2.0,
            ),
            T.AmplitudeToDB(),
        )
    if feats_type == "raw":
        return None
    raise ValueError(
        "feats_type must be one of: raw, spectrogram, melspectrogram"
    )


def build_model(cfg: Dict[str, Any], ckpt_path: str, model_type: str, device_id: int):
    del cfg  # Kept in signature for compatibility with the original script.

    if model_type == "AudioTokenizer":
        return AudioTokenizer.load_from_pretrained(ckpt_path, device_id)
    if model_type == "AudioTokenizerARPrior":
        return AudioTokenizerARPrior.load_from_pretrained(ckpt_path, device_id)
    if model_type == "AudioTokenizerAR_FINAL":
        return AudioTokenizerAR_FINAL.load_from_pretrained(ckpt_path, device_id)
    raise ValueError(f"Unknown model_type: {model_type}")


@torch.no_grad()
def tokenize_features(
    model,
    features: torch.Tensor,
    max_len: int,
    beam_size: int,
) -> List[List[int]]:
    output = model.tokenize(
        features,
        max_len=max_len,
        beam_size=beam_size,
        strip_special_tokens=True,
    )
    tokens = normalise_token_batch(output)
    if len(tokens) != features.size(0):
        raise RuntimeError(
            f"Tokenizer returned {len(tokens)} sequences for batch size "
            f"{features.size(0)}"
        )
    return tokens


# -----------------------------------------------------------------------------
# CSV generation and summary
# -----------------------------------------------------------------------------


OUTPUT_FIELDS = [
    "file_id",
    "anc_tokenization",
    "pos_tokenization",
    "neg_tokenization",
    "negative_type",
    "anchor_path",
    "negative_file_id",
    "negative_path",
    "anchor_phone_sequence",
    "negative_phone_sequence",
    "shared_phone_overlap_units",
    "rearrangement_order_1based",
    "anchor_valid_samples",
    "anc_length",
    "pos_length",
    "neg_length",
    "anc_pos_edit_distance",
    "anc_pos_normalized_edit_distance",
    "anc_neg_edit_distance",
    "anc_neg_normalized_edit_distance",
    "pos_neg_edit_distance",
    "pos_neg_normalized_edit_distance",
    "pos_to_anc_length_ratio",
    "neg_to_anc_length_ratio",
    "anc_pos_exact_match",
    "anc_neg_exact_match",
]


def mean_or_none(values: Sequence[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


@torch.no_grad()
def generate_pair_tokenizations(
    model,
    loader: DataLoader,
    feature_extractor: Optional[torch.nn.Module],
    output_csv: str,
    summary_json: str,
    device: torch.device,
    max_len: int,
    beam_size: int,
    run_metadata: Dict[str, Any],
) -> None:
    model.eval().to(device)
    if feature_extractor is not None:
        feature_extractor.eval()

    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(summary_json) or ".", exist_ok=True)

    anc_pos_neds: List[float] = []
    anc_neg_neds: List[float] = []
    pos_neg_neds: List[float] = []
    anc_lengths: List[float] = []
    pos_lengths: List[float] = []
    neg_lengths: List[float] = []
    shared_phone_counts: List[float] = []
    written = 0

    with open(output_csv, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()

        progress = tqdm(loader, desc="Generating anchor/positive/negative tokens", unit="batch")
        for batch in progress:
            anchor_wav = batch["anchor_wav"]
            positive_wav = batch["positive_wav"]
            negative_wav = batch["negative_wav"]

            if feature_extractor is None:
                anchor_features = anchor_wav
                positive_features = positive_wav
                negative_features = negative_wav
            else:
                # Match the retrieval script: feature extraction is performed on
                # CPU and the feature tensors are moved to the model device.
                anchor_features = feature_extractor(anchor_wav)
                positive_features = feature_extractor(positive_wav)
                negative_features = feature_extractor(negative_wav)

            anchor_features = anchor_features.to(device, non_blocking=True).contiguous()
            positive_features = positive_features.to(device, non_blocking=True).contiguous()
            negative_features = negative_features.to(device, non_blocking=True).contiguous()

            anchor_tokens = tokenize_features(
                model, anchor_features, max_len=max_len, beam_size=beam_size
            )
            positive_tokens = tokenize_features(
                model, positive_features, max_len=max_len, beam_size=beam_size
            )
            negative_tokens = tokenize_features(
                model, negative_features, max_len=max_len, beam_size=beam_size
            )

            batch_size = len(batch["metadata"])
            if not (
                len(anchor_tokens)
                == len(positive_tokens)
                == len(negative_tokens)
                == batch_size
            ):
                raise RuntimeError("Anchor/positive/negative token batch-size mismatch")

            for meta, anc, pos, neg in zip(
                batch["metadata"], anchor_tokens, positive_tokens, negative_tokens
            ):
                anc_pos_ed = edit_distance(anc, pos)
                anc_neg_ed = edit_distance(anc, neg)
                pos_neg_ed = edit_distance(pos, neg)

                anc_pos_ned = anc_pos_ed / max(len(anc), len(pos), 1)
                anc_neg_ned = anc_neg_ed / max(len(anc), len(neg), 1)
                pos_neg_ned = pos_neg_ed / max(len(pos), len(neg), 1)

                writer.writerow(
                    {
                        "file_id": meta["file_id"],
                        "anc_tokenization": serialize_tokens(anc),
                        "pos_tokenization": serialize_tokens(pos),
                        "neg_tokenization": serialize_tokens(neg),
                        "negative_type": meta["negative_type"],
                        "anchor_path": meta["anchor_path"],
                        "negative_file_id": meta["negative_file_id"],
                        "negative_path": meta["negative_path"],
                        "anchor_phone_sequence": meta["anchor_phones"],
                        "negative_phone_sequence": meta["negative_phones"],
                        "shared_phone_overlap_units": meta[
                            "shared_phone_overlap_units"
                        ],
                        "rearrangement_order_1based": meta["rearrangement_order"],
                        "anchor_valid_samples": meta["anchor_valid_samples"],
                        "anc_length": len(anc),
                        "pos_length": len(pos),
                        "neg_length": len(neg),
                        "anc_pos_edit_distance": anc_pos_ed,
                        "anc_pos_normalized_edit_distance": f"{anc_pos_ned:.8f}",
                        "anc_neg_edit_distance": anc_neg_ed,
                        "anc_neg_normalized_edit_distance": f"{anc_neg_ned:.8f}",
                        "pos_neg_edit_distance": pos_neg_ed,
                        "pos_neg_normalized_edit_distance": f"{pos_neg_ned:.8f}",
                        "pos_to_anc_length_ratio": (
                            f"{len(pos) / len(anc):.8f}" if len(anc) else ""
                        ),
                        "neg_to_anc_length_ratio": (
                            f"{len(neg) / len(anc):.8f}" if len(anc) else ""
                        ),
                        "anc_pos_exact_match": int(anc == pos),
                        "anc_neg_exact_match": int(anc == neg),
                    }
                )

                anc_pos_neds.append(anc_pos_ned)
                anc_neg_neds.append(anc_neg_ned)
                pos_neg_neds.append(pos_neg_ned)
                anc_lengths.append(float(len(anc)))
                pos_lengths.append(float(len(pos)))
                neg_lengths.append(float(len(neg)))
                shared_phone_counts.append(
                    float(meta["shared_phone_overlap_units"])
                )
                written += 1

            progress.set_postfix(written=written)

    if written == 0:
        raise RuntimeError("No rows were written to the output CSV")

    summary = dict(run_metadata)
    summary.update(
        {
            "num_rows": written,
            "output_csv": os.path.abspath(output_csv),
            "mean_anchor_length": mean_or_none(anc_lengths),
            "mean_positive_length": mean_or_none(pos_lengths),
            "mean_negative_length": mean_or_none(neg_lengths),
            "mean_anchor_positive_ned": mean_or_none(anc_pos_neds),
            "mean_anchor_negative_ned": mean_or_none(anc_neg_neds),
            "mean_positive_negative_ned": mean_or_none(pos_neg_neds),
            "mean_shared_phone_overlap_units": mean_or_none(shared_phone_counts),
            "ordering_or_content_probe_margin": (
                mean_or_none(anc_neg_neds) - mean_or_none(anc_pos_neds)
                if anc_neg_neds and anc_pos_neds
                else None
            ),
        }
    )

    with open(summary_json, "w") as handle:
        json.dump(summary, handle, indent=2)

    print(f"Saved pairwise-consistency CSV: {output_csv}")
    print(f"Saved summary JSON: {summary_json}")
    print(json.dumps(summary, indent=2))


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate pairwise consistency tokenizations with negatives"
    )

    parser.add_argument(
        "--root_dir",
        type=str,
        default=os.path.dirname(os.getcwd()),
        help="Parent working directory",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="mainFull.yaml",
        help="Config path or file name inside root_dir/config/",
    )
    parser.add_argument("--ckpt", type=str, required=True, help="Checkpoint path")
    parser.add_argument(
        "--split",
        type=str,
        default="valid",
        choices=["train", "valid", "test"],
        help="Config split to tokenize",
    )
    parser.add_argument(
        "--output_csv", type=str, required=True, help="Output CSV path"
    )
    parser.add_argument(
        "--summary_json",
        type=str,
        default=None,
        help="Optional summary JSON path; defaults beside output CSV",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="AudioTokenizerAR_FINAL",
        choices=["AudioTokenizer", "AudioTokenizerARPrior", "AudioTokenizerAR_FINAL"],
    )
    parser.add_argument("--device", type=int, default=0, help="CUDA device id")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--max_len", type=int, default=60)
    parser.add_argument("--beam_size", type=int, default=4)
    parser.add_argument("--path_col", type=str, default="path")
    parser.add_argument("--seed", type=int, default=13)

    parser.add_argument(
        "--negative_mode",
        type=str,
        default="auto",
        choices=["auto", "phn", "rearrange"],
        help=(
            "auto: PHN mode only when every file has .PHN; otherwise rearrange. "
            "phn: force phone-disjoint content negatives. rearrange: force "
            "within-anchor temporal-order negatives."
        ),
    )
    parser.add_argument(
        "--phone_overlap_unit",
        type=str,
        default="ngram",
        choices=["phone", "ngram"],
        help=(
            "phone forbids any shared phone label. ngram forbids any shared "
            "contiguous phone sequence of length --phone_ngram."
        ),
    )
    parser.add_argument(
        "--phone_ngram",
        type=int,
        default=3,
        help="Phone n-gram length used when --phone_overlap_unit ngram",
    )
    parser.add_argument(
        "--silence_phones",
        type=str,
        default="h#,pau,epi",
        help="Comma-separated PHN labels excluded from overlap checks",
    )
    parser.add_argument(
        "--no_phn_negative_policy",
        type=str,
        default="skip",
        choices=["error", "skip", "min_overlap"],
        help=(
            "What to do when an anchor has no zero-overlap PHN negative. "
            "skip preserves strictness; min_overlap is an explicit relaxed fallback."
        ),
    )
    parser.add_argument(
        "--rearrange_chunks",
        type=int,
        default=6,
        help="Number of contiguous waveform chunks to derange in rearrange mode",
    )

    noise_group = parser.add_mutually_exclusive_group()
    noise_group.add_argument(
        "--add_noise",
        dest="add_noise",
        action="store_true",
        help="Augment the positive waveform",
    )
    noise_group.add_argument(
        "--no_add_noise",
        dest="add_noise",
        action="store_false",
        help="Use a clean positive copy",
    )
    parser.set_defaults(add_noise=True)
    parser.add_argument(
        "--use_time_compression",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Slightly compress only the positive view and right-pad it back to 3 seconds.",
    )

    parser.add_argument(
        "--dist_root",
        type=str,
        default=None,
        help="Optional distortion root used by utils.dataset.AudioAugmenter.",
    )
    parser.add_argument(
        "--feats_type",
        type=str,
        default="melspectrogram",
        choices=["raw", "spectrogram", "melspectrogram"],
    )

    args = parser.parse_args()

    config_path = resolve_config_path(args.root_dir, args.config)
    with open(config_path, "r") as handle:
        cfg = yaml.load(handle, Loader=yaml.FullLoader)

    if args.split not in cfg:
        raise ValueError(f"Split `{args.split}` not found in config")

    split_cfg = cfg[args.split]
    dataset_csv = resolve_dataset_csv(
        root_dir=args.root_dir,
        config_path=config_path,
        csv_arg=split_cfg["dataset_csv"],
    )

    seed = int(args.seed if args.seed is not None else cfg.get("seed", 313))
    seed_everything(seed, workers=True)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    sample_rate = int(cfg["fs"])
    context_sec = float(cfg["seglen"])
    batch_size = (
        int(args.batch_size)
        if args.batch_size is not None
        else int(cfg["batchsize"])
    )
    num_workers = (
        int(args.num_workers)
        if args.num_workers is not None
        else int(cfg["load_workers"])
    )

    paths = read_audio_paths(dataset_csv, path_col=args.path_col)
    phn_exists = [find_existing_phn_path(path) is not None for path in paths]
    all_phn = bool(paths) and all(phn_exists)
    any_phn = any(phn_exists)

    if args.negative_mode == "auto":
        selected_mode = "phn" if all_phn else "rearrange"
        if any_phn and not all_phn:
            warnings.warn(
                "Only some files have .PHN labels. Auto mode therefore uses "
                "rearrangement negatives for the entire dataset."
            )
    elif args.negative_mode == "phn":
        missing_phn = [
            phn_path_from_audio(path)
            for path, exists in zip(paths, phn_exists)
            if not exists
        ]
        if missing_phn:
            preview = "\n  ".join(missing_phn[:10])
            raise FileNotFoundError(
                "PHN mode requires a .PHN file for every audio file. Missing:\n  "
                + preview
                + ("\n  ..." if len(missing_phn) > 10 else "")
            )
        selected_mode = "phn"
    else:
        selected_mode = "rearrange"

    silence_phones = {
        phone.strip()
        for phone in args.silence_phones.split(",")
        if phone.strip()
    }

    metadata = build_segment_metadata(
        paths=paths,
        target_sample_rate=sample_rate,
        context_sec=context_sec,
        seed=seed,
        split=args.split,
        include_phn=(selected_mode == "phn"),
        silence_phones=silence_phones,
        phone_overlap_unit=args.phone_overlap_unit,
        phone_ngram=args.phone_ngram,
    )

    negative_map: Dict[int, int] = {}
    phn_sampling_diagnostics: Dict[str, Any] = {}

    if selected_mode == "phn":
        negative_map, skipped, phn_sampling_diagnostics = build_phn_negative_map(
            metadata=metadata,
            seed=seed,
            no_candidate_policy=args.no_phn_negative_policy,
        )
        anchor_indices = sorted(negative_map.keys())

        # In auto mode, fall back to rearrangement if strict PHN sampling cannot
        # produce any usable pair. Explicit --negative_mode phn remains strict.
        if not anchor_indices and args.negative_mode == "auto":
            warnings.warn(
                "PHN files exist, but no valid phone-disjoint negatives were "
                "found. Auto mode is falling back to temporal rearrangement."
            )
            selected_mode = "rearrange"
            metadata = build_segment_metadata(
                paths=paths,
                target_sample_rate=sample_rate,
                context_sec=context_sec,
                seed=seed,
                split=args.split,
                include_phn=False,
                silence_phones=silence_phones,
                phone_overlap_unit=args.phone_overlap_unit,
                phone_ngram=args.phone_ngram,
            )
            negative_map = {}
            anchor_indices = list(range(len(metadata)))
        elif not anchor_indices:
            raise RuntimeError(
                "No PHN-negative pairs are available under the selected overlap rule"
            )
    else:
        anchor_indices = list(range(len(metadata)))

    positive_augmenter, positive_augmenter_name = build_positive_augmenter(
        add_noise=args.add_noise,
        sample_rate=sample_rate,
        dist_root=args.dist_root,
        use_time_compression=args.use_time_compression,
    )

    dataset = PairwiseNegativeDataset(
        metadata=metadata,
        anchor_indices=anchor_indices,
        negative_mode=selected_mode,
        negative_map=negative_map,
        target_sample_rate=sample_rate,
        context_sec=context_sec,
        positive_augmenter=positive_augmenter,
        seed=seed,
        rearrange_chunks=args.rearrange_chunks,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=pair_negative_collate,
    )

    rf_size = int(round(float(cfg["win_size"]) * sample_rate))
    stride = int(round(float(cfg["win_hop"]) * sample_rate))
    n_mels = int(cfg["n_mels"])
    feature_extractor = build_batch_featextract(
        sample_rate=sample_rate,
        feats_type=args.feats_type,
        rf_size=rf_size,
        stride=stride,
        n_mels=n_mels,
    )

    device = torch.device(
        f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    )


    model = build_model(
        cfg=cfg,
        ckpt_path=args.ckpt,
        model_type=args.model_type,
        device_id=args.device,
    )

    if args.summary_json is None:
        stem, _ = os.path.splitext(args.output_csv)
        summary_json = stem + "_summary.json"
    else:
        summary_json = args.summary_json

    run_metadata = {
        "config_path": os.path.abspath(config_path),
        "dataset_csv": os.path.abspath(dataset_csv),
        "checkpoint": os.path.abspath(args.ckpt),
        "split": args.split,
        "seed": seed,
        "device": str(device),
        "requested_negative_mode": args.negative_mode,
        "selected_negative_mode": selected_mode,
        "num_audio_files": len(paths),
        "num_anchor_items": len(anchor_indices),
        "all_phn_files_available": all_phn,
        "any_phn_files_available": any_phn,
        "phone_overlap_unit": args.phone_overlap_unit,
        "phone_ngram": args.phone_ngram,
        "silence_phones": sorted(silence_phones),
        "no_phn_negative_policy": args.no_phn_negative_policy,
        "rearrange_chunks": args.rearrange_chunks,
        "positive_augmentation_enabled": args.add_noise,
        "positive_time_compression_enabled": args.use_time_compression,
        "positive_augmenter": positive_augmenter_name,
        "sample_rate": sample_rate,
        "context_sec": context_sec,
        "feature_type": args.feats_type,
        "max_len": args.max_len,
        "beam_size": args.beam_size,
        "batch_size": batch_size,
        "phn_sampling": phn_sampling_diagnostics,
    }

    print("==== Pairwise consistency negative generation ====")
    print(json.dumps(run_metadata, indent=2))

    generate_pair_tokenizations(
        model=model,
        loader=loader,
        feature_extractor=feature_extractor,
        output_csv=args.output_csv,
        summary_json=summary_json,
        device=device,
        max_len=args.max_len,
        beam_size=args.beam_size,
        run_metadata=run_metadata,
    )


if __name__ == "__main__":
    main()
