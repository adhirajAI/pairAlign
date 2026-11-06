#!/usr/bin/env python3
"""TIMIT + external real-noise consistency with controlled negatives.

Triplet construction
--------------------
Anchor:
    A deterministic clean 3-second TIMIT crop.

Positive:
    The same crop mixed with one external ``noise_only`` recording.

Negative modes:
    rearrange
        The anchor crop is split into contiguous waveform chunks and permuted
        with a Sattolo derangement. Every chunk moves. Default: 12 chunks.

    phone
        A crop from another TIMIT utterance that shares no nonsilence phone
        labels with the anchor crop.

    trigram
        A crop from another TIMIT utterance that shares no contiguous
        nonsilence phone trigram with the anchor crop. Individual phones may
        overlap; phone trigrams may not.

For all modes, the negative receives the same external noise segment at the
same requested SNR as the positive. This noise-matches positive and negative,
so separation must arise from speech content/order rather than noise identity.

Keep this file beside:
    evaluate_timit_real_noise_pairwise_consistency.py
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
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from lightning.pytorch import seed_everything
from tqdm.auto import tqdm
import torchaudio
import yaml

from utils import AudioAugmenter, AugConfig

from evaluate_timit_real_noise_pairwise_consistency import (
    DEFAULT_NOISE_DIR,
    active_rms,
    build_feature_extractor,
    build_model,
    crop_or_pad,
    crop_start,
    decode_hf_audio,
    describe,
    edit_distance,
    file_id,
    find_parquet_files,
    index_noise_only,
    load_audio,
    load_local_noise_dataset,
    noise_crop_or_repeat,
    normalize_rms,
    parse_deduplicate,
    read_audio_paths,
    remove_dc,
    resample,
    resolve_config_path,
    resolve_dataset_csv,
    serialize_tokens,
    token_jaccard,
    tokenize,
)


SILENCE_PHONES_DEFAULT = {"h#", "pau", "epi"}


# -----------------------------------------------------------------------------
# PHN metadata and controlled negative sampling
# -----------------------------------------------------------------------------


def find_existing_phn_path(audio_path: str) -> Optional[str]:
    stem = os.path.splitext(audio_path)[0]
    for extension in (".PHN", ".phn", ".Phn", ".pHN"):
        candidate = stem + extension
        if os.path.isfile(candidate):
            return candidate
    return None


def canonical_phn_path(audio_path: str) -> str:
    return os.path.splitext(audio_path)[0] + ".PHN"


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
            intervals.append((int(start), int(end), str(phone)))
    return intervals


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
        start_target = int(
            round(start_original * target_sample_rate / original_sample_rate)
        )
        end_target = int(
            round(end_original * target_sample_rate / original_sample_rate)
        )
        overlaps_crop = (
            end_target > crop_start_target and start_target < crop_end_target
        )
        if overlaps_crop and phone not in silence_phones:
            phones.append(phone)
    return tuple(phones)


def phone_overlap_units(
    phones: Sequence[str],
    negative_mode: str,
) -> Set[Any]:
    if negative_mode == "phone":
        return set(phones)

    if negative_mode == "trigram":
        n = 3
        if len(phones) < n:
            # Do not return an empty set, since that would make every candidate
            # appear valid for a short phone sequence.
            return {tuple(phones)} if phones else set()
        return {
            tuple(phones[index : index + n])
            for index in range(len(phones) - n + 1)
        }

    if negative_mode == "rearrange":
        return set()

    raise ValueError(f"Unknown negative mode: {negative_mode}")


@dataclass(frozen=True)
class SegmentMeta:
    dataset_index: int
    path: str
    file_id: str
    original_sample_rate: int
    crop_start: int
    crop_end: int
    valid_samples: int
    phones: Tuple[str, ...]
    overlap_units: frozenset


def build_segment_metadata(
    paths: Sequence[str],
    target_sample_rate: int,
    context_sec: float,
    seed: int,
    split: str,
    negative_mode: str,
    silence_phones: Set[str],
) -> List[SegmentMeta]:
    target_len = int(round(float(context_sec) * target_sample_rate))
    include_phn = negative_mode in {"phone", "trigram"}
    metadata: List[SegmentMeta] = []

    for index, path in enumerate(
        tqdm(paths, desc="Reading TIMIT/PHN metadata", unit="file")
    ):
        info = torchaudio.info(path)
        original_sr = int(info.sample_rate)
        original_frames = int(info.num_frames)
        resampled_samples = int(
            round(original_frames * target_sample_rate / original_sr)
        )
        start = crop_start(
            num_samples=resampled_samples,
            target_len=target_len,
            seed=seed,
            index=index,
            split=split,
        )
        end = start + target_len
        valid_samples = max(0, min(target_len, resampled_samples - start))

        phones: Tuple[str, ...] = ()
        units: Set[Any] = set()
        if include_phn:
            phones = phones_for_crop(
                audio_path=path,
                original_sample_rate=original_sr,
                target_sample_rate=target_sample_rate,
                crop_start_target=start,
                crop_end_target=end,
                silence_phones=silence_phones,
            )
            units = phone_overlap_units(phones, negative_mode)

        metadata.append(
            SegmentMeta(
                dataset_index=index,
                path=path,
                file_id=file_id(path),
                original_sample_rate=original_sr,
                crop_start=start,
                crop_end=end,
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
    """Map each eligible anchor to a zero-overlap PHN negative."""
    eligible = [
        meta.dataset_index for meta in metadata if len(meta.overlap_units) > 0
    ]
    eligible_set = set(eligible)

    unit_to_indices: Dict[Any, Set[int]] = defaultdict(set)
    for meta in metadata:
        for unit in meta.overlap_units:
            unit_to_indices[unit].add(meta.dataset_index)

    negative_map: Dict[int, int] = {}
    skipped: List[int] = []
    minimum_overlap_fallback_count = 0
    fallback_shared_units: List[int] = []

    for anchor_index in tqdm(
        eligible,
        desc="Sampling PHN-controlled negatives",
        unit="anchor",
    ):
        anchor = metadata[anchor_index]
        forbidden = {anchor_index}
        for unit in anchor.overlap_units:
            forbidden.update(unit_to_indices[unit])

        candidates = sorted(eligible_set.difference(forbidden))

        if not candidates:
            if no_candidate_policy == "error":
                raise RuntimeError(
                    "No strict PHN-disjoint negative exists for anchor "
                    f"{anchor.path}. Use --no_phn_negative_policy skip or "
                    "min_overlap, or use --negative_mode trigram."
                )

            if no_candidate_policy == "skip":
                skipped.append(anchor_index)
                continue

            if no_candidate_policy == "min_overlap":
                other_indices = [
                    candidate
                    for candidate in eligible
                    if candidate != anchor_index
                ]
                if not other_indices:
                    skipped.append(anchor_index)
                    continue

                minimum_shared = min(
                    len(
                        anchor.overlap_units.intersection(
                            metadata[candidate].overlap_units
                        )
                    )
                    for candidate in other_indices
                )
                candidates = [
                    candidate
                    for candidate in other_indices
                    if len(
                        anchor.overlap_units.intersection(
                            metadata[candidate].overlap_units
                        )
                    )
                    == minimum_shared
                ]
                minimum_overlap_fallback_count += 1
                fallback_shared_units.append(minimum_shared)
            else:
                raise ValueError(
                    f"Unknown no_candidate_policy: {no_candidate_policy}"
                )

        rng = random.Random(seed + 2_000_003 * anchor_index)
        negative_map[anchor_index] = candidates[rng.randrange(len(candidates))]

    diagnostics = {
        "eligible_phone_anchors": len(eligible),
        "sampled_negative_pairs": len(negative_map),
        "skipped_no_candidate": len(skipped),
        "minimum_overlap_fallback_count": minimum_overlap_fallback_count,
        "mean_fallback_shared_units": (
            float(np.mean(fallback_shared_units))
            if fallback_shared_units
            else 0.0
        ),
    }
    return negative_map, skipped, diagnostics


# -----------------------------------------------------------------------------
# Rearrangement negatives
# -----------------------------------------------------------------------------


def sattolo_order(num_items: int, rng: random.Random) -> List[int]:
    if num_items < 2:
        raise ValueError("A Sattolo derangement requires at least two chunks.")

    order = list(range(num_items))
    for index in range(num_items - 1, 0, -1):
        swap_index = rng.randrange(index)  # never equal to index
        order[index], order[swap_index] = order[swap_index], order[index]

    if any(position == source for position, source in enumerate(order)):
        raise RuntimeError(f"Rearrangement has a fixed point: {order}")
    return order


def rearrange_waveform_chunks(
    waveform: torch.Tensor,
    valid_samples: int,
    num_chunks: int,
    rng: random.Random,
) -> Tuple[torch.Tensor, List[int]]:
    valid_samples = min(
        max(int(valid_samples), 0),
        int(waveform.size(-1)),
    )
    if valid_samples < 2:
        return waveform.clone(), [0]

    effective_chunks = min(max(int(num_chunks), 2), valid_samples)
    order = sattolo_order(effective_chunks, rng)
    valid_audio = waveform[..., :valid_samples]
    chunks = list(
        torch.tensor_split(valid_audio, effective_chunks, dim=-1)
    )
    rearranged_valid = torch.cat(
        [chunks[source_index] for source_index in order],
        dim=-1,
    )

    output = waveform.clone()
    output[..., :valid_samples] = rearranged_valid
    return output.contiguous(), order


# -----------------------------------------------------------------------------
# Noise-matched triplet mixing
# -----------------------------------------------------------------------------


def mix_noise_matched_triplet(
    anchor_speech: torch.Tensor,
    negative_speech: torch.Tensor,
    noise: torch.Tensor,
    anchor_valid_samples: int,
    negative_valid_samples: int,
    target_rms_dbfs: float,
    snr_db: float,
    peak_limit: float,
    gate_db: float,
) -> Dict[str, Any]:
    """Apply one shared external-noise condition to positive and negative."""
    anchor_speech = remove_dc(anchor_speech, anchor_valid_samples)
    negative_speech = remove_dc(negative_speech, negative_valid_samples)
    noise = remove_dc(noise, None)

    anchor_speech, anchor_gain_db = normalize_rms(
        anchor_speech,
        target_rms_dbfs,
        anchor_valid_samples,
        gate_db,
    )
    negative_speech, negative_gain_db = normalize_rms(
        negative_speech,
        target_rms_dbfs,
        negative_valid_samples,
        gate_db,
    )
    noise, noise_gain_db = normalize_rms(
        noise,
        target_rms_dbfs,
        None,
        gate_db,
    )

    scaled_noise = noise * (10.0 ** (-float(snr_db) / 20.0))
    positive = anchor_speech + scaled_noise
    negative = negative_speech + scaled_noise

    maximum_peak = max(
        float(anchor_speech.abs().max().item()),
        float(positive.abs().max().item()),
        float(negative.abs().max().item()),
        1e-12,
    )
    common_scale = min(1.0, float(peak_limit) / maximum_peak)

    anchor_speech = (anchor_speech * common_scale).contiguous()
    positive = (positive * common_scale).contiguous()
    negative_speech = (negative_speech * common_scale).contiguous()
    negative = (negative * common_scale).contiguous()
    scaled_noise = (scaled_noise * common_scale).contiguous()

    noise_rms = float(active_rms(scaled_noise, None, gate_db).item())
    anchor_rms = float(
        active_rms(anchor_speech, anchor_valid_samples, gate_db).item()
    )
    negative_rms = float(
        active_rms(negative_speech, negative_valid_samples, gate_db).item()
    )

    anchor_snr = 20.0 * math.log10(
        max(anchor_rms / max(noise_rms, 1e-12), 1e-12)
    )
    negative_snr = 20.0 * math.log10(
        max(negative_rms / max(noise_rms, 1e-12), 1e-12)
    )

    return {
        "anchor": anchor_speech,
        "positive": positive,
        "negative": negative,
        "anchor_speech_gain_db": anchor_gain_db,
        "negative_speech_gain_db": negative_gain_db,
        "noise_gain_db": noise_gain_db,
        "requested_snr_db": float(snr_db),
        "anchor_achieved_snr_db": anchor_snr,
        "negative_achieved_snr_db": negative_snr,
        "common_peak_scale": common_scale,
        "positive_peak": float(positive.abs().max().item()),
        "negative_peak": float(negative.abs().max().item()),
    }


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------


class TimitControlledNegativeDataset(Dataset):
    def __init__(
        self,
        metadata: Sequence[SegmentMeta],
        anchor_indices: Sequence[int],
        negative_mode: str,
        negative_map: Optional[Dict[int, int]],
        noise_dataset,
        noise_rows: Sequence[Dict[str, Any]],
        noise_dataset_dir: str,
        sample_rate: int,
        context_sec: float,
        seed: int,
        snr_min: float,
        snr_max: float,
        target_rms_dbfs: float,
        peak_limit: float,
        gate_db: float,
        rearrange_chunks: int,
        use_time_compression: bool = False,
    ):
        self.metadata = list(metadata)
        self.anchor_indices = list(anchor_indices)
        self.negative_mode = str(negative_mode)
        self.negative_map = negative_map or {}
        self.noise_dataset = noise_dataset
        self.noise_rows = list(noise_rows)
        self.noise_dataset_dir = str(noise_dataset_dir)
        self.sample_rate = int(sample_rate)
        self.target_len = int(round(float(context_sec) * self.sample_rate))
        self.seed = int(seed)
        self.snr_min = float(snr_min)
        self.snr_max = float(snr_max)
        self.target_rms_dbfs = float(target_rms_dbfs)
        self.peak_limit = float(peak_limit)
        self.gate_db = float(gate_db)
        self.rearrange_chunks = int(rearrange_chunks)
        self.use_time_compression = bool(use_time_compression)
        self.time_compressor = AudioAugmenter(
            sr=self.sample_rate,
            cfg=AugConfig(
                p_gain=0.0,
                p_noise=0.0,
                p_reverb=0.0,
                p_real_rir=0.0,
                p_bandpass=0.0,
                p_time_mask=0.0,
                max_ops_per_sample=0,
            ),
            use_time_compression=True,
        ) if self.use_time_compression else None

        if not self.noise_rows:
            raise ValueError("No external noise_only rows are available.")
        if not self.anchor_indices:
            raise ValueError("No anchor items are available for evaluation.")
        if self.snr_max < self.snr_min:
            raise ValueError("snr_max must be >= snr_min.")
        if self.rearrange_chunks < 2:
            raise ValueError("rearrange_chunks must be at least 2.")

    def __len__(self) -> int:
        return len(self.anchor_indices)

    def _load_crop(self, meta: SegmentMeta) -> Tuple[torch.Tensor, int]:
        waveform, original_sr = load_audio(meta.path)
        waveform = resample(waveform, original_sr, self.sample_rate)
        waveform = waveform[..., meta.crop_start : meta.crop_end]
        waveform, valid_samples = crop_or_pad(
            waveform,
            start=0,
            target_len=self.target_len,
        )
        return waveform, min(valid_samples, meta.valid_samples)

    def __getitem__(self, item: int) -> Dict[str, Any]:
        anchor_index = self.anchor_indices[item]
        anchor_meta = self.metadata[anchor_index]
        anchor_speech, anchor_valid = self._load_crop(anchor_meta)

        negative_meta: Optional[SegmentMeta] = None
        rearrangement_order: List[int] = []

        if self.negative_mode == "rearrange":
            rearrange_rng = random.Random(
                self.seed + 5_000_081 * anchor_index
            )
            negative_speech, rearrangement_order = rearrange_waveform_chunks(
                waveform=anchor_speech,
                valid_samples=anchor_valid,
                num_chunks=self.rearrange_chunks,
                rng=rearrange_rng,
            )
            negative_valid = anchor_valid
            negative_type = f"rearrange_{len(rearrangement_order)}_chunks"
        elif self.negative_mode in {"phone", "trigram"}:
            negative_index = self.negative_map[anchor_index]
            negative_meta = self.metadata[negative_index]
            negative_speech, negative_valid = self._load_crop(negative_meta)
            negative_type = (
                "phone_disjoint_content_negative"
                if self.negative_mode == "phone"
                else "phone_trigram_disjoint_content_negative"
            )
        else:
            raise ValueError(f"Unknown negative mode: {self.negative_mode}")

        noise_rng = random.Random(self.seed + 3_000_017 * anchor_index)
        noise_meta = self.noise_rows[
            noise_rng.randrange(len(self.noise_rows))
        ]
        noise_row = self.noise_dataset[noise_meta["row_index"]]
        noise, noise_original_sr = decode_hf_audio(
            noise_row["audio"],
            self.noise_dataset_dir,
            noise_meta["filename"],
        )
        noise = resample(noise, noise_original_sr, self.sample_rate)
        noise, noise_start = noise_crop_or_repeat(
            noise,
            self.target_len,
            self.seed,
            anchor_index,
        )

        requested_snr = random.Random(
            self.seed + 4_000_037 * anchor_index
        ).uniform(self.snr_min, self.snr_max)

        triplet = mix_noise_matched_triplet(
            anchor_speech=anchor_speech,
            negative_speech=negative_speech,
            noise=noise,
            anchor_valid_samples=anchor_valid,
            negative_valid_samples=negative_valid,
            target_rms_dbfs=self.target_rms_dbfs,
            snr_db=requested_snr,
            peak_limit=self.peak_limit,
            gate_db=self.gate_db,
        )

        compression_info = {
            "time_compression_applied": False,
            "time_compression_rate": 1.0,
            "positive_right_pad_samples": 0,
        }
        positive_wav = triplet["positive"]
        if self.time_compressor is not None:
            positive_wav, compression_info = self.time_compressor(
                positive_wav,
                rng=random.Random(self.seed + 6_000_083 * anchor_index),
                return_info=True,
            )

        shared_units = 0
        if negative_meta is not None:
            shared_units = len(
                anchor_meta.overlap_units.intersection(
                    negative_meta.overlap_units
                )
            )

        return {
            "anchor_wav": triplet["anchor"],
            "positive_wav": positive_wav,
            "negative_wav": triplet["negative"],
            "file_id": anchor_meta.file_id,
            "anchor_path": anchor_meta.path,
            "negative_file_id": (
                negative_meta.file_id if negative_meta is not None else ""
            ),
            "negative_path": (
                negative_meta.path if negative_meta is not None else ""
            ),
            "negative_dataset_index": (
                negative_meta.dataset_index if negative_meta is not None else -1
            ),
            "negative_mode": self.negative_mode,
            "negative_type": negative_type,
            "anchor_phone_sequence": " ".join(anchor_meta.phones),
            "negative_phone_sequence": (
                " ".join(negative_meta.phones)
                if negative_meta is not None
                else ""
            ),
            "shared_phone_overlap_units": shared_units,
            "rearrangement_order_1based": " ".join(
                str(index + 1) for index in rearrangement_order
            ),
            "rearrange_chunks_requested": self.rearrange_chunks,
            "rearrange_chunks_effective": len(rearrangement_order),
            "noise_filename": noise_meta["filename"],
            "noise_row_index": noise_meta["row_index"],
            "anchor_original_sr": anchor_meta.original_sample_rate,
            "negative_original_sr": (
                negative_meta.original_sample_rate
                if negative_meta is not None
                else anchor_meta.original_sample_rate
            ),
            "noise_original_sr": noise_original_sr,
            "anchor_crop_start": anchor_meta.crop_start,
            "negative_crop_start": (
                negative_meta.crop_start if negative_meta is not None else anchor_meta.crop_start
            ),
            "noise_crop_start": noise_start,
            "anchor_valid_samples": anchor_valid,
            "negative_valid_samples": negative_valid,
            "requested_snr_db": triplet["requested_snr_db"],
            "anchor_achieved_snr_db": triplet[
                "anchor_achieved_snr_db"
            ],
            "negative_achieved_snr_db": triplet[
                "negative_achieved_snr_db"
            ],
            "anchor_speech_normalization_gain_db": triplet[
                "anchor_speech_gain_db"
            ],
            "negative_speech_normalization_gain_db": triplet[
                "negative_speech_gain_db"
            ],
            "noise_normalization_gain_db": triplet["noise_gain_db"],
            "common_peak_scale": triplet["common_peak_scale"],
            "positive_peak": triplet["positive_peak"],
            "time_compression_applied": compression_info["time_compression_applied"],
            "time_compression_rate": compression_info["time_compression_rate"],
            "positive_right_pad_samples": compression_info["positive_right_pad_samples"],
            "negative_peak": triplet["negative_peak"],
        }


def collate_triplets(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "anchor_wav": torch.stack(
            [item["anchor_wav"] for item in batch],
            dim=0,
        ),
        "positive_wav": torch.stack(
            [item["positive_wav"] for item in batch],
            dim=0,
        ),
        "negative_wav": torch.stack(
            [item["negative_wav"] for item in batch],
            dim=0,
        ),
        "metadata": [
            {
                key: value
                for key, value in item.items()
                if key not in {
                    "anchor_wav",
                    "positive_wav",
                    "negative_wav",
                }
            }
            for item in batch
        ],
    }


# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------


FIELDS = [
    "file_id",
    "anchor_path",
    "negative_file_id",
    "negative_path",
    "negative_dataset_index",
    "negative_mode",
    "negative_type",
    "anchor_phone_sequence",
    "negative_phone_sequence",
    "shared_phone_overlap_units",
    "rearrangement_order_1based",
    "rearrange_chunks_requested",
    "rearrange_chunks_effective",
    "noise_filename",
    "noise_row_index",
    "anchor_original_sr",
    "negative_original_sr",
    "noise_original_sr",
    "anchor_crop_start",
    "negative_crop_start",
    "noise_crop_start",
    "anchor_valid_samples",
    "negative_valid_samples",
    "requested_snr_db",
    "anchor_achieved_snr_db",
    "negative_achieved_snr_db",
    "anchor_speech_normalization_gain_db",
    "negative_speech_normalization_gain_db",
    "noise_normalization_gain_db",
    "common_peak_scale",
    "positive_peak",
    "time_compression_applied",
    "time_compression_rate",
    "positive_right_pad_samples",
    "negative_peak",
    "anc_tokenization",
    "pos_tokenization",
    "neg_tokenization",
    "anc_length",
    "pos_length",
    "neg_length",
    "anc_pos_edit_distance",
    "anc_pos_normalized_edit_distance",
    "anc_pos_edit_similarity",
    "anc_pos_token_set_jaccard",
    "anc_pos_exact_match",
    "anc_neg_edit_distance",
    "anc_neg_normalized_edit_distance",
    "anc_neg_edit_similarity",
    "anc_neg_token_set_jaccard",
    "anc_neg_exact_match",
    "pos_neg_edit_distance",
    "pos_neg_normalized_edit_distance",
    "pos_neg_edit_similarity",
    "pos_neg_token_set_jaccard",
    "pos_neg_exact_match",
    "negative_distance_margin",
    "positive_similarity_margin",
    "triplet_correct",
    "triplet_tie",
]


def pair_metrics(
    first: Sequence[int],
    second: Sequence[int],
) -> Dict[str, float]:
    distance = edit_distance(first, second)
    normalized_distance = distance / max(len(first), len(second), 1)
    return {
        "distance": float(distance),
        "normalized_distance": float(normalized_distance),
        "similarity": float(1.0 - normalized_distance),
        "jaccard": float(token_jaccard(first, second)),
        "exact": float(first == second),
    }


@torch.no_grad()
def evaluate_with_negatives(
    model,
    loader: DataLoader,
    feature_extractor,
    output_csv: str,
    summary_json: str,
    summary_csv: str,
    device: torch.device,
    max_len: int,
    beam_size: int,
    deduplicate: Optional[bool],
    run_metadata: Dict[str, Any],
) -> None:
    model.eval().to(device)
    if feature_extractor is not None:
        feature_extractor.eval()

    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(summary_json) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(summary_csv) or ".", exist_ok=True)

    collections: Dict[str, List[float]] = {
        "anchor_length": [],
        "positive_length": [],
        "negative_length": [],
        "anchor_positive_normalized_edit_distance": [],
        "anchor_negative_normalized_edit_distance": [],
        "positive_negative_normalized_edit_distance": [],
        "anchor_positive_edit_similarity": [],
        "anchor_negative_edit_similarity": [],
        "positive_negative_edit_similarity": [],
        "anchor_positive_token_set_jaccard": [],
        "anchor_negative_token_set_jaccard": [],
        "positive_negative_token_set_jaccard": [],
        "negative_distance_margin": [],
        "positive_similarity_margin": [],
        "triplet_correct": [],
        "triplet_tie": [],
        "achieved_snr_db": [],
        "shared_phone_overlap_units": [],
    }
    anc_pos_exact: List[float] = []
    anc_neg_exact: List[float] = []
    pos_neg_exact: List[float] = []
    written = 0

    with open(output_csv, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()

        progress = tqdm(
            loader,
            desc="TIMIT external-noise controlled-negative analysis",
            unit="batch",
        )

        for batch in progress:
            anchor = batch["anchor_wav"]
            positive = batch["positive_wav"]
            negative = batch["negative_wav"]

            if feature_extractor is not None:
                anchor = feature_extractor(anchor)
                positive = feature_extractor(positive)
                negative = feature_extractor(negative)

            anchor = anchor.to(device, non_blocking=True).contiguous()
            positive = positive.to(device, non_blocking=True).contiguous()
            negative = negative.to(device, non_blocking=True).contiguous()

            anchor_tokens = tokenize(
                model, anchor, max_len, beam_size, deduplicate
            )
            positive_tokens = tokenize(
                model, positive, max_len, beam_size, deduplicate
            )
            negative_tokens = tokenize(
                model, negative, max_len, beam_size, deduplicate
            )

            if not (
                len(anchor_tokens)
                == len(positive_tokens)
                == len(negative_tokens)
                == len(batch["metadata"])
            ):
                raise RuntimeError(
                    "Anchor/positive/negative batch-size mismatch."
                )

            for meta, anc, pos, neg in zip(
                batch["metadata"],
                anchor_tokens,
                positive_tokens,
                negative_tokens,
            ):
                ap = pair_metrics(anc, pos)
                an = pair_metrics(anc, neg)
                pn = pair_metrics(pos, neg)

                distance_margin = (
                    an["normalized_distance"] - ap["normalized_distance"]
                )
                similarity_margin = ap["similarity"] - an["similarity"]
                correct = float(
                    ap["normalized_distance"] < an["normalized_distance"]
                )
                tie = float(
                    math.isclose(
                        ap["normalized_distance"],
                        an["normalized_distance"],
                        rel_tol=0.0,
                        abs_tol=1e-12,
                    )
                )

                writer.writerow(
                    {
                        **meta,
                        "requested_snr_db": f"{meta['requested_snr_db']:.8f}",
                        "anchor_achieved_snr_db": (
                            f"{meta['anchor_achieved_snr_db']:.8f}"
                        ),
                        "negative_achieved_snr_db": (
                            f"{meta['negative_achieved_snr_db']:.8f}"
                        ),
                        "anchor_speech_normalization_gain_db": (
                            f"{meta['anchor_speech_normalization_gain_db']:.8f}"
                        ),
                        "negative_speech_normalization_gain_db": (
                            f"{meta['negative_speech_normalization_gain_db']:.8f}"
                        ),
                        "noise_normalization_gain_db": (
                            f"{meta['noise_normalization_gain_db']:.8f}"
                        ),
                        "common_peak_scale": (
                            f"{meta['common_peak_scale']:.8f}"
                        ),
                        "positive_peak": f"{meta['positive_peak']:.8f}",
                        "negative_peak": f"{meta['negative_peak']:.8f}",
                        "anc_tokenization": serialize_tokens(anc),
                        "pos_tokenization": serialize_tokens(pos),
                        "neg_tokenization": serialize_tokens(neg),
                        "anc_length": len(anc),
                        "pos_length": len(pos),
                        "neg_length": len(neg),
                        "anc_pos_edit_distance": int(ap["distance"]),
                        "anc_pos_normalized_edit_distance": (
                            f"{ap['normalized_distance']:.8f}"
                        ),
                        "anc_pos_edit_similarity": (
                            f"{ap['similarity']:.8f}"
                        ),
                        "anc_pos_token_set_jaccard": (
                            f"{ap['jaccard']:.8f}"
                        ),
                        "anc_pos_exact_match": int(ap["exact"]),
                        "anc_neg_edit_distance": int(an["distance"]),
                        "anc_neg_normalized_edit_distance": (
                            f"{an['normalized_distance']:.8f}"
                        ),
                        "anc_neg_edit_similarity": (
                            f"{an['similarity']:.8f}"
                        ),
                        "anc_neg_token_set_jaccard": (
                            f"{an['jaccard']:.8f}"
                        ),
                        "anc_neg_exact_match": int(an["exact"]),
                        "pos_neg_edit_distance": int(pn["distance"]),
                        "pos_neg_normalized_edit_distance": (
                            f"{pn['normalized_distance']:.8f}"
                        ),
                        "pos_neg_edit_similarity": (
                            f"{pn['similarity']:.8f}"
                        ),
                        "pos_neg_token_set_jaccard": (
                            f"{pn['jaccard']:.8f}"
                        ),
                        "pos_neg_exact_match": int(pn["exact"]),
                        "negative_distance_margin": (
                            f"{distance_margin:.8f}"
                        ),
                        "positive_similarity_margin": (
                            f"{similarity_margin:.8f}"
                        ),
                        "triplet_correct": int(correct),
                        "triplet_tie": int(tie),
                    }
                )

                collections["anchor_length"].append(float(len(anc)))
                collections["positive_length"].append(float(len(pos)))
                collections["negative_length"].append(float(len(neg)))
                collections[
                    "anchor_positive_normalized_edit_distance"
                ].append(ap["normalized_distance"])
                collections[
                    "anchor_negative_normalized_edit_distance"
                ].append(an["normalized_distance"])
                collections[
                    "positive_negative_normalized_edit_distance"
                ].append(pn["normalized_distance"])
                collections["anchor_positive_edit_similarity"].append(
                    ap["similarity"]
                )
                collections["anchor_negative_edit_similarity"].append(
                    an["similarity"]
                )
                collections["positive_negative_edit_similarity"].append(
                    pn["similarity"]
                )
                collections["anchor_positive_token_set_jaccard"].append(
                    ap["jaccard"]
                )
                collections["anchor_negative_token_set_jaccard"].append(
                    an["jaccard"]
                )
                collections["positive_negative_token_set_jaccard"].append(
                    pn["jaccard"]
                )
                collections["negative_distance_margin"].append(
                    distance_margin
                )
                collections["positive_similarity_margin"].append(
                    similarity_margin
                )
                collections["triplet_correct"].append(correct)
                collections["triplet_tie"].append(tie)
                collections["achieved_snr_db"].append(
                    float(meta["anchor_achieved_snr_db"])
                )
                collections["shared_phone_overlap_units"].append(
                    float(meta["shared_phone_overlap_units"])
                )
                anc_pos_exact.append(ap["exact"])
                anc_neg_exact.append(an["exact"])
                pos_neg_exact.append(pn["exact"])
                written += 1

            progress.set_postfix(written=written)

    if written == 0:
        raise RuntimeError("No triplets were evaluated.")

    summary = dict(run_metadata)
    summary.update(
        {
            "num_triplets": written,
            "output_csv": os.path.abspath(output_csv),
            "triplet_accuracy": float(
                np.mean(collections["triplet_correct"])
            ),
            "triplet_tie_rate": float(
                np.mean(collections["triplet_tie"])
            ),
            "anchor_positive_exact_match_rate": float(
                np.mean(anc_pos_exact)
            ),
            "anchor_negative_exact_match_rate": float(
                np.mean(anc_neg_exact)
            ),
            "positive_negative_exact_match_rate": float(
                np.mean(pos_neg_exact)
            ),
        }
    )
    for prefix, values in collections.items():
        summary.update(describe(values, prefix))

    with open(summary_json, "w") as handle:
        json.dump(summary, handle, indent=2)

    table_row = {
        "run_name": summary["run_name"],
        "model_type": summary["model_type"],
        "negative_mode": summary["selected_negative_mode"],
        "num_triplets": written,
        "anchor_positive_edit_similarity": summary[
            "anchor_positive_edit_similarity_mean"
        ],
        "anchor_negative_edit_similarity": summary[
            "anchor_negative_edit_similarity_mean"
        ],
        "positive_negative_edit_similarity": summary[
            "positive_negative_edit_similarity_mean"
        ],
        "anchor_positive_jaccard": summary[
            "anchor_positive_token_set_jaccard_mean"
        ],
        "anchor_negative_jaccard": summary[
            "anchor_negative_token_set_jaccard_mean"
        ],
        "positive_negative_jaccard": summary[
            "positive_negative_token_set_jaccard_mean"
        ],
        "negative_distance_margin": summary[
            "negative_distance_margin_mean"
        ],
        "positive_similarity_margin": summary[
            "positive_similarity_margin_mean"
        ],
        "triplet_accuracy": summary["triplet_accuracy"],
        "triplet_tie_rate": summary["triplet_tie_rate"],
        "anchor_positive_exact_match_rate": summary[
            "anchor_positive_exact_match_rate"
        ],
        "anchor_negative_exact_match_rate": summary[
            "anchor_negative_exact_match_rate"
        ],
        "mean_anchor_length": summary["anchor_length_mean"],
        "mean_positive_length": summary["positive_length_mean"],
        "mean_negative_length": summary["negative_length_mean"],
        "mean_shared_phone_overlap_units": summary[
            "shared_phone_overlap_units_mean"
        ],
        "mean_achieved_snr_db": summary["achieved_snr_db_mean"],
    }

    with open(summary_csv, "w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(table_row.keys()),
        )
        writer.writeheader()
        writer.writerow(table_row)

    print(f"Saved per-triplet CSV: {output_csv}")
    print(f"Saved summary JSON: {summary_json}")
    print(f"Saved table summary CSV: {summary_csv}")
    print(json.dumps(summary, indent=2))


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate TIMIT + external noise with rearrange, phone-disjoint, "
            "or phone-trigram-disjoint negatives."
        )
    )
    parser.add_argument("--root_dir", default=".")
    parser.add_argument("--config", default="mainFull.yaml")
    parser.add_argument(
        "--split",
        choices=["train", "valid", "test"],
        default="valid",
    )
    parser.add_argument("--path_col", default="path")
    parser.add_argument("--noise_dataset_dir", default=DEFAULT_NOISE_DIR)
    parser.add_argument("--hf_cache_dir", default=None)

    parser.add_argument("--ckpt", required=True)
    parser.add_argument(
        "--model_type",
        choices=["AudioTokenizer", "AudioTokenizerAR_FINAL"],
        default="AudioTokenizerAR_FINAL",
    )
    parser.add_argument("--run_name", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--summary_json", default=None)
    parser.add_argument("--summary_csv", default=None)

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_items", type=int, default=None)
    parser.add_argument("--context_sec", type=float, default=None)

    parser.add_argument("--snr_db_min", type=float, default=15.0)
    parser.add_argument("--snr_db_max", type=float, default=30.0)
    parser.add_argument("--target_rms_dbfs", type=float, default=-25.0)
    parser.add_argument("--peak_limit", type=float, default=0.99)
    parser.add_argument("--gate_db_below_peak", type=float, default=40.0)

    parser.add_argument("--max_len", type=int, default=60)
    parser.add_argument("--beam_size", type=int, default=4)
    parser.add_argument(
        "--deduplicate",
        choices=["auto", "true", "false"],
        default="auto",
    )
    parser.add_argument(
        "--feats_type",
        choices=["raw", "spectrogram", "melspectrogram"],
        default="melspectrogram",
    )
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--use_time_compression", action=argparse.BooleanOptionalAction, default=False, help="Slightly compress only the positive view and right-pad it back to 3 seconds.")

    parser.add_argument(
        "--negative_mode",
        choices=["rearrange", "phone", "trigram"],
        required=True,
        help=(
            "rearrange: within-anchor chunk permutation; phone: no shared "
            "nonsilence phone labels; trigram: no shared nonsilence phone "
            "trigrams."
        ),
    )
    parser.add_argument(
        "--rearrange_chunks",
        type=int,
        default=12,
        help="Number of chunks for rearrange mode (default: 12).",
    )
    parser.add_argument(
        "--silence_phones",
        default="h#,pau,epi",
        help="Comma-separated PHN labels ignored in phone overlap checks.",
    )
    parser.add_argument(
        "--no_phn_negative_policy",
        choices=["error", "skip", "min_overlap"],
        default="skip",
        help=(
            "Behavior when no strict phone/trigram-disjoint candidate exists. "
            "skip preserves strictness; min_overlap is an explicit fallback."
        ),
    )
    args = parser.parse_args()

    if args.snr_db_max < args.snr_db_min:
        raise ValueError("snr_db_max must be >= snr_db_min")
    if args.rearrange_chunks < 2:
        raise ValueError("rearrange_chunks must be at least 2")

    config_path = resolve_config_path(args.root_dir, args.config)
    with open(config_path, "r") as handle:
        cfg = yaml.load(handle, Loader=yaml.FullLoader)

    if args.split not in cfg:
        raise ValueError(f"Split {args.split!r} not found in config")

    timit_csv = resolve_dataset_csv(
        args.root_dir,
        config_path,
        cfg[args.split]["dataset_csv"],
    )
    timit_paths = read_audio_paths(timit_csv, args.path_col)
    if args.max_items is not None:
        timit_paths = timit_paths[: int(args.max_items)]

    if len(timit_paths) < 1:
        raise ValueError("No TIMIT files are available.")

    seed_everything(args.seed, workers=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    sample_rate = int(cfg["fs"])
    context_sec = (
        float(args.context_sec)
        if args.context_sec is not None
        else float(cfg["seglen"])
    )
    batch_size = (
        int(args.batch_size)
        if args.batch_size is not None
        else int(cfg["batchsize"])
    )

    phn_exists = [
        find_existing_phn_path(path) is not None for path in timit_paths
    ]
    all_phn = bool(timit_paths) and all(phn_exists)
    any_phn = any(phn_exists)

    if args.negative_mode in {"phone", "trigram"} and not all_phn:
        missing = [
            canonical_phn_path(path)
            for path, exists in zip(timit_paths, phn_exists)
            if not exists
        ]
        preview = "\n  ".join(missing[:10])
        raise FileNotFoundError(
            f"--negative_mode {args.negative_mode} requires a PHN sidecar "
            "for every selected TIMIT file. Missing:\n  "
            + preview
            + ("\n  ..." if len(missing) > 10 else "")
        )

    silence_phones = {
        phone.strip()
        for phone in args.silence_phones.split(",")
        if phone.strip()
    }

    segment_metadata = build_segment_metadata(
        paths=timit_paths,
        target_sample_rate=sample_rate,
        context_sec=context_sec,
        seed=args.seed,
        split=args.split,
        negative_mode=args.negative_mode,
        silence_phones=silence_phones,
    )

    negative_map: Dict[int, int] = {}
    phn_sampling_diagnostics: Dict[str, Any] = {}

    if args.negative_mode in {"phone", "trigram"}:
        negative_map, skipped, phn_sampling_diagnostics = (
            build_phn_negative_map(
                metadata=segment_metadata,
                seed=args.seed,
                no_candidate_policy=args.no_phn_negative_policy,
            )
        )
        anchor_indices = sorted(negative_map.keys())
        if not anchor_indices:
            raise RuntimeError(
                "No PHN-controlled negative pairs are available under the "
                "selected overlap rule."
            )
    else:
        anchor_indices = list(range(len(segment_metadata)))

    parquet_files = find_parquet_files(args.noise_dataset_dir)
    noise_dataset = load_local_noise_dataset(
        parquet_files,
        args.hf_cache_dir,
    )
    noise_rows, noise_diagnostics = index_noise_only(noise_dataset)

    dataset = TimitControlledNegativeDataset(
        metadata=segment_metadata,
        anchor_indices=anchor_indices,
        negative_mode=args.negative_mode,
        negative_map=negative_map,
        noise_dataset=noise_dataset,
        noise_rows=noise_rows,
        noise_dataset_dir=args.noise_dataset_dir,
        sample_rate=sample_rate,
        context_sec=context_sec,
        seed=args.seed,
        snr_min=args.snr_db_min,
        snr_max=args.snr_db_max,
        target_rms_dbfs=args.target_rms_dbfs,
        peak_limit=args.peak_limit,
        gate_db=args.gate_db_below_peak,
        rearrange_chunks=args.rearrange_chunks,
        use_time_compression=args.use_time_compression,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_triplets,
    )

    rf_size = int(round(float(cfg["win_size"]) * sample_rate))
    stride = int(round(float(cfg["win_hop"]) * sample_rate))
    n_mels = int(cfg["n_mels"])
    feature_extractor = build_feature_extractor(
        sample_rate,
        args.feats_type,
        rf_size,
        stride,
        n_mels,
    )

    device = torch.device(
        f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    )
    model = build_model(args.ckpt, args.model_type, args.device)
    deduplicate = parse_deduplicate(args.deduplicate)

    stem, _ = os.path.splitext(args.output_csv)
    summary_json = args.summary_json or stem + "_summary.json"
    summary_csv = args.summary_csv or stem + "_summary.csv"

    negative_definition = {
        "rearrange": (
            f"anchor waveform rearranged with a {args.rearrange_chunks}-chunk "
            "Sattolo derangement + same external noise segment and SNR"
        ),
        "phone": (
            "different TIMIT crop with zero shared nonsilence phone labels + "
            "same external noise segment and SNR"
        ),
        "trigram": (
            "different TIMIT crop with zero shared nonsilence phone trigrams + "
            "same external noise segment and SNR"
        ),
    }[args.negative_mode]

    run_metadata = {
        "run_name": args.run_name,
        "model_type": args.model_type,
        "checkpoint": os.path.abspath(args.ckpt),
        "config_path": os.path.abspath(config_path),
        "timit_csv": os.path.abspath(timit_csv),
        "noise_dataset_dir": os.path.abspath(args.noise_dataset_dir),
        "noise_index": noise_diagnostics,
        "num_timit_files": len(timit_paths),
        "num_anchor_items": len(anchor_indices),
        "num_triplets_selected": len(dataset),
        "split": args.split,
        "seed": args.seed,
        "sample_rate": sample_rate,
        "context_sec": context_sec,
        "snr_db_min": args.snr_db_min,
        "snr_db_max": args.snr_db_max,
        "target_rms_dbfs": args.target_rms_dbfs,
        "peak_limit": args.peak_limit,
        "gate_db_below_peak": args.gate_db_below_peak,
        "feature_type": args.feats_type,
        "max_len": args.max_len,
        "beam_size": args.beam_size,
        "deduplicate": args.deduplicate,
        "requested_negative_mode": args.negative_mode,
        "selected_negative_mode": args.negative_mode,
        "all_phn_files_available": all_phn,
        "any_phn_files_available": any_phn,
        "phone_overlap_unit": (
            "phone"
            if args.negative_mode == "phone"
            else "trigram"
            if args.negative_mode == "trigram"
            else None
        ),
        "phone_ngram": 3 if args.negative_mode == "trigram" else None,
        "silence_phones": sorted(silence_phones),
        "no_phn_negative_policy": args.no_phn_negative_policy,
        "phn_sampling": phn_sampling_diagnostics,
        "rearrange_chunks": args.rearrange_chunks,
        "positive_definition": (
            "same TIMIT crop + RMS-normalized external noise_only recording"
        ),
        "negative_definition": negative_definition,
        "noise_matching": (
            "positive and negative use the same external noise segment and SNR"
        ),
        "additional_augmentation": ("positive_time_compression" if args.use_time_compression else "none"),
    }

    print("==== TIMIT external-noise controlled-negative analysis ====")
    print(json.dumps(run_metadata, indent=2))

    evaluate_with_negatives(
        model=model,
        loader=loader,
        feature_extractor=feature_extractor,
        output_csv=args.output_csv,
        summary_json=summary_json,
        summary_csv=summary_csv,
        device=device,
        max_len=args.max_len,
        beam_size=args.beam_size,
        deduplicate=deduplicate,
        run_metadata=run_metadata,
    )


if __name__ == "__main__":
    main()
