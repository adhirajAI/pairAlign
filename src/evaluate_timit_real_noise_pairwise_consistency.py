#!/usr/bin/env python3
"""Pairwise consistency on TIMIT with real noise-only recordings.

Anchor:
    A deterministic 3-second TIMIT crop.

Positive:
    The same crop mixed with one `noise_only` recording from
    haydarkadioglu/speech-noise-dataset.

No other augmentation is applied. Speech and noise are independently
active-RMS normalized, the noise is attenuated to the requested SNR, and a
shared peak-protection scale is applied to both the clean anchor and mixture.
Thus, the pair differs only by the added real noise.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import random
import warnings
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from lightning.pytorch import seed_everything
from tqdm.auto import tqdm

import torchaudio
import torchaudio.functional as AF
import torchaudio.transforms as T
import yaml

try:
    from datasets import Audio, load_dataset
except ImportError as exc:
    raise ImportError(
        "Install dependencies with: pip install -U 'datasets[audio]' pyarrow soundfile"
    ) from exc

from train import AudioTokenizer, AudioTokenizerAR_FINAL
from utils import AudioAugmenter, AugConfig


DEFAULT_NOISE_DIR = "data/speech-noise-dataset"


# -----------------------------------------------------------------------------
# Paths and basic utilities
# -----------------------------------------------------------------------------


def resolve_config_path(root_dir: str, config_arg: str) -> str:
    path = config_arg if os.path.isabs(config_arg) else os.path.join(root_dir, "config", config_arg)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Config not found: {path}")
    return os.path.abspath(path)


def resolve_dataset_csv(root_dir: str, config_path: str, csv_arg: str) -> str:
    candidates = [
        csv_arg,
        os.path.join(root_dir, csv_arg),
        os.path.join(os.path.dirname(config_path), csv_arg),
    ]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return os.path.abspath(candidate)
    raise FileNotFoundError("TIMIT CSV not found. Tried:\n  " + "\n  ".join(candidates))


def read_audio_paths(csv_path: str, path_col: str) -> List[str]:
    paths: List[str] = []
    with open(csv_path, "r", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or path_col not in reader.fieldnames:
            raise ValueError(f"CSV must contain `{path_col}`. Found: {reader.fieldnames}")
        for row in reader:
            raw = str(row[path_col]).strip()
            if not raw:
                continue
            path = os.path.expanduser(raw)
            if not os.path.isabs(path):
                path = os.path.join(os.path.dirname(csv_path), path)
            path = os.path.abspath(path)
            if not os.path.isfile(path):
                warnings.warn(f"Skipping missing TIMIT file: {path}")
                continue
            paths.append(path)
    if not paths:
        raise ValueError(f"No valid TIMIT files found in {csv_path}")
    return paths


def find_parquet_files(dataset_dir: str) -> List[str]:
    root = Path(dataset_dir).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Noise dataset directory not found: {root}")
    files = sorted(str(path) for path in root.rglob("*.parquet") if ".cache" not in path.parts)
    if not files:
        raise FileNotFoundError(f"No Parquet files found under {root}")
    return files


def stable_int(text: str) -> int:
    digest = hashlib.sha1(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def file_id(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def serialize_tokens(tokens: Sequence[int]) -> str:
    return " ".join(str(int(token)) for token in tokens)


def normalize_token_batch(output: Any) -> List[List[int]]:
    def convert(item: Any) -> List[int]:
        if isinstance(item, torch.Tensor):
            return [int(x) for x in item.detach().cpu().reshape(-1).tolist()]
        if isinstance(item, np.ndarray):
            return [int(x) for x in item.reshape(-1).tolist()]
        if isinstance(item, (list, tuple)):
            return [int(x) for x in item]
        raise TypeError(f"Unsupported token sequence: {type(item)}")

    if isinstance(output, torch.Tensor):
        output = output.detach().cpu()
        return [convert(output)] if output.ndim == 1 else [convert(output[i]) for i in range(output.size(0))]
    if isinstance(output, np.ndarray):
        return [convert(output)] if output.ndim == 1 else [convert(output[i]) for i in range(output.shape[0])]
    if isinstance(output, (list, tuple)):
        return [convert(item) for item in output]
    if isinstance(output, dict):
        for key in ("tokens", "token_ids", "sequences", "seqs"):
            if key in output:
                return normalize_token_batch(output[key])
    raise TypeError(f"Unsupported tokenize output: {type(output)}")


# -----------------------------------------------------------------------------
# Edit-distance diagnostics
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
                cost = 0 if a[i - 1] == b[j - 1] else 1
                current[j] = min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost)
            previous, current = current, previous
        return previous[m]


def token_jaccard(a: Sequence[int], b: Sequence[int]) -> float:
    first, second = set(a), set(b)
    union = first | second
    return 1.0 if not union else len(first & second) / len(union)


# -----------------------------------------------------------------------------
# Audio decoding
# -----------------------------------------------------------------------------


def channel_first_mono(waveform: torch.Tensor) -> torch.Tensor:
    waveform = torch.as_tensor(waveform).float()
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    elif waveform.ndim != 2:
        raise ValueError(f"Expected [channels,time] or [time], got {tuple(waveform.shape)}")
    if waveform.size(0) > 8 and waveform.size(1) <= 8:
        waveform = waveform.transpose(0, 1)
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform.contiguous()


def load_audio(path: str) -> Tuple[torch.Tensor, int]:
    waveform, sr = torchaudio.load(path)
    return channel_first_mono(waveform), int(sr)


def decode_hf_audio(audio_value: Any, dataset_dir: str, filename: str) -> Tuple[torch.Tensor, int]:
    if hasattr(audio_value, "get_all_samples"):
        samples = audio_value.get_all_samples()
        return channel_first_mono(samples.data), int(samples.sample_rate)

    audio_bytes = None
    audio_path = None
    if isinstance(audio_value, dict):
        audio_bytes = audio_value.get("bytes")
        audio_path = audio_value.get("path")
    elif isinstance(audio_value, (bytes, bytearray, memoryview)):
        audio_bytes = bytes(audio_value)
    elif isinstance(audio_value, str):
        audio_path = audio_value
    else:
        raise TypeError(f"Unsupported Hugging Face audio value: {type(audio_value)}")

    if audio_bytes is not None:
        buffer = io.BytesIO(bytes(audio_bytes))
        try:
            waveform, sr = torchaudio.load(buffer)
            return channel_first_mono(waveform), int(sr)
        except Exception as first_error:
            try:
                import soundfile as sf
                buffer.seek(0)
                array, sr = sf.read(buffer, dtype="float32", always_2d=True)
                return channel_first_mono(torch.from_numpy(array.T.copy())), int(sr)
            except Exception as second_error:
                raise RuntimeError(
                    f"Could not decode embedded noise audio. torchaudio={first_error}; soundfile={second_error}"
                ) from second_error

    candidates: List[Path] = []
    if audio_path:
        raw = Path(str(audio_path))
        candidates.append(raw if raw.is_absolute() else Path(dataset_dir) / str(audio_path).replace("\\", "/"))
    candidates.append(Path(dataset_dir) / str(filename).replace("\\", "/"))
    for candidate in candidates:
        if candidate.is_file():
            return load_audio(str(candidate))
    raise FileNotFoundError(f"Noise audio unavailable: filename={filename!r}, path={audio_path!r}")


def resample(waveform: torch.Tensor, source_sr: int, target_sr: int) -> torch.Tensor:
    return waveform if source_sr == target_sr else AF.resample(waveform, source_sr, target_sr)


# -----------------------------------------------------------------------------
# Local Hugging Face noise-only dataset
# -----------------------------------------------------------------------------


def load_local_noise_dataset(parquet_files: Sequence[str], cache_dir: Optional[str]):
    dataset = load_dataset(
        "parquet",
        data_files={"train": list(parquet_files)},
        split="train",
        cache_dir=cache_dir,
        keep_in_memory=False,
    )
    required = {"filename", "label", "audio"}
    missing = required.difference(dataset.column_names)
    if missing:
        raise ValueError(f"Missing dataset columns {sorted(missing)}; found {dataset.column_names}")
    try:
        return dataset.decode(False)
    except (AttributeError, TypeError):
        return dataset.cast_column("audio", Audio(decode=False))


def is_noise_only(label: Any, filename: str) -> bool:
    label_text = str(label).strip().lower()
    normalized = str(filename).replace("\\", "/").lower()
    parts = PurePosixPath(normalized).parts
    return label_text == "noise_only" or (parts and parts[0] == "noise_only")


def index_noise_only(dataset) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    metadata = dataset.remove_columns(["audio"])
    rows: List[Dict[str, Any]] = []
    labels: Dict[str, int] = {}
    for index, row in enumerate(tqdm(metadata, desc="Indexing noise_only", unit="row")):
        label = str(row["label"]).strip()
        filename = str(row["filename"]).replace("\\", "/").strip()
        labels[label] = labels.get(label, 0) + 1
        if is_noise_only(label, filename):
            rows.append({"row_index": index, "filename": filename, "label": label})
    if not rows:
        raise RuntimeError(f"No noise_only rows found. Labels: {labels}")
    return rows, {"num_dataset_rows": len(dataset), "num_noise_only_rows": len(rows), "label_counts": labels}


# -----------------------------------------------------------------------------
# Deterministic crop and loudness-normalized mixing
# -----------------------------------------------------------------------------


def crop_start(num_samples: int, target_len: int, seed: int, index: int, split: str) -> int:
    if num_samples <= target_len:
        return 0
    maximum = num_samples - target_len
    if split == "train":
        return random.Random(seed + 1_000_003 * index).randint(0, maximum)
    return maximum // 2


def crop_or_pad(waveform: torch.Tensor, start: int, target_len: int) -> Tuple[torch.Tensor, int]:
    crop = waveform[..., start : start + target_len]
    valid = int(crop.size(-1))
    if valid < target_len:
        crop = F.pad(crop, (0, target_len - valid))
    return crop.contiguous(), valid


def noise_crop_or_repeat(waveform: torch.Tensor, target_len: int, seed: int, index: int) -> Tuple[torch.Tensor, int]:
    length = int(waveform.size(-1))
    if length <= 0:
        raise ValueError("Empty noise recording")
    rng = random.Random(seed + 2_000_033 * index)
    if length < target_len:
        waveform = waveform.repeat(1, math.ceil(target_len / length))
        length = int(waveform.size(-1))
    maximum = length - target_len
    start = rng.randint(0, maximum) if maximum > 0 else 0
    return waveform[..., start : start + target_len].contiguous(), start


def remove_dc(waveform: torch.Tensor, valid_samples: Optional[int]) -> torch.Tensor:
    region = waveform if valid_samples is None else waveform[..., : max(1, min(valid_samples, waveform.size(-1)))]
    return waveform - region.mean(dim=-1, keepdim=True)


def active_rms(waveform: torch.Tensor, valid_samples: Optional[int], gate_db: float) -> torch.Tensor:
    region = waveform.reshape(-1) if valid_samples is None else waveform[..., : max(1, min(valid_samples, waveform.size(-1)))].reshape(-1)
    absolute = region.abs()
    peak = absolute.max().clamp_min(1e-8)
    threshold = peak * (10.0 ** (-gate_db / 20.0))
    active = region[absolute >= threshold]
    if active.numel() < max(32, int(0.01 * region.numel())):
        active = region
    return active.pow(2).mean().sqrt().clamp_min(1e-8)


def normalize_rms(waveform: torch.Tensor, target_dbfs: float, valid_samples: Optional[int], gate_db: float) -> Tuple[torch.Tensor, float]:
    current = active_rms(waveform, valid_samples, gate_db)
    target = 10.0 ** (target_dbfs / 20.0)
    gain = target / float(current.item())
    return waveform * gain, 20.0 * math.log10(max(gain, 1e-12))


def mix_pair(
    speech: torch.Tensor,
    noise: torch.Tensor,
    valid_speech_samples: int,
    target_rms_dbfs: float,
    snr_db: float,
    peak_limit: float,
    gate_db: float,
) -> Dict[str, Any]:
    speech = remove_dc(speech, valid_speech_samples)
    noise = remove_dc(noise, None)
    speech, speech_gain_db = normalize_rms(speech, target_rms_dbfs, valid_speech_samples, gate_db)
    noise, noise_gain_db = normalize_rms(noise, target_rms_dbfs, None, gate_db)
    scaled_noise = noise * (10.0 ** (-snr_db / 20.0))
    mixture = speech + scaled_noise
    common_scale = min(1.0, peak_limit / max(float(mixture.abs().max().item()), 1e-12))
    speech = (speech * common_scale).contiguous()
    mixture = (mixture * common_scale).contiguous()
    scaled_noise = (scaled_noise * common_scale).contiguous()
    achieved = 20.0 * math.log10(
        max(
            float(active_rms(speech, valid_speech_samples, gate_db).item())
            / max(float(active_rms(scaled_noise, None, gate_db).item()), 1e-12),
            1e-12,
        )
    )
    return {
        "anchor": speech,
        "positive": mixture,
        "speech_gain_db": speech_gain_db,
        "noise_gain_db": noise_gain_db,
        "requested_snr_db": snr_db,
        "achieved_snr_db": achieved,
        "common_peak_scale": common_scale,
        "mixture_peak": float(mixture.abs().max().item()),
    }


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------


class TimitNoisePairDataset(Dataset):
    def __init__(
        self,
        timit_paths: Sequence[str],
        noise_dataset,
        noise_rows: Sequence[Dict[str, Any]],
        noise_dataset_dir: str,
        sample_rate: int,
        context_sec: float,
        split: str,
        seed: int,
        snr_min: float,
        snr_max: float,
        target_rms_dbfs: float,
        peak_limit: float,
        gate_db: float,
        max_items: Optional[int],
        use_time_compression: bool = False,
    ):
        self.timit_paths = list(timit_paths)[:max_items] if max_items is not None else list(timit_paths)
        self.noise_dataset = noise_dataset
        self.noise_rows = list(noise_rows)
        self.noise_dataset_dir = noise_dataset_dir
        self.sample_rate = sample_rate
        self.target_len = int(round(context_sec * sample_rate))
        self.split = split
        self.seed = seed
        self.snr_min = snr_min
        self.snr_max = snr_max
        self.target_rms_dbfs = target_rms_dbfs
        self.peak_limit = peak_limit
        self.gate_db = gate_db
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

    def __len__(self) -> int:
        return len(self.timit_paths)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        speech_path = self.timit_paths[index]
        speech, speech_sr = load_audio(speech_path)
        speech = resample(speech, speech_sr, self.sample_rate)
        speech_start = crop_start(speech.size(-1), self.target_len, self.seed, index, self.split)
        speech, valid_speech = crop_or_pad(speech, speech_start, self.target_len)

        noise_rng = random.Random(self.seed + 3_000_017 * index)
        noise_meta = self.noise_rows[noise_rng.randrange(len(self.noise_rows))]
        noise_row = self.noise_dataset[noise_meta["row_index"]]
        noise, noise_sr = decode_hf_audio(noise_row["audio"], self.noise_dataset_dir, noise_meta["filename"])
        noise = resample(noise, noise_sr, self.sample_rate)
        noise, noise_start = noise_crop_or_repeat(noise, self.target_len, self.seed, index)

        snr = random.Random(self.seed + 4_000_037 * index).uniform(self.snr_min, self.snr_max)
        mixed = mix_pair(
            speech=speech,
            noise=noise,
            valid_speech_samples=valid_speech,
            target_rms_dbfs=self.target_rms_dbfs,
            snr_db=snr,
            peak_limit=self.peak_limit,
            gate_db=self.gate_db,
        )

        compression_info = {
            "time_compression_applied": False,
            "time_compression_rate": 1.0,
            "positive_right_pad_samples": 0,
        }
        positive_wav = mixed["positive"]
        if self.time_compressor is not None:
            positive_wav, compression_info = self.time_compressor(
                positive_wav,
                rng=random.Random(self.seed + 5_000_081 * index),
                return_info=True,
            )

        return {
            "anchor_wav": mixed["anchor"],
            "positive_wav": positive_wav,
            "file_id": file_id(speech_path),
            "anchor_path": speech_path,
            "noise_filename": noise_meta["filename"],
            "noise_row_index": noise_meta["row_index"],
            "speech_original_sr": speech_sr,
            "noise_original_sr": noise_sr,
            "speech_crop_start": speech_start,
            "noise_crop_start": noise_start,
            "valid_speech_samples": valid_speech,
            "requested_snr_db": mixed["requested_snr_db"],
            "achieved_snr_db": mixed["achieved_snr_db"],
            "speech_normalization_gain_db": mixed["speech_gain_db"],
            "noise_normalization_gain_db": mixed["noise_gain_db"],
            "common_peak_scale": mixed["common_peak_scale"],
            "mixture_peak": mixed["mixture_peak"],
            "time_compression_applied": compression_info["time_compression_applied"],
            "time_compression_rate": compression_info["time_compression_rate"],
            "positive_right_pad_samples": compression_info["positive_right_pad_samples"],
        }


def collate_pairs(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "anchor_wav": torch.stack([item["anchor_wav"] for item in batch]),
        "positive_wav": torch.stack([item["positive_wav"] for item in batch]),
        "metadata": [
            {key: value for key, value in item.items() if key not in {"anchor_wav", "positive_wav"}}
            for item in batch
        ],
    }


# -----------------------------------------------------------------------------
# Model and feature extraction
# -----------------------------------------------------------------------------


def build_feature_extractor(sample_rate: int, feats_type: str, rf_size: int, stride: int, n_mels: int):
    if feats_type == "melspectrogram":
        return torch.nn.Sequential(
            T.MelSpectrogram(sample_rate=sample_rate, n_fft=rf_size, hop_length=stride, n_mels=n_mels, center=True),
            T.AmplitudeToDB(),
        )
    if feats_type == "spectrogram":
        return torch.nn.Sequential(
            T.Spectrogram(n_fft=rf_size, hop_length=stride, center=True, power=2.0),
            T.AmplitudeToDB(),
        )
    if feats_type == "raw":
        return None
    raise ValueError(f"Unknown feats_type: {feats_type}")


def build_model(ckpt: str, model_type: str, device_id: int):
    if model_type == "AudioTokenizer":
        return AudioTokenizer.load_from_pretrained(ckpt, device_id)
    if model_type == "AudioTokenizerAR_FINAL":
        return AudioTokenizerAR_FINAL.load_from_pretrained(ckpt, device_id)
    raise ValueError(f"Unknown model_type: {model_type}")


def parse_deduplicate(value: str) -> Optional[bool]:
    return None if value == "auto" else value == "true"


@torch.no_grad()
def tokenize(model, features: torch.Tensor, max_len: int, beam_size: int, deduplicate: Optional[bool]) -> List[List[int]]:
    kwargs = {"max_len": max_len, "beam_size": beam_size, "strip_special_tokens": True}
    if deduplicate is not None:
        kwargs["deduplicate"] = deduplicate
    output = model.tokenize(features, **kwargs)
    tokens = normalize_token_batch(output)
    if len(tokens) != features.size(0):
        raise RuntimeError(f"Tokenizer returned {len(tokens)} sequences for batch size {features.size(0)}")
    return tokens


# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------


FIELDS = [
    "file_id", "anchor_path", "noise_filename", "noise_row_index",
    "speech_original_sr", "noise_original_sr", "speech_crop_start", "noise_crop_start",
    "valid_speech_samples", "requested_snr_db", "achieved_snr_db",
    "speech_normalization_gain_db", "noise_normalization_gain_db",
    "common_peak_scale", "mixture_peak",
    "time_compression_applied", "time_compression_rate", "positive_right_pad_samples",
    "anc_tokenization", "pos_tokenization", "anc_length", "pos_length",
    "absolute_length_difference", "pos_to_anc_length_ratio",
    "anc_pos_edit_distance", "anc_pos_normalized_edit_distance",
    "anc_pos_edit_similarity", "anc_pos_exact_match", "anc_pos_token_set_jaccard",
]


def describe(values: Sequence[float], prefix: str) -> Dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        f"{prefix}_mean": float(array.mean()),
        f"{prefix}_std": float(array.std()),
        f"{prefix}_median": float(np.median(array)),
        f"{prefix}_q25": float(np.quantile(array, 0.25)),
        f"{prefix}_q75": float(np.quantile(array, 0.75)),
    }


@torch.no_grad()
def evaluate(
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
    metadata: Dict[str, Any],
) -> None:
    model.eval().to(device)
    if feature_extractor is not None:
        feature_extractor.eval()

    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    lengths_a: List[float] = []
    lengths_p: List[float] = []
    length_diffs: List[float] = []
    length_ratios: List[float] = []
    neds: List[float] = []
    similarities: List[float] = []
    exact: List[float] = []
    jaccards: List[float] = []
    snrs: List[float] = []
    peak_scales: List[float] = []
    written = 0

    with open(output_csv, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for batch in tqdm(loader, desc="TIMIT real-noise consistency", unit="batch"):
            anchor = batch["anchor_wav"]
            positive = batch["positive_wav"]
            if feature_extractor is not None:
                anchor = feature_extractor(anchor)
                positive = feature_extractor(positive)
            anchor = anchor.to(device, non_blocking=True).contiguous()
            positive = positive.to(device, non_blocking=True).contiguous()
            anchor_tokens = tokenize(model, anchor, max_len, beam_size, deduplicate)
            positive_tokens = tokenize(model, positive, max_len, beam_size, deduplicate)

            for meta, anc, pos in zip(batch["metadata"], anchor_tokens, positive_tokens):
                ed = edit_distance(anc, pos)
                ned = ed / max(len(anc), len(pos), 1)
                ratio = len(pos) / len(anc) if len(anc) else float("nan")
                diff = abs(len(pos) - len(anc))
                similarity = 1.0 - ned
                exact_match = int(anc == pos)
                jaccard = token_jaccard(anc, pos)
                writer.writerow({
                    **meta,
                    "requested_snr_db": f"{meta['requested_snr_db']:.8f}",
                    "achieved_snr_db": f"{meta['achieved_snr_db']:.8f}",
                    "speech_normalization_gain_db": f"{meta['speech_normalization_gain_db']:.8f}",
                    "noise_normalization_gain_db": f"{meta['noise_normalization_gain_db']:.8f}",
                    "common_peak_scale": f"{meta['common_peak_scale']:.8f}",
                    "mixture_peak": f"{meta['mixture_peak']:.8f}",
                    "anc_tokenization": serialize_tokens(anc),
                    "pos_tokenization": serialize_tokens(pos),
                    "anc_length": len(anc),
                    "pos_length": len(pos),
                    "absolute_length_difference": diff,
                    "pos_to_anc_length_ratio": f"{ratio:.8f}",
                    "anc_pos_edit_distance": ed,
                    "anc_pos_normalized_edit_distance": f"{ned:.8f}",
                    "anc_pos_edit_similarity": f"{similarity:.8f}",
                    "anc_pos_exact_match": exact_match,
                    "anc_pos_token_set_jaccard": f"{jaccard:.8f}",
                })
                lengths_a.append(len(anc))
                lengths_p.append(len(pos))
                length_diffs.append(diff)
                length_ratios.append(ratio)
                neds.append(ned)
                similarities.append(similarity)
                exact.append(exact_match)
                jaccards.append(jaccard)
                snrs.append(meta["achieved_snr_db"])
                peak_scales.append(meta["common_peak_scale"])
                written += 1

    if written == 0:
        raise RuntimeError("No pairs were evaluated")

    summary = dict(metadata)
    summary.update({
        "num_pairs": written,
        "output_csv": os.path.abspath(output_csv),
        "exact_match_rate": float(np.mean(exact)),
        "fraction_peak_protected": float(np.mean(np.asarray(peak_scales) < 1.0)),
    })
    summary.update(describe(lengths_a, "anchor_length"))
    summary.update(describe(lengths_p, "positive_length"))
    summary.update(describe(length_diffs, "absolute_length_difference"))
    summary.update(describe(length_ratios, "positive_to_anchor_length_ratio"))
    summary.update(describe(neds, "normalized_edit_distance"))
    summary.update(describe(similarities, "edit_similarity"))
    summary.update(describe(jaccards, "token_set_jaccard"))
    summary.update(describe(snrs, "achieved_snr_db"))

    with open(summary_json, "w") as handle:
        json.dump(summary, handle, indent=2)

    row = {
        "run_name": summary["run_name"],
        "model_type": summary["model_type"],
        "num_pairs": written,
        "snr_db_min": summary["snr_db_min"],
        "snr_db_max": summary["snr_db_max"],
        "mean_achieved_snr_db": summary["achieved_snr_db_mean"],
        "mean_anchor_length": summary["anchor_length_mean"],
        "mean_positive_length": summary["positive_length_mean"],
        "mean_normalized_edit_distance": summary["normalized_edit_distance_mean"],
        "median_normalized_edit_distance": summary["normalized_edit_distance_median"],
        "std_normalized_edit_distance": summary["normalized_edit_distance_std"],
        "mean_edit_similarity": summary["edit_similarity_mean"],
        "exact_match_rate": summary["exact_match_rate"],
        "mean_token_set_jaccard": summary["token_set_jaccard_mean"],
        "mean_positive_to_anchor_length_ratio": summary["positive_to_anchor_length_ratio_mean"],
        "mean_absolute_length_difference": summary["absolute_length_difference_mean"],
    }
    os.makedirs(os.path.dirname(summary_csv) or ".", exist_ok=True)
    with open(summary_csv, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)

    print(f"Saved per-pair CSV: {output_csv}")
    print(f"Saved summary JSON: {summary_json}")
    print(f"Saved table summary CSV: {summary_csv}")
    print(json.dumps(summary, indent=2))


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", default=".")
    parser.add_argument("--config", default="mainFull.yaml")
    parser.add_argument("--split", choices=["train", "valid", "test"], default="valid")
    parser.add_argument("--path_col", default="path")
    parser.add_argument("--noise_dataset_dir", default=DEFAULT_NOISE_DIR)
    parser.add_argument("--hf_cache_dir", default=None)

    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--model_type", choices=["AudioTokenizer", "AudioTokenizerAR_FINAL"], default="AudioTokenizerAR_FINAL")
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
    parser.add_argument("--deduplicate", choices=["auto", "true", "false"], default="auto")
    parser.add_argument("--feats_type", choices=["raw", "spectrogram", "melspectrogram"], default="melspectrogram")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--use_time_compression", action=argparse.BooleanOptionalAction, default=False, help="Slightly compress only the positive view and right-pad it back to 3 seconds.")
    args = parser.parse_args()

    if args.snr_db_max < args.snr_db_min:
        raise ValueError("snr_db_max must be >= snr_db_min")

    config_path = resolve_config_path(args.root_dir, args.config)
    with open(config_path, "r") as handle:
        cfg = yaml.load(handle, Loader=yaml.FullLoader)
    if args.split not in cfg:
        raise ValueError(f"Split {args.split!r} not found in config")

    timit_csv = resolve_dataset_csv(args.root_dir, config_path, cfg[args.split]["dataset_csv"])
    timit_paths = read_audio_paths(timit_csv, args.path_col)

    seed_everything(args.seed, workers=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    sample_rate = int(cfg["fs"])
    context_sec = float(args.context_sec) if args.context_sec is not None else float(cfg["seglen"])
    batch_size = int(args.batch_size) if args.batch_size is not None else int(cfg["batchsize"])

    parquet_files = find_parquet_files(args.noise_dataset_dir)
    noise_dataset = load_local_noise_dataset(parquet_files, args.hf_cache_dir)
    noise_rows, noise_diagnostics = index_noise_only(noise_dataset)

    dataset = TimitNoisePairDataset(
        timit_paths=timit_paths,
        noise_dataset=noise_dataset,
        noise_rows=noise_rows,
        noise_dataset_dir=args.noise_dataset_dir,
        sample_rate=sample_rate,
        context_sec=context_sec,
        split=args.split,
        seed=args.seed,
        snr_min=args.snr_db_min,
        snr_max=args.snr_db_max,
        target_rms_dbfs=args.target_rms_dbfs,
        peak_limit=args.peak_limit,
        gate_db=args.gate_db_below_peak,
        max_items=args.max_items,
        use_time_compression=args.use_time_compression,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_pairs,
    )

    rf_size = int(round(float(cfg["win_size"]) * sample_rate))
    stride = int(round(float(cfg["win_hop"]) * sample_rate))
    n_mels = int(cfg["n_mels"])
    feature_extractor = build_feature_extractor(sample_rate, args.feats_type, rf_size, stride, n_mels)

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    model = build_model(args.ckpt, args.model_type, args.device)
    deduplicate = parse_deduplicate(args.deduplicate)

    stem, _ = os.path.splitext(args.output_csv)
    summary_json = args.summary_json or stem + "_summary.json"
    summary_csv = args.summary_csv or stem + "_summary.csv"

    metadata = {
        "run_name": args.run_name,
        "model_type": args.model_type,
        "checkpoint": os.path.abspath(args.ckpt),
        "config_path": config_path,
        "timit_csv": timit_csv,
        "noise_dataset_dir": os.path.abspath(args.noise_dataset_dir),
        "noise_index": noise_diagnostics,
        "num_timit_files": len(timit_paths),
        "num_items_selected": len(dataset),
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
        "additional_augmentation": ("positive_time_compression" if args.use_time_compression else "none"),
        "positive_definition": "same TIMIT crop + RMS-normalized real noise_only recording",
    }
    print("==== TIMIT + real noise_only pairwise consistency ====")
    print(json.dumps(metadata, indent=2))

    evaluate(
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
        metadata=metadata,
    )


if __name__ == "__main__":
    main()
