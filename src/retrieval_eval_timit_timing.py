#!/usr/bin/env python3
"""Retrieval evaluation with tokenization and retrieval efficiency metrics.

Saved metrics include archive/query mean sequence length, tokens per second of
input audio, tokenizer throughput, batch-amortized per-item latency, CUDA memory,
per-query ranking time, and a length-product proxy for quadratic edit-distance
work. Per-item tokenization timings are amortized over each batch; use
--batch_size 1 for true single-item latency.
"""

import os
import sys
import csv
import json
import math
import time
import random
import argparse
import logging
import resource
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import yaml
from tqdm.auto import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchaudio
import torchaudio.transforms as T
import torchaudio.functional as AF

from train import AudioTokenizerAR_FINAL, AudioTokenizer


# ============================================================
# Logging and generic helpers
# ============================================================

def setup_logger(log_path: str) -> logging.Logger:
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    logger = logging.getLogger("retrieval_eval")
    logger.setLevel(logging.INFO)
    logger.handlers = []
    logger.propagate = False

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cuda_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def safe_rate(numerator: float, denominator: float) -> Optional[float]:
    return float(numerator) / float(denominator) if denominator > 0 else None


def percentile(values: Sequence[float], q: float) -> Optional[float]:
    if not values:
        return None
    xs = sorted(float(v) for v in values)
    if len(xs) == 1:
        return xs[0]
    position = (len(xs) - 1) * q
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return xs[lower]
    weight = position - lower
    return xs[lower] * (1.0 - weight) + xs[upper] * weight


def population_std(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((x - mean) ** 2 for x in values) / len(values))


def process_peak_rss_mb() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1024.0 ** 2) if sys.platform == "darwin" else peak / 1024.0


def runtime_environment(device: torch.device) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "device": str(device),
        "torch_version": torch.__version__,
        "torchaudio_version": torchaudio.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
    }
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        props = torch.cuda.get_device_properties(index)
        out.update({
            "gpu_type": props.name,
            "gpu_total_memory_gib": props.total_memory / (1024.0 ** 3),
            "gpu_compute_capability": f"{props.major}.{props.minor}",
        })
    else:
        out.update({
            "gpu_type": None,
            "gpu_total_memory_gib": None,
            "gpu_compute_capability": None,
        })
    return out


def resolve_config_path(root_dir: str, config_arg: str) -> str:
    path = config_arg if os.path.isabs(config_arg) else os.path.join(root_dir, "config", config_arg)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Config not found: {path}")
    return path


# ============================================================
# Audio utilities
# ============================================================

def load_audio_mono(path: str) -> Tuple[torch.Tensor, int]:
    wav, sr = torchaudio.load(path)
    if wav.dtype != torch.float32:
        wav = wav.float()
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    return wav, sr


def ensure_sr(wav: torch.Tensor, sr: int, target_sr: int) -> Tuple[torch.Tensor, int]:
    if sr == target_sr:
        return wav, sr
    return AF.resample(wav, orig_freq=sr, new_freq=target_sr), target_sr


def read_paths_from_csv(csv_path: str, path_col: str = "path") -> List[str]:
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    paths: List[str] = []
    with open(csv_path, "r", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or path_col not in reader.fieldnames:
            raise ValueError(f"CSV must contain `{path_col}`. Found: {reader.fieldnames}")
        for row in reader:
            path = row[path_col].strip()
            if path:
                paths.append(path)
    return paths


# ============================================================
# Metadata
# ============================================================

@dataclass
class GlobalPhoneInterval:
    start: int
    end: int
    phone: str


@dataclass
class ArchiveFileInfo:
    path: str
    phn_path: Optional[str]
    global_start: int
    global_end: int
    num_samples: int


@dataclass
class SegmentRecord:
    seg_idx: int
    start: int
    end: int
    duration_sec: float
    phones: List[str]
    phone_str: str
    source_files: List[str]


@dataclass
class TokenizationTimingRecord:
    item_order: int
    seg_idx: int
    batch_index: int
    batch_size: int
    num_tokens: int
    audio_duration_sec: float
    amortized_data_loading_ms: float
    amortized_feature_extraction_ms: float
    amortized_host_to_device_ms: float
    amortized_model_tokenize_ms: float
    amortized_postprocess_ms: float
    amortized_end_to_end_ms: float


# ============================================================
# PHN parsing and archive construction
# ============================================================

def phn_path_from_wav(wav_path: str) -> str:
    return os.path.splitext(wav_path)[0] + ".PHN"


def read_phn_file(phn_path: str) -> List[Tuple[int, int, str]]:
    output: List[Tuple[int, int, str]] = []
    with open(phn_path) as handle:
        for line in handle:
            line = line.strip()
            if line:
                start, end, phone = line.split()
                output.append((int(start), int(end), phone))
    return output


def build_archive_metadata(
    audio_csv: str,
    fs: int,
    path_col: str,
    logger: logging.Logger,
    require_phn: bool = True,
) -> Tuple[List[ArchiveFileInfo], List[GlobalPhoneInterval]]:
    paths = [p for p in read_paths_from_csv(audio_csv, path_col) if os.path.isfile(p)]
    if not paths:
        raise ValueError("No valid audio files found in CSV.")

    file_infos: List[ArchiveFileInfo] = []
    global_phones: List[GlobalPhoneInterval] = []
    current = 0
    found_phn = 0
    missing_phn = 0

    for wav_path in tqdm(paths, desc="Stage 1/7 | Archive metadata", unit="file"):
        phn_path = phn_path_from_wav(wav_path)
        if os.path.isfile(phn_path):
            phn_path_or_none: Optional[str] = phn_path
            found_phn += 1
        else:
            phn_path_or_none = None
            missing_phn += 1
            if require_phn:
                raise FileNotFoundError(
                    f"Missing PHN: {phn_path}. Use --segment_only without PHN files."
                )

        info = torchaudio.info(wav_path)
        file_sr = info.sample_rate
        frames = info.num_frames
        num_samples = int(round(frames / file_sr * fs)) if file_sr != fs else int(frames)
        global_start, global_end = current, current + num_samples

        file_infos.append(ArchiveFileInfo(
            path=wav_path,
            phn_path=phn_path_or_none,
            global_start=global_start,
            global_end=global_end,
            num_samples=num_samples,
        ))

        if phn_path_or_none is not None:
            for start, end, phone in read_phn_file(phn_path_or_none):
                if file_sr != fs:
                    start = int(round(start * fs / file_sr))
                    end = int(round(end * fs / file_sr))
                global_phones.append(GlobalPhoneInterval(
                    start=global_start + start,
                    end=global_start + end,
                    phone=phone,
                ))
        current = global_end

    logger.info(f"Files: {len(file_infos)}")
    logger.info(f"Total samples: {current} ({current / fs / 3600:.4f} h)")
    logger.info(f"PHN files found: {found_phn}")
    logger.info(f"PHN files missing: {missing_phn}")
    logger.info(f"Phone intervals: {len(global_phones)}")
    return file_infos, global_phones


def build_segments(
    file_infos: List[ArchiveFileInfo],
    global_phones: List[GlobalPhoneInterval],
    fs: int,
    window_sec: float,
    hop_sec: float,
    silence_set: Optional[Set[str]],
    logger: logging.Logger,
) -> List[SegmentRecord]:
    total_len = file_infos[-1].global_end
    target_len = int(round(window_sec * fs))
    hop_len = int(round(hop_sec * fs))
    if total_len < target_len:
        raise ValueError("Archive shorter than one window.")

    starts = list(range(0, total_len - target_len + 1, hop_len))
    segments: List[SegmentRecord] = []
    phone_ptr = 0
    file_ptr = 0
    has_phones = bool(global_phones)

    for seg_idx, seg_start in enumerate(tqdm(starts, desc="Stage 2/7 | Building segments", unit="seg")):
        seg_end = seg_start + target_len
        phones: List[str] = []

        if has_phones:
            while phone_ptr < len(global_phones) and global_phones[phone_ptr].end <= seg_start:
                phone_ptr += 1
            pointer = phone_ptr
            while pointer < len(global_phones) and global_phones[pointer].start < seg_end:
                phone = global_phones[pointer].phone
                if silence_set is None or phone not in silence_set:
                    phones.append(phone)
                pointer += 1

        while file_ptr + 1 < len(file_infos) and file_infos[file_ptr].global_end <= seg_start:
            file_ptr += 1
        source_files: List[str] = []
        pointer = file_ptr
        while pointer < len(file_infos) and file_infos[pointer].global_start < seg_end:
            source_files.append(file_infos[pointer].path)
            pointer += 1

        segments.append(SegmentRecord(
            seg_idx=seg_idx,
            start=seg_start,
            end=seg_end,
            duration_sec=(seg_end - seg_start) / fs,
            phones=phones,
            phone_str=" ".join(phones),
            source_files=source_files,
        ))

    logger.info(f"Segments: {len(segments)}")
    return segments


# ============================================================
# Feature extraction and datasets
# ============================================================

def build_batch_featextract(
    fs: int,
    feats_type: str,
    rf_size: int,
    stride: int,
    n_mels: int,
) -> Optional[torch.nn.Module]:
    if feats_type == "melspectrogram":
        return torch.nn.Sequential(
            T.MelSpectrogram(
                sample_rate=fs,
                n_fft=rf_size,
                hop_length=stride,
                n_mels=n_mels,
                center=True,
            ),
            T.AmplitudeToDB(),
        )
    if feats_type == "spectrogram":
        return torch.nn.Sequential(
            T.Spectrogram(n_fft=rf_size, hop_length=stride, center=True, power=2.0),
            T.AmplitudeToDB(),
        )
    if feats_type == "raw":
        return None
    raise ValueError("feats_type must be raw, spectrogram, or melspectrogram")


class ArchiveDataset(Dataset):
    def __init__(
        self,
        file_infos: List[ArchiveFileInfo],
        segments: List[SegmentRecord],
        fs: int = 16000,
        context_sec: float = 3.0,
    ):
        super().__init__()
        self.file_infos = file_infos
        self.segments = segments
        self.fs = fs
        self.target_len = int(round(context_sec * fs))
        self._seg_to_file: List[int] = [0] * len(segments)

        file_index = 0
        for i, segment in enumerate(segments):
            while file_index + 1 < len(file_infos) and file_infos[file_index].global_end <= segment.start:
                file_index += 1
            self._seg_to_file[i] = file_index

    def _load_segment_wav(self, segment: SegmentRecord) -> torch.Tensor:
        remaining_start, remaining_end = segment.start, segment.end
        parts: List[torch.Tensor] = []
        file_index = self._seg_to_file[segment.seg_idx]

        while remaining_start < remaining_end and file_index < len(self.file_infos):
            info = self.file_infos[file_index]
            if info.global_start >= remaining_end:
                break
            local_start = max(remaining_start, info.global_start) - info.global_start
            local_end = min(remaining_end, info.global_end) - info.global_start
            wav, sr = load_audio_mono(info.path)
            wav, _ = ensure_sr(wav, sr, self.fs)
            parts.append(wav[:, local_start:local_end])
            remaining_start = info.global_end
            file_index += 1

        if not parts:
            return torch.zeros((1, self.target_len), dtype=torch.float32)
        output = torch.cat(parts, dim=-1)
        return F.pad(output, (0, self.target_len - output.size(-1))) if output.size(-1) < self.target_len else output[..., :self.target_len]

    def __len__(self) -> int:
        return len(self.segments)

    def __getitem__(self, idx: int):
        segment = self.segments[idx]
        return self._load_segment_wav(segment), segment.seg_idx


class QueryDataset(Dataset):
    def __init__(self, archive_ds: ArchiveDataset, query_indices: List[int], augmenter, seed: int):
        super().__init__()
        self.archive_ds = archive_ds
        self.query_indices = query_indices
        self.augmenter = augmenter
        self.seed = seed

    def __len__(self) -> int:
        return len(self.query_indices)

    def __getitem__(self, idx: int):
        seg_idx = self.query_indices[idx]
        segment = self.archive_ds.segments[seg_idx]
        wav = self.archive_ds._load_segment_wav(segment)
        augmented = self.augmenter(wav, rng=random.Random(self.seed + seg_idx))
        return augmented, seg_idx


def collate_batch(batch):
    return torch.stack([item[0] for item in batch], dim=0), [item[1] for item in batch]


# ============================================================
# Tokenization
# ============================================================

def normalise_tokens(output: Any) -> List[List[int]]:
    def to_ints(item):
        if isinstance(item, torch.Tensor):
            return [int(v) for v in item.detach().cpu().reshape(-1).tolist()]
        return [int(v) for v in item]

    if isinstance(output, torch.Tensor):
        output = output.detach().cpu()
        return [to_ints(output)] if output.ndim == 1 else [to_ints(output[i]) for i in range(output.size(0))]
    if isinstance(output, (list, tuple)):
        return [to_ints(item) for item in output]
    if isinstance(output, dict):
        for key in ("tokens", "token_ids", "sequences", "seqs"):
            if key in output:
                return normalise_tokens(output[key])
    raise TypeError(f"Unsupported tokenize() output type: {type(output)}")


def summarize_token_sequences(tokens: List[List[int]], audio_sec_per_item: float) -> Dict[str, Any]:
    lengths = [len(sequence) for sequence in tokens]
    count = len(lengths)
    total_tokens = sum(lengths)
    total_audio_sec = count * audio_sec_per_item
    return {
        "num_items": count,
        "total_tokens": total_tokens,
        "mean_sequence_length": total_tokens / count if count else 0.0,
        "std_sequence_length": population_std(lengths),
        "median_sequence_length": percentile(lengths, 0.50),
        "p05_sequence_length": percentile(lengths, 0.05),
        "p95_sequence_length": percentile(lengths, 0.95),
        "min_sequence_length": min(lengths) if lengths else 0,
        "max_sequence_length": max(lengths) if lengths else 0,
        "audio_sec_per_item": audio_sec_per_item,
        "total_audio_sec": total_audio_sec,
        "token_rate_tokens_per_audio_sec": safe_rate(total_tokens, total_audio_sec),
    }


def log_tokenization_stats(logger: logging.Logger, name: str, stats: Dict[str, Any]) -> None:
    logger.info(f"==== {name} tokenization efficiency ====")
    logger.info(
        "items=%s | tokens=%s | mean_len=%.4f | token_rate=%.4f tokens/audio-sec",
        stats.get("num_items"),
        stats.get("total_tokens"),
        stats.get("mean_sequence_length", 0.0),
        stats.get("token_rate_tokens_per_audio_sec", 0.0),
    )
    if stats.get("timing_available", False):
        logger.info(
            "wall=%.4fs | model.tokenize=%.4fs | items/s=%.4f | generated_tokens/s=%.4f",
            stats["wall_clock_sec"],
            stats["model_tokenize_sec"],
            stats["items_per_wall_sec"],
            stats["generated_tokens_per_wall_sec"],
        )
        logger.info(
            "amortized model latency: mean=%.4f ms | p50=%.4f ms | p95=%.4f ms",
            stats["mean_model_tokenize_ms_per_item"],
            stats["p50_model_tokenize_ms_per_item"],
            stats["p95_model_tokenize_ms_per_item"],
        )
    logger.info(json.dumps(stats, indent=2))


@torch.no_grad()
def tokenize_dataset(
    model,
    dataset: Dataset,
    featextract: Optional[torch.nn.Module],
    batch_size: int,
    device: torch.device,
    max_len: int,
    beam_size: int,
    strip_special_tokens: bool,
    desc: str,
    audio_sec_per_item: float,
) -> Tuple[List[int], List[List[int]], Dict[str, Any], List[TokenizationTimingRecord]]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=collate_batch,
    )
    model.eval().to(device)
    if featextract is not None:
        featextract.eval()

    ids_out: List[int] = []
    tokens_out: List[List[int]] = []
    records: List[TokenizationTimingRecord] = []
    totals = {"data": 0.0, "feature": 0.0, "transfer": 0.0, "model": 0.0, "post": 0.0}

    if device.type == "cuda":
        cuda_sync(device)
        torch.cuda.reset_peak_memory_stats(device)
        allocated_before = torch.cuda.memory_allocated(device) / (1024.0 ** 2)
        reserved_before = torch.cuda.memory_reserved(device) / (1024.0 ** 2)
    else:
        allocated_before = reserved_before = None

    start_all = time.perf_counter()
    iterator = iter(loader)
    progress = tqdm(total=len(loader), desc=desc, unit="batch")
    item_order = 0
    batch_index = 0

    while True:
        data_start = time.perf_counter()
        try:
            wavs, seg_ids = next(iterator)
        except StopIteration:
            break
        data_sec = time.perf_counter() - data_start
        batch_start = time.perf_counter()

        feature_start = time.perf_counter()
        feats = featextract(wavs) if featextract is not None else wavs
        feature_sec = time.perf_counter() - feature_start

        transfer_start = time.perf_counter()
        feats = feats.to(device).contiguous()
        cuda_sync(device)
        transfer_sec = time.perf_counter() - transfer_start

        cuda_sync(device)
        model_start = time.perf_counter()
        raw_tokens = model.tokenize(
            feats,
            max_len=max_len,
            beam_size=beam_size,
            strip_special_tokens=strip_special_tokens,
        )
        cuda_sync(device)
        model_sec = time.perf_counter() - model_start

        post_start = time.perf_counter()
        batch_tokens = normalise_tokens(raw_tokens)
        post_sec = time.perf_counter() - post_start

        if len(batch_tokens) != len(seg_ids):
            raise RuntimeError(f"{len(batch_tokens)} token sequences for {len(seg_ids)} inputs")

        batch_end_to_end_sec = data_sec + (time.perf_counter() - batch_start)
        current_batch_size = len(seg_ids)
        for seg_id, sequence in zip(seg_ids, batch_tokens):
            records.append(TokenizationTimingRecord(
                item_order=item_order,
                seg_idx=int(seg_id),
                batch_index=batch_index,
                batch_size=current_batch_size,
                num_tokens=len(sequence),
                audio_duration_sec=audio_sec_per_item,
                amortized_data_loading_ms=1000.0 * data_sec / current_batch_size,
                amortized_feature_extraction_ms=1000.0 * feature_sec / current_batch_size,
                amortized_host_to_device_ms=1000.0 * transfer_sec / current_batch_size,
                amortized_model_tokenize_ms=1000.0 * model_sec / current_batch_size,
                amortized_postprocess_ms=1000.0 * post_sec / current_batch_size,
                amortized_end_to_end_ms=1000.0 * batch_end_to_end_sec / current_batch_size,
            ))
            item_order += 1

        ids_out.extend(int(seg_id) for seg_id in seg_ids)
        tokens_out.extend(batch_tokens)
        totals["data"] += data_sec
        totals["feature"] += feature_sec
        totals["transfer"] += transfer_sec
        totals["model"] += model_sec
        totals["post"] += post_sec
        batch_index += 1
        progress.update(1)

    progress.close()
    cuda_sync(device)
    wall_sec = time.perf_counter() - start_all

    stats = summarize_token_sequences(tokens_out, audio_sec_per_item)
    num_items = len(tokens_out)
    total_tokens = sum(len(sequence) for sequence in tokens_out)
    model_ms = [record.amortized_model_tokenize_ms for record in records]
    end_to_end_ms = [record.amortized_end_to_end_ms for record in records]
    stats.update({
        "timing_available": True,
        "timing_note": "Per-item latency is batch-amortized; use batch_size=1 for single-item latency.",
        "batch_size": batch_size,
        "beam_size": beam_size,
        "max_decode_length": max_len,
        "num_batches": batch_index,
        "wall_clock_sec": wall_sec,
        "data_loading_sec": totals["data"],
        "feature_extraction_sec": totals["feature"],
        "host_to_device_sec": totals["transfer"],
        "model_tokenize_sec": totals["model"],
        "postprocess_sec": totals["post"],
        "items_per_wall_sec": safe_rate(num_items, wall_sec),
        "items_per_model_tokenize_sec": safe_rate(num_items, totals["model"]),
        "generated_tokens_per_wall_sec": safe_rate(total_tokens, wall_sec),
        "generated_tokens_per_model_tokenize_sec": safe_rate(total_tokens, totals["model"]),
        "mean_wall_ms_per_item": 1000.0 * wall_sec / num_items if num_items else None,
        "mean_model_tokenize_ms_per_item": sum(model_ms) / len(model_ms) if model_ms else None,
        "p50_model_tokenize_ms_per_item": percentile(model_ms, 0.50),
        "p95_model_tokenize_ms_per_item": percentile(model_ms, 0.95),
        "mean_end_to_end_ms_per_item": sum(end_to_end_ms) / len(end_to_end_ms) if end_to_end_ms else None,
        "p50_end_to_end_ms_per_item": percentile(end_to_end_ms, 0.50),
        "p95_end_to_end_ms_per_item": percentile(end_to_end_ms, 0.95),
    })

    if device.type == "cuda":
        stats.update({
            "gpu_memory_allocated_before_mb": allocated_before,
            "gpu_memory_reserved_before_mb": reserved_before,
            "peak_gpu_memory_allocated_mb": torch.cuda.max_memory_allocated(device) / (1024.0 ** 2),
            "peak_gpu_memory_reserved_mb": torch.cuda.max_memory_reserved(device) / (1024.0 ** 2),
            "gpu_memory_allocated_after_mb": torch.cuda.memory_allocated(device) / (1024.0 ** 2),
            "gpu_memory_reserved_after_mb": torch.cuda.memory_reserved(device) / (1024.0 ** 2),
        })
    else:
        stats.update({key: None for key in (
            "gpu_memory_allocated_before_mb",
            "gpu_memory_reserved_before_mb",
            "peak_gpu_memory_allocated_mb",
            "peak_gpu_memory_reserved_mb",
            "gpu_memory_allocated_after_mb",
            "gpu_memory_reserved_after_mb",
        )})

    return ids_out, tokens_out, stats, records


# ============================================================
# Cache
# ============================================================

def save_cache(
    path: str,
    segments: List[SegmentRecord],
    tokens: List[List[int]],
    tokenization_stats: Dict[str, Any],
    timing_records: List[TokenizationTimingRecord],
) -> None:
    torch.save({
        "segments": [asdict(segment) for segment in segments],
        "tokens": tokens,
        "tokenization_stats": tokenization_stats,
        "timing_records": [asdict(record) for record in timing_records],
    }, path)


def load_cache(path: str) -> Tuple[List[SegmentRecord], List[List[int]], Optional[Dict[str, Any]], List[TokenizationTimingRecord]]:
    data = torch.load(path, map_location="cpu")
    return (
        [SegmentRecord(**segment) for segment in data["segments"]],
        data["tokens"],
        data.get("tokenization_stats"),
        [TokenizationTimingRecord(**record) for record in (data.get("timing_records") or [])],
    )


# ============================================================
# Edit distance, relevance and ranking
# ============================================================

try:
    from rapidfuzz.distance import Levenshtein as RapidLevenshtein
    EDIT_DISTANCE_BACKEND = "rapidfuzz"

    def edit_distance(a, b) -> int:
        return int(RapidLevenshtein.distance(a, b))

except Exception:
    EDIT_DISTANCE_BACKEND = "python_dynamic_programming"

    def edit_distance(a, b) -> int:
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


def ned(a, b) -> float:
    return edit_distance(a, b) / max(len(a), len(b), 1)


def overlap_relevant(query: SegmentRecord, segments: List[SegmentRecord]) -> Set[int]:
    return {
        segment.seg_idx for segment in segments
        if min(query.end, segment.end) - max(query.start, segment.start) > 0
    }


def exact_phone_map(segments: List[SegmentRecord]) -> Dict[str, List[int]]:
    mapping: Dict[str, List[int]] = {}
    for segment in segments:
        mapping.setdefault(segment.phone_str, []).append(segment.seg_idx)
    return mapping


def relaxed_phone_relevant(query_phones: List[str], segments: List[SegmentRecord], threshold: float) -> Set[int]:
    return {segment.seg_idx for segment in segments if ned(query_phones, segment.phones) <= threshold}


def rank_by_ned(query_tokens: List[int], archive_tokens: List[List[int]]) -> Tuple[List[Tuple[int, float]], Dict[str, int]]:
    query_len = len(query_tokens)
    scores: List[Tuple[int, float]] = []
    edit_calls = 0
    length_bound_skips = 0
    length_product_proxy = 0

    for index, candidate in enumerate(archive_tokens):
        candidate_len = len(candidate)
        lower_bound = abs(query_len - candidate_len) / max(query_len, candidate_len, 1)
        if lower_bound >= 1.0:
            score = lower_bound
            length_bound_skips += 1
        else:
            edit_calls += 1
            length_product_proxy += query_len * candidate_len
            score = ned(query_tokens, candidate)
        scores.append((index, score))

    scores.sort(key=lambda item: (item[1], item[0]))
    return scores, {
        "num_candidates": len(archive_tokens),
        "edit_distance_calls": edit_calls,
        "length_bound_skips": length_bound_skips,
        "edit_length_product_proxy": length_product_proxy,
    }


def first_hit(ranked: Sequence[int], relevant: Set[int]) -> Optional[int]:
    for rank, index in enumerate(ranked, 1):
        if index in relevant:
            return rank
    return None


# ============================================================
# Retrieval metrics and output writers
# ============================================================

def compute_metrics(first_ranks: List[Optional[int]], ks: Sequence[int]) -> Dict[str, float]:
    count = len(first_ranks)
    mrr = 0.0
    hits = {k: 0 for k in ks}
    finite: List[int] = []
    for rank in first_ranks:
        if rank is not None:
            finite.append(rank)
            mrr += 1.0 / rank
            for k in ks:
                if rank <= k:
                    hits[k] += 1
    output = {f"Recall@{k}": hits[k] / count if count else 0.0 for k in ks}
    output["MRR"] = mrr / count if count else 0.0
    output["HitRate"] = len(finite) / count if count else 0.0
    output["MeanFirstRelevantRank"] = sum(finite) / len(finite) if finite else float("inf")
    output["MedianFirstRelevantRank"] = float(sorted(finite)[len(finite) // 2]) if finite else float("inf")
    return output


def write_tokenization_timing_csv(path: str, records: List[TokenizationTimingRecord]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fields = list(TokenizationTimingRecord.__dataclass_fields__.keys())
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))


def write_query_runtime_csv(
    path: str,
    query_indices: List[int],
    query_tokens: List[List[int]],
    tokenization_records: List[TokenizationTimingRecord],
    ranking_times_sec: List[float],
    full_query_times_sec: List[float],
    diagnostics: List[Dict[str, int]],
) -> None:
    timing_by_seg = {record.seg_idx: record for record in tokenization_records}
    fields = [
        "query_num", "query_seg_idx", "query_token_len", "query_token_rate_tokens_per_audio_sec",
        "amortized_data_loading_ms", "amortized_feature_extraction_ms", "amortized_host_to_device_ms",
        "amortized_model_tokenize_ms", "amortized_postprocess_ms", "amortized_tokenization_end_to_end_ms",
        "ranking_only_wall_clock_ms", "full_query_retrieval_wall_clock_ms", "num_archive_candidates",
        "edit_distance_calls", "length_bound_skips", "edit_length_product_proxy",
    ]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for query_num, seg_idx in enumerate(query_indices):
            timing = timing_by_seg.get(seg_idx)
            audio_sec = timing.audio_duration_sec if timing is not None else None
            diag = diagnostics[query_num]
            writer.writerow({
                "query_num": query_num,
                "query_seg_idx": seg_idx,
                "query_token_len": len(query_tokens[query_num]),
                "query_token_rate_tokens_per_audio_sec": len(query_tokens[query_num]) / audio_sec if audio_sec else None,
                "amortized_data_loading_ms": timing.amortized_data_loading_ms if timing else None,
                "amortized_feature_extraction_ms": timing.amortized_feature_extraction_ms if timing else None,
                "amortized_host_to_device_ms": timing.amortized_host_to_device_ms if timing else None,
                "amortized_model_tokenize_ms": timing.amortized_model_tokenize_ms if timing else None,
                "amortized_postprocess_ms": timing.amortized_postprocess_ms if timing else None,
                "amortized_tokenization_end_to_end_ms": timing.amortized_end_to_end_ms if timing else None,
                "ranking_only_wall_clock_ms": ranking_times_sec[query_num] * 1000.0,
                "full_query_retrieval_wall_clock_ms": full_query_times_sec[query_num] * 1000.0,
                "num_archive_candidates": diag["num_candidates"],
                "edit_distance_calls": diag["edit_distance_calls"],
                "length_bound_skips": diag["length_bound_skips"],
                "edit_length_product_proxy": diag["edit_length_product_proxy"],
            })


def write_query_csv(
    path: str,
    query_indices: List[int],
    query_tokens: List[List[int]],
    segments: List[SegmentRecord],
    ranked_lists: List[List[Tuple[int, float]]],
    overlap_first: List[Optional[int]],
    exact_first: List[Optional[int]],
    relaxed_first: List[Optional[int]],
    topk: int,
    segment_only: bool,
) -> None:
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        if segment_only:
            writer.writerow([
                "query_num", "query_seg_idx", "query_start", "query_end", "query_token_len",
                "segment_overlap_first_rank", "topk_rank", "retrieved_seg_idx", "retrieved_start",
                "retrieved_end", "retrieved_ned", "retrieved_source_files",
            ])
            for query_num, query_seg_idx in enumerate(query_indices):
                query_segment = segments[query_seg_idx]
                for rank_position, (retrieved_idx, score) in enumerate(ranked_lists[query_num][:topk], 1):
                    retrieved = segments[retrieved_idx]
                    writer.writerow([
                        query_num, query_seg_idx, query_segment.start, query_segment.end,
                        len(query_tokens[query_num]), overlap_first[query_num], rank_position,
                        retrieved_idx, retrieved.start, retrieved.end, f"{score:.6f}",
                        "|".join(retrieved.source_files),
                    ])
        else:
            writer.writerow([
                "query_num", "query_seg_idx", "query_phone_str", "query_token_len",
                "overlap_first_rank", "exact_first_rank", "relaxed_first_rank", "topk_rank",
                "retrieved_seg_idx", "retrieved_ned", "retrieved_phone_str",
            ])
            for query_num, query_seg_idx in enumerate(query_indices):
                query_segment = segments[query_seg_idx]
                for rank_position, (retrieved_idx, score) in enumerate(ranked_lists[query_num][:topk], 1):
                    writer.writerow([
                        query_num, query_seg_idx, query_segment.phone_str, len(query_tokens[query_num]),
                        overlap_first[query_num], exact_first[query_num], relaxed_first[query_num],
                        rank_position, retrieved_idx, f"{score:.6f}", segments[retrieved_idx].phone_str,
                    ])


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser("Retrieval evaluation with inference timing")
    parser.add_argument("--root_dir", type=str, default=os.path.dirname(os.getcwd()))
    parser.add_argument("--config", type=str, default="mainFull.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--audio_csv", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--hop_sec", type=float, default=1.5)
    parser.add_argument("--n_queries", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--relaxed_threshold", type=float, default=0.35)
    parser.add_argument("--topk_dump", type=int, default=20)
    parser.add_argument("--use_cache", action="store_true")
    parser.add_argument("--ignore_silence", action="store_true")
    parser.add_argument("--segment_only", action="store_true")
    parser.add_argument("--path_col", type=str, default="path")
    parser.add_argument("--max_token_len", type=int, default=60)
    parser.add_argument("--beam_size", type=int, default=4)
    parser.add_argument("--dist_root", type=str, required=True, help="Distortion root used to construct positive query views.")
    parser.add_argument("--use_time_compression", action=argparse.BooleanOptionalAction, default=False, help="Slightly compress only positive query views and right-pad them to 3 seconds.")
    args = parser.parse_args()

    run_start = time.perf_counter()
    stage_times: Dict[str, float] = {}
    os.makedirs(args.save_dir, exist_ok=True)
    logger = setup_logger(os.path.join(args.save_dir, "retrieval_eval.log"))
    set_seed(args.seed)
    logger.info("==== Retrieval eval started ====")
    logger.info(json.dumps(vars(args), indent=2))

    config_path = resolve_config_path(args.root_dir, args.config)
    with open(config_path, "r") as handle:
        cfg = yaml.load(handle, Loader=yaml.FullLoader)

    fs = int(cfg["fs"])
    seg_sec = float(cfg["seglen"])
    n_mels = int(cfg["n_mels"])
    rf_size = int(round(float(cfg["win_size"]) * fs))
    stride = int(round(float(cfg["win_hop"]) * fs))
    featextract = build_batch_featextract(fs, "melspectrogram", rf_size, stride, n_mels)

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    environment = runtime_environment(device)
    logger.info("==== Runtime environment ====")
    logger.info(json.dumps(environment, indent=2))

    load_start = time.perf_counter()

    model = AudioTokenizerAR_FINAL.load_from_pretrained(args.checkpoint, args.device)


    # model = AudioTokenizer.load_from_pretrained(args.checkpoint, args.device)

    model.eval().to(device)
    cuda_sync(device)
    stage_times["model_load_sec"] = time.perf_counter() - load_start

    silence_set = {"h#", "pau", "epi"} if args.ignore_silence else None
    cache_path = os.path.join(args.save_dir, "archive_cache.pt")
    metrics_path = os.path.join(args.save_dir, "metrics.json")
    topk_path = os.path.join(args.save_dir, "per_query_topk.csv")
    archive_timing_path = os.path.join(args.save_dir, "archive_tokenization_timings.csv")
    query_timing_path = os.path.join(args.save_dir, "query_tokenization_timings.csv")
    query_runtime_path = os.path.join(args.save_dir, "query_retrieval_timings.csv")

    start = time.perf_counter()
    file_infos, global_phones = build_archive_metadata(
        args.audio_csv, fs, args.path_col, logger, require_phn=not args.segment_only
    )
    stage_times["stage_1_archive_metadata_sec"] = time.perf_counter() - start
    if not args.segment_only and not global_phones:
        raise RuntimeError("No PHN intervals found; provide PHN files or use --segment_only")

    start = time.perf_counter()
    segments = build_segments(file_infos, global_phones, fs, seg_sec, args.hop_sec, silence_set, logger)
    stage_times["stage_2_build_segments_sec"] = time.perf_counter() - start

    archive_records: List[TokenizationTimingRecord] = []
    if args.use_cache and os.path.isfile(cache_path):
        start = time.perf_counter()
        segments, archive_tokens, archive_stats, archive_records = load_cache(cache_path)
        stage_times["stage_3_archive_cache_load_sec"] = time.perf_counter() - start
        if archive_stats is None:
            archive_stats = summarize_token_sequences(archive_tokens, seg_sec)
            archive_stats.update({
                "timing_available": False,
                "timing_note": "Old cache contains no tokenization timings; regenerate the cache.",
            })
    else:
        archive_dataset = ArchiveDataset(file_infos, segments, fs, seg_sec)
        archive_ids, archive_unsorted, archive_stats, archive_records = tokenize_dataset(
            model, archive_dataset, featextract, args.batch_size, device,
            args.max_token_len, args.beam_size, True,
            "Stage 3/7 | Tokenising archive", seg_sec,
        )
        archive_optional: List[Optional[List[int]]] = [None] * len(segments)
        for seg_id, sequence in zip(archive_ids, archive_unsorted):
            archive_optional[seg_id] = sequence
        missing = sum(sequence is None for sequence in archive_optional)
        if missing:
            raise RuntimeError(f"Missing tokenisations for {missing} archive segments")
        archive_tokens = [sequence for sequence in archive_optional if sequence is not None]
        stage_times["stage_3_archive_tokenization_sec"] = archive_stats["wall_clock_sec"]
        if args.use_cache:
            save_cache(cache_path, segments, archive_tokens, archive_stats, archive_records)
            logger.info(f"Archive cache saved to {cache_path}")

    log_tokenization_stats(logger, "Archive", archive_stats)
    if archive_records:
        write_tokenization_timing_csv(archive_timing_path, archive_records)
    else:
        logger.warning("Archive timing CSV unavailable because the loaded cache has no timing records")

    start = time.perf_counter()
    eligible = [segment.seg_idx for segment in segments] if args.segment_only else [
        segment.seg_idx for segment in segments if segment.phones
    ]
    if len(eligible) < args.n_queries:
        raise ValueError(f"Only {len(eligible)} eligible segments; need {args.n_queries}")
    query_indices = random.Random(args.seed).sample(eligible, args.n_queries)
    stage_times["stage_4_sample_queries_sec"] = time.perf_counter() - start

    from utils.dataset import AudioAugmenter, AugConfig, _collect_audio_files_from_dirs
    dist_root = args.dist_root
    augmenter = AudioAugmenter(
        sr=fs,
        cfg=AugConfig(),
        use_time_compression=args.use_time_compression,
        noise_files=_collect_audio_files_from_dirs([
            os.path.join(dist_root, directory)
            for directory in ("noise_16k", "pointsource_noises", "real_rirs_isotropic_noises")
        ], exts=(".wav", ".flac")),
        rir_files=_collect_audio_files_from_dirs(
            [os.path.join(dist_root, "rir_16k")], exts=(".wav", ".flac")
        ),
    )

    query_archive_dataset = ArchiveDataset(file_infos, segments, fs, seg_sec)
    query_dataset = QueryDataset(query_archive_dataset, query_indices, augmenter, args.seed + 100_000)
    query_ids, query_unsorted, query_stats, query_records = tokenize_dataset(
        model, query_dataset, featextract, args.batch_size, device,
        args.max_token_len, args.beam_size, True,
        "Stage 5/7 | Tokenising queries", seg_sec,
    )
    query_map = {seg_id: sequence for seg_id, sequence in zip(query_ids, query_unsorted)}
    query_tokens = [query_map[index] for index in query_indices]
    stage_times["stage_5_query_tokenization_sec"] = query_stats["wall_clock_sec"]
    log_tokenization_stats(logger, "Query", query_stats)
    write_tokenization_timing_csv(query_timing_path, query_records)

    start = time.perf_counter()
    ranked_lists: List[List[Tuple[int, float]]] = []
    overlap_first: List[Optional[int]] = []
    exact_first: List[Optional[int]] = []
    relaxed_first: List[Optional[int]] = []
    ranking_times: List[float] = []
    full_query_times: List[float] = []
    diagnostics: List[Dict[str, int]] = []
    phone_map = exact_phone_map(segments) if not args.segment_only else None

    for query_num in tqdm(range(len(query_indices)), desc="Stage 6/7 | Ranking", unit="query"):
        query_start = time.perf_counter()
        query_segment = segments[query_indices[query_num]]
        query_sequence = query_tokens[query_num]

        rank_start = time.perf_counter()
        ranked, diag = rank_by_ned(query_sequence, archive_tokens)
        ranking_times.append(time.perf_counter() - rank_start)
        diagnostics.append(diag)
        ranked_lists.append(ranked)
        ranked_indices = [index for index, _ in ranked]

        overlap_first.append(first_hit(ranked_indices, overlap_relevant(query_segment, segments)))
        if args.segment_only:
            exact_first.append(None)
            relaxed_first.append(None)
        else:
            assert phone_map is not None
            exact_first.append(first_hit(ranked_indices, set(phone_map.get(query_segment.phone_str, []))))
            relaxed_first.append(first_hit(
                ranked_indices,
                relaxed_phone_relevant(query_segment.phones, segments, args.relaxed_threshold),
            ))
        full_query_times.append(time.perf_counter() - query_start)

    retrieval_stage_sec = time.perf_counter() - start
    stage_times["stage_6_retrieval_sec"] = retrieval_stage_sec
    ranking_only_sec = sum(ranking_times)
    full_query_sum_sec = sum(full_query_times)
    comparisons = sum(item["num_candidates"] for item in diagnostics)
    edit_calls = sum(item["edit_distance_calls"] for item in diagnostics)
    length_skips = sum(item["length_bound_skips"] for item in diagnostics)
    length_product = sum(item["edit_length_product_proxy"] for item in diagnostics)

    retrieval_efficiency = {
        "edit_distance_backend": EDIT_DISTANCE_BACKEND,
        "retrieval_stage_wall_clock_sec": retrieval_stage_sec,
        "ranking_only_wall_clock_sec": ranking_only_sec,
        "summed_full_query_retrieval_sec": full_query_sum_sec,
        "mean_ranking_ms_per_query": 1000.0 * ranking_only_sec / len(query_indices),
        "p50_ranking_ms_per_query": 1000.0 * percentile(ranking_times, 0.50),
        "p95_ranking_ms_per_query": 1000.0 * percentile(ranking_times, 0.95),
        "mean_full_retrieval_ms_per_query": 1000.0 * full_query_sum_sec / len(query_indices),
        "p50_full_retrieval_ms_per_query": 1000.0 * percentile(full_query_times, 0.50),
        "p95_full_retrieval_ms_per_query": 1000.0 * percentile(full_query_times, 0.95),
        "queries_per_ranking_sec": safe_rate(len(query_indices), ranking_only_sec),
        "queries_per_retrieval_stage_sec": safe_rate(len(query_indices), retrieval_stage_sec),
        "num_archive_segments": len(archive_tokens),
        "num_queries": len(query_tokens),
        "num_archive_query_comparisons": comparisons,
        "comparisons_per_ranking_sec": safe_rate(comparisons, ranking_only_sec),
        "edit_distance_calls": edit_calls,
        "edit_distance_calls_per_sec": safe_rate(edit_calls, ranking_only_sec),
        "length_bound_skips": length_skips,
        "edit_length_product_proxy": length_product,
        "mean_edit_length_product_per_edit_call": safe_rate(length_product, edit_calls),
        "edit_length_product_per_ranking_sec": safe_rate(length_product, ranking_only_sec),
        "archive_mean_sequence_length": archive_stats["mean_sequence_length"],
        "query_mean_sequence_length": query_stats["mean_sequence_length"],
        "archive_token_rate_tokens_per_audio_sec": archive_stats["token_rate_tokens_per_audio_sec"],
        "query_token_rate_tokens_per_audio_sec": query_stats["token_rate_tokens_per_audio_sec"],
    }
    logger.info("==== Retrieval efficiency ====")
    logger.info(json.dumps(retrieval_efficiency, indent=2))
    write_query_runtime_csv(
        query_runtime_path, query_indices, query_tokens, query_records,
        ranking_times, full_query_times, diagnostics,
    )

    start = time.perf_counter()
    ks = [1, 5, 10, 20]
    metrics: Dict[str, Any] = {
        "num_archive_segments": len(segments),
        "num_queries": len(query_indices),
        "window_sec": seg_sec,
        "hop_sec": args.hop_sec,
        "ranking_metric": "normalized_edit_distance_on_tokens",
        "segment_only": args.segment_only,
        "segment_overlap": compute_metrics(overlap_first, ks),
        "efficiency": {
            "runtime_environment": environment,
            "archive_tokenization": archive_stats,
            "query_tokenization": query_stats,
            "retrieval": retrieval_efficiency,
        },
    }
    if not args.segment_only:
        metrics["relaxed_threshold"] = args.relaxed_threshold
        metrics["phoneme_exact"] = compute_metrics(exact_first, ks)
        metrics["phoneme_relaxed"] = compute_metrics(relaxed_first, ks)

    write_query_csv(
        topk_path, query_indices, query_tokens, segments, ranked_lists,
        overlap_first, exact_first, relaxed_first, args.topk_dump, args.segment_only,
    )
    stage_times["stage_7_metrics_and_save_sec"] = time.perf_counter() - start
    metrics["efficiency"]["stage_wall_clock_sec"] = stage_times
    metrics["efficiency"]["process_peak_rss_mb"] = process_peak_rss_mb()
    metrics["efficiency"]["total_evaluation_wall_clock_sec"] = time.perf_counter() - run_start

    allocated_peaks = [
        value for value in (
            archive_stats.get("peak_gpu_memory_allocated_mb"),
            query_stats.get("peak_gpu_memory_allocated_mb"),
        ) if value is not None
    ]
    reserved_peaks = [
        value for value in (
            archive_stats.get("peak_gpu_memory_reserved_mb"),
            query_stats.get("peak_gpu_memory_reserved_mb"),
        ) if value is not None
    ]
    metrics["efficiency"]["evaluation_peak_gpu_memory_allocated_mb"] = max(allocated_peaks) if allocated_peaks else None
    metrics["efficiency"]["evaluation_peak_gpu_memory_reserved_mb"] = max(reserved_peaks) if reserved_peaks else None

    with open(metrics_path, "w") as handle:
        json.dump(metrics, handle, indent=2, allow_nan=True)

    logger.info("==== Final metrics ====")
    logger.info(json.dumps(metrics, indent=2))
    logger.info(f"Metrics -> {metrics_path}")
    logger.info(f"Per-query top-k -> {topk_path}")
    logger.info(f"Archive tokenization timings -> {archive_timing_path}")
    logger.info(f"Query tokenization timings -> {query_timing_path}")
    logger.info(f"Query retrieval timings -> {query_runtime_path}")
    logger.info("==== Retrieval eval finished ====")


if __name__ == "__main__":
    main()
