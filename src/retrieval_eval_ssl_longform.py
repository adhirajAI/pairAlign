#!/usr/bin/env python3
"""
SSL-KMeans retrieval evaluation.

This is the SSL-tokenizer-only counterpart of retrieval_eval_timit_longform.py.

Supported tokenizers:
  --ssl_model hubert
  --ssl_model wav2vec2
  --ssl_model wavlm

The script:
  1. Builds an overlapping long-form archive.
  2. Tokenizes archive segments using pretrained SSL + LibriSpeech KMeans.
  3. Samples augmented query segments.
  4. Tokenizes queries using the same SSL tokenizer.
  5. Ranks archive segments by normalized edit distance over token sequences.
  6. Computes segment-overlap retrieval, and optionally PHN-based retrieval.

For non-TIMIT / non-PHN datasets such as Tamil, use:
  --segment_only
"""

import os
import csv
import json
import time
import random
import argparse
import logging
import warnings
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import yaml
import joblib
import numpy as np
from tqdm.auto import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchaudio
import torchaudio.functional as AF

from huggingface_hub import hf_hub_download

from speechbrain.integrations.huggingface.hubert import HuBERT
from speechbrain.integrations.huggingface.wav2vec2 import Wav2Vec2
from speechbrain.integrations.huggingface.wavlm import WavLM


# ============================================================
# SSL tokenizer registry
# ============================================================

SSL_MODEL_REGISTRY = {
    "hubert": {
        "class": HuBERT,
        "source": "facebook/hubert-base-ls960",
        "kmeans_name": "hubert",
    },
    "wav2vec2": {
        "class": Wav2Vec2,
        "source": "facebook/wav2vec2-base",
        "kmeans_name": "wav2vec",
    },
    # chaanks/Kmeans LibriSpeech_wavlm_k512_L7.pt has 1024-dim centers,
    # therefore use WavLM-large, not WavLM-base.
    "wavlm": {
        "class": WavLM,
        "source": "microsoft/wavlm-large",
        "kmeans_name": "wavlm",
    },
}


@dataclass
class SSLKMeansConfig:
    ssl_model: str
    k: int = 512
    layer_num: int = 7
    dataset: str = "LibriSpeech"
    kmeans_repo_id: str = "chaanks/Kmeans"
    save_root: str = "pretrained_ssl"
    dedup: bool = True
    source_override: Optional[str] = None


class SSLKMeansTokenizer(torch.nn.Module):
    """
    SpeechBrain SSL encoder + external KMeans tokenizer.

    Input to tokenize():
        wavs: [B, T] or [B, 1, T], sampled at 16 kHz.

    Output:
        List[List[int]], one deduplicated token sequence per waveform.
    """

    def __init__(self, config: SSLKMeansConfig, device: torch.device):
        super().__init__()

        self.config = config
        self.device = device

        ssl_model = config.ssl_model.lower()
        if ssl_model not in SSL_MODEL_REGISTRY:
            raise ValueError(
                f"Unknown ssl_model={ssl_model}. "
                f"Choose from {list(SSL_MODEL_REGISTRY.keys())}."
            )

        model_info = SSL_MODEL_REGISTRY[ssl_model]
        model_class = model_info["class"]
        source = config.source_override or model_info["source"]
        kmeans_name = model_info["kmeans_name"]

        self.kmeans_filename = (
            f"{config.dataset}_{kmeans_name}_k{config.k}_L{config.layer_num}.pt"
        )

        print(f"SSL model:       {ssl_model}")
        print(f"HF source:       {source}")
        print(f"KMeans repo:     {config.kmeans_repo_id}")
        print(f"KMeans file:     {self.kmeans_filename}")
        print(f"SSL layer:       {config.layer_num}")
        print(f"Dedup:           {config.dedup}")

        self.ssl = model_class(
            source=source,
            save_path=os.path.join(config.save_root, ssl_model),
            output_norm=False,
            freeze=True,
            output_all_hiddens=True,
        ).to(device)

        self.ssl.eval()

        kmeans_path = hf_hub_download(
            repo_id=config.kmeans_repo_id,
            filename=self.kmeans_filename,
        )

        centers = load_kmeans_centers(kmeans_path)
        self.register_buffer("centers", centers.to(device), persistent=False)

        print("KMeans centers:", tuple(self.centers.shape))

    @staticmethod
    def _dedup_consecutive(tok: torch.Tensor) -> torch.Tensor:
        tok = tok.detach().cpu().reshape(-1)

        if tok.numel() == 0:
            return tok

        keep = torch.ones(tok.numel(), dtype=torch.bool)
        keep[1:] = tok[1:] != tok[:-1]
        return tok[keep]

    @torch.no_grad()
    def tokenize(self, wavs: torch.Tensor) -> List[List[int]]:
        # [B, 1, T] or [B, C, T] -> [B, T]
        if wavs.ndim == 3:
            wavs = wavs.mean(dim=1)

        if wavs.ndim != 2:
            raise RuntimeError(f"Expected wavs with shape [B, T] or [B, C, T], got {wavs.shape}")

        wavs = wavs.to(self.device).contiguous()

        hidden_states = self.ssl(wavs)

        if isinstance(hidden_states, (tuple, list)):
            hidden_states = torch.stack(list(hidden_states), dim=0)

        if hidden_states.ndim != 4:
            raise RuntimeError(
                f"Expected hidden states with shape [L, B, T, D], "
                f"got {hidden_states.shape}"
            )

        if self.config.layer_num >= hidden_states.shape[0]:
            raise RuntimeError(
                f"Requested layer {self.config.layer_num}, but model returned "
                f"only {hidden_states.shape[0]} hidden-state tensors."
            )

        feats = hidden_states[self.config.layer_num]  # [B, T_ssl, D]

        if feats.shape[-1] != self.centers.shape[-1]:
            raise RuntimeError(
                f"Feature dim mismatch. SSL features have dim {feats.shape[-1]}, "
                f"but KMeans centers have dim {self.centers.shape[-1]}. "
                f"Check ssl_model/source and KMeans file."
            )

        out: List[List[int]] = []

        for b in range(feats.shape[0]):
            dists = torch.cdist(feats[b].float(), self.centers.float())  # [T_ssl, K]
            tok = dists.argmin(dim=-1).detach().cpu()

            if self.config.dedup:
                tok = self._dedup_consecutive(tok)

            out.append([int(x) for x in tok.tolist()])

        return out


def load_kmeans_centers(path: str) -> torch.Tensor:
    """
    chaanks/Kmeans files are sklearn/joblib-style serialized objects,
    not torch.load checkpoints.
    """
    obj = joblib.load(path)

    print("Loaded KMeans object type:", type(obj))

    if hasattr(obj, "cluster_centers_"):
        return torch.tensor(obj.cluster_centers_, dtype=torch.float32)

    if isinstance(obj, dict):
        print("KMeans dict keys:", obj.keys())

        for key in [
            "cluster_centers_",
            "cluster_centers",
            "centroids",
            "centers",
            "kmeans",
            "model",
        ]:
            if key not in obj:
                continue

            value = obj[key]

            if hasattr(value, "cluster_centers_"):
                return torch.tensor(value.cluster_centers_, dtype=torch.float32)

            if isinstance(value, np.ndarray):
                return torch.tensor(value, dtype=torch.float32)

            if torch.is_tensor(value):
                return value.float()

    if isinstance(obj, np.ndarray):
        return torch.tensor(obj, dtype=torch.float32)

    raise RuntimeError(f"Could not extract KMeans centers from: {type(obj)}")


# ============================================================
# Logging
# ============================================================

def setup_logger(log_path: str) -> logging.Logger:
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    logger = logging.getLogger("retrieval_eval_ssl")
    logger.setLevel(logging.INFO)
    logger.handlers = []
    logger.propagate = False

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)

    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(sh)

    return logger


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# Audio utils
# ============================================================

def _load_audio_mono(path: str) -> Tuple[torch.Tensor, int]:
    wav, sr = torchaudio.load(path)

    if wav.dtype != torch.float32:
        wav = wav.float()

    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)

    return wav, sr


def _ensure_sr(wav: torch.Tensor, sr: int, target_sr: int) -> Tuple[torch.Tensor, int]:
    if sr == target_sr:
        return wav, sr

    wav = AF.resample(wav, orig_freq=sr, new_freq=target_sr)
    return wav, target_sr


def _read_paths_from_csv(csv_path: str, path_col: str = "path") -> List[str]:
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    paths: List[str] = []

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)

        if path_col not in reader.fieldnames:
            raise ValueError(
                f"CSV must contain column `{path_col}`. Found: {reader.fieldnames}"
            )

        for row in reader:
            p = row[path_col].strip()
            if p:
                paths.append(p)

    return paths


# ============================================================
# Metadata containers
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


# ============================================================
# PHN parsing
# ============================================================

def phn_path_from_wav(wav_path: str) -> str:
    return os.path.splitext(wav_path)[0] + ".PHN"


def read_phn_file(phn_path: str) -> List[Tuple[int, int, str]]:
    out: List[Tuple[int, int, str]] = []

    with open(phn_path) as f:
        for line in f:
            line = line.strip()
            if line:
                s, e, ph = line.split()
                out.append((int(s), int(e), ph))

    return out


# ============================================================
# Build archive metadata
# ============================================================

def build_archive_metadata(
    audio_csv: str,
    fs: int,
    path_col: str,
    logger: logging.Logger,
    require_phn: bool = True,
) -> Tuple[List[ArchiveFileInfo], List[GlobalPhoneInterval]]:

    paths = _read_paths_from_csv(audio_csv, path_col=path_col)
    paths = [p for p in paths if os.path.isfile(p)]

    if not paths:
        raise ValueError("No valid audio files found in CSV.")

    file_infos: List[ArchiveFileInfo] = []
    global_phones: List[GlobalPhoneInterval] = []

    cur = 0
    missing_phn = 0
    found_phn = 0

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
                    f"Missing PHN: {phn_path}. "
                    f"Use --segment_only if you want segment-level retrieval without PHN files."
                )

        info = torchaudio.info(wav_path)
        file_sr = info.sample_rate
        file_frames = info.num_frames

        if file_sr != fs:
            num_samples = int(round(file_frames / file_sr * fs))
        else:
            num_samples = int(file_frames)

        gs, ge = cur, cur + num_samples

        file_infos.append(
            ArchiveFileInfo(
                path=wav_path,
                phn_path=phn_path_or_none,
                global_start=gs,
                global_end=ge,
                num_samples=num_samples,
            )
        )

        if phn_path_or_none is not None:
            for s, e, ph in read_phn_file(phn_path_or_none):
                if file_sr != fs:
                    s = int(round(s * fs / file_sr))
                    e = int(round(e * fs / file_sr))

                global_phones.append(
                    GlobalPhoneInterval(
                        start=gs + s,
                        end=gs + e,
                        phone=ph,
                    )
                )

        cur = ge

    logger.info(f"Files: {len(file_infos)}")
    logger.info(f"Total samples: {cur}  ({cur / fs / 3600:.4f} h)")
    logger.info(f"PHN files found: {found_phn}")
    logger.info(f"PHN files missing: {missing_phn}")
    logger.info(f"Phone intervals: {len(global_phones)}")

    if len(global_phones) == 0:
        logger.info("No PHN annotations available. Segment-level retrieval only is active/expected.")

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

    has_phones = len(global_phones) > 0

    for seg_idx, seg_start in enumerate(
        tqdm(starts, desc="Stage 2/7 | Building segments", unit="seg")
    ):
        seg_end = seg_start + target_len

        phones: List[str] = []

        if has_phones:
            while phone_ptr < len(global_phones) and global_phones[phone_ptr].end <= seg_start:
                phone_ptr += 1

            p = phone_ptr
            while p < len(global_phones) and global_phones[p].start < seg_end:
                ph = global_phones[p].phone

                if silence_set is None or ph not in silence_set:
                    phones.append(ph)

                p += 1

        while file_ptr + 1 < len(file_infos) and file_infos[file_ptr].global_end <= seg_start:
            file_ptr += 1

        src: List[str] = []
        k = file_ptr

        while k < len(file_infos) and file_infos[k].global_start < seg_end:
            src.append(file_infos[k].path)
            k += 1

        segments.append(
            SegmentRecord(
                seg_idx=seg_idx,
                start=seg_start,
                end=seg_end,
                duration_sec=(seg_end - seg_start) / fs,
                phones=phones,
                phone_str=" ".join(phones),
                source_files=src,
            )
        )

    logger.info(f"Segments: {len(segments)}")

    return segments


# ============================================================
# Datasets
# ============================================================

class ArchiveDataset(Dataset):
    """
    Archive tokenization dataset.

    Returns:
      wav: (1, N)
      seg_idx
    """

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

        fi = 0
        for i, seg in enumerate(segments):
            while fi + 1 < len(file_infos) and file_infos[fi].global_end <= seg.start:
                fi += 1

            self._seg_to_file[i] = fi

    def _load_segment_wav(self, seg: SegmentRecord) -> torch.Tensor:
        rem_start, rem_end = seg.start, seg.end

        parts: List[torch.Tensor] = []
        fi = self._seg_to_file[seg.seg_idx]

        while rem_start < rem_end and fi < len(self.file_infos):
            f = self.file_infos[fi]

            if f.global_start >= rem_end:
                break

            local_s = max(rem_start, f.global_start) - f.global_start
            local_e = min(rem_end, f.global_end) - f.global_start

            wav, sr = _load_audio_mono(f.path)
            wav, _ = _ensure_sr(wav, sr, self.fs)

            parts.append(wav[:, local_s:local_e])

            rem_start = f.global_end
            fi += 1

        if not parts:
            return torch.zeros((1, self.target_len), dtype=torch.float32)

        out = torch.cat(parts, dim=-1)

        if out.size(-1) < self.target_len:
            out = F.pad(out, (0, self.target_len - out.size(-1)))
        else:
            out = out[..., :self.target_len]

        return out

    def __len__(self) -> int:
        return len(self.segments)

    def __getitem__(self, idx: int):
        seg = self.segments[idx]
        wav = self._load_segment_wav(seg)
        return wav, seg.seg_idx


class QueryDataset(Dataset):
    """
    Query dataset — augmented versions of selected archive segments.

    Returns:
      aug_wav: (1, N)
      seg_idx
    """

    def __init__(
        self,
        archive_ds: ArchiveDataset,
        query_indices: List[int],
        augmenter,
        seed: int,
    ):
        super().__init__()

        self.archive_ds = archive_ds
        self.query_indices = query_indices
        self.augmenter = augmenter
        self.seed = seed

    def __len__(self) -> int:
        return len(self.query_indices)

    def __getitem__(self, idx: int):
        seg_idx = self.query_indices[idx]
        seg = self.archive_ds.segments[seg_idx]

        wav = self.archive_ds._load_segment_wav(seg)

        rng = random.Random(self.seed + seg_idx)
        aug_wav = self.augmenter(wav, rng=rng)

        return aug_wav, seg_idx


def _collate(batch):
    wavs = torch.stack([x[0] for x in batch], dim=0)  # [B, 1, N]
    ids = [x[1] for x in batch]
    return wavs, ids


# ============================================================
# Token normalization / tokenization runner
# ============================================================

def _normalise_tokens(out: Any) -> List[List[int]]:
    def _to_ints(x):
        if isinstance(x, torch.Tensor):
            return [int(v) for v in x.detach().cpu().reshape(-1).tolist()]
        return [int(v) for v in x]

    if isinstance(out, torch.Tensor):
        out = out.detach().cpu()

        if out.ndim == 1:
            return [_to_ints(out)]

        return [_to_ints(out[i]) for i in range(out.size(0))]

    if isinstance(out, (list, tuple)):
        return [_to_ints(item) for item in out]

    if isinstance(out, dict):
        for k in ("tokens", "token_ids", "sequences", "seqs"):
            if k in out:
                return _normalise_tokens(out[k])

    raise TypeError(f"Unsupported tokenize() output type: {type(out)}")


@torch.no_grad()
def tokenize_dataset(
    tokenizer,
    dataset: Dataset,
    batch_size: int,
    device: torch.device,
    desc: str,
) -> Tuple[List[int], List[List[int]]]:

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=_collate,
    )

    tokenizer.eval()
    tokenizer.to(device)

    seg_ids_out: List[int] = []
    tokens_out: List[List[int]] = []

    pbar = tqdm(loader, total=len(loader), desc=desc, unit="batch")

    for wavs, seg_ids in pbar:
        wavs = wavs.to(device).contiguous()

        batch_tokens = tokenizer.tokenize(wavs)
        batch_tokens = _normalise_tokens(batch_tokens)

        if len(batch_tokens) != len(seg_ids):
            raise RuntimeError(
                f"Batch size mismatch: {len(batch_tokens)} tokens for {len(seg_ids)} inputs"
            )

        seg_ids_out.extend(seg_ids)
        tokens_out.extend(batch_tokens)

    return seg_ids_out, tokens_out


# ============================================================
# Cache helpers
# ============================================================

def save_cache(path: str, segments: List[SegmentRecord], tokens: List[List[int]], meta: Dict[str, Any]) -> None:
    torch.save(
        {
            "segments": [asdict(s) for s in segments],
            "tokens": tokens,
            "meta": meta,
        },
        path,
    )


def load_cache(path: str) -> Tuple[List[SegmentRecord], List[List[int]], Dict[str, Any]]:
    d = torch.load(path, map_location="cpu")
    return [SegmentRecord(**s) for s in d["segments"]], d["tokens"], d.get("meta", {})


# ============================================================
# Edit distance
# ============================================================

try:
    from rapidfuzz.distance import Levenshtein as _RFL

    def edit_distance(a, b) -> int:
        return int(_RFL.distance(a, b))

except Exception:

    def edit_distance(a, b) -> int:
        n, m = len(a), len(b)

        if n == 0:
            return m

        if m == 0:
            return n

        if m > n:
            a, b, n, m = b, a, m, n

        prev = list(range(m + 1))
        cur = [0] * (m + 1)

        for i in range(1, n + 1):
            cur[0] = i

            for j in range(1, m + 1):
                cost = 0 if a[i - 1] == b[j - 1] else 1

                cur[j] = min(
                    prev[j] + 1,
                    cur[j - 1] + 1,
                    prev[j - 1] + cost,
                )

            prev, cur = cur, prev

        return prev[m]


def ned(a, b) -> float:
    return edit_distance(a, b) / max(len(a), len(b), 1)


# ============================================================
# Relevance oracles
# ============================================================

def overlap_relevant(query: SegmentRecord, segments: List[SegmentRecord]) -> Set[int]:
    return {
        s.seg_idx
        for s in segments
        if min(query.end, s.end) - max(query.start, s.start) > 0
    }


def exact_phone_map(segments: List[SegmentRecord]) -> Dict[str, List[int]]:
    mp: Dict[str, List[int]] = {}

    for s in segments:
        mp.setdefault(s.phone_str, []).append(s.seg_idx)

    return mp


def relaxed_phone_relevant(
    query_phones: List[str],
    segments: List[SegmentRecord],
    threshold: float,
) -> Set[int]:
    return {
        s.seg_idx
        for s in segments
        if ned(query_phones, s.phones) <= threshold
    }


# ============================================================
# Ranking
# ============================================================

def rank_by_ned(
    query_tokens: List[int],
    archive_tokens: List[List[int]],
) -> List[Tuple[int, float]]:

    ql = len(query_tokens)
    scores = []

    for idx, cand in enumerate(archive_tokens):
        lb = abs(ql - len(cand)) / max(ql, len(cand), 1)

        if lb >= 1.0:
            score = lb
        else:
            score = ned(query_tokens, cand)

        scores.append((idx, score))

    scores.sort(key=lambda x: (x[1], x[0]))

    return scores


def first_hit(ranked: Sequence[int], relevant: Set[int]) -> Optional[int]:
    for r, idx in enumerate(ranked, 1):
        if idx in relevant:
            return r

    return None


# ============================================================
# Metrics
# ============================================================

def compute_metrics(
    first_ranks: List[Optional[int]],
    ks: Sequence[int],
) -> Dict[str, float]:

    n = len(first_ranks)

    mrr = 0.0
    hits: Dict[int, int] = {k: 0 for k in ks}
    finite: List[int] = []

    for r in first_ranks:
        if r is not None:
            finite.append(r)
            mrr += 1.0 / r

            for k in ks:
                if r <= k:
                    hits[k] += 1

    out: Dict[str, float] = {
        f"Recall@{k}": hits[k] / n
        for k in ks
    }

    out["MRR"] = mrr / n
    out["HitRate"] = len(finite) / n
    out["MeanFirstRelevantRank"] = sum(finite) / len(finite) if finite else float("inf")
    out["MedianFirstRelevantRank"] = (
        float(sorted(finite)[len(finite) // 2])
        if finite
        else float("inf")
    )

    return out


# ============================================================
# Output helpers
# ============================================================

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

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    with open(path, "w", newline="") as f:
        w = csv.writer(f)

        if segment_only:
            w.writerow(
                [
                    "query_num",
                    "query_seg_idx",
                    "query_start",
                    "query_end",
                    "query_token_len",
                    "segment_overlap_first_rank",
                    "topk_rank",
                    "retrieved_seg_idx",
                    "retrieved_start",
                    "retrieved_end",
                    "retrieved_ned",
                    "retrieved_source_files",
                ]
            )

            for qi, qseg_idx in enumerate(query_indices):
                qseg = segments[qseg_idx]

                for rank_pos, (ret_idx, score) in enumerate(ranked_lists[qi][:topk], 1):
                    rseg = segments[ret_idx]

                    w.writerow(
                        [
                            qi,
                            qseg_idx,
                            qseg.start,
                            qseg.end,
                            len(query_tokens[qi]),
                            overlap_first[qi],
                            rank_pos,
                            ret_idx,
                            rseg.start,
                            rseg.end,
                            f"{score:.6f}",
                            "|".join(rseg.source_files),
                        ]
                    )

        else:
            w.writerow(
                [
                    "query_num",
                    "query_seg_idx",
                    "query_phone_str",
                    "query_token_len",
                    "overlap_first_rank",
                    "exact_first_rank",
                    "relaxed_first_rank",
                    "topk_rank",
                    "retrieved_seg_idx",
                    "retrieved_ned",
                    "retrieved_phone_str",
                ]
            )

            for qi, qseg_idx in enumerate(query_indices):
                qseg = segments[qseg_idx]

                for rank_pos, (ret_idx, score) in enumerate(ranked_lists[qi][:topk], 1):
                    w.writerow(
                        [
                            qi,
                            qseg_idx,
                            qseg.phone_str,
                            len(query_tokens[qi]),
                            overlap_first[qi],
                            exact_first[qi],
                            relaxed_first[qi],
                            rank_pos,
                            ret_idx,
                            f"{score:.6f}",
                            segments[ret_idx].phone_str,
                        ]
                    )


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        "SSL-KMeans retrieval eval — selectable HuBERT / wav2vec2 / WavLM tokenizer"
    )

    parser.add_argument("--root_dir", type=str, default=os.path.dirname(os.getcwd()))
    parser.add_argument("--config", type=str, default="mainFull.yaml")

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

    parser.add_argument(
        "--segment_only",
        action="store_true",
        help="Run only segment-level overlap retrieval. Does not require PHN files.",
    )

    parser.add_argument(
        "--path_col",
        type=str,
        default="path",
        help="Column name in audio_csv containing audio paths.",
    )

    # SSL tokenizer args.
    parser.add_argument(
        "--ssl_model",
        type=str,
        required=True,
        choices=["hubert", "wav2vec2", "wavlm"],
        help="SSL semantic tokenizer to evaluate.",
    )

    parser.add_argument(
        "--layer_num",
        type=int,
        default=7,
        help="SSL hidden layer used for KMeans. For chaanks LibriSpeech files, use 7.",
    )

    parser.add_argument(
        "--k",
        type=int,
        default=512,
        help="Number of KMeans clusters. For these experiments use 512.",
    )

    parser.add_argument(
        "--kmeans_repo_id",
        type=str,
        default="chaanks/Kmeans",
        help="Hugging Face repo containing KMeans files.",
    )

    parser.add_argument(
        "--kmeans_dataset",
        type=str,
        default="LibriSpeech",
        help="KMeans dataset prefix. Usually LibriSpeech.",
    )

    parser.add_argument(
        "--save_root",
        type=str,
        default="pretrained_ssl",
        help="Where to cache SpeechBrain/HF SSL models.",
    )

    parser.add_argument(
        "--source_override",
        type=str,
        default=None,
        help="Optional HF source override. Usually leave unset.",
    )

    parser.add_argument(
        "--no_dedup",
        action="store_true",
        help="Disable consecutive duplicate removal from SSL-KMeans token streams.",
    )
    parser.add_argument(
        "--dist_root",
        type=str,
        required=True,
        help="Distortion root used to construct positive query views.",
    )
    parser.add_argument(
        "--use_time_compression",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Slightly compress only positive query views and right-pad them to 3 seconds.",
    )

    args = parser.parse_args()

    warnings.filterwarnings("default")

    os.makedirs(args.save_dir, exist_ok=True)

    logger = setup_logger(os.path.join(args.save_dir, "retrieval_eval_ssl.log"))
    set_seed(args.seed)

    logger.info("==== SSL retrieval eval started ====")
    logger.info(json.dumps(vars(args), indent=2))

    with open(os.path.join(args.root_dir, "config", args.config)) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    fs = int(cfg["fs"])
    seg_sec = float(cfg["seglen"])

    if fs != 16000:
        logger.warning(
            f"Config fs={fs}. SSL encoders expect 16 kHz. "
            f"The dataset loader will resample each audio file to fs={fs}; "
            f"for standard SSL tokenization, fs should be 16000."
        )

    device = torch.device(
        f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    )

    logger.info(f"Device: {device}")

    logger.info("Loading SSL-KMeans tokenizer …")

    ssl_cfg = SSLKMeansConfig(
        ssl_model=args.ssl_model,
        k=args.k,
        layer_num=args.layer_num,
        dataset=args.kmeans_dataset,
        kmeans_repo_id=args.kmeans_repo_id,
        save_root=args.save_root,
        dedup=not args.no_dedup,
        source_override=args.source_override,
    )

    tokenizer = SSLKMeansTokenizer(
        config=ssl_cfg,
        device=device,
    )

    tokenizer.eval()
    tokenizer.to(device)

    silence_set = {"h#", "pau", "epi"} if args.ignore_silence else None

    dedup_tag = "rawunits" if args.no_dedup else "dedup"
    cache_path = os.path.join(
        args.save_dir,
        f"archive_cache_ssl_{args.ssl_model}_k{args.k}_L{args.layer_num}_{dedup_tag}.pt",
    )
    metrics_path = os.path.join(args.save_dir, "metrics.json")
    csv_path = os.path.join(args.save_dir, "per_query_topk.csv")

    # --------------------------------------------------------
    # Stage 1: archive metadata
    # --------------------------------------------------------

    t = time.time()

    file_infos, global_phones = build_archive_metadata(
        audio_csv=args.audio_csv,
        fs=fs,
        path_col=args.path_col,
        logger=logger,
        require_phn=not args.segment_only,
    )

    logger.info(f"Stage 1 done in {time.time() - t:.1f}s")

    if args.segment_only:
        logger.info("Running in SEGMENT-ONLY mode. PHN metrics will be skipped.")
    else:
        if len(global_phones) == 0:
            raise RuntimeError(
                "No phone intervals found, but --segment_only was not set. "
                "Either provide PHN files or run with --segment_only."
            )

    # --------------------------------------------------------
    # Stage 2: segments
    # --------------------------------------------------------

    t = time.time()

    segments = build_segments(
        file_infos=file_infos,
        global_phones=global_phones,
        fs=fs,
        window_sec=seg_sec,
        hop_sec=args.hop_sec,
        silence_set=silence_set,
        logger=logger,
    )

    logger.info(f"Stage 2 done in {time.time() - t:.1f}s")

    # --------------------------------------------------------
    # Stage 3: tokenize archive
    # --------------------------------------------------------

    cache_meta = {
        "ssl_model": args.ssl_model,
        "k": args.k,
        "layer_num": args.layer_num,
        "kmeans_repo_id": args.kmeans_repo_id,
        "kmeans_dataset": args.kmeans_dataset,
        "dedup": not args.no_dedup,
        "fs": fs,
        "seg_sec": seg_sec,
        "hop_sec": args.hop_sec,
        "audio_csv": args.audio_csv,
    }

    if args.use_cache and os.path.isfile(cache_path):
        logger.info(f"Loading cache from {cache_path}")
        segments, archive_tokens, loaded_meta = load_cache(cache_path)
        logger.info(f"Loaded cache meta: {json.dumps(loaded_meta, indent=2)}")

    else:
        t = time.time()

        archive_ds = ArchiveDataset(
            file_infos=file_infos,
            segments=segments,
            fs=fs,
            context_sec=seg_sec,
        )

        seg_ids, tokens_unsorted = tokenize_dataset(
            tokenizer=tokenizer,
            dataset=archive_ds,
            batch_size=args.batch_size,
            device=device,
            desc="Stage 3/7 | Tokenising archive with SSL-KMeans",
        )

        archive_tokens: List[Optional[List[int]]] = [None] * len(segments)

        for sid, tok in zip(seg_ids, tokens_unsorted):
            archive_tokens[sid] = tok

        missing = sum(x is None for x in archive_tokens)

        if missing:
            raise RuntimeError(f"Missing tokenisations for {missing} segments")

        archive_tokens = list(archive_tokens)

        logger.info(f"Stage 3 done in {time.time() - t:.1f}s")

        if args.use_cache:
            save_cache(cache_path, segments, archive_tokens, cache_meta)
            logger.info(f"Cache saved to {cache_path}")

    # --------------------------------------------------------
    # Stage 4: sample queries
    # --------------------------------------------------------

    if args.segment_only:
        eligible = [s.seg_idx for s in segments]
    else:
        eligible = [s.seg_idx for s in segments if s.phones]

    if len(eligible) < args.n_queries:
        raise ValueError(
            f"Only {len(eligible)} eligible segments, need {args.n_queries}"
        )

    query_indices = random.Random(args.seed).sample(eligible, args.n_queries)

    logger.info(f"Sampled {len(query_indices)} queries")

    # --------------------------------------------------------
    # Stage 5: tokenize augmented queries
    # --------------------------------------------------------

    from utils.dataset import (
        AudioAugmenter,
        AugConfig,
        _collect_audio_files_from_dirs,
    )

    dist_root = args.dist_root

    augmenter = AudioAugmenter(
        sr=fs,
        cfg=AugConfig(),
        use_time_compression=args.use_time_compression,
        noise_files=_collect_audio_files_from_dirs(
            [
                os.path.join(dist_root, d)
                for d in (
                    "noise_16k",
                    "pointsource_noises",
                    "real_rirs_isotropic_noises",
                )
            ],
            exts=(".wav", ".flac"),
        ),
        rir_files=_collect_audio_files_from_dirs(
            [os.path.join(dist_root, "rir_16k")],
            exts=(".wav", ".flac"),
        ),
    )

    query_archive_ds = ArchiveDataset(
        file_infos=file_infos,
        segments=segments,
        fs=fs,
        context_sec=seg_sec,
    )

    query_ds = QueryDataset(
        archive_ds=query_archive_ds,
        query_indices=query_indices,
        augmenter=augmenter,
        seed=args.seed + 100_000,
    )

    t = time.time()

    q_seg_ids, q_tokens_unsorted = tokenize_dataset(
        tokenizer=tokenizer,
        dataset=query_ds,
        batch_size=args.batch_size,
        device=device,
        desc="Stage 5/7 | Tokenising queries with SSL-KMeans",
    )

    q_token_map = {
        sid: tok
        for sid, tok in zip(q_seg_ids, q_tokens_unsorted)
    }

    query_tokens = [q_token_map[i] for i in query_indices]

    logger.info(f"Stage 5 done in {time.time() - t:.1f}s")

    # --------------------------------------------------------
    # Stage 6: ranking
    # --------------------------------------------------------

    t = time.time()

    ranked_lists: List[List[Tuple[int, float]]] = []
    overlap_first: List[Optional[int]] = []
    exact_first: List[Optional[int]] = []
    relaxed_first: List[Optional[int]] = []

    if not args.segment_only:
        ep_map = exact_phone_map(segments)
    else:
        ep_map = None

    for qi in tqdm(
        range(len(query_indices)),
        desc="Stage 6/7 | Ranking",
        unit="query",
    ):
        qseg = segments[query_indices[qi]]
        qtok = query_tokens[qi]

        ranked = rank_by_ned(qtok, archive_tokens)
        ranked_lists.append(ranked)

        ranked_idx = [i for i, _ in ranked]

        overlap_first.append(
            first_hit(
                ranked_idx,
                overlap_relevant(qseg, segments),
            )
        )

        if args.segment_only:
            exact_first.append(None)
            relaxed_first.append(None)
        else:
            assert ep_map is not None

            exact_first.append(
                first_hit(
                    ranked_idx,
                    set(ep_map.get(qseg.phone_str, [])),
                )
            )

            relaxed_first.append(
                first_hit(
                    ranked_idx,
                    relaxed_phone_relevant(
                        qseg.phones,
                        segments,
                        args.relaxed_threshold,
                    ),
                )
            )

    logger.info(f"Stage 6 done in {time.time() - t:.1f}s")

    # --------------------------------------------------------
    # Stage 7: metrics + save
    # --------------------------------------------------------

    t = time.time()

    ks = [1, 5, 10, 20]

    metrics: Dict[str, Any] = {
        "tokenizer": {
            "type": "ssl_kmeans",
            "ssl_model": args.ssl_model,
            "k": args.k,
            "layer_num": args.layer_num,
            "kmeans_repo_id": args.kmeans_repo_id,
            "kmeans_dataset": args.kmeans_dataset,
            "dedup": not args.no_dedup,
        },
        "num_archive_segments": len(segments),
        "num_queries": len(query_indices),
        "window_sec": seg_sec,
        "hop_sec": args.hop_sec,
        "ranking_metric": "normalized_edit_distance_on_tokens",
        "segment_only": args.segment_only,
        "segment_overlap": compute_metrics(overlap_first, ks),
        "archive_token_length": {
            "mean": float(sum(len(x) for x in archive_tokens) / max(len(archive_tokens), 1)),
            "min": int(min(len(x) for x in archive_tokens)) if archive_tokens else 0,
            "max": int(max(len(x) for x in archive_tokens)) if archive_tokens else 0,
        },
        "query_token_length": {
            "mean": float(sum(len(x) for x in query_tokens) / max(len(query_tokens), 1)),
            "min": int(min(len(x) for x in query_tokens)) if query_tokens else 0,
            "max": int(max(len(x) for x in query_tokens)) if query_tokens else 0,
        },
    }

    if not args.segment_only:
        metrics["relaxed_threshold"] = args.relaxed_threshold
        metrics["phoneme_exact"] = compute_metrics(exact_first, ks)
        metrics["phoneme_relaxed"] = compute_metrics(relaxed_first, ks)

    logger.info("==== Final Metrics ====")
    logger.info(json.dumps(metrics, indent=2))

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    write_query_csv(
        path=csv_path,
        query_indices=query_indices,
        query_tokens=query_tokens,
        segments=segments,
        ranked_lists=ranked_lists,
        overlap_first=overlap_first,
        exact_first=exact_first,
        relaxed_first=relaxed_first,
        topk=args.topk_dump,
        segment_only=args.segment_only,
    )

    logger.info(f"Metrics  → {metrics_path}")
    logger.info(f"Per-query CSV → {csv_path}")
    logger.info(f"Stage 7 done in {time.time() - t:.1f}s")
    logger.info("==== SSL retrieval eval finished ====")


if __name__ == "__main__":
    main()
