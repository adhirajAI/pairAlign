#!/usr/bin/env python3
"""
Unified raw-vs-Beta cross-attention analysis for TIMIT.

This script combines:
  * phone-boundary and segment-alignment analysis;
  * HuBERT-style token/phone statistics;
  * speaker leakage and leave-one-speaker-out analysis;
  * paired bootstrap raw-vs-Beta comparisons;
  * discrete ABX-style analysis;
  * monotonicity summaries inherited from the artifact generator;
  * per-speaker raw/Beta alignment summaries.

Expected artifact contract
--------------------------
Each item_*.pt should come from the metadata-aware cross-attention artifact
generator and contain, for each requested side ("anc" and/or "pos"):

  tokens
  raw_attn / beta_attn
  raw_frame2tok / beta_frame2tok            (preferred; fallback is supported)
  raw_viterbi_path / beta_viterbi_path      (preferred; fallback is supported)
  raw_frame_token_ids / beta_frame_token_ids
  raw_metrics / beta_metrics
  phone_labels_61 / phone_labels_39
  phone_alignment_valid

The artifact generator already stores raw and Beta attention separately. Beta
is an analysis-only positional reweighting; it is never treated here as the
model's native attention.

Outputs
-------
Core outputs:
  aligned_sequences/*.pt
  per_item_alignment_metrics.csv
  global_hubert_metrics.csv
  paired_beta_improvement.csv
  token_phone_mapping_raw.csv
  token_phone_mapping_beta.csv
  abx_discrete_metrics.csv                  (unless --skip_abx)

Additional monotonicity/speaker outputs:
  monotonicity_summary.csv
  per_speaker_alignment_summary.csv

Optional:
  frame_alignment.csv.gz                    (with --write_frame_csv)

Terminology
-----------
"ABX-style" is deliberate. Official acoustic ABX generally evaluates
continuous frame representations with DTW. Here the positional prior changes
token-to-frame alignment, not the underlying encoder representation, so the
raw-vs-Beta comparison is performed on aligned discrete token sequences.
"""

import argparse
import csv
import gzip
import json
import math
import os
import random
from collections import Counter, defaultdict
from glob import glob
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from tqdm.auto import tqdm


# -----------------------------------------------------------------------------
# Metric definitions
# -----------------------------------------------------------------------------

SILENCE_39 = {"sil"}
IGNORE_PHONES = {"<ignore>", "<outside>", "<pad>", ""}

HIGHER_IS_BETTER = {
    "boundary_f1_10ms",
    "boundary_f1_20ms",
    "boundary_f1_30ms",
    "boundary_precision_10ms",
    "boundary_precision_20ms",
    "boundary_precision_30ms",
    "boundary_recall_10ms",
    "boundary_recall_20ms",
    "boundary_recall_30ms",
    "token_segment_phone_purity",
    "phone_segment_token_purity",
    "expected_frame_corr",
    "viterbi_log_score_mean",
}

LOWER_IS_BETTER = {
    "boundary_mae_frames_20ms",
    "boundary_mae_ms_20ms",
    "oversegmentation_ratio_abs_log",
    "mean_phones_per_token_segment",
    "mean_tokens_per_phone_segment",
    "expected_frame_violation_rate",
    "expected_frame_violation_mag",
    "peak_frame_violation_rate",
    "peak_frame_violation_mag",
    "mean_abs_diagonal_error",
    "mean_peak_diagonal_error",
    "frame_to_token_expected_violation_rate",
    "frame_to_token_expected_violation_mag",
    "frame_to_token_peak_violation_rate",
    "frame_to_token_peak_violation_mag",
}

DEFAULT_CORE_MONOTONICITY_METRICS = [
    "expected_frame_violation_rate",
    "peak_frame_violation_rate",
    "frame_to_token_expected_violation_rate",
    "frame_to_token_peak_violation_rate",
    "mean_abs_diagonal_error",
    "expected_frame_corr",
]

DEFAULT_SUPPORT_MONOTONICITY_METRICS = [
    "expected_frame_violation_mag",
    "peak_frame_violation_mag",
    "frame_to_token_expected_violation_mag",
    "frame_to_token_peak_violation_mag",
    "mean_peak_diagonal_error",
    "viterbi_log_score_mean",
]


# -----------------------------------------------------------------------------
# Generic I/O / conversion helpers
# -----------------------------------------------------------------------------

def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if not rows:
        return

    keys: List[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)

    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def load_items(input_dir: str) -> List[Dict[str, Any]]:
    paths = sorted(glob(os.path.join(input_dir, "item_*.pt")))
    if not paths:
        raise FileNotFoundError(f"No item_*.pt files in {input_dir}")

    items = []
    for path in tqdm(paths, desc="Loading artifacts"):
        item = torch.load(path, map_location="cpu")
        item["_path"] = path
        items.append(item)
    return items


def as_long_vector(x: Any) -> torch.LongTensor:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().reshape(-1).long()
    return torch.tensor(x, dtype=torch.long).reshape(-1)


def as_float_matrix(x: Any) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().float()
    return torch.tensor(x, dtype=torch.float32)


def safe_float(x: Any) -> float:
    if isinstance(x, torch.Tensor):
        if x.numel() != 1:
            raise ValueError(
                f"Expected scalar tensor, got shape {tuple(x.shape)}"
            )
        return float(x.detach().cpu().item())
    return float(x)


# -----------------------------------------------------------------------------
# Alignment reconstruction
# -----------------------------------------------------------------------------

def frame_to_token_distribution(
    token_to_frame: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Convert token->frame attention (M,T) to row-normalized frame->token scores
    (T,M).
    """
    attention = (
        token_to_frame.detach().cpu().float().clamp_min(0.0)
    )
    if attention.numel() == 0:
        return torch.empty(0, 0)

    attention = (
        attention
        / attention.sum(dim=-1, keepdim=True).clamp_min(eps)
    )

    frame_to_token = attention.transpose(0, 1).contiguous()
    return (
        frame_to_token
        / frame_to_token.sum(dim=-1, keepdim=True).clamp_min(eps)
    )


def viterbi_monotonic_path(
    frame_to_token: torch.Tensor,
    eps: float = 1e-12,
) -> torch.LongTensor:
    """
    Highest-scoring monotonic frame->token path with transitions:
        stay at token t, or advance from t-1 to t.

    The path starts at token 0 and must terminate at the final token.
    """
    scores = frame_to_token.detach().cpu().float()
    frame_count, token_count = scores.shape

    if frame_count == 0 or token_count == 0:
        return torch.empty(0, dtype=torch.long)

    if token_count > frame_count:
        raise ValueError(
            "Cannot visit every token with stay/advance-by-one transitions when "
            f"token_count > frame_count ({token_count} > {frame_count})."
        )

    log_scores = scores.clamp_min(eps).log()

    dp = torch.full(
        (frame_count, token_count),
        float("-inf"),
        dtype=log_scores.dtype,
    )
    backpointer = torch.full(
        (frame_count, token_count),
        -1,
        dtype=torch.long,
    )

    dp[0, 0] = log_scores[0, 0]
    backpointer[0, 0] = 0

    for frame in range(1, frame_count):
        dp[frame, 0] = dp[frame - 1, 0] + log_scores[frame, 0]
        backpointer[frame, 0] = 0

        max_reachable = min(token_count - 1, frame)
        for token in range(1, max_reachable + 1):
            stay = dp[frame - 1, token]
            advance = dp[frame - 1, token - 1]

            if advance > stay:
                dp[frame, token] = advance + log_scores[frame, token]
                backpointer[frame, token] = token - 1
            else:
                dp[frame, token] = stay + log_scores[frame, token]
                backpointer[frame, token] = token

    if not torch.isfinite(dp[-1, -1]):
        raise RuntimeError(
            "No valid monotonic Viterbi path reached the final token."
        )

    path = torch.empty(frame_count, dtype=torch.long)
    token = token_count - 1
    path[-1] = token

    for frame in range(frame_count - 1, 0, -1):
        token = int(backpointer[frame, token].item())
        if token < 0:
            raise RuntimeError("Invalid Viterbi backpointer.")
        path[frame - 1] = token

    return path


def get_frame_to_token_distribution(
    side_obj: Dict[str, Any],
    mode: str,
) -> torch.Tensor:
    frame2tok_key = f"{mode}_frame2tok"
    attn_key = f"{mode}_attn"

    if frame2tok_key in side_obj:
        scores = as_float_matrix(side_obj[frame2tok_key]).clamp_min(0.0)
        if scores.numel() == 0:
            return scores
        return scores / scores.sum(dim=-1, keepdim=True).clamp_min(1e-12)

    if attn_key not in side_obj:
        raise KeyError(
            f"Neither {frame2tok_key} nor {attn_key} exists in artifact."
        )

    return frame_to_token_distribution(as_float_matrix(side_obj[attn_key]))


def get_alignment(
    side_obj: Dict[str, Any],
    mode: str,
) -> Tuple[torch.LongTensor, torch.LongTensor]:
    token_ids = as_long_vector(side_obj["tokens"])
    path_key = f"{mode}_viterbi_path"
    frame_token_key = f"{mode}_frame_token_ids"

    if path_key in side_obj:
        path = as_long_vector(side_obj[path_key])
    else:
        path = viterbi_monotonic_path(
            get_frame_to_token_distribution(side_obj, mode)
        )

    if frame_token_key in side_obj:
        frame_token_ids = as_long_vector(side_obj[frame_token_key])
    else:
        if token_ids.numel() == 0:
            frame_token_ids = torch.empty_like(path)
        else:
            if path.numel() and int(path.max()) >= token_ids.numel():
                raise ValueError(
                    "Alignment path contains token position beyond token list."
                )
            frame_token_ids = token_ids[path]

    return path, frame_token_ids


def expected_token_per_frame(
    frame_to_token: torch.Tensor,
) -> torch.Tensor:
    if frame_to_token.numel() == 0:
        return torch.empty(0)

    token_count = frame_to_token.size(1)
    token_positions = torch.arange(
        token_count,
        dtype=frame_to_token.dtype,
    )
    return (
        frame_to_token
        * token_positions.unsqueeze(0)
    ).sum(dim=-1)


def peak_token_per_frame(
    frame_to_token: torch.Tensor,
) -> torch.LongTensor:
    if frame_to_token.numel() == 0:
        return torch.empty(0, dtype=torch.long)
    return frame_to_token.argmax(dim=-1).long()


# -----------------------------------------------------------------------------
# Boundary / segment metrics
# -----------------------------------------------------------------------------

def boundaries_from_labels(
    labels: Sequence[Any],
    ignored: set = IGNORE_PHONES,
) -> np.ndarray:
    boundaries = []

    for index in range(1, len(labels)):
        left = str(labels[index - 1])
        right = str(labels[index])

        if left in ignored or right in ignored:
            continue
        if left != right:
            boundaries.append(index)

    return np.asarray(boundaries, dtype=np.int64)


def boundaries_from_token_positions(
    positions: Sequence[int],
) -> np.ndarray:
    values = np.asarray(positions, dtype=np.int64)

    if values.size < 2:
        return np.empty(0, dtype=np.int64)

    return np.flatnonzero(values[1:] != values[:-1]) + 1


def one_to_one_boundary_match(
    predicted: np.ndarray,
    reference: np.ndarray,
    tolerance_frames: int,
) -> Tuple[int, int, int, List[int]]:
    predicted = np.asarray(predicted, dtype=np.int64)
    reference = np.asarray(reference, dtype=np.int64)

    candidates = []
    for predicted_index, boundary in enumerate(predicted):
        left = np.searchsorted(
            reference,
            boundary - tolerance_frames,
            side="left",
        )
        right = np.searchsorted(
            reference,
            boundary + tolerance_frames,
            side="right",
        )

        for reference_index in range(left, right):
            candidates.append(
                (
                    abs(
                        int(boundary)
                        - int(reference[reference_index])
                    ),
                    predicted_index,
                    reference_index,
                )
            )

    candidates.sort()

    used_predicted = set()
    used_reference = set()
    distances: List[int] = []

    for distance, predicted_index, reference_index in candidates:
        if (
            predicted_index not in used_predicted
            and reference_index not in used_reference
        ):
            used_predicted.add(predicted_index)
            used_reference.add(reference_index)
            distances.append(distance)

    true_positive = len(distances)
    false_positive = len(predicted) - true_positive
    false_negative = len(reference) - true_positive

    return true_positive, false_positive, false_negative, distances


def boundary_metrics(
    token_positions: torch.LongTensor,
    phone_labels_61: Sequence[str],
    hop_ms: float,
) -> Dict[str, float]:
    predicted = boundaries_from_token_positions(
        token_positions.numpy()
    )
    reference = boundaries_from_labels(phone_labels_61)

    output: Dict[str, float] = {
        "num_token_boundaries": float(len(predicted)),
        "num_phone_boundaries": float(len(reference)),
    }

    for tolerance in (1, 2, 3):
        tp, fp, fn, distances = one_to_one_boundary_match(
            predicted,
            reference,
            tolerance,
        )

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = (
            2.0 * precision * recall
            / max(precision + recall, 1e-12)
        )

        tolerance_ms = int(round(tolerance * hop_ms))

        output[f"boundary_precision_{tolerance_ms}ms"] = precision
        output[f"boundary_recall_{tolerance_ms}ms"] = recall
        output[f"boundary_f1_{tolerance_ms}ms"] = f1

        if tolerance == 2:
            output["boundary_mae_frames_20ms"] = (
                float(np.mean(distances))
                if distances
                else float("nan")
            )
            output["boundary_mae_ms_20ms"] = (
                float(np.mean(distances) * hop_ms)
                if distances
                else float("nan")
            )

    ratio = len(predicted) / max(len(reference), 1)
    output["token_to_phone_boundary_count_ratio"] = ratio
    output["oversegmentation_ratio_abs_log"] = abs(
        math.log(max(ratio, 1e-12))
    )

    return output


def runs(values: Sequence[Any]) -> List[Tuple[int, int, Any]]:
    if len(values) == 0:
        return []

    output = []
    start = 0
    previous = values[0]

    for index in range(1, len(values)):
        if values[index] != previous:
            output.append((start, index, previous))
            start = index
            previous = values[index]

    output.append((start, len(values), previous))
    return output


def segment_alignment_metrics(
    token_positions: torch.LongTensor,
    phone_labels_39: Sequence[str],
) -> Dict[str, float]:
    positions = token_positions.numpy().tolist()
    length = min(len(positions), len(phone_labels_39))

    positions = positions[:length]
    phones = list(phone_labels_39[:length])

    token_runs = runs(positions)
    phone_runs = [
        (start, end, phone)
        for start, end, phone in runs(phones)
        if phone not in IGNORE_PHONES
    ]

    token_majority_correct = 0
    token_valid_frames = 0
    phones_per_token = []

    for start, end, _ in token_runs:
        valid_phones = [
            phone
            for phone in phones[start:end]
            if phone not in IGNORE_PHONES
        ]
        if not valid_phones:
            continue

        counts = Counter(valid_phones)
        token_majority_correct += max(counts.values())
        token_valid_frames += len(valid_phones)
        phones_per_token.append(len(counts))

    phone_majority_correct = 0
    phone_valid_frames = 0
    tokens_per_phone = []

    for start, end, _ in phone_runs:
        values = positions[start:end]
        if not values:
            continue

        counts = Counter(values)
        phone_majority_correct += max(counts.values())
        phone_valid_frames += len(values)
        tokens_per_phone.append(len(counts))

    return {
        "token_segment_phone_purity": (
            token_majority_correct / max(token_valid_frames, 1)
        ),
        "phone_segment_token_purity": (
            phone_majority_correct / max(phone_valid_frames, 1)
        ),
        "mean_phones_per_token_segment": (
            float(np.mean(phones_per_token))
            if phones_per_token
            else float("nan")
        ),
        "mean_tokens_per_phone_segment": (
            float(np.mean(tokens_per_phone))
            if tokens_per_phone
            else float("nan")
        ),
    }


# -----------------------------------------------------------------------------
# HuBERT-style token / phone / speaker metrics
# -----------------------------------------------------------------------------

def entropy_from_counter(counter: Counter) -> float:
    total = sum(counter.values())

    if total <= 0:
        return 0.0

    probabilities = np.asarray(
        [
            count / total
            for count in counter.values()
            if count > 0
        ],
        dtype=np.float64,
    )

    return float(
        -(probabilities * np.log(probabilities)).sum()
    )


def mutual_information_from_joint(joint: Counter) -> float:
    total = sum(joint.values())

    if total <= 0:
        return 0.0

    first_counts = Counter()
    second_counts = Counter()

    for (first, second), count in joint.items():
        first_counts[first] += count
        second_counts[second] += count

    mutual_information = 0.0

    for (first, second), count in joint.items():
        p_joint = count / total
        p_first = first_counts[first] / total
        p_second = second_counts[second] / total

        mutual_information += p_joint * math.log(
            p_joint / max(p_first * p_second, 1e-300)
        )

    return float(mutual_information)


def conditional_mi_token_speaker_given_phone(
    triples: Counter,
) -> Tuple[float, float]:
    """
    Return I(Token;Speaker | Phone) and H(Speaker | Phone).
    """
    total = sum(triples.values())

    if total <= 0:
        return 0.0, 0.0

    phone_counts = Counter()
    token_speaker_by_phone = defaultdict(Counter)
    speaker_by_phone = defaultdict(Counter)

    for (phone, token, speaker), count in triples.items():
        phone_counts[phone] += count
        token_speaker_by_phone[phone][(token, speaker)] += count
        speaker_by_phone[phone][speaker] += count

    conditional_mi = 0.0
    conditional_speaker_entropy = 0.0

    for phone, phone_count in phone_counts.items():
        weight = phone_count / total

        conditional_mi += (
            weight
            * mutual_information_from_joint(
                token_speaker_by_phone[phone]
            )
        )
        conditional_speaker_entropy += (
            weight
            * entropy_from_counter(
                speaker_by_phone[phone]
            )
        )

    return conditional_mi, conditional_speaker_entropy


def hubert_style_metrics(
    records: List[Dict[str, Any]],
    mode: str,
) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    joint = Counter()
    token_counts = Counter()
    phone_counts = Counter()
    speaker_token_joint = Counter()
    triples = Counter()
    speaker_joint = defaultdict(Counter)

    for record in records:
        speaker = record["speaker_id"]
        tokens = record[f"{mode}_token_ids"]
        phones = record["phone_labels_39"]

        for token, phone in zip(tokens.tolist(), phones):
            if phone in IGNORE_PHONES:
                continue

            token = int(token)

            joint[(phone, token)] += 1
            token_counts[token] += 1
            phone_counts[phone] += 1
            speaker_token_joint[(speaker, token)] += 1
            triples[(phone, token, speaker)] += 1
            speaker_joint[speaker][(phone, token)] += 1

    valid_frames = sum(joint.values())
    if valid_frames == 0:
        raise ValueError("No valid phone-token frames available.")

    # HuBERT-style phone purity: dominant phone inside each token class.
    phone_purity = (
        sum(
            max(joint[(phone, token)] for phone in phone_counts)
            for token in token_counts
        )
        / valid_frames
    )

    # HuBERT-style cluster purity: dominant token inside each phone class.
    cluster_purity = (
        sum(
            max(joint[(phone, token)] for token in token_counts)
            for phone in phone_counts
        )
        / valid_frames
    )

    mi = mutual_information_from_joint(joint)
    phone_entropy = entropy_from_counter(phone_counts)
    pnmi = mi / max(phone_entropy, 1e-12)

    token_entropy = entropy_from_counter(token_counts)
    code_perplexity = math.exp(token_entropy)
    active_vocab = len(token_counts)

    speaker_counts = Counter()
    for (speaker, _), count in speaker_token_joint.items():
        speaker_counts[speaker] += count

    speaker_mi = mutual_information_from_joint(speaker_token_joint)
    speaker_entropy = entropy_from_counter(speaker_counts)
    speaker_nmi = speaker_mi / max(speaker_entropy, 1e-12)

    conditional_mi, speaker_entropy_given_phone = (
        conditional_mi_token_speaker_given_phone(triples)
    )
    conditional_speaker_nmi = (
        conditional_mi / max(speaker_entropy_given_phone, 1e-12)
    )

    silence_joint = Counter()
    for (phone, token), count in joint.items():
        label = "silence" if phone in SILENCE_39 else "speech"
        silence_joint[(label, token)] += count

    silence_purity = (
        sum(
            max(
                silence_joint[(label, token)]
                for label in ("silence", "speech")
            )
            for token in token_counts
        )
        / valid_frames
    )

    # Leave-one-speaker-out token->phone majority mapping.
    loso_correct = 0
    loso_total = 0
    global_phone_majority = phone_counts.most_common(1)[0][0]

    for heldout_speaker in sorted(speaker_joint):
        heldout_joint = speaker_joint[heldout_speaker]

        train_joint = joint.copy()
        for key, count in heldout_joint.items():
            train_joint[key] -= count
            if train_joint[key] <= 0:
                del train_joint[key]

        train_by_token = defaultdict(Counter)
        train_phone_counts = Counter()

        for (phone, token), count in train_joint.items():
            train_by_token[token][phone] += count
            train_phone_counts[phone] += count

        fallback = (
            train_phone_counts.most_common(1)[0][0]
            if train_phone_counts
            else global_phone_majority
        )

        mapping = {
            token: counts.most_common(1)[0][0]
            for token, counts in train_by_token.items()
        }

        for (phone, token), count in heldout_joint.items():
            prediction = mapping.get(token, fallback)
            if prediction == phone:
                loso_correct += count
            loso_total += count

    mapping_rows = []
    by_token = defaultdict(Counter)

    for (phone, token), count in joint.items():
        by_token[token][phone] += count

    for token in sorted(by_token):
        counts = by_token[token]
        total = sum(counts.values())
        dominant_phone, dominant_count = counts.most_common(1)[0]

        mapping_rows.append(
            {
                "mode": mode,
                "token_id": token,
                "frames": total,
                "dominant_phone": dominant_phone,
                "token_phone_purity": dominant_count / total,
                "num_phone_classes": len(counts),
                "phone_distribution": json.dumps(
                    dict(counts),
                    sort_keys=True,
                ),
            }
        )

    metrics = {
        "mode": mode,
        "num_valid_frames": valid_frames,
        "active_vocab": active_vocab,
        "codebook_perplexity": code_perplexity,
        "normalized_codebook_perplexity": (
            code_perplexity / max(active_vocab, 1)
        ),
        "phone_purity": phone_purity,
        "cluster_purity": cluster_purity,
        "pnmi": pnmi,
        "silence_binary_purity": silence_purity,
        "speaker_nmi": speaker_nmi,
        "conditional_speaker_nmi_given_phone": conditional_speaker_nmi,
        "loso_token_to_phone_accuracy": (
            loso_correct / max(loso_total, 1)
        ),
    }

    return metrics, mapping_rows


# -----------------------------------------------------------------------------
# Paired raw-vs-Beta summaries
# -----------------------------------------------------------------------------

def bootstrap_mean_delta(
    raw: np.ndarray,
    beta: np.ndarray,
    higher_is_better: bool,
    n_boot: int,
    seed: int,
) -> Tuple[float, float, float, float]:
    finite = np.isfinite(raw) & np.isfinite(beta)
    raw = raw[finite]
    beta = beta[finite]

    if len(raw) == 0:
        return (
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
        )

    improvement = (
        beta - raw
        if higher_is_better
        else raw - beta
    )

    rng = np.random.default_rng(seed)
    bootstrap_indices = rng.integers(
        0,
        len(improvement),
        size=(n_boot, len(improvement)),
    )
    bootstrap_means = improvement[bootstrap_indices].mean(axis=1)

    return (
        float(improvement.mean()),
        float(np.quantile(bootstrap_means, 0.025)),
        float(np.quantile(bootstrap_means, 0.975)),
        float((improvement > 0).mean()),
    )


def paired_improvement_summary(
    per_item_rows: List[Dict[str, Any]],
    n_boot: int,
    seed: int,
) -> List[Dict[str, Any]]:
    metric_names = sorted(
        {
            key[len("raw_"):]
            for row in per_item_rows
            for key in row
            if (
                key.startswith("raw_")
                and f"beta_{key[len('raw_'):]}" in row
            )
        }
    )

    rows = []

    for metric in metric_names:
        if metric in HIGHER_IS_BETTER:
            higher = True
        elif metric in LOWER_IS_BETTER:
            higher = False
        else:
            continue

        raw = np.asarray(
            [
                row.get(f"raw_{metric}", np.nan)
                for row in per_item_rows
            ],
            dtype=float,
        )
        beta = np.asarray(
            [
                row.get(f"beta_{metric}", np.nan)
                for row in per_item_rows
            ],
            dtype=float,
        )

        improvement, ci_low, ci_high, fraction_better = (
            bootstrap_mean_delta(
                raw,
                beta,
                higher,
                n_boot,
                seed,
            )
        )

        finite = np.isfinite(raw) & np.isfinite(beta)

        rows.append(
            {
                "metric": metric,
                "direction": "higher" if higher else "lower",
                "n": int(finite.sum()),
                "raw_mean": float(np.nanmean(raw)),
                "beta_mean": float(np.nanmean(beta)),
                "beta_minus_raw": float(np.nanmean(beta - raw)),
                "mean_improvement": improvement,
                "improvement_ci95_low": ci_low,
                "improvement_ci95_high": ci_high,
                "fraction_items_beta_better": fraction_better,
            }
        )

    return rows


def metric_improvement(
    metric: str,
    raw_value: float,
    beta_value: float,
) -> Optional[float]:
    if metric in HIGHER_IS_BETTER:
        return beta_value - raw_value
    if metric in LOWER_IS_BETTER:
        return raw_value - beta_value
    return None


def available_paired_metrics(
    rows: List[Dict[str, Any]],
) -> List[str]:
    if not rows:
        return []

    metrics = set()
    for row in rows:
        for key in row:
            if not key.startswith("raw_"):
                continue
            metric = key[len("raw_"):]
            if f"beta_{metric}" in row:
                metrics.add(metric)

    return sorted(metrics)


def monotonicity_summary(
    per_item_rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Long-form raw/Beta summary for artifact-native monotonicity metrics.
    """
    monotonicity_metrics = (
        DEFAULT_CORE_MONOTONICITY_METRICS
        + DEFAULT_SUPPORT_MONOTONICITY_METRICS
    )

    output = []

    for side in ("anc", "pos", "both"):
        rows = (
            per_item_rows
            if side == "both"
            else [
                row
                for row in per_item_rows
                if row["side"] == side
            ]
        )

        if not rows:
            continue

        for metric in monotonicity_metrics:
            raw_key = f"raw_{metric}"
            beta_key = f"beta_{metric}"

            valid_rows = [
                row
                for row in rows
                if raw_key in row and beta_key in row
            ]
            if not valid_rows:
                continue

            raw = np.asarray(
                [safe_float(row[raw_key]) for row in valid_rows],
                dtype=np.float64,
            )
            beta = np.asarray(
                [safe_float(row[beta_key]) for row in valid_rows],
                dtype=np.float64,
            )

            finite = np.isfinite(raw) & np.isfinite(beta)
            raw = raw[finite]
            beta = beta[finite]

            if len(raw) == 0:
                continue

            improvements = np.asarray(
                [
                    metric_improvement(metric, r, b)
                    for r, b in zip(raw, beta)
                ],
                dtype=np.float64,
            )

            output.append(
                {
                    "side": side,
                    "metric": metric,
                    "n": len(raw),
                    "direction": (
                        "higher"
                        if metric in HIGHER_IS_BETTER
                        else "lower"
                    ),
                    "raw_mean": float(raw.mean()),
                    "raw_std": float(raw.std()),
                    "beta_mean": float(beta.mean()),
                    "beta_std": float(beta.std()),
                    "beta_minus_raw": float((beta - raw).mean()),
                    "mean_improvement": float(improvements.mean()),
                    "improvement_std": float(improvements.std()),
                    "improvement_median": float(np.median(improvements)),
                    "fraction_items_beta_better": float(
                        (improvements > 0).mean()
                    ),
                }
            )

    return output


def per_speaker_alignment_summary(
    per_item_rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Long-form per-speaker raw/Beta summary for every directional metric.
    """
    grouped = defaultdict(list)

    for row in per_item_rows:
        grouped[
            (
                str(row["speaker_id"]),
                str(row["side"]),
            )
        ].append(row)

    output = []

    for (speaker, side), rows in sorted(grouped.items()):
        for metric in available_paired_metrics(rows):
            if metric not in HIGHER_IS_BETTER and metric not in LOWER_IS_BETTER:
                continue

            raw = np.asarray(
                [
                    row.get(f"raw_{metric}", np.nan)
                    for row in rows
                ],
                dtype=np.float64,
            )
            beta = np.asarray(
                [
                    row.get(f"beta_{metric}", np.nan)
                    for row in rows
                ],
                dtype=np.float64,
            )

            finite = np.isfinite(raw) & np.isfinite(beta)
            raw = raw[finite]
            beta = beta[finite]

            if len(raw) == 0:
                continue

            if metric in HIGHER_IS_BETTER:
                improvement = beta - raw
                direction = "higher"
            else:
                improvement = raw - beta
                direction = "lower"

            output.append(
                {
                    "speaker_id": speaker,
                    "side": side,
                    "metric": metric,
                    "direction": direction,
                    "n_items": len(raw),
                    "raw_mean": float(raw.mean()),
                    "beta_mean": float(beta.mean()),
                    "beta_minus_raw": float((beta - raw).mean()),
                    "mean_improvement": float(improvement.mean()),
                    "median_improvement": float(np.median(improvement)),
                    "fraction_items_beta_better": float(
                        (improvement > 0).mean()
                    ),
                }
            )

    return output


# -----------------------------------------------------------------------------
# Discrete ABX-style analysis
# -----------------------------------------------------------------------------

def collapse_consecutive(
    values: Sequence[int],
) -> Tuple[int, ...]:
    if not values:
        return tuple()

    output = [int(values[0])]

    for value in values[1:]:
        value = int(value)
        if value != output[-1]:
            output.append(value)

    return tuple(output)


def extract_triphone_occurrences(
    records: List[Dict[str, Any]],
    exclude_silence: bool,
) -> List[Dict[str, Any]]:
    occurrences = []
    occurrence_id = 0

    for record in records:
        phones = record["phone_labels_39"]
        segments = runs(phones)

        for index in range(1, len(segments) - 1):
            left_start, left_end, left = segments[index - 1]
            center_start, center_end, center = segments[index]
            right_start, right_end, right = segments[index + 1]

            del left_end, center_start, center_end

            if (
                left in IGNORE_PHONES
                or center in IGNORE_PHONES
                or right in IGNORE_PHONES
            ):
                continue

            if exclude_silence and (
                left in SILENCE_39
                or center in SILENCE_39
                or right in SILENCE_39
            ):
                continue

            raw_sequence = collapse_consecutive(
                record["raw_token_ids"][
                    left_start:right_end
                ].tolist()
            )
            beta_sequence = collapse_consecutive(
                record["beta_token_ids"][
                    left_start:right_end
                ].tolist()
            )

            if not raw_sequence or not beta_sequence:
                continue

            occurrences.append(
                {
                    "id": occurrence_id,
                    "context": (left, right),
                    "center": center,
                    "speaker": record["speaker_id"],
                    "raw": raw_sequence,
                    "beta": beta_sequence,
                }
            )
            occurrence_id += 1

    return occurrences


def normalized_levenshtein(
    first: Tuple[int, ...],
    second: Tuple[int, ...],
) -> float:
    if first == second:
        return 0.0
    if not first:
        return 1.0 if second else 0.0
    if not second:
        return 1.0

    previous = list(range(len(second) + 1))

    for row_index, first_value in enumerate(first, start=1):
        current = [row_index]

        for column_index, second_value in enumerate(second, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column_index] + 1,
                    previous[column_index - 1]
                    + (first_value != second_value),
                )
            )

        previous = current

    return previous[-1] / max(len(first), len(second), 1)


def sampled_discrete_abx(
    occurrences: List[Dict[str, Any]],
    max_triplets: int,
    seed: int,
) -> List[Dict[str, Any]]:
    rng = random.Random(seed)

    grouped = defaultdict(
        lambda: defaultdict(
            lambda: defaultdict(list)
        )
    )
    by_id = {}

    for occurrence in occurrences:
        grouped[
            occurrence["context"]
        ][
            occurrence["center"]
        ][
            occurrence["speaker"]
        ].append(occurrence)

        by_id[occurrence["id"]] = occurrence

    pair_specs = []

    for context, categories in grouped.items():
        centers = sorted(categories)

        for first_index in range(len(centers)):
            for second_index in range(
                first_index + 1,
                len(centers),
            ):
                pair_specs.append(
                    (
                        context,
                        centers[first_index],
                        centers[second_index],
                    )
                )

    if not pair_specs:
        return []

    per_pair = max(
        1,
        max_triplets // max(len(pair_specs), 1),
    )

    triplets = {
        "within": [],
        "across": [],
    }

    for context, first_center, second_center in pair_specs:
        for target, contrast in (
            (first_center, second_center),
            (second_center, first_center),
        ):
            target_by_speaker = grouped[context][target]
            contrast_by_speaker = grouped[context][contrast]

            valid_within_speakers = [
                speaker
                for speaker in target_by_speaker
                if (
                    len(target_by_speaker[speaker]) >= 2
                    and len(
                        contrast_by_speaker.get(
                            speaker,
                            [],
                        )
                    ) >= 1
                )
            ]

            for _ in range(per_pair):
                if not valid_within_speakers:
                    break

                speaker = rng.choice(valid_within_speakers)
                A, X = rng.sample(
                    target_by_speaker[speaker],
                    2,
                )
                B = rng.choice(
                    contrast_by_speaker[speaker]
                )

                triplets["within"].append(
                    (
                        A["id"],
                        B["id"],
                        X["id"],
                        context,
                        target,
                        contrast,
                    )
                )

            base_speakers = [
                speaker
                for speaker in target_by_speaker
                if (
                    target_by_speaker[speaker]
                    and contrast_by_speaker.get(speaker)
                )
            ]

            valid_across_pairs = [
                (first_speaker, second_speaker)
                for first_speaker in base_speakers
                for second_speaker in target_by_speaker
                if (
                    second_speaker != first_speaker
                    and target_by_speaker[second_speaker]
                )
            ]

            for _ in range(per_pair):
                if not valid_across_pairs:
                    break

                first_speaker, second_speaker = rng.choice(
                    valid_across_pairs
                )

                A = rng.choice(
                    target_by_speaker[first_speaker]
                )
                B = rng.choice(
                    contrast_by_speaker[first_speaker]
                )
                X = rng.choice(
                    target_by_speaker[second_speaker]
                )

                triplets["across"].append(
                    (
                        A["id"],
                        B["id"],
                        X["id"],
                        context,
                        target,
                        contrast,
                    )
                )

    distance_cache: Dict[Tuple[str, int, int], float] = {}

    def distance(mode: str, first_id: int, second_id: int) -> float:
        key = (
            mode,
            min(first_id, second_id),
            max(first_id, second_id),
        )

        if key not in distance_cache:
            distance_cache[key] = normalized_levenshtein(
                by_id[first_id][mode],
                by_id[second_id][mode],
            )

        return distance_cache[key]

    output = []

    for condition, triples in triplets.items():
        if len(triples) > max_triplets:
            triples = rng.sample(
                triples,
                max_triplets,
            )

        raw_errors = []
        beta_errors = []
        pair_errors = defaultdict(
            lambda: {
                "raw": [],
                "beta": [],
            }
        )

        for A, B, X, context, target, contrast in triples:
            raw_ax = distance("raw", A, X)
            raw_bx = distance("raw", B, X)
            beta_ax = distance("beta", A, X)
            beta_bx = distance("beta", B, X)

            raw_error = (
                1.0
                if raw_ax > raw_bx
                else (
                    0.5
                    if raw_ax == raw_bx
                    else 0.0
                )
            )
            beta_error = (
                1.0
                if beta_ax > beta_bx
                else (
                    0.5
                    if beta_ax == beta_bx
                    else 0.0
                )
            )

            raw_errors.append(raw_error)
            beta_errors.append(beta_error)

            pair_key = (
                context,
                tuple(sorted((target, contrast))),
            )
            pair_errors[pair_key]["raw"].append(raw_error)
            pair_errors[pair_key]["beta"].append(beta_error)

        if not raw_errors:
            continue

        raw_macro = np.mean(
            [
                np.mean(values["raw"])
                for values in pair_errors.values()
            ]
        )
        beta_macro = np.mean(
            [
                np.mean(values["beta"])
                for values in pair_errors.values()
            ]
        )

        raw_errors_array = np.asarray(raw_errors)
        beta_errors_array = np.asarray(beta_errors)

        output.append(
            {
                "condition": condition,
                "num_occurrences": len(occurrences),
                "num_context_phone_pairs": len(pair_errors),
                "num_sampled_triplets": len(raw_errors),
                "raw_abx_error_macro": float(raw_macro),
                "beta_abx_error_macro": float(beta_macro),
                "abx_error_reduction_raw_minus_beta": float(
                    raw_macro - beta_macro
                ),
                "raw_abx_error_micro": float(raw_errors_array.mean()),
                "beta_abx_error_micro": float(beta_errors_array.mean()),
                "fraction_triplets_beta_better": float(
                    (beta_errors_array < raw_errors_array).mean()
                ),
                "fraction_triplets_beta_worse": float(
                    (beta_errors_array > raw_errors_array).mean()
                ),
            }
        )

    return output


# -----------------------------------------------------------------------------
# Per-item aligned record construction
# -----------------------------------------------------------------------------

def collect_aligned_records(
    items: List[Dict[str, Any]],
    output_dir: str,
    sides: Sequence[str],
    expected_frames: int,
    write_frame_csv: bool,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    aligned_dir = os.path.join(
        output_dir,
        "aligned_sequences",
    )
    os.makedirs(aligned_dir, exist_ok=True)

    per_item_rows = []
    records = []

    frame_csv_file = None
    frame_writer = None

    if write_frame_csv:
        frame_csv_file = gzip.open(
            os.path.join(
                output_dir,
                "frame_alignment.csv.gz",
            ),
            "wt",
            newline="",
        )

        frame_writer = csv.DictWriter(
            frame_csv_file,
            fieldnames=[
                "item_index",
                "file_id",
                "speaker_id",
                "side",
                "encoder_frame",
                "time_sec",
                "phone_61",
                "phone_39",
                "raw_token_pos",
                "raw_token_id",
                "raw_peak_token_pos",
                "raw_peak_token_id",
                "raw_expected_token_pos",
                "beta_token_pos",
                "beta_token_id",
                "beta_peak_token_pos",
                "beta_peak_token_id",
                "beta_expected_token_pos",
            ],
        )
        frame_writer.writeheader()

    try:
        for item in tqdm(
            items,
            desc="Aligning tokens and phones",
        ):
            meta = item.get("meta", {})
            speaker = str(
                meta.get(
                    "speaker_id",
                    "unknown",
                )
            )
            hop_samples = int(
                meta.get(
                    "hop_samples",
                    160,
                )
            )
            sample_rate = int(
                meta.get(
                    "sample_rate",
                    16000,
                )
            )
            hop_ms = 1000.0 * hop_samples / sample_rate

            for side in sides:
                if side not in item:
                    continue

                side_obj = item[side]

                if not side_obj.get(
                    "phone_alignment_valid",
                    False,
                ):
                    continue

                if (
                    "phone_labels_61" not in side_obj
                    or "phone_labels_39" not in side_obj
                ):
                    raise KeyError(
                        f"{item['_path']} has no phone labels. "
                        "Re-run the metadata-aware artifact generator."
                    )

                raw_positions, raw_ids = get_alignment(
                    side_obj,
                    "raw",
                )
                beta_positions, beta_ids = get_alignment(
                    side_obj,
                    "beta",
                )

                raw_frame2tok = get_frame_to_token_distribution(
                    side_obj,
                    "raw",
                )
                beta_frame2tok = get_frame_to_token_distribution(
                    side_obj,
                    "beta",
                )

                raw_peak_positions = peak_token_per_frame(
                    raw_frame2tok
                )
                beta_peak_positions = peak_token_per_frame(
                    beta_frame2tok
                )
                raw_expected_positions = expected_token_per_frame(
                    raw_frame2tok
                )
                beta_expected_positions = expected_token_per_frame(
                    beta_frame2tok
                )

                token_ids = as_long_vector(side_obj["tokens"])
                if token_ids.numel() == 0:
                    raw_peak_ids = torch.empty_like(raw_peak_positions)
                    beta_peak_ids = torch.empty_like(beta_peak_positions)
                else:
                    if (
                        raw_peak_positions.numel()
                        and int(raw_peak_positions.max()) >= token_ids.numel()
                    ):
                        raise ValueError(
                            "Raw peak-token position exceeds token inventory."
                        )
                    if (
                        beta_peak_positions.numel()
                        and int(beta_peak_positions.max()) >= token_ids.numel()
                    ):
                        raise ValueError(
                            "Beta peak-token position exceeds token inventory."
                        )
                    raw_peak_ids = token_ids[raw_peak_positions]
                    beta_peak_ids = token_ids[beta_peak_positions]

                phones_61 = list(
                    side_obj["phone_labels_61"]
                )
                phones_39 = list(
                    side_obj["phone_labels_39"]
                )

                lengths = [
                    raw_positions.numel(),
                    beta_positions.numel(),
                    raw_ids.numel(),
                    beta_ids.numel(),
                    raw_peak_positions.numel(),
                    beta_peak_positions.numel(),
                    raw_expected_positions.numel(),
                    beta_expected_positions.numel(),
                    len(phones_61),
                    len(phones_39),
                ]

                if len(set(lengths)) != 1:
                    raise ValueError(
                        f"Length mismatch in {item['_path']} "
                        f"side={side}: {lengths}"
                    )

                if (
                    expected_frames > 0
                    and lengths[0] != expected_frames
                ):
                    raise ValueError(
                        f"Expected {expected_frames} aligned frames, "
                        f"got {lengths[0]} in {item['_path']} side={side}"
                    )

                record = {
                    "item_index": int(
                        item.get(
                            "item_index",
                            -1,
                        )
                    ),
                    "file_id": str(
                        item.get(
                            "file_id",
                            "",
                        )
                    ),
                    "speaker_id": speaker,
                    "side": side,
                    "raw_token_pos": raw_positions,
                    "beta_token_pos": beta_positions,
                    "raw_token_ids": raw_ids,
                    "beta_token_ids": beta_ids,
                    "raw_peak_token_pos": raw_peak_positions,
                    "beta_peak_token_pos": beta_peak_positions,
                    "raw_peak_token_ids": raw_peak_ids,
                    "beta_peak_token_ids": beta_peak_ids,
                    "raw_expected_token_pos": raw_expected_positions,
                    "beta_expected_token_pos": beta_expected_positions,
                    "phone_labels_61": phones_61,
                    "phone_labels_39": phones_39,
                    "hop_ms": hop_ms,
                    "artifact_path": item["_path"],
                }
                records.append(record)

                save_name = (
                    f"item_{record['item_index']:06d}_{side}.pt"
                    if record["item_index"] >= 0
                    else f"{Path(item['_path']).stem}_{side}.pt"
                )

                torch.save(
                    record,
                    os.path.join(
                        aligned_dir,
                        save_name,
                    ),
                )

                row = {
                    "path": item["_path"],
                    "item_index": record["item_index"],
                    "file_id": record["file_id"],
                    "speaker_id": speaker,
                    "side": side,
                    "num_frames": lengths[0],
                    "num_decoder_tokens": int(
                        as_long_vector(
                            side_obj["tokens"]
                        ).numel()
                    ),
                }

                for mode, positions in (
                    ("raw", raw_positions),
                    ("beta", beta_positions),
                ):
                    boundary = boundary_metrics(
                        positions,
                        phones_61,
                        hop_ms,
                    )
                    segment = segment_alignment_metrics(
                        positions,
                        phones_39,
                    )

                    for key, value in {
                        **boundary,
                        **segment,
                    }.items():
                        row[f"{mode}_{key}"] = value

                    original_metrics = side_obj.get(
                        f"{mode}_metrics",
                        {},
                    )
                    for key, value in original_metrics.items():
                        try:
                            row[f"{mode}_{key}"] = safe_float(value)
                        except (TypeError, ValueError):
                            continue

                per_item_rows.append(row)

                if frame_writer is not None:
                    for frame in range(lengths[0]):
                        frame_writer.writerow(
                            {
                                "item_index": record["item_index"],
                                "file_id": record["file_id"],
                                "speaker_id": speaker,
                                "side": side,
                                "encoder_frame": frame,
                                "time_sec": (
                                    frame
                                    * hop_samples
                                    / sample_rate
                                ),
                                "phone_61": phones_61[frame],
                                "phone_39": phones_39[frame],
                                "raw_token_pos": int(
                                    raw_positions[frame]
                                ),
                                "raw_token_id": int(
                                    raw_ids[frame]
                                ),
                                "raw_peak_token_pos": int(
                                    raw_peak_positions[frame]
                                ),
                                "raw_peak_token_id": int(
                                    raw_peak_ids[frame]
                                ),
                                "raw_expected_token_pos": float(
                                    raw_expected_positions[frame]
                                ),
                                "beta_token_pos": int(
                                    beta_positions[frame]
                                ),
                                "beta_token_id": int(
                                    beta_ids[frame]
                                ),
                                "beta_peak_token_pos": int(
                                    beta_peak_positions[frame]
                                ),
                                "beta_peak_token_id": int(
                                    beta_peak_ids[frame]
                                ),
                                "beta_expected_token_pos": float(
                                    beta_expected_positions[frame]
                                ),
                            }
                        )
    finally:
        if frame_csv_file is not None:
            frame_csv_file.close()

    return records, per_item_rows


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_dir",
        default="cross_attention_phone_output_TIMIT",
    )
    parser.add_argument(
        "--output_dir",
        default="cross_attention_phone_analysis_TIMIT",
    )
    parser.add_argument(
        "--sides",
        default="anc",
        help="anc, pos, or anc,pos",
    )
    parser.add_argument(
        "--expected_frames",
        type=int,
        default=301,
        help="Set <= 0 to disable the frame-count assertion.",
    )
    parser.add_argument(
        "--write_frame_csv",
        action="store_true",
    )

    parser.add_argument(
        "--bootstrap_samples",
        type=int,
        default=2000,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=13,
    )

    parser.add_argument(
        "--skip_abx",
        action="store_true",
    )
    parser.add_argument(
        "--abx_max_triplets",
        type=int,
        default=100000,
    )
    parser.add_argument(
        "--abx_include_silence",
        action="store_true",
    )

    args = parser.parse_args()

    if args.bootstrap_samples < 1:
        raise ValueError("--bootstrap_samples must be >= 1")
    if args.abx_max_triplets < 1:
        raise ValueError("--abx_max_triplets must be >= 1")
    os.makedirs(
        args.output_dir,
        exist_ok=True,
    )

    sides = [
        side.strip()
        for side in args.sides.split(",")
        if side.strip()
    ]

    if not sides:
        raise ValueError("--sides must contain anc and/or pos")

    for side in sides:
        if side not in {"anc", "pos"}:
            raise ValueError(f"Invalid side: {side}")

    items = load_items(args.input_dir)

    records, per_item = collect_aligned_records(
        items=items,
        output_dir=args.output_dir,
        sides=sides,
        expected_frames=args.expected_frames,
        write_frame_csv=args.write_frame_csv,
    )

    if not records:
        raise RuntimeError(
            "No valid phone-aligned records. For positive examples, "
            "pos_time_preserving=True should be set only for transforms that "
            "preserve the original time axis."
        )

    # ------------------------------------------------------------------
    # Per-item metrics
    # ------------------------------------------------------------------
    per_item_path = os.path.join(
        args.output_dir,
        "per_item_alignment_metrics.csv",
    )
    write_csv(
        per_item_path,
        per_item,
    )

    # ------------------------------------------------------------------
    # Global token/phone/speaker metrics
    # ------------------------------------------------------------------
    global_rows = []

    for mode in ("raw", "beta"):
        metrics, mapping = hubert_style_metrics(
            records,
            mode,
        )
        global_rows.append(metrics)

        write_csv(
            os.path.join(
                args.output_dir,
                f"token_phone_mapping_{mode}.csv",
            ),
            mapping,
        )

    global_metrics_path = os.path.join(
        args.output_dir,
        "global_hubert_metrics.csv",
    )
    write_csv(
        global_metrics_path,
        global_rows,
    )

    # ------------------------------------------------------------------
    # Paired bootstrap comparison
    # ------------------------------------------------------------------
    paired = paired_improvement_summary(
        per_item,
        n_boot=args.bootstrap_samples,
        seed=args.seed,
    )
    paired_path = os.path.join(
        args.output_dir,
        "paired_beta_improvement.csv",
    )
    write_csv(
        paired_path,
        paired,
    )

    # ------------------------------------------------------------------
    # Full-dataset monotonicity and per-speaker summaries
    # ------------------------------------------------------------------
    monotonicity_rows = monotonicity_summary(
        per_item
    )
    monotonicity_path = os.path.join(
        args.output_dir,
        "monotonicity_summary.csv",
    )
    write_csv(
        monotonicity_path,
        monotonicity_rows,
    )

    speaker_rows = per_speaker_alignment_summary(
        per_item
    )
    speaker_path = os.path.join(
        args.output_dir,
        "per_speaker_alignment_summary.csv",
    )
    write_csv(
        speaker_path,
        speaker_rows,
    )

    # ------------------------------------------------------------------
    # ABX-style discrete analysis
    # ------------------------------------------------------------------
    abx_path = os.path.join(
        args.output_dir,
        "abx_discrete_metrics.csv",
    )

    if not args.skip_abx:
        occurrences = extract_triphone_occurrences(
            records,
            exclude_silence=(
                not args.abx_include_silence
            ),
        )

        abx_rows = sampled_discrete_abx(
            occurrences,
            max_triplets=args.abx_max_triplets,
            seed=args.seed,
        )
        write_csv(
            abx_path,
            abx_rows,
        )

    # ------------------------------------------------------------------
    # Console summary
    # ------------------------------------------------------------------
    print(f"Artifacts loaded: {len(items)}")
    print(f"Aligned side records: {len(records)}")
    print(f"Per-item metrics: {per_item_path}")
    print(f"HuBERT-style metrics: {global_metrics_path}")
    print(f"Paired Beta comparison: {paired_path}")
    print(f"Monotonicity summary: {monotonicity_path}")
    print(f"Per-speaker summary: {speaker_path}")
    print(
        "Aligned sequences: "
        f"{os.path.join(args.output_dir, 'aligned_sequences')}"
    )

    if args.write_frame_csv:
        print(
            "Frame alignment CSV: "
            f"{os.path.join(args.output_dir, 'frame_alignment.csv.gz')}"
        )

    if not args.skip_abx:
        print(f"ABX-style metrics: {abx_path}")


if __name__ == "__main__":
    main()
