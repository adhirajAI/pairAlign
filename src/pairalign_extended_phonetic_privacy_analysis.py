#!/usr/bin/env python3
"""
Extended TIMIT analysis for raw versus Beta-aligned PairAlign tokens.

Preserves the original boundary, HuBERT-style phone-association, bootstrap,
and discrete ABX analyses, and adds:

  * ngram_phone_metrics.csv
      token unigram/bigram/trigram -> central-phone association, PNMI,
      purity and leave-one-speaker-out majority prediction.

  * ngram_phone_sequence_metrics.csv
      token n-gram -> aligned phone n-gram association for n=1,2,3.

  * token_transition_metrics.csv
      H(Z), H(Z_t|Z_{t-1}), H(Z_t|Z_{t-2},Z_{t-1}), adjacent-token MI,
      active n-gram counts and conditional perplexities.

  * speaker_ngram_probe.csv
      lightweight closed-set speaker identification from per-segment
      bag-of-token unigram/bigram/trigram features using stratified CV.

  * speaker_verification_metrics.csv
      lightweight same/different-speaker verification using cosine similarity
      between sparse symbolic n-gram count vectors, with ROC-AUC and EER.

  * privacy_summary.csv
      compact summary combining unigram NMI/conditional NMI and attacker-style
      speaker probes.

Important: lower speaker-ID accuracy, lower verification AUC, or higher EER
means less recoverable speaker information under THESE probes only. It is not
a privacy guarantee and compactness must not be interpreted as anonymity.

Dependencies: numpy, torch, tqdm, scikit-learn
"""

import os
import csv
import json
import math
import random
import argparse
from glob import glob
from pathlib import Path
from collections import Counter, defaultdict
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
from tqdm.auto import tqdm

try:
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        f1_score,
        roc_auc_score,
        roc_curve,
    )
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import LabelEncoder, normalize
except ImportError as exc:
    raise ImportError(
        "This script requires scikit-learn. Install with: pip install scikit-learn"
    ) from exc


SILENCE_39 = {"sil"}
IGNORE_PHONES = {"<ignore>", "<outside>", "<pad>", ""}

HIGHER_IS_BETTER = {
    "boundary_f1_10ms", "boundary_f1_20ms", "boundary_f1_30ms",
    "boundary_precision_10ms", "boundary_precision_20ms", "boundary_precision_30ms",
    "boundary_recall_10ms", "boundary_recall_20ms", "boundary_recall_30ms",
    "token_segment_phone_purity", "phone_segment_token_purity",
    "expected_frame_corr", "viterbi_log_score_mean",
}

LOWER_IS_BETTER = {
    "boundary_mae_frames_20ms", "boundary_mae_ms_20ms",
    "oversegmentation_ratio_abs_log",
    "mean_phones_per_token_segment", "mean_tokens_per_phone_segment",
    "expected_frame_violation_rate", "expected_frame_violation_mag",
    "peak_frame_violation_rate", "peak_frame_violation_mag",
    "mean_abs_diagonal_error", "mean_peak_diagonal_error",
    "frame_to_token_expected_violation_rate",
    "frame_to_token_expected_violation_mag",
    "frame_to_token_peak_violation_rate",
    "frame_to_token_peak_violation_mag",
}


def write_csv(path: str, rows: List[Dict[str, Any]]):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if not rows:
        return
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
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


def viterbi_monotonic_path(P_frame2tok: torch.Tensor, eps: float = 1e-12) -> torch.LongTensor:
    P = P_frame2tok.detach().cpu().float()
    T, M = P.shape
    if T == 0 or M == 0:
        return torch.empty(0, dtype=torch.long)
    logP = P.clamp_min(eps).log()
    dp = torch.full((T, M), float("-inf"), dtype=logP.dtype)
    bp = torch.full((T, M), -1, dtype=torch.long)
    dp[0, 0] = logP[0, 0]
    bp[0, 0] = 0
    for f in range(1, T):
        dp[f, 0] = dp[f - 1, 0] + logP[f, 0]
        bp[f, 0] = 0
        for t in range(1, M):
            stay, advance = dp[f - 1, t], dp[f - 1, t - 1]
            if advance > stay:
                dp[f, t] = advance + logP[f, t]
                bp[f, t] = t - 1
            else:
                dp[f, t] = stay + logP[f, t]
                bp[f, t] = t
    path = torch.empty(T, dtype=torch.long)
    t = M - 1
    path[-1] = t
    for f in range(T - 1, 0, -1):
        t = int(bp[f, t])
        path[f - 1] = t
    return path


def get_alignment(side_obj: Dict[str, Any], mode: str) -> Tuple[torch.LongTensor, torch.LongTensor]:
    token_ids = as_long_vector(side_obj["tokens"])
    path_key = f"{mode}_viterbi_path"
    frame_token_key = f"{mode}_frame_token_ids"

    if path_key in side_obj:
        path = as_long_vector(side_obj[path_key])
    elif f"{mode}_frame2tok" in side_obj:
        path = viterbi_monotonic_path(as_float_matrix(side_obj[f"{mode}_frame2tok"]))
    else:
        A = as_float_matrix(side_obj[f"{mode}_attn"]).clamp_min(0)
        A = A / (A.sum(-1, keepdim=True) + 1e-12)
        P = A.T.contiguous()
        P = P / (P.sum(-1, keepdim=True) + 1e-12)
        path = viterbi_monotonic_path(P)

    if frame_token_key in side_obj:
        frame_token_ids = as_long_vector(side_obj[frame_token_key])
    elif token_ids.numel() == 0:
        frame_token_ids = torch.empty_like(path)
    else:
        frame_token_ids = token_ids[path.clamp(0, token_ids.numel() - 1)]
    return path, frame_token_ids


def boundaries_from_labels(labels: Sequence[Any], ignored: set = IGNORE_PHONES) -> np.ndarray:
    out = []
    for i in range(1, len(labels)):
        a, b = str(labels[i - 1]), str(labels[i])
        if a in ignored or b in ignored:
            continue
        if a != b:
            out.append(i)
    return np.asarray(out, dtype=np.int64)


def boundaries_from_token_positions(pos: Sequence[int]) -> np.ndarray:
    x = np.asarray(pos, dtype=np.int64)
    if x.size < 2:
        return np.empty(0, dtype=np.int64)
    return np.flatnonzero(x[1:] != x[:-1]) + 1


def one_to_one_boundary_match(pred: np.ndarray, ref: np.ndarray, tolerance_frames: int):
    pred = np.asarray(pred, dtype=np.int64)
    ref = np.asarray(ref, dtype=np.int64)
    candidates = []
    for i, p in enumerate(pred):
        left = np.searchsorted(ref, p - tolerance_frames, side="left")
        right = np.searchsorted(ref, p + tolerance_frames, side="right")
        for j in range(left, right):
            candidates.append((abs(int(p) - int(ref[j])), i, j))
    candidates.sort()
    used_pred, used_ref, distances = set(), set(), []
    for dist, i, j in candidates:
        if i not in used_pred and j not in used_ref:
            used_pred.add(i)
            used_ref.add(j)
            distances.append(dist)
    tp = len(distances)
    return tp, len(pred) - tp, len(ref) - tp, distances


def boundary_metrics(token_positions: torch.LongTensor, phone_labels_61: Sequence[str], hop_ms: float):
    pred = boundaries_from_token_positions(token_positions.numpy())
    ref = boundaries_from_labels(phone_labels_61)
    out = {
        "num_token_boundaries": float(len(pred)),
        "num_phone_boundaries": float(len(ref)),
    }
    for tol in [1, 2, 3]:
        tp, fp, fn, distances = one_to_one_boundary_match(pred, ref, tol)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        ms = int(round(tol * hop_ms))
        out[f"boundary_precision_{ms}ms"] = precision
        out[f"boundary_recall_{ms}ms"] = recall
        out[f"boundary_f1_{ms}ms"] = f1
        if tol == 2:
            out["boundary_mae_frames_20ms"] = float(np.mean(distances)) if distances else float("nan")
            out["boundary_mae_ms_20ms"] = float(np.mean(distances) * hop_ms) if distances else float("nan")
    ratio = len(pred) / max(len(ref), 1)
    out["token_to_phone_boundary_count_ratio"] = ratio
    out["oversegmentation_ratio_abs_log"] = abs(math.log(max(ratio, 1e-12)))
    return out


def runs(values: Sequence[Any]):
    if len(values) == 0:
        return []
    out, start, prev = [], 0, values[0]
    for i in range(1, len(values)):
        if values[i] != prev:
            out.append((start, i, prev))
            start, prev = i, values[i]
    out.append((start, len(values), prev))
    return out


def segment_alignment_metrics(token_positions: torch.LongTensor, phone_labels_39: Sequence[str]):
    pos = token_positions.numpy().tolist()
    n = min(len(pos), len(phone_labels_39))
    pos, phones = pos[:n], list(phone_labels_39[:n])
    token_runs = runs(pos)
    phone_runs = [(s, e, p) for s, e, p in runs(phones) if p not in IGNORE_PHONES]

    token_majority_correct = 0
    token_valid_frames = 0
    phones_per_token = []
    for s, e, _ in token_runs:
        valid = [p for p in phones[s:e] if p not in IGNORE_PHONES]
        if not valid:
            continue
        c = Counter(valid)
        token_majority_correct += max(c.values())
        token_valid_frames += len(valid)
        phones_per_token.append(len(c))

    phone_majority_correct = 0
    phone_valid_frames = 0
    tokens_per_phone = []
    for s, e, _ in phone_runs:
        vals = pos[s:e]
        if not vals:
            continue
        c = Counter(vals)
        phone_majority_correct += max(c.values())
        phone_valid_frames += len(vals)
        tokens_per_phone.append(len(c))

    return {
        "token_segment_phone_purity": token_majority_correct / max(token_valid_frames, 1),
        "phone_segment_token_purity": phone_majority_correct / max(phone_valid_frames, 1),
        "mean_phones_per_token_segment": float(np.mean(phones_per_token)) if phones_per_token else float("nan"),
        "mean_tokens_per_phone_segment": float(np.mean(tokens_per_phone)) if tokens_per_phone else float("nan"),
    }


def entropy_from_counter(counter: Counter) -> float:
    total = sum(counter.values())
    if total <= 0:
        return 0.0
    probs = np.asarray([v / total for v in counter.values() if v > 0], dtype=np.float64)
    return float(-(probs * np.log(probs)).sum())


def mutual_information_from_joint(joint: Counter) -> float:
    total = sum(joint.values())
    if total <= 0:
        return 0.0
    a_counts, b_counts = Counter(), Counter()
    for (a, b), n in joint.items():
        a_counts[a] += n
        b_counts[b] += n
    mi = 0.0
    for (a, b), n in joint.items():
        p_ab = n / total
        p_a = a_counts[a] / total
        p_b = b_counts[b] / total
        mi += p_ab * math.log(p_ab / max(p_a * p_b, 1e-300))
    return float(mi)


def conditional_mi_token_speaker_given_phone(triple: Counter):
    total = sum(triple.values())
    if total <= 0:
        return 0.0, 0.0
    phone_counts = Counter()
    p_ts = defaultdict(Counter)
    p_s = defaultdict(Counter)
    for (phone, token, speaker), n in triple.items():
        phone_counts[phone] += n
        p_ts[phone][(token, speaker)] += n
        p_s[phone][speaker] += n
    cmi, h_s_given_p = 0.0, 0.0
    for phone, n_phone in phone_counts.items():
        weight = n_phone / total
        cmi += weight * mutual_information_from_joint(p_ts[phone])
        h_s_given_p += weight * entropy_from_counter(p_s[phone])
    return cmi, h_s_given_p


def hubert_style_metrics(records: List[Dict[str, Any]], mode: str):
    joint = Counter()
    token_counts = Counter()
    phone_counts = Counter()
    speaker_token_joint = Counter()
    triple = Counter()
    speaker_joint = defaultdict(Counter)

    for r in records:
        speaker = r["speaker_id"]
        tokens = r[f"{mode}_token_ids"]
        phones = r["phone_labels_39"]
        for z, y in zip(tokens.tolist(), phones):
            if y in IGNORE_PHONES:
                continue
            joint[(y, int(z))] += 1
            token_counts[int(z)] += 1
            phone_counts[y] += 1
            speaker_token_joint[(speaker, int(z))] += 1
            triple[(y, int(z), speaker)] += 1
            speaker_joint[speaker][(y, int(z))] += 1

    N = sum(joint.values())
    if N == 0:
        raise ValueError("No valid phone-token frames available.")

    phone_purity = sum(max(joint[(y, z)] for y in phone_counts) for z in token_counts) / N
    cluster_purity = sum(max(joint[(y, z)] for z in token_counts) for y in phone_counts) / N
    mi = mutual_information_from_joint(joint)
    h_phone = entropy_from_counter(phone_counts)
    pnmi = mi / max(h_phone, 1e-12)
    h_token = entropy_from_counter(token_counts)
    code_perplexity = math.exp(h_token)
    active_vocab = len(token_counts)

    speaker_counts = Counter()
    for (s, _), n in speaker_token_joint.items():
        speaker_counts[s] += n
    speaker_mi = mutual_information_from_joint(speaker_token_joint)
    h_speaker = entropy_from_counter(speaker_counts)
    speaker_nmi = speaker_mi / max(h_speaker, 1e-12)
    cmi, h_s_given_phone = conditional_mi_token_speaker_given_phone(triple)
    conditional_speaker_nmi = cmi / max(h_s_given_phone, 1e-12)

    silence_joint = Counter()
    for (y, z), n in joint.items():
        silence_joint[("silence" if y in SILENCE_39 else "speech", z)] += n
    silence_purity = sum(
        max(silence_joint[(label, z)] for label in ["silence", "speech"])
        for z in token_counts
    ) / N

    loso_correct, loso_total = 0, 0
    global_phone_majority = phone_counts.most_common(1)[0][0]
    for heldout in sorted(speaker_joint):
        heldout_joint = speaker_joint[heldout]
        train_joint = joint.copy()
        for key, n in heldout_joint.items():
            train_joint[key] -= n
            if train_joint[key] <= 0:
                del train_joint[key]
        train_by_token = defaultdict(Counter)
        train_phone_counts = Counter()
        for (y, z), n in train_joint.items():
            train_by_token[z][y] += n
            train_phone_counts[y] += n
        fallback = train_phone_counts.most_common(1)[0][0] if train_phone_counts else global_phone_majority
        mapping = {z: counts.most_common(1)[0][0] for z, counts in train_by_token.items()}
        for (y, z), n in heldout_joint.items():
            pred = mapping.get(z, fallback)
            loso_correct += n if pred == y else 0
            loso_total += n

    mapping_rows = []
    by_token = defaultdict(Counter)
    for (y, z), n in joint.items():
        by_token[z][y] += n
    for z in sorted(by_token):
        counts = by_token[z]
        total = sum(counts.values())
        dominant, count = counts.most_common(1)[0]
        mapping_rows.append({
            "mode": mode,
            "token_id": z,
            "frames": total,
            "dominant_phone": dominant,
            "token_phone_purity": count / total,
            "num_phone_classes": len(counts),
            "phone_distribution": json.dumps(dict(counts), sort_keys=True),
        })

    metrics = {
        "mode": mode,
        "num_valid_frames": N,
        "active_vocab": active_vocab,
        "codebook_perplexity": code_perplexity,
        "normalized_codebook_perplexity": code_perplexity / max(active_vocab, 1),
        "phone_purity": phone_purity,
        "cluster_purity": cluster_purity,
        "pnmi": pnmi,
        "silence_binary_purity": silence_purity,
        "speaker_nmi": speaker_nmi,
        "conditional_speaker_nmi_given_phone": conditional_speaker_nmi,
        "loso_token_to_phone_accuracy": loso_correct / max(loso_total, 1),
    }
    return metrics, mapping_rows


def bootstrap_mean_delta(raw, beta, higher_is_better, n_boot, seed):
    mask = np.isfinite(raw) & np.isfinite(beta)
    raw, beta = raw[mask], beta[mask]
    if len(raw) == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    improvement = beta - raw if higher_is_better else raw - beta
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(improvement), size=(n_boot, len(improvement)))
    boot = improvement[idx].mean(axis=1)
    return (
        float(improvement.mean()),
        float(np.quantile(boot, 0.025)),
        float(np.quantile(boot, 0.975)),
        float((improvement > 0).mean()),
    )


def paired_improvement_summary(per_item_rows, n_boot, seed):
    metric_names = sorted({
        k[len("raw_"):]
        for r in per_item_rows for k in r
        if k.startswith("raw_") and f"beta_{k[len('raw_') :]}" in r
    })
    rows = []
    for metric in metric_names:
        raw = np.asarray([r.get(f"raw_{metric}", np.nan) for r in per_item_rows], dtype=float)
        beta = np.asarray([r.get(f"beta_{metric}", np.nan) for r in per_item_rows], dtype=float)
        if metric in HIGHER_IS_BETTER:
            higher = True
        elif metric in LOWER_IS_BETTER:
            higher = False
        else:
            continue
        imp, lo, hi, frac = bootstrap_mean_delta(raw, beta, higher, n_boot, seed)
        rows.append({
            "metric": metric,
            "direction": "higher" if higher else "lower",
            "n": int(np.sum(np.isfinite(raw) & np.isfinite(beta))),
            "raw_mean": float(np.nanmean(raw)),
            "beta_mean": float(np.nanmean(beta)),
            "beta_minus_raw": float(np.nanmean(beta - raw)),
            "mean_improvement": imp,
            "improvement_ci95_low": lo,
            "improvement_ci95_high": hi,
            "fraction_items_beta_better": frac,
        })
    return rows


def collapse_consecutive(values: Sequence[int]) -> Tuple[int, ...]:
    if not values:
        return tuple()
    out = [int(values[0])]
    for x in values[1:]:
        if int(x) != out[-1]:
            out.append(int(x))
    return tuple(out)


def extract_triphone_occurrences(records, exclude_silence):
    occurrences = []
    occ_id = 0
    for r in records:
        phones = r["phone_labels_39"]
        segs = runs(phones)
        for i in range(1, len(segs) - 1):
            ls, le, left = segs[i - 1]
            cs, ce, center = segs[i]
            rs, re, right = segs[i + 1]
            if left in IGNORE_PHONES or center in IGNORE_PHONES or right in IGNORE_PHONES:
                continue
            if exclude_silence and (left in SILENCE_39 or center in SILENCE_39 or right in SILENCE_39):
                continue
            raw_seq = collapse_consecutive(r["raw_token_ids"][ls:re].tolist())
            beta_seq = collapse_consecutive(r["beta_token_ids"][ls:re].tolist())
            if not raw_seq or not beta_seq:
                continue
            occurrences.append({
                "id": occ_id,
                "context": (left, right),
                "center": center,
                "speaker": r["speaker_id"],
                "raw": raw_seq,
                "beta": beta_seq,
            })
            occ_id += 1
    return occurrences


def normalized_levenshtein(a, b):
    if a == b:
        return 0.0
    if not a:
        return 1.0 if b else 0.0
    if not b:
        return 1.0
    prev = list(range(len(b) + 1))
    for i, xa in enumerate(a, 1):
        cur = [i]
        for j, xb in enumerate(b, 1):
            cur.append(min(cur[-1] + 1, prev[j] + 1, prev[j - 1] + (xa != xb)))
        prev = cur
    return prev[-1] / max(len(a), len(b), 1)


def sampled_discrete_abx(occurrences, max_triplets, seed):
    rng = random.Random(seed)
    grouped = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    by_id = {}
    for o in occurrences:
        grouped[o["context"]][o["center"]][o["speaker"]].append(o)
        by_id[o["id"]] = o

    pair_specs = []
    for context, cats in grouped.items():
        centers = sorted(cats)
        for i in range(len(centers)):
            for j in range(i + 1, len(centers)):
                pair_specs.append((context, centers[i], centers[j]))
    if not pair_specs:
        return []

    per_pair = max(1, max_triplets // max(len(pair_specs), 1))
    triplets = {"within": [], "across": []}
    for context, c1, c2 in pair_specs:
        for target, contrast in [(c1, c2), (c2, c1)]:
            target_spk = grouped[context][target]
            contrast_spk = grouped[context][contrast]
            valid_within = [
                s for s in target_spk
                if len(target_spk[s]) >= 2 and len(contrast_spk.get(s, [])) >= 1
            ]
            for _ in range(per_pair):
                if not valid_within:
                    break
                s = rng.choice(valid_within)
                A, X = rng.sample(target_spk[s], 2)
                B = rng.choice(contrast_spk[s])
                triplets["within"].append((A["id"], B["id"], X["id"], context, target, contrast))

            base_speakers = [s for s in target_spk if target_spk[s] and contrast_spk.get(s)]
            valid_across_pairs = [
                (s1, s2) for s1 in base_speakers for s2 in target_spk
                if s2 != s1 and target_spk[s2]
            ]
            for _ in range(per_pair):
                if not valid_across_pairs:
                    break
                s1, s2 = rng.choice(valid_across_pairs)
                A = rng.choice(target_spk[s1])
                B = rng.choice(contrast_spk[s1])
                X = rng.choice(target_spk[s2])
                triplets["across"].append((A["id"], B["id"], X["id"], context, target, contrast))

    distance_cache = {}
    def dist(mode, i, j):
        key = (mode, min(i, j), max(i, j))
        if key not in distance_cache:
            distance_cache[key] = normalized_levenshtein(by_id[i][mode], by_id[j][mode])
        return distance_cache[key]

    output = []
    for condition, triples in triplets.items():
        if len(triples) > max_triplets:
            triples = rng.sample(triples, max_triplets)
        raw_errors, beta_errors = [], []
        per_pair_errors = defaultdict(lambda: {"raw": [], "beta": []})
        for A, B, X, context, target, contrast in triples:
            raw_ax, raw_bx = dist("raw", A, X), dist("raw", B, X)
            beta_ax, beta_bx = dist("beta", A, X), dist("beta", B, X)
            raw_e = 1.0 if raw_ax > raw_bx else (0.5 if raw_ax == raw_bx else 0.0)
            beta_e = 1.0 if beta_ax > beta_bx else (0.5 if beta_ax == beta_bx else 0.0)
            raw_errors.append(raw_e)
            beta_errors.append(beta_e)
            pair_key = (context, tuple(sorted((target, contrast))))
            per_pair_errors[pair_key]["raw"].append(raw_e)
            per_pair_errors[pair_key]["beta"].append(beta_e)
        if not raw_errors:
            continue
        raw_macro = np.mean([np.mean(v["raw"]) for v in per_pair_errors.values()])
        beta_macro = np.mean([np.mean(v["beta"]) for v in per_pair_errors.values()])
        output.append({
            "condition": condition,
            "num_occurrences": len(occurrences),
            "num_context_phone_pairs": len(per_pair_errors),
            "num_sampled_triplets": len(raw_errors),
            "raw_abx_error_macro": float(raw_macro),
            "beta_abx_error_macro": float(beta_macro),
            "abx_error_reduction_raw_minus_beta": float(raw_macro - beta_macro),
            "raw_abx_error_micro": float(np.mean(raw_errors)),
            "beta_abx_error_micro": float(np.mean(beta_errors)),
            "fraction_triplets_beta_better": float(np.mean(np.asarray(beta_errors) < np.asarray(raw_errors))),
            "fraction_triplets_beta_worse": float(np.mean(np.asarray(beta_errors) > np.asarray(raw_errors))),
        })
    return output


# ------------------------ New n-gram phone analyses ------------------------

def sliding_ngrams(values, n):
    vals = list(values)
    if n <= 0 or len(vals) < n:
        return []
    return [tuple(vals[i:i+n]) for i in range(len(vals) - n + 1)]


def _purity_x_to_y(joint):
    total = sum(joint.values())
    if total <= 0:
        return float("nan")
    by_x = defaultdict(Counter)
    for (y, x), c in joint.items():
        by_x[x][y] += c
    return float(sum(max(v.values()) for v in by_x.values()) / total)


def _purity_y_to_x(joint):
    total = sum(joint.values())
    if total <= 0:
        return float("nan")
    by_y = defaultdict(Counter)
    for (y, x), c in joint.items():
        by_y[y][x] += c
    return float(sum(max(v.values()) for v in by_y.values()) / total)


def loso_majority_accuracy(by_speaker, global_joint):
    y_counts = Counter()
    for (y, _), n in global_joint.items():
        y_counts[y] += n
    if not y_counts:
        return float("nan")
    global_y = y_counts.most_common(1)[0][0]
    correct, total = 0, 0
    for heldout, heldout_joint in by_speaker.items():
        train_joint = global_joint.copy()
        for key, n in heldout_joint.items():
            train_joint[key] -= n
            if train_joint[key] <= 0:
                del train_joint[key]
        by_x = defaultdict(Counter)
        train_y = Counter()
        for (y, x), n in train_joint.items():
            by_x[x][y] += n
            train_y[y] += n
        fallback = train_y.most_common(1)[0][0] if train_y else global_y
        mapping = {x: c.most_common(1)[0][0] for x, c in by_x.items()}
        for (y, x), n in heldout_joint.items():
            if mapping.get(x, fallback) == y:
                correct += n
            total += n
    return correct / max(total, 1)


def ngram_phone_metrics(records, mode, max_n=3):
    """Frame-aligned token n-gram -> central 39-phone association."""
    rows = []
    for n in range(1, max_n + 1):
        joint, x_counts, y_counts = Counter(), Counter(), Counter()
        by_speaker = defaultdict(Counter)
        for r in records:
            toks = r[f"{mode}_token_ids"].tolist()
            phones = list(r["phone_labels_39"])
            m = min(len(toks), len(phones))
            toks, phones = toks[:m], phones[:m]
            for i in range(m - n + 1):
                center = i + n // 2   # right-center for even n
                y = phones[center]
                if y in IGNORE_PHONES:
                    continue
                x = tuple(int(v) for v in toks[i:i+n])
                joint[(y, x)] += 1
                x_counts[x] += 1
                y_counts[y] += 1
                by_speaker[r["speaker_id"]][(y, x)] += 1
        total = sum(joint.values())
        if not total:
            continue
        mi = mutual_information_from_joint(joint)
        h_y = entropy_from_counter(y_counts)
        h_x = entropy_from_counter(x_counts)
        rows.append({
            "mode": mode,
            "ngram_order": n,
            "target": "central_phone_39",
            "num_examples": total,
            "active_token_ngrams": len(x_counts),
            "token_ngram_perplexity": math.exp(h_x),
            "token_ngram_to_phone_purity": _purity_x_to_y(joint),
            "phone_to_token_ngram_cluster_purity": _purity_y_to_x(joint),
            "mutual_information_nats": mi,
            "phone_normalized_mi": mi / max(h_y, 1e-12),
            "loso_token_ngram_to_phone_accuracy": loso_majority_accuracy(by_speaker, joint),
        })
    return rows


def ngram_phone_sequence_metrics(records, mode, max_n=3):
    """Frame-aligned token n-gram -> aligned 39-phone n-gram association."""
    rows = []
    for n in range(1, max_n + 1):
        joint, x_counts, y_counts = Counter(), Counter(), Counter()
        by_speaker = defaultdict(Counter)
        for r in records:
            toks = r[f"{mode}_token_ids"].tolist()
            phones = list(r["phone_labels_39"])
            m = min(len(toks), len(phones))
            toks, phones = toks[:m], phones[:m]
            for i in range(m - n + 1):
                y = tuple(phones[i:i+n])
                if any(p in IGNORE_PHONES for p in y):
                    continue
                x = tuple(int(v) for v in toks[i:i+n])
                joint[(y, x)] += 1
                x_counts[x] += 1
                y_counts[y] += 1
                by_speaker[r["speaker_id"]][(y, x)] += 1
        total = sum(joint.values())
        if not total:
            continue
        mi = mutual_information_from_joint(joint)
        h_y = entropy_from_counter(y_counts)
        rows.append({
            "mode": mode,
            "ngram_order": n,
            "target": f"aligned_phone_{n}gram_39",
            "num_examples": total,
            "active_token_ngrams": len(x_counts),
            "active_phone_ngrams": len(y_counts),
            "token_ngram_to_phone_ngram_purity": _purity_x_to_y(joint),
            "phone_ngram_to_token_ngram_cluster_purity": _purity_y_to_x(joint),
            "mutual_information_nats": mi,
            "phone_ngram_normalized_mi": mi / max(h_y, 1e-12),
            "loso_token_ngram_to_phone_ngram_accuracy": loso_majority_accuracy(by_speaker, joint),
        })
    return rows


# ------------------------ New transition analyses --------------------------

def conditional_entropy_from_ngram_counts(ngram_counts, prefix_counts):
    total = sum(ngram_counts.values())
    if total <= 0:
        return float("nan")
    h = 0.0
    for ng, c in ngram_counts.items():
        prefix = ng[:-1]
        p_joint = c / total
        p_cond = c / max(prefix_counts[prefix], 1)
        h -= p_joint * math.log(max(p_cond, 1e-300))
    return float(h)


def transition_metrics(records, mode):
    uni, bi, tri = Counter(), Counter(), Counter()
    bi_prefix, tri_prefix = Counter(), Counter()
    for r in records:
        seq = collapse_consecutive(r[f"{mode}_token_ids"].tolist())
        for z in seq:
            uni[(z,)] += 1
        for bg in sliding_ngrams(seq, 2):
            bi[bg] += 1
            bi_prefix[(bg[0],)] += 1
        for tg in sliding_ngrams(seq, 3):
            tri[tg] += 1
            tri_prefix[(tg[0], tg[1])] += 1
    h1 = entropy_from_counter(uni)
    h2_joint = entropy_from_counter(bi)
    h3_joint = entropy_from_counter(tri)
    h_next_prev = conditional_entropy_from_ngram_counts(bi, bi_prefix)
    h_next_prev2 = conditional_entropy_from_ngram_counts(tri, tri_prefix)
    adjacent_mi = h1 - h_next_prev if np.isfinite(h_next_prev) else float("nan")
    return {
        "mode": mode,
        "num_unigram_events": sum(uni.values()),
        "num_bigram_events": sum(bi.values()),
        "num_trigram_events": sum(tri.values()),
        "active_unigrams": len(uni),
        "active_bigrams": len(bi),
        "active_trigrams": len(tri),
        "unigram_entropy_nats": h1,
        "bigram_joint_entropy_nats": h2_joint,
        "trigram_joint_entropy_nats": h3_joint,
        "conditional_entropy_next_given_prev_nats": h_next_prev,
        "conditional_entropy_next_given_prev2_nats": h_next_prev2,
        "adjacent_token_mutual_information_nats": adjacent_mi,
        "unigram_perplexity": math.exp(h1),
        "conditional_bigram_perplexity": math.exp(h_next_prev) if np.isfinite(h_next_prev) else float("nan"),
        "conditional_trigram_perplexity": math.exp(h_next_prev2) if np.isfinite(h_next_prev2) else float("nan"),
    }


# ------------------------ New speaker/privacy probes -----------------------

def segment_symbolic_sequence(record, mode):
    return collapse_consecutive(record[f"{mode}_token_ids"].tolist())


def ngram_count_dict(seq, n):
    counts = Counter(sliding_ngrams(seq, n))
    return {"|".join(map(str, ng)): float(c) for ng, c in counts.items()}


def build_speaker_probe_dataset(records, mode, n):
    X_dict, y = [], []
    for r in records:
        seq = segment_symbolic_sequence(r, mode)
        feats = ngram_count_dict(seq, n)
        if not feats:
            continue
        X_dict.append(feats)
        y.append(r["speaker_id"])
    return X_dict, np.asarray(y)


def safe_cv_splits(y, requested_folds):
    counts = Counter(y.tolist())
    if len(counts) < 2:
        return 0
    return max(0, min(requested_folds, min(counts.values())))


def run_speaker_identification_probe(records, mode, n, cv_folds, seed, max_iter, C):
    X_dict, y_raw = build_speaker_probe_dataset(records, mode, n)
    if len(y_raw) == 0:
        return {"mode": mode, "ngram_order": n, "status": "no_examples"}

    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    n_speakers = len(le.classes_)
    folds = safe_cv_splits(y, cv_folds)
    if n_speakers < 2 or folds < 2:
        return {
            "mode": mode,
            "ngram_order": n,
            "status": "insufficient_per_speaker_segments_for_cv",
            "num_segments": len(y),
            "num_speakers": n_speakers,
            "usable_cv_folds": folds,
        }

    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    y_true_all, y_pred_all = [], []
    top5_hits, fold_acc, fold_f1, fold_bal = [], [], [], []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(np.zeros(len(y)), y)):
        vec = DictVectorizer(sparse=True)
        X_train = normalize(vec.fit_transform([X_dict[i] for i in train_idx]), norm="l2", copy=False)
        X_test = normalize(vec.transform([X_dict[i] for i in test_idx]), norm="l2", copy=False)

        clf = LogisticRegression(
            C=C,
            max_iter=max_iter,
            solver="liblinear" if n_speakers <= 50 else "saga",
            multi_class="ovr" if n_speakers <= 50 else "auto",
            random_state=seed + fold_idx,
            n_jobs=1,
        )
        clf.fit(X_train, y[train_idx])
        pred = clf.predict(X_test)
        y_true_all.extend(y[test_idx].tolist())
        y_pred_all.extend(pred.tolist())
        fold_acc.append(accuracy_score(y[test_idx], pred))
        fold_f1.append(f1_score(y[test_idx], pred, average="macro", zero_division=0))
        fold_bal.append(balanced_accuracy_score(y[test_idx], pred))

        prob = clf.predict_proba(X_test)
        classes = clf.classes_
        k = min(5, len(classes))
        top_idx = np.argpartition(prob, -k, axis=1)[:, -k:]
        top_classes = classes[top_idx]
        for truth, cand in zip(y[test_idx], top_classes):
            top5_hits.append(float(truth in set(cand.tolist())))

    y_true_all = np.asarray(y_true_all)
    y_pred_all = np.asarray(y_pred_all)
    return {
        "mode": mode,
        "ngram_order": n,
        "status": "ok",
        "num_segments": len(y),
        "num_speakers": n_speakers,
        "cv_folds": folds,
        "chance_accuracy_uniform": 1.0 / max(n_speakers, 1),
        "accuracy_micro": float(accuracy_score(y_true_all, y_pred_all)),
        "macro_f1": float(f1_score(y_true_all, y_pred_all, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true_all, y_pred_all)),
        "top5_accuracy": float(np.mean(top5_hits)) if top5_hits else float("nan"),
        "fold_accuracy_mean": float(np.mean(fold_acc)),
        "fold_accuracy_std": float(np.std(fold_acc, ddof=1)) if len(fold_acc) > 1 else 0.0,
        "fold_macro_f1_mean": float(np.mean(fold_f1)),
        "fold_balanced_accuracy_mean": float(np.mean(fold_bal)),
        "feature": f"bag_of_{n}gram_counts_l2",
        "classifier": "logistic_regression",
    }


def sample_verification_pairs(speakers, max_pairs_per_class, seed):
    rng = random.Random(seed)
    by_spk = defaultdict(list)
    for i, s in enumerate(speakers):
        by_spk[str(s)].append(i)

    same = []
    for ids in by_spk.values():
        for a in range(len(ids)):
            for b in range(a + 1, len(ids)):
                same.append((ids[a], ids[b]))
    if len(same) > max_pairs_per_class:
        same = rng.sample(same, max_pairs_per_class)

    target = len(same)
    spks = sorted(by_spk)
    diff, seen = [], set()
    attempts = 0
    max_attempts = max(10000, target * 30)
    while len(diff) < target and attempts < max_attempts and len(spks) >= 2:
        s1, s2 = rng.sample(spks, 2)
        i = rng.choice(by_spk[s1])
        j = rng.choice(by_spk[s2])
        p = (min(i, j), max(i, j))
        if p not in seen:
            seen.add(p)
            diff.append(p)
        attempts += 1

    n_bal = min(len(same), len(diff))
    if len(same) > n_bal:
        same = rng.sample(same, n_bal)
    if len(diff) > n_bal:
        diff = rng.sample(diff, n_bal)
    return same, diff


def sparse_cosine_pairs(X, pairs):
    scores = np.empty(len(pairs), dtype=np.float64)
    for k, (i, j) in enumerate(pairs):
        scores[k] = float(X[i].multiply(X[j]).sum())
    return scores


def compute_eer(labels, scores):
    if len(np.unique(labels)) < 2:
        return float("nan"), float("nan")
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1.0 - tpr
    idx = int(np.nanargmin(np.abs(fnr - fpr)))
    return float(0.5 * (fpr[idx] + fnr[idx])), float(thresholds[idx])


def run_speaker_verification_probe(records, mode, n, max_pairs_per_class, seed):
    X_dict, y = build_speaker_probe_dataset(records, mode, n)
    if len(y) == 0:
        return {"mode": mode, "ngram_order": n, "status": "no_examples"}

    vec = DictVectorizer(sparse=True)
    X = normalize(vec.fit_transform(X_dict), norm="l2", copy=False)
    same_pairs, diff_pairs = sample_verification_pairs(y.tolist(), max_pairs_per_class, seed)
    if not same_pairs or not diff_pairs:
        return {
            "mode": mode,
            "ngram_order": n,
            "status": "insufficient_pairs",
            "num_segments": len(y),
            "num_speakers": len(set(y.tolist())),
        }

    same_scores = sparse_cosine_pairs(X, same_pairs)
    diff_scores = sparse_cosine_pairs(X, diff_pairs)
    labels = np.concatenate([np.ones(len(same_scores), dtype=np.int64), np.zeros(len(diff_scores), dtype=np.int64)])
    scores = np.concatenate([same_scores, diff_scores])
    auc = roc_auc_score(labels, scores)
    eer, threshold = compute_eer(labels, scores)
    return {
        "mode": mode,
        "ngram_order": n,
        "status": "ok",
        "num_segments": len(y),
        "num_speakers": len(set(y.tolist())),
        "num_same_pairs": len(same_scores),
        "num_different_pairs": len(diff_scores),
        "roc_auc": float(auc),
        "eer": float(eer),
        "eer_threshold": float(threshold),
        "mean_same_speaker_cosine": float(np.mean(same_scores)),
        "mean_different_speaker_cosine": float(np.mean(diff_scores)),
        "cosine_gap_same_minus_different": float(np.mean(same_scores) - np.mean(diff_scores)),
        "feature": f"bag_of_{n}gram_counts_l2",
        "score": "cosine_similarity",
    }


def privacy_summary_rows(global_hubert_rows, speaker_id_rows, verification_rows):
    hubert = {r["mode"]: r for r in global_hubert_rows}
    sid = {(r.get("mode"), r.get("ngram_order")): r for r in speaker_id_rows}
    ver = {(r.get("mode"), r.get("ngram_order")): r for r in verification_rows}
    rows = []
    for mode in ["raw", "beta"]:
        for n in [1, 2, 3]:
            s = sid.get((mode, n), {})
            v = ver.get((mode, n), {})
            h = hubert.get(mode, {})
            rows.append({
                "mode": mode,
                "ngram_order": n,
                "speaker_nmi": h.get("speaker_nmi", "") if n == 1 else "",
                "conditional_speaker_nmi_given_phone": h.get("conditional_speaker_nmi_given_phone", "") if n == 1 else "",
                "speaker_id_accuracy": s.get("accuracy_micro", float("nan")),
                "speaker_id_macro_f1": s.get("macro_f1", float("nan")),
                "speaker_id_balanced_accuracy": s.get("balanced_accuracy", float("nan")),
                "speaker_id_chance": s.get("chance_accuracy_uniform", float("nan")),
                "speaker_verification_auc": v.get("roc_auc", float("nan")),
                "speaker_verification_eer": v.get("eer", float("nan")),
                "mean_same_speaker_cosine": v.get("mean_same_speaker_cosine", float("nan")),
                "mean_different_speaker_cosine": v.get("mean_different_speaker_cosine", float("nan")),
                "note": "Probe-specific leakage diagnostic; not a formal privacy guarantee.",
            })
    return rows


# ----------------------------- Collection ---------------------------------

def collect_aligned_records(items, output_dir, sides, expected_frames, write_frame_csv):
    aligned_dir = os.path.join(output_dir, "aligned_sequences")
    os.makedirs(aligned_dir, exist_ok=True)
    per_item_rows, records = [], []

    frame_csv_file = None
    frame_writer = None
    if write_frame_csv:
        import gzip
        frame_csv_file = gzip.open(os.path.join(output_dir, "frame_alignment.csv.gz"), "wt", newline="")
        frame_writer = csv.DictWriter(frame_csv_file, fieldnames=[
            "item_index", "file_id", "speaker_id", "side", "encoder_frame",
            "time_sec", "phone_61", "phone_39", "raw_token_pos", "raw_token_id",
            "beta_token_pos", "beta_token_id",
        ])
        frame_writer.writeheader()

    try:
        for item in tqdm(items, desc="Aligning tokens and phones"):
            meta = item.get("meta", {})
            speaker = str(meta.get("speaker_id", "unknown"))
            hop_samples = int(meta.get("hop_samples", 160))
            sample_rate = int(meta.get("sample_rate", 16000))
            hop_ms = 1000.0 * hop_samples / sample_rate

            for side in sides:
                if side not in item:
                    continue
                side_obj = item[side]
                if not side_obj.get("phone_alignment_valid", False):
                    continue
                if "phone_labels_61" not in side_obj or "phone_labels_39" not in side_obj:
                    raise KeyError(f"{item['_path']} has no phone labels. Re-run the metadata-aware generator.")

                raw_pos, raw_ids = get_alignment(side_obj, "raw")
                beta_pos, beta_ids = get_alignment(side_obj, "beta")
                phones61 = list(side_obj["phone_labels_61"])
                phones39 = list(side_obj["phone_labels_39"])
                lengths = [raw_pos.numel(), beta_pos.numel(), len(phones61), len(phones39)]
                if len(set(lengths)) != 1:
                    raise ValueError(f"Length mismatch in {item['_path']} side={side}: {lengths}")
                if expected_frames > 0 and lengths[0] != expected_frames:
                    raise ValueError(f"Expected {expected_frames} frames, got {lengths[0]} in {item['_path']} side={side}")

                record = {
                    "item_index": int(item.get("item_index", -1)),
                    "file_id": str(item.get("file_id", "")),
                    "speaker_id": speaker,
                    "side": side,
                    "raw_token_pos": raw_pos,
                    "beta_token_pos": beta_pos,
                    "raw_token_ids": raw_ids,
                    "beta_token_ids": beta_ids,
                    "phone_labels_61": phones61,
                    "phone_labels_39": phones39,
                    "hop_ms": hop_ms,
                }
                records.append(record)

                save_name = f"item_{record['item_index']:06d}_{side}.pt" if record["item_index"] >= 0 else f"{Path(item['_path']).stem}_{side}.pt"
                torch.save(record, os.path.join(aligned_dir, save_name))

                row = {
                    "path": item["_path"],
                    "item_index": record["item_index"],
                    "file_id": record["file_id"],
                    "speaker_id": speaker,
                    "side": side,
                    "num_frames": lengths[0],
                    "num_decoder_tokens": int(as_long_vector(side_obj["tokens"]).numel()),
                }
                for mode, positions in [("raw", raw_pos), ("beta", beta_pos)]:
                    metrics = {**boundary_metrics(positions, phones61, hop_ms), **segment_alignment_metrics(positions, phones39)}
                    for k, v in metrics.items():
                        row[f"{mode}_{k}"] = v
                    for k, v in side_obj.get(f"{mode}_metrics", {}).items():
                        try:
                            row[f"{mode}_{k}"] = float(v)
                        except Exception:
                            pass
                per_item_rows.append(row)

                if frame_writer is not None:
                    for f in range(lengths[0]):
                        frame_writer.writerow({
                            "item_index": record["item_index"],
                            "file_id": record["file_id"],
                            "speaker_id": speaker,
                            "side": side,
                            "encoder_frame": f,
                            "time_sec": f * hop_samples / sample_rate,
                            "phone_61": phones61[f],
                            "phone_39": phones39[f],
                            "raw_token_pos": int(raw_pos[f]),
                            "raw_token_id": int(raw_ids[f]),
                            "beta_token_pos": int(beta_pos[f]),
                            "beta_token_id": int(beta_ids[f]),
                        })
    finally:
        if frame_csv_file is not None:
            frame_csv_file.close()

    return records, per_item_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="cross_attention_phone_output_TIMIT")
    parser.add_argument("--output_dir", default="cross_attention_phone_analysis_TIMIT")
    parser.add_argument("--sides", default="anc", help="anc, pos, or anc,pos")
    parser.add_argument("--expected_frames", type=int, default=301)
    parser.add_argument("--write_frame_csv", action="store_true")
    parser.add_argument("--bootstrap_samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--skip_abx", action="store_true")
    parser.add_argument("--abx_max_triplets", type=int, default=100000)
    parser.add_argument("--abx_include_silence", action="store_true")

    parser.add_argument("--max_ngram_order", type=int, default=3)
    parser.add_argument("--skip_speaker_probe", action="store_true")
    parser.add_argument("--speaker_cv_folds", type=int, default=5)
    parser.add_argument("--speaker_probe_max_iter", type=int, default=1000)
    parser.add_argument("--speaker_probe_C", type=float, default=1.0)
    parser.add_argument("--verification_max_pairs_per_class", type=int, default=50000)
    args = parser.parse_args()

    if args.max_ngram_order < 1:
        raise ValueError("--max_ngram_order must be >= 1")

    os.makedirs(args.output_dir, exist_ok=True)
    sides = [x.strip() for x in args.sides.split(",") if x.strip()]
    for side in sides:
        if side not in {"anc", "pos"}:
            raise ValueError(f"Invalid side: {side}")

    items = load_items(args.input_dir)
    records, per_item = collect_aligned_records(
        items, args.output_dir, sides, args.expected_frames, args.write_frame_csv
    )
    if not records:
        raise RuntimeError("No valid phone-aligned records.")

    write_csv(os.path.join(args.output_dir, "per_item_alignment_metrics.csv"), per_item)

    global_rows = []
    for mode in ["raw", "beta"]:
        metrics, mapping = hubert_style_metrics(records, mode)
        global_rows.append(metrics)
        write_csv(os.path.join(args.output_dir, f"token_phone_mapping_{mode}.csv"), mapping)
    write_csv(os.path.join(args.output_dir, "global_hubert_metrics.csv"), global_rows)

    paired = paired_improvement_summary(per_item, args.bootstrap_samples, args.seed)
    write_csv(os.path.join(args.output_dir, "paired_beta_improvement.csv"), paired)

    if not args.skip_abx:
        occurrences = extract_triphone_occurrences(records, exclude_silence=not args.abx_include_silence)
        abx_rows = sampled_discrete_abx(occurrences, args.abx_max_triplets, args.seed)
        write_csv(os.path.join(args.output_dir, "abx_discrete_metrics.csv"), abx_rows)

    ngram_phone_rows, ngram_phone_seq_rows, transition_rows = [], [], []
    for mode in ["raw", "beta"]:
        ngram_phone_rows.extend(ngram_phone_metrics(records, mode, args.max_ngram_order))
        ngram_phone_seq_rows.extend(ngram_phone_sequence_metrics(records, mode, args.max_ngram_order))
        transition_rows.append(transition_metrics(records, mode))
    write_csv(os.path.join(args.output_dir, "ngram_phone_metrics.csv"), ngram_phone_rows)
    write_csv(os.path.join(args.output_dir, "ngram_phone_sequence_metrics.csv"), ngram_phone_seq_rows)
    write_csv(os.path.join(args.output_dir, "token_transition_metrics.csv"), transition_rows)

    speaker_id_rows, verification_rows = [], []
    if not args.skip_speaker_probe:
        for mode in ["raw", "beta"]:
            for n in range(1, args.max_ngram_order + 1):
                print(f"[speaker-ID] mode={mode} n={n}")
                speaker_id_rows.append(run_speaker_identification_probe(
                    records, mode, n, args.speaker_cv_folds, args.seed,
                    args.speaker_probe_max_iter, args.speaker_probe_C
                ))
                print(f"[speaker-verification] mode={mode} n={n}")
                verification_rows.append(run_speaker_verification_probe(
                    records, mode, n, args.verification_max_pairs_per_class, args.seed
                ))
        write_csv(os.path.join(args.output_dir, "speaker_ngram_probe.csv"), speaker_id_rows)
        write_csv(os.path.join(args.output_dir, "speaker_verification_metrics.csv"), verification_rows)
        write_csv(os.path.join(args.output_dir, "privacy_summary.csv"), privacy_summary_rows(global_rows, speaker_id_rows, verification_rows))

    print("\n=== COMPLETE ===")
    print(f"Artifacts loaded: {len(items)}")
    print(f"Aligned side records: {len(records)}")
    print(f"Output directory: {args.output_dir}")
    print("New outputs:")
    print("  ngram_phone_metrics.csv")
    print("  ngram_phone_sequence_metrics.csv")
    print("  token_transition_metrics.csv")
    if not args.skip_speaker_probe:
        print("  speaker_ngram_probe.csv")
        print("  speaker_verification_metrics.csv")
        print("  privacy_summary.csv")
    print("\nPrivacy caution: these are leakage probes, not formal privacy guarantees.")


if __name__ == "__main__":
    main()
