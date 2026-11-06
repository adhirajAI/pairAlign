#!/usr/bin/env python3
"""
Stage~I Geometric phonetic / privacy analyzer for pre-aligned TIMIT records.

Input
-----
A directory containing flattened aligned record files, e.g.

    geometric_phone_analysis_TIMIT/Stage_I_Geometric/aligned_sequences/*.pt

Each .pt record is expected to contain at least:
    item_index
    file_id
    speaker_id
    side                  # "anc" or "pos"
    phone_labels_39
    phone_labels_61
and one aligned token-id sequence under one of:
    geometric_token_ids
    token_ids
    frame_token_ids
    raw_token_ids
    beta_token_ids

Optionally, a token-position sequence may also be present under one of:
    geometric_token_pos
    token_pos
    token_positions
    raw_token_pos
    beta_token_pos

The script auto-detects the token-id / token-position keys and reports them.

Outputs
-------
1. geometric_global_phone_metrics.csv
   HuBERT-style phone purity, cluster purity, PNMI, codebook perplexity,
   silence purity, speaker NMI, conditional speaker leakage, and LOSO
   token-to-phone accuracy.

2. geometric_ngram_phone_metrics.csv
   Unigram/bigram/trigram token-context -> central-phone association.

3. geometric_ngram_phone_sequence_metrics.csv
   Token n-gram -> aligned phone n-gram association.

4. geometric_token_transition_metrics.csv
   H(Z), H(Z_t|Z_{t-1}), H(Z_t|Z_{t-2},Z_{t-1}), adjacent-token MI,
   active unigram/bigram/trigram counts and perplexities.

5. geometric_speaker_ngram_probe.csv
   Closed-set speaker-ID using bag-of-token unigram/bigram/trigram statistics.

6. geometric_speaker_verification_metrics.csv
   Symbolic speaker verification from cosine similarity with ROC-AUC and EER.

7. geometric_privacy_summary.csv
   Convenience summary of speaker leakage probes.

8. geometric_boundary_metrics.csv
   If token positions are available, boundary precision/recall/F1/MAE and
   segmentation purity versus TIMIT phone boundaries.

Interpretation
--------------
These probes quantify phone-correlated structure and recoverable speaker
information. They do NOT establish formal privacy preservation.

Dependencies
------------
numpy, torch, tqdm, scikit-learn
"""

import os
import csv
import math
import random
import argparse
from glob import glob
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


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------

def write_csv(path: str, rows: List[Dict[str, Any]]):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if not rows:
        return
    keys = []
    for r in rows:
        for k in r.keys():
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def safe_torch_load(path: str):
    # Explicit weights_only=False suppresses the future-default ambiguity while
    # preserving compatibility with arbitrary Python objects in trusted local artifacts.
    return torch.load(path, map_location="cpu", weights_only=False)


def load_pre_aligned_records(input_dir: str, requested_sides: Sequence[str]) -> List[Dict[str, Any]]:
    paths = sorted(glob(os.path.join(input_dir, "*.pt")))
    if not paths:
        raise FileNotFoundError(f"No .pt files found in {input_dir}")

    records = []
    detected_token_keys = Counter()
    detected_pos_keys = Counter()
    skipped_side = 0
    skipped_schema = 0

    token_id_candidates = [
        "geometric_token_ids",
        "token_ids",
        "frame_token_ids",
        "raw_token_ids",
        "beta_token_ids",
    ]
    token_pos_candidates = [
        "geometric_token_pos",
        "token_pos",
        "token_positions",
        "raw_token_pos",
        "beta_token_pos",
    ]

    for path in tqdm(paths, desc="Loading pre-aligned geometric records"):
        obj = safe_torch_load(path)

        side = str(obj.get("side", "anc"))
        if requested_sides and side not in requested_sides:
            skipped_side += 1
            continue

        token_key = next((k for k in token_id_candidates if k in obj), None)
        if token_key is None:
            skipped_schema += 1
            continue

        pos_key = next((k for k in token_pos_candidates if k in obj), None)

        if "phone_labels_39" not in obj:
            skipped_schema += 1
            continue

        rec = dict(obj)
        rec["_path"] = path
        rec["_token_key"] = token_key
        rec["_pos_key"] = pos_key
        rec["geometric_token_ids"] = as_long_vector(obj[token_key])
        if pos_key is not None:
            rec["geometric_token_pos"] = as_long_vector(obj[pos_key])

        rec["speaker_id"] = str(obj.get("speaker_id", "unknown"))
        rec["side"] = side
        rec["file_id"] = str(obj.get("file_id", os.path.basename(path)))
        rec["item_index"] = int(obj.get("item_index", -1))

        detected_token_keys[token_key] += 1
        if pos_key is not None:
            detected_pos_keys[pos_key] += 1
        records.append(rec)

    print("\n=== INPUT DETECTION ===")
    print("Detected input schema: pre-aligned flattened records")
    print(f"Files scanned: {len(paths)}")
    print(f"Records accepted: {len(records)}")
    print(f"Skipped by side filter: {skipped_side}")
    print(f"Skipped due to missing token/phone fields: {skipped_schema}")
    print(f"Token-ID keys detected: {dict(detected_token_keys)}")
    print(f"Token-position keys detected: {dict(detected_pos_keys)}")

    side_counts = Counter(r["side"] for r in records)
    print(f"Accepted sides: {dict(side_counts)}")

    if not records:
        raise RuntimeError(
            "No valid pre-aligned geometric records were found. "
            "Expected flattened records with phone_labels_39 and one recognized token-id key."
        )

    return records


def as_long_vector(x: Any) -> torch.LongTensor:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().reshape(-1).long()
    return torch.tensor(x, dtype=torch.long).reshape(-1)


# ---------------------------------------------------------------------------
# Core information-theoretic helpers
# ---------------------------------------------------------------------------

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


def conditional_mi_token_speaker_given_phone(triple: Counter) -> Tuple[float, float]:
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


# ---------------------------------------------------------------------------
# HuBERT-style phone / speaker association
# ---------------------------------------------------------------------------

def global_phone_metrics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    joint = Counter()
    token_counts = Counter()
    phone_counts = Counter()
    speaker_token_joint = Counter()
    triple = Counter()
    speaker_joint = defaultdict(Counter)

    for r in tqdm(records, desc="Global phone/speaker statistics"):
        tokens = r["geometric_token_ids"].tolist()
        phones = list(r["phone_labels_39"])
        speaker = r["speaker_id"]

        for z, y in zip(tokens, phones):
            if y in IGNORE_PHONES:
                continue
            z = int(z)
            joint[(y, z)] += 1
            token_counts[z] += 1
            phone_counts[y] += 1
            speaker_token_joint[(speaker, z)] += 1
            triple[(y, z, speaker)] += 1
            speaker_joint[speaker][(y, z)] += 1

    N = sum(joint.values())
    if N == 0:
        raise RuntimeError("No valid phone-token frames found.")

    phone_purity = sum(
        max(joint[(y, z)] for y in phone_counts)
        for z in token_counts
    ) / N

    cluster_purity = sum(
        max(joint[(y, z)] for z in token_counts)
        for y in phone_counts
    ) / N

    mi = mutual_information_from_joint(joint)
    h_phone = entropy_from_counter(phone_counts)
    pnmi = mi / max(h_phone, 1e-12)

    h_token = entropy_from_counter(token_counts)
    code_perplexity = math.exp(h_token)

    speaker_counts = Counter()
    for (s, _), n in speaker_token_joint.items():
        speaker_counts[s] += n
    h_speaker = entropy_from_counter(speaker_counts)
    speaker_nmi = mutual_information_from_joint(speaker_token_joint) / max(h_speaker, 1e-12)

    cmi, h_s_given_phone = conditional_mi_token_speaker_given_phone(triple)
    conditional_speaker_nmi = cmi / max(h_s_given_phone, 1e-12)

    silence_joint = Counter()
    for (y, z), n in joint.items():
        silence_joint[("silence" if y in SILENCE_39 else "speech", z)] += n

    silence_purity = sum(
        max(silence_joint[(label, z)] for label in ["silence", "speech"])
        for z in token_counts
    ) / N

    # LOSO token->phone majority mapping
    speakers = sorted(speaker_joint)
    global_phone_majority = phone_counts.most_common(1)[0][0]
    loso_correct = 0
    loso_total = 0

    for heldout in tqdm(speakers, desc="LOSO token→phone", leave=False):
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

        fallback = (
            train_phone_counts.most_common(1)[0][0]
            if train_phone_counts else global_phone_majority
        )
        mapping = {
            z: cnt.most_common(1)[0][0]
            for z, cnt in train_by_token.items()
        }

        for (y, z), n in heldout_joint.items():
            if mapping.get(z, fallback) == y:
                loso_correct += n
            loso_total += n

    return {
        "model": "Stage_I_Geometric",
        "num_valid_frames": N,
        "active_vocab": len(token_counts),
        "codebook_perplexity": code_perplexity,
        "normalized_codebook_perplexity": code_perplexity / max(len(token_counts), 1),
        "phone_purity": phone_purity,
        "cluster_purity": cluster_purity,
        "pnmi": pnmi,
        "silence_binary_purity": silence_purity,
        "speaker_nmi": speaker_nmi,
        "conditional_speaker_nmi_given_phone": conditional_speaker_nmi,
        "loso_token_to_phone_accuracy": loso_correct / max(loso_total, 1),
    }


# ---------------------------------------------------------------------------
# N-gram phone analyses
# ---------------------------------------------------------------------------

def sliding_ngrams(values: Sequence[Any], n: int):
    vals = list(values)
    if len(vals) < n:
        return []
    return [tuple(vals[i:i+n]) for i in range(len(vals) - n + 1)]


def center_index_for_ngram(start: int, n: int) -> int:
    return start + n // 2


def purity_x_to_y(joint: Counter) -> float:
    total = sum(joint.values())
    by_x = defaultdict(Counter)
    for (y, x), n in joint.items():
        by_x[x][y] += n
    return sum(max(c.values()) for c in by_x.values()) / max(total, 1)


def purity_y_to_x(joint: Counter) -> float:
    total = sum(joint.values())
    by_y = defaultdict(Counter)
    for (y, x), n in joint.items():
        by_y[y][x] += n
    return sum(max(c.values()) for c in by_y.values()) / max(total, 1)


def loso_majority_accuracy(by_speaker: Dict[str, Counter], joint: Counter) -> float:
    y_counts = Counter()
    for (y, _), n in joint.items():
        y_counts[y] += n
    if not y_counts:
        return float("nan")

    global_y = y_counts.most_common(1)[0][0]
    correct = 0
    total = 0

    for heldout, heldout_joint in tqdm(
        by_speaker.items(), desc="LOSO n-gram prediction", leave=False
    ):
        train_joint = joint.copy()
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


def ngram_phone_metrics(records: List[Dict[str, Any]], max_n: int = 3):
    rows = []

    for n in tqdm(range(1, max_n + 1), desc="Token n-gram→central-phone"):
        joint = Counter()
        x_counts = Counter()
        y_counts = Counter()
        by_speaker = defaultdict(Counter)

        for r in tqdm(records, desc=f"Scanning n={n}", leave=False):
            toks = r["geometric_token_ids"].tolist()
            phones = list(r["phone_labels_39"])
            m = min(len(toks), len(phones))
            toks, phones = toks[:m], phones[:m]

            for i in range(m - n + 1):
                center = center_index_for_ngram(i, n)
                y = phones[center]
                if y in IGNORE_PHONES:
                    continue
                x = tuple(int(v) for v in toks[i:i+n])
                joint[(y, x)] += 1
                x_counts[x] += 1
                y_counts[y] += 1
                by_speaker[r["speaker_id"]][(y, x)] += 1

        mi = mutual_information_from_joint(joint)
        h_y = entropy_from_counter(y_counts)
        h_x = entropy_from_counter(x_counts)

        rows.append({
            "model": "Stage_I_Geometric",
            "ngram_order": n,
            "target": "central_phone_39",
            "num_examples": sum(joint.values()),
            "active_token_ngrams": len(x_counts),
            "token_ngram_perplexity": math.exp(h_x),
            "token_ngram_to_phone_purity": purity_x_to_y(joint),
            "phone_to_token_ngram_cluster_purity": purity_y_to_x(joint),
            "mutual_information_nats": mi,
            "phone_normalized_mi": mi / max(h_y, 1e-12),
            "loso_token_ngram_to_phone_accuracy": loso_majority_accuracy(
                by_speaker, joint
            ),
        })
    return rows


def ngram_phone_sequence_metrics(records: List[Dict[str, Any]], max_n: int = 3):
    rows = []

    for n in tqdm(range(1, max_n + 1), desc="Token n-gram→phone n-gram"):
        joint = Counter()
        x_counts = Counter()
        y_counts = Counter()
        by_speaker = defaultdict(Counter)

        for r in tqdm(records, desc=f"Scanning n={n}", leave=False):
            toks = r["geometric_token_ids"].tolist()
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

        mi = mutual_information_from_joint(joint)
        h_y = entropy_from_counter(y_counts)

        rows.append({
            "model": "Stage_I_Geometric",
            "ngram_order": n,
            "target": f"aligned_phone_{n}gram_39",
            "num_examples": sum(joint.values()),
            "active_token_ngrams": len(x_counts),
            "active_phone_ngrams": len(y_counts),
            "token_ngram_to_phone_ngram_purity": purity_x_to_y(joint),
            "phone_ngram_to_token_ngram_cluster_purity": purity_y_to_x(joint),
            "mutual_information_nats": mi,
            "phone_ngram_normalized_mi": mi / max(h_y, 1e-12),
            "loso_token_ngram_to_phone_ngram_accuracy": loso_majority_accuracy(
                by_speaker, joint
            ),
        })
    return rows


# ---------------------------------------------------------------------------
# Sequence / transition metrics
# ---------------------------------------------------------------------------

def collapse_consecutive(values: Sequence[int]) -> Tuple[int, ...]:
    if not values:
        return tuple()
    out = [int(values[0])]
    for x in values[1:]:
        x = int(x)
        if x != out[-1]:
            out.append(x)
    return tuple(out)


def conditional_entropy_from_ngram_counts(ngram_counts: Counter, prefix_counts: Counter):
    total = sum(ngram_counts.values())
    if total <= 0:
        return float("nan")
    h = 0.0
    for ng, c in ngram_counts.items():
        prefix = ng[:-1]
        p_joint = c / total
        p_cond = c / max(prefix_counts[prefix], 1)
        h -= p_joint * math.log(max(p_cond, 1e-300))
    return h


def transition_metrics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    uni = Counter()
    bi = Counter()
    tri = Counter()
    bi_prefix = Counter()
    tri_prefix = Counter()

    for r in tqdm(records, desc="Token transition statistics"):
        seq = collapse_consecutive(r["geometric_token_ids"].tolist())

        for z in seq:
            uni[(z,)] += 1
        for bg in sliding_ngrams(seq, 2):
            bi[bg] += 1
            bi_prefix[(bg[0],)] += 1
        for tg in sliding_ngrams(seq, 3):
            tri[tg] += 1
            tri_prefix[(tg[0], tg[1])] += 1

    h1 = entropy_from_counter(uni)
    h2 = entropy_from_counter(bi)
    h3 = entropy_from_counter(tri)
    h_next_prev = conditional_entropy_from_ngram_counts(bi, bi_prefix)
    h_next_prev2 = conditional_entropy_from_ngram_counts(tri, tri_prefix)

    return {
        "model": "Stage_I_Geometric",
        "num_unigram_events": sum(uni.values()),
        "num_bigram_events": sum(bi.values()),
        "num_trigram_events": sum(tri.values()),
        "active_unigrams": len(uni),
        "active_bigrams": len(bi),
        "active_trigrams": len(tri),
        "unigram_entropy_nats": h1,
        "bigram_joint_entropy_nats": h2,
        "trigram_joint_entropy_nats": h3,
        "conditional_entropy_next_given_prev_nats": h_next_prev,
        "conditional_entropy_next_given_prev2_nats": h_next_prev2,
        "adjacent_token_mutual_information_nats": (
            h1 - h_next_prev if np.isfinite(h_next_prev) else float("nan")
        ),
        "unigram_perplexity": math.exp(h1),
        "conditional_bigram_perplexity": (
            math.exp(h_next_prev) if np.isfinite(h_next_prev) else float("nan")
        ),
        "conditional_trigram_perplexity": (
            math.exp(h_next_prev2) if np.isfinite(h_next_prev2) else float("nan")
        ),
    }


# ---------------------------------------------------------------------------
# Boundary / segmentation analysis, when positions exist
# ---------------------------------------------------------------------------

def runs(values: Sequence[Any]):
    if len(values) == 0:
        return []
    out = []
    start = 0
    prev = values[0]
    for i in range(1, len(values)):
        if values[i] != prev:
            out.append((start, i, prev))
            start = i
            prev = values[i]
    out.append((start, len(values), prev))
    return out


def boundaries_from_labels(labels):
    out = []
    for i in range(1, len(labels)):
        a, b = str(labels[i-1]), str(labels[i])
        if a in IGNORE_PHONES or b in IGNORE_PHONES:
            continue
        if a != b:
            out.append(i)
    return np.asarray(out, dtype=np.int64)


def boundaries_from_token_positions(pos):
    x = np.asarray(pos, dtype=np.int64)
    if x.size < 2:
        return np.empty(0, dtype=np.int64)
    return np.flatnonzero(x[1:] != x[:-1]) + 1


def one_to_one_boundary_match(pred, ref, tolerance_frames):
    candidates = []
    for i, p in enumerate(pred):
        left = np.searchsorted(ref, p - tolerance_frames, side="left")
        right = np.searchsorted(ref, p + tolerance_frames, side="right")
        for j in range(left, right):
            candidates.append((abs(int(p) - int(ref[j])), i, j))
    candidates.sort()

    used_pred, used_ref, distances = set(), set(), []
    for d, i, j in candidates:
        if i not in used_pred and j not in used_ref:
            used_pred.add(i)
            used_ref.add(j)
            distances.append(d)
    tp = len(distances)
    return tp, len(pred)-tp, len(ref)-tp, distances


def segment_alignment_metrics(token_positions, phone_labels_39):
    pos = token_positions.numpy().tolist()
    n = min(len(pos), len(phone_labels_39))
    pos = pos[:n]
    phones = list(phone_labels_39[:n])

    token_runs = runs(pos)
    phone_runs = [(s,e,p) for s,e,p in runs(phones) if p not in IGNORE_PHONES]

    token_majority_correct = 0
    token_valid_frames = 0
    phones_per_token = []
    for s,e,_ in token_runs:
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
    for s,e,_ in phone_runs:
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


def boundary_rows(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []

    usable = [r for r in records if "geometric_token_pos" in r and "phone_labels_61" in r]
    if not usable:
        return rows

    for r in tqdm(usable, desc="Boundary / segmentation metrics"):
        pos = r["geometric_token_pos"]
        phones61 = list(r["phone_labels_61"])
        phones39 = list(r["phone_labels_39"])
        hop_ms = float(r.get("hop_ms", 10.0))

        pred = boundaries_from_token_positions(pos.numpy())
        ref = boundaries_from_labels(phones61)

        row = {
            "item_index": r["item_index"],
            "file_id": r["file_id"],
            "speaker_id": r["speaker_id"],
            "side": r["side"],
            "num_token_boundaries": len(pred),
            "num_phone_boundaries": len(ref),
        }

        for tol in [1,2,3]:
            tp, fp, fn, distances = one_to_one_boundary_match(pred, ref, tol)
            p = tp / max(tp+fp, 1)
            rr = tp / max(tp+fn, 1)
            f1 = 2*p*rr / max(p+rr, 1e-12)
            ms = int(round(tol * hop_ms))
            row[f"boundary_precision_{ms}ms"] = p
            row[f"boundary_recall_{ms}ms"] = rr
            row[f"boundary_f1_{ms}ms"] = f1
            if tol == 2:
                row["boundary_mae_ms_20ms"] = (
                    float(np.mean(distances) * hop_ms)
                    if distances else float("nan")
                )

        ratio = len(pred) / max(len(ref), 1)
        row["token_to_phone_boundary_count_ratio"] = ratio
        row["oversegmentation_ratio_abs_log"] = abs(math.log(max(ratio, 1e-12)))
        row.update(segment_alignment_metrics(pos, phones39))
        rows.append(row)

    return rows


# ---------------------------------------------------------------------------
# Speaker attacker probes
# ---------------------------------------------------------------------------

def segment_symbolic_sequence(r):
    return collapse_consecutive(r["geometric_token_ids"].tolist())


def ngram_count_dict(seq, n):
    c = Counter(sliding_ngrams(seq, n))
    return {"|".join(map(str, ng)): float(v) for ng, v in c.items()}


def build_speaker_dataset(records, n):
    X_dict, y = [], []
    for r in records:
        seq = segment_symbolic_sequence(r)
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


def run_speaker_id_probe(records, n, cv_folds, seed, max_iter, C):
    X_dict, y_raw = build_speaker_dataset(records, n)
    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    folds = safe_cv_splits(y, cv_folds)
    if len(le.classes_) < 2 or folds < 2:
        return {
            "model": "Stage_I_Geometric",
            "ngram_order": n,
            "status": "insufficient_per_speaker_segments_for_cv",
            "num_segments": len(y),
            "num_speakers": len(le.classes_),
        }

    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    yt_all, yp_all = [], []
    fold_acc, fold_f1, fold_bal = [], [], []

    for fold_idx, (tr, te) in enumerate(
        tqdm(skf.split(np.zeros(len(y)), y), total=folds,
             desc=f"Speaker-ID CV n={n}", leave=False)
    ):
        vec = DictVectorizer(sparse=True)
        Xtr = vec.fit_transform([X_dict[i] for i in tr])
        Xte = vec.transform([X_dict[i] for i in te])

        Xtr = normalize(Xtr, norm="l2", copy=False)
        Xte = normalize(Xte, norm="l2", copy=False)

        clf = LogisticRegression(
            C=C,
            max_iter=max_iter,
            solver="liblinear" if len(le.classes_) <= 50 else "saga",
            multi_class="ovr" if len(le.classes_) <= 50 else "auto",
            random_state=seed + fold_idx,
            n_jobs=1,
        )
        clf.fit(Xtr, y[tr])
        pred = clf.predict(Xte)

        yt_all.extend(y[te].tolist())
        yp_all.extend(pred.tolist())
        fold_acc.append(accuracy_score(y[te], pred))
        fold_f1.append(f1_score(y[te], pred, average="macro", zero_division=0))
        fold_bal.append(balanced_accuracy_score(y[te], pred))

    yt_all = np.asarray(yt_all)
    yp_all = np.asarray(yp_all)

    return {
        "model": "Stage_I_Geometric",
        "ngram_order": n,
        "status": "ok",
        "num_segments": len(y),
        "num_speakers": len(le.classes_),
        "cv_folds": folds,
        "chance_accuracy_uniform": 1.0 / len(le.classes_),
        "accuracy_micro": float(accuracy_score(yt_all, yp_all)),
        "macro_f1": float(f1_score(yt_all, yp_all, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(yt_all, yp_all)),
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
    for ids in tqdm(by_spk.values(), desc="Enumerating same-speaker pairs", leave=False):
        for i in range(len(ids)):
            for j in range(i+1, len(ids)):
                same.append((ids[i], ids[j]))

    if len(same) > max_pairs_per_class:
        same = rng.sample(same, max_pairs_per_class)

    target = len(same)
    different = []
    seen = set()
    spks = sorted(by_spk)
    max_attempts = max(10000, target * 30)

    for _ in tqdm(range(max_attempts), desc="Sampling different-speaker pairs", leave=False):
        if len(different) >= target:
            break
        if len(spks) < 2:
            break
        s1, s2 = rng.sample(spks, 2)
        a = rng.choice(by_spk[s1])
        b = rng.choice(by_spk[s2])
        pair = (min(a,b), max(a,b))
        if pair not in seen:
            seen.add(pair)
            different.append(pair)

    n_bal = min(len(same), len(different))
    if len(same) > n_bal:
        same = rng.sample(same, n_bal)
    if len(different) > n_bal:
        different = rng.sample(different, n_bal)
    return same, different


def sparse_cosine_pairs(X, pairs, desc):
    scores = np.empty(len(pairs), dtype=np.float64)
    for k, (i,j) in enumerate(tqdm(pairs, desc=desc, leave=False)):
        scores[k] = float(X[i].multiply(X[j]).sum())
    return scores


def compute_eer(labels, scores):
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1.0 - tpr
    idx = int(np.nanargmin(np.abs(fnr - fpr)))
    eer = 0.5 * (fpr[idx] + fnr[idx])
    return float(eer), float(thresholds[idx])


def run_speaker_verification(records, n, max_pairs_per_class, seed):
    X_dict, y = build_speaker_dataset(records, n)

    vec = DictVectorizer(sparse=True)
    X = vec.fit_transform(X_dict)
    X = normalize(X, norm="l2", copy=False)

    same_pairs, diff_pairs = sample_verification_pairs(
        y.tolist(), max_pairs_per_class, seed
    )

    if not same_pairs or not diff_pairs:
        return {
            "model": "Stage_I_Geometric",
            "ngram_order": n,
            "status": "insufficient_pairs",
        }

    same_scores = sparse_cosine_pairs(X, same_pairs, f"Same-speaker cosine n={n}")
    diff_scores = sparse_cosine_pairs(X, diff_pairs, f"Different-speaker cosine n={n}")

    labels = np.concatenate([
        np.ones(len(same_scores), dtype=np.int64),
        np.zeros(len(diff_scores), dtype=np.int64),
    ])
    scores = np.concatenate([same_scores, diff_scores])

    auc = roc_auc_score(labels, scores)
    eer, threshold = compute_eer(labels, scores)

    return {
        "model": "Stage_I_Geometric",
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
        "cosine_gap_same_minus_different": float(
            np.mean(same_scores) - np.mean(diff_scores)
        ),
        "feature": f"bag_of_{n}gram_counts_l2",
        "score": "cosine_similarity",
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_dir",
        default="geometric_phone_analysis_TIMIT/Stage_I_Geometric/aligned_sequences",
    )
    parser.add_argument(
        "--output_dir",
        default="geometric_phone_analysis_TIMIT_PRIVACY",
    )
    parser.add_argument("--sides", default="anc,pos")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--max_ngram_order", type=int, default=3)

    parser.add_argument("--skip_speaker_probe", action="store_true")
    parser.add_argument("--speaker_cv_folds", type=int, default=5)
    parser.add_argument("--speaker_probe_max_iter", type=int, default=1000)
    parser.add_argument("--speaker_probe_C", type=float, default=1.0)
    parser.add_argument("--verification_max_pairs_per_class", type=int, default=50000)

    args = parser.parse_args()

    sides = [s.strip() for s in args.sides.split(",") if s.strip()]
    os.makedirs(args.output_dir, exist_ok=True)

    records = load_pre_aligned_records(args.input_dir, sides)

    global_metrics = global_phone_metrics(records)
    write_csv(
        os.path.join(args.output_dir, "geometric_global_phone_metrics.csv"),
        [global_metrics],
    )

    ngram_rows = ngram_phone_metrics(records, args.max_ngram_order)
    write_csv(
        os.path.join(args.output_dir, "geometric_ngram_phone_metrics.csv"),
        ngram_rows,
    )

    ngram_seq_rows = ngram_phone_sequence_metrics(records, args.max_ngram_order)
    write_csv(
        os.path.join(args.output_dir, "geometric_ngram_phone_sequence_metrics.csv"),
        ngram_seq_rows,
    )

    trans = transition_metrics(records)
    write_csv(
        os.path.join(args.output_dir, "geometric_token_transition_metrics.csv"),
        [trans],
    )

    b_rows = boundary_rows(records)
    if b_rows:
        write_csv(
            os.path.join(args.output_dir, "geometric_boundary_metrics.csv"),
            b_rows,
        )
    else:
        print("No token-position field detected; skipping boundary metrics.")

    speaker_id_rows = []
    verification_rows = []

    if not args.skip_speaker_probe:
        for n in tqdm(
            range(1, args.max_ngram_order + 1),
            desc="Speaker/privacy probes"
        ):
            speaker_id_rows.append(
                run_speaker_id_probe(
                    records,
                    n=n,
                    cv_folds=args.speaker_cv_folds,
                    seed=args.seed,
                    max_iter=args.speaker_probe_max_iter,
                    C=args.speaker_probe_C,
                )
            )
            verification_rows.append(
                run_speaker_verification(
                    records,
                    n=n,
                    max_pairs_per_class=args.verification_max_pairs_per_class,
                    seed=args.seed,
                )
            )

        write_csv(
            os.path.join(args.output_dir, "geometric_speaker_ngram_probe.csv"),
            speaker_id_rows,
        )
        write_csv(
            os.path.join(args.output_dir, "geometric_speaker_verification_metrics.csv"),
            verification_rows,
        )

        privacy_rows = []
        for n in range(1, args.max_ngram_order + 1):
            sid = next((r for r in speaker_id_rows if r.get("ngram_order") == n), {})
            sv = next((r for r in verification_rows if r.get("ngram_order") == n), {})
            privacy_rows.append({
                "model": "Stage_I_Geometric",
                "ngram_order": n,
                "speaker_nmi": global_metrics["speaker_nmi"] if n == 1 else "",
                "conditional_speaker_nmi_given_phone": (
                    global_metrics["conditional_speaker_nmi_given_phone"]
                    if n == 1 else ""
                ),
                "speaker_id_accuracy": sid.get("accuracy_micro", float("nan")),
                "speaker_id_macro_f1": sid.get("macro_f1", float("nan")),
                "speaker_id_balanced_accuracy": sid.get("balanced_accuracy", float("nan")),
                "speaker_id_chance": sid.get("chance_accuracy_uniform", float("nan")),
                "speaker_verification_auc": sv.get("roc_auc", float("nan")),
                "speaker_verification_eer": sv.get("eer", float("nan")),
                "mean_same_speaker_cosine": sv.get("mean_same_speaker_cosine", float("nan")),
                "mean_different_speaker_cosine": sv.get("mean_different_speaker_cosine", float("nan")),
                "note": (
                    "Lower ID accuracy/AUC and higher EER indicate less recoverable "
                    "speaker information under these specific lightweight probes; "
                    "this is not a formal privacy guarantee."
                ),
            })

        write_csv(
            os.path.join(args.output_dir, "geometric_privacy_summary.csv"),
            privacy_rows,
        )

    print("\n=== COMPLETE ===")
    print(f"Records analyzed: {len(records)}")
    print(f"Output directory: {args.output_dir}")
    print("Generated:")
    print("  geometric_global_phone_metrics.csv")
    print("  geometric_ngram_phone_metrics.csv")
    print("  geometric_ngram_phone_sequence_metrics.csv")
    print("  geometric_token_transition_metrics.csv")
    if b_rows:
        print("  geometric_boundary_metrics.csv")
    if not args.skip_speaker_probe:
        print("  geometric_speaker_ngram_probe.csv")
        print("  geometric_speaker_verification_metrics.csv")
        print("  geometric_privacy_summary.csv")


if __name__ == "__main__":
    main()
