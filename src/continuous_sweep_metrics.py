#!/usr/bin/env python3
"""
Continuous-sweep tokenization analysis with optional compositional n-gram code space.

Input CSV format:
file_id,window_id,tokenization
SA1,0,134 70 73 ...
SI648,0,134 70 ...
SI648,1,134 331 ...

This script compares adjacent windows within the same file_id:
    window m vs window m+1

Metrics:
    - normalized edit distance
    - normalized edit similarity = 1 - normalized edit distance
    - unigram Jaccard similarity
    - length difference |L_{m+1} - L_m|
    - length ratio min(L1, L2) / max(L1, L2)

Edit-operation analysis:
    - substitutions S
    - insertions I
    - deletions D
    - normalized substitution / insertion / deletion rates
    - thresholded fractions for |ΔL|, edit distance, S, I, D

Optional compositional analysis:
    Raw token sequence:
        [a, b, c, d]

    Bigram sequence:
        [(a,b), (b,c), (c,d)]

    Trigram sequence:
        [(a,b,c), (b,c,d)]

    Each unique bigram/trigram is mapped to a new code ID:
        (a,b)     -> 0
        (b,c)     -> 1
        (a,b,c)   -> 2
        ...

Usage:
    python continuous_sweep_metrics.py \
        --csv_path sweep_tokenizations.csv \
        --pairwise_out pairwise_metrics.csv \
        --file_out per_file_metrics.csv

    python continuous_sweep_metrics.py \
        --csv_path sweep_tokenizations.csv \
        --composition_order both \
        --pairwise_out pairwise_ngram_metrics.csv \
        --file_out per_file_ngram_metrics.csv \
        --ngram_vocab_out ngram_vocab.csv
"""

import argparse
import csv
from collections import defaultdict
from statistics import mean, pstdev, median
from typing import List, Dict, Any, Tuple, Optional


# Thresholds used for distributional summaries.
DEFAULT_COUNT_THRESHOLDS = [0, 1, 2, 3, 5, 10, 20]


def parse_tokenization(s: str) -> List[int]:
    """
    Converts a whitespace-separated token string into a list of ints.
    """
    s = s.strip()
    if not s:
        return []
    return [int(x) for x in s.split()]


# ============================================================
# Edit distance and edit-operation decomposition
# ============================================================

def edit_distance(a: List[int], b: List[int]) -> int:
    """
    Standard Levenshtein edit distance between two token sequences.
    Uses O(min(len(a), len(b))) memory.

    This is kept for compatibility, but pairwise analysis below uses
    edit_distance_with_ops so that the distance and edit-operation counts
    are computed together.
    """
    n, m = len(a), len(b)

    if n == 0:
        return m
    if m == 0:
        return n

    if m > n:
        a, b = b, a
        n, m = m, n

    prev = list(range(m + 1))

    for i in range(1, n + 1):
        cur = [i] + [0] * m
        ai = a[i - 1]

        for j in range(1, m + 1):
            cost = 0 if ai == b[j - 1] else 1

            cur[j] = min(
                prev[j] + 1,          # deletion
                cur[j - 1] + 1,       # insertion
                prev[j - 1] + cost,   # substitution / match
            )

        prev = cur

    return prev[m]


def _better_edit_state(
    cand: Tuple[int, int, int, int],
    best: Optional[Tuple[int, int, int, int]],
) -> Tuple[int, int, int, int]:
    """
    Selects one deterministic optimal edit state.

    State format:
        (total_cost, substitutions, insertions, deletions)

    Primary criterion:
        minimize total edit cost.

    Tie-breaking:
        among equal-cost scripts, prefer fewer insertion/deletion operations,
        then fewer substitutions, then fewer insertions.

    Rationale:
        The continuous-sweep analysis uses insertions/deletions as evidence of
        token-rate or segmentation change. When multiple optimal scripts exist,
        this tie-break avoids over-attributing a pure relabeling event to a
        compensating insertion/deletion pair.
    """
    if best is None:
        return cand

    c_cost, c_sub, c_ins, c_del = cand
    b_cost, b_sub, b_ins, b_del = best

    c_indel = c_ins + c_del
    b_indel = b_ins + b_del

    c_key = (c_cost, c_indel, c_sub, c_ins, c_del)
    b_key = (b_cost, b_indel, b_sub, b_ins, b_del)

    return cand if c_key < b_key else best


def edit_distance_with_ops(a: List[int], b: List[int]) -> Tuple[int, int, int, int]:
    """
    Computes Levenshtein edit distance and a deterministic optimal edit-operation
    decomposition.

    Returns:
        edit_distance, substitutions, insertions, deletions

    Interpretation:
        - substitutions: token identity changes under the optimal alignment
        - insertions: tokens inserted into sequence a to obtain b
        - deletions: tokens deleted from sequence a to obtain b

    Note:
        If multiple optimal edit scripts exist, the deterministic tie-break in
        _better_edit_state is used. This makes the operation statistics stable
        across runs.
    """
    n, m = len(a), len(b)

    # DP state:
    #   (cost, substitutions, insertions, deletions)
    prev = [(j, 0, j, 0) for j in range(m + 1)]

    for i in range(1, n + 1):
        cur = [(i, 0, 0, i)] + [None] * m
        ai = a[i - 1]

        for j in range(1, m + 1):
            bj = b[j - 1]

            # Delete ai from a.
            p_cost, p_sub, p_ins, p_del = prev[j]
            delete_state = (p_cost + 1, p_sub, p_ins, p_del + 1)

            # Insert bj into a.
            l_cost, l_sub, l_ins, l_del = cur[j - 1]
            insert_state = (l_cost + 1, l_sub, l_ins + 1, l_del)

            # Match or substitute ai -> bj.
            d_cost, d_sub, d_ins, d_del = prev[j - 1]
            if ai == bj:
                diag_state = (d_cost, d_sub, d_ins, d_del)
            else:
                diag_state = (d_cost + 1, d_sub + 1, d_ins, d_del)

            best = None
            best = _better_edit_state(delete_state, best)
            best = _better_edit_state(insert_state, best)
            best = _better_edit_state(diag_state, best)

            cur[j] = best

        prev = cur

    ed, subs, ins, dels = prev[m]
    return ed, subs, ins, dels


def normalized_edit_distance(a: List[int], b: List[int]) -> float:
    """
    Normalized edit distance in [0, 1], using max sequence length.
    """
    denom = max(len(a), len(b))

    if denom == 0:
        return 0.0

    return edit_distance(a, b) / denom


def normalized_edit_similarity(a: List[int], b: List[int]) -> float:
    """
    Normalized edit similarity in [0, 1].
    Higher is better.
    """
    return 1.0 - normalized_edit_distance(a, b)


def unigram_jaccard(a: List[int], b: List[int], deduplicate: bool = True) -> float:
    """
    Jaccard similarity over the current token space.

    Important:
        If compositional mode is used, these are no longer original unigrams.
        They are compositional n-gram code IDs.
    """
    if len(a) == 0 and len(b) == 0:
        return 1.0

    if deduplicate:
        sa, sb = set(a), set(b)
        if not sa and not sb:
            return 1.0
        return len(sa & sb) / len(sa | sb)

    from collections import Counter

    ca, cb = Counter(a), Counter(b)
    keys = set(ca.keys()) | set(cb.keys())

    inter = sum(min(ca[k], cb[k]) for k in keys)
    union = sum(max(ca[k], cb[k]) for k in keys)

    if union == 0:
        return 1.0

    return inter / union


def safe_mean(xs: List[float]) -> Optional[float]:
    return mean(xs) if xs else None


def safe_std(xs: List[float]) -> Optional[float]:
    return pstdev(xs) if len(xs) > 1 else 0.0 if len(xs) == 1 else None


def safe_median(xs: List[float]) -> Optional[float]:
    return median(xs) if xs else None


def safe_fraction_le(xs: List[float], threshold: float) -> Optional[float]:
    """
    Fraction of values <= threshold.
    """
    if not xs:
        return None
    return sum(1 for x in xs if x <= threshold) / len(xs)


def add_threshold_fractions(
    out: Dict[str, Any],
    values: List[float],
    prefix: str,
    thresholds: List[int] = DEFAULT_COUNT_THRESHOLDS,
) -> None:
    """
    Adds thresholded fraction columns to a summary dictionary.

    Example:
        prefix='delta_len'
        adds:
            frac_delta_len_le_0
            frac_delta_len_le_1
            ...
    """
    for k in thresholds:
        out[f"frac_{prefix}_le_{k}"] = safe_fraction_le(values, k)


def add_basic_stats(
    out: Dict[str, Any],
    values: List[float],
    prefix: str,
) -> None:
    """
    Adds mean/std/median columns to a summary dictionary.

    Example:
        prefix='substitutions'
        adds:
            mean_substitutions
            std_substitutions
            median_substitutions
    """
    out[f"mean_{prefix}"] = safe_mean(values)
    out[f"std_{prefix}"] = safe_std(values)
    out[f"median_{prefix}"] = safe_median(values)


# ============================================================
# Compositional code-space utilities
# ============================================================

def extract_ngrams(tokens: List[int], n: int) -> List[Tuple[int, ...]]:
    """
    Extracts overlapping n-grams from a token sequence.

    Example:
        tokens = [1, 2, 3, 4], n = 2
        returns [(1,2), (2,3), (3,4)]
    """
    if n <= 0:
        raise ValueError("n must be positive.")

    if len(tokens) < n:
        return []

    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def extract_ngrams2(tokens: List[int], n: int) -> List[Tuple[int, ...]]:
    """
    Extracts non-overlapping n-gram chunks from a token sequence.

    Example:
        tokens = [1, 2, 3, 4, 5], n = 2
        returns [(1,2), (3,4)]

        tokens = [1, 2, 3, 4, 5, 6, 7], n = 3
        returns [(1,2,3), (4,5,6)]

    Incomplete trailing chunks are dropped by default.
    This makes each n-gram act like a compressed symbolic unit.

    Note:
        This function is not used by default. The main compositional analysis
        uses overlapping n-grams through extract_ngrams.
    """
    if n <= 0:
        raise ValueError("n must be positive.")

    if len(tokens) < n:
        return []

    return [
        tuple(tokens[i:i + n])
        for i in range(0, len(tokens) - n + 1, n)
    ]


def make_compositional_units(
    tokens: List[int],
    composition_order: str,
) -> List[Tuple[int, Tuple[int, ...]]]:
    """
    Converts raw tokens into compositional units.

    Returns typed units:
        (2, (a,b)) for bigrams
        (3, (a,b,c)) for trigrams

    The type prefix prevents accidental collision between:
        bigram (1, 23)
    and
        trigram (1, 2, 3)
    when both are used.
    """
    if composition_order == "none":
        raise ValueError("make_compositional_units should not be called with composition_order='none'.")

    units = []

    if composition_order in {"2", "both"}:
        units.extend((2, ng) for ng in extract_ngrams(tokens, 2))

    if composition_order in {"3", "both"}:
        units.extend((3, ng) for ng in extract_ngrams(tokens, 3))

    return units


def build_ngram_vocab(
    grouped: Dict[str, List[Dict[str, Any]]],
    composition_order: str,
) -> Dict[Tuple[int, Tuple[int, ...]], int]:
    """
    Builds a global hash-set vocabulary over all bigrams/trigrams in the dataset,
    then maps each unique n-gram to a new code ID from 0 to V-1.
    """
    vocab_set = set()

    for rows in grouped.values():
        for row in rows:
            tokens = row["raw_tokens"]
            units = make_compositional_units(tokens, composition_order)
            vocab_set.update(units)

    # Deterministic ordering for reproducibility.
    vocab_items = sorted(vocab_set, key=lambda x: (x[0], x[1]))

    return {unit: idx for idx, unit in enumerate(vocab_items)}


def convert_grouped_to_compositional_space(
    grouped: Dict[str, List[Dict[str, Any]]],
    composition_order: str,
) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[Tuple[int, Tuple[int, ...]], int]]:
    """
    Converts every raw token sequence to the new compositional code space.

    For example:
        raw tokens: [10, 20, 30]
        bigrams:    [(10,20), (20,30)]
        new IDs:    [5, 12]
    """
    ngram_vocab = build_ngram_vocab(grouped, composition_order)

    converted = defaultdict(list)

    for file_id, rows in grouped.items():
        for row in rows:
            raw_tokens = row["raw_tokens"]
            units = make_compositional_units(raw_tokens, composition_order)
            new_tokens = [ngram_vocab[u] for u in units]

            converted[file_id].append(
                {
                    "file_id": row["file_id"],
                    "window_id": row["window_id"],
                    "raw_tokens": raw_tokens,
                    "tokens": new_tokens,
                    "length": len(new_tokens),
                    "raw_length": len(raw_tokens),
                }
            )

    for file_id in converted:
        converted[file_id].sort(key=lambda x: x["window_id"])

    return dict(converted), ngram_vocab


def write_ngram_vocab(
    path: str,
    ngram_vocab: Dict[Tuple[int, Tuple[int, ...]], int],
) -> None:
    """
    Writes the compositional vocabulary to CSV.

    Columns:
        code_id
        ngram_order
        ngram
    """
    rows = []

    inverse = sorted(ngram_vocab.items(), key=lambda x: x[1])

    for (order, ng), code_id in inverse:
        rows.append(
            {
                "code_id": code_id,
                "ngram_order": order,
                "ngram": " ".join(str(x) for x in ng),
            }
        )

    write_csv(path, rows)


# ============================================================
# Main analysis
# ============================================================

def read_sweep_csv(csv_path: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Reads CSV and groups rows by file_id.

    Each row contains:
        file_id
        window_id
        raw_tokens
        tokens

    Initially:
        raw_tokens == tokens

    If compositional mode is enabled later:
        raw_tokens stays original
        tokens becomes compositional-code sequence
    """
    grouped = defaultdict(list)

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)

        required = {"file_id", "window_id", "tokenization"}
        missing = required - set(reader.fieldnames or [])

        if missing:
            raise ValueError(f"CSV is missing required columns: {missing}")

        for row in reader:
            file_id = row["file_id"]
            window_id = int(row["window_id"])
            tokens = parse_tokenization(row["tokenization"])

            grouped[file_id].append(
                {
                    "file_id": file_id,
                    "window_id": window_id,
                    "raw_tokens": tokens,
                    "tokens": tokens,
                    "length": len(tokens),
                    "raw_length": len(tokens),
                }
            )

    for file_id in grouped:
        grouped[file_id].sort(key=lambda x: x["window_id"])

    return dict(grouped)


def compute_pairwise_metrics(
    grouped: Dict[str, List[Dict[str, Any]]],
    dedup_jaccard: bool = True,
    require_consecutive: bool = True,
    token_space_name: str = "raw",
) -> List[Dict[str, Any]]:
    """
    Computes metrics for adjacent windows in each file.

    If token_space_name='raw':
        metrics are over original token IDs.

    If token_space_name is compositional:
        metrics are over new n-gram code IDs.

    Edit-operation counts are therefore token-space specific:
        raw                  -> edits over original decoded tokens
        bigram_code_space    -> edits over bigram symbolic units
        trigram_code_space   -> edits over trigram symbolic units
    """
    pairwise_rows = []

    for file_id, rows in grouped.items():
        if len(rows) < 2:
            continue

        for r1, r2 in zip(rows[:-1], rows[1:]):
            w1 = r1["window_id"]
            w2 = r2["window_id"]

            if require_consecutive and (w2 != w1 + 1):
                continue

            t1 = r1["tokens"]
            t2 = r2["tokens"]

            raw_l1 = r1.get("raw_length", len(r1.get("raw_tokens", [])))
            raw_l2 = r2.get("raw_length", len(r2.get("raw_tokens", [])))

            l1 = len(t1)
            l2 = len(t2)
            denom = max(l1, l2)

            ed, substitutions, insertions, deletions = edit_distance_with_ops(t1, t2)

            ned = ed / denom if denom > 0 else 0.0
            nes = 1.0 - ned
            jac = unigram_jaccard(t1, t2, deduplicate=dedup_jaccard)

            delta_l = abs(l2 - l1)

            if denom == 0:
                length_ratio = 1.0
                substitution_rate = 0.0
                insertion_rate = 0.0
                deletion_rate = 0.0
            else:
                length_ratio = min(l1, l2) / denom
                substitution_rate = substitutions / denom
                insertion_rate = insertions / denom
                deletion_rate = deletions / denom

            pairwise_rows.append(
                {
                    "token_space": token_space_name,
                    "file_id": file_id,
                    "window_id_1": w1,
                    "window_id_2": w2,

                    "raw_len_1": raw_l1,
                    "raw_len_2": raw_l2,
                    "raw_delta_len": abs(raw_l2 - raw_l1),

                    "len_1": l1,
                    "len_2": l2,
                    "delta_len": delta_l,
                    "length_ratio": length_ratio,

                    "edit_distance": ed,
                    "norm_edit_distance": ned,
                    "norm_edit_similarity": nes,
                    "jaccard": jac,

                    "substitution_count": substitutions,
                    "insertion_count": insertions,
                    "deletion_count": deletions,
                    "edit_ops_total": substitutions + insertions + deletions,

                    "substitution_rate": substitution_rate,
                    "insertion_rate": insertion_rate,
                    "deletion_rate": deletion_rate,
                }
            )

    return pairwise_rows


def summarize_by_file(pairwise_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Computes per-file summary metrics.
    """
    grouped = defaultdict(list)

    for row in pairwise_rows:
        grouped[row["file_id"]].append(row)

    summaries = []

    for file_id, rows in grouped.items():
        sim = [r["norm_edit_similarity"] for r in rows]
        ned = [r["norm_edit_distance"] for r in rows]
        jac = [r["jaccard"] for r in rows]
        dlen = [r["delta_len"] for r in rows]
        raw_dlen = [r["raw_delta_len"] for r in rows]
        lratio = [r["length_ratio"] for r in rows]

        ed = [r["edit_distance"] for r in rows]
        subs = [r["substitution_count"] for r in rows]
        ins = [r["insertion_count"] for r in rows]
        dels = [r["deletion_count"] for r in rows]
        ops_total = [r["edit_ops_total"] for r in rows]

        sub_rate = [r["substitution_rate"] for r in rows]
        ins_rate = [r["insertion_rate"] for r in rows]
        del_rate = [r["deletion_rate"] for r in rows]

        out = {
            "token_space": rows[0]["token_space"],
            "file_id": file_id,
            "num_adjacent_pairs": len(rows),
        }

        add_basic_stats(out, sim, "norm_edit_similarity")
        add_basic_stats(out, ned, "norm_edit_distance")
        add_basic_stats(out, jac, "jaccard")

        add_basic_stats(out, dlen, "delta_len")
        add_basic_stats(out, raw_dlen, "raw_delta_len")
        add_basic_stats(out, lratio, "length_ratio")

        add_basic_stats(out, ed, "edit_distance")
        add_basic_stats(out, subs, "substitutions")
        add_basic_stats(out, ins, "insertions")
        add_basic_stats(out, dels, "deletions")
        add_basic_stats(out, ops_total, "edit_ops_total")

        add_basic_stats(out, sub_rate, "substitution_rate")
        add_basic_stats(out, ins_rate, "insertion_rate")
        add_basic_stats(out, del_rate, "deletion_rate")

        add_threshold_fractions(out, dlen, "delta_len")
        add_threshold_fractions(out, ed, "edit_distance")
        add_threshold_fractions(out, subs, "substitutions")
        add_threshold_fractions(out, ins, "insertions")
        add_threshold_fractions(out, dels, "deletions")
        add_threshold_fractions(out, ops_total, "edit_ops_total")

        summaries.append(out)

    summaries.sort(key=lambda x: x["file_id"])
    return summaries


def summarize_global(pairwise_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Computes global summary across all adjacent pairs.
    """
    token_space = pairwise_rows[0]["token_space"] if pairwise_rows else "unknown"

    out = {
        "token_space": token_space,
        "num_adjacent_pairs": len(pairwise_rows),
    }

    if not pairwise_rows:
        return out

    sim = [r["norm_edit_similarity"] for r in pairwise_rows]
    ned = [r["norm_edit_distance"] for r in pairwise_rows]
    jac = [r["jaccard"] for r in pairwise_rows]
    dlen = [r["delta_len"] for r in pairwise_rows]
    raw_dlen = [r["raw_delta_len"] for r in pairwise_rows]
    lratio = [r["length_ratio"] for r in pairwise_rows]

    ed = [r["edit_distance"] for r in pairwise_rows]
    subs = [r["substitution_count"] for r in pairwise_rows]
    ins = [r["insertion_count"] for r in pairwise_rows]
    dels = [r["deletion_count"] for r in pairwise_rows]
    ops_total = [r["edit_ops_total"] for r in pairwise_rows]

    sub_rate = [r["substitution_rate"] for r in pairwise_rows]
    ins_rate = [r["insertion_rate"] for r in pairwise_rows]
    del_rate = [r["deletion_rate"] for r in pairwise_rows]

    add_basic_stats(out, sim, "norm_edit_similarity")
    add_basic_stats(out, ned, "norm_edit_distance")
    add_basic_stats(out, jac, "jaccard")

    add_basic_stats(out, dlen, "delta_len")
    add_basic_stats(out, raw_dlen, "raw_delta_len")
    add_basic_stats(out, lratio, "length_ratio")

    add_basic_stats(out, ed, "edit_distance")
    add_basic_stats(out, subs, "substitutions")
    add_basic_stats(out, ins, "insertions")
    add_basic_stats(out, dels, "deletions")
    add_basic_stats(out, ops_total, "edit_ops_total")

    add_basic_stats(out, sub_rate, "substitution_rate")
    add_basic_stats(out, ins_rate, "insertion_rate")
    add_basic_stats(out, del_rate, "deletion_rate")

    add_threshold_fractions(out, dlen, "delta_len")
    add_threshold_fractions(out, ed, "edit_distance")
    add_threshold_fractions(out, subs, "substitutions")
    add_threshold_fractions(out, ins, "insertions")
    add_threshold_fractions(out, dels, "deletions")
    add_threshold_fractions(out, ops_total, "edit_ops_total")

    return out


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    """
    Writes list of dictionaries to CSV.
    """
    if not rows:
        print(f"[Warning] No rows to write for {path}")
        return

    fieldnames = list(rows[0].keys())

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_threshold_block(
    global_summary: Dict[str, Any],
    prefix: str,
    label: str,
    thresholds: List[int] = DEFAULT_COUNT_THRESHOLDS,
) -> None:
    """
    Prints thresholded fractions from the global summary.
    """
    vals = []
    for k in thresholds:
        key = f"frac_{prefix}_le_{k}"
        if key in global_summary and global_summary[key] is not None:
            vals.append(f"≤{k}: {global_summary[key]:.4f}")

    if vals:
        print(f"{label:<34s}: " + " | ".join(vals))


def print_summary(global_summary: Dict[str, Any]) -> None:
    """
    Pretty-prints global metrics.
    """
    print("\n========== Global Continuous-Sweep Summary ==========")
    print(f"Token space                    : {global_summary['token_space']}")
    print(f"Adjacent pairs                 : {global_summary['num_adjacent_pairs']}")

    if global_summary["num_adjacent_pairs"] == 0:
        print("[Warning] No adjacent pairs found.")
        print("====================================================\n")
        return

    print(
        f"Norm edit similarity            : "
        f"{global_summary['mean_norm_edit_similarity']:.4f} "
        f"± {global_summary['std_norm_edit_similarity']:.4f} "
        f"(median {global_summary['median_norm_edit_similarity']:.4f})"
    )

    print(
        f"Norm edit distance              : "
        f"{global_summary['mean_norm_edit_distance']:.4f} "
        f"± {global_summary['std_norm_edit_distance']:.4f} "
        f"(median {global_summary['median_norm_edit_distance']:.4f})"
    )

    print(
        f"Jaccard                         : "
        f"{global_summary['mean_jaccard']:.4f} "
        f"± {global_summary['std_jaccard']:.4f} "
        f"(median {global_summary['median_jaccard']:.4f})"
    )

    print(
        f"Adjacent length change |ΔL|     : "
        f"{global_summary['mean_delta_len']:.4f} "
        f"± {global_summary['std_delta_len']:.4f} "
        f"(median {global_summary['median_delta_len']:.4f})"
    )

    print(
        f"Adjacent length ratio           : "
        f"{global_summary['mean_length_ratio']:.4f} "
        f"± {global_summary['std_length_ratio']:.4f} "
        f"(median {global_summary['median_length_ratio']:.4f})"
    )

    print("\n---------- Edit-operation decomposition ----------")

    print(
        f"Substitutions S                 : "
        f"{global_summary['mean_substitutions']:.4f} "
        f"± {global_summary['std_substitutions']:.4f} "
        f"(median {global_summary['median_substitutions']:.4f})"
    )

    print(
        f"Insertions I                    : "
        f"{global_summary['mean_insertions']:.4f} "
        f"± {global_summary['std_insertions']:.4f} "
        f"(median {global_summary['median_insertions']:.4f})"
    )

    print(
        f"Deletions D                     : "
        f"{global_summary['mean_deletions']:.4f} "
        f"± {global_summary['std_deletions']:.4f} "
        f"(median {global_summary['median_deletions']:.4f})"
    )

    print(
        f"Edit ops total S+I+D            : "
        f"{global_summary['mean_edit_ops_total']:.4f} "
        f"± {global_summary['std_edit_ops_total']:.4f} "
        f"(median {global_summary['median_edit_ops_total']:.4f})"
    )

    print(
        f"Substitution rate               : "
        f"{global_summary['mean_substitution_rate']:.4f} "
        f"± {global_summary['std_substitution_rate']:.4f} "
        f"(median {global_summary['median_substitution_rate']:.4f})"
    )

    print(
        f"Insertion rate                  : "
        f"{global_summary['mean_insertion_rate']:.4f} "
        f"± {global_summary['std_insertion_rate']:.4f} "
        f"(median {global_summary['median_insertion_rate']:.4f})"
    )

    print(
        f"Deletion rate                   : "
        f"{global_summary['mean_deletion_rate']:.4f} "
        f"± {global_summary['std_deletion_rate']:.4f} "
        f"(median {global_summary['median_deletion_rate']:.4f})"
    )

    print("\n---------- Thresholded distribution summaries ----------")
    print_threshold_block(global_summary, "delta_len", "Fraction |ΔL|")
    print_threshold_block(global_summary, "edit_distance", "Fraction edit distance")
    print_threshold_block(global_summary, "substitutions", "Fraction substitutions")
    print_threshold_block(global_summary, "insertions", "Fraction insertions")
    print_threshold_block(global_summary, "deletions", "Fraction deletions")
    print_threshold_block(global_summary, "edit_ops_total", "Fraction S+I+D")

    print("====================================================\n")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--csv_path",
        type=str,
        required=True,
        help="Path to continuous-sweep tokenization CSV.",
    )

    parser.add_argument(
        "--pairwise_out",
        type=str,
        default=None,
        help="Optional path to save adjacent-pair metrics CSV.",
    )

    parser.add_argument(
        "--file_out",
        type=str,
        default=None,
        help="Optional path to save per-file summary metrics CSV.",
    )

    parser.add_argument(
        "--global_out",
        type=str,
        default=None,
        help="Optional path to save one-row global summary metrics CSV.",
    )

    parser.add_argument(
        "--ngram_vocab_out",
        type=str,
        default=None,
        help="Optional path to save the learned bigram/trigram code vocabulary.",
    )

    parser.add_argument(
        "--composition_order",
        type=str,
        default="none",
        choices=["none", "2", "3", "both"],
        help=(
            "Use raw tokens or convert to compositional n-gram code space. "
            "'2' = bigrams, '3' = trigrams, 'both' = bigrams + trigrams."
        ),
    )

    parser.add_argument(
        "--multiset_jaccard",
        action="store_true",
        help="Use multiset Jaccard instead of deduplicated set Jaccard.",
    )

    parser.add_argument(
        "--allow_nonconsecutive",
        action="store_true",
        help="Compare adjacent sorted rows even if window IDs are not consecutive.",
    )

    args = parser.parse_args()

    grouped = read_sweep_csv(args.csv_path)

    ngram_vocab = None

    if args.composition_order == "none":
        token_space_name = "raw"
    else:
        grouped, ngram_vocab = convert_grouped_to_compositional_space(
            grouped,
            composition_order=args.composition_order,
        )

        if args.composition_order == "2":
            token_space_name = "bigram_code_space"
        elif args.composition_order == "3":
            token_space_name = "trigram_code_space"
        else:
            token_space_name = "bigram_trigram_code_space"

        print("\n========== Compositional Vocabulary ==========")
        print(f"Composition order               : {args.composition_order}")
        print(f"Unique compositional codes       : {len(ngram_vocab)}")
        print("==============================================\n")

    pairwise_rows = compute_pairwise_metrics(
        grouped,
        dedup_jaccard=not args.multiset_jaccard,
        require_consecutive=not args.allow_nonconsecutive,
        token_space_name=token_space_name,
    )

    file_summaries = summarize_by_file(pairwise_rows)
    global_summary = summarize_global(pairwise_rows)

    print_summary(global_summary)

    print("Per-file summary:")
    for row in file_summaries:
        print(
            f"{row['file_id']:>12s} | "
            f"pairs={row['num_adjacent_pairs']:>3d} | "
            f"Sim_edit={row['mean_norm_edit_similarity']:.4f} | "
            f"Jaccard={row['mean_jaccard']:.4f} | "
            f"|ΔL|={row['mean_delta_len']:.4f} | "
            f"S={row['mean_substitutions']:.2f} | "
            f"I={row['mean_insertions']:.2f} | "
            f"D={row['mean_deletions']:.2f}"
        )

    if args.pairwise_out is not None:
        write_csv(args.pairwise_out, pairwise_rows)
        print(f"\nSaved pairwise metrics to: {args.pairwise_out}")

    if args.file_out is not None:
        write_csv(args.file_out, file_summaries)
        print(f"Saved per-file metrics to: {args.file_out}")

    if args.global_out is not None:
        write_csv(args.global_out, [global_summary])
        print(f"Saved global metrics to: {args.global_out}")

    if args.ngram_vocab_out is not None:
        if ngram_vocab is None:
            print("[Warning] --ngram_vocab_out was given, but composition_order='none'. No vocab saved.")
        else:
            write_ngram_vocab(args.ngram_vocab_out, ngram_vocab)
            print(f"Saved compositional n-gram vocabulary to: {args.ngram_vocab_out}")


if __name__ == "__main__":
    main()