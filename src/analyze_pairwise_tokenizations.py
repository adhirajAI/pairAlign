#!/usr/bin/env python3
"""
analyze_pairwise_tokenizations.py

Reproduce paired-tokenization consistency and collapse/collision metrics from
one or more CSV run files.

Expected CSV columns by default:
    file_id, anc_tokenization, pos_tokenization

Example:
    python analyze_pairwise_tokenizations.py run1.csv run2.csv run3.csv

The script reports:
  1. Per-run metrics.
  2. Across-run mean and sample standard deviation (ddof=1).
  3. Counts useful for auditing exact-match and exact-collision rates.

Definitions
-----------
For paired token sequences A_i and P_i:

Set Jaccard similarity:
    |set(A_i) ∩ set(P_i)| / |set(A_i) ∪ set(P_i)|

Normalized edit similarity:
    1 - ED(A_i, P_i) / max(|A_i|, |P_i|)

Exact-match rate:
    (1/N) * sum_i 1[A_i == P_i]

Mean sequence length:
    mean over all anchor and positive sequence lengths:
    (sum_i |A_i| + sum_i |P_i|) / (2N)

Active vocabulary:
    number of distinct token IDs appearing in either anchor or positive stream.

Low-diversity sequence:
    |unique tokens in U| / |U| <= threshold
    (default threshold = 0.2)

Collapsed-pair rate:
    fraction of paired examples for which the anchor OR positive sequence is
    low-diversity.

Within-stream exact-collision rate:
    1 - N_unique / N
    = (N - N_unique) / N

This collision definition counts redundant sequence realizations beyond the
number of unique sequences. It is NOT the fraction of examples belonging to
any duplicate group and is NOT the pairwise probability of equality.

Notes
-----
- Sequences are analyzed exactly as stored in the CSV; the script does not add
  or remove EOS or any other token.
- Across-run variability is computed over run-level aggregate metrics.
- With >=2 runs, the reported standard deviation is the sample standard
  deviation (ddof=1), matching the reported LibriSpeech analysis.
- With a single run, across-run SD is reported as 0 because there is no
  across-run variability to estimate from the provided inputs.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

_INTEGER_RE = re.compile(r"-?\d+")


def parse_token_sequence(value) -> Tuple[int, ...]:
    """Parse a token sequence stored as whitespace/comma/bracket separated IDs."""
    if pd.isna(value):
        return tuple()

    text = str(value).strip()
    if not text:
        return tuple()

    # Works for:
    #   "1 2 3"
    #   "1,2,3"
    #   "[1, 2, 3]"
    return tuple(int(x) for x in _INTEGER_RE.findall(text))


# ---------------------------------------------------------------------------
# Sequence metrics
# ---------------------------------------------------------------------------

def levenshtein_distance(a: Sequence[int], b: Sequence[int]) -> int:
    """
    Exact Levenshtein distance with insertion/deletion/substitution cost 1.

    Uses O(min(len(a), len(b))) memory and has no external dependency.
    """
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    # Keep b as the shorter sequence for lower memory use.
    if len(a) < len(b):
        a, b = b, a

    previous = list(range(len(b) + 1))

    for i, token_a in enumerate(a, start=1):
        current = [i]
        for j, token_b in enumerate(b, start=1):
            insertion = current[j - 1] + 1
            deletion = previous[j] + 1
            substitution = previous[j - 1] + (token_a != token_b)
            current.append(min(insertion, deletion, substitution))
        previous = current

    return previous[-1]


def jaccard_similarity(a: Sequence[int], b: Sequence[int]) -> float:
    """Set-based Jaccard similarity."""
    sa = set(a)
    sb = set(b)
    union = sa | sb
    if not union:
        return 1.0
    return len(sa & sb) / len(union)


def normalized_edit_similarity(a: Sequence[int], b: Sequence[int]) -> float:
    """1 - Levenshtein distance / max sequence length."""
    denom = max(len(a), len(b))
    if denom == 0:
        return 1.0
    return 1.0 - (levenshtein_distance(a, b) / denom)


def is_low_diversity(seq: Sequence[int], threshold: float) -> bool:
    """True when unique-token ratio <= threshold."""
    if len(seq) == 0:
        # Empty decoded sequences are treated as degenerate.
        return True
    return (len(set(seq)) / len(seq)) <= threshold


def exact_collision_rate(sequences: Sequence[Tuple[int, ...]]) -> float:
    """
    Fraction of redundant realizations under exact sequence equality:
        (N - N_unique) / N.
    """
    n = len(sequences)
    if n == 0:
        return float("nan")
    n_unique = len(set(sequences))
    return (n - n_unique) / n


# ---------------------------------------------------------------------------
# Run analysis
# ---------------------------------------------------------------------------

METRIC_ORDER = [
    "jaccard_similarity",
    "edit_similarity",
    "exact_match_rate",
    "mean_sequence_length",
    "active_vocabulary",
    "anchor_low_diversity_rate",
    "positive_low_diversity_rate",
    "collapsed_pair_rate",
    "anchor_exact_collision_rate",
    "positive_exact_collision_rate",
]


def analyze_run(
    csv_path: Path,
    anc_col: str,
    pos_col: str,
    id_col: str,
    low_diversity_threshold: float,
) -> Tuple[Dict[str, float], Dict[str, int]]:
    df = pd.read_csv(csv_path)

    required = [anc_col, pos_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"{csv_path}: missing required column(s): {', '.join(missing)}. "
            f"Available columns: {', '.join(map(str, df.columns))}"
        )

    n = len(df)
    if n == 0:
        raise ValueError(f"{csv_path}: CSV contains no examples.")

    anchors = [parse_token_sequence(x) for x in df[anc_col]]
    positives = [parse_token_sequence(x) for x in df[pos_col]]

    jaccards: List[float] = []
    edit_sims: List[float] = []
    exacts: List[bool] = []

    anchor_low: List[bool] = []
    positive_low: List[bool] = []
    collapsed_pairs: List[bool] = []

    vocab = set()

    anchor_token_total = 0
    positive_token_total = 0

    for a, p in zip(anchors, positives):
        jaccards.append(jaccard_similarity(a, p))
        edit_sims.append(normalized_edit_similarity(a, p))
        exacts.append(a == p)

        a_low = is_low_diversity(a, low_diversity_threshold)
        p_low = is_low_diversity(p, low_diversity_threshold)
        anchor_low.append(a_low)
        positive_low.append(p_low)
        collapsed_pairs.append(a_low or p_low)

        vocab.update(a)
        vocab.update(p)

        anchor_token_total += len(a)
        positive_token_total += len(p)

    anchor_unique = len(set(anchors))
    positive_unique = len(set(positives))
    exact_count = int(sum(exacts))

    metrics: Dict[str, float] = {
        "jaccard_similarity": float(np.mean(jaccards)),
        "edit_similarity": float(np.mean(edit_sims)),
        "exact_match_rate": float(np.mean(exacts)),
        "mean_sequence_length": (
            anchor_token_total + positive_token_total
        ) / (2.0 * n),
        "active_vocabulary": float(len(vocab)),
        "anchor_low_diversity_rate": float(np.mean(anchor_low)),
        "positive_low_diversity_rate": float(np.mean(positive_low)),
        "collapsed_pair_rate": float(np.mean(collapsed_pairs)),
        "anchor_exact_collision_rate": (n - anchor_unique) / n,
        "positive_exact_collision_rate": (n - positive_unique) / n,
    }

    counts: Dict[str, int] = {
        "num_examples": n,
        "exact_match_count": exact_count,
        "anchor_total_tokens": anchor_token_total,
        "positive_total_tokens": positive_token_total,
        "anchor_unique_sequences": anchor_unique,
        "positive_unique_sequences": positive_unique,
        "anchor_redundant_sequences": n - anchor_unique,
        "positive_redundant_sequences": n - positive_unique,
        "active_vocabulary": len(vocab),
    }

    if id_col in df.columns:
        counts["unique_file_ids"] = int(df[id_col].nunique(dropna=False))
        counts["duplicate_file_ids"] = int(n - df[id_col].nunique(dropna=False))

    return metrics, counts


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def across_run_summary(
    per_run: Sequence[Dict[str, float]],
) -> Dict[str, Tuple[float, float]]:
    """
    Return metric -> (mean, sample SD across run-level metrics).

    For a single supplied run, SD is set to 0.0 because an across-run SD cannot
    be estimated from one observation.
    """
    summary: Dict[str, Tuple[float, float]] = {}

    for metric in METRIC_ORDER:
        values = np.asarray([run[metric] for run in per_run], dtype=float)
        mean = float(np.mean(values))
        sd = float(np.std(values, ddof=1)) if len(values) >= 2 else 0.0
        summary[metric] = (mean, sd)

    return summary


def fmt(x: float, digits: int = 10) -> str:
    if math.isnan(x):
        return "nan"
    return f"{x:.{digits}g}"


def print_run_report(
    path: Path,
    metrics: Dict[str, float],
    counts: Dict[str, int],
) -> None:
    print(f"\n=== {path.name} ===")
    print(f"N examples                  : {counts['num_examples']}")
    if "unique_file_ids" in counts:
        print(f"Unique file_id values       : {counts['unique_file_ids']}")
        print(f"Duplicate file_id rows      : {counts['duplicate_file_ids']}")

    print(f"Jaccard similarity          : {metrics['jaccard_similarity']:.12f}")
    print(f"Edit similarity             : {metrics['edit_similarity']:.12f}")
    print(
        f"Exact match rate            : {metrics['exact_match_rate']:.12f} "
        f"({counts['exact_match_count']}/{counts['num_examples']})"
    )
    print(f"Mean sequence length        : {metrics['mean_sequence_length']:.12f}")
    print(f"Active vocabulary           : {int(metrics['active_vocabulary'])}")

    print(
        f"Anchor low-diversity rate   : "
        f"{metrics['anchor_low_diversity_rate']:.12f}"
    )
    print(
        f"Positive low-diversity rate : "
        f"{metrics['positive_low_diversity_rate']:.12f}"
    )
    print(
        f"Collapsed-pair rate         : "
        f"{metrics['collapsed_pair_rate']:.12f}"
    )

    print(
        f"Anchor exact-collision rate : "
        f"{metrics['anchor_exact_collision_rate']:.12f} "
        f"({counts['anchor_redundant_sequences']} redundant; "
        f"{counts['anchor_unique_sequences']} unique)"
    )
    print(
        f"Positive exact-collision rate: "
        f"{metrics['positive_exact_collision_rate']:.12f} "
        f"({counts['positive_redundant_sequences']} redundant; "
        f"{counts['positive_unique_sequences']} unique)"
    )


def print_summary(summary: Dict[str, Tuple[float, float]], nruns: int) -> None:
    print("\n=== Across-run aggregate ===")
    print(f"Number of runs: {nruns}")
    print("Mean ± sample SD across run-level aggregate metrics")
    if nruns == 1:
        print("(Only one run supplied; SD displayed as 0.0.)")

    labels = {
        "jaccard_similarity": "Jaccard similarity",
        "edit_similarity": "Edit similarity",
        "exact_match_rate": "Exact match rate",
        "mean_sequence_length": "Mean sequence length",
        "active_vocabulary": "Active vocabulary",
        "anchor_low_diversity_rate": "Anchor low-diversity rate",
        "positive_low_diversity_rate": "Positive low-diversity rate",
        "collapsed_pair_rate": "Collapsed-pair rate",
        "anchor_exact_collision_rate": "Anchor exact-collision rate",
        "positive_exact_collision_rate": "Positive exact-collision rate",
    }

    for metric in METRIC_ORDER:
        mean, sd = summary[metric]
        print(f"{labels[metric]:29s}: {mean:.12g} ± {sd:.12g}")


def save_results(
    output_prefix: Path,
    paths: Sequence[Path],
    per_run: Sequence[Dict[str, float]],
    summary: Dict[str, Tuple[float, float]],
) -> None:
    per_run_rows = []
    for path, metrics in zip(paths, per_run):
        row = {"run_file": str(path)}
        row.update(metrics)
        per_run_rows.append(row)

    per_run_path = output_prefix.with_name(output_prefix.name + "_per_run.csv")
    summary_path = output_prefix.with_name(output_prefix.name + "_summary.csv")

    pd.DataFrame(per_run_rows).to_csv(per_run_path, index=False)

    summary_rows = [
        {
            "metric": metric,
            "mean_across_runs": summary[metric][0],
            "sample_sd_across_runs": summary[metric][1],
        }
        for metric in METRIC_ORDER
    ]
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)

    print(f"\nSaved per-run metrics : {per_run_path}")
    print(f"Saved aggregate metrics: {summary_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compute paired-tokenization consistency and collapse/collision "
            "metrics from any number of run CSV files."
        )
    )
    parser.add_argument(
        "run_files",
        nargs="+",
        type=Path,
        help="One or more CSV run files.",
    )
    parser.add_argument(
        "--anc-col",
        default="anc_tokenization",
        help="Anchor tokenization column (default: anc_tokenization).",
    )
    parser.add_argument(
        "--pos-col",
        default="pos_tokenization",
        help="Positive tokenization column (default: pos_tokenization).",
    )
    parser.add_argument(
        "--id-col",
        default="file_id",
        help="Example ID column used for audit counts (default: file_id).",
    )
    parser.add_argument(
        "--low-diversity-threshold",
        type=float,
        default=0.2,
        help=(
            "Unique-token-ratio threshold for low-diversity sequences "
            "(default: 0.2)."
        ),
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=None,
        help=(
            "Optional prefix for saving '<prefix>_per_run.csv' and "
            "'<prefix>_summary.csv'."
        ),
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()

    if not (0.0 <= args.low_diversity_threshold <= 1.0):
        raise ValueError("--low-diversity-threshold must be in [0, 1].")

    paths = [p.expanduser().resolve() for p in args.run_files]
    for p in paths:
        if not p.is_file():
            raise FileNotFoundError(f"Run file not found: {p}")

    per_run_metrics: List[Dict[str, float]] = []

    for path in paths:
        metrics, counts = analyze_run(
            path,
            anc_col=args.anc_col,
            pos_col=args.pos_col,
            id_col=args.id_col,
            low_diversity_threshold=args.low_diversity_threshold,
        )
        per_run_metrics.append(metrics)
        print_run_report(path, metrics, counts)

    summary = across_run_summary(per_run_metrics)
    print_summary(summary, len(paths))

    if args.output_prefix is not None:
        save_results(
            args.output_prefix.expanduser(),
            paths,
            per_run_metrics,
            summary,
        )

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
