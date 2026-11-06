# geometric_bpe_consistency_timed.py

import os
import csv
import json
import time
import yaml
import argparse
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional

import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
import torchaudio.functional as AF
from torch.utils.data import Dataset, DataLoader
from lightning.pytorch import seed_everything
from tqdm.auto import tqdm

from train import AudioTokenizer
from utils import AudioPairAugDataset


# ================================================================
# General utilities
# ================================================================

def path_to_file_id(path: str) -> str:
    return os.path.splitext(os.path.basename(str(path)))[0]


def sequence_to_string(seq: List[int]) -> str:
    return " ".join(map(str, seq))


def mean_sequence_length(sequences: List[List[int]]) -> float:
    if len(sequences) == 0:
        return 0.0
    return sum(len(x) for x in sequences) / len(sequences)


def cuda_sync(device: torch.device):
    """Synchronize CUDA only when timing CUDA work."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


# ================================================================
# Dataset collate
# ================================================================

def pair_collate(batch):
    first = batch[0]

    if isinstance(first, dict):
        anc_keys = ["anc", "anchor", "anchor_feat", "x"]
        pos_keys = ["pos", "positive", "positive_feat", "aug", "x_aug"]
        id_keys = ["file_id", "path", "audio_path"]

        def find_key(keys):
            for k in keys:
                if k in first:
                    return k
            return None

        anc_key = find_key(anc_keys)
        pos_key = find_key(pos_keys)
        id_key = find_key(id_keys)

        if anc_key is None or pos_key is None:
            raise ValueError(
                f"Could not infer anchor/positive keys: {first.keys()}"
            )

        anc_feats = torch.stack([b[anc_key] for b in batch], dim=0)
        pos_feats = torch.stack([b[pos_key] for b in batch], dim=0)

        if id_key is not None:
            raw_ids = [b[id_key] for b in batch]
            file_ids = [
                path_to_file_id(x)
                if "/" in str(x) or "." in str(x)
                else str(x)
                for x in raw_ids
            ]
        else:
            file_ids = ["unknown"] * len(batch)

        return anc_feats, pos_feats, file_ids

    if isinstance(first, (tuple, list)):
        if len(first) < 2:
            raise ValueError(
                f"Dataset item needs anc and pos. Got: {first}"
            )

        anc_feats = torch.stack([b[0] for b in batch], dim=0)
        pos_feats = torch.stack([b[1] for b in batch], dim=0)

        if len(first) >= 3:
            raw_ids = [b[2] for b in batch]
            file_ids = [
                path_to_file_id(x)
                if "/" in str(x) or "." in str(x)
                else str(x)
                for x in raw_ids
            ]
        else:
            file_ids = ["unknown"] * len(batch)

        return anc_feats, pos_feats, file_ids

    raise ValueError(
        f"Unsupported dataset output type: {type(first)}"
    )


# ================================================================
# Geometric tokenizer handling
# ================================================================

def build_geometric_model(
    ckpt_path: str,
    device_id: int,
):
    model = AudioTokenizer.load_from_pretrained(
        ckpt_path,
        device_id
    )
    return model


def normalize_token_batch(
    tokens: Any,
    batch_size: int,
) -> List[List[int]]:
    """
    Convert AudioTokenizer.tokenize() output to List[List[int]].
    """

    if isinstance(tokens, torch.Tensor):
        tokens = tokens.detach().cpu()

        if tokens.ndim == 1:
            if batch_size != 1:
                raise ValueError(
                    f"1D token tensor but batch_size={batch_size}"
                )
            return [[int(x) for x in tokens.tolist()]]

        if tokens.ndim == 2:
            return [
                [int(x) for x in row.tolist()]
                for row in tokens
            ]

        raise ValueError(
            f"Unexpected token tensor shape: {tokens.shape}"
        )

    if isinstance(tokens, (list, tuple)):
        if batch_size == 1:
            if (
                len(tokens) == 1
                and isinstance(tokens[0], (list, tuple))
            ):
                return [[int(x) for x in tokens[0]]]

            if (
                len(tokens) == 1
                and isinstance(tokens[0], torch.Tensor)
            ):
                return [[
                    int(x)
                    for x in tokens[0]
                    .detach()
                    .cpu()
                    .reshape(-1)
                    .tolist()
                ]]

            if len(tokens) > 0 and isinstance(
                tokens[0], (int, float)
            ):
                return [[int(x) for x in tokens]]

        output = []

        for seq in tokens:
            if isinstance(seq, torch.Tensor):
                seq = (
                    seq.detach()
                    .cpu()
                    .reshape(-1)
                    .tolist()
                )
            output.append([int(x) for x in seq])

        if len(output) != batch_size:
            raise RuntimeError(
                f"Token batch mismatch: "
                f"{len(output)} vs {batch_size}"
            )

        return output

    raise ValueError(
        f"Unsupported token output type: {type(tokens)}"
    )


@torch.no_grad()
def geometric_tokenize(
    model,
    feats: torch.Tensor,
) -> List[List[int]]:
    """
    Stage-I / geometric tokenization.
    """
    tokens = model.tokenize(feats)

    return normalize_token_batch(
        tokens,
        batch_size=feats.shape[0],
    )


# ================================================================
# Integer BPE
# ================================================================

def replace_pair(
    sequence: List[int],
    pair: Tuple[int, int],
    new_token: int,
) -> List[int]:

    if len(sequence) < 2:
        return sequence.copy()

    a, b = pair
    output = []
    i = 0

    while i < len(sequence):
        if (
            i < len(sequence) - 1
            and sequence[i] == a
            and sequence[i + 1] == b
        ):
            output.append(new_token)
            i += 2
        else:
            output.append(sequence[i])
            i += 1

    return output


def pair_statistics(
    corpus: List[List[int]],
) -> Counter:

    counts = Counter()

    for seq in corpus:
        for i in range(len(seq) - 1):
            counts[(seq[i], seq[i + 1])] += 1

    return counts


def learn_bpe(
    corpus: List[List[int]],
    target_mean_length: float,
    min_pair_frequency: int = 2,
    max_merges: int = 10000,
    verbose: bool = True,
) -> Dict:

    corpus = [list(seq) for seq in corpus]

    if len(corpus) == 0:
        raise ValueError("Empty BPE training corpus.")

    original_mean = mean_sequence_length(corpus)

    all_tokens = {
        token
        for seq in corpus
        for token in seq
    }

    if len(all_tokens) == 0:
        raise ValueError("BPE corpus contains no tokens.")

    next_token = max(all_tokens) + 1
    merges = []
    current_mean = original_mean

    print()
    print("=" * 70)
    print("BPE TRAINING")
    print("=" * 70)
    print(f"Original mean length : {original_mean:.4f}")
    print(f"Target mean length   : {target_mean_length:.4f}")
    print(f"Initial vocabulary   : {len(all_tokens)}")
    print()

    if current_mean <= target_mean_length:
        print(
            "Original stream is already at/below target length. "
            "No BPE merges required."
        )

        return {
            "target_mean_length": target_mean_length,
            "original_mean_length": original_mean,
            "achieved_mean_length": current_mean,
            "merges": [],
        }

    iterator = range(max_merges)

    if verbose:
        iterator = tqdm(
            iterator,
            desc="Learning BPE merges",
        )

    for _ in iterator:
        counts = pair_statistics(corpus)

        if len(counts) == 0:
            break

        best_pair, best_frequency = counts.most_common(1)[0]

        if best_frequency < min_pair_frequency:
            print(
                "\nStopping because most frequent pair occurs only "
                f"{best_frequency} times."
            )
            break

        previous_mean = current_mean

        new_corpus = [
            replace_pair(
                seq,
                best_pair,
                next_token,
            )
            for seq in corpus
        ]

        new_mean = mean_sequence_length(new_corpus)

        if new_mean < target_mean_length:
            previous_error = abs(
                previous_mean - target_mean_length
            )
            new_error = abs(
                new_mean - target_mean_length
            )

            if previous_error <= new_error:
                break

        merges.append({
            "left": int(best_pair[0]),
            "right": int(best_pair[1]),
            "new": int(next_token),
            "frequency_at_fit": int(best_frequency),
        })

        corpus = new_corpus
        current_mean = new_mean
        next_token += 1

        if verbose:
            iterator.set_postfix(
                mean_len=f"{current_mean:.3f}",
                merges=len(merges),
                pair_freq=best_frequency,
            )

        if current_mean <= target_mean_length:
            break

    print()
    print(f"Learned merges       : {len(merges)}")
    print(f"Achieved mean length : {current_mean:.4f}")
    print(
        f"Compression ratio    : "
        f"{original_mean / max(current_mean, 1e-8):.4f}x"
    )
    print("=" * 70)
    print()

    return {
        "target_mean_length": target_mean_length,
        "original_mean_length": original_mean,
        "achieved_mean_length": current_mean,
        "merges": merges,
    }


def apply_bpe(
    sequence: List[int],
    bpe_model: Dict,
) -> List[int]:

    output = list(sequence)

    for merge in bpe_model["merges"]:
        pair = (
            int(merge["left"]),
            int(merge["right"]),
        )
        new_token = int(merge["new"])

        output = replace_pair(
            output,
            pair,
            new_token,
        )

    return output


def apply_bpe_batch(
    sequences: List[List[int]],
    bpe_model: Dict,
) -> List[List[int]]:
    return [
        apply_bpe(seq, bpe_model)
        for seq in sequences
    ]


def save_bpe_model(
    bpe_model: Dict,
    path: str,
):
    os.makedirs(
        os.path.dirname(path) or ".",
        exist_ok=True,
    )

    with open(path, "w") as f:
        json.dump(
            bpe_model,
            f,
            indent=2,
        )

    print(f"Saved BPE model: {path}")


def load_bpe_model(
    path: str,
) -> Dict:

    with open(path, "r") as f:
        model = json.load(f)

    print(f"Loaded BPE model: {path}")
    return model


# ================================================================
# Collect BPE fitting corpus
# ================================================================

@torch.no_grad()
def collect_bpe_training_sequences(
    model,
    loader,
    device,
    max_sequences: Optional[int],
):

    model.eval()
    model.to(device)

    available = len(loader.dataset)
    total = (
        available
        if max_sequences is None
        else int(max_sequences)
    )

    if total <= 0:
        raise ValueError(
            "BPE fitting requires at least one sequence."
        )

    if total > available:
        raise ValueError(
            f"Requested {total} BPE fitting sequences, "
            f"but the training dataset contains only {available}."
        )

    if total % 2 != 0:
        raise ValueError(
            "BPE fitting corpus size must be even so clean and "
            "augmented subsets have equal size."
        )

    num_clean = total // 2
    num_augmented = total // 2

    clean_sequences = []
    augmented_sequences = []

    pbar = tqdm(
        loader,
        desc="Collecting BPE training tokens",
    )

    for anc_feats, pos_feats, _ in pbar:
        clean_needed = num_clean - len(clean_sequences)
        augmented_needed = (
            num_augmented - len(augmented_sequences)
        )

        if clean_needed <= 0 and augmented_needed <= 0:
            break

        batch_size = anc_feats.shape[0]
        clean_count = min(batch_size, max(0, clean_needed))
        augmented_start = clean_count
        augmented_count = min(
            batch_size - augmented_start,
            max(0, augmented_needed),
        )

        if clean_count > 0:
            clean_feats = anc_feats[:clean_count].to(
                device,
                non_blocking=True,
            )
            clean_sequences.extend(
                geometric_tokenize(
                    model,
                    clean_feats,
                )
            )

        if augmented_count > 0:
            augmented_feats = pos_feats[
                augmented_start:
                augmented_start + augmented_count
            ].to(
                device,
                non_blocking=True,
            )
            augmented_sequences.extend(
                geometric_tokenize(
                    model,
                    augmented_feats,
                )
            )

        pbar.set_postfix(
            clean=len(clean_sequences),
            augmented=len(augmented_sequences),
        )

    if len(clean_sequences) != num_clean:
        raise RuntimeError(
            f"Collected {len(clean_sequences)} clean sequences; "
            f"expected {num_clean}."
        )

    if len(augmented_sequences) != num_augmented:
        raise RuntimeError(
            f"Collected {len(augmented_sequences)} augmented sequences; "
            f"expected {num_augmented}."
        )

    sequences = clean_sequences + augmented_sequences

    return sequences, {
        "clean": len(clean_sequences),
        "augmented": len(augmented_sequences),
    }


# ================================================================
# Consistency metrics
# ================================================================

def jaccard_similarity(
    a: List[int],
    b: List[int],
) -> float:

    set_a = set(a)
    set_b = set(b)
    union = set_a | set_b

    if len(union) == 0:
        return 1.0

    return len(set_a & set_b) / len(union)


def levenshtein_distance(
    a: List[int],
    b: List[int],
) -> int:

    if len(a) < len(b):
        a, b = b, a

    if len(b) == 0:
        return len(a)

    previous = list(range(len(b) + 1))

    for i, token_a in enumerate(a, start=1):
        current = [i]

        for j, token_b in enumerate(b, start=1):
            insertion = current[j - 1] + 1
            deletion = previous[j] + 1
            substitution = (
                previous[j - 1]
                + (token_a != token_b)
            )

            current.append(
                min(
                    insertion,
                    deletion,
                    substitution,
                )
            )

        previous = current

    return previous[-1]


def edit_similarity(
    a: List[int],
    b: List[int],
) -> float:

    denominator = max(
        len(a),
        len(b),
    )

    if denominator == 0:
        return 1.0

    distance = levenshtein_distance(
        a,
        b,
    )

    return 1.0 - distance / denominator


def calculate_consistency_metrics(
    anchors: List[List[int]],
    positives: List[List[int]],
    segment_sec: float,
) -> Dict[str, float]:

    if len(anchors) != len(positives):
        raise ValueError(
            "Anchor/positive count mismatch."
        )

    n = len(anchors)

    if n == 0:
        raise ValueError(
            "No sequences for evaluation."
        )

    jaccards = []
    edits = []
    exacts = []

    active_tokens = set()
    total_length = 0

    for anc, pos in zip(
        anchors,
        positives,
    ):
        jaccards.append(
            jaccard_similarity(anc, pos)
        )
        edits.append(
            edit_similarity(anc, pos)
        )
        exacts.append(
            float(anc == pos)
        )

        total_length += len(anc)
        total_length += len(pos)

        active_tokens.update(anc)
        active_tokens.update(pos)

    mean_length = total_length / (2 * n)

    return {
        "num_pairs": n,
        "jaccard_similarity":
            sum(jaccards) / n,
        "edit_similarity":
            sum(edits) / n,
        "exact_match_rate":
            sum(exacts) / n,
        "mean_sequence_length":
            mean_length,
        "tokens_per_second":
            mean_length / segment_sec,
        "active_vocabulary_size":
            len(active_tokens),
    }


# ================================================================
# Timing helpers
# ================================================================

def empty_timing_stats() -> Dict[str, float]:
    return {
        "num_archive_segments": 0,
        "geometric_model_wall_s": 0.0,
        "bpe_post_wall_s": 0.0,
        "geometric_plus_bpe_wall_s": 0.0,
        "archive_loop_wall_s": 0.0,
    }


def finalize_timing_stats(
    timing: Dict[str, float],
    segment_sec: float,
) -> Dict[str, float]:
    """
    Compute archive quantities.

    geometric_model_*:
        model.tokenize() only, batch-amortized.

    bpe_post_*:
        CPU BPE transform only.

    geometric_plus_bpe_*:
        model.tokenize() + BPE transform, excluding DataLoader wait
        and host->device transfer.

    archive_loop_*:
        full evaluation-loop archive timing, including DataLoader wait,
        host->device transfer, model tokenization, BPE and Python handling.

    Retrieval-style comparisons should generally use geometric_plus_bpe
    throughput/wall time if the PairAlign archive
    measurement times tokenizer computation rather than feature extraction.
    """

    n = timing["num_archive_segments"]

    if n <= 0:
        return timing

    model_s = timing["geometric_model_wall_s"]
    bpe_s = timing["bpe_post_wall_s"]
    combined_s = timing["geometric_plus_bpe_wall_s"]
    loop_s = timing["archive_loop_wall_s"]

    timing.update({
        "segment_duration_sec": segment_sec,

        "geometric_model_ms_per_segment":
            1000.0 * model_s / n,

        "bpe_post_ms_per_segment":
            1000.0 * bpe_s / n,

        "geometric_plus_bpe_ms_per_segment":
            1000.0 * combined_s / n,

        "geometric_model_segments_per_s":
            n / model_s if model_s > 0 else float("inf"),

        "geometric_plus_bpe_segments_per_s":
            n / combined_s if combined_s > 0 else float("inf"),

        "archive_loop_ms_per_segment":
            1000.0 * loop_s / n,

        "archive_loop_segments_per_s":
            n / loop_s if loop_s > 0 else float("inf"),
    })

    return timing


# ================================================================
# Evaluation + archive timing
# ================================================================

@torch.no_grad()
def evaluate_geometric_and_bpe(
    model,
    loader,
    bpe_model,
    output_csv,
    device,
    segment_sec,
    timing_warmup_batches: int = 5,
):

    model.eval()
    model.to(device)

    os.makedirs(
        os.path.dirname(output_csv) or ".",
        exist_ok=True,
    )

    raw_anc_all = []
    raw_pos_all = []

    bpe_anc_all = []
    bpe_pos_all = []

    global_index = 0
    timing = empty_timing_stats()

    # ------------------------------------------------------------
    # IMPORTANT:
    # Archive timing is based on ANCHORS ONLY.
    #
    # The positive stream is still tokenized for consistency metrics,
    # but its time is NOT included in archive timing.
    #
    # Warm-up is performed BEFORE the first measured archive batch by
    # repeatedly tokenizing the first anchor batch. Therefore every
    # real evaluation/archive segment is included in timing totals.
    # ------------------------------------------------------------

    loop_start_after_warmup = None
    warmup_done = False

    with open(
        output_csv,
        "w",
        newline="",
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "file_id",

            "raw_anc_tokenization",
            "raw_pos_tokenization",

            "bpe_anc_tokenization",
            "bpe_pos_tokenization",

            "raw_anc_length",
            "raw_pos_length",

            "bpe_anc_length",
            "bpe_pos_length",

            # Batch-amortized timing for the ARCHIVE/ANCHOR view.
            "anc_geometric_model_ms_per_segment",
            "anc_bpe_post_ms_per_segment",
            "anc_geometric_plus_bpe_ms_per_segment",

            # All archive rows are timed after one-time warm-up.
            "timing_included",
        ])

        pbar = tqdm(
            enumerate(loader),
            total=len(loader),
            desc="Evaluating geometric+BPE",
        )

        for batch_idx, (
            anc_feats,
            pos_feats,
            file_ids,
        ) in pbar:

            batch_size_actual = anc_feats.shape[0]

            anc_feats = anc_feats.to(
                device,
                non_blocking=True,
            )

            pos_feats = pos_feats.to(
                device,
                non_blocking=True,
            )

            # ----------------------------------------------------
            # One-time warm-up using the first actual archive batch.
            # These extra calls are NOT included in timing totals.
            # The same batch is then processed once more below as a
            # normal measured archive batch, so no dataset segments
            # are discarded from the timing count.
            # ----------------------------------------------------
            if not warmup_done:
                for _ in range(max(0, timing_warmup_batches)):
                    _ = geometric_tokenize(
                        model,
                        anc_feats,
                    )
                    cuda_sync(device)

                warmup_done = True
                cuda_sync(device)
                loop_start_after_warmup = time.perf_counter()

            # ====================================================
            # ARCHIVE / ANCHOR GEOMETRIC TOKENIZATION TIMING
            # ====================================================

            cuda_sync(device)
            t_model_start = time.perf_counter()

            raw_anc = geometric_tokenize(
                model,
                anc_feats,
            )

            cuda_sync(device)
            t_model_end = time.perf_counter()

            anc_model_wall_s = (
                t_model_end - t_model_start
            )

            # ====================================================
            # ARCHIVE / ANCHOR BPE POST-PROCESSING TIMING
            # ====================================================

            t_bpe_start = time.perf_counter()

            bpe_anc = apply_bpe_batch(
                raw_anc,
                bpe_model,
            )

            t_bpe_end = time.perf_counter()

            anc_bpe_wall_s = (
                t_bpe_end - t_bpe_start
            )

            anc_combined_wall_s = (
                anc_model_wall_s
                + anc_bpe_wall_s
            )

            # ====================================================
            # POSITIVE VIEW
            # Required for consistency evaluation.
            # Deliberately not included in archive timing.
            # ====================================================

            raw_pos = geometric_tokenize(
                model,
                pos_feats,
            )

            bpe_pos = apply_bpe_batch(
                raw_pos,
                bpe_model,
            )

            if len(raw_anc) != len(raw_pos):
                raise RuntimeError(
                    "Anchor/positive batch mismatch."
                )

            timing["num_archive_segments"] += (
                batch_size_actual
            )

            timing["geometric_model_wall_s"] += (
                anc_model_wall_s
            )

            timing["bpe_post_wall_s"] += (
                anc_bpe_wall_s
            )

            timing["geometric_plus_bpe_wall_s"] += (
                anc_combined_wall_s
            )

            # Batch-amortized timing. Every row from this batch gets
            # the same amortized per-segment value.
            anc_model_ms_per_segment = (
                1000.0
                * anc_model_wall_s
                / batch_size_actual
            )

            anc_bpe_ms_per_segment = (
                1000.0
                * anc_bpe_wall_s
                / batch_size_actual
            )

            anc_combined_ms_per_segment = (
                1000.0
                * anc_combined_wall_s
                / batch_size_actual
            )

            # ====================================================
            # Store sequences
            # ====================================================

            raw_anc_all.extend(raw_anc)
            raw_pos_all.extend(raw_pos)

            bpe_anc_all.extend(bpe_anc)
            bpe_pos_all.extend(bpe_pos)

            for (
                fid,
                ra,
                rp,
                ba,
                bp,
            ) in zip(
                file_ids,
                raw_anc,
                raw_pos,
                bpe_anc,
                bpe_pos,
            ):

                if fid == "unknown":
                    fid = (
                        f"sample_"
                        f"{global_index:09d}"
                    )

                writer.writerow([
                    fid,

                    sequence_to_string(ra),
                    sequence_to_string(rp),

                    sequence_to_string(ba),
                    sequence_to_string(bp),

                    len(ra),
                    len(rp),

                    len(ba),
                    len(bp),

                    f"{anc_model_ms_per_segment:.9f}",
                    f"{anc_bpe_ms_per_segment:.9f}",
                    f"{anc_combined_ms_per_segment:.9f}",

                    1,
                ])

                global_index += 1

            pbar.set_postfix(
                samples=global_index,
                timed_segments=timing[
                    "num_archive_segments"
                ],
            )

    # Full timed loop wall time after warmup.
    if loop_start_after_warmup is not None:
        cuda_sync(device)
        timing["archive_loop_wall_s"] = (
            time.perf_counter()
            - loop_start_after_warmup
        )

    timing = finalize_timing_stats(
        timing,
        segment_sec=segment_sec,
    )

    raw_metrics = calculate_consistency_metrics(
        raw_anc_all,
        raw_pos_all,
        segment_sec=segment_sec,
    )

    bpe_metrics = calculate_consistency_metrics(
        bpe_anc_all,
        bpe_pos_all,
        segment_sec=segment_sec,
    )

    return (
        raw_metrics,
        bpe_metrics,
        timing,
    )



# ================================================================
# Retrieval-efficiency-style archive timing
# ================================================================
#
# This section deliberately mirrors the timing context used by the
# retrieval-efficiency script that produced the Stage-I / PairAlign
# archive-tokenization table:
#
#   raw waveform DataLoader
#   -> feature extraction
#   -> host-to-device transfer
#   -> model.tokenize()
#   -> token normalization / optional BPE
#
# `wall_clock_sec` surrounds the complete DataLoader iteration.
# DataLoader settings are intentionally:
#       num_workers = 0
#       pin_memory = False
# exactly as in the reference retrieval timing code.
#
# IMPORTANT:
# This is a SEPARATE archive-only benchmark. The consistency analysis
# above remains unchanged and still uses AudioPairAugDataset.
# ================================================================


@dataclass
class TimingArchiveFileInfo:
    path: str
    global_start: int
    global_end: int
    num_samples: int


@dataclass
class TimingSegmentRecord:
    seg_idx: int
    start: int
    end: int
    duration_sec: float


@dataclass
class RetrievalStyleTimingRecord:
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


def read_paths_from_csv(
    csv_path: str,
    path_col: str = "path",
) -> List[str]:
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(
            f"CSV not found: {csv_path}"
        )

    paths = []

    with open(csv_path, "r", newline="") as handle:
        reader = csv.DictReader(handle)

        if (
            reader.fieldnames is None
            or path_col not in reader.fieldnames
        ):
            raise ValueError(
                f"CSV must contain `{path_col}`. "
                f"Found: {reader.fieldnames}"
            )

        for row in reader:
            path = row[path_col].strip()
            if path:
                paths.append(path)

    return paths


def load_audio_mono(
    path: str,
) -> Tuple[torch.Tensor, int]:

    wav, sr = torchaudio.load(path)

    if wav.dtype != torch.float32:
        wav = wav.float()

    if wav.size(0) > 1:
        wav = wav.mean(
            dim=0,
            keepdim=True,
        )

    return wav, sr


def ensure_sr(
    wav: torch.Tensor,
    sr: int,
    target_sr: int,
) -> Tuple[torch.Tensor, int]:

    if sr == target_sr:
        return wav, sr

    return (
        AF.resample(
            wav,
            orig_freq=sr,
            new_freq=target_sr,
        ),
        target_sr,
    )


def build_timing_archive_metadata(
    audio_csv: str,
    fs: int,
    path_col: str,
) -> List[TimingArchiveFileInfo]:
    """
    Mirrors the archive construction in the retrieval-efficiency script,
    except PHN metadata is unnecessary because this function is timing-only.
    """

    paths = [
        p
        for p in read_paths_from_csv(
            audio_csv,
            path_col,
        )
        if os.path.isfile(p)
    ]

    if not paths:
        raise ValueError(
            "No valid audio files found in timing CSV."
        )

    file_infos = []
    current = 0

    for wav_path in tqdm(
        paths,
        desc="Timing | Archive metadata",
        unit="file",
    ):
        info = torchaudio.info(wav_path)

        file_sr = info.sample_rate
        frames = info.num_frames

        if file_sr != fs:
            num_samples = int(
                round(
                    frames / file_sr * fs
                )
            )
        else:
            num_samples = int(frames)

        global_start = current
        global_end = (
            current + num_samples
        )

        file_infos.append(
            TimingArchiveFileInfo(
                path=wav_path,
                global_start=global_start,
                global_end=global_end,
                num_samples=num_samples,
            )
        )

        current = global_end

    return file_infos


def build_timing_segments(
    file_infos: List[TimingArchiveFileInfo],
    fs: int,
    window_sec: float,
    hop_sec: float,
) -> List[TimingSegmentRecord]:
    """
    Same contiguous-archive sliding-window construction as the
    retrieval-efficiency benchmark.
    """

    total_len = file_infos[-1].global_end

    target_len = int(
        round(window_sec * fs)
    )

    hop_len = int(
        round(hop_sec * fs)
    )

    if total_len < target_len:
        raise ValueError(
            "Archive shorter than one window."
        )

    starts = list(
        range(
            0,
            total_len - target_len + 1,
            hop_len,
        )
    )

    segments = [
        TimingSegmentRecord(
            seg_idx=seg_idx,
            start=seg_start,
            end=seg_start + target_len,
            duration_sec=window_sec,
        )
        for seg_idx, seg_start in enumerate(starts)
    ]

    return segments


class RetrievalStyleArchiveDataset(Dataset):
    """
    Raw-waveform archive dataset matching the timing context of the
    retrieval-efficiency script.

    Segments may cross source-file boundaries, exactly as in that script.
    """

    def __init__(
        self,
        file_infos: List[TimingArchiveFileInfo],
        segments: List[TimingSegmentRecord],
        fs: int,
        context_sec: float,
    ):
        super().__init__()

        self.file_infos = file_infos
        self.segments = segments
        self.fs = fs
        self.target_len = int(
            round(context_sec * fs)
        )

        self._seg_to_file = (
            [0] * len(segments)
        )

        file_index = 0

        for i, segment in enumerate(
            segments
        ):
            while (
                file_index + 1
                < len(file_infos)
                and file_infos[
                    file_index
                ].global_end
                <= segment.start
            ):
                file_index += 1

            self._seg_to_file[i] = (
                file_index
            )

    def _load_segment_wav(
        self,
        segment: TimingSegmentRecord,
    ) -> torch.Tensor:

        remaining_start = segment.start
        remaining_end = segment.end

        parts = []

        file_index = self._seg_to_file[
            segment.seg_idx
        ]

        while (
            remaining_start < remaining_end
            and file_index
            < len(self.file_infos)
        ):

            info = self.file_infos[
                file_index
            ]

            if (
                info.global_start
                >= remaining_end
            ):
                break

            local_start = (
                max(
                    remaining_start,
                    info.global_start,
                )
                - info.global_start
            )

            local_end = (
                min(
                    remaining_end,
                    info.global_end,
                )
                - info.global_start
            )

            wav, sr = load_audio_mono(
                info.path
            )

            wav, _ = ensure_sr(
                wav,
                sr,
                self.fs,
            )

            parts.append(
                wav[
                    :,
                    local_start:local_end,
                ]
            )

            remaining_start = (
                info.global_end
            )

            file_index += 1

        if not parts:
            return torch.zeros(
                (1, self.target_len),
                dtype=torch.float32,
            )

        output = torch.cat(
            parts,
            dim=-1,
        )

        if output.size(-1) < self.target_len:
            output = F.pad(
                output,
                (
                    0,
                    self.target_len
                    - output.size(-1),
                ),
            )
        else:
            output = output[
                ...,
                :self.target_len,
            ]

        return output

    def __len__(self):
        return len(self.segments)

    def __getitem__(
        self,
        idx: int,
    ):
        segment = self.segments[idx]

        return (
            self._load_segment_wav(
                segment
            ),
            segment.seg_idx,
        )


def retrieval_timing_collate(
    batch,
):
    return (
        torch.stack(
            [
                item[0]
                for item in batch
            ],
            dim=0,
        ),
        [
            item[1]
            for item in batch
        ],
    )


def build_retrieval_style_featextract(
    fs: int,
    rf_size: int,
    stride: int,
    n_mels: int,
):
    """
    Exact mel feature-extraction structure used by the retrieval benchmark.
    """

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


def _percentile(
    values: List[float],
    q: float,
) -> Optional[float]:

    if not values:
        return None

    xs = sorted(
        float(v)
        for v in values
    )

    if len(xs) == 1:
        return xs[0]

    position = (
        (len(xs) - 1) * q
    )

    lower = int(position)
    upper = min(
        lower + 1,
        len(xs) - 1,
    )

    weight = (
        position - lower
    )

    return (
        xs[lower] * (1.0 - weight)
        + xs[upper] * weight
    )


@torch.no_grad()
def measure_archive_tokenization_retrieval_style(
    model,
    bpe_model: Dict,
    audio_csv: str,
    cfg: Dict,
    batch_size: int,
    device: torch.device,
    hop_sec: float,
    path_col: str = "path",
    apply_bpe_transform: bool = False,
) -> Tuple[
    Dict[str, Any],
    List[RetrievalStyleTimingRecord],
]:
    """
    Apples-to-apples archive timing relative to the retrieval-efficiency code.

    Measurement definition
    ----------------------
    wall_clock_sec:
        Starts BEFORE iterating the DataLoader and stops after the entire
        archive has been tokenized. Therefore it includes:
            - DataLoader / audio loading
            - resampling if needed
            - mel feature extraction
            - host -> device transfer
            - model.tokenize()
            - token normalization
            - BPE when apply_bpe_transform=True
            - ordinary Python loop overhead

    model_tokenize_sec:
        CUDA-synchronized complete model.tokenize() time only.

    postprocess_sec:
        Token normalization, and for the BPE row, frozen BPE application.

    No explicit warm-up is performed, because the retrieval-efficiency
    benchmark used for the PairAlign table did not perform one.
    """

    fs = int(cfg["fs"])
    seg_sec = float(cfg["seglen"])
    n_mels = int(cfg["n_mels"])

    rf_size = int(
        round(
            float(cfg["win_size"])
            * fs
        )
    )

    stride = int(
        round(
            float(cfg["win_hop"])
            * fs
        )
    )

    file_infos = (
        build_timing_archive_metadata(
            audio_csv=audio_csv,
            fs=fs,
            path_col=path_col,
        )
    )

    segments = build_timing_segments(
        file_infos=file_infos,
        fs=fs,
        window_sec=seg_sec,
        hop_sec=hop_sec,
    )

    dataset = (
        RetrievalStyleArchiveDataset(
            file_infos=file_infos,
            segments=segments,
            fs=fs,
            context_sec=seg_sec,
        )
    )

    # Keep these exactly aligned with the retrieval-efficiency code.
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=(
            retrieval_timing_collate
        ),
    )

    featextract = (
        build_retrieval_style_featextract(
            fs=fs,
            rf_size=rf_size,
            stride=stride,
            n_mels=n_mels,
        )
    )

    model.eval().to(device)
    featextract.eval()

    totals = {
        "data": 0.0,
        "feature": 0.0,
        "transfer": 0.0,
        "model": 0.0,
        "post": 0.0,
    }

    records = []
    token_lengths = []

    if device.type == "cuda":
        cuda_sync(device)

        torch.cuda.reset_peak_memory_stats(
            device
        )

        allocated_before = (
            torch.cuda.memory_allocated(
                device
            )
            / (1024.0 ** 2)
        )

        reserved_before = (
            torch.cuda.memory_reserved(
                device
            )
            / (1024.0 ** 2)
        )

    else:
        allocated_before = None
        reserved_before = None

    # ------------------------------------------------------------
    # IMPORTANT:
    # Same wall-clock placement as retrieval-efficiency tokenize_dataset().
    # ------------------------------------------------------------

    start_all = time.perf_counter()

    iterator = iter(loader)

    progress = tqdm(
        total=len(loader),
        desc=(
            "Timing archive | "
            + (
                "Geometric+BPE"
                if apply_bpe_transform
                else "Geometric"
            )
        ),
        unit="batch",
    )

    item_order = 0
    batch_index = 0

    while True:

        # --------------------------------------------------------
        # Data loading timing
        # --------------------------------------------------------

        data_start = (
            time.perf_counter()
        )

        try:
            wavs, seg_ids = next(
                iterator
            )

        except StopIteration:
            break

        data_sec = (
            time.perf_counter()
            - data_start
        )

        batch_start = (
            time.perf_counter()
        )

        # --------------------------------------------------------
        # Feature extraction timing
        # --------------------------------------------------------

        feature_start = (
            time.perf_counter()
        )

        feats = featextract(wavs)

        feature_sec = (
            time.perf_counter()
            - feature_start
        )

        # --------------------------------------------------------
        # Host -> device timing
        # --------------------------------------------------------

        transfer_start = (
            time.perf_counter()
        )

        feats = (
            feats.to(device)
            .contiguous()
        )

        cuda_sync(device)

        transfer_sec = (
            time.perf_counter()
            - transfer_start
        )

        # --------------------------------------------------------
        # model.tokenize() timing
        # --------------------------------------------------------

        cuda_sync(device)

        model_start = (
            time.perf_counter()
        )

        raw_tokens = model.tokenize(
            feats
        )

        cuda_sync(device)

        model_sec = (
            time.perf_counter()
            - model_start
        )

        # --------------------------------------------------------
        # Post-processing timing
        #
        # Raw row:
        #     normalize tokens only
        #
        # BPE row:
        #     normalize tokens + frozen BPE
        # --------------------------------------------------------

        post_start = (
            time.perf_counter()
        )

        batch_tokens = (
            normalize_token_batch(
                raw_tokens,
                batch_size=len(seg_ids),
            )
        )

        if apply_bpe_transform:
            batch_tokens = (
                apply_bpe_batch(
                    batch_tokens,
                    bpe_model,
                )
            )

        post_sec = (
            time.perf_counter()
            - post_start
        )

        if (
            len(batch_tokens)
            != len(seg_ids)
        ):
            raise RuntimeError(
                f"{len(batch_tokens)} token sequences "
                f"for {len(seg_ids)} archive inputs"
            )

        batch_end_to_end_sec = (
            data_sec
            + (
                time.perf_counter()
                - batch_start
            )
        )

        current_batch_size = (
            len(seg_ids)
        )

        for seg_id, sequence in zip(
            seg_ids,
            batch_tokens,
        ):
            records.append(
                RetrievalStyleTimingRecord(
                    item_order=item_order,
                    seg_idx=int(seg_id),
                    batch_index=batch_index,
                    batch_size=(
                        current_batch_size
                    ),
                    num_tokens=len(sequence),
                    audio_duration_sec=(
                        seg_sec
                    ),
                    amortized_data_loading_ms=(
                        1000.0
                        * data_sec
                        / current_batch_size
                    ),
                    amortized_feature_extraction_ms=(
                        1000.0
                        * feature_sec
                        / current_batch_size
                    ),
                    amortized_host_to_device_ms=(
                        1000.0
                        * transfer_sec
                        / current_batch_size
                    ),
                    amortized_model_tokenize_ms=(
                        1000.0
                        * model_sec
                        / current_batch_size
                    ),
                    amortized_postprocess_ms=(
                        1000.0
                        * post_sec
                        / current_batch_size
                    ),
                    amortized_end_to_end_ms=(
                        1000.0
                        * batch_end_to_end_sec
                        / current_batch_size
                    ),
                )
            )

            token_lengths.append(
                len(sequence)
            )

            item_order += 1

        totals["data"] += data_sec
        totals["feature"] += feature_sec
        totals["transfer"] += transfer_sec
        totals["model"] += model_sec
        totals["post"] += post_sec

        batch_index += 1
        progress.update(1)

    progress.close()

    cuda_sync(device)

    wall_sec = (
        time.perf_counter()
        - start_all
    )

    num_items = len(
        token_lengths
    )

    total_tokens = sum(
        token_lengths
    )

    model_ms = [
        record.amortized_model_tokenize_ms
        for record in records
    ]

    post_ms = [
        record.amortized_postprocess_ms
        for record in records
    ]

    end_to_end_ms = [
        record.amortized_end_to_end_ms
        for record in records
    ]

    stats = {
        "timing_protocol":
            "retrieval_efficiency_apples_to_apples",

        "model_name": (
            "Stage I Geometric + BPE"
            if apply_bpe_transform
            else "Stage I Geometric"
        ),

        "bpe_applied":
            bool(apply_bpe_transform),

        "audio_csv":
            audio_csv,

        "batch_size":
            batch_size,

        "num_workers":
            0,

        "pin_memory":
            False,

        "window_sec":
            seg_sec,

        "hop_sec":
            hop_sec,

        "num_archive_segments":
            num_items,

        "num_batches":
            batch_index,

        "total_tokens":
            total_tokens,

        "mean_sequence_length":
            (
                total_tokens / num_items
                if num_items
                else 0.0
            ),

        "token_rate_tokens_per_audio_sec":
            (
                total_tokens
                / (
                    num_items
                    * seg_sec
                )
                if num_items
                else 0.0
            ),

        # --------------------------------------------------------
        # These are the exact concepts used by the PairAlign
        # retrieval-efficiency timing.
        # --------------------------------------------------------

        "wall_clock_sec":
            wall_sec,

        "data_loading_sec":
            totals["data"],

        "feature_extraction_sec":
            totals["feature"],

        "host_to_device_sec":
            totals["transfer"],

        "model_tokenize_sec":
            totals["model"],

        "postprocess_sec":
            totals["post"],

        "items_per_wall_sec":
            (
                num_items / wall_sec
                if wall_sec > 0
                else None
            ),

        "items_per_model_tokenize_sec":
            (
                num_items
                / totals["model"]
                if totals["model"] > 0
                else None
            ),

        "mean_wall_ms_per_item":
            (
                1000.0
                * wall_sec
                / num_items
                if num_items
                else None
            ),

        "mean_model_tokenize_ms_per_item":
            (
                sum(model_ms)
                / len(model_ms)
                if model_ms
                else None
            ),

        "p50_model_tokenize_ms_per_item":
            _percentile(
                model_ms,
                0.50,
            ),

        "p95_model_tokenize_ms_per_item":
            _percentile(
                model_ms,
                0.95,
            ),

        "mean_postprocess_ms_per_item":
            (
                sum(post_ms)
                / len(post_ms)
                if post_ms
                else None
            ),

        "p50_postprocess_ms_per_item":
            _percentile(
                post_ms,
                0.50,
            ),

        "p95_postprocess_ms_per_item":
            _percentile(
                post_ms,
                0.95,
            ),

        "mean_end_to_end_ms_per_item":
            (
                sum(end_to_end_ms)
                / len(end_to_end_ms)
                if end_to_end_ms
                else None
            ),

        "p50_end_to_end_ms_per_item":
            _percentile(
                end_to_end_ms,
                0.50,
            ),

        "p95_end_to_end_ms_per_item":
            _percentile(
                end_to_end_ms,
                0.95,
            ),
    }

    if device.type == "cuda":
        stats.update({
            "gpu_memory_allocated_before_mb":
                allocated_before,

            "gpu_memory_reserved_before_mb":
                reserved_before,

            "peak_gpu_memory_allocated_mb":
                torch.cuda.max_memory_allocated(
                    device
                )
                / (1024.0 ** 2),

            "peak_gpu_memory_reserved_mb":
                torch.cuda.max_memory_reserved(
                    device
                )
                / (1024.0 ** 2),

            "gpu_memory_allocated_after_mb":
                torch.cuda.memory_allocated(
                    device
                )
                / (1024.0 ** 2),

            "gpu_memory_reserved_after_mb":
                torch.cuda.memory_reserved(
                    device
                )
                / (1024.0 ** 2),
        })

    else:
        stats.update({
            "gpu_memory_allocated_before_mb":
                None,
            "gpu_memory_reserved_before_mb":
                None,
            "peak_gpu_memory_allocated_mb":
                None,
            "peak_gpu_memory_reserved_mb":
                None,
            "gpu_memory_allocated_after_mb":
                None,
            "gpu_memory_reserved_after_mb":
                None,
        })

    return stats, records


def write_retrieval_style_timing_csv(
    path: str,
    records: List[
        RetrievalStyleTimingRecord
    ],
):
    os.makedirs(
        os.path.dirname(path) or ".",
        exist_ok=True,
    )

    fields = list(
        RetrievalStyleTimingRecord
        .__dataclass_fields__
        .keys()
    )

    with open(
        path,
        "w",
        newline="",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
        )

        writer.writeheader()

        for record in records:
            writer.writerow(
                record.__dict__
            )


# ================================================================
# Pretty printing
# ================================================================

def print_metrics(
    name: str,
    metrics: Dict,
):

    print()
    print("=" * 74)
    print(name)
    print("=" * 74)

    print(
        f"Pairs                  : "
        f"{metrics['num_pairs']}"
    )
    print(
        f"Jaccard Similarity     : "
        f"{metrics['jaccard_similarity']:.6f}"
    )
    print(
        f"Edit Similarity        : "
        f"{metrics['edit_similarity']:.6f}"
    )
    print(
        f"Exact Match Rate       : "
        f"{metrics['exact_match_rate']:.6f}"
    )
    print(
        f"Mean Sequence Length   : "
        f"{metrics['mean_sequence_length']:.4f}"
    )
    print(
        f"Token Rate             : "
        f"{metrics['tokens_per_second']:.4f} tokens/sec"
    )
    print(
        f"Active Vocabulary Size : "
        f"{metrics['active_vocabulary_size']}"
    )


def print_retrieval_style_timing(
    timing: Dict,
):
    """
    Print using the same interpretation as retrieval efficiency:
        Archive Throughput = items_per_wall_sec
        Archive Wall time  = wall_clock_sec
    """

    print()
    print("=" * 78)
    print(
        "RETRIEVAL-STYLE ARCHIVE TOKENIZATION TIMING | "
        f"{timing['model_name']}"
    )
    print("=" * 78)

    print(
        f"Archive segments                    : "
        f"{timing['num_archive_segments']}"
    )

    print(
        f"Batch size                          : "
        f"{timing['batch_size']}"
    )

    print(
        f"Window / hop                        : "
        f"{timing['window_sec']:.3f} s / "
        f"{timing['hop_sec']:.3f} s"
    )

    print()

    print(
        f"ARCHIVE THROUGHPUT [TABLE]          : "
        f"{timing['items_per_wall_sec']:.4f} segments/s"
    )

    print(
        f"ARCHIVE WALL TIME [TABLE]           : "
        f"{timing['wall_clock_sec']:.4f} s"
    )

    print()

    print(
        f"Data loading                        : "
        f"{timing['data_loading_sec']:.4f} s"
    )

    print(
        f"Feature extraction                  : "
        f"{timing['feature_extraction_sec']:.4f} s"
    )

    print(
        f"Host -> device                      : "
        f"{timing['host_to_device_sec']:.4f} s"
    )

    print(
        f"model.tokenize()                    : "
        f"{timing['model_tokenize_sec']:.4f} s"
    )

    print(
        f"Post-processing"
        f"{' + BPE' if timing['bpe_applied'] else ''}"
        f"                 : "
        f"{timing['postprocess_sec']:.4f} s"
    )

    print()

    print(
        f"Batch-amortized model latency       : "
        f"{timing['mean_model_tokenize_ms_per_item']:.4f} "
        f"ms / 3-s segment"
    )

    print(
        f"Batch-amortized postprocess latency : "
        f"{timing['mean_postprocess_ms_per_item']:.4f} "
        f"ms / 3-s segment"
    )

    print(
        f"Batch-amortized end-to-end latency  : "
        f"{timing['mean_end_to_end_ms_per_item']:.4f} "
        f"ms / 3-s segment"
    )

    print()

    peak = timing.get(
        "peak_gpu_memory_allocated_mb"
    )

    if peak is not None:
        print(
            f"Peak allocated GPU memory           : "
            f"{peak:.2f} MB"
        )

    print()
    print(
        "Use ARCHIVE THROUGHPUT and ARCHIVE WALL TIME above "
        "for direct comparison with the PairAlign archive columns."
    )


def save_summary(
    path: str,
    dataset_name: str,
    raw_metrics: Dict,
    bpe_metrics: Dict,
    raw_timing: Dict,
    bpe_timing: Dict,
):

    os.makedirs(
        os.path.dirname(path) or ".",
        exist_ok=True,
    )

    fields = [
        "dataset",
        "model",
        "jaccard_similarity",
        "edit_similarity",
        "exact_match_rate",
        "mean_sequence_length",
        "tokens_per_second",
        "active_vocabulary_size",

        # Retrieval-efficiency-compatible timing
        "archive_num_segments",
        "archive_wall_time_s",
        "archive_throughput_segments_per_s",

        "data_loading_s",
        "feature_extraction_s",
        "host_to_device_s",
        "model_tokenize_s",
        "postprocess_s",

        "model_tokenize_ms_per_segment",
        "postprocess_ms_per_segment",
        "end_to_end_ms_per_segment",

        "peak_gpu_memory_allocated_mb",
    ]

    with open(
        path,
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()

        for (
            name,
            metrics,
            timing,
        ) in [
            (
                "Stage I Geometric",
                raw_metrics,
                raw_timing,
            ),
            (
                "Stage I Geometric + BPE",
                bpe_metrics,
                bpe_timing,
            ),
        ]:

            writer.writerow({
                "dataset":
                    dataset_name,

                "model":
                    name,

                "jaccard_similarity":
                    metrics[
                        "jaccard_similarity"
                    ],

                "edit_similarity":
                    metrics[
                        "edit_similarity"
                    ],

                "exact_match_rate":
                    metrics[
                        "exact_match_rate"
                    ],

                "mean_sequence_length":
                    metrics[
                        "mean_sequence_length"
                    ],

                "tokens_per_second":
                    metrics[
                        "tokens_per_second"
                    ],

                "active_vocabulary_size":
                    metrics[
                        "active_vocabulary_size"
                    ],

                "archive_num_segments":
                    timing[
                        "num_archive_segments"
                    ],

                "archive_wall_time_s":
                    timing[
                        "wall_clock_sec"
                    ],

                "archive_throughput_segments_per_s":
                    timing[
                        "items_per_wall_sec"
                    ],

                "data_loading_s":
                    timing[
                        "data_loading_sec"
                    ],

                "feature_extraction_s":
                    timing[
                        "feature_extraction_sec"
                    ],

                "host_to_device_s":
                    timing[
                        "host_to_device_sec"
                    ],

                "model_tokenize_s":
                    timing[
                        "model_tokenize_sec"
                    ],

                "postprocess_s":
                    timing[
                        "postprocess_sec"
                    ],

                "model_tokenize_ms_per_segment":
                    timing[
                        "mean_model_tokenize_ms_per_item"
                    ],

                "postprocess_ms_per_segment":
                    timing[
                        "mean_postprocess_ms_per_item"
                    ],

                "end_to_end_ms_per_segment":
                    timing[
                        "mean_end_to_end_ms_per_item"
                    ],

                "peak_gpu_memory_allocated_mb":
                    timing.get(
                        "peak_gpu_memory_allocated_mb"
                    ),
            })

    print(
        f"\nSaved summary: {path}"
    )


def save_timing_json(
    path: str,
    raw_timing: Dict,
    bpe_timing: Dict,
):
    os.makedirs(
        os.path.dirname(path) or ".",
        exist_ok=True,
    )

    payload = {
        "timing_protocol":
            "retrieval_efficiency_apples_to_apples",

        "raw_geometric":
            raw_timing,

        "geometric_plus_bpe":
            bpe_timing,
    }

    with open(path, "w") as f:
        json.dump(
            payload,
            f,
            indent=2,
        )

    print(
        f"Saved timing JSON: {path}"
    )


# ================================================================
# DataLoader helper
# ================================================================

def build_loader(
    cfg,
    split_name,
    batch_size,
    num_workers,
    add_noise,
    use_time_compression=False,
    shuffle=False,
    seed=13,
):

    split_cfg = cfg[split_name]
    dataset_csv = split_cfg["dataset_csv"]

    dataset = AudioPairAugDataset(
        audio_csv=dataset_csv,
        fs=cfg["fs"],
        context_sec=cfg["seglen"],
        add_noise=add_noise,
        use_time_compression=use_time_compression,
        return_path=True,
        feats_type="melspectrogram",
        rf_size=int(
            round(
                cfg["win_size"]
                * cfg["fs"]
            )
        ),
        stride=int(
            round(
                cfg["win_hop"]
                * cfg["fs"]
            )
        ),
        n_mels=cfg["n_mels"],
        split=(
            "train"
            if split_name == "train"
            else "val"
        ),
    )

    generator = None
    if shuffle:
        generator = torch.Generator()
        generator.manual_seed(int(seed))

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=pair_collate,
        generator=generator,
    )


# ================================================================
# Main
# ================================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root_dir",
        type=str,
        default=os.path.dirname(
            os.getcwd()
        ),
    )

    parser.add_argument(
        "--config",
        type=str,
        default="mainFull.yaml",
    )

    parser.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="Stage-I geometric AudioTokenizer checkpoint",
    )

    parser.add_argument(
        "--eval_split",
        type=str,
        default="valid",
        choices=[
            "train",
            "valid",
            "test",
        ],
    )

    # ------------------------------------------------------------
    # BPE
    # ------------------------------------------------------------

    parser.add_argument(
        "--bpe_model",
        type=str,
        required=True,
        help="Path to save/load BPE JSON model",
    )

    parser.add_argument(
        "--fit_bpe",
        action="store_true",
        help=(
            "Learn BPE from equal-sized disjoint clean and mildly "
            "augmented TRAIN subsets"
        ),
    )

    parser.add_argument(
        "--target_token_rate",
        type=float,
        default=9.0,
        help="Target BPE token rate in tokens/sec",
    )

    parser.add_argument(
        "--target_mean_len",
        type=float,
        default=None,
        help=(
            "Optional exact target sequence length. "
            "Overrides --target_token_rate."
        ),
    )

    parser.add_argument(
        "--bpe_max_sequences",
        type=int,
        default=25000,
        help=(
            "Total number of TRAIN sequences used for BPE fitting, "
            "split equally between disjoint clean and augmented subsets. "
            "Use <=0 for all available sequences."
        ),
    )

    parser.add_argument(
        "--max_bpe_merges",
        type=int,
        default=1000000,
    )

    parser.add_argument(
        "--min_pair_frequency",
        type=int,
        default=2,
    )

    # ------------------------------------------------------------
    # Evaluation / timing
    # ------------------------------------------------------------

    parser.add_argument(
        "--output_csv",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--summary_csv",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--timing_json",
        type=str,
        default=None,
        help="Optional JSON file containing archive timing measurements.",
    )

    parser.add_argument(
        "--timing_hop_sec",
        type=float,
        default=1.5,
        help=(
            "Archive hop in seconds for the retrieval-style timing pass. "
            "Default 1.5 matches the PairAlign retrieval-efficiency benchmark."
        ),
    )

    parser.add_argument(
        "--timing_path_col",
        type=str,
        default="path",
        help="Audio-path column for the retrieval-style timing CSV.",
    )

    parser.add_argument(
        "--raw_archive_timing_csv",
        type=str,
        default=None,
        help=(
            "Optional per-segment timing CSV for raw Stage-I Geometric "
            "using the retrieval-efficiency timing protocol."
        ),
    )

    parser.add_argument(
        "--bpe_archive_timing_csv",
        type=str,
        default=None,
        help=(
            "Optional per-segment timing CSV for Stage-I Geometric+BPE "
            "using the retrieval-efficiency timing protocol."
        ),
    )

    parser.add_argument(
        "--dataset_name",
        type=str,
        default="Dataset",
    )

    parser.add_argument(
        "--device",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--add_noise",
        action="store_true",
    )

    parser.add_argument(
        "--no_add_noise",
        action="store_true",
    )
    parser.add_argument(
        "--use_time_compression",
        action=argparse.BooleanOptionalAction,
        default=False,
    )

    args = parser.parse_args()

    # ------------------------------------------------------------
    # Config
    # ------------------------------------------------------------

    config_path = os.path.join(
        args.root_dir,
        "config",
        args.config,
    )

    with open(config_path, "r") as f:
        cfg = yaml.load(
            f,
            Loader=yaml.FullLoader,
        )

    seed_everything(
        13,
        workers=True,
    )

    batch_size = (
        args.batch_size
        if args.batch_size is not None
        else int(cfg["batchsize"])
    )

    num_workers = (
        args.num_workers
        if args.num_workers is not None
        else int(cfg["load_workers"])
    )

    segment_sec = float(
        cfg["seglen"]
    )

    if args.no_add_noise:
        eval_add_noise = False
    elif args.add_noise:
        eval_add_noise = True
    else:
        eval_add_noise = True

    device = torch.device(
        f"cuda:{args.device}"
        if torch.cuda.is_available()
        else "cpu"
    )

    print()
    print(f"Geometric checkpoint : {args.ckpt}")
    print(f"Segment duration     : {segment_sec:.4f} s")
    print(f"Batch size           : {batch_size}")
    print(
        f"Timing protocol      : retrieval-efficiency apples-to-apples"
    )
    print(
        f"Timing archive hop   : {args.timing_hop_sec:.4f} s"
    )

    model = build_geometric_model(
        ckpt_path=args.ckpt,
        device_id=args.device,
    )

    # ============================================================
    # FIT BPE
    # ============================================================

    if args.fit_bpe:
        print()
        print(
            "Fitting BPE using disjoint clean and augmented TRAIN subsets."
        )

        train_loader = build_loader(
            cfg=cfg,
            split_name="train",
            batch_size=batch_size,
            num_workers=num_workers,
            add_noise=False,
            use_time_compression=False,
            shuffle=True,
            seed=13,
        )

        max_sequences = (
            None
            if args.bpe_max_sequences <= 0
            else args.bpe_max_sequences
        )

        fit_sequences, fit_composition = (
            collect_bpe_training_sequences(
                model=model,
                loader=train_loader,
                device=device,
                max_sequences=max_sequences,
            )
        )

        fit_raw_mean = mean_sequence_length(
            fit_sequences
        )

        print(
            f"\nBPE fitting sequences : "
            f"{len(fit_sequences)}"
        )
        print(
            f"Clean sequences       : "
            f"{fit_composition['clean']}"
        )
        print(
            f"Augmented sequences   : "
            f"{fit_composition['augmented']}"
        )

        print(
            f"Raw fitting mean len  : "
            f"{fit_raw_mean:.4f}"
        )

        if args.target_mean_len is not None:
            target_mean_length = (
                args.target_mean_len
            )
        else:
            target_mean_length = (
                args.target_token_rate
                * segment_sec
            )

        print(
            f"Target mean length    : "
            f"{target_mean_length:.4f}"
        )

        print(
            f"Target token rate     : "
            f"{target_mean_length / segment_sec:.4f} token/s"
        )

        bpe_model = learn_bpe(
            corpus=fit_sequences,
            target_mean_length=target_mean_length,
            min_pair_frequency=(
                args.min_pair_frequency
            ),
            max_merges=(
                args.max_bpe_merges
            ),
        )

        bpe_model[
            "segment_sec"
        ] = segment_sec

        bpe_model[
            "target_token_rate"
        ] = (
            target_mean_length
            / segment_sec
        )

        bpe_model[
            "fit_num_sequences"
        ] = len(fit_sequences)

        bpe_model[
            "fit_num_clean_sequences"
        ] = fit_composition["clean"]

        bpe_model[
            "fit_num_augmented_sequences"
        ] = fit_composition["augmented"]

        bpe_model[
            "fit_disjoint_subsets"
        ] = True

        bpe_model[
            "source_checkpoint"
        ] = os.path.basename(args.ckpt)

        save_bpe_model(
            bpe_model,
            args.bpe_model,
        )

    # ============================================================
    # LOAD EXISTING BPE
    # ============================================================

    else:
        if not os.path.exists(
            args.bpe_model
        ):
            raise FileNotFoundError(
                f"BPE model does not exist: "
                f"{args.bpe_model}\n"
                f"Use --fit_bpe first."
            )

        bpe_model = load_bpe_model(
            args.bpe_model
        )


    eval_loader = build_loader(
        cfg=cfg,
        split_name=args.eval_split,
        batch_size=batch_size,
        num_workers=num_workers,
        add_noise=eval_add_noise,
        use_time_compression=args.use_time_compression,
    )

    raw_metrics, bpe_metrics, _consistency_timing = (
        evaluate_geometric_and_bpe(
            model=model,
            loader=eval_loader,
            bpe_model=bpe_model,
            output_csv=args.output_csv,
            device=device,
            segment_sec=segment_sec,
            timing_warmup_batches=0,
        )
    )

    # ============================================================
    # RETRIEVAL-EFFICIENCY-STYLE ARCHIVE TIMING
    # ============================================================
    #
    # This is deliberately a separate clean archive-only pass using
    # the exact timing context of the PairAlign retrieval-efficiency
    # code:
    #
    # raw waveform loading -> mel features -> H2D -> model.tokenize()
    # -> post-processing, with whole-loop wall_clock_sec.
    #
    # No positive views, no augmentation, no explicit warm-up.
    # ============================================================

    timing_audio_csv = (
        cfg[args.eval_split][
            "dataset_csv"
        ]
    )

    print()
    print("=" * 78)
    print("RUNNING APPLES-TO-APPLES ARCHIVE TIMING")
    print("=" * 78)
    print(
        f"Timing audio CSV      : {timing_audio_csv}"
    )
    print(
        f"Timing window / hop   : "
        f"{segment_sec:.3f} s / "
        f"{args.timing_hop_sec:.3f} s"
    )
    print(
        f"Timing DataLoader     : "
        f"batch={batch_size}, num_workers=0, pin_memory=False"
    )
    print(
        "No explicit warm-up is used, matching the retrieval-efficiency code."
    )

    if device.type == "cuda":
        torch.cuda.empty_cache()

    raw_timing, raw_timing_records = (
        measure_archive_tokenization_retrieval_style(
            model=model,
            bpe_model=bpe_model,
            audio_csv=timing_audio_csv,
            cfg=cfg,
            batch_size=batch_size,
            device=device,
            hop_sec=args.timing_hop_sec,
            path_col=args.timing_path_col,
            apply_bpe_transform=False,
        )
    )

    if device.type == "cuda":
        torch.cuda.empty_cache()

    bpe_timing, bpe_timing_records = (
        measure_archive_tokenization_retrieval_style(
            model=model,
            bpe_model=bpe_model,
            audio_csv=timing_audio_csv,
            cfg=cfg,
            batch_size=batch_size,
            device=device,
            hop_sec=args.timing_hop_sec,
            path_col=args.timing_path_col,
            apply_bpe_transform=True,
        )
    )

    if (
        args.raw_archive_timing_csv
        is not None
    ):
        write_retrieval_style_timing_csv(
            args.raw_archive_timing_csv,
            raw_timing_records,
        )

        print(
            f"Saved raw archive timing CSV: "
            f"{args.raw_archive_timing_csv}"
        )

    if (
        args.bpe_archive_timing_csv
        is not None
    ):
        write_retrieval_style_timing_csv(
            args.bpe_archive_timing_csv,
            bpe_timing_records,
        )

        print(
            f"Saved BPE archive timing CSV: "
            f"{args.bpe_archive_timing_csv}"
        )

    print_metrics(
        "Stage I Geometric",
        raw_metrics,
    )

    print_metrics(
        "Stage I Geometric + BPE",
        bpe_metrics,
    )

    compression_ratio = (
        raw_metrics[
            "mean_sequence_length"
        ]
        /
        max(
            bpe_metrics[
                "mean_sequence_length"
            ],
            1e-8,
        )
    )

    print()
    print("=" * 74)
    print("RATE MATCHING")
    print("=" * 74)

    print(
        f"Raw token rate       : "
        f"{raw_metrics['tokens_per_second']:.4f} token/s"
    )

    print(
        f"BPE token rate       : "
        f"{bpe_metrics['tokens_per_second']:.4f} token/s"
    )

    print(
        f"Compression factor   : "
        f"{compression_ratio:.4f}x"
    )

    print(
        f"Number BPE merges    : "
        f"{len(bpe_model['merges'])}"
    )

    print_retrieval_style_timing(
        raw_timing
    )

    print_retrieval_style_timing(
        bpe_timing
    )

    print()
    print("=" * 78)
    print("TABLE VALUES FOR APPLES-TO-APPLES ARCHIVE COMPARISON")
    print("=" * 78)

    print(
        "Stage I Geometric:"
    )
    print(
        f"  Archive throughput = "
        f"{raw_timing['items_per_wall_sec']:.4f} segments/s"
    )
    print(
        f"  Archive wall time  = "
        f"{raw_timing['wall_clock_sec']:.4f} s"
    )

    print(
        "Stage I Geometric + BPE:"
    )
    print(
        f"  Archive throughput = "
        f"{bpe_timing['items_per_wall_sec']:.4f} segments/s"
    )
    print(
        f"  Archive wall time  = "
        f"{bpe_timing['wall_clock_sec']:.4f} s"
    )

    if args.summary_csv is not None:
        save_summary(
            path=args.summary_csv,
            dataset_name=args.dataset_name,
            raw_metrics=raw_metrics,
            bpe_metrics=bpe_metrics,
            raw_timing=raw_timing,
            bpe_timing=bpe_timing,
        )

    if args.timing_json is not None:
        save_timing_json(
            path=args.timing_json,
            raw_timing=raw_timing,
            bpe_timing=bpe_timing,
        )


if __name__ == "__main__":
    main()
