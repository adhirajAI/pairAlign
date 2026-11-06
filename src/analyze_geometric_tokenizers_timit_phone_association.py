#!/usr/bin/env python3
"""
TIMIT phone/speaker/silence/ABX analysis for geometric tokenizers only.

Supported variants:
  * AudioTokenizer -> Stage I Geometric
  * AudioTokenizer -> Stage I+ Geometric

Stage I and Stage I+ use the same AudioTokenizer class and differ only in
their checkpoint paths.

The essential geometric-tokenizer call is:

    model.tokenize(..., strip_special_tokens=True, deduplicate=False)

With deduplicate=False, the output is required to contain exactly one token for
all 301 encoder/Mel frames of a 3-second centered-STFT segment.
"""

import os
import csv
import json
import math
import gzip
import random
import argparse
from pathlib import Path
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader
from lightning.pytorch import seed_everything
from tqdm.auto import tqdm

from train import AudioTokenizer
from utils import AudioPairAugDataset


TIMIT_61_TO_39 = {
    "aa": "aa", "ae": "ae", "ah": "ah", "ao": "aa", "aw": "aw",
    "ax": "ah", "ax-h": "ah", "axr": "er", "ay": "ay",
    "b": "b", "bcl": "sil", "ch": "ch", "d": "d", "dcl": "sil",
    "dh": "dh", "dx": "dx", "eh": "eh", "el": "l", "em": "m",
    "en": "n", "eng": "ng", "epi": "sil", "er": "er", "ey": "ey",
    "f": "f", "g": "g", "gcl": "sil", "h#": "sil", "hh": "hh",
    "hv": "hh", "ih": "ih", "ix": "ih", "iy": "iy", "jh": "jh",
    "k": "k", "kcl": "sil", "l": "l", "m": "m", "n": "n",
    "ng": "ng", "nx": "n", "ow": "ow", "oy": "oy", "p": "p",
    "pau": "sil", "pcl": "sil", "q": "<ignore>", "r": "r",
    "s": "s", "sh": "sh", "t": "t", "tcl": "sil", "th": "th",
    "uh": "uh", "uw": "uw", "ux": "uw", "v": "v", "w": "w",
    "y": "y", "z": "z", "zh": "sh",
}
SILENCE_39 = {"sil"}
IGNORE_PHONES = {"<ignore>", "<outside>", "<pad>", ""}


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if not rows:
        return
    keys: List[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


# -----------------------------------------------------------------------------
# Dataset parsing
# -----------------------------------------------------------------------------

def item_to_features_and_meta(item: Any) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    if isinstance(item, dict):
        anc_key = next((k for k in ["anc", "anchor", "anchor_feat", "x"] if k in item), None)
        pos_key = next((k for k in ["pos", "positive", "positive_feat", "aug", "x_aug"] if k in item), None)
        if anc_key is None or pos_key is None:
            raise ValueError(f"Could not infer anchor/positive keys from {list(item.keys())}")
        meta = dict(item.get("metadata", item.get("meta", {})) or {})
        for key in [
            "audio_path", "path", "wav_path", "phn_path", "file_id",
            "crop_start_sample", "start_sample", "anc_crop_start_sample",
            "pos_crop_start_sample", "segment_num_samples", "sample_rate",
            "phn_sample_rate", "speaker_id", "pos_time_preserving",
        ]:
            if key in item:
                meta[key] = item[key]
        return item[anc_key], item[pos_key], meta

    if isinstance(item, (tuple, list)):
        if len(item) < 2:
            raise ValueError("Dataset item must contain anchor and positive features.")
        meta: Dict[str, Any] = {}
        if len(item) >= 3:
            if isinstance(item[2], dict):
                meta.update(item[2])
            else:
                meta["audio_path"] = str(item[2])
        if len(item) >= 4:
            if not isinstance(item[3], dict):
                raise TypeError("The fourth dataset item must be a metadata dictionary.")
            meta.update(item[3])
        return item[0], item[1], meta

    raise TypeError(f"Unsupported dataset item type: {type(item)}")


def pair_collate(batch):
    anchors, positives, metadata = [], [], []
    for item in batch:
        anc, pos, meta = item_to_features_and_meta(item)
        anchors.append(anc)
        positives.append(pos)
        metadata.append(meta)
    return torch.stack(anchors, 0), torch.stack(positives, 0), metadata


# -----------------------------------------------------------------------------
# Model/tokenization
# -----------------------------------------------------------------------------

def build_model(checkpoint: str, device_id: int):
    """
    Stage I Geometric and Stage I+ Geometric both use AudioTokenizer.
    The variants differ only through the supplied checkpoint.
    """
    model = AudioTokenizer.load_from_pretrained(checkpoint, device_id)
    model.eval()
    return model


def normalize_token_batch(tokens: Any, batch_size: int) -> List[torch.LongTensor]:
    if isinstance(tokens, torch.Tensor):
        tokens = tokens.detach().cpu()
        if tokens.ndim == 1:
            if batch_size != 1:
                raise ValueError(f"One token sequence returned for batch size {batch_size}.")
            return [tokens.long().reshape(-1)]
        if tokens.ndim == 2:
            if tokens.size(0) != batch_size:
                raise ValueError(f"Token batch dimension {tokens.size(0)} != {batch_size}.")
            return [tokens[b].long().reshape(-1) for b in range(batch_size)]
        if tokens.ndim == 3 and tokens.size(1) == 1:
            if tokens.size(0) != batch_size:
                raise ValueError(f"Token batch dimension {tokens.size(0)} != {batch_size}.")
            return [tokens[b, 0].long().reshape(-1) for b in range(batch_size)]
        raise ValueError(f"Unsupported token tensor shape: {tuple(tokens.shape)}")

    if isinstance(tokens, (list, tuple)):
        if len(tokens) != batch_size:
            if batch_size == 1:
                return [torch.as_tensor(tokens).long().reshape(-1)]
            raise ValueError(f"Token output length {len(tokens)} != batch size {batch_size}.")
        return [
            t.detach().cpu().long().reshape(-1)
            if isinstance(t, torch.Tensor)
            else torch.as_tensor(t).long().reshape(-1)
            for t in tokens
        ]

    raise TypeError(f"Unsupported tokenize() output type: {type(tokens)}")


@torch.no_grad()
def tokenize_geometric_batch(
    model,
    features: torch.Tensor,
    expected_frames: int,
    max_len: int,
    beam_size: int,
) -> List[torch.LongTensor]:
    outputs = model.tokenize(
        features,
        max_len=max_len,
        beam_size=beam_size,
        strip_special_tokens=True,
        deduplicate=False,
    )
    sequences = normalize_token_batch(outputs, features.size(0))
    for batch_index, sequence in enumerate(sequences):
        if sequence.numel() != expected_frames:
            raise ValueError(
                f"Expected {expected_frames} frame-aligned geometric tokens with "
                f"deduplicate=False, but batch item {batch_index} has "
                f"{sequence.numel()}. Check tokenize() output and special-token handling."
            )
    return sequences


# -----------------------------------------------------------------------------
# PHN alignment
# -----------------------------------------------------------------------------

def infer_audio_path(meta: Dict[str, Any]) -> str:
    for key in ["audio_path", "wav_path", "path"]:
        if meta.get(key) is not None:
            return str(meta[key])
    raise KeyError("Dataset metadata is missing audio_path.")


def infer_phn_path(audio_path: str, explicit: Optional[str]) -> str:
    candidates: List[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    path = Path(audio_path)
    candidates += [path.with_suffix(".PHN"), path.with_suffix(".phn"), path.with_suffix(".Phn")]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    raise FileNotFoundError(
        f"No .PHN file found for {audio_path}. Tried: " + ", ".join(str(x) for x in candidates)
    )


def infer_speaker_id(audio_path: str, explicit: Optional[str]) -> str:
    return str(explicit) if explicit else Path(audio_path).parent.name


def resolve_crop_start(meta: Dict[str, Any], side: str) -> int:
    keys = [f"{side}_crop_start_sample", "crop_start_sample", "start_sample"]
    for key in keys:
        if meta.get(key) is not None:
            return int(meta[key])
    raise KeyError(f"Missing crop start for {side}; expected one of {keys}.")


def side_is_time_preserving(meta: Dict[str, Any], side: str) -> bool:
    return side == "anc" or bool(meta.get("pos_time_preserving", False))


def read_timit_phn(path: str) -> List[Tuple[int, int, str]]:
    records = []
    with open(path, "r") as f:
        for line_number, line in enumerate(f, 1):
            parts = line.strip().split()
            if not parts:
                continue
            if len(parts) != 3:
                raise ValueError(f"Malformed .PHN line {line_number} in {path}: {line!r}")
            start, end, phone = int(parts[0]), int(parts[1]), parts[2]
            if end <= start:
                raise ValueError(f"Non-positive .PHN interval at {path}:{line_number}")
            records.append((start, end, phone))
    if not records:
        raise ValueError(f"No phone intervals found in {path}.")
    return records


def map_phone_39(phone: str, q_policy: str) -> str:
    if phone == "q" and q_policy == "sil":
        return "sil"
    return TIMIT_61_TO_39.get(phone, phone)


def align_phn_to_frames(
    phn_records: List[Tuple[int, int, str]],
    crop_start_sample: int,
    num_frames: int,
    hop_samples: int,
    win_samples: int,
    segment_num_samples: int,
    stft_center: bool,
    q_policy: str,
) -> Dict[str, Any]:
    offsets = torch.arange(num_frames, dtype=torch.long) * int(hop_samples)
    if not stft_center:
        offsets = offsets + int(win_samples // 2)
    offsets = offsets.clamp(0, int(segment_num_samples) - 1)
    absolute_samples = offsets + int(crop_start_sample)

    starts = torch.tensor([x[0] for x in phn_records], dtype=torch.long)
    ends = torch.tensor([x[1] for x in phn_records], dtype=torch.long)
    phones = [x[2] for x in phn_records]
    indices = torch.searchsorted(ends, absolute_samples, right=True)

    labels_61: List[str] = []
    labels_39: List[str] = []
    for sample, index in zip(absolute_samples.tolist(), indices.tolist()):
        if index >= len(phn_records) or not (int(starts[index]) <= sample < int(ends[index])):
            labels_61.append("<outside>")
            labels_39.append("<outside>")
        else:
            phone = phones[index]
            labels_61.append(phone)
            labels_39.append(map_phone_39(phone, q_policy))

    crop_end = int(crop_start_sample) + int(segment_num_samples)
    boundary_samples_crop: List[int] = []
    boundary_frames_float: List[float] = []
    for index in range(len(phn_records) - 1):
        _, left_end, _ = phn_records[index]
        right_start, _, _ = phn_records[index + 1]
        boundary = int(right_start if right_start != left_end else left_end)
        if not (int(crop_start_sample) < boundary < crop_end):
            continue
        offset = boundary - int(crop_start_sample)
        frame = (
            offset / float(hop_samples)
            if stft_center
            else (offset - win_samples / 2.0) / float(hop_samples)
        )
        boundary_samples_crop.append(offset)
        boundary_frames_float.append(float(frame))

    return {
        "phone_labels_61": labels_61,
        "phone_labels_39": labels_39,
        "frame_center_samples_in_crop": offsets,
        "frame_center_samples_in_utterance": absolute_samples,
        "phone_boundary_samples_in_crop": boundary_samples_crop,
        "phone_boundary_frames_float": boundary_frames_float,
    }


# -----------------------------------------------------------------------------
# Boundary and local alignment metrics
# -----------------------------------------------------------------------------

def token_boundaries(token_ids: Sequence[int]) -> np.ndarray:
    values = np.asarray(token_ids, dtype=np.int64)
    if values.size < 2:
        return np.empty(0, dtype=np.float64)
    return np.flatnonzero(values[1:] != values[:-1]).astype(np.float64) + 0.5


def match_boundaries(
    predicted: np.ndarray,
    reference: np.ndarray,
    tolerance_frames: float,
) -> Tuple[int, int, int, List[float]]:
    predicted = np.sort(np.asarray(predicted, dtype=np.float64))
    reference = np.sort(np.asarray(reference, dtype=np.float64))
    candidates = []
    for pred_index, prediction in enumerate(predicted):
        left = np.searchsorted(reference, prediction - tolerance_frames, side="left")
        right = np.searchsorted(reference, prediction + tolerance_frames, side="right")
        for ref_index in range(left, right):
            candidates.append((abs(float(prediction) - float(reference[ref_index])), pred_index, ref_index))
    candidates.sort()
    used_pred, used_ref, distances = set(), set(), []
    for distance, pred_index, ref_index in candidates:
        if pred_index not in used_pred and ref_index not in used_ref:
            used_pred.add(pred_index)
            used_ref.add(ref_index)
            distances.append(float(distance))
    true_positive = len(distances)
    return true_positive, len(predicted) - true_positive, len(reference) - true_positive, distances


def boundary_metrics(
    token_ids: torch.LongTensor,
    reference_boundaries: Sequence[float],
    hop_ms: float,
) -> Dict[str, float]:
    predicted = token_boundaries(token_ids.tolist())
    reference = np.asarray(reference_boundaries, dtype=np.float64)
    reference = reference[np.isfinite(reference)]
    output: Dict[str, float] = {
        "num_token_boundaries": float(len(predicted)),
        "num_phone_boundaries": float(len(reference)),
    }
    for tolerance_ms in [10, 20, 30]:
        tp, fp, fn, distances = match_boundaries(
            predicted, reference, tolerance_ms / float(hop_ms)
        )
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
        output[f"boundary_precision_{tolerance_ms}ms"] = precision
        output[f"boundary_recall_{tolerance_ms}ms"] = recall
        output[f"boundary_f1_{tolerance_ms}ms"] = f1
        if tolerance_ms == 20:
            output["boundary_mae_frames_20ms"] = float(np.mean(distances)) if distances else float("nan")
            output["boundary_mae_ms_20ms"] = (
                float(np.mean(distances) * hop_ms) if distances else float("nan")
            )
    ratio = len(predicted) / max(len(reference), 1)
    output["token_to_phone_boundary_count_ratio"] = ratio
    output["oversegmentation_ratio_abs_log"] = abs(math.log(max(ratio, 1e-12)))
    return output


def runs(values: Sequence[Any]) -> List[Tuple[int, int, Any]]:
    if not values:
        return []
    output = []
    start, previous = 0, values[0]
    for index in range(1, len(values)):
        if values[index] != previous:
            output.append((start, index, previous))
            start, previous = index, values[index]
    output.append((start, len(values), previous))
    return output


def segment_metrics(token_ids: torch.LongTensor, phone_labels_39: Sequence[str]) -> Dict[str, float]:
    n = min(token_ids.numel(), len(phone_labels_39))
    tokens = token_ids[:n].tolist()
    phones = list(phone_labels_39[:n])

    token_correct = token_frames = 0
    phones_per_token = []
    for start, end, _ in runs(tokens):
        valid = [phone for phone in phones[start:end] if phone not in IGNORE_PHONES]
        if not valid:
            continue
        counts = Counter(valid)
        token_correct += max(counts.values())
        token_frames += len(valid)
        phones_per_token.append(len(counts))

    phone_correct = phone_frames = 0
    tokens_per_phone = []
    for start, end, phone in runs(phones):
        if phone in IGNORE_PHONES:
            continue
        values = tokens[start:end]
        if not values:
            continue
        counts = Counter(values)
        phone_correct += max(counts.values())
        phone_frames += len(values)
        tokens_per_phone.append(len(counts))

    return {
        "token_segment_phone_purity": token_correct / max(token_frames, 1),
        "phone_segment_token_purity": phone_correct / max(phone_frames, 1),
        "mean_phones_per_token_segment": float(np.mean(phones_per_token)) if phones_per_token else float("nan"),
        "mean_tokens_per_phone_segment": float(np.mean(tokens_per_phone)) if tokens_per_phone else float("nan"),
    }


# -----------------------------------------------------------------------------
# HuBERT-style, silence, and speaker metrics
# -----------------------------------------------------------------------------

def entropy(counter: Counter) -> float:
    total = sum(counter.values())
    if total <= 0:
        return 0.0
    probabilities = np.asarray([value / total for value in counter.values() if value > 0], dtype=np.float64)
    return float(-(probabilities * np.log(probabilities)).sum())


def mutual_information(joint: Counter) -> float:
    total = sum(joint.values())
    if total <= 0:
        return 0.0
    left_counts, right_counts = Counter(), Counter()
    for (left, right), count in joint.items():
        left_counts[left] += count
        right_counts[right] += count
    result = 0.0
    for (left, right), count in joint.items():
        p_joint = count / total
        p_left = left_counts[left] / total
        p_right = right_counts[right] / total
        result += p_joint * math.log(p_joint / max(p_left * p_right, 1e-300))
    return float(result)


def conditional_speaker_information(triples: Counter) -> Tuple[float, float]:
    total = sum(triples.values())
    phone_counts = Counter()
    token_speaker_by_phone = defaultdict(Counter)
    speaker_by_phone = defaultdict(Counter)
    for (phone, token, speaker), count in triples.items():
        phone_counts[phone] += count
        token_speaker_by_phone[phone][(token, speaker)] += count
        speaker_by_phone[phone][speaker] += count
    conditional_mi = conditional_entropy = 0.0
    for phone, count in phone_counts.items():
        weight = count / max(total, 1)
        conditional_mi += weight * mutual_information(token_speaker_by_phone[phone])
        conditional_entropy += weight * entropy(speaker_by_phone[phone])
    return conditional_mi, conditional_entropy


def global_metrics(records: List[Dict[str, Any]], variant: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    phone_token = Counter()
    token_counts, phone_counts = Counter(), Counter()
    speaker_token = Counter()
    triples = Counter()
    heldout_by_speaker = defaultdict(Counter)

    for record in records:
        speaker = record["speaker_id"]
        for token, phone in zip(record["token_ids"].tolist(), record["phone_labels_39"]):
            if phone in IGNORE_PHONES:
                continue
            token = int(token)
            phone_token[(phone, token)] += 1
            token_counts[token] += 1
            phone_counts[phone] += 1
            speaker_token[(speaker, token)] += 1
            triples[(phone, token, speaker)] += 1
            heldout_by_speaker[speaker][(phone, token)] += 1

    num_frames = sum(phone_token.values())
    if num_frames == 0:
        raise ValueError("No valid phone-token frames were collected.")

    phone_purity = sum(max(phone_token[(phone, token)] for phone in phone_counts) for token in token_counts) / num_frames
    cluster_purity = sum(max(phone_token[(phone, token)] for token in token_counts) for phone in phone_counts) / num_frames
    pnmi = mutual_information(phone_token) / max(entropy(phone_counts), 1e-12)
    codebook_perplexity = math.exp(entropy(token_counts))
    active_vocab = len(token_counts)

    speaker_counts = Counter()
    for (speaker, _), count in speaker_token.items():
        speaker_counts[speaker] += count
    speaker_nmi = mutual_information(speaker_token) / max(entropy(speaker_counts), 1e-12)
    conditional_mi, conditional_entropy = conditional_speaker_information(triples)

    silence_joint = Counter()
    for (phone, token), count in phone_token.items():
        silence_joint[("silence" if phone in SILENCE_39 else "speech", token)] += count
    silence_purity = sum(
        max(silence_joint[("silence", token)], silence_joint[("speech", token)])
        for token in token_counts
    ) / num_frames

    global_majority = phone_counts.most_common(1)[0][0]
    loso_correct = loso_total = 0
    for heldout_speaker, heldout_joint in heldout_by_speaker.items():
        train_joint = phone_token.copy()
        for key, count in heldout_joint.items():
            train_joint[key] -= count
            if train_joint[key] <= 0:
                del train_joint[key]
        train_by_token = defaultdict(Counter)
        train_phone_counts = Counter()
        for (phone, token), count in train_joint.items():
            train_by_token[token][phone] += count
            train_phone_counts[phone] += count
        fallback = train_phone_counts.most_common(1)[0][0] if train_phone_counts else global_majority
        mapping = {token: counts.most_common(1)[0][0] for token, counts in train_by_token.items()}
        for (phone, token), count in heldout_joint.items():
            if mapping.get(token, fallback) == phone:
                loso_correct += count
            loso_total += count

    mapping_rows = []
    phones_by_token = defaultdict(Counter)
    for (phone, token), count in phone_token.items():
        phones_by_token[token][phone] += count
    for token in sorted(phones_by_token):
        counts = phones_by_token[token]
        total = sum(counts.values())
        dominant, dominant_count = counts.most_common(1)[0]
        mapping_rows.append({
            "variant": variant,
            "token_id": token,
            "frames": total,
            "dominant_phone": dominant,
            "token_phone_purity": dominant_count / total,
            "num_phone_classes": len(counts),
            "phone_distribution": json.dumps(dict(counts), sort_keys=True),
        })

    return {
        "variant": variant,
        "num_valid_frames": num_frames,
        "active_vocab": active_vocab,
        "codebook_perplexity": codebook_perplexity,
        "normalized_codebook_perplexity": codebook_perplexity / max(active_vocab, 1),
        "phone_purity": phone_purity,
        "cluster_purity": cluster_purity,
        "pnmi": pnmi,
        "silence_binary_purity": silence_purity,
        "speaker_nmi": speaker_nmi,
        "conditional_speaker_nmi_given_phone": conditional_mi / max(conditional_entropy, 1e-12),
        "loso_token_to_phone_accuracy": loso_correct / max(loso_total, 1),
    }, mapping_rows


# -----------------------------------------------------------------------------
# Discrete ABX-style metric
# -----------------------------------------------------------------------------

def collapse_consecutive(values: Sequence[int]) -> Tuple[int, ...]:
    if not values:
        return tuple()
    output = [int(values[0])]
    for value in values[1:]:
        value = int(value)
        if value != output[-1]:
            output.append(value)
    return tuple(output)


def triphone_occurrences(records: List[Dict[str, Any]], exclude_silence: bool) -> List[Dict[str, Any]]:
    output = []
    occurrence_id = 0
    for record in records:
        phone_runs = runs(record["phone_labels_39"])
        tokens = record["token_ids"].tolist()
        for index in range(1, len(phone_runs) - 1):
            left_start, _, left = phone_runs[index - 1]
            _, _, center = phone_runs[index]
            _, right_end, right = phone_runs[index + 1]
            if left in IGNORE_PHONES or center in IGNORE_PHONES or right in IGNORE_PHONES:
                continue
            if exclude_silence and (left in SILENCE_39 or center in SILENCE_39 or right in SILENCE_39):
                continue
            sequence = collapse_consecutive(tokens[left_start:right_end])
            if not sequence:
                continue
            output.append({
                "id": occurrence_id,
                "context": (left, right),
                "center": center,
                "speaker": record["speaker_id"],
                "sequence": sequence,
            })
            occurrence_id += 1
    return output


def normalized_levenshtein(first: Tuple[int, ...], second: Tuple[int, ...]) -> float:
    if first == second:
        return 0.0
    if not first or not second:
        return 1.0
    previous = list(range(len(second) + 1))
    for row_index, first_value in enumerate(first, 1):
        current = [row_index]
        for column_index, second_value in enumerate(second, 1):
            current.append(min(
                current[-1] + 1,
                previous[column_index] + 1,
                previous[column_index - 1] + (first_value != second_value),
            ))
        previous = current
    return previous[-1] / max(len(first), len(second), 1)


def discrete_abx(
    occurrences: List[Dict[str, Any]],
    variant: str,
    max_triplets: int,
    seed: int,
) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    grouped = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    by_id = {}
    for occurrence in occurrences:
        grouped[occurrence["context"]][occurrence["center"]][occurrence["speaker"]].append(occurrence)
        by_id[occurrence["id"]] = occurrence

    pair_specs = []
    for context, centers in grouped.items():
        labels = sorted(centers)
        for first in range(len(labels)):
            for second in range(first + 1, len(labels)):
                pair_specs.append((context, labels[first], labels[second]))
    if not pair_specs:
        return []

    budget = max(1, max_triplets // max(len(pair_specs), 1))
    triplets = {"within": [], "across": []}
    for context, first_center, second_center in pair_specs:
        for target, contrast in [(first_center, second_center), (second_center, first_center)]:
            target_by_speaker = grouped[context][target]
            contrast_by_speaker = grouped[context][contrast]

            valid_within = [
                speaker for speaker in target_by_speaker
                if len(target_by_speaker[speaker]) >= 2 and contrast_by_speaker.get(speaker)
            ]
            for _ in range(budget):
                if not valid_within:
                    break
                speaker = rng.choice(valid_within)
                anchor, query = rng.sample(target_by_speaker[speaker], 2)
                negative = rng.choice(contrast_by_speaker[speaker])
                triplets["within"].append((anchor["id"], negative["id"], query["id"], context, target, contrast))

            base_speakers = [
                speaker for speaker in target_by_speaker
                if target_by_speaker[speaker] and contrast_by_speaker.get(speaker)
            ]
            valid_across = [
                (source, query_speaker)
                for source in base_speakers
                for query_speaker in target_by_speaker
                if source != query_speaker and target_by_speaker[query_speaker]
            ]
            for _ in range(budget):
                if not valid_across:
                    break
                source, query_speaker = rng.choice(valid_across)
                anchor = rng.choice(target_by_speaker[source])
                negative = rng.choice(contrast_by_speaker[source])
                query = rng.choice(target_by_speaker[query_speaker])
                triplets["across"].append((anchor["id"], negative["id"], query["id"], context, target, contrast))

    distance_cache: Dict[Tuple[int, int], float] = {}
    def distance(first_id: int, second_id: int) -> float:
        key = (min(first_id, second_id), max(first_id, second_id))
        if key not in distance_cache:
            distance_cache[key] = normalized_levenshtein(
                by_id[first_id]["sequence"], by_id[second_id]["sequence"]
            )
        return distance_cache[key]

    rows = []
    for condition, samples in triplets.items():
        if len(samples) > max_triplets:
            samples = rng.sample(samples, max_triplets)
        errors = []
        errors_by_pair = defaultdict(list)
        for anchor_id, negative_id, query_id, context, target, contrast in samples:
            same_distance = distance(anchor_id, query_id)
            contrast_distance = distance(negative_id, query_id)
            error = 1.0 if same_distance > contrast_distance else 0.5 if same_distance == contrast_distance else 0.0
            errors.append(error)
            errors_by_pair[(context, tuple(sorted((target, contrast))))].append(error)
        if not errors:
            continue
        rows.append({
            "variant": variant,
            "condition": condition,
            "num_occurrences": len(occurrences),
            "num_context_phone_pairs": len(errors_by_pair),
            "num_sampled_triplets": len(errors),
            "abx_error_macro": float(np.mean([np.mean(values) for values in errors_by_pair.values()])),
            "abx_error_micro": float(np.mean(errors)),
        })
    return rows


# -----------------------------------------------------------------------------
# Per-variant evaluation
# -----------------------------------------------------------------------------

def summarize_per_item(rows: List[Dict[str, Any]], variant: str) -> List[Dict[str, Any]]:
    identity = {"variant", "path", "file_id", "speaker_id", "item_index", "side", "num_frames"}
    metrics = sorted({
        key for row in rows for key, value in row.items()
        if key not in identity and isinstance(value, (int, float, np.integer, np.floating))
    })
    output = []
    for metric in metrics:
        values = np.asarray([row.get(metric, np.nan) for row in rows], dtype=np.float64)
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        output.append({
            "variant": variant,
            "metric": metric,
            "n": int(values.size),
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "median": float(np.median(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        })
    return output


@torch.no_grad()
def evaluate_variant(
    model,
    loader,
    variant: str,
    output_root: str,
    device: torch.device,
    fs: int,
    hop_samples: int,
    win_samples: int,
    default_segment_samples: int,
    expected_frames: int,
    max_len: int,
    beam_size: int,
    sides: Sequence[str],
    stft_center: bool,
    q_policy: str,
    max_items: Optional[int],
    write_frame_csv: bool,
    skip_abx: bool,
    abx_max_triplets: int,
    abx_include_silence: bool,
    seed: int,
) -> Dict[str, List[Dict[str, Any]]]:
    model.eval().to(device)
    variant_dir = os.path.join(output_root, variant)
    aligned_dir = os.path.join(variant_dir, "aligned_sequences")
    os.makedirs(aligned_dir, exist_ok=True)

    records, per_item_rows = [], []
    frame_file = frame_writer = None
    if write_frame_csv:
        frame_file = gzip.open(os.path.join(variant_dir, "frame_alignment.csv.gz"), "wt", newline="")
        frame_writer = csv.DictWriter(frame_file, fieldnames=[
            "variant", "item_index", "file_id", "speaker_id", "side",
            "encoder_frame", "time_sec", "phone_61", "phone_39", "token_id",
        ])
        frame_writer.writeheader()

    global_index = 0
    progress = tqdm(loader, desc=f"Analyzing {variant}", unit="batch")
    try:
        for anc_features, pos_features, metadata_batch in progress:
            if max_items is not None and global_index >= max_items:
                break
            anc_features = anc_features.to(device, non_blocking=True)
            pos_features = pos_features.to(device, non_blocking=True)
            token_batches: Dict[str, List[torch.LongTensor]] = {}
            if "anc" in sides:
                token_batches["anc"] = tokenize_geometric_batch(
                    model, anc_features, expected_frames, max_len, beam_size
                )
            if "pos" in sides:
                token_batches["pos"] = tokenize_geometric_batch(
                    model, pos_features, expected_frames, max_len, beam_size
                )

            for batch_index in range(anc_features.size(0)):
                if max_items is not None and global_index >= max_items:
                    break
                meta = dict(metadata_batch[batch_index])
                audio_path = infer_audio_path(meta)
                phn_path = infer_phn_path(audio_path, meta.get("phn_path"))
                speaker_id = infer_speaker_id(audio_path, meta.get("speaker_id"))
                file_id = str(meta.get("file_id") or f"{speaker_id}/{Path(audio_path).stem}")
                phn_records = read_timit_phn(phn_path)

                for side in sides:
                    if not side_is_time_preserving(meta, side):
                        continue
                    token_ids = token_batches[side][batch_index]
                    crop_start = resolve_crop_start(meta, side)
                    segment_samples = int(meta.get("segment_num_samples", default_segment_samples))
                    alignment = align_phn_to_frames(
                        phn_records, crop_start, expected_frames, hop_samples,
                        win_samples, segment_samples, stft_center, q_policy,
                    )
                    record = {
                        "variant": variant,
                        "item_index": global_index + 1,
                        "file_id": file_id,
                        "speaker_id": speaker_id,
                        "side": side,
                        "token_ids": token_ids.detach().cpu(),
                        "phone_labels_61": alignment["phone_labels_61"],
                        "phone_labels_39": alignment["phone_labels_39"],
                        "phone_boundary_frames_float": alignment["phone_boundary_frames_float"],
                        "phone_boundary_samples_in_crop": alignment["phone_boundary_samples_in_crop"],
                        "crop_start_sample": crop_start,
                        "hop_ms": 1000.0 * hop_samples / fs,
                    }
                    records.append(record)
                    torch.save(record, os.path.join(aligned_dir, f"item_{global_index + 1:06d}_{side}.pt"))

                    item_row = {
                        "variant": variant,
                        "path": audio_path,
                        "file_id": file_id,
                        "speaker_id": speaker_id,
                        "item_index": global_index + 1,
                        "side": side,
                        "num_frames": token_ids.numel(),
                    }
                    item_row.update(boundary_metrics(
                        token_ids, alignment["phone_boundary_frames_float"], record["hop_ms"]
                    ))
                    item_row.update(segment_metrics(token_ids, alignment["phone_labels_39"]))
                    per_item_rows.append(item_row)

                    if frame_writer is not None:
                        for frame in range(expected_frames):
                            frame_writer.writerow({
                                "variant": variant,
                                "item_index": global_index + 1,
                                "file_id": file_id,
                                "speaker_id": speaker_id,
                                "side": side,
                                "encoder_frame": frame,
                                "time_sec": frame * hop_samples / fs,
                                "phone_61": alignment["phone_labels_61"][frame],
                                "phone_39": alignment["phone_labels_39"][frame],
                                "token_id": int(token_ids[frame]),
                            })
                global_index += 1
                progress.set_postfix(items=global_index)
    finally:
        if frame_file is not None:
            frame_file.close()

    if not records:
        raise RuntimeError(f"No valid aligned records produced for {variant}.")

    write_csv(os.path.join(variant_dir, "per_item_alignment_metrics.csv"), per_item_rows)
    alignment_summary = summarize_per_item(per_item_rows, variant)
    write_csv(os.path.join(variant_dir, "alignment_summary.csv"), alignment_summary)
    hubert_metrics, mapping_rows = global_metrics(records, variant)
    write_csv(os.path.join(variant_dir, "global_hubert_metrics.csv"), [hubert_metrics])
    write_csv(os.path.join(variant_dir, "token_phone_mapping.csv"), mapping_rows)

    abx_rows: List[Dict[str, Any]] = []
    if not skip_abx:
        occurrences = triphone_occurrences(records, exclude_silence=not abx_include_silence)
        abx_rows = discrete_abx(occurrences, variant, abx_max_triplets, seed)
        write_csv(os.path.join(variant_dir, "abx_discrete_metrics.csv"), abx_rows)

    print(f"[{variant}] aligned records: {len(records)}")
    print(f"[{variant}] output: {variant_dir}")
    return {
        "alignment_summary": alignment_summary,
        "global_metrics": [hubert_metrics],
        "abx": abx_rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", default=os.path.dirname(os.getcwd()))
    parser.add_argument("--config", default="mainFull.yaml")
    parser.add_argument("--split", default="valid", choices=["train", "valid", "test"])
    parser.add_argument("--output_dir", default="geometric_phone_analysis_TIMIT")

    parser.add_argument("--stage1_ckpt", default=None, help="AudioTokenizer checkpoint.")
    parser.add_argument("--stage1_plus_ckpt", default=None, help="AudioTokenizer checkpoint for Stage I+ Geometric.")

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--max_items", type=int, default=None)
    parser.add_argument("--expected_frames", type=int, default=301)
    parser.add_argument("--max_len", type=int, default=301)
    parser.add_argument("--beam_size", type=int, default=4)
    parser.add_argument("--sides", default="anc", help="anc, pos, or anc,pos")
    parser.add_argument("--stft_center", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--q_policy", choices=["ignore", "sil"], default="ignore")
    parser.add_argument("--add_noise", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use_time_compression", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--write_frame_csv", action="store_true")
    parser.add_argument("--skip_abx", action="store_true")
    parser.add_argument("--abx_max_triplets", type=int, default=100000)
    parser.add_argument("--abx_include_silence", action="store_true")
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    if args.stage1_ckpt is None and args.stage1_plus_ckpt is None:
        parser.error("Provide --stage1_ckpt and/or --stage1_plus_ckpt.")

    sides = [value.strip() for value in args.sides.split(",") if value.strip()]
    for side in sides:
        if side not in {"anc", "pos"}:
            raise ValueError(f"Invalid side: {side}")

    config_path = os.path.join(args.root_dir, "config", args.config)
    with open(config_path, "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    if args.split not in cfg:
        raise ValueError(f"Split {args.split!r} not found in config.")

    seed_everything(args.seed, workers=True)
    split_cfg = cfg[args.split]
    dataset_csv = split_cfg["dataset_csv"]
    batch_size = args.batch_size if args.batch_size is not None else int(cfg["batchsize"])
    num_workers = args.num_workers if args.num_workers is not None else int(cfg["load_workers"])

    fs = int(cfg["fs"])
    hop_samples = int(round(float(cfg["win_hop"]) * fs))
    win_samples = int(round(float(cfg["win_size"]) * fs))
    segment_samples = int(round(float(cfg["seglen"]) * fs))

    dataset = AudioPairAugDataset(
        audio_csv=dataset_csv,
        fs=fs,
        context_sec=cfg["seglen"],
        add_noise=args.add_noise,
        use_time_compression=args.use_time_compression,
        return_path=True,
        return_metadata=True,
        feats_type="melspectrogram",
        rf_size=win_samples,
        stride=hop_samples,
        n_mels=cfg["n_mels"],
        split="train" if args.split == "train" else "val",
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=pair_collate,
    )

    # Validate the PHN-aware dataset interface before loading large checkpoints.
    _, _, first_meta = item_to_features_and_meta(dataset[0])
    missing = [key for key in ["audio_path", "crop_start_sample"] if key not in first_meta]
    if missing:
        raise RuntimeError(
            f"AudioPairAugDataset metadata is incomplete: missing {missing}. "
            "Use the PHN-aware return_metadata=True dataset implementation."
        )

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    variants = []
    if args.stage1_ckpt:
        variants.append(("Stage_I_Geometric", args.stage1_ckpt))
    if args.stage1_plus_ckpt:
        variants.append(("Stage_I_Plus_Geometric", args.stage1_plus_ckpt))

    os.makedirs(args.output_dir, exist_ok=True)
    combined_alignment, combined_hubert, combined_abx = [], [], []

    print(f"Config: {config_path}")
    print(f"Dataset: {dataset_csv}")
    print(f"Sides: {sides}")
    print(f"Expected framewise token length: {args.expected_frames}")
    print("Both Stage I and Stage I+ are loaded with AudioTokenizer.")
    print("Geometric tokenize call uses deduplicate=False.")

    for variant, checkpoint in variants:
        print("=" * 80)
        print(f"Variant: {variant}")
        print(f"Checkpoint: {checkpoint}")
        model = build_model(checkpoint, args.device)
        result = evaluate_variant(
            model=model,
            loader=loader,
            variant=variant,
            output_root=args.output_dir,
            device=device,
            fs=fs,
            hop_samples=hop_samples,
            win_samples=win_samples,
            default_segment_samples=segment_samples,
            expected_frames=args.expected_frames,
            max_len=args.max_len,
            beam_size=args.beam_size,
            sides=sides,
            stft_center=args.stft_center,
            q_policy=args.q_policy,
            max_items=args.max_items,
            write_frame_csv=args.write_frame_csv,
            skip_abx=args.skip_abx,
            abx_max_triplets=args.abx_max_triplets,
            abx_include_silence=args.abx_include_silence,
            seed=args.seed,
        )
        combined_alignment.extend(result["alignment_summary"])
        combined_hubert.extend(result["global_metrics"])
        combined_abx.extend(result["abx"])
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    write_csv(os.path.join(args.output_dir, "geometric_alignment_summary.csv"), combined_alignment)
    write_csv(os.path.join(args.output_dir, "geometric_global_hubert_metrics.csv"), combined_hubert)
    if combined_abx:
        write_csv(os.path.join(args.output_dir, "geometric_abx_discrete_metrics.csv"), combined_abx)

    print("=" * 80)
    print(f"Combined alignment summary: {os.path.join(args.output_dir, 'geometric_alignment_summary.csv')}")
    print(f"Combined HuBERT-style metrics: {os.path.join(args.output_dir, 'geometric_global_hubert_metrics.csv')}")
    if combined_abx:
        print(f"Combined ABX metrics: {os.path.join(args.output_dir, 'geometric_abx_discrete_metrics.csv')}")


if __name__ == "__main__":
    main()
