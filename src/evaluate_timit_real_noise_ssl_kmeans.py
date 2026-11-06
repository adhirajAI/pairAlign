#!/usr/bin/env python3
"""Evaluate HuBERT/WavLM K-means tokens on TIMIT + external real noise.

This script uses the exact TIMIT + ``noise_only`` mixing implementation from
``evaluate_timit_real_noise_pairwise_consistency.py``:

Anchor
    A deterministic clean 3-second TIMIT crop.

Positive
    The same crop mixed with a real ``noise_only`` recording from
    ``haydarkadioglu/speech-noise-dataset`` after active-RMS normalization.

No other augmentation is applied. Consecutive duplicate SSL K-means tokens are
removed before all sequence-level measurements, matching the supplied SSL
baseline code.

Outputs
-------
* ``<output_prefix>_<ssl_model>.csv``: per-pair tokenizations and diagnostics.
* ``<output_prefix>_<ssl_model>_summary.json``: complete aggregate results.
* ``<output_prefix>_<ssl_model>_summary.csv``: one-row table-ready summary.

The summary contains the metrics required by both manuscript tables:

Consistency/compactness
    Jaccard similarity, edit similarity, exact-match rate, mean sequence
    length, and active vocabulary size.

Collapse behavior
    Anchor/positive low-diversity sequence rates, collapsed-pair rate, and
    anchor/positive exact-collision rates.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import torch
import yaml
from huggingface_hub import hf_hub_download
from lightning.pytorch import seed_everything
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from speechbrain.integrations.huggingface.hubert import HuBERT
from speechbrain.integrations.huggingface.wavlm import WavLM

try:
    from evaluate_timit_real_noise_pairwise_consistency import (
        DEFAULT_NOISE_DIR,
        TimitNoisePairDataset,
        collate_pairs,
        edit_distance,
        find_parquet_files,
        index_noise_only,
        load_local_noise_dataset,
        read_audio_paths,
        resolve_config_path,
        resolve_dataset_csv,
        serialize_tokens,
        token_jaccard,
    )
except ImportError as exc:
    raise ImportError(
        "Place this SSL evaluator and "
        "evaluate_timit_real_noise_pairwise_consistency.py in the same "
        "directory, or add that directory to PYTHONPATH."
    ) from exc


SSL_MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "hubert": {
        "class": HuBERT,
        "source": "facebook/hubert-base-ls960",
        "kmeans_name": "hubert",
    },
    # chaanks/Kmeans LibriSpeech_wavlm_k512_L7.pt has 1024-dimensional
    # centers, so it must be paired with WavLM Large rather than Base.
    "wavlm": {
        "class": WavLM,
        "source": "microsoft/wavlm-large",
        "kmeans_name": "wavlm",
    },
}


@dataclass(frozen=True)
class SSLKMeansConfig:
    ssl_model: str
    k: int = 512
    layer_num: int = 7
    kmeans_dataset: str = "LibriSpeech"
    kmeans_repo_id: str = "chaanks/Kmeans"
    save_root: str = "pretrained_ssl"


# -----------------------------------------------------------------------------
# K-means and hidden-state helpers
# -----------------------------------------------------------------------------


def load_kmeans_centers(path: str) -> torch.Tensor:
    """Load centers from the sklearn/joblib objects in chaanks/Kmeans."""
    obj = joblib.load(path)

    if hasattr(obj, "cluster_centers_"):
        centers = obj.cluster_centers_
    elif isinstance(obj, dict):
        centers = None
        for key in (
            "cluster_centers_",
            "cluster_centers",
            "centroids",
            "centers",
            "kmeans",
            "model",
        ):
            if key not in obj:
                continue
            value = obj[key]
            if hasattr(value, "cluster_centers_"):
                centers = value.cluster_centers_
                break
            if isinstance(value, np.ndarray) or torch.is_tensor(value):
                centers = value
                break
        if centers is None:
            raise RuntimeError(
                f"Could not find K-means centers in dictionary keys: "
                f"{sorted(obj.keys())}"
            )
    elif isinstance(obj, np.ndarray) or torch.is_tensor(obj):
        centers = obj
    else:
        raise RuntimeError(
            f"Could not extract K-means centers from object type {type(obj)}"
        )

    centers_tensor = torch.as_tensor(centers, dtype=torch.float32)
    if centers_tensor.ndim != 2:
        raise ValueError(
            f"Expected K-means centers [K, D], got {tuple(centers_tensor.shape)}"
        )
    return centers_tensor.contiguous()


def deduplicate_consecutive_tokens(tokens: torch.Tensor) -> torch.Tensor:
    tokens = tokens.detach().cpu().reshape(-1)
    if tokens.numel() == 0:
        return tokens
    keep = torch.ones(tokens.numel(), dtype=torch.bool)
    keep[1:] = tokens[1:] != tokens[:-1]
    return tokens[keep]


def _find_hidden_sequence(value: Any) -> Optional[List[torch.Tensor]]:
    """Find a nested sequence of layer tensors shaped [B, T, D]."""
    if isinstance(value, (list, tuple)):
        tensors = [item for item in value if torch.is_tensor(item)]
        if tensors and len(tensors) == len(value) and all(
            item.ndim == 3 for item in tensors
        ):
            return tensors

        # SpeechBrain/Transformers versions sometimes return a tuple containing
        # the last hidden state and a nested tuple of all hidden states.
        for item in reversed(value):
            result = _find_hidden_sequence(item)
            if result is not None:
                return result

    if isinstance(value, dict):
        for key in ("hidden_states", "all_hiddens", "states"):
            if key in value:
                result = _find_hidden_sequence(value[key])
                if result is not None:
                    return result

    return None


def select_ssl_layer(output: Any, layer_num: int) -> torch.Tensor:
    """Return one hidden layer as [B, T, D] across SpeechBrain versions."""
    if torch.is_tensor(output):
        if output.ndim == 4:
            # Most SpeechBrain wrappers return [L, B, T, D]. A minority return
            # [B, L, T, D]; distinguish using the small layer dimension.
            if output.shape[0] <= 64:
                if layer_num >= output.shape[0]:
                    raise IndexError(
                        f"Requested layer {layer_num}, but output has "
                        f"{output.shape[0]} layers"
                    )
                return output[layer_num]
            if output.shape[1] <= 64:
                if layer_num >= output.shape[1]:
                    raise IndexError(
                        f"Requested layer {layer_num}, but output has "
                        f"{output.shape[1]} layers"
                    )
                return output[:, layer_num]
        if output.ndim == 3:
            if layer_num not in (0, -1):
                raise RuntimeError(
                    "SSL wrapper returned only one hidden tensor [B,T,D], "
                    f"so layer {layer_num} cannot be selected. Ensure "
                    "output_all_hiddens=True is supported by this SpeechBrain "
                    "version."
                )
            return output

    hidden_sequence = _find_hidden_sequence(output)
    if hidden_sequence is not None:
        if layer_num >= len(hidden_sequence):
            raise IndexError(
                f"Requested layer {layer_num}, but wrapper returned "
                f"{len(hidden_sequence)} hidden tensors"
            )
        return hidden_sequence[layer_num]

    if isinstance(output, dict):
        for key in ("hidden_states", "all_hiddens", "states"):
            if key in output:
                return select_ssl_layer(output[key], layer_num)

    raise RuntimeError(
        "Could not identify all-layer SSL output. Received type "
        f"{type(output)}."
    )


class SSLKMeansTokenizer(torch.nn.Module):
    def __init__(self, config: SSLKMeansConfig, device: torch.device):
        super().__init__()
        self.config = config
        self.device = device

        model_name = config.ssl_model.lower()
        if model_name not in SSL_MODEL_REGISTRY:
            raise ValueError(
                f"Unknown SSL model {model_name!r}; choose from "
                f"{sorted(SSL_MODEL_REGISTRY)}"
            )

        model_info = SSL_MODEL_REGISTRY[model_name]
        model_class = model_info["class"]
        source = model_info["source"]
        kmeans_name = model_info["kmeans_name"]

        self.kmeans_filename = (
            f"{config.kmeans_dataset}_{kmeans_name}_k{config.k}_"
            f"L{config.layer_num}.pt"
        )

        print(f"SSL model:       {model_name}")
        print(f"HF source:       {source}")
        print(f"SSL layer:       {config.layer_num}")
        print(f"K-means repo:    {config.kmeans_repo_id}")
        print(f"K-means file:    {self.kmeans_filename}")

        self.ssl = model_class(
            source=source,
            save_path=os.path.join(config.save_root, model_name),
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
        self.register_buffer(
            "centers",
            centers.to(device),
            persistent=False,
        )
        self.register_buffer(
            "centers_squared_norm",
            self.centers.pow(2).sum(dim=1),
            persistent=False,
        )
        print(f"K-means centers: {tuple(self.centers.shape)}")

    @torch.no_grad()
    def tokenize(self, waveforms: torch.Tensor) -> List[List[int]]:
        """Quantize [B,T] 16-kHz waveforms and collapse adjacent repeats."""
        if waveforms.ndim != 2:
            raise ValueError(
                f"Expected waveform batch [B,T], got {tuple(waveforms.shape)}"
            )

        waveforms = waveforms.to(self.device, non_blocking=True)
        output = self.ssl(waveforms)
        features = select_ssl_layer(output, self.config.layer_num)

        if features.ndim != 3:
            raise RuntimeError(
                f"Expected selected features [B,T,D], got {features.shape}"
            )
        if features.shape[-1] != self.centers.shape[-1]:
            raise RuntimeError(
                f"Feature dimension mismatch: SSL={features.shape[-1]}, "
                f"K-means={self.centers.shape[-1]}. Check the SSL source and "
                "K-means filename."
            )

        # Squared Euclidean assignment without materializing [T,K,D].
        tokens: List[List[int]] = []
        centers_t = self.centers.float().transpose(0, 1)
        centers_norm = self.centers_squared_norm.float()

        for item_features in features:
            item_features = item_features.float()
            distances = (
                item_features.pow(2).sum(dim=1, keepdim=True)
                + centers_norm.unsqueeze(0)
                - 2.0 * item_features @ centers_t
            )
            frame_tokens = distances.argmin(dim=1)
            collapsed = deduplicate_consecutive_tokens(frame_tokens)
            tokens.append([int(x) for x in collapsed.tolist()])

        return tokens


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------


def sequence_unique_ratio(tokens: Sequence[int]) -> float:
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


def exact_collision_rate(sequences: Sequence[Sequence[int]]) -> float:
    if not sequences:
        return 0.0
    unique_sequences = {tuple(sequence) for sequence in sequences}
    return 1.0 - len(unique_sequences) / len(sequences)


def describe(values: Sequence[float], prefix: str) -> Dict[str, Optional[float]]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {
            f"{prefix}_mean": None,
            f"{prefix}_std": None,
            f"{prefix}_median": None,
            f"{prefix}_q25": None,
            f"{prefix}_q75": None,
        }
    return {
        f"{prefix}_mean": float(array.mean()),
        f"{prefix}_std": float(array.std()),
        f"{prefix}_median": float(np.median(array)),
        f"{prefix}_q25": float(np.quantile(array, 0.25)),
        f"{prefix}_q75": float(np.quantile(array, 0.75)),
    }


OUTPUT_FIELDS = [
    # Exact metadata schema returned by TimitNoisePairDataset.
    "file_id",
    "anchor_path",
    "noise_filename",
    "noise_row_index",
    "speech_original_sr",
    "noise_original_sr",
    "speech_crop_start",
    "noise_crop_start",
    "valid_speech_samples",
    "requested_snr_db",
    "achieved_snr_db",
    "speech_normalization_gain_db",
    "noise_normalization_gain_db",
    "common_peak_scale",
    "mixture_peak",
    "time_compression_applied",
    "time_compression_rate",
    "positive_right_pad_samples",

    # SSL token and pairwise-consistency diagnostics.
    "anc_tokenization",
    "pos_tokenization",
    "anc_length",
    "pos_length",
    "anc_unique_token_ratio",
    "pos_unique_token_ratio",
    "anc_low_diversity",
    "pos_low_diversity",
    "collapsed_pair",
    "absolute_length_difference",
    "pos_to_anc_length_ratio",
    "anc_pos_edit_distance",
    "anc_pos_normalized_edit_distance",
    "anc_pos_edit_similarity",
    "anc_pos_exact_match",
    "anc_pos_token_set_jaccard",
]


@torch.no_grad()
def evaluate_ssl_tokenizer(
    tokenizer: SSLKMeansTokenizer,
    loader: DataLoader,
    output_csv: str,
    summary_json: str,
    summary_csv: str,
    low_diversity_threshold: float,
    metadata: Dict[str, Any],
) -> None:
    tokenizer.eval()
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)

    anchor_sequences: List[List[int]] = []
    positive_sequences: List[List[int]] = []
    anchor_lengths: List[float] = []
    positive_lengths: List[float] = []
    all_lengths: List[float] = []
    edit_similarities: List[float] = []
    normalized_edit_distances: List[float] = []
    exact_matches: List[float] = []
    jaccards: List[float] = []
    length_ratios: List[float] = []
    length_differences: List[float] = []
    achieved_snrs: List[float] = []
    anchor_low_diversity: List[float] = []
    positive_low_diversity: List[float] = []
    collapsed_pairs: List[float] = []
    active_tokens: set[int] = set()

    written = 0
    with open(output_csv, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()

        progress = tqdm(
            loader,
            desc=f"Generating {metadata['ssl_model']} tokens",
            unit="batch",
        )
        for batch in progress:
            anchor_waveforms = batch["anchor_wav"]
            positive_waveforms = batch["positive_wav"]
            if anchor_waveforms.ndim == 3:
                anchor_waveforms = anchor_waveforms.mean(dim=1)
            if positive_waveforms.ndim == 3:
                positive_waveforms = positive_waveforms.mean(dim=1)

            anchor_tokens = tokenizer.tokenize(anchor_waveforms)
            positive_tokens = tokenizer.tokenize(positive_waveforms)

            if not (
                len(anchor_tokens)
                == len(positive_tokens)
                == len(batch["metadata"])
            ):
                raise RuntimeError("Anchor/positive SSL token batch mismatch")

            for item_meta, anchor, positive in zip(
                batch["metadata"], anchor_tokens, positive_tokens
            ):
                distance = edit_distance(anchor, positive)
                normalized_distance = distance / max(len(anchor), len(positive), 1)
                edit_similarity = 1.0 - normalized_distance
                exact_match = int(anchor == positive)
                jaccard = token_jaccard(anchor, positive)
                length_difference = abs(len(anchor) - len(positive))
                length_ratio = len(positive) / max(len(anchor), 1)

                anchor_unique_ratio = sequence_unique_ratio(anchor)
                positive_unique_ratio = sequence_unique_ratio(positive)
                anchor_is_low = int(
                    anchor_unique_ratio <= low_diversity_threshold
                )
                positive_is_low = int(
                    positive_unique_ratio <= low_diversity_threshold
                )
                pair_is_collapsed = int(anchor_is_low or positive_is_low)

                required_pair_metadata = {
                    "file_id",
                    "anchor_path",
                    "noise_filename",
                    "noise_row_index",
                    "speech_original_sr",
                    "noise_original_sr",
                    "speech_crop_start",
                    "noise_crop_start",
                    "valid_speech_samples",
                    "requested_snr_db",
                    "achieved_snr_db",
                    "speech_normalization_gain_db",
                    "noise_normalization_gain_db",
                    "common_peak_scale",
                    "mixture_peak",
                }
                missing_pair_metadata = required_pair_metadata.difference(
                    item_meta
                )
                if missing_pair_metadata:
                    raise KeyError(
                        "SSL evaluator received an unexpected pair-metadata "
                        f"schema. Missing: {sorted(missing_pair_metadata)}; "
                        f"available: {sorted(item_meta.keys())}"
                    )

                writer.writerow({
                    # Use the exact metadata keys returned by the PairAlign
                    # TimitNoisePairDataset implementation.
                    **item_meta,
                    "requested_snr_db": (
                        f"{float(item_meta['requested_snr_db']):.8f}"
                    ),
                    "achieved_snr_db": (
                        f"{float(item_meta['achieved_snr_db']):.8f}"
                    ),
                    "speech_normalization_gain_db": (
                        f"{float(item_meta['speech_normalization_gain_db']):.8f}"
                    ),
                    "noise_normalization_gain_db": (
                        f"{float(item_meta['noise_normalization_gain_db']):.8f}"
                    ),
                    "common_peak_scale": (
                        f"{float(item_meta['common_peak_scale']):.8f}"
                    ),
                    "mixture_peak": (
                        f"{float(item_meta['mixture_peak']):.8f}"
                    ),
                    "anc_tokenization": serialize_tokens(anchor),
                    "pos_tokenization": serialize_tokens(positive),
                    "anc_length": len(anchor),
                    "pos_length": len(positive),
                    "anc_unique_token_ratio": f"{anchor_unique_ratio:.8f}",
                    "pos_unique_token_ratio": f"{positive_unique_ratio:.8f}",
                    "anc_low_diversity": anchor_is_low,
                    "pos_low_diversity": positive_is_low,
                    "collapsed_pair": pair_is_collapsed,
                    "absolute_length_difference": length_difference,
                    "pos_to_anc_length_ratio": f"{length_ratio:.8f}",
                    "anc_pos_edit_distance": distance,
                    "anc_pos_normalized_edit_distance": (
                        f"{normalized_distance:.8f}"
                    ),
                    "anc_pos_edit_similarity": f"{edit_similarity:.8f}",
                    "anc_pos_exact_match": exact_match,
                    "anc_pos_token_set_jaccard": f"{jaccard:.8f}",
                })

                anchor_sequences.append(anchor)
                positive_sequences.append(positive)
                anchor_lengths.append(float(len(anchor)))
                positive_lengths.append(float(len(positive)))
                all_lengths.extend((float(len(anchor)), float(len(positive))))
                edit_similarities.append(edit_similarity)
                normalized_edit_distances.append(normalized_distance)
                exact_matches.append(float(exact_match))
                jaccards.append(jaccard)
                length_ratios.append(length_ratio)
                length_differences.append(float(length_difference))
                achieved_snrs.append(float(item_meta["achieved_snr_db"]))
                anchor_low_diversity.append(float(anchor_is_low))
                positive_low_diversity.append(float(positive_is_low))
                collapsed_pairs.append(float(pair_is_collapsed))
                active_tokens.update(anchor)
                active_tokens.update(positive)
                written += 1

            progress.set_postfix(written=written)

    if written == 0:
        raise RuntimeError("No SSL token pairs were evaluated")

    summary = dict(metadata)
    summary.update({
        "num_pairs": written,
        "output_csv": os.path.abspath(output_csv),
        "mean_sequence_length": float(np.mean(all_lengths)),
        "active_vocabulary_size": len(active_tokens),
        "exact_match_rate": float(np.mean(exact_matches)),
        "anchor_low_diversity_sequence_rate": float(
            np.mean(anchor_low_diversity)
        ),
        "positive_low_diversity_sequence_rate": float(
            np.mean(positive_low_diversity)
        ),
        "collapsed_pair_rate": float(np.mean(collapsed_pairs)),
        "anchor_exact_collision_rate": exact_collision_rate(anchor_sequences),
        "positive_exact_collision_rate": exact_collision_rate(
            positive_sequences
        ),
    })
    summary.update(describe(anchor_lengths, "anchor_length"))
    summary.update(describe(positive_lengths, "positive_length"))
    summary.update(describe(edit_similarities, "edit_similarity"))
    summary.update(
        describe(normalized_edit_distances, "normalized_edit_distance")
    )
    summary.update(describe(jaccards, "token_set_jaccard"))
    summary.update(describe(length_ratios, "positive_to_anchor_length_ratio"))
    summary.update(describe(length_differences, "absolute_length_difference"))
    summary.update(describe(achieved_snrs, "achieved_snr_db"))

    os.makedirs(os.path.dirname(summary_json) or ".", exist_ok=True)
    with open(summary_json, "w") as handle:
        json.dump(summary, handle, indent=2)

    table_row = {
        "ssl_model": summary["ssl_model"],
        "source": summary["ssl_source"],
        "layer_num": summary["layer_num"],
        "kmeans_file": summary["kmeans_file"],
        "num_pairs": summary["num_pairs"],
        "jaccard_similarity": summary["token_set_jaccard_mean"],
        "edit_similarity": summary["edit_similarity_mean"],
        "exact_match_rate": summary["exact_match_rate"],
        "mean_sequence_length": summary["mean_sequence_length"],
        "active_vocabulary_size": summary["active_vocabulary_size"],
        "anchor_low_diversity_sequence_rate": summary[
            "anchor_low_diversity_sequence_rate"
        ],
        "positive_low_diversity_sequence_rate": summary[
            "positive_low_diversity_sequence_rate"
        ],
        "collapsed_pair_rate": summary["collapsed_pair_rate"],
        "anchor_exact_collision_rate": summary[
            "anchor_exact_collision_rate"
        ],
        "positive_exact_collision_rate": summary[
            "positive_exact_collision_rate"
        ],
        "mean_achieved_snr_db": summary["achieved_snr_db_mean"],
    }
    os.makedirs(os.path.dirname(summary_csv) or ".", exist_ok=True)
    with open(summary_csv, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table_row.keys()))
        writer.writeheader()
        writer.writerow(table_row)

    print(f"Saved per-pair CSV: {output_csv}")
    print(f"Saved summary JSON: {summary_json}")
    print(f"Saved table-ready CSV: {summary_csv}")
    print(json.dumps(table_row, indent=2))


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_models(value: str) -> List[str]:
    if value == "all":
        return ["hubert", "wavlm"]
    return [value]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate HuBERT/WavLM K-means tokens on TIMIT mixed with "
            "external real noise_only recordings."
        )
    )
    parser.add_argument(
        "--root_dir",
        default=".",
    )
    parser.add_argument("--config", default="mainFull.yaml")
    parser.add_argument(
        "--split",
        choices=["train", "valid", "test"],
        default="valid",
    )
    parser.add_argument("--path_col", default="path")
    parser.add_argument("--noise_dataset_dir", default=DEFAULT_NOISE_DIR)
    parser.add_argument("--hf_cache_dir", default=None)

    parser.add_argument(
        "--ssl_model",
        choices=["hubert", "wavlm", "all"],
        required=True,
    )
    parser.add_argument("--layer_num", type=int, default=7)
    parser.add_argument("--k", type=int, default=512)
    parser.add_argument("--kmeans_dataset", default="LibriSpeech")
    parser.add_argument("--kmeans_repo_id", default="chaanks/Kmeans")
    parser.add_argument("--save_root", default="pretrained_ssl")

    parser.add_argument(
        "--output_prefix",
        required=True,
        help=(
            "Output path without model suffix, e.g. "
            "retrieval_logs/timit_external_noise_ssl"
        ),
    )
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
    parser.add_argument(
        "--use_time_compression",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Slightly compress only the positive view and right-pad it back to 3 seconds.",
    )
    parser.add_argument("--low_diversity_threshold", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    if args.snr_db_max < args.snr_db_min:
        raise ValueError("snr_db_max must be >= snr_db_min")
    if not 0.0 <= args.low_diversity_threshold <= 1.0:
        raise ValueError("low_diversity_threshold must be in [0,1]")

    config_path = resolve_config_path(args.root_dir, args.config)
    with open(config_path, "r") as handle:
        cfg = yaml.load(handle, Loader=yaml.FullLoader)
    if args.split not in cfg:
        raise ValueError(f"Split {args.split!r} not found in config")

    timit_csv = resolve_dataset_csv(
        args.root_dir,
        config_path,
        cfg[args.split]["dataset_csv"],
    )
    timit_paths = read_audio_paths(timit_csv, args.path_col)

    seed_everything(args.seed, workers=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    sample_rate = int(cfg["fs"])
    if sample_rate != 16000:
        raise ValueError(
            f"HuBERT/WavLM baselines require 16 kHz audio; config fs={sample_rate}"
        )
    context_sec = (
        float(args.context_sec)
        if args.context_sec is not None
        else float(cfg["seglen"])
    )
    batch_size = (
        int(args.batch_size)
        if args.batch_size is not None
        else int(cfg["batchsize"])
    )

    parquet_files = find_parquet_files(args.noise_dataset_dir)
    noise_dataset = load_local_noise_dataset(
        parquet_files,
        args.hf_cache_dir,
    )
    noise_rows, noise_diagnostics = index_noise_only(noise_dataset)

    pair_dataset = TimitNoisePairDataset(
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
        pair_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_pairs,
    )

    device = torch.device(
        f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    )

    for model_name in parse_models(args.ssl_model):
        # Re-seed before each model so --ssl_model all uses exactly the same
        # deterministic noise selection and SNR sequence as separate runs.
        seed_everything(args.seed, workers=True)
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

        ssl_config = SSLKMeansConfig(
            ssl_model=model_name,
            k=args.k,
            layer_num=args.layer_num,
            kmeans_dataset=args.kmeans_dataset,
            kmeans_repo_id=args.kmeans_repo_id,
            save_root=args.save_root,
        )
        tokenizer = SSLKMeansTokenizer(ssl_config, device)

        model_info = SSL_MODEL_REGISTRY[model_name]
        output_stem = f"{args.output_prefix}_{model_name}"
        output_csv = output_stem + ".csv"
        summary_json = output_stem + "_summary.json"
        summary_csv = output_stem + "_summary.csv"

        metadata = {
            "ssl_model": model_name,
            "model_type": "SSLKMeansTokenizer",
            "ssl_source": model_info["source"],
            "layer_num": args.layer_num,
            "k": args.k,
            "kmeans_repo_id": args.kmeans_repo_id,
            "kmeans_file": tokenizer.kmeans_filename,
            "config_path": config_path,
            "timit_csv": timit_csv,
            "noise_dataset_dir": os.path.abspath(args.noise_dataset_dir),
            "noise_index": noise_diagnostics,
            "num_timit_files": len(timit_paths),
            "num_items_selected": len(pair_dataset),
            "split": args.split,
            "seed": args.seed,
            "sample_rate": sample_rate,
            "context_sec": context_sec,
            "snr_db_min": args.snr_db_min,
            "snr_db_max": args.snr_db_max,
            "target_rms_dbfs": args.target_rms_dbfs,
            "peak_limit": args.peak_limit,
            "gate_db_below_peak": args.gate_db_below_peak,
            "low_diversity_threshold": args.low_diversity_threshold,
            "deduplicate_consecutive_tokens": True,
            "additional_augmentation": (
                "positive_time_compression"
                if args.use_time_compression
                else "none"
            ),
            "positive_definition": (
                "same TIMIT crop + RMS-normalized real noise_only recording"
            ),
        }

        print(f"==== {model_name.upper()} TIMIT + external noise ====")
        print(json.dumps(metadata, indent=2))
        evaluate_ssl_tokenizer(
            tokenizer=tokenizer,
            loader=loader,
            output_csv=output_csv,
            summary_json=summary_json,
            summary_csv=summary_csv,
            low_diversity_threshold=args.low_diversity_threshold,
            metadata=metadata,
        )

        # Release the large WavLM/HuBERT model before constructing the next one.
        del tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
