#!/usr/bin/env python3
"""
Generate Stage-III decoder cross-attention artifacts with exact TIMIT .PHN
alignment.

The model must use the unified TextDecoder implementation, where
``return_cross_attn=True`` exposes cross-attention without replacing the
training decoder.

IMPORTANT TOKEN/ATTENTION ALIGNMENT
-----------------------------------
The decoder is autoregressive. For a generated sequence

    [BOS, y1, y2, ..., EOS, PAD, ...]

decoder row t predicts token t+1. Therefore artifacts are built from

    decoder_input = indices[:, :-1]
    predicted_token = indices[:, 1:]

and cross-attention row t is associated with ``predicted_token[:, t]``.

This avoids the off-by-one error produced by pairing attention row t with the
input token at the same decoder position.

IMPORTANT DATASET CONTRACT
--------------------------
Exact phone alignment requires the crop start in the ORIGINAL TIMIT/.PHN
waveform timebase. ``AudioPairAugDataset`` must return either:

1) a dict containing anchor/positive features plus metadata, or
2) ``(anc_features, pos_features, audio_path, metadata_dict)``.

Required metadata:
    audio_path
    crop_start_sample (or anc_crop_start_sample)

For positive-view phone alignment, the dataset must additionally declare
``pos_time_preserving=True``. ``pos_crop_start_sample`` may be supplied when
different from the anchor crop.

Optional metadata:
    phn_path
    speaker_id
    file_id
    phn_sample_rate
    segment_num_samples

``crop_start_sample`` is interpreted in the .PHN waveform timebase.
``segment_num_samples`` is interpreted in the feature/model sample-rate
timebase.
"""

import argparse
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader
from lightning.pytorch import seed_everything
from tqdm.auto import tqdm
import yaml

from train.trainer_autoreg_pairalign import AudioTokenizerAR_FINAL
from utils import AudioPairAugDataset


# -----------------------------------------------------------------------------
# TIMIT phone mapping
# -----------------------------------------------------------------------------


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


# -----------------------------------------------------------------------------
# Dataset metadata
# -----------------------------------------------------------------------------


def _first_present(
    values: Dict[str, Any],
    keys: List[str],
    default=None,
):
    for key in keys:
        if key in values and values[key] is not None:
            return values[key]
    return default


def _item_to_features_and_meta(
    item: Any,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    Dict[str, Any],
]:
    if isinstance(item, dict):
        anc_key = next(
            (
                key
                for key in [
                    "anc",
                    "anchor",
                    "anchor_feat",
                    "x",
                ]
                if key in item
            ),
            None,
        )
        pos_key = next(
            (
                key
                for key in [
                    "pos",
                    "positive",
                    "positive_feat",
                    "aug",
                    "x_aug",
                ]
                if key in item
            ),
            None,
        )

        if anc_key is None or pos_key is None:
            raise ValueError(
                "Could not infer anchor/positive keys from "
                f"{list(item.keys())}"
            )

        meta = (
            dict(item["meta"])
            if isinstance(
                item.get("meta"),
                dict,
            )
            else {}
        )

        metadata_keys = [
            "audio_path",
            "path",
            "wav_path",
            "phn_path",
            "file_id",
            "crop_start_sample",
            "start_sample",
            "anc_crop_start_sample",
            "pos_crop_start_sample",
            "sample_rate",
            "phn_sample_rate",
            "segment_num_samples",
            "speaker_id",
            "time_preserving",
            "pos_time_preserving",
        ]

        for key in metadata_keys:
            if key in item:
                meta[key] = item[key]

        raw_path = _first_present(
            meta,
            [
                "audio_path",
                "wav_path",
                "path",
            ],
        )
        if raw_path is not None:
            meta["audio_path"] = str(
                raw_path
            )

        return (
            item[anc_key],
            item[pos_key],
            meta,
        )

    if isinstance(
        item,
        (tuple, list),
    ):
        if len(item) < 2:
            raise ValueError(
                "Dataset tuple must contain "
                "anchor and positive features."
            )

        meta: Dict[str, Any] = {}

        if len(item) >= 3:
            if isinstance(
                item[2],
                dict,
            ):
                meta.update(item[2])
            else:
                meta["audio_path"] = str(
                    item[2]
                )

        if len(item) >= 4:
            if not isinstance(
                item[3],
                dict,
            ):
                raise TypeError(
                    "The fourth tuple element must "
                    "be a metadata dict."
                )
            meta.update(item[3])

        return (
            item[0],
            item[1],
            meta,
        )

    raise TypeError(
        "Unsupported dataset item type: "
        f"{type(item)}"
    )


def pair_collate(batch):
    anc_list = []
    pos_list = []
    metas = []

    for item in batch:
        anc, pos, meta = (
            _item_to_features_and_meta(
                item
            )
        )
        anc_list.append(anc)
        pos_list.append(pos)
        metas.append(meta)

    return (
        torch.stack(
            anc_list,
            dim=0,
        ),
        torch.stack(
            pos_list,
            dim=0,
        ),
        metas,
    )


def canonicalize_feats(
    feats: torch.Tensor,
    n_mels: int,
) -> torch.Tensor:
    """Return features as (B,n_mels,T)."""
    if (
        feats.dim() == 4
        and feats.size(1) == 1
    ):
        feats = feats.squeeze(1)

    if feats.dim() != 3:
        raise ValueError(
            "Expected 3D/4D features, got "
            f"{tuple(feats.shape)}"
        )

    if (
        feats.size(-1) == n_mels
        and feats.size(1) != n_mels
    ):
        feats = (
            feats.permute(
                0,
                2,
                1,
            )
            .contiguous()
        )

    if feats.size(1) != n_mels:
        raise ValueError(
            "Could not canonicalize to "
            "(B,n_mels,T). "
            f"Shape={tuple(feats.shape)}, "
            f"n_mels={n_mels}"
        )

    return feats


def infer_phn_path(
    audio_path: str,
    explicit_phn_path: Optional[
        str
    ] = None,
) -> str:
    candidates: List[Path] = []

    if explicit_phn_path:
        candidates.append(
            Path(explicit_phn_path)
        )

    path = Path(audio_path)
    candidates.extend(
        [
            path.with_suffix(".PHN"),
            path.with_suffix(".phn"),
            path.with_suffix(".Phn"),
        ]
    )

    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)

    raise FileNotFoundError(
        "No .PHN file found for "
        f"{audio_path}. Tried: "
        + ", ".join(
            str(candidate)
            for candidate in candidates
        )
    )


def infer_speaker_id(
    audio_path: str,
    explicit: Optional[str] = None,
) -> str:
    if explicit:
        return str(explicit)
    return Path(audio_path).parent.name


def make_unique_file_id(
    audio_path: str,
    speaker_id: str,
    explicit: Optional[str] = None,
) -> str:
    if explicit:
        return str(explicit)

    return (
        f"{speaker_id}/"
        f"{Path(audio_path).stem}"
    )


def resolve_crop_start(
    meta: Dict[str, Any],
    side: str,
) -> int:
    keys = [
        f"{side}_crop_start_sample",
        "crop_start_sample",
        "start_sample",
    ]
    value = _first_present(
        meta,
        keys,
    )

    if value is None:
        raise KeyError(
            f"Missing crop start for side={side}. "
            f"Expected one of {keys}."
        )

    value = int(value)
    if value < 0:
        raise ValueError(
            "crop_start_sample must be "
            f"non-negative, got {value}"
        )

    return value


def side_is_time_preserving(
    meta: Dict[str, Any],
    side: str,
) -> bool:
    if side == "anc":
        return True

    return bool(
        _first_present(
            meta,
            [
                "pos_time_preserving",
                "time_preserving",
            ],
            False,
        )
    )


# -----------------------------------------------------------------------------
# TIMIT .PHN alignment
# -----------------------------------------------------------------------------


def read_timit_phn(
    phn_path: str,
) -> List[
    Tuple[int, int, str]
]:
    records = []

    with open(
        phn_path,
        "r",
        encoding="utf-8",
    ) as handle:
        for line_number, line in enumerate(
            handle,
            start=1,
        ):
            fields = line.strip().split()

            if not fields:
                continue

            if len(fields) != 3:
                raise ValueError(
                    "Malformed .PHN line "
                    f"{line_number} in {phn_path}: "
                    f"{line!r}"
                )

            start = int(fields[0])
            end = int(fields[1])
            phone = fields[2]

            if end <= start:
                raise ValueError(
                    "Non-positive .PHN interval "
                    f"at {phn_path}:{line_number}"
                )

            records.append(
                (
                    start,
                    end,
                    phone,
                )
            )

    if not records:
        raise ValueError(
            "No phone intervals found in "
            f"{phn_path}"
        )

    return records


def map_phone_39(
    phone: str,
    q_policy: str = "ignore",
) -> str:
    if (
        phone == "q"
        and q_policy == "sil"
    ):
        return "sil"

    return TIMIT_61_TO_39.get(
        phone,
        phone,
    )


def align_phn_to_encoder_frames(
    phn_records: List[
        Tuple[int, int, str]
    ],
    crop_start_sample: int,
    num_encoder_frames: int,
    hop_samples: int,
    win_samples: int,
    segment_num_samples: int,
    feature_sample_rate: int,
    phn_sample_rate: int,
    stft_center: bool,
    q_policy: str,
) -> Dict[str, Any]:
    """
    Map each length-preserving encoder frame to the TIMIT .PHN interval
    containing that frame center.

    ``crop_start_sample`` is in the .PHN/original-waveform timebase.
    Frame offsets are first computed in the feature timebase and then converted
    to the .PHN timebase when the sample rates differ.
    """
    if num_encoder_frames <= 0:
        raise ValueError(
            "num_encoder_frames must be positive"
        )
    if (
        hop_samples <= 0
        or win_samples <= 0
        or segment_num_samples <= 0
    ):
        raise ValueError(
            "Invalid feature-frame geometry"
        )
    if (
        feature_sample_rate <= 0
        or phn_sample_rate <= 0
    ):
        raise ValueError(
            "Sample rates must be positive"
        )

    frame_offsets_feature = (
        torch.arange(
            num_encoder_frames,
            dtype=torch.long,
        )
        * int(hop_samples)
    )

    if not stft_center:
        frame_offsets_feature = (
            frame_offsets_feature
            + int(win_samples // 2)
        )

    frame_offsets_feature = (
        frame_offsets_feature.clamp(
            min=0,
            max=(
                segment_num_samples
                - 1
            ),
        )
    )

    rate_ratio = (
        float(phn_sample_rate)
        / float(feature_sample_rate)
    )
    frame_offsets_phn = torch.round(
        frame_offsets_feature.float()
        * rate_ratio
    ).long()

    absolute_samples_phn = (
        frame_offsets_phn
        + int(crop_start_sample)
    )

    starts = torch.tensor(
        [
            interval[0]
            for interval
            in phn_records
        ],
        dtype=torch.long,
    )
    ends = torch.tensor(
        [
            interval[1]
            for interval
            in phn_records
        ],
        dtype=torch.long,
    )
    labels_61_all = [
        interval[2]
        for interval
        in phn_records
    ]

    interval_indices = (
        torch.searchsorted(
            ends,
            absolute_samples_phn,
            right=True,
        )
    )

    labels_61: List[str] = []
    labels_39: List[str] = []

    for query, interval_index in zip(
        absolute_samples_phn.tolist(),
        interval_indices.tolist(),
    ):
        if (
            interval_index
            >= len(phn_records)
            or not (
                int(
                    starts[
                        interval_index
                    ]
                )
                <= query
                < int(
                    ends[
                        interval_index
                    ]
                )
            )
        ):
            labels_61.append(
                "<outside>"
            )
            labels_39.append(
                "<outside>"
            )
            continue

        phone_61 = labels_61_all[
            interval_index
        ]
        labels_61.append(
            phone_61
        )
        labels_39.append(
            map_phone_39(
                phone_61,
                q_policy=q_policy,
            )
        )

    return {
        "phone_labels_61": labels_61,
        "phone_labels_39": labels_39,
        "frame_center_samples_in_crop_feature_timebase": (
            frame_offsets_feature
        ),
        "frame_center_samples_in_crop_phn_timebase": (
            frame_offsets_phn
        ),
        "frame_center_samples_in_utterance": (
            absolute_samples_phn
        ),
    }


# -----------------------------------------------------------------------------
# Cross-attention extraction
# -----------------------------------------------------------------------------


@torch.no_grad()
def extract_prediction_aligned_cross_attention(
    decoder,
    indices: torch.LongTensor,
    z_ctx: torch.Tensor,
) -> Tuple[
    List[torch.Tensor],
    List[torch.Tensor],
]:
    """
    Extract cross-attention rows aligned to generated content tokens.

    For autoregressive teacher forcing:
        decoder_input[:, t] predicts targets[:, t] = indices[:, t+1].

    Therefore cross-attention row t is paired with ``indices[:, t+1]``.
    BOS-input attention correctly aligns to the first generated content token.
    """
    if indices.dim() != 2:
        raise ValueError(
            "indices must have shape (B,L), "
            f"got {tuple(indices.shape)}"
        )
    if indices.size(1) < 2:
        raise ValueError(
            "Generated sequences must contain "
            "at least BOS and one target token."
        )

    decoder_input = (
        indices[:, :-1]
    )
    predicted_tokens = (
        indices[:, 1:]
    )

    result = decoder(
        x=decoder_input,
        xa=z_ctx,
        return_cross_attn=True,
    )

    if (
        not isinstance(
            result,
            tuple,
        )
        or len(result) != 2
    ):
        raise RuntimeError(
            "Unified TextDecoder must return "
            "(logits, cross_attn) when "
            "return_cross_attn=True."
        )

    _, cross_attn = result

    if cross_attn.shape[:2] != (
        predicted_tokens.size(0),
        predicted_tokens.size(1),
    ):
        raise RuntimeError(
            "Cross-attention/target length mismatch: "
            f"attention={tuple(cross_attn.shape)}, "
            f"targets={tuple(predicted_tokens.shape)}"
        )

    pad_idx = decoder.PAD_IDX
    bos_idx = decoder.BOS_IDX
    eos_idx = decoder.EOS_IDX
    mask_idx = getattr(
        decoder,
        "MASK_IDX",
        None,
    )

    valid = (
        predicted_tokens.ne(pad_idx)
        & predicted_tokens.ne(bos_idx)
        & predicted_tokens.ne(eos_idx)
    )
    if mask_idx is not None:
        valid &= predicted_tokens.ne(
            mask_idx
        )

    attention_rows: List[
        torch.Tensor
    ] = []
    token_ids: List[
        torch.Tensor
    ] = []

    for row in range(
        indices.size(0)
    ):
        row_attention = (
            cross_attn[
                row,
                valid[row],
                :,
            ]
            .float()
            .clamp_min(0.0)
        )

        row_attention = (
            row_attention
            / (
                row_attention.sum(
                    dim=-1,
                    keepdim=True,
                )
                + 1e-12
            )
        )

        row_tokens = (
            predicted_tokens[
                row,
                valid[row],
            ]
        )

        if (
            row_attention.size(0)
            != row_tokens.numel()
        ):
            raise RuntimeError(
                "Token/cross-attention row "
                "alignment failed."
            )

        attention_rows.append(
            row_attention.detach().cpu()
        )
        token_ids.append(
            row_tokens.detach().cpu()
        )

    return (
        attention_rows,
        token_ids,
    )


def apply_2d_beta_prior_ragged(
    ragged_attn: List[
        torch.Tensor
    ],
    alpha_scale: float = 2.0,
    beta_scale: float = 2.0,
    strength: float = 0.4,
    eps: float = 1e-12,
) -> List[torch.Tensor]:
    """Apply a post-hoc diagonal positional prior to token-to-frame attention."""
    output = []

    for attention in ragged_attn:
        if attention.numel() == 0:
            output.append(attention)
            continue

        token_count, frame_count = (
            attention.shape
        )
        dtype = attention.dtype
        device = attention.device

        token_position = (
            torch.arange(
                token_count,
                dtype=dtype,
                device=device,
            )
            / max(
                token_count - 1,
                1,
            )
        )
        frame_position = (
            torch.arange(
                frame_count,
                dtype=dtype,
                device=device,
            )
            / max(
                frame_count - 1,
                1,
            )
        )

        mu = token_position[:, None]
        alpha = (
            1.0
            + alpha_scale * mu
        )
        beta = (
            1.0
            + beta_scale
            * (1.0 - mu)
        )

        prior = (
            (
                frame_position[None]
                + eps
            ).pow(
                alpha - 1.0
            )
            * (
                1.0
                - frame_position[None]
                + eps
            ).pow(
                beta - 1.0
            )
        )
        prior = (
            prior
            / (
                prior.sum(
                    dim=1,
                    keepdim=True,
                )
                + eps
            )
        )

        adjusted = (
            attention
            * prior.pow(strength)
        )
        adjusted = (
            adjusted
            / (
                adjusted.sum(
                    dim=1,
                    keepdim=True,
                )
                + eps
            )
        )

        output.append(
            adjusted
        )

    return output


def frames_to_tokens_scores(
    token_to_frame: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    frame_to_token = (
        token_to_frame
        .transpose(0, 1)
        .contiguous()
    )

    return (
        frame_to_token
        / (
            frame_to_token.sum(
                dim=1,
                keepdim=True,
            )
            + eps
        )
    )


def viterbi_monotonic_path(
    frame_to_token: torch.Tensor,
    eps: float = 1e-12,
) -> torch.LongTensor:
    """
    Monotonic stay/advance Viterbi path constrained to start at token 0 and
    finish at the final token.
    """
    frame_count, token_count = (
        frame_to_token.shape
    )

    if (
        frame_count == 0
        or token_count == 0
    ):
        return torch.empty(
            0,
            dtype=torch.long,
        )

    if frame_count < token_count:
        raise ValueError(
            "Monotonic stay/advance path is "
            "impossible when frame_count < token_count: "
            f"{frame_count} < {token_count}"
        )

    log_prob = (
        frame_to_token
        .clamp_min(eps)
        .log()
    )

    dp = torch.full(
        (
            frame_count,
            token_count,
        ),
        float("-inf"),
        device=(
            frame_to_token.device
        ),
        dtype=log_prob.dtype,
    )
    backpointer = torch.full(
        (
            frame_count,
            token_count,
        ),
        -1,
        device=(
            frame_to_token.device
        ),
        dtype=torch.long,
    )

    dp[0, 0] = log_prob[0, 0]
    backpointer[0, 0] = 0

    for frame in range(
        1,
        frame_count,
    ):
        dp[frame, 0] = (
            dp[frame - 1, 0]
            + log_prob[frame, 0]
        )
        backpointer[frame, 0] = 0

        max_token = min(
            frame,
            token_count - 1,
        )

        for token in range(
            1,
            max_token + 1,
        ):
            stay = dp[
                frame - 1,
                token,
            ]
            advance = dp[
                frame - 1,
                token - 1,
            ]

            if advance > stay:
                dp[
                    frame,
                    token,
                ] = (
                    advance
                    + log_prob[
                        frame,
                        token,
                    ]
                )
                backpointer[
                    frame,
                    token,
                ] = token - 1
            else:
                dp[
                    frame,
                    token,
                ] = (
                    stay
                    + log_prob[
                        frame,
                        token,
                    ]
                )
                backpointer[
                    frame,
                    token,
                ] = token

    if not torch.isfinite(
        dp[
            frame_count - 1,
            token_count - 1,
        ]
    ):
        raise RuntimeError(
            "No valid monotonic Viterbi path "
            "reached the final token."
        )

    path = torch.empty(
        frame_count,
        device=(
            frame_to_token.device
        ),
        dtype=torch.long,
    )

    token_position = (
        token_count - 1
    )
    path[
        frame_count - 1
    ] = token_position

    for frame in range(
        frame_count - 1,
        0,
        -1,
    ):
        token_position = int(
            backpointer[
                frame,
                token_position,
            ].item()
        )
        path[
            frame - 1
        ] = token_position

    return path.cpu()


def path_indices_to_token_ids(
    path: torch.LongTensor,
    token_ids: torch.LongTensor,
) -> torch.LongTensor:
    if (
        path.numel() == 0
        or token_ids.numel() == 0
    ):
        return torch.empty(
            0,
            dtype=torch.long,
        )

    return token_ids.detach().cpu()[
        path.clamp(
            0,
            token_ids.numel() - 1,
        )
    ]


def monotonicity_metrics_single(
    token_to_frame: torch.Tensor,
    eps: float = 1e-12,
) -> Dict[str, float]:
    metric_names = [
        "tok_len",
        "frame_len",
        "expected_frame_violation_rate",
        "expected_frame_violation_mag",
        "peak_frame_violation_rate",
        "peak_frame_violation_mag",
        "mean_abs_diagonal_error",
        "mean_peak_diagonal_error",
        "expected_frame_corr",
        "frame_to_token_expected_violation_rate",
        "frame_to_token_expected_violation_mag",
        "frame_to_token_peak_violation_rate",
        "frame_to_token_peak_violation_mag",
        "viterbi_log_score_mean",
    ]

    if token_to_frame.numel() == 0:
        return {
            name: 0.0
            for name in metric_names
        }

    attention = (
        token_to_frame
        .float()
        .clamp_min(0.0)
    )
    attention = (
        attention
        / (
            attention.sum(
                dim=-1,
                keepdim=True,
            )
            + eps
        )
    )

    token_count, frame_count = (
        attention.shape
    )

    token_position = torch.arange(
        token_count,
        dtype=attention.dtype,
        device=attention.device,
    )
    frame_position = torch.arange(
        frame_count,
        dtype=attention.dtype,
        device=attention.device,
    )

    token_norm = (
        token_position
        / max(
            token_count - 1,
            1,
        )
    )

    expected_frame = (
        attention
        * frame_position[None]
    ).sum(dim=-1)
    expected_frame_norm = (
        expected_frame
        / max(
            frame_count - 1,
            1,
        )
    )

    peak_frame = attention.argmax(
        dim=-1
    ).to(attention.dtype)
    peak_frame_norm = (
        peak_frame
        / max(
            frame_count - 1,
            1,
        )
    )

    if token_count > 1:
        expected_diff = (
            expected_frame[1:]
            - expected_frame[:-1]
        )
        peak_diff = (
            peak_frame[1:]
            - peak_frame[:-1]
        )

        expected_bad = (
            expected_diff < 0
        )
        peak_bad = (
            peak_diff < 0
        )

        expected_rate = (
            expected_bad.float()
            .mean()
            .item()
        )
        peak_rate = (
            peak_bad.float()
            .mean()
            .item()
        )

        expected_mag = (
            (
                -expected_diff[
                    expected_bad
                ]
            )
            .mean()
            .item()
            if expected_bad.any()
            else 0.0
        )
        peak_mag = (
            (
                -peak_diff[
                    peak_bad
                ]
            )
            .mean()
            .item()
            if peak_bad.any()
            else 0.0
        )

        centered_token = (
            token_norm
            - token_norm.mean()
        )
        centered_frame = (
            expected_frame_norm
            - expected_frame_norm.mean()
        )

        corr = (
            (
                centered_token
                * centered_frame
            ).sum()
            / torch.sqrt(
                (
                    centered_token.square().sum()
                    + eps
                )
                * (
                    centered_frame.square().sum()
                    + eps
                )
            )
        ).item()
    else:
        expected_rate = 0.0
        peak_rate = 0.0
        expected_mag = 0.0
        peak_mag = 0.0
        corr = 0.0

    frame_to_token = (
        frames_to_tokens_scores(
            attention,
            eps=eps,
        )
    )

    token_positions = (
        torch.arange(
            token_count,
            dtype=attention.dtype,
            device=attention.device,
        )
    )

    expected_token = (
        frame_to_token
        * token_positions[None]
    ).sum(dim=-1)

    peak_token = (
        frame_to_token.argmax(
            dim=-1
        ).to(attention.dtype)
    )

    if frame_count > 1:
        expected_token_diff = (
            expected_token[1:]
            - expected_token[:-1]
        )
        peak_token_diff = (
            peak_token[1:]
            - peak_token[:-1]
        )

        frame_expected_bad = (
            expected_token_diff < 0
        )
        frame_peak_bad = (
            peak_token_diff < 0
        )

        frame_expected_rate = (
            frame_expected_bad.float()
            .mean()
            .item()
        )
        frame_peak_rate = (
            frame_peak_bad.float()
            .mean()
            .item()
        )

        frame_expected_mag = (
            (
                -expected_token_diff[
                    frame_expected_bad
                ]
            )
            .mean()
            .item()
            if frame_expected_bad.any()
            else 0.0
        )
        frame_peak_mag = (
            (
                -peak_token_diff[
                    frame_peak_bad
                ]
            )
            .mean()
            .item()
            if frame_peak_bad.any()
            else 0.0
        )
    else:
        frame_expected_rate = 0.0
        frame_peak_rate = 0.0
        frame_expected_mag = 0.0
        frame_peak_mag = 0.0

    path = viterbi_monotonic_path(
        frame_to_token,
        eps=eps,
    )

    if path.numel() > 0:
        frame_indices = torch.arange(
            frame_to_token.size(0),
        )
        viterbi_score = (
            frame_to_token[
                frame_indices,
                path,
            ]
            .clamp_min(eps)
            .log()
            .mean()
            .item()
        )
    else:
        viterbi_score = 0.0

    return {
        "tok_len": float(
            token_count
        ),
        "frame_len": float(
            frame_count
        ),
        "expected_frame_violation_rate": (
            expected_rate
        ),
        "expected_frame_violation_mag": (
            expected_mag
        ),
        "peak_frame_violation_rate": (
            peak_rate
        ),
        "peak_frame_violation_mag": (
            peak_mag
        ),
        "mean_abs_diagonal_error": (
            (
                expected_frame_norm
                - token_norm
            )
            .abs()
            .mean()
            .item()
        ),
        "mean_peak_diagonal_error": (
            (
                peak_frame_norm
                - token_norm
            )
            .abs()
            .mean()
            .item()
        ),
        "expected_frame_corr": corr,
        "frame_to_token_expected_violation_rate": (
            frame_expected_rate
        ),
        "frame_to_token_expected_violation_mag": (
            frame_expected_mag
        ),
        "frame_to_token_peak_violation_rate": (
            frame_peak_rate
        ),
        "frame_to_token_peak_violation_mag": (
            frame_peak_mag
        ),
        "viterbi_log_score_mean": (
            viterbi_score
        ),
    }


def build_item_payload(
    indices: torch.LongTensor,
    z_ctx: torch.Tensor,
    decoder,
    beta_alpha_scale: float,
    beta_beta_scale: float,
    beta_strength: float,
) -> List[
    Dict[str, Any]
]:
    raw_attention, token_ids = (
        extract_prediction_aligned_cross_attention(
            decoder=decoder,
            indices=indices,
            z_ctx=z_ctx,
        )
    )

    beta_attention = (
        apply_2d_beta_prior_ragged(
            raw_attention,
            alpha_scale=(
                beta_alpha_scale
            ),
            beta_scale=(
                beta_beta_scale
            ),
            strength=beta_strength,
        )
    )

    items = []

    for (
        raw_attn,
        beta_attn,
        tokens,
    ) in zip(
        raw_attention,
        beta_attention,
        token_ids,
    ):
        raw_frame_to_token = (
            frames_to_tokens_scores(
                raw_attn
            )
        )
        beta_frame_to_token = (
            frames_to_tokens_scores(
                beta_attn
            )
        )

        raw_path = (
            viterbi_monotonic_path(
                raw_frame_to_token
            )
        )
        beta_path = (
            viterbi_monotonic_path(
                beta_frame_to_token
            )
        )

        items.append(
            {
                "tokens": tokens,
                "indices": None,
                "attention_alignment": (
                    "cross-attention row t is "
                    "aligned to generated target "
                    "indices[t+1]"
                ),
                "raw_attn": raw_attn,
                "beta_attn": beta_attn,
                "raw_frame2tok": (
                    raw_frame_to_token
                ),
                "beta_frame2tok": (
                    beta_frame_to_token
                ),
                "raw_viterbi_path": (
                    raw_path
                ),
                "beta_viterbi_path": (
                    beta_path
                ),
                "raw_frame_token_ids": (
                    path_indices_to_token_ids(
                        raw_path,
                        tokens,
                    )
                ),
                "beta_frame_token_ids": (
                    path_indices_to_token_ids(
                        beta_path,
                        tokens,
                    )
                ),
                "raw_metrics": (
                    monotonicity_metrics_single(
                        raw_attn
                    )
                ),
                "beta_metrics": (
                    monotonicity_metrics_single(
                        beta_attn
                    )
                ),
            }
        )

    # Preserve each full generated sequence for reproducibility.
    for row, item in enumerate(items):
        item["indices"] = (
            indices[row]
            .detach()
            .cpu()
        )

    return items


# -----------------------------------------------------------------------------
# Artifact generation
# -----------------------------------------------------------------------------


@torch.no_grad()
def generate_artifacts(
    model,
    loader,
    output_dir: str,
    device: torch.device,
    cfg: Dict[str, Any],
    max_items: Optional[int],
    max_len: int,
    beam_size: int,
    length_cap_ratio: float,
    rep_gamma: float,
    length_norm_alpha: float,
    beta_alpha_scale: float,
    beta_beta_scale: float,
    beta_strength: float,
    expected_encoder_frames: Optional[
        int
    ],
    stft_center: bool,
    q_policy: str,
):
    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    model.eval().to(device)
    model.encoder_teacher.eval()
    model.decoder_teacher.eval()
    model.decoder_teacher.set_self_attn_drop_p(
        0.0
    )
    decoder = model.decoder_teacher

    feature_sample_rate = int(
        cfg["fs"]
    )
    hop_samples = int(
        round(
            float(
                cfg["win_hop"]
            )
            * feature_sample_rate
        )
    )
    win_samples = int(
        round(
            float(
                cfg["win_size"]
            )
            * feature_sample_rate
        )
    )
    default_segment_samples = int(
        round(
            float(
                cfg["seglen"]
            )
            * feature_sample_rate
        )
    )

    global_index = 0

    progress = tqdm(
        loader,
        desc=(
            "Saving cross-attention + "
            "phone artifacts"
        ),
        unit="batch",
    )

    for (
        anc_features,
        pos_features,
        metas,
    ) in progress:
        if (
            max_items is not None
            and global_index
            >= max_items
        ):
            break

        anc_bft = canonicalize_feats(
            anc_features,
            int(cfg["n_mels"]),
        ).to(
            device,
            non_blocking=True,
        )
        pos_bft = canonicalize_feats(
            pos_features,
            int(cfg["n_mels"]),
        ).to(
            device,
            non_blocking=True,
        )

        anc_encoder_input = (
            anc_bft.permute(
                0,
                2,
                1,
            )
            .contiguous()
        )
        pos_encoder_input = (
            pos_bft.permute(
                0,
                2,
                1,
            )
            .contiguous()
        )

        z_anc = model.encoder_teacher(
            anc_encoder_input
        )
        z_pos = model.encoder_teacher(
            pos_encoder_input
        )

        # Exact STFT-frame/.PHN alignment below assumes the encoder preserves
        # the feature time axis one-for-one.
        if (
            z_anc.size(1)
            != anc_bft.size(-1)
            or z_pos.size(1)
            != pos_bft.size(-1)
        ):
            raise ValueError(
                "Exact .PHN frame alignment assumes "
                "a length-preserving encoder. "
                f"anc feature/encoder frames="
                f"{anc_bft.size(-1)}/"
                f"{z_anc.size(1)}, "
                f"pos={pos_bft.size(-1)}/"
                f"{z_pos.size(1)}."
            )

        if (
            expected_encoder_frames
            is not None
            and (
                z_anc.size(1)
                != expected_encoder_frames
                or z_pos.size(1)
                != expected_encoder_frames
            )
        ):
            raise ValueError(
                "Unexpected encoder length: "
                f"expected "
                f"{expected_encoder_frames}, "
                f"got anc={z_anc.size(1)}, "
                f"pos={z_pos.size(1)}"
            )

        anc_indices = (
            model.generate_teacher_beam_tokens(
                z_ctx=z_anc,
                max_len=max_len,
                beam_size=beam_size,
                length_cap_ratio=(
                    length_cap_ratio
                ),
                rep_gamma=rep_gamma,
                length_norm_alpha=(
                    length_norm_alpha
                ),
            )
        )

        pos_indices = (
            model.generate_teacher_beam_tokens(
                z_ctx=z_pos,
                max_len=max_len,
                beam_size=beam_size,
                length_cap_ratio=(
                    length_cap_ratio
                ),
                rep_gamma=rep_gamma,
                length_norm_alpha=(
                    length_norm_alpha
                ),
            )
        )

        anc_items = build_item_payload(
            indices=anc_indices,
            z_ctx=z_anc,
            decoder=decoder,
            beta_alpha_scale=(
                beta_alpha_scale
            ),
            beta_beta_scale=(
                beta_beta_scale
            ),
            beta_strength=(
                beta_strength
            ),
        )
        pos_items = build_item_payload(
            indices=pos_indices,
            z_ctx=z_pos,
            decoder=decoder,
            beta_alpha_scale=(
                beta_alpha_scale
            ),
            beta_beta_scale=(
                beta_beta_scale
            ),
            beta_strength=(
                beta_strength
            ),
        )

        batch_size = z_anc.size(0)

        for row in range(batch_size):
            if (
                max_items is not None
                and global_index
                >= max_items
            ):
                break

            meta = dict(
                metas[row]
            )

            audio_path = _first_present(
                meta,
                [
                    "audio_path",
                    "wav_path",
                    "path",
                ],
            )
            if audio_path is None:
                raise KeyError(
                    "Dataset metadata has no "
                    "audio_path. Full TIMIT paths "
                    "must be preserved."
                )

            audio_path = str(
                audio_path
            )
            phn_path = infer_phn_path(
                audio_path,
                meta.get("phn_path"),
            )
            speaker_id = infer_speaker_id(
                audio_path,
                meta.get("speaker_id"),
            )
            file_id = make_unique_file_id(
                audio_path,
                speaker_id,
                meta.get("file_id"),
            )
            phn_records = read_timit_phn(
                phn_path
            )

            phn_sample_rate = int(
                meta.get(
                    "phn_sample_rate",
                    feature_sample_rate,
                )
            )

            item = {
                "file_id": file_id,
                "item_index": (
                    global_index + 1
                ),
                "anc": anc_items[row],
                "pos": pos_items[row],
                "meta": {
                    "audio_path": audio_path,
                    "phn_path": phn_path,
                    "speaker_id": (
                        speaker_id
                    ),
                    "feature_sample_rate": (
                        feature_sample_rate
                    ),
                    "phn_sample_rate": (
                        phn_sample_rate
                    ),
                    "hop_samples": (
                        hop_samples
                    ),
                    "win_samples": (
                        win_samples
                    ),
                    "stft_center": (
                        stft_center
                    ),
                    "encoder_frames": int(
                        z_anc.size(1)
                    ),
                    "max_len": max_len,
                    "beam_size": (
                        beam_size
                    ),
                    "length_cap_ratio": (
                        length_cap_ratio
                    ),
                    "rep_gamma": rep_gamma,
                    "length_norm_alpha": (
                        length_norm_alpha
                    ),
                    "beta_alpha_scale": (
                        beta_alpha_scale
                    ),
                    "beta_beta_scale": (
                        beta_beta_scale
                    ),
                    "beta_strength": (
                        beta_strength
                    ),
                    "q_policy": q_policy,
                    "decoder_attention_alignment": (
                        "decoder row t -> "
                        "generated token t+1"
                    ),
                },
            }

            for side, z_side in [
                ("anc", z_anc),
                ("pos", z_pos),
            ]:
                valid_alignment = (
                    side_is_time_preserving(
                        meta,
                        side,
                    )
                )

                item[side][
                    "phone_alignment_valid"
                ] = bool(
                    valid_alignment
                )

                if not valid_alignment:
                    item[side][
                        "phone_alignment_reason"
                    ] = (
                        "Positive transform was not "
                        "declared time-preserving."
                    )
                    continue

                crop_start = (
                    resolve_crop_start(
                        meta,
                        side,
                    )
                )

                segment_num_samples = int(
                    meta.get(
                        "segment_num_samples",
                        default_segment_samples,
                    )
                )

                alignment = (
                    align_phn_to_encoder_frames(
                        phn_records=(
                            phn_records
                        ),
                        crop_start_sample=(
                            crop_start
                        ),
                        num_encoder_frames=int(
                            z_side.size(1)
                        ),
                        hop_samples=(
                            hop_samples
                        ),
                        win_samples=(
                            win_samples
                        ),
                        segment_num_samples=(
                            segment_num_samples
                        ),
                        feature_sample_rate=(
                            feature_sample_rate
                        ),
                        phn_sample_rate=(
                            phn_sample_rate
                        ),
                        stft_center=(
                            stft_center
                        ),
                        q_policy=q_policy,
                    )
                )

                item[side].update(
                    alignment
                )
                item[side][
                    "crop_start_sample"
                ] = crop_start
                item[side][
                    "segment_num_samples"
                ] = segment_num_samples

            output_path = os.path.join(
                output_dir,
                (
                    f"item_"
                    f"{global_index + 1:06d}"
                    f".pt"
                ),
            )
            torch.save(
                item,
                output_path,
            )

            global_index += 1
            progress.set_postfix(
                saved=global_index
            )

    print(
        f"Saved {global_index} artifacts "
        f"to {output_dir}"
    )


def build_model(
    checkpoint_path: str,
    device_id: int,
):
    model = (
        AudioTokenizerAR_FINAL
        .load_from_pretrained(
            checkpoint_path,
            device_id,
        )
    )
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root_dir",
        default=os.path.dirname(
            os.getcwd()
        ),
    )
    parser.add_argument(
        "--config",
        default="mainFull.yaml",
    )
    parser.add_argument(
        "--ckpt",
        required=True,
    )
    parser.add_argument(
        "--split",
        default="valid",
        choices=[
            "train",
            "valid",
            "test",
        ],
    )
    parser.add_argument(
        "--output_dir",
        default=(
            "cross_attention_phone_output_TIMIT"
        ),
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
        "--max_items",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--max_len",
        type=int,
        default=60,
    )
    parser.add_argument(
        "--beam_size",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--length_cap_ratio",
        type=float,
        default=0.15,
    )
    parser.add_argument(
        "--rep_gamma",
        type=float,
        default=1.3,
    )
    parser.add_argument(
        "--length_norm_alpha",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--beta_alpha_scale",
        type=float,
        default=2.0,
    )
    parser.add_argument(
        "--beta_beta_scale",
        type=float,
        default=2.0,
    )
    parser.add_argument(
        "--beta_strength",
        type=float,
        default=0.4,
    )

    parser.add_argument(
        "--expected_encoder_frames",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--stft_center",
        action=(
            argparse.BooleanOptionalAction
        ),
        default=True,
    )
    parser.add_argument(
        "--q_policy",
        choices=[
            "ignore",
            "sil",
        ],
        default="ignore",
    )
    parser.add_argument(
        "--add_noise",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--use_time_compression",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Slightly compress only the positive view and right-pad it back to 3 seconds.",
    )

    args = parser.parse_args()

    if args.rep_gamma < 1.0:
        raise ValueError(
            "--rep_gamma must be >= 1.0"
        )

    config_path = os.path.join(
        args.root_dir,
        "config",
        args.config,
    )

    with open(
        config_path,
        "r",
        encoding="utf-8",
    ) as handle:
        cfg = yaml.safe_load(
            handle
        )

    seed_everything(
        cfg.get(
            "seed",
            13,
        ),
        workers=True,
    )

    split_cfg = cfg[
        args.split
    ]
    dataset_csv = split_cfg[
        "dataset_csv"
    ]

    batch_size = (
        args.batch_size
        or int(
            cfg["batchsize"]
        )
    )
    num_workers = (
        args.num_workers
        if args.num_workers is not None
        else int(
            cfg["load_workers"]
        )
    )

    print(
        f"Config: {config_path}"
    )
    print(
        f"Dataset CSV: {dataset_csv}"
    )
    print(
        f"Checkpoint: {args.ckpt}"
    )
    print(
        f"Output: {args.output_dir}"
    )
    print(
        "Exact phone alignment requires "
        "full audio_path and crop-start metadata."
    )

    dataset = AudioPairAugDataset(
        audio_csv=dataset_csv,
        fs=cfg["fs"],
        context_sec=cfg["seglen"],
        add_noise=args.add_noise,
        use_time_compression=args.use_time_compression,
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
            if args.split == "train"
            else "val"
        ),
        return_metadata=True,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=(
            torch.cuda.is_available()
        ),
        collate_fn=pair_collate,
    )

    device = torch.device(
        (
            f"cuda:{args.device}"
            if torch.cuda.is_available()
            else "cpu"
        )
    )

    model = build_model(
        args.ckpt,
        args.device,
    )

    generate_artifacts(
        model=model,
        loader=loader,
        output_dir=(
            args.output_dir
        ),
        device=device,
        cfg=cfg,
        max_items=args.max_items,
        max_len=args.max_len,
        beam_size=args.beam_size,
        length_cap_ratio=(
            args.length_cap_ratio
        ),
        rep_gamma=args.rep_gamma,
        length_norm_alpha=(
            args.length_norm_alpha
        ),
        beta_alpha_scale=(
            args.beta_alpha_scale
        ),
        beta_beta_scale=(
            args.beta_beta_scale
        ),
        beta_strength=(
            args.beta_strength
        ),
        expected_encoder_frames=(
            args.expected_encoder_frames
        ),
        stft_center=(
            args.stft_center
        ),
        q_policy=args.q_policy,
    )


if __name__ == "__main__":
    main()
