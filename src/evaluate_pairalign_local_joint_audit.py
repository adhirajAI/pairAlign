#!/usr/bin/env python3
"""
evaluate_pairalign_local_joint_audit.py

Held-out LOCAL audit for the revised PairAlign Stage-III appendix.

This script is deliberately row-local and hard-neighborhood-local. It does
NOT estimate any dataset-global slope, residual bound, or edit-distance margin.

It audits three things on the same held-out anchor/positive population:

(1) Centered local likelihood--edit calibration

      delta_ij,V = S_ii,V - S_ij,V
      d_ij       = ED(Y_i, Y_j) / max(|Y_i|, |Y_j|)

    Within H_k^V(i), fit THROUGH THE ORIGIN

      delta_ij,V = a_i,V^(k) d_ij + r_ij,V^(k).

    The fit is repeated for k in {5,8,15} by default.

(2) Leave-one-candidate-out (LOO) local certificates

    For each held-out hard candidate j, fit a on the other k-1 candidates and
    set beta to the maximum absolute residual on that fitting neighborhood.
    Then test whether the held-out residual is covered by that LOCAL beta.

    Candidate-specific local lower bound:

      q_obs = [ (delta_ij,V - beta_loo) / a_loo ]_+

    Row-NCE local lower bound:

      Delta_i,V^NCE
          = tau * log(1 / (exp(ell_i^V) - 1))
          = -tau * log sum_{j != i}
                    exp((S_ij,V - S_ii,V) / tau)

      q_nce = [ (Delta_i,V^NCE - beta_loo) / a_loo ]_+

    where ell_i^V is the realized row-wise negative-log-softmax loss computed
    from the current score matrix.

    IMPORTANT:
    Delta_i,V^NCE is NOT evaluated through expm1(ell), because when ell is
    extremely small that representation suffers catastrophic cancellation.
    Instead we evaluate the mathematically equivalent off-diagonal
    log-sum-exp directly using the shifted stabilization

        exp(z - max(z)).

(3) Joint positive-contraction + local-separation audit

    The EMA teacher generates ONE anchor-derived target Y_i.

    To mirror the supplied Stage-III training code, a single stochastic
    corrupted teacher-forcing prefix is generated per realization and is reused
    for BOTH anchor and positive conditioning. The decoder forwards themselves
    remain stochastic because structured self-attention dropout and ordinary
    training-mode dropout are sampled during each forward.

      e_A  = ED(pred_A, Y_i)
      e_P  = ED(pred_P, Y_i)
      d_AP = ED(pred_A, pred_P)

    Conservative triangle-inequality threshold:

      C_cons = e_A + e_P + max(e_A, e_P)

    Tighter realized-positive-contraction threshold:

      C_real = d_AP + max(e_A, e_P)

    For each local negative j:

      Gamma_geom_cons = ED(Y_i,Y_j) - C_cons
      Gamma_obs_cons  = L_ij*q_obs - C_cons
      Gamma_nce_cons  = L_ij*q_nce - C_cons

    and

      Gamma_geom_real = ED(Y_i,Y_j) - C_real
      Gamma_obs_real  = L_ij*q_obs - C_real
      Gamma_nce_real  = L_ij*q_nce - C_real

    Positive calibrated Gamma values, together with LOO residual coverage and
    a_loo > 0, provide LOCAL sufficient certificates for

      ED(pred_A,pred_P)
          < min{
              ED(pred_A,Y_j),
              ED(pred_P,Y_j)
            }.

NUMERICAL STABILIZATION
-----------------------
All probability normalizations relevant to this audit explicitly use shifted
exponentials

      exp(z - max(z))

rather than raw exp(z).

This is used for:
  * top-p softmax probabilities;
  * row-wise NCE log-sum-exp;
  * off-diagonal NCE mass used to compute Delta_i,V^NCE.

The NCE calculation is additionally promoted to float64.

IMPORTANT IMPLEMENTATION MATCHING
---------------------------------
The defaults mirror the supplied current PairAlign Stage III:
  * EMA target source: anchor only
  * top-p target generation:
        max_len=60, top_p=0.88, temperature=0.90,
        length_cap_ratio=0.15, rep_gamma=1.15,
        explore_steps=4,
        top_p_start=0.97, top_p_end=0.88,
        temperature_start=1.30, temperature_end=1.00,
        rep_gamma_start=1.03, rep_gamma_end=1.30
  * NCE temperature tau=0.5
  * clean NCE audit scoring: prefix corruption OFF,
        structured-SA drop OFF, decoder eval mode
  * positive-fitting audit:
        prefix mask=0.1,
        structured-SA drop=0.1
  * the SAME corrupted prefix is used for A/P within each MC realization
  * 16 independent stochastic positive-fitting realizations
  * validation DataLoader seed=1234
  * evaluation seed=13
  * batch size=16 -> 15 in-batch mismatches on full batches

The script assumes:

    from train import AudioTokenizerAR_FINAL
    from utils import AudioPairAugDataset

Outputs in --out_dir:
    local_row_fits.csv
    local_loo_joint.csv
    contraction_rows.csv
    combined_joint.csv
    summary.csv
    run_config.json

No training is performed.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from lightning.pytorch import seed_everything
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from train import AudioTokenizerAR_FINAL
from utils import AudioPairAugDataset


# ============================================================
# Reproducibility / data
# ============================================================


def path_to_file_id(path: str) -> str:
    return os.path.splitext(os.path.basename(str(path)))[0]


def seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def pair_collate(batch):
    """Flexible anchor/positive collation used by the PairAlign audits."""
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
                f"Could not infer anchor/positive keys from {first.keys()}"
            )

        anc_feats = torch.stack([b[anc_key] for b in batch], dim=0)
        pos_feats = torch.stack([b[pos_key] for b in batch], dim=0)

        if id_key is None:
            file_ids = ["unknown"] * len(batch)
        else:
            raw = [b[id_key] for b in batch]
            file_ids = [
                path_to_file_id(x)
                if "/" in str(x) or "." in str(x)
                else str(x)
                for x in raw
            ]

        return anc_feats, pos_feats, file_ids

    if isinstance(first, (tuple, list)):
        if len(first) < 2:
            raise ValueError(
                "Dataset item must contain anchor and positive"
            )

        anc_feats = torch.stack([b[0] for b in batch], dim=0)
        pos_feats = torch.stack([b[1] for b in batch], dim=0)

        if len(first) >= 3:
            raw = [b[2] for b in batch]
            file_ids = [
                path_to_file_id(x)
                if "/" in str(x) or "." in str(x)
                else str(x)
                for x in raw
            ]
        else:
            file_ids = ["unknown"] * len(batch)

        return anc_feats, pos_feats, file_ids

    raise ValueError(
        f"Unsupported dataset item type: {type(first)}"
    )


def make_loader(
    cfg: Dict[str, Any],
    split: str,
    batch_size: int,
    num_workers: int,
    dataloader_seed: int,
    add_noise: bool,
    use_time_compression: bool = False,
):
    split_cfg = cfg[split]

    dataset = AudioPairAugDataset(
        audio_csv=split_cfg["dataset_csv"],
        fs=cfg["fs"],
        context_sec=cfg["seglen"],
        add_noise=add_noise,
        use_time_compression=use_time_compression,
        return_path=True,
        feats_type="melspectrogram",
        rf_size=int(round(cfg["win_size"] * cfg["fs"])),
        stride=int(round(cfg["win_hop"] * cfg["fs"])),
        n_mels=cfg["n_mels"],
        split="train" if split == "train" else "val",
    )

    g = torch.Generator()
    g.manual_seed(dataloader_seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=pair_collate,
        worker_init_fn=seed_worker,
        generator=g,
        persistent_workers=False,
    )


def prepare_model_input(
    model,
    feats: torch.Tensor,
) -> torch.Tensor:
    """
    Use project-specific inference preparation when available.

    Fallback matches supplied Stage-III training:
        (B,1,n_mels,T) -> (B,T,n_mels).
    """
    if hasattr(model, "_prepare_inference_input"):
        return model._prepare_inference_input(feats)

    x = feats

    if x.ndim == 4 and x.size(1) == 1:
        x = x.squeeze(1)

    if x.ndim == 3:
        return x.permute(0, 2, 1).contiguous()

    raise ValueError(
        "Could not prepare model input. Expected "
        "model._prepare_inference_input or a 3D/4D mel tensor, "
        f"got shape {tuple(feats.shape)}"
    )


@contextmanager
def torch_rng_scope(
    seed: int,
    device: torch.device,
):
    """
    Isolate stochastic model evaluation from DataLoader RNG state.
    """
    devices: List[int] = []

    if device.type == "cuda":
        devices = [
            device.index
            if device.index is not None
            else 0
        ]

    with torch.random.fork_rng(
        devices=devices,
        enabled=True,
    ):
        torch.manual_seed(int(seed))

        if device.type == "cuda":
            torch.cuda.manual_seed_all(int(seed))

        yield


# ============================================================
# Stable shifted softmax / log-sum-exp
# ============================================================


def stable_shifted_softmax(
    logits: torch.Tensor,
    dim: int = -1,
) -> torch.Tensor:
    """
    Numerically stable softmax:

        p_i = exp(z_i - m)
              ----------------
              sum_j exp(z_j - m),

        m = max_j z_j.

    Subtracting the maximum guarantees the largest exponent is exp(0)=1,
    preventing exponential overflow. Very negative shifted logits may
    underflow to zero, which is numerically harmless and corresponds to
    negligible probability mass.

    -inf logits are allowed.
    NaN or +inf logits are rejected explicitly.
    """
    if torch.isnan(logits).any():
        raise FloatingPointError(
            "NaN encountered in logits before stable softmax"
        )

    if torch.isposinf(logits).any():
        raise FloatingPointError(
            "+inf encountered in logits before stable softmax"
        )

    m = torch.amax(
        logits,
        dim=dim,
        keepdim=True,
    )

    if not torch.isfinite(m).all():
        raise FloatingPointError(
            "At least one softmax row contains no finite logit"
        )

    shifted = logits - m

    # Explicit requested stabilization:
    # exp(z - max(z))
    exp_shifted = torch.exp(shifted)

    denom = exp_shifted.sum(
        dim=dim,
        keepdim=True,
    )

    if (denom <= 0).any() or not torch.isfinite(denom).all():
        raise FloatingPointError(
            "Invalid denominator in shifted softmax"
        )

    return exp_shifted / denom


def stable_shifted_logsumexp(
    z: torch.Tensor,
    dim: int = -1,
) -> torch.Tensor:
    """
    Numerically stable

        log sum_j exp(z_j)

    evaluated as

        m + log sum_j exp(z_j - m),

        m = max_j z_j.

    This explicitly uses exp(z - max(z)).

    Input is promoted to float64 because this function is used for the NCE
    audit. -inf entries are allowed as masks, provided every reduction row
    contains at least one finite value.
    """
    z64 = z.to(dtype=torch.float64)

    if torch.isnan(z64).any():
        raise FloatingPointError(
            "NaN encountered before shifted log-sum-exp"
        )

    if torch.isposinf(z64).any():
        raise FloatingPointError(
            "+inf encountered before shifted log-sum-exp"
        )

    m = torch.amax(
        z64,
        dim=dim,
        keepdim=True,
    )

    if not torch.isfinite(m).all():
        raise FloatingPointError(
            "At least one log-sum-exp row contains no finite value"
        )

    shifted = z64 - m

    # Explicit requested stabilization:
    # exp(z - max(z))
    exp_shifted = torch.exp(shifted)

    sum_exp = exp_shifted.sum(
        dim=dim,
        keepdim=True,
    )

    if (sum_exp <= 0).any() or not torch.isfinite(sum_exp).all():
        raise FloatingPointError(
            "Invalid shifted exponential sum in log-sum-exp"
        )

    out = m + torch.log(sum_exp)

    return out.squeeze(dim)


# ============================================================
# Exact Stage-III top-p target generation
# ============================================================


def sample_top_p(
    logits: torch.Tensor,
    top_p: float,
) -> torch.Tensor:
    """
    Top-p sampling using explicit shifted-softmax stabilization.

    probabilities are computed as

        exp(z - max(z))
        ----------------
        sum exp(z - max(z)).
    """
    if not (0.0 < top_p <= 1.0):
        raise ValueError(
            f"top_p must be in (0,1], got {top_p}"
        )

    probs = stable_shifted_softmax(
        logits,
        dim=-1,
    )

    sorted_probs, sorted_idx = torch.sort(
        probs,
        dim=-1,
        descending=True,
    )

    cdf = torch.cumsum(
        sorted_probs,
        dim=-1,
    )

    keep = cdf <= top_p

    # Always retain the largest-probability token.
    keep[:, 0] = True

    # Retain the first token crossing the nucleus threshold.
    first_over = (~keep).float().argmax(dim=-1)
    row_ids = torch.arange(
        logits.size(0),
        device=logits.device,
    )
    keep[row_ids, first_over] = True

    filtered_probs = torch.where(
        keep,
        sorted_probs,
        torch.zeros_like(sorted_probs),
    )

    denom = filtered_probs.sum(
        dim=-1,
        keepdim=True,
    ).clamp_min(
        torch.finfo(filtered_probs.dtype).tiny
    )

    filtered_probs = filtered_probs / denom

    sampled_sorted = torch.multinomial(
        filtered_probs,
        num_samples=1,
    ).squeeze(-1)

    return sorted_idx.gather(
        1,
        sampled_sorted[:, None],
    ).squeeze(1)


def _linear_schedule(
    step: int,
    num_steps: int,
    start: float,
    end: float,
) -> float:
    """
    Preserves the supplied target-generation implementation.
    """
    if step <= num_steps:
        return float(start)

    return float(end)


def apply_compounding_repetition_penalty(
    logits: torch.Tensor,
    prev_tokens: torch.Tensor,
    gamma: float,
    ignore_tokens: Tuple[int, ...],
) -> torch.Tensor:
    if gamma is None or gamma <= 1.0:
        return logits

    if prev_tokens.dtype != torch.long:
        prev_tokens = prev_tokens.long()

    N, V = logits.shape

    counts = torch.zeros(
        (N, V),
        device=logits.device,
        dtype=torch.float32,
    )

    counts.scatter_add_(
        1,
        prev_tokens,
        torch.ones_like(
            prev_tokens,
            dtype=torch.float32,
        ),
    )

    exponent = (
        counts - 1.0
    ).clamp_min(0.0)

    penalty_factor = torch.pow(
        torch.tensor(
            gamma,
            device=logits.device,
            dtype=logits.dtype,
        ),
        exponent.to(logits.dtype),
    )

    if ignore_tokens:
        ignore = torch.tensor(
            ignore_tokens,
            device=logits.device,
            dtype=torch.long,
        )

        penalty_factor.index_fill_(
            1,
            ignore,
            1.0,
        )

    out = logits.clone()

    positive = logits > 0

    out[positive] = (
        out[positive]
        / penalty_factor[positive]
    )

    out[~positive] = (
        out[~positive]
        * penalty_factor[~positive]
    )

    return out


@torch.no_grad()
def top_p_teacher_targets_single(
    decoder,
    xa: torch.Tensor,
    cond_lens: torch.Tensor,
    bos_idx: int,
    eos_idx: int,
    pad_idx: int,
    mask_idx: Optional[int] = None,
    max_len: int = 60,
    top_p: float = 0.88,
    temperature: float = 0.90,
    length_cap_ratio: float = 0.15,
    min_cap: int = 2,
    rep_gamma: float = 1.15,
    xa_kpm: Optional[torch.Tensor] = None,
    explore_steps: int = 4,
    top_p_start: float = 0.97,
    top_p_end: float = 0.88,
    temperature_start: float = 1.30,
    temperature_end: float = 1.00,
    rep_gamma_start: float = 1.03,
    rep_gamma_end: float = 1.30,
):
    device = xa.device
    B = xa.size(0)

    cap = (
        length_cap_ratio
        * cond_lens.float()
    ).long()

    cap = torch.clamp(
        cap,
        min=min_cap,
        max=max_len - 1,
    )

    seq = torch.full(
        (B, 1),
        bos_idx,
        dtype=torch.long,
        device=device,
    )

    finished = torch.zeros(
        B,
        dtype=torch.bool,
        device=device,
    )

    ignore_for_rep = (
        bos_idx,
        eos_idx,
        pad_idx,
    )

    if mask_idx is not None:
        ignore_for_rep = (
            ignore_for_rep
            + (mask_idx,)
        )

    for step in range(1, max_len):

        try:
            logits_full = decoder(
                x=seq,
                xa=xa,
                xa_kpm=xa_kpm,
            )
        except TypeError:
            logits_full = decoder(
                x=seq,
                xa=xa,
            )

        logits = logits_full[:, -1, :]

        if step <= explore_steps:
            rep_now = _linear_schedule(
                step,
                explore_steps,
                rep_gamma_start,
                rep_gamma_end,
            )

            temp_now = _linear_schedule(
                step,
                explore_steps,
                temperature_start,
                temperature_end,
            )

            p_now = _linear_schedule(
                step,
                explore_steps,
                top_p_start,
                top_p_end,
            )

        else:
            rep_now = float(
                rep_gamma_end
            )

            temp_now = float(
                temperature_end
            )

            p_now = float(
                top_p_end
            )

        logits = apply_compounding_repetition_penalty(
            logits,
            seq,
            gamma=rep_now,
            ignore_tokens=ignore_for_rep,
        )

        logits = (
            logits
            / temp_now
        )

        logits = logits.clone()

        # PAD unavailable before termination.
        logits[
            ~finished,
            pad_idx,
        ] = -1e9

        # Force EOS at the cap.
        must_eos = (
            (~finished)
            & (step >= cap)
        )

        if must_eos.any():
            logits[
                must_eos,
                :,
            ] = -1e9

            logits[
                must_eos,
                eos_idx,
            ] = 0.0

        # Finished rows emit PAD.
        if finished.any():
            logits[
                finished,
                :,
            ] = -1e9

            logits[
                finished,
                pad_idx,
            ] = 0.0

        next_tok = sample_top_p(
            logits,
            p_now,
        )

        seq = torch.cat(
            [
                seq,
                next_tok[:, None],
            ],
            dim=1,
        )

        finished |= next_tok.eq(
            eos_idx
        )

        if bool(
            finished.all()
        ):
            break

    if seq.size(1) < max_len:
        pad = torch.full(
            (
                B,
                max_len - seq.size(1),
            ),
            pad_idx,
            dtype=torch.long,
            device=device,
        )

        seq = torch.cat(
            [
                seq,
                pad,
            ],
            dim=1,
        )

    else:
        seq = seq[:, :max_len]

    # Normalize everything after first EOS to PAD.
    for b in range(B):

        eos_pos = (
            seq[b] == eos_idx
        ).nonzero(
            as_tuple=True
        )[0]

        if eos_pos.numel() == 0:
            raise RuntimeError(
                f"Teacher target {b} contains no EOS"
            )

        end = (
            int(
                eos_pos[0].item()
            )
            + 1
        )

        if end < max_len:
            seq[
                b,
                end:,
            ] = pad_idx

    return seq.unsqueeze(1)


@torch.no_grad()
def generate_anchor_target_bank(
    model,
    anc: torch.Tensor,
    max_len: int,
    length_cap_ratio: float,
    min_cap: int,
) -> torch.Tensor:

    model.encoder_teacher.eval()
    model.decoder_teacher.eval()

    if hasattr(
        model.decoder_teacher,
        "set_self_attn_drop_p",
    ):
        model.decoder_teacher.set_self_attn_drop_p(
            0.0
        )

    z = model.encoder_teacher(
        anc
    )

    B, T = z.shape[:2]

    cond_lens = torch.full(
        (B,),
        T,
        dtype=torch.long,
        device=z.device,
    )

    return top_p_teacher_targets_single(
        decoder=model.decoder_teacher,
        xa=z,
        cond_lens=cond_lens,
        bos_idx=model.decoder_teacher.BOS_IDX,
        eos_idx=model.decoder_teacher.EOS_IDX,
        pad_idx=model.decoder_teacher.PAD_IDX,
        mask_idx=getattr(
            model.decoder_teacher,
            "MASK_IDX",
            None,
        ),
        max_len=max_len,
        top_p=0.88,
        temperature=0.90,
        length_cap_ratio=length_cap_ratio,
        min_cap=min_cap,
        rep_gamma=1.15,
        explore_steps=4,
        top_p_start=0.97,
        top_p_end=0.88,
        temperature_start=1.30,
        temperature_end=1.00,
        rep_gamma_start=1.03,
        rep_gamma_end=1.30,
    )[:, 0, :]


# ============================================================
# Clean NCE score matrix
# ============================================================


@torch.no_grad()
def per_seq_mean_nll_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pad_idx: int,
) -> torch.Tensor:

    B, T, V = logits.shape

    nll_tok = F.cross_entropy(
        logits.reshape(-1, V),
        targets.reshape(-1),
        reduction="none",
        label_smoothing=0.0,
    ).view(
        B,
        T,
    )

    mask = targets.ne(
        pad_idx
    ).float()

    denom = mask.sum(
        dim=1
    ).clamp_min(1.0)

    return (
        (nll_tok * mask).sum(dim=1)
        / denom
    )


@torch.no_grad()
def build_clean_ll_matrix(
    decoder,
    reps: torch.Tensor,
    tok: torch.Tensor,
    pad_idx: int,
    row_chunk: int = 4,
) -> torch.Tensor:
    """
    ll[i,j] =
        mean clean-prefix log-likelihood
        of target j under condition i.

    Decoder is evaluated deterministically:
      * eval mode;
      * structured-SA dropout = 0;
      * no prefix corruption.
    """

    decoder.eval()

    if hasattr(
        decoder,
        "set_self_attn_drop_p",
    ):
        decoder.set_self_attn_drop_p(
            0.0
        )

    device = reps.device

    tok = tok.to(
        device
    )

    B, T, D = reps.shape
    _, L = tok.shape

    x = tok[:, :-1]
    y = tok[:, 1:]

    rows = []

    for r0 in range(
        0,
        B,
        row_chunk,
    ):

        r1 = min(
            B,
            r0 + row_chunk,
        )

        R = r1 - r0

        reps_r = reps[
            r0:r1
        ]

        x_big = (
            x.unsqueeze(0)
            .expand(
                R,
                B,
                L - 1,
            )
            .reshape(
                R * B,
                L - 1,
            )
        )

        y_big = (
            y.unsqueeze(0)
            .expand(
                R,
                B,
                L - 1,
            )
            .reshape(
                R * B,
                L - 1,
            )
        )

        xa_big = (
            reps_r.unsqueeze(1)
            .expand(
                R,
                B,
                T,
                D,
            )
            .reshape(
                R * B,
                T,
                D,
            )
        )

        logits = decoder(
            x=x_big,
            xa=xa_big,
        )

        nll = per_seq_mean_nll_from_logits(
            logits,
            y_big,
            pad_idx,
        )

        rows.append(
            (-nll).view(
                R,
                B,
            )
        )

    if hasattr(
        decoder,
        "set_self_attn_drop_p",
    ):
        decoder.set_self_attn_drop_p(
            0.0
        )

    return torch.cat(
        rows,
        dim=0,
    )


def row_nce_loss_and_margin(
    S: torch.Tensor,
    tau: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Numerically stable exact per-row NCE loss and realized row margin.

    S:
        (B,B) score matrix.

    S[i,i]:
        paired target score.

    S[i,j], j != i:
        mismatched candidate scores.

    Define

        z_ij = (S_ij - S_ii) / tau.

    Then

        ell_i
        =
        log[
            1 +
            sum_{j != i} exp(z_ij)
        ].

    Since z_ii = 0, equivalently

        ell_i
        =
        logsumexp_j z_ij.

    The exact realized row margin is

        Delta_i
        =
        tau * log[
            1 / (exp(ell_i)-1)
        ]

    but evaluating it via expm1(ell_i) is numerically unstable when ell_i is
    tiny. Instead use the exactly equivalent expression

        Delta_i
        =
        -tau *
        logsumexp_{j != i}
        [
            (S_ij-S_ii)/tau
        ].

    BOTH log-sum-exp terms are evaluated with

        exp(z - max(z))

    stabilization in float64.
    """

    if tau <= 0:
        raise ValueError(
            f"tau must be > 0, got {tau}"
        )

    if S.ndim != 2:
        raise ValueError(
            f"S must be 2-D, got {S.shape}"
        )

    B0, B1 = S.shape

    if B0 != B1:
        raise ValueError(
            f"S must be square, got {S.shape}"
        )

    if B0 < 2:
        raise ValueError(
            "NCE margin requires at least one negative candidate"
        )

    S64 = (
        S.detach()
        .to(dtype=torch.float64)
    )

    if not torch.isfinite(S64).all():
        bad = (
            ~torch.isfinite(S64)
        ).nonzero(
            as_tuple=False
        )

        raise FloatingPointError(
            "Non-finite score encountered in NCE matrix. "
            f"First bad positions: {bad[:10].tolist()}"
        )

    diag = torch.diagonal(
        S64,
        offset=0,
    )

    # Relative logits:
    #
    #   z_ij = (S_ij-S_ii)/tau
    #
    # Diagonal is exactly zero.
    relative = (
        S64
        - diag[:, None]
    ) / float(tau)

    # --------------------------------------------------------
    # Exact row loss
    #
    # ell_i = logsumexp_j(relative_ij)
    #
    # stable_shifted_logsumexp explicitly evaluates:
    #
    # max(z) + log sum exp(z-max(z)).
    # --------------------------------------------------------
    ell = stable_shifted_logsumexp(
        relative,
        dim=1,
    )

    # --------------------------------------------------------
    # Exact NCE row-margin lower bound
    #
    # Remove diagonal, then:
    #
    # log_neg_mass =
    #     logsumexp_{j != i} relative_ij
    #
    # Delta_i = -tau * log_neg_mass.
    # --------------------------------------------------------
    eye = torch.eye(
        B0,
        dtype=torch.bool,
        device=S64.device,
    )

    neg_relative = relative.masked_fill(
        eye,
        float("-inf"),
    )

    log_neg_mass = stable_shifted_logsumexp(
        neg_relative,
        dim=1,
    )

    Delta = (
        -float(tau)
        * log_neg_mass
    )

    # --------------------------------------------------------
    # Numerical consistency check:
    #
    # Delta <= min_j!=i(S_ii-S_ij)
    #
    # should hold algebraically.
    # --------------------------------------------------------
    neg_margins = (
        diag[:, None]
        - S64
    ).masked_fill(
        eye,
        float("inf"),
    )

    min_margin = neg_margins.min(
        dim=1
    ).values

    numeric_eps = 1e-10

    bad = (
        Delta
        > min_margin + numeric_eps
    )

    if bad.any():
        ids = bad.nonzero(
            as_tuple=False
        ).flatten()

        examples = []

        for idx in ids[:10]:
            i = int(idx.item())

            examples.append(
                {
                    "row": i,
                    "Delta": float(
                        Delta[i].item()
                    ),
                    "min_margin": float(
                        min_margin[i].item()
                    ),
                    "ell": float(
                        ell[i].item()
                    ),
                }
            )

        raise FloatingPointError(
            "Stable NCE margin identity violated numerically. "
            "This should not occur for finite scores. "
            f"Examples: {examples}"
        )

    return (
        ell.cpu().numpy(),
        Delta.cpu().numpy(),
    )


# ============================================================
# Positive-fitting stochastic predictions
# ============================================================


def token_dropout_to_fraction2(
    x: torch.Tensor,
    pad_idx: int,
    bos_idx: int,
    eos_idx: int,
    mask_idx: int,
    target_frac: float,
    min_keep: int = 1,
    max_force_k: int = 3,
) -> torch.Tensor:
    """
    Mirrors supplied Stage-III prefix-masking implementation.
    """

    assert (
        0.0
        <= target_frac
        <= 1.0
    )

    B, L = x.shape

    x2 = x.clone()

    valid = (
        (x != pad_idx)
        & (x != bos_idx)
        & (x != eos_idx)
    )

    valid_count = valid.sum(
        dim=1
    )

    drop_n = torch.round(
        valid_count.float()
        * target_frac
    ).long()

    max_drop = (
        valid_count
        - min_keep
    ).clamp_min(0)

    drop_n = torch.minimum(
        drop_n,
        max_drop,
    )

    if (
        drop_n.max().item()
        <= 0
    ):
        return x2

    if max_force_k > 0:
        k = torch.randint(
            1,
            max_force_k + 1,
            (B,),
            device=x.device,
        )
    else:
        k = torch.zeros(
            B,
            dtype=torch.long,
            device=x.device,
        )

    valid_rank = valid.cumsum(
        dim=1
    )

    first_k_mask = (
        valid
        & (
            valid_rank
            <= k.unsqueeze(1)
        )
    )

    forced_n = torch.minimum(
        first_k_mask.sum(dim=1),
        drop_n,
    )

    # Preserve supplied code exactly.
    drop_mask = first_k_mask.clone()

    remain_n = (
        drop_n
        - forced_n
    )

    if (
        remain_n.max().item()
        > 0
    ):

        candidates = (
            valid
            & (~drop_mask)
        )

        scores = torch.rand(
            (B, L),
            device=x.device,
        )

        scores = scores.masked_fill(
            ~candidates,
            float("inf"),
        )

        order = scores.argsort(
            dim=1
        )

        rank = (
            torch.arange(
                L,
                device=x.device,
            )
            .unsqueeze(0)
            .expand(
                B,
                L,
            )
        )

        take = (
            rank
            < remain_n.unsqueeze(1)
        )

        extra_mask = torch.zeros(
            (B, L),
            dtype=torch.bool,
            device=x.device,
        )

        extra_mask.scatter_(
            1,
            order,
            take,
        )

        drop_mask |= extra_mask

    x2[
        drop_mask
    ] = mask_idx

    return x2


@torch.no_grad()
def stochastic_prediction_pair_batch(
    model,
    target: torch.Tensor,
    z_anc: torch.Tensor,
    z_pos: torch.Tensor,
    prefix_mask_frac: float,
    sa_drop_p: float,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """
    One stochastic positive-fitting realization for an MC-expanded batch.

    IMPORTANT:
    The supplied Stage-III code creates t_anc ONCE and then uses that same
    corrupted prefix in

        decoder_student(x=t_anc, xa=z_ctx_anc)
        decoder_student(x=t_anc, xa=z_ctx_pos)

    Therefore this audit does the same.

    The corrupted teacher-forcing prefix is shared between A/P.

    Decoder dropout remains stochastic because the A/P decoder forwards are
    performed sequentially in train mode.

    Returns
    -------
    pred_A:
        (N,L-1)

    pred_P:
        (N,L-1)

    labels:
        (N,L-1)
    """

    dec = model.decoder_student

    dec.train()

    if hasattr(
        dec,
        "set_self_attn_drop_p",
    ):
        dec.set_self_attn_drop_p(
            float(sa_drop_p)
        )

    prefix = target[
        :,
        :-1,
    ].clone()

    labels = target[
        :,
        1:,
    ].clone()

    if prefix_mask_frac > 0.0:
        prefix = token_dropout_to_fraction2(
            prefix,
            pad_idx=dec.PAD_IDX,
            bos_idx=dec.BOS_IDX,
            eos_idx=dec.EOS_IDX,
            mask_idx=dec.MASK_IDX,
            target_frac=float(
                prefix_mask_frac
            ),
            min_keep=2,
            max_force_k=3,
        )

    # Same corrupted prefix, different acoustic condition.
    # Training-mode stochastic layers are resampled naturally between calls.
    logits_A = dec(
        x=prefix,
        xa=z_anc,
    )

    logits_P = dec(
        x=prefix,
        xa=z_pos,
    )

    pred_A = logits_A.argmax(
        dim=-1
    )

    pred_P = logits_P.argmax(
        dim=-1
    )

    dec.eval()

    if hasattr(
        dec,
        "set_self_attn_drop_p",
    ):
        dec.set_self_attn_drop_p(
            0.0
        )

    return (
        pred_A,
        pred_P,
        labels,
    )


@torch.no_grad()
def collect_mc_positive_predictions(
    model,
    target: torch.Tensor,
    z_anc: torch.Tensor,
    z_pos: torch.Tensor,
    prefix_mask_frac: float,
    sa_drop_p: float,
    mc_samples: int,
    mc_chunk: int,
    seed_base: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """
    Return pred_A/pred_P/labels with shape

        (M,B,L-1).

    One common corrupted prefix is generated for A/P within every MC
    realization, matching the supplied Stage-III training path.
    """

    B = target.size(0)

    out_A = []
    out_P = []
    out_Y = []

    done = 0
    chunk_id = 0

    while done < mc_samples:

        m = min(
            mc_chunk,
            mc_samples - done,
        )

        t_rep = target.repeat(
            m,
            1,
        )

        za_rep = z_anc.repeat(
            m,
            1,
            1,
        )

        zp_rep = z_pos.repeat(
            m,
            1,
            1,
        )

        with torch_rng_scope(
            seed_base
            + 1009 * chunk_id
            + 11,
            device,
        ):
            pred_A, pred_P, labels = (
                stochastic_prediction_pair_batch(
                    model=model,
                    target=t_rep,
                    z_anc=za_rep,
                    z_pos=zp_rep,
                    prefix_mask_frac=prefix_mask_frac,
                    sa_drop_p=sa_drop_p,
                )
            )

        Lm = pred_A.size(1)

        out_A.append(
            pred_A.view(
                m,
                B,
                Lm,
            ).cpu()
        )

        out_P.append(
            pred_P.view(
                m,
                B,
                Lm,
            ).cpu()
        )

        out_Y.append(
            labels.view(
                m,
                B,
                Lm,
            ).cpu()
        )

        done += m
        chunk_id += 1

    return {
        "pred_A": torch.cat(
            out_A,
            dim=0,
        ),
        "pred_P": torch.cat(
            out_P,
            dim=0,
        ),
        "labels": torch.cat(
            out_Y,
            dim=0,
        ),
    }


# ============================================================
# Sequence / edit helpers
# ============================================================


def levenshtein(
    a: Sequence[int],
    b: Sequence[int],
) -> int:

    if len(a) == 0:
        return len(b)

    if len(b) == 0:
        return len(a)

    if len(a) > len(b):
        a, b = b, a

    prev = list(
        range(
            len(a) + 1
        )
    )

    for i, tb in enumerate(
        b,
        start=1,
    ):

        cur = [i]

        for j, ta in enumerate(
            a,
            start=1,
        ):

            cur.append(
                min(
                    cur[-1] + 1,
                    prev[j] + 1,
                    prev[j - 1]
                    + (ta != tb),
                )
            )

        prev = cur

    return int(
        prev[-1]
    )


def extract_scored_target(
    seq: torch.Tensor,
    bos_idx: int,
    eos_idx: int,
    pad_idx: int,
) -> List[int]:
    """
    Return

        Y=[content...,EOS].

    BOS/PAD removed.
    First EOS INCLUDED.
    """

    xs = [
        int(x)
        for x in seq.detach()
        .cpu()
        .tolist()
    ]

    if (
        xs
        and xs[0] == bos_idx
    ):
        xs = xs[1:]

    out: List[int] = []

    for t in xs:

        if t == pad_idx:
            break

        if t == bos_idx:
            continue

        out.append(t)

        if t == eos_idx:
            return out

    raise RuntimeError(
        "Generated target has no EOS before PAD"
    )


def extract_valid_prediction(
    pred_row: torch.Tensor,
    label_row: torch.Tensor,
    pad_idx: int,
) -> List[int]:

    valid = label_row.ne(
        pad_idx
    )

    return [
        int(x)
        for x in pred_row[
            valid
        ].tolist()
    ]


def extract_valid_labels(
    label_row: torch.Tensor,
    pad_idx: int,
) -> List[int]:

    valid = label_row.ne(
        pad_idx
    )

    return [
        int(x)
        for x in label_row[
            valid
        ].tolist()
    ]


def target_distance_matrices(
    targets: List[List[int]],
):
    B = len(targets)

    raw = np.zeros(
        (B, B),
        dtype=np.float64,
    )

    norm = np.zeros(
        (B, B),
        dtype=np.float64,
    )

    Lij = np.zeros(
        (B, B),
        dtype=np.float64,
    )

    for i in range(B):

        for j in range(
            i + 1,
            B,
        ):

            e = levenshtein(
                targets[i],
                targets[j],
            )

            L = max(
                len(targets[i]),
                len(targets[j]),
                1,
            )

            raw[i, j] = e
            raw[j, i] = e

            norm[i, j] = (
                e / L
            )

            norm[j, i] = (
                e / L
            )

            Lij[i, j] = L
            Lij[j, i] = L

    for i in range(B):
        Lij[i, i] = max(
            len(targets[i]),
            1,
        )

    return (
        raw,
        norm,
        Lij,
    )

# ============================================================
# Local monotonicity statistics
# ============================================================


def average_ranks(
    x: np.ndarray,
) -> np.ndarray:
    """
    Average ranks with tie handling, used for Spearman correlation.
    """
    x = np.asarray(
        x,
        dtype=np.float64,
    )

    order = np.argsort(
        x,
        kind="mergesort",
    )

    sx = x[
        order
    ]

    ranks_sorted = np.empty(
        len(x),
        dtype=np.float64,
    )

    start = 0

    while start < len(x):

        end = start + 1

        while (
            end < len(x)
            and sx[end] == sx[start]
        ):
            end += 1

        avg = 0.5 * (
            (start + 1)
            + end
        )

        ranks_sorted[
            start:end
        ] = avg

        start = end

    ranks = np.empty(
        len(x),
        dtype=np.float64,
    )

    ranks[
        order
    ] = ranks_sorted

    return ranks


def spearman_np(
    x: np.ndarray,
    y: np.ndarray,
) -> float:
    """
    Descriptive row-local monotonicity diagnostic.

    This is NOT used to construct the formal certificate. The certificate is
    constructed exclusively from the LOO monotone lower envelope.
    """
    x = np.asarray(
        x,
        dtype=np.float64,
    )

    y = np.asarray(
        y,
        dtype=np.float64,
    )

    if len(x) < 2:
        return float("nan")

    rx = average_ranks(
        x
    )

    ry = average_ranks(
        y
    )

    if (
        np.std(rx) <= 0
        or np.std(ry) <= 0
    ):
        return float("nan")

    return float(
        np.corrcoef(
            rx,
            ry,
        )[0, 1]
    )


def top_negative_indices(
    scores_row: np.ndarray,
    i: int,
) -> np.ndarray:
    """
    Rank mismatched candidates by conditional likelihood, highest first.

    Thus the beginning of the returned array contains the hardest negatives.
    """
    idx = np.arange(
        len(scores_row)
    )

    neg = idx[
        idx != i
    ]

    return neg[
        np.argsort(
            scores_row[
                neg
            ]
        )[::-1]
    ]


# ============================================================
# LOO maximal monotone lower envelope
# ============================================================


def monotone_lower_envelope_query(
    train_delta: np.ndarray,
    train_d: np.ndarray,
    query: float,
    tol: float = 1e-12,
) -> Dict[str, Any]:
    """
    Evaluate the maximal non-decreasing empirical lower envelope

        kappa_hat(u)
            =
            min_{m : delta_m >= u} d_m

    using ONLY the supplied calibration candidates.

    This is exactly the LOO estimator used in the revised theory.

    Parameters
    ----------
    train_delta:
        Paired-versus-candidate likelihood margins of the calibration
        candidates. Candidate j must already have been removed.

    train_d:
        Corresponding normalized edit distances.

    query:
        Margin u at which to evaluate kappa_hat.

    tol:
        Small numerical tolerance. The eligible-set comparison uses

            delta_m >= u - tol

        which can only make the empirical lower envelope weakly more
        conservative.

    Returns
    -------
    dict containing

        supported:
            Whether the query lies inside the non-extrapolated domain

                u <= max_m delta_m.

        q:
            kappa_hat(u) if supported, otherwise NaN.

        support_count:
            Number of calibration candidates satisfying delta_m >= u.

        train_min_delta, train_max_delta:
            Range of calibration margins.

        train_min_d, train_max_d:
            Range of calibration edit distances.

    Notes
    -----
    We deliberately do NOT extrapolate for

        u > max_m delta_m.

    Such queries are marked unsupported rather than assigned a bound.
    """

    train_delta = np.asarray(
        train_delta,
        dtype=np.float64,
    )

    train_d = np.asarray(
        train_d,
        dtype=np.float64,
    )

    if train_delta.ndim != 1 or train_d.ndim != 1:
        raise ValueError(
            "train_delta and train_d must be one-dimensional"
        )

    if len(train_delta) != len(train_d):
        raise ValueError(
            "train_delta and train_d must have identical length"
        )

    finite = (
        np.isfinite(train_delta)
        & np.isfinite(train_d)
    )

    train_delta = train_delta[
        finite
    ]

    train_d = train_d[
        finite
    ]

    if len(train_delta) == 0:
        return {
            "supported": False,
            "q": float("nan"),
            "support_count": 0,
            "train_min_delta": float("nan"),
            "train_max_delta": float("nan"),
            "train_min_d": float("nan"),
            "train_max_d": float("nan"),
        }

    if not math.isfinite(
        float(query)
    ):
        return {
            "supported": False,
            "q": float("nan"),
            "support_count": 0,
            "train_min_delta": float(
                np.min(train_delta)
            ),
            "train_max_delta": float(
                np.max(train_delta)
            ),
            "train_min_d": float(
                np.min(train_d)
            ),
            "train_max_d": float(
                np.max(train_d)
            ),
        }

    train_min_delta = float(
        np.min(
            train_delta
        )
    )

    train_max_delta = float(
        np.max(
            train_delta
        )
    )

    train_min_d = float(
        np.min(
            train_d
        )
    )

    train_max_d = float(
        np.max(
            train_d
        )
    )

    # No extrapolation above the largest calibration margin.
    supported = bool(
        float(query)
        <= train_max_delta + tol
    )

    if not supported:
        return {
            "supported": False,
            "q": float("nan"),
            "support_count": 0,
            "train_min_delta": train_min_delta,
            "train_max_delta": train_max_delta,
            "train_min_d": train_min_d,
            "train_max_d": train_max_d,
        }

    # Slightly conservative numerical comparison.
    eligible = (
        train_delta
        >= float(query) - tol
    )

    support_count = int(
        eligible.sum()
    )

    if support_count <= 0:
        # This should not happen if the support check above succeeds, but fail
        # safely rather than silently extrapolating.
        return {
            "supported": False,
            "q": float("nan"),
            "support_count": 0,
            "train_min_delta": train_min_delta,
            "train_max_delta": train_max_delta,
            "train_min_d": train_min_d,
            "train_max_d": train_max_d,
        }

    q = float(
        np.min(
            train_d[
                eligible
            ]
        )
    )

    return {
        "supported": True,
        "q": q,
        "support_count": support_count,
        "train_min_delta": train_min_delta,
        "train_max_delta": train_max_delta,
        "train_min_d": train_min_d,
        "train_max_d": train_max_d,
    }


# ============================================================
# Aggregation helpers
# ============================================================


@dataclass
class MetricBag:

    vals: Dict[
        str,
        List[float],
    ] = field(
        default_factory=dict
    )

    counts: Dict[
        str,
        int,
    ] = field(
        default_factory=dict
    )

    def add_value(
        self,
        name: str,
        value: float,
    ):
        if value is None:
            return

        value = float(
            value
        )

        if math.isfinite(
            value
        ):
            self.vals.setdefault(
                name,
                [],
            ).append(
                value
            )

    def add_count(
        self,
        name: str,
        value: int = 1,
    ):
        self.counts[
            name
        ] = (
            self.counts.get(
                name,
                0,
            )
            + int(value)
        )

    def summarize(
        self,
    ) -> Dict[str, Any]:

        out: Dict[
            str,
            Any,
        ] = dict(
            self.counts
        )

        for name, xs in self.vals.items():

            if not xs:
                continue

            a = np.asarray(
                xs,
                dtype=np.float64,
            )

            out[
                f"{name}_mean"
            ] = float(
                np.mean(a)
            )

            out[
                f"{name}_median"
            ] = float(
                np.median(a)
            )

            out[
                f"{name}_p25"
            ] = float(
                np.quantile(
                    a,
                    0.25,
                )
            )

            out[
                f"{name}_p75"
            ] = float(
                np.quantile(
                    a,
                    0.75,
                )
            )

        return out


# ============================================================
# Main audit
# ============================================================


@torch.no_grad()
def run_audit(
    args,
):

    root_dir = args.root_dir

    config_path = os.path.join(
        root_dir,
        "config",
        args.config,
    )

    with open(
        config_path,
        "r",
    ) as f:

        cfg = yaml.load(
            f,
            Loader=yaml.FullLoader,
        )

    if args.split not in cfg:
        raise ValueError(
            f"Split {args.split!r} not found in config"
        )

    num_workers = (
        args.num_workers
        if args.num_workers is not None
        else int(
            cfg["load_workers"]
        )
    )

    add_noise = (
        not args.no_add_noise
    )

    device = torch.device(
        f"cuda:{args.device}"
        if torch.cuda.is_available()
        else "cpu"
    )

    # --------------------------------------------------------
    # Load trained PairAlign model
    # --------------------------------------------------------

    if device.type == "cuda":

        model = (
            AudioTokenizerAR_FINAL
            .load_from_pretrained(
                args.ckpt,
                args.device,
            )
        )

    else:

        if not hasattr(
            AudioTokenizerAR_FINAL,
            "load_from_checkpoint",
        ):
            raise RuntimeError(
                "CPU path requires Lightning load_from_checkpoint"
            )

        model = (
            AudioTokenizerAR_FINAL
            .load_from_checkpoint(
                args.ckpt,
                map_location="cpu",
                strict=False,
            )
        )

    model.to(
        device
    )

    for p in model.parameters():
        p.requires_grad_(
            False
        )

    model.eval()

    model.encoder_teacher.eval()
    model.encoder_student.eval()
    model.decoder_teacher.eval()
    model.decoder_student.eval()

    if hasattr(
        model.decoder_teacher,
        "set_self_attn_drop_p",
    ):
        model.decoder_teacher.set_self_attn_drop_p(
            0.0
        )

    if hasattr(
        model.decoder_student,
        "set_self_attn_drop_p",
    ):
        model.decoder_student.set_self_attn_drop_p(
            0.0
        )

    BOS = (
        model.decoder_student.BOS_IDX
    )

    EOS = (
        model.decoder_student.EOS_IDX
    )

    PAD = (
        model.decoder_student.PAD_IDX
    )

    # --------------------------------------------------------
    # Output paths
    # --------------------------------------------------------

    out_dir = Path(
        args.out_dir
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    row_path = (
        out_dir
        / "local_row_monotonicity.csv"
    )

    loo_path = (
        out_dir
        / "local_loo_monotone_joint.csv"
    )

    contraction_path = (
        out_dir
        / "contraction_rows.csv"
    )

    combined_path = (
        out_dir
        / "combined_joint.csv"
    )

    summary_path = (
        out_dir
        / "summary.csv"
    )

    config_out = (
        out_dir
        / "run_config.json"
    )

    run_cfg = vars(
        args
    ).copy()

    run_cfg.update(
        {
            "device_resolved": str(
                device
            ),
            "dataset_csv": cfg[
                args.split
            ]["dataset_csv"],
            "fs": cfg["fs"],
            "seglen": cfg["seglen"],
            "n_mels": cfg["n_mels"],
            "add_noise": add_noise,

            "local_calibration": (
                "leave-one-candidate-out maximal "
                "non-decreasing empirical lower envelope"
            ),

            "envelope_definition": (
                "kappa_hat(u) = "
                "min_{m != j : delta_m >= u} d_m"
            ),

            "envelope_extrapolation": (
                "disabled above max calibration margin"
            ),

            "nce_stabilization": (
                "float64 shifted logsumexp "
                "using exp(z-max(z))"
            ),

            "softmax_stabilization": (
                "shifted exp(z-max(z))"
            ),

            "positive_prefix_pairing": (
                "same corrupted prefix reused "
                "for anchor and positive path"
            ),
        }
    )

    with open(
        config_out,
        "w",
    ) as f:

        json.dump(
            run_cfg,
            f,
            indent=2,
        )

    # ========================================================
    # CSV schemas
    # ========================================================

    row_fields = [
        "bank",
        "batch",
        "global_row",
        "file_id",
        "view",
        "k_requested",
        "k_eff",
        "batch_size",

        "nce_loss_row",
        "delta_nce_row",
        "nce_margin_positive",
        "min_observed_margin_all",
        "nce_margin_violation",

        # Descriptive only. Not used for certificates.
        "spearman_full",
        "delta_mean",
        "delta_median",
        "delta_p25",
        "delta_p75",
        "d_norm_mean",
        "d_norm_median",
        "d_norm_p25",
        "d_norm_p75",
    ]

    loo_fields = [
        "bank",
        "batch",
        "global_row",
        "file_id",
        "view",
        "k_requested",
        "k_eff",
        "candidate_col",
        "candidate_file_id",
        "candidate_rank",
        "batch_size",

        "d_norm",
        "ed_target_negative",
        "L_ij",
        "delta_observed",
        "nce_loss_row",
        "delta_nce_row",

        # LOO monotone-envelope construction.
        "envelope_train_min_delta",
        "envelope_train_max_delta",

        "obs_query_supported",
        "obs_support_count",
        "q_obs_predicted",

        "nce_query_supported",
        "nce_support_count",
        "q_nce_predicted",

        # Held-out validation.
        "calibration_slack",
        "loo_calibration_valid",

        # Validated q values. Zero when calibration does not validate.
        "q_obs_validated",
        "q_nce_validated",
        "q_obs_positive",
        "q_nce_positive",

        # Positive-contraction realization.
        "mc_samples",
        "mean_eA",
        "mean_eP",
        "mean_ed_AP",
        "mean_C_conservative",
        "mean_C_realized",
        "triangle_violation_rate",

        # Direct empirical target relation.
        "actual_order_rate",

        # Direct target-distance sufficient condition.
        "geom_gap_positive_rate",
        "geom_certificate_violation_rate",

        # Conservative calibrated certificates.
        "formal_obs_certificate_rate",
        "formal_nce_certificate_rate",
        "formal_obs_certificate_violation_rate",
        "formal_nce_certificate_violation_rate",
        "mean_gamma_geom",
        "mean_gamma_obs",
        "mean_gamma_nce",

        # Realized-contraction calibrated certificates.
        "realized_geom_gap_positive_rate",
        "realized_geom_certificate_violation_rate",
        "realized_formal_obs_certificate_rate",
        "realized_formal_nce_certificate_rate",
        "realized_formal_obs_certificate_violation_rate",
        "realized_formal_nce_certificate_violation_rate",
        "mean_gamma_geom_realized",
        "mean_gamma_obs_realized",
        "mean_gamma_nce_realized",
    ]

    contraction_fields = [
        "bank",
        "batch",
        "global_row",
        "file_id",
        "mc_samples",

        "mean_eA",
        "mean_eP",
        "mean_ed_AP",

        "mean_positive_triangle_rhs",
        "triangle_violation_rate",

        "mean_C_conservative",
        "mean_C_realized",
    ]

    combined_fields = [
        "bank",
        "batch",
        "global_row",
        "file_id",
        "k_requested",
        "candidate_col",
        "candidate_file_id",

        "valid_views_calibration",
        "valid_views_obs_positive",
        "valid_views_nce_positive",

        "q_obs_combined",
        "q_nce_combined",

        "ed_target_negative",
        "L_ij",
        "mc_samples",

        "actual_order_rate",

        # Conservative.
        "combined_obs_certificate_rate",
        "combined_nce_certificate_rate",
        "combined_obs_certificate_violation_rate",
        "combined_nce_certificate_violation_rate",
        "mean_gamma_obs_combined",
        "mean_gamma_nce_combined",

        # Realized.
        "realized_combined_obs_certificate_rate",
        "realized_combined_nce_certificate_rate",
        "realized_combined_obs_certificate_violation_rate",
        "realized_combined_nce_certificate_violation_rate",
        "mean_gamma_obs_combined_realized",
        "mean_gamma_nce_combined_realized",
    ]

    # --------------------------------------------------------
    # Open output CSVs
    # --------------------------------------------------------

    row_f = open(
        row_path,
        "w",
        newline="",
    )

    loo_f = open(
        loo_path,
        "w",
        newline="",
    )

    con_f = open(
        contraction_path,
        "w",
        newline="",
    )

    comb_f = open(
        combined_path,
        "w",
        newline="",
    )

    row_w = csv.DictWriter(
        row_f,
        fieldnames=row_fields,
    )

    loo_w = csv.DictWriter(
        loo_f,
        fieldnames=loo_fields,
    )

    con_w = csv.DictWriter(
        con_f,
        fieldnames=contraction_fields,
    )

    comb_w = csv.DictWriter(
        comb_f,
        fieldnames=combined_fields,
    )

    row_w.writeheader()
    loo_w.writeheader()
    con_w.writeheader()
    comb_w.writeheader()

    # --------------------------------------------------------
    # Summary accumulators
    # --------------------------------------------------------

    row_bags: Dict[
        Tuple[str, int],
        MetricBag,
    ] = {}

    loo_bags: Dict[
        Tuple[str, int],
        MetricBag,
    ] = {}

    combined_bags: Dict[
        int,
        MetricBag,
    ] = {}

    contraction_bag = MetricBag()

    # ========================================================
    # Audit
    # ========================================================

    try:

        for bank in range(
            args.target_banks
        ):

            # Recreate exactly the same held-out A/P population for each
            # independently sampled EMA target bank.
            seed_everything(
                args.eval_seed,
                workers=False,
            )

            loader = make_loader(
                cfg=cfg,
                split=args.split,
                batch_size=args.batch_size,
                num_workers=num_workers,
                dataloader_seed=args.dataloader_seed,
                add_noise=add_noise,
                use_time_compression=args.use_time_compression,
            )

            bank_global_row = 0

            pbar = tqdm(
                loader,
                desc=(
                    f"LOO monotone audit bank "
                    f"{bank + 1}/"
                    f"{args.target_banks}"
                ),
                unit="batch",
            )

            for (
                batch_idx,
                (
                    anc_feats,
                    pos_feats,
                    file_ids,
                ),
            ) in enumerate(
                pbar
            ):

                if (
                    args.max_batches is not None
                    and batch_idx >= args.max_batches
                ):
                    break

                # ------------------------------------------------
                # Prepare anchor / positive acoustic inputs
                # ------------------------------------------------

                anc_feats = anc_feats.to(
                    device,
                    non_blocking=True,
                )

                pos_feats = pos_feats.to(
                    device,
                    non_blocking=True,
                )

                anc = prepare_model_input(
                    model,
                    anc_feats,
                )

                pos = prepare_model_input(
                    model,
                    pos_feats,
                )

                # ------------------------------------------------
                # Independently sampled anchor-derived EMA bank
                # ------------------------------------------------

                target_seed = (
                    args.eval_seed
                    + bank * 10_000_019
                    + batch_idx * 100_003
                )

                with torch_rng_scope(
                    target_seed,
                    device,
                ):

                    target = generate_anchor_target_bank(
                        model=model,
                        anc=anc,
                        max_len=args.max_len,
                        length_cap_ratio=(
                            args.length_cap_ratio
                        ),
                        min_cap=args.min_cap,
                    )

                model.encoder_student.eval()

                zA = model.encoder_student(
                    anc
                )

                zP = model.encoder_student(
                    pos
                )

                B = target.size(
                    0
                )

                if B < 3:
                    continue

                # ------------------------------------------------
                # Clean likelihood matrices
                # ------------------------------------------------

                SA = build_clean_ll_matrix(
                    model.decoder_student,
                    zA,
                    target,
                    PAD,
                    row_chunk=(
                        args.score_row_chunk
                    ),
                )

                SP = build_clean_ll_matrix(
                    model.decoder_student,
                    zP,
                    target,
                    PAD,
                    row_chunk=(
                        args.score_row_chunk
                    ),
                )

                ellA, DeltaA = row_nce_loss_and_margin(
                    SA,
                    args.tau,
                )

                ellP, DeltaP = row_nce_loss_and_margin(
                    SP,
                    args.tau,
                )

                score_np = {
                    "A": (
                        SA.detach()
                        .double()
                        .cpu()
                        .numpy()
                    ),
                    "P": (
                        SP.detach()
                        .double()
                        .cpu()
                        .numpy()
                    ),
                }

                ell_np = {
                    "A": ellA,
                    "P": ellP,
                }

                Delta_np = {
                    "A": DeltaA,
                    "P": DeltaP,
                }

                # ------------------------------------------------
                # EMA strings + exact target edit geometry
                # ------------------------------------------------

                target_lists = [
                    extract_scored_target(
                        target[i],
                        BOS,
                        EOS,
                        PAD,
                    )
                    for i in range(B)
                ]

                (
                    D_raw,
                    D_norm,
                    Lij,
                ) = target_distance_matrices(
                    target_lists
                )

                # ------------------------------------------------
                # Positive-contraction MC realizations
                # ------------------------------------------------

                mc_seed = (
                    args.eval_seed
                    + bank * 30_000_017
                    + batch_idx * 700_001
                )

                mc = collect_mc_positive_predictions(
                    model=model,
                    target=target,
                    z_anc=zA,
                    z_pos=zP,
                    prefix_mask_frac=(
                        args.prefix_mask_frac
                    ),
                    sa_drop_p=(
                        args.sa_drop_p
                    ),
                    mc_samples=(
                        args.mc_samples
                    ),
                    mc_chunk=(
                        args.mc_chunk
                    ),
                    seed_base=mc_seed,
                    device=device,
                )

                # ------------------------------------------------
                # Extract realized contraction quantities
                # ------------------------------------------------

                contraction: List[
                    Dict[str, Any]
                ] = []

                for i in range(
                    B
                ):

                    eA_list = []
                    eP_list = []
                    edAP_list = []

                    C_cons_list = []
                    C_real_list = []

                    predA_lists = []
                    predP_lists = []

                    for m in range(
                        args.mc_samples
                    ):

                        labels_i = (
                            mc["labels"][
                                m,
                                i,
                            ]
                        )

                        Y_i = extract_valid_labels(
                            labels_i,
                            PAD,
                        )

                        if (
                            Y_i
                            != target_lists[i]
                        ):
                            raise RuntimeError(
                                "MC labels do not match "
                                "the scored EMA target"
                            )

                        pA = extract_valid_prediction(
                            mc["pred_A"][
                                m,
                                i,
                            ],
                            labels_i,
                            PAD,
                        )

                        pP = extract_valid_prediction(
                            mc["pred_P"][
                                m,
                                i,
                            ],
                            labels_i,
                            PAD,
                        )

                        eA = levenshtein(
                            pA,
                            Y_i,
                        )

                        eP = levenshtein(
                            pP,
                            Y_i,
                        )

                        edAP = levenshtein(
                            pA,
                            pP,
                        )

                        C_cons = (
                            eA
                            + eP
                            + max(
                                eA,
                                eP,
                            )
                        )

                        C_real = (
                            edAP
                            + max(
                                eA,
                                eP,
                            )
                        )

                        predA_lists.append(
                            pA
                        )

                        predP_lists.append(
                            pP
                        )

                        eA_list.append(
                            eA
                        )

                        eP_list.append(
                            eP
                        )

                        edAP_list.append(
                            edAP
                        )

                        C_cons_list.append(
                            C_cons
                        )

                        C_real_list.append(
                            C_real
                        )

                    eA_arr = np.asarray(
                        eA_list,
                        dtype=np.float64,
                    )

                    eP_arr = np.asarray(
                        eP_list,
                        dtype=np.float64,
                    )

                    edAP_arr = np.asarray(
                        edAP_list,
                        dtype=np.float64,
                    )

                    C_cons_arr = np.asarray(
                        C_cons_list,
                        dtype=np.float64,
                    )

                    C_real_arr = np.asarray(
                        C_real_list,
                        dtype=np.float64,
                    )

                    tri_rhs = (
                        eA_arr
                        + eP_arr
                    )

                    tri_violation = (
                        edAP_arr
                        > tri_rhs
                        + args.tol
                    )

                    contraction.append(
                        {
                            "predA": predA_lists,
                            "predP": predP_lists,
                            "eA": eA_arr,
                            "eP": eP_arr,
                            "edAP": edAP_arr,
                            "C_cons": C_cons_arr,
                            "C_real": C_real_arr,
                        }
                    )

                    con_w.writerow(
                        {
                            "bank": bank,
                            "batch": batch_idx,
                            "global_row": (
                                bank_global_row
                                + i
                            ),
                            "file_id": (
                                file_ids[i]
                            ),
                            "mc_samples": (
                                args.mc_samples
                            ),
                            "mean_eA": float(
                                eA_arr.mean()
                            ),
                            "mean_eP": float(
                                eP_arr.mean()
                            ),
                            "mean_ed_AP": float(
                                edAP_arr.mean()
                            ),
                            "mean_positive_triangle_rhs": float(
                                tri_rhs.mean()
                            ),
                            "triangle_violation_rate": float(
                                tri_violation.mean()
                            ),
                            "mean_C_conservative": float(
                                C_cons_arr.mean()
                            ),
                            "mean_C_realized": float(
                                C_real_arr.mean()
                            ),
                        }
                    )

                    contraction_bag.add_value(
                        "eA",
                        eA_arr.mean(),
                    )

                    contraction_bag.add_value(
                        "eP",
                        eP_arr.mean(),
                    )

                    contraction_bag.add_value(
                        "ed_AP",
                        edAP_arr.mean(),
                    )

                    contraction_bag.add_value(
                        "C_conservative",
                        C_cons_arr.mean(),
                    )

                    contraction_bag.add_value(
                        "C_realized",
                        C_real_arr.mean(),
                    )

                    contraction_bag.add_count(
                        "triangle_trials",
                        len(
                            tri_violation
                        ),
                    )

                    contraction_bag.add_count(
                        "triangle_violations",
                        int(
                            tri_violation.sum()
                        ),
                    )

                # ------------------------------------------------
                # Cache direct positive-vs-negative ordering
                # ------------------------------------------------

                joint_cache: Dict[
                    Tuple[int, int],
                    Dict[str, np.ndarray],
                ] = {}

                def joint_for(
                    i: int,
                    j: int,
                ) -> Dict[
                    str,
                    np.ndarray,
                ]:

                    key = (
                        i,
                        j,
                    )

                    if key in joint_cache:
                        return joint_cache[
                            key
                        ]

                    c = contraction[
                        i
                    ]

                    Yj = target_lists[
                        j
                    ]

                    edAneg = []
                    edPneg = []
                    actual = []

                    for m in range(
                        args.mc_samples
                    ):

                        ea = levenshtein(
                            c["predA"][m],
                            Yj,
                        )

                        ep = levenshtein(
                            c["predP"][m],
                            Yj,
                        )

                        edAneg.append(
                            ea
                        )

                        edPneg.append(
                            ep
                        )

                        actual.append(
                            c["edAP"][m]
                            < min(
                                ea,
                                ep,
                            )
                        )

                    ans = {
                        "edAneg": np.asarray(
                            edAneg,
                            dtype=np.float64,
                        ),
                        "edPneg": np.asarray(
                            edPneg,
                            dtype=np.float64,
                        ),
                        "actual": np.asarray(
                            actual,
                            dtype=bool,
                        ),
                    }

                    joint_cache[
                        key
                    ] = ans

                    return ans

                # ------------------------------------------------
                # Store valid A/P local bounds before combination
                # ------------------------------------------------

                combined_local: Dict[
                    Tuple[int, int, int],
                    Dict[
                        str,
                        Dict[str, Any],
                    ],
                ] = {}

                # ====================================================
                # Per-view row + LOO monotone audit
                # ====================================================

                for view in (
                    "A",
                    "P",
                ):

                    S = score_np[
                        view
                    ]

                    ell = ell_np[
                        view
                    ]

                    Delta = Delta_np[
                        view
                    ]

                    for i in range(
                        B
                    ):

                        ranking = top_negative_indices(
                            S[i],
                            i,
                        )

                        all_delta = (
                            S[i, i]
                            - S[
                                i,
                                ranking,
                            ]
                        )

                        min_obs_all = float(
                            np.min(
                                all_delta
                            )
                        )

                        # Exact NCE inequality check.
                        nce_violation = bool(
                            min_obs_all
                            + args.tol
                            < float(
                                Delta[i]
                            )
                        )

                        for k_req in (
                            args.k_values
                        ):

                            k_eff = min(
                                int(k_req),
                                B - 1,
                            )

                            if (
                                k_eff
                                < args.min_local_k
                            ):
                                continue

                            neigh = ranking[
                                :k_eff
                            ]

                            d = (
                                D_norm[
                                    i,
                                    neigh,
                                ]
                                .astype(
                                    np.float64
                                )
                            )

                            delta = (
                                S[i, i]
                                - S[
                                    i,
                                    neigh,
                                ]
                            ).astype(
                                np.float64
                            )

                            # ----------------------------------------
                            # Row-level descriptive monotonicity only
                            # ----------------------------------------

                            rho = spearman_np(
                                delta,
                                d,
                            )

                            row_w.writerow(
                                {
                                    "bank": bank,
                                    "batch": batch_idx,
                                    "global_row": (
                                        bank_global_row
                                        + i
                                    ),
                                    "file_id": (
                                        file_ids[i]
                                    ),
                                    "view": view,
                                    "k_requested": (
                                        k_req
                                    ),
                                    "k_eff": (
                                        k_eff
                                    ),
                                    "batch_size": (
                                        B
                                    ),

                                    "nce_loss_row": float(
                                        ell[i]
                                    ),
                                    "delta_nce_row": float(
                                        Delta[i]
                                    ),
                                    "nce_margin_positive": int(
                                        Delta[i]
                                        > 0
                                    ),
                                    "min_observed_margin_all": (
                                        min_obs_all
                                    ),
                                    "nce_margin_violation": int(
                                        nce_violation
                                    ),

                                    "spearman_full": rho,

                                    "delta_mean": float(
                                        np.mean(
                                            delta
                                        )
                                    ),
                                    "delta_median": float(
                                        np.median(
                                            delta
                                        )
                                    ),
                                    "delta_p25": float(
                                        np.quantile(
                                            delta,
                                            0.25,
                                        )
                                    ),
                                    "delta_p75": float(
                                        np.quantile(
                                            delta,
                                            0.75,
                                        )
                                    ),

                                    "d_norm_mean": float(
                                        np.mean(
                                            d
                                        )
                                    ),
                                    "d_norm_median": float(
                                        np.median(
                                            d
                                        )
                                    ),
                                    "d_norm_p25": float(
                                        np.quantile(
                                            d,
                                            0.25,
                                        )
                                    ),
                                    "d_norm_p75": float(
                                        np.quantile(
                                            d,
                                            0.75,
                                        )
                                    ),
                                }
                            )

                            rb = row_bags.setdefault(
                                (
                                    view,
                                    int(
                                        k_req
                                    ),
                                ),
                                MetricBag(),
                            )

                            rb.add_count(
                                "rows",
                                1,
                            )

                            rb.add_count(
                                "nce_positive_margin_rows",
                                int(
                                    Delta[i]
                                    > 0
                                ),
                            )

                            rb.add_count(
                                "nce_margin_violations",
                                int(
                                    nce_violation
                                ),
                            )

                            rb.add_value(
                                "spearman",
                                rho,
                            )

                            rb.add_value(
                                "delta",
                                np.mean(
                                    delta
                                ),
                            )

                            rb.add_value(
                                "d_norm",
                                np.mean(
                                    d
                                ),
                            )

                            rb.add_value(
                                "delta_nce",
                                Delta[i],
                            )

                            rb.add_value(
                                "min_observed_margin_all",
                                min_obs_all,
                            )

                            # ========================================
                            # Candidate-wise LOO monotone calibration
                            # ========================================

                            for (
                                rank0,
                                j,
                            ) in enumerate(
                                neigh
                            ):

                                keep = np.ones(
                                    k_eff,
                                    dtype=bool,
                                )

                                keep[
                                    rank0
                                ] = False

                                train_delta = delta[
                                    keep
                                ]

                                train_d = d[
                                    keep
                                ]

                                d_j = float(
                                    d[
                                        rank0
                                    ]
                                )

                                delta_j = float(
                                    delta[
                                        rank0
                                    ]
                                )

                                # --------------------------------
                                # Candidate-specific LOO prediction
                                # --------------------------------

                                obs_env = monotone_lower_envelope_query(
                                    train_delta=train_delta,
                                    train_d=train_d,
                                    query=delta_j,
                                    tol=args.tol,
                                )

                                obs_supported = bool(
                                    obs_env[
                                        "supported"
                                    ]
                                )

                                q_obs_pred = (
                                    float(
                                        obs_env["q"]
                                    )
                                    if obs_supported
                                    else float(
                                        "nan"
                                    )
                                )

                                # --------------------------------
                                # NCE-derived LOO prediction
                                # --------------------------------

                                nce_env = monotone_lower_envelope_query(
                                    train_delta=train_delta,
                                    train_d=train_d,
                                    query=float(
                                        Delta[i]
                                    ),
                                    tol=args.tol,
                                )

                                nce_supported = bool(
                                    nce_env[
                                        "supported"
                                    ]
                                )

                                q_nce_pred = (
                                    float(
                                        nce_env["q"]
                                    )
                                    if nce_supported
                                    else float(
                                        "nan"
                                    )
                                )

                                # --------------------------------
                                # Sanity: Delta <= delta_j
                                # --------------------------------

                                if (
                                    float(
                                        Delta[i]
                                    )
                                    > delta_j
                                    + 10.0 * args.tol
                                ):
                                    raise RuntimeError(
                                        "NCE row margin exceeds the "
                                        "held-out observed candidate margin. "
                                        f"view={view}, i={i}, j={j}, "
                                        f"Delta={Delta[i]}, delta_j={delta_j}"
                                    )

                                # If delta_j is supported then Delta <= delta_j
                                # must also be supported under the domain
                                # u <= max(train_delta).
                                if (
                                    obs_supported
                                    and not nce_supported
                                ):
                                    raise RuntimeError(
                                        "Observed query is supported but the "
                                        "smaller NCE query is not supported. "
                                        "This contradicts the monotone envelope "
                                        "domain construction."
                                    )

                                # --------------------------------
                                # Non-vacuous held-out calibration event
                                #
                                # C_ij,V:
                                # d_j >= kappa_hat^{-j}(delta_j)
                                # --------------------------------

                                if obs_supported:

                                    calibration_slack = (
                                        d_j
                                        - q_obs_pred
                                    )

                                    calibration_valid = bool(
                                        d_j
                                        + args.tol
                                        >= q_obs_pred
                                    )

                                else:

                                    calibration_slack = float(
                                        "nan"
                                    )

                                    calibration_valid = False

                                # --------------------------------
                                # Validated local q values
                                # --------------------------------

                                if calibration_valid:

                                    q_obs = float(
                                        q_obs_pred
                                    )

                                    if not nce_supported:
                                        raise RuntimeError(
                                            "Validated observed envelope but "
                                            "NCE query unexpectedly unsupported"
                                        )

                                    q_nce = float(
                                        q_nce_pred
                                    )

                                    # Monotonicity should give
                                    # q_obs >= q_nce.
                                    if (
                                        q_nce
                                        > q_obs
                                        + 10.0 * args.tol
                                    ):
                                        raise RuntimeError(
                                            "Monotone certificate hierarchy "
                                            "violated numerically: "
                                            f"q_nce={q_nce} > q_obs={q_obs}"
                                        )

                                else:

                                    # Invalid/unsupported candidates are NOT
                                    # assigned formal lower bounds.
                                    q_obs = 0.0
                                    q_nce = 0.0

                                q_obs_positive = bool(
                                    calibration_valid
                                    and q_obs
                                    > args.tol
                                )

                                q_nce_positive = bool(
                                    calibration_valid
                                    and q_nce
                                    > args.tol
                                )

                                # --------------------------------
                                # Cache per-view validated bounds
                                # --------------------------------

                                combined_local.setdefault(
                                    (
                                        i,
                                        int(
                                            k_req
                                        ),
                                        int(j),
                                    ),
                                    {},
                                )[view] = {
                                    "valid": bool(
                                        calibration_valid
                                    ),
                                    "obs_supported": bool(
                                        obs_supported
                                    ),
                                    "nce_supported": bool(
                                        nce_supported
                                    ),
                                    "q_obs": float(
                                        q_obs
                                    ),
                                    "q_nce": float(
                                        q_nce
                                    ),
                                }

                                # --------------------------------
                                # Joint contraction / separation
                                # --------------------------------

                                jc = joint_for(
                                    i,
                                    int(j),
                                )

                                c = contraction[
                                    i
                                ]

                                C_cons = c[
                                    "C_cons"
                                ]

                                C_real = c[
                                    "C_real"
                                ]

                                actual = jc[
                                    "actual"
                                ]

                                target_neg_dist = float(
                                    D_raw[
                                        i,
                                        j,
                                    ]
                                )

                                raw_obs_lb = (
                                    float(
                                        Lij[
                                            i,
                                            j,
                                        ]
                                    )
                                    * q_obs
                                )

                                raw_nce_lb = (
                                    float(
                                        Lij[
                                            i,
                                            j,
                                        ]
                                    )
                                    * q_nce
                                )

                                # --------------------------------
                                # Conservative thresholds
                                # --------------------------------

                                gamma_geom = (
                                    target_neg_dist
                                    - C_cons
                                )

                                gamma_obs = (
                                    raw_obs_lb
                                    - C_cons
                                )

                                gamma_nce = (
                                    raw_nce_lb
                                    - C_cons
                                )

                                geom_pos = (
                                    gamma_geom
                                    > args.tol
                                )

                                formal_obs = (
                                    (
                                        gamma_obs
                                        > args.tol
                                    )
                                    & calibration_valid
                                    & q_obs_positive
                                )

                                formal_nce = (
                                    (
                                        gamma_nce
                                        > args.tol
                                    )
                                    & calibration_valid
                                    & q_nce_positive
                                )

                                # --------------------------------
                                # Realized-contraction thresholds
                                # --------------------------------

                                gamma_geom_real = (
                                    target_neg_dist
                                    - C_real
                                )

                                gamma_obs_real = (
                                    raw_obs_lb
                                    - C_real
                                )

                                gamma_nce_real = (
                                    raw_nce_lb
                                    - C_real
                                )

                                geom_real_pos = (
                                    gamma_geom_real
                                    > args.tol
                                )

                                formal_obs_real = (
                                    (
                                        gamma_obs_real
                                        > args.tol
                                    )
                                    & calibration_valid
                                    & q_obs_positive
                                )

                                formal_nce_real = (
                                    (
                                        gamma_nce_real
                                        > args.tol
                                    )
                                    & calibration_valid
                                    & q_nce_positive
                                )

                                # --------------------------------
                                # Violation helper
                                # --------------------------------

                                def violation_rate(
                                    mask: np.ndarray,
                                ) -> float:

                                    n = int(
                                        mask.sum()
                                    )

                                    if n == 0:
                                        return float(
                                            "nan"
                                        )

                                    return float(
                                        (
                                            mask
                                            & (~actual)
                                        ).sum()
                                        / n
                                    )

                                # --------------------------------
                                # Per-candidate CSV
                                # --------------------------------

                                loo_w.writerow(
                                    {
                                        "bank": bank,
                                        "batch": batch_idx,
                                        "global_row": (
                                            bank_global_row
                                            + i
                                        ),
                                        "file_id": (
                                            file_ids[i]
                                        ),
                                        "view": view,
                                        "k_requested": (
                                            k_req
                                        ),
                                        "k_eff": (
                                            k_eff
                                        ),
                                        "candidate_col": int(
                                            j
                                        ),
                                        "candidate_file_id": (
                                            file_ids[
                                                int(j)
                                            ]
                                        ),
                                        "candidate_rank": (
                                            rank0 + 1
                                        ),
                                        "batch_size": (
                                            B
                                        ),

                                        "d_norm": (
                                            d_j
                                        ),
                                        "ed_target_negative": (
                                            target_neg_dist
                                        ),
                                        "L_ij": float(
                                            Lij[
                                                i,
                                                j,
                                            ]
                                        ),
                                        "delta_observed": (
                                            delta_j
                                        ),
                                        "nce_loss_row": float(
                                            ell[i]
                                        ),
                                        "delta_nce_row": float(
                                            Delta[i]
                                        ),

                                        "envelope_train_min_delta": (
                                            obs_env[
                                                "train_min_delta"
                                            ]
                                        ),
                                        "envelope_train_max_delta": (
                                            obs_env[
                                                "train_max_delta"
                                            ]
                                        ),

                                        "obs_query_supported": int(
                                            obs_supported
                                        ),
                                        "obs_support_count": int(
                                            obs_env[
                                                "support_count"
                                            ]
                                        ),
                                        "q_obs_predicted": (
                                            q_obs_pred
                                        ),

                                        "nce_query_supported": int(
                                            nce_supported
                                        ),
                                        "nce_support_count": int(
                                            nce_env[
                                                "support_count"
                                            ]
                                        ),
                                        "q_nce_predicted": (
                                            q_nce_pred
                                        ),

                                        "calibration_slack": (
                                            calibration_slack
                                        ),
                                        "loo_calibration_valid": int(
                                            calibration_valid
                                        ),

                                        "q_obs_validated": (
                                            q_obs
                                        ),
                                        "q_nce_validated": (
                                            q_nce
                                        ),
                                        "q_obs_positive": int(
                                            q_obs_positive
                                        ),
                                        "q_nce_positive": int(
                                            q_nce_positive
                                        ),

                                        "mc_samples": (
                                            args.mc_samples
                                        ),
                                        "mean_eA": float(
                                            c["eA"].mean()
                                        ),
                                        "mean_eP": float(
                                            c["eP"].mean()
                                        ),
                                        "mean_ed_AP": float(
                                            c["edAP"].mean()
                                        ),
                                        "mean_C_conservative": float(
                                            C_cons.mean()
                                        ),
                                        "mean_C_realized": float(
                                            C_real.mean()
                                        ),
                                        "triangle_violation_rate": float(
                                            (
                                                c["edAP"]
                                                > (
                                                    c["eA"]
                                                    + c["eP"]
                                                    + args.tol
                                                )
                                            ).mean()
                                        ),

                                        "actual_order_rate": float(
                                            actual.mean()
                                        ),

                                        "geom_gap_positive_rate": float(
                                            geom_pos.mean()
                                        ),
                                        "geom_certificate_violation_rate": violation_rate(
                                            geom_pos
                                        ),

                                        "formal_obs_certificate_rate": float(
                                            formal_obs.mean()
                                        ),
                                        "formal_nce_certificate_rate": float(
                                            formal_nce.mean()
                                        ),
                                        "formal_obs_certificate_violation_rate": violation_rate(
                                            formal_obs
                                        ),
                                        "formal_nce_certificate_violation_rate": violation_rate(
                                            formal_nce
                                        ),

                                        "mean_gamma_geom": float(
                                            gamma_geom.mean()
                                        ),
                                        "mean_gamma_obs": float(
                                            gamma_obs.mean()
                                        ),
                                        "mean_gamma_nce": float(
                                            gamma_nce.mean()
                                        ),

                                        "realized_geom_gap_positive_rate": float(
                                            geom_real_pos.mean()
                                        ),
                                        "realized_geom_certificate_violation_rate": violation_rate(
                                            geom_real_pos
                                        ),

                                        "realized_formal_obs_certificate_rate": float(
                                            formal_obs_real.mean()
                                        ),
                                        "realized_formal_nce_certificate_rate": float(
                                            formal_nce_real.mean()
                                        ),
                                        "realized_formal_obs_certificate_violation_rate": violation_rate(
                                            formal_obs_real
                                        ),
                                        "realized_formal_nce_certificate_violation_rate": violation_rate(
                                            formal_nce_real
                                        ),

                                        "mean_gamma_geom_realized": float(
                                            gamma_geom_real.mean()
                                        ),
                                        "mean_gamma_obs_realized": float(
                                            gamma_obs_real.mean()
                                        ),
                                        "mean_gamma_nce_realized": float(
                                            gamma_nce_real.mean()
                                        ),
                                    }
                                )

                                # --------------------------------
                                # Aggregate LOO statistics
                                # --------------------------------

                                lb = loo_bags.setdefault(
                                    (
                                        view,
                                        int(
                                            k_req
                                        ),
                                    ),
                                    MetricBag(),
                                )

                                lb.add_count(
                                    "loo_candidates",
                                    1,
                                )

                                lb.add_count(
                                    "obs_query_supported",
                                    int(
                                        obs_supported
                                    ),
                                )

                                lb.add_count(
                                    "nce_query_supported",
                                    int(
                                        nce_supported
                                    ),
                                )

                                lb.add_count(
                                    "calibration_valid",
                                    int(
                                        calibration_valid
                                    ),
                                )

                                lb.add_count(
                                    "q_obs_positive_candidates",
                                    int(
                                        q_obs_positive
                                    ),
                                )

                                lb.add_count(
                                    "q_nce_positive_candidates",
                                    int(
                                        q_nce_positive
                                    ),
                                )

                                if obs_supported:

                                    lb.add_value(
                                        "obs_support_count",
                                        obs_env[
                                            "support_count"
                                        ],
                                    )

                                    lb.add_value(
                                        "q_obs_predicted",
                                        q_obs_pred,
                                    )

                                    lb.add_value(
                                        "calibration_slack",
                                        calibration_slack,
                                    )

                                if nce_supported:

                                    lb.add_value(
                                        "nce_support_count",
                                        nce_env[
                                            "support_count"
                                        ],
                                    )

                                    lb.add_value(
                                        "q_nce_predicted",
                                        q_nce_pred,
                                    )

                                if calibration_valid:

                                    lb.add_value(
                                        "q_obs_validated",
                                        q_obs,
                                    )

                                    lb.add_value(
                                        "q_nce_validated",
                                        q_nce,
                                    )

                                # Every local candidate contributes MC trials
                                # to the observed ordering statistic.
                                lb.add_count(
                                    "mc_trials",
                                    args.mc_samples,
                                )

                                lb.add_count(
                                    "actual_order_success",
                                    int(
                                        actual.sum()
                                    ),
                                )

                                # Only calibrated candidates contribute to the
                                # conditional-on-valid denominator.
                                if calibration_valid:

                                    lb.add_count(
                                        "validated_mc_trials",
                                        args.mc_samples,
                                    )

                                # Direct geometric sufficient condition.
                                lb.add_count(
                                    "geom_positive",
                                    int(
                                        geom_pos.sum()
                                    ),
                                )

                                lb.add_count(
                                    "geom_cert_violations",
                                    int(
                                        (
                                            geom_pos
                                            & (~actual)
                                        ).sum()
                                    ),
                                )

                                # Conservative calibrated certificates.
                                lb.add_count(
                                    "formal_obs_cert",
                                    int(
                                        formal_obs.sum()
                                    ),
                                )

                                lb.add_count(
                                    "formal_nce_cert",
                                    int(
                                        formal_nce.sum()
                                    ),
                                )

                                lb.add_count(
                                    "formal_obs_cert_violations",
                                    int(
                                        (
                                            formal_obs
                                            & (~actual)
                                        ).sum()
                                    ),
                                )

                                lb.add_count(
                                    "formal_nce_cert_violations",
                                    int(
                                        (
                                            formal_nce
                                            & (~actual)
                                        ).sum()
                                    ),
                                )

                                # Realized-contraction certificates.
                                lb.add_count(
                                    "realized_geom_positive",
                                    int(
                                        geom_real_pos.sum()
                                    ),
                                )

                                lb.add_count(
                                    "realized_geom_cert_violations",
                                    int(
                                        (
                                            geom_real_pos
                                            & (~actual)
                                        ).sum()
                                    ),
                                )

                                lb.add_count(
                                    "realized_formal_obs_cert",
                                    int(
                                        formal_obs_real.sum()
                                    ),
                                )

                                lb.add_count(
                                    "realized_formal_nce_cert",
                                    int(
                                        formal_nce_real.sum()
                                    ),
                                )

                                lb.add_count(
                                    "realized_formal_obs_cert_violations",
                                    int(
                                        (
                                            formal_obs_real
                                            & (~actual)
                                        ).sum()
                                    ),
                                )

                                lb.add_count(
                                    "realized_formal_nce_cert_violations",
                                    int(
                                        (
                                            formal_nce_real
                                            & (~actual)
                                        ).sum()
                                    ),
                                )

                                lb.add_value(
                                    "gamma_geom",
                                    gamma_geom.mean(),
                                )

                                if calibration_valid:

                                    lb.add_value(
                                        "gamma_obs",
                                        gamma_obs.mean(),
                                    )

                                    lb.add_value(
                                        "gamma_nce",
                                        gamma_nce.mean(),
                                    )

                                    lb.add_value(
                                        "gamma_obs_realized",
                                        gamma_obs_real.mean(),
                                    )

                                    lb.add_value(
                                        "gamma_nce_realized",
                                        gamma_nce_real.mean(),
                                    )

                                lb.add_value(
                                    "gamma_geom_realized",
                                    gamma_geom_real.mean(),
                                )

                # ====================================================
                # Combined A/P validated local certificates
                # ====================================================

                for (
                    i,
                    k_req,
                    j,
                ), by_view in combined_local.items():

                    # A view is admitted to the combined theorem only if the
                    # held-out LOO calibration condition validated.
                    valid_views = [
                        v
                        for v, z in by_view.items()
                        if z[
                            "valid"
                        ]
                    ]

                    valid_obs_positive = [
                        v
                        for v in valid_views
                        if by_view[v][
                            "q_obs"
                        ] > args.tol
                    ]

                    valid_nce_positive = [
                        v
                        for v in valid_views
                        if by_view[v][
                            "q_nce"
                        ] > args.tol
                    ]

                    q_obs_comb = max(
                        [
                            by_view[v][
                                "q_obs"
                            ]
                            for v in valid_views
                        ],
                        default=0.0,
                    )

                    q_nce_comb = max(
                        [
                            by_view[v][
                                "q_nce"
                            ]
                            for v in valid_views
                        ],
                        default=0.0,
                    )

                    # Because each member of valid_views is itself a validated
                    # lower bound on the same d_ij, the maximum remains valid.
                    if (
                        q_nce_comb
                        > q_obs_comb
                        + 10.0 * args.tol
                    ):
                        raise RuntimeError(
                            "Combined NCE certificate exceeds "
                            "combined candidate-specific certificate"
                        )

                    c = contraction[
                        i
                    ]

                    C_cons = c[
                        "C_cons"
                    ]

                    C_real = c[
                        "C_real"
                    ]

                    actual = joint_for(
                        i,
                        j,
                    )[
                        "actual"
                    ]

                    raw_obs_lb = (
                        float(
                            Lij[
                                i,
                                j,
                            ]
                        )
                        * q_obs_comb
                    )

                    raw_nce_lb = (
                        float(
                            Lij[
                                i,
                                j,
                            ]
                        )
                        * q_nce_comb
                    )

                    # --------------------------------------------
                    # Conservative certificates
                    # --------------------------------------------

                    gamma_obs_c = (
                        raw_obs_lb
                        - C_cons
                    )

                    gamma_nce_c = (
                        raw_nce_lb
                        - C_cons
                    )

                    cert_obs_c = (
                        (
                            gamma_obs_c
                            > args.tol
                        )
                        & bool(
                            valid_views
                        )
                        & (
                            q_obs_comb
                            > args.tol
                        )
                    )

                    cert_nce_c = (
                        (
                            gamma_nce_c
                            > args.tol
                        )
                        & bool(
                            valid_views
                        )
                        & (
                            q_nce_comb
                            > args.tol
                        )
                    )

                    # --------------------------------------------
                    # Realized-contraction certificates
                    # --------------------------------------------

                    gamma_obs_real_c = (
                        raw_obs_lb
                        - C_real
                    )

                    gamma_nce_real_c = (
                        raw_nce_lb
                        - C_real
                    )

                    cert_obs_real_c = (
                        (
                            gamma_obs_real_c
                            > args.tol
                        )
                        & bool(
                            valid_views
                        )
                        & (
                            q_obs_comb
                            > args.tol
                        )
                    )

                    cert_nce_real_c = (
                        (
                            gamma_nce_real_c
                            > args.tol
                        )
                        & bool(
                            valid_views
                        )
                        & (
                            q_nce_comb
                            > args.tol
                        )
                    )

                    def _vr(
                        mask: np.ndarray,
                    ) -> float:

                        den = int(
                            mask.sum()
                        )

                        if den == 0:
                            return float(
                                "nan"
                            )

                        return float(
                            (
                                mask
                                & (~actual)
                            ).sum()
                            / den
                        )

                    # --------------------------------------------
                    # Combined CSV
                    # --------------------------------------------

                    comb_w.writerow(
                        {
                            "bank": bank,
                            "batch": batch_idx,
                            "global_row": (
                                bank_global_row
                                + i
                            ),
                            "file_id": (
                                file_ids[i]
                            ),
                            "k_requested": (
                                k_req
                            ),
                            "candidate_col": (
                                j
                            ),
                            "candidate_file_id": (
                                file_ids[j]
                            ),

                            "valid_views_calibration": (
                                "+".join(
                                    valid_views
                                )
                            ),
                            "valid_views_obs_positive": (
                                "+".join(
                                    valid_obs_positive
                                )
                            ),
                            "valid_views_nce_positive": (
                                "+".join(
                                    valid_nce_positive
                                )
                            ),

                            "q_obs_combined": (
                                q_obs_comb
                            ),
                            "q_nce_combined": (
                                q_nce_comb
                            ),

                            "ed_target_negative": float(
                                D_raw[
                                    i,
                                    j,
                                ]
                            ),

                            "L_ij": float(
                                Lij[
                                    i,
                                    j,
                                ]
                            ),

                            "mc_samples": (
                                args.mc_samples
                            ),

                            "actual_order_rate": float(
                                actual.mean()
                            ),

                            "combined_obs_certificate_rate": float(
                                cert_obs_c.mean()
                            ),
                            "combined_nce_certificate_rate": float(
                                cert_nce_c.mean()
                            ),
                            "combined_obs_certificate_violation_rate": _vr(
                                cert_obs_c
                            ),
                            "combined_nce_certificate_violation_rate": _vr(
                                cert_nce_c
                            ),

                            "mean_gamma_obs_combined": float(
                                gamma_obs_c.mean()
                            ),
                            "mean_gamma_nce_combined": float(
                                gamma_nce_c.mean()
                            ),

                            "realized_combined_obs_certificate_rate": float(
                                cert_obs_real_c.mean()
                            ),
                            "realized_combined_nce_certificate_rate": float(
                                cert_nce_real_c.mean()
                            ),
                            "realized_combined_obs_certificate_violation_rate": _vr(
                                cert_obs_real_c
                            ),
                            "realized_combined_nce_certificate_violation_rate": _vr(
                                cert_nce_real_c
                            ),

                            "mean_gamma_obs_combined_realized": float(
                                gamma_obs_real_c.mean()
                            ),
                            "mean_gamma_nce_combined_realized": float(
                                gamma_nce_real_c.mean()
                            ),
                        }
                    )

                    # --------------------------------------------
                    # Aggregate combined statistics
                    # --------------------------------------------

                    cb = combined_bags.setdefault(
                        int(
                            k_req
                        ),
                        MetricBag(),
                    )

                    cb.add_count(
                        "combined_candidates",
                        1,
                    )

                    cb.add_count(
                        "combined_has_valid_view",
                        int(
                            bool(
                                valid_views
                            )
                        ),
                    )

                    cb.add_count(
                        "combined_obs_positive_q",
                        int(
                            bool(
                                valid_obs_positive
                            )
                        ),
                    )

                    cb.add_count(
                        "combined_nce_positive_q",
                        int(
                            bool(
                                valid_nce_positive
                            )
                        ),
                    )

                    cb.add_count(
                        "mc_trials",
                        args.mc_samples,
                    )

                    cb.add_count(
                        "actual_order_success",
                        int(
                            actual.sum()
                        ),
                    )

                    if valid_views:

                        cb.add_count(
                            "validated_mc_trials",
                            args.mc_samples,
                        )

                        cb.add_value(
                            "q_obs_combined",
                            q_obs_comb,
                        )

                        cb.add_value(
                            "q_nce_combined",
                            q_nce_comb,
                        )

                    # Conservative.
                    cb.add_count(
                        "obs_cert",
                        int(
                            cert_obs_c.sum()
                        ),
                    )

                    cb.add_count(
                        "nce_cert",
                        int(
                            cert_nce_c.sum()
                        ),
                    )

                    cb.add_count(
                        "obs_cert_violations",
                        int(
                            (
                                cert_obs_c
                                & (~actual)
                            ).sum()
                        ),
                    )

                    cb.add_count(
                        "nce_cert_violations",
                        int(
                            (
                                cert_nce_c
                                & (~actual)
                            ).sum()
                        ),
                    )

                    # Realized.
                    cb.add_count(
                        "realized_obs_cert",
                        int(
                            cert_obs_real_c.sum()
                        ),
                    )

                    cb.add_count(
                        "realized_nce_cert",
                        int(
                            cert_nce_real_c.sum()
                        ),
                    )

                    cb.add_count(
                        "realized_obs_cert_violations",
                        int(
                            (
                                cert_obs_real_c
                                & (~actual)
                            ).sum()
                        ),
                    )

                    cb.add_count(
                        "realized_nce_cert_violations",
                        int(
                            (
                                cert_nce_real_c
                                & (~actual)
                            ).sum()
                        ),
                    )

                    if valid_views:

                        cb.add_value(
                            "gamma_obs_combined",
                            gamma_obs_c.mean(),
                        )

                        cb.add_value(
                            "gamma_nce_combined",
                            gamma_nce_c.mean(),
                        )

                        cb.add_value(
                            "gamma_obs_combined_realized",
                            gamma_obs_real_c.mean(),
                        )

                        cb.add_value(
                            "gamma_nce_combined_realized",
                            gamma_nce_real_c.mean(),
                        )

                bank_global_row += B

                # Restore deterministic score state.
                model.decoder_student.eval()

                if hasattr(
                    model.decoder_student,
                    "set_self_attn_drop_p",
                ):
                    model.decoder_student.set_self_attn_drop_p(
                        0.0
                    )

    finally:

        row_f.close()
        loo_f.close()
        con_f.close()
        comb_f.close()

    # ============================================================
    # Paper-facing summary
    # ============================================================

    summary_rows: List[
        Dict[str, Any]
    ] = []

    all_keys = sorted(
        set(
            row_bags
        )
        | set(
            loo_bags
        ),
        key=lambda x: (
            x[0],
            x[1],
        ),
    )

    # --------------------------------------------------------
    # Per-view rows
    # --------------------------------------------------------

    for key in all_keys:

        view, k = key

        rsum = row_bags.get(
            key,
            MetricBag(),
        ).summarize()

        lsum = loo_bags.get(
            key,
            MetricBag(),
        ).summarize()

        row: Dict[
            str,
            Any,
        ] = {
            "view": view,
            "k_requested": k,
        }

        row.update(
            {
                f"row_{a}": b
                for a, b in rsum.items()
            }
        )

        row.update(
            {
                f"loo_{a}": b
                for a, b in lsum.items()
            }
        )

        # --------------------------------------------
        # Row-level NCE statistics
        # --------------------------------------------

        nr = rsum.get(
            "rows",
            0,
        )

        if nr:

            row[
                "row_nce_positive_margin_rate"
            ] = (
                rsum.get(
                    "nce_positive_margin_rows",
                    0,
                )
                / nr
            )

            row[
                "row_nce_margin_violation_rate"
            ] = (
                rsum.get(
                    "nce_margin_violations",
                    0,
                )
                / nr
            )

        # --------------------------------------------
        # LOO support / calibration statistics
        # --------------------------------------------

        nc = lsum.get(
            "loo_candidates",
            0,
        )

        if nc:

            row[
                "loo_obs_support_rate"
            ] = (
                lsum.get(
                    "obs_query_supported",
                    0,
                )
                / nc
            )

            row[
                "loo_nce_support_rate"
            ] = (
                lsum.get(
                    "nce_query_supported",
                    0,
                )
                / nc
            )

            row[
                "loo_calibration_valid_rate_all"
            ] = (
                lsum.get(
                    "calibration_valid",
                    0,
                )
                / nc
            )

            row[
                "loo_q_obs_positive_rate_all"
            ] = (
                lsum.get(
                    "q_obs_positive_candidates",
                    0,
                )
                / nc
            )

            row[
                "loo_q_nce_positive_rate_all"
            ] = (
                lsum.get(
                    "q_nce_positive_candidates",
                    0,
                )
                / nc
            )

        supported = lsum.get(
            "obs_query_supported",
            0,
        )

        if supported:

            row[
                "loo_calibration_coverage_rate_supported"
            ] = (
                lsum.get(
                    "calibration_valid",
                    0,
                )
                / supported
            )

        n_valid = lsum.get(
            "calibration_valid",
            0,
        )

        if n_valid:

            row[
                "loo_q_obs_positive_rate_given_valid"
            ] = (
                lsum.get(
                    "q_obs_positive_candidates",
                    0,
                )
                / n_valid
            )

            row[
                "loo_q_nce_positive_rate_given_valid"
            ] = (
                lsum.get(
                    "q_nce_positive_candidates",
                    0,
                )
                / n_valid
            )

        # --------------------------------------------
        # Direct empirical ordering
        # --------------------------------------------

        nmc = lsum.get(
            "mc_trials",
            0,
        )

        if nmc:

            row[
                "joint_actual_order_rate"
            ] = (
                lsum.get(
                    "actual_order_success",
                    0,
                )
                / nmc
            )

            row[
                "joint_geom_positive_rate"
            ] = (
                lsum.get(
                    "geom_positive",
                    0,
                )
                / nmc
            )

            row[
                "joint_formal_obs_certificate_rate_all"
            ] = (
                lsum.get(
                    "formal_obs_cert",
                    0,
                )
                / nmc
            )

            row[
                "joint_formal_nce_certificate_rate_all"
            ] = (
                lsum.get(
                    "formal_nce_cert",
                    0,
                )
                / nmc
            )

            row[
                "joint_realized_geom_positive_rate"
            ] = (
                lsum.get(
                    "realized_geom_positive",
                    0,
                )
                / nmc
            )

            row[
                "joint_realized_formal_obs_certificate_rate_all"
            ] = (
                lsum.get(
                    "realized_formal_obs_cert",
                    0,
                )
                / nmc
            )

            row[
                "joint_realized_formal_nce_certificate_rate_all"
            ] = (
                lsum.get(
                    "realized_formal_nce_cert",
                    0,
                )
                / nmc
            )

        # Certificate rate conditioned on a validated local envelope.
        n_valid_mc = lsum.get(
            "validated_mc_trials",
            0,
        )

        if n_valid_mc:

            row[
                "joint_formal_obs_certificate_rate_given_valid"
            ] = (
                lsum.get(
                    "formal_obs_cert",
                    0,
                )
                / n_valid_mc
            )

            row[
                "joint_formal_nce_certificate_rate_given_valid"
            ] = (
                lsum.get(
                    "formal_nce_cert",
                    0,
                )
                / n_valid_mc
            )

            row[
                "joint_realized_formal_obs_certificate_rate_given_valid"
            ] = (
                lsum.get(
                    "realized_formal_obs_cert",
                    0,
                )
                / n_valid_mc
            )

            row[
                "joint_realized_formal_nce_certificate_rate_given_valid"
            ] = (
                lsum.get(
                    "realized_formal_nce_cert",
                    0,
                )
                / n_valid_mc
            )

        # --------------------------------------------
        # Certificate violation rates
        # --------------------------------------------

        for (
            cert_name,
            viol_name,
            out_name,
        ) in [
            (
                "geom_positive",
                "geom_cert_violations",
                "joint_geom_certificate_violation_rate",
            ),
            (
                "formal_obs_cert",
                "formal_obs_cert_violations",
                "joint_formal_obs_certificate_violation_rate",
            ),
            (
                "formal_nce_cert",
                "formal_nce_cert_violations",
                "joint_formal_nce_certificate_violation_rate",
            ),
            (
                "realized_geom_positive",
                "realized_geom_cert_violations",
                "joint_realized_geom_certificate_violation_rate",
            ),
            (
                "realized_formal_obs_cert",
                "realized_formal_obs_cert_violations",
                "joint_realized_formal_obs_certificate_violation_rate",
            ),
            (
                "realized_formal_nce_cert",
                "realized_formal_nce_cert_violations",
                "joint_realized_formal_nce_certificate_violation_rate",
            ),
        ]:

            den = lsum.get(
                cert_name,
                0,
            )

            row[
                out_name
            ] = (
                lsum.get(
                    viol_name,
                    0,
                )
                / den
                if den
                else float(
                    "nan"
                )
            )

        summary_rows.append(
            row
        )

    # --------------------------------------------------------
    # Combined A/P theorem-facing rows
    # --------------------------------------------------------

    for k in sorted(
        combined_bags
    ):

        cs = combined_bags[
            k
        ].summarize()

        row: Dict[
            str,
            Any,
        ] = {
            "view": "COMBINED",
            "k_requested": k,
        }

        row.update(
            {
                f"combined_{a}": b
                for a, b in cs.items()
            }
        )

        nc = cs.get(
            "combined_candidates",
            0,
        )

        if nc:

            row[
                "combined_valid_view_rate"
            ] = (
                cs.get(
                    "combined_has_valid_view",
                    0,
                )
                / nc
            )

            row[
                "combined_obs_positive_q_rate"
            ] = (
                cs.get(
                    "combined_obs_positive_q",
                    0,
                )
                / nc
            )

            row[
                "combined_nce_positive_q_rate"
            ] = (
                cs.get(
                    "combined_nce_positive_q",
                    0,
                )
                / nc
            )

        nmc = cs.get(
            "mc_trials",
            0,
        )

        if nmc:

            row[
                "joint_actual_order_rate"
            ] = (
                cs.get(
                    "actual_order_success",
                    0,
                )
                / nmc
            )

            row[
                "joint_formal_obs_certificate_rate_all"
            ] = (
                cs.get(
                    "obs_cert",
                    0,
                )
                / nmc
            )

            row[
                "joint_formal_nce_certificate_rate_all"
            ] = (
                cs.get(
                    "nce_cert",
                    0,
                )
                / nmc
            )

            row[
                "joint_realized_formal_obs_certificate_rate_all"
            ] = (
                cs.get(
                    "realized_obs_cert",
                    0,
                )
                / nmc
            )

            row[
                "joint_realized_formal_nce_certificate_rate_all"
            ] = (
                cs.get(
                    "realized_nce_cert",
                    0,
                )
                / nmc
            )

        n_valid_mc = cs.get(
            "validated_mc_trials",
            0,
        )

        if n_valid_mc:

            row[
                "joint_formal_obs_certificate_rate_given_valid"
            ] = (
                cs.get(
                    "obs_cert",
                    0,
                )
                / n_valid_mc
            )

            row[
                "joint_formal_nce_certificate_rate_given_valid"
            ] = (
                cs.get(
                    "nce_cert",
                    0,
                )
                / n_valid_mc
            )

            row[
                "joint_realized_formal_obs_certificate_rate_given_valid"
            ] = (
                cs.get(
                    "realized_obs_cert",
                    0,
                )
                / n_valid_mc
            )

            row[
                "joint_realized_formal_nce_certificate_rate_given_valid"
            ] = (
                cs.get(
                    "realized_nce_cert",
                    0,
                )
                / n_valid_mc
            )

        # Violation rates.
        for (
            cert_name,
            viol_name,
            out_name,
        ) in [
            (
                "obs_cert",
                "obs_cert_violations",
                "joint_formal_obs_certificate_violation_rate",
            ),
            (
                "nce_cert",
                "nce_cert_violations",
                "joint_formal_nce_certificate_violation_rate",
            ),
            (
                "realized_obs_cert",
                "realized_obs_cert_violations",
                "joint_realized_formal_obs_certificate_violation_rate",
            ),
            (
                "realized_nce_cert",
                "realized_nce_cert_violations",
                "joint_realized_formal_nce_certificate_violation_rate",
            ),
        ]:

            den = cs.get(
                cert_name,
                0,
            )

            row[
                out_name
            ] = (
                cs.get(
                    viol_name,
                    0,
                )
                / den
                if den
                else float(
                    "nan"
                )
            )

        summary_rows.append(
            row
        )

    # --------------------------------------------------------
    # Positive-contraction summary
    # --------------------------------------------------------

    csum = (
        contraction_bag
        .summarize()
    )

    c_row: Dict[
        str,
        Any,
    ] = {
        "view": "POSITIVE_CONTRACTION",
        "k_requested": "",
    }

    c_row.update(
        {
            f"contraction_{k}": v
            for k, v in csum.items()
        }
    )

    tri_den = csum.get(
        "triangle_trials",
        0,
    )

    if tri_den:

        c_row[
            "contraction_triangle_violation_rate"
        ] = (
            csum.get(
                "triangle_violations",
                0,
            )
            / tri_den
        )

    summary_rows.append(
        c_row
    )

    # --------------------------------------------------------
    # Write union-schema summary
    # --------------------------------------------------------

    fields: List[
        str
    ] = []

    for row in summary_rows:

        for k in row:

            if k not in fields:
                fields.append(
                    k
                )

    with open(
        summary_path,
        "w",
        newline="",
    ) as f:

        w = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        w.writeheader()

        for row in summary_rows:

            w.writerow(
                row
            )

    # ============================================================
    # Final diagnostics
    # ============================================================

    print(
        "\n"
        + "=" * 96
    )

    print(
        "PAIRALIGN LOO MONOTONE LOCAL JOINT AUDIT COMPLETE"
    )

    print(
        "=" * 96
    )

    print(
        f"row monotonicity     : {row_path}"
    )

    print(
        f"LOO monotone audit   : {loo_path}"
    )

    print(
        f"contraction rows     : {contraction_path}"
    )

    print(
        f"combined A/P joint   : {combined_path}"
    )

    print(
        f"paper summary        : {summary_path}"
    )

    print(
        f"run configuration    : {config_out}"
    )

    print(
        "\nInterpretation:"
    )

    print(
        "  * row_spearman_* is descriptive only; no linear calibration is assumed."
    )

    print(
        "  * loo_obs_support_rate measures how often the held-out observed margin"
    )

    print(
        "    lies inside the non-extrapolated LOO envelope domain."
    )

    print(
        "  * loo_calibration_coverage_rate_supported measures how often the"
    )

    print(
        "    held-out candidate satisfies d_j >= kappa_hat^{-j}(delta_j)."
    )

    print(
        "  * q_obs_validated and q_nce_validated are used only after that"
    )

    print(
        "    held-out calibration event succeeds."
    )

    print(
        "  * certificate_rate_all uses every evaluated local comparison as"
    )

    print(
        "    denominator; certificate_rate_given_valid conditions on candidates"
    )

    print(
        "    with a validated LOO monotone envelope."
    )

    print(
        "  * conservative certificates use"
    )

    print(
        "        e_A + e_P + max(e_A,e_P)"
    )

    print(
        "  * realized certificates use"
    )

    print(
        "        ED(pred_A,pred_P) + max(e_A,e_P)"
    )

    print(
        "  * row_nce_margin_violation_rate should be 0."
    )

    print(
        "  * positively validated certificate violation rates should be 0"
    )

    print(
        "    up to numerical tolerance."
    )


# ============================================================
# CLI
# ============================================================


def parse_args():

    p = argparse.ArgumentParser(
        description=(
            "PairAlign held-out LOO monotone "
            "likelihood/edit + joint contraction/NCE audit"
        )
    )

    p.add_argument(
        "--root_dir",
        type=str,
        default=os.path.dirname(
            os.getcwd()
        ),
    )

    p.add_argument(
        "--config",
        type=str,
        default="mainFull.yaml",
    )

    p.add_argument(
        "--ckpt",
        type=str,
        required=True,
    )

    p.add_argument(
        "--out_dir",
        type=str,
        required=True,
    )

    p.add_argument(
        "--split",
        type=str,
        default="valid",
        choices=[
            "train",
            "valid",
            "test",
        ],
    )

    p.add_argument(
        "--device",
        type=int,
        default=0,
    )

    p.add_argument(
        "--batch_size",
        type=int,
        default=16,
    )

    p.add_argument(
        "--num_workers",
        type=int,
        default=None,
    )

    p.add_argument(
        "--no_add_noise",
        action="store_true",
        help=(
            "Disable standard content-preserving "
            "positive augmentation"
        ),
    )
    p.add_argument(
        "--use_time_compression",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Slightly compress only the positive view and right-pad it back to 3 seconds.",
    )

    p.add_argument(
        "--target_banks",
        type=int,
        default=3,
    )

    p.add_argument(
        "--k_values",
        type=int,
        nargs="+",
        default=[
            5,
            8,
            15,
        ],
    )

    p.add_argument(
        "--min_local_k",
        type=int,
        default=3,
        help=(
            "Minimum local neighborhood size. "
            "LOO then uses k-1 calibration candidates."
        ),
    )

    p.add_argument(
        "--tau",
        type=float,
        default=0.5,
    )

    p.add_argument(
        "--score_row_chunk",
        type=int,
        default=4,
        help=(
            "Rows of the BxB clean likelihood "
            "matrix scored at once; lower if GPU OOM"
        ),
    )

    p.add_argument(
        "--prefix_mask_frac",
        type=float,
        default=0.1,
    )

    p.add_argument(
        "--sa_drop_p",
        type=float,
        default=0.1,
    )

    p.add_argument(
        "--mc_samples",
        type=int,
        default=16,
    )

    p.add_argument(
        "--mc_chunk",
        type=int,
        default=4,
        help=(
            "Number of stochastic positive-fitting "
            "realizations vectorized per decoder forward"
        ),
    )

    p.add_argument(
        "--max_len",
        type=int,
        default=60,
    )

    p.add_argument(
        "--length_cap_ratio",
        type=float,
        default=0.15,
    )

    p.add_argument(
        "--min_cap",
        type=int,
        default=2,
    )

    p.add_argument(
        "--eval_seed",
        type=int,
        default=13,
    )

    p.add_argument(
        "--dataloader_seed",
        type=int,
        default=1234,
    )

    p.add_argument(
        "--tol",
        type=float,
        default=1e-7,
    )

    p.add_argument(
        "--max_batches",
        type=int,
        default=None,
        help="Debug only",
    )

    return p.parse_args()


if __name__ == "__main__":

    run_audit(
        parse_args()
    )