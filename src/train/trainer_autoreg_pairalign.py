import copy
import math
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio.transforms as T
import lightning as L
from torch.optim import lr_scheduler

from models import TextDecoder
from train.trainer_autoreg_prior import AudioTokenizerARPrior


# -----------------------------------------------------------------------------
# Token helpers
# -----------------------------------------------------------------------------


def _clean_codes(
    codes: torch.Tensor,
    pad_idx: int,
    bos_idx: int,
    eos_idx: int,
    mask_idx: Optional[int],
) -> List[torch.Tensor]:
    """Remove decoder special symbols from a padded batch."""
    keep = (
        codes.ne(pad_idx)
        & codes.ne(bos_idx)
        & codes.ne(eos_idx)
    )
    if mask_idx is not None:
        keep &= codes.ne(mask_idx)
    return [seq[keep[i]] for i, seq in enumerate(codes)]


def _assert_eos_exists(
    targets: torch.Tensor,
    eos_idx: int,
) -> None:
    if not targets.eq(eos_idx).any(dim=1).all():
        raise RuntimeError("Every teacher target must contain EOS.")


# -----------------------------------------------------------------------------
# Prefix corruption
# -----------------------------------------------------------------------------


def _token_dropout_with_forced_prefix(
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
    Mask approximately ``target_frac`` of content tokens.

    A small random number of earliest valid tokens is masked first. Any
    remaining mask budget is then filled from the other valid positions.
    BOS/EOS/PAD are never masked and at least ``min_keep`` content tokens remain.
    """
    if not 0.0 <= target_frac <= 1.0:
        raise ValueError(f"target_frac must be in [0, 1], got {target_frac}")

    batch_size, seq_len = x.shape
    out = x.clone()

    valid = x.ne(pad_idx) & x.ne(bos_idx) & x.ne(eos_idx)
    valid_count = valid.sum(dim=1)

    drop_count = torch.round(valid_count.float() * target_frac).long()
    drop_count = torch.minimum(
        drop_count,
        (valid_count - min_keep).clamp_min(0),
    )

    if drop_count.max().item() <= 0:
        return out

    if max_force_k > 0:
        force_k = torch.randint(
            1,
            max_force_k + 1,
            (batch_size,),
            device=x.device,
        )
    else:
        force_k = torch.zeros(
            batch_size,
            dtype=torch.long,
            device=x.device,
        )

    valid_rank = valid.cumsum(dim=1)
    forced_mask = valid & (valid_rank <= force_k.unsqueeze(1))
    forced_rank = forced_mask.cumsum(dim=1)
    forced_mask &= forced_rank <= drop_count.unsqueeze(1)

    drop_mask = forced_mask.clone()
    remaining = drop_count - forced_mask.sum(dim=1)

    if remaining.max().item() > 0:
        candidates = valid & ~drop_mask
        scores = torch.rand((batch_size, seq_len), device=x.device)
        scores.masked_fill_(~candidates, float("inf"))
        order = scores.argsort(dim=1)

        rank = torch.arange(
            seq_len,
            device=x.device,
        ).unsqueeze(0).expand(batch_size, seq_len)
        take = rank < remaining.unsqueeze(1)

        extra_mask = torch.zeros(
            (batch_size, seq_len),
            dtype=torch.bool,
            device=x.device,
        )
        extra_mask.scatter_(1, order, take)
        drop_mask |= extra_mask

    out[drop_mask] = mask_idx
    return out


# -----------------------------------------------------------------------------
# Shared Stage-II / Stage-III repetition control
# -----------------------------------------------------------------------------


def _prefix_token_counts(
    prefix: torch.Tensor,
    vocab_size: int,
    excluded_token_ids: Optional[Sequence[int]] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """
    Count token occurrences in each decoder prefix.

    Tokens listed in ``excluded_token_ids`` are assigned an effective count
    of zero.
    """
    if prefix.dim() != 2:
        raise ValueError(
            f"prefix must have shape (B, L), got {tuple(prefix.shape)}"
        )

    if dtype is None:
        dtype = torch.float32

    counts = torch.zeros(
        (prefix.size(0), vocab_size),
        dtype=dtype,
        device=prefix.device,
    )
    counts.scatter_add_(
        1,
        prefix,
        torch.ones_like(prefix, dtype=dtype),
    )

    if excluded_token_ids:
        valid_ids = [
            int(token_id)
            for token_id in excluded_token_ids
            if token_id is not None
            and 0 <= int(token_id) < vocab_size
        ]
        if valid_ids:
            counts[:, valid_ids] = 0.0

    return counts


def _apply_logit_repetition_penalty(
    logits: torch.Tensor,
    prefix: torch.Tensor,
    rep_gamma: float,
    excluded_token_ids: Optional[Sequence[int]] = None,
) -> torch.Tensor:
    r"""
    Apply the count-compounded repetition penalty directly to logits:

        adjusted_logit(v) = logit(v) - c(v) * log(rep_gamma)

    This is the same repetition rule used by the Stage-II AR prior.
    BOS/EOS/PAD/MASK can be excluded by zeroing their effective counts.
    """
    if logits.dim() != 2:
        raise ValueError(
            f"logits must have shape (B, V), got {tuple(logits.shape)}"
        )
    if prefix.dim() != 2:
        raise ValueError(
            f"prefix must have shape (B, L), got {tuple(prefix.shape)}"
        )
    if logits.size(0) != prefix.size(0):
        raise ValueError("logits and prefix must have the same batch size")
    if rep_gamma < 1.0:
        raise ValueError(f"rep_gamma must be >= 1.0, got {rep_gamma}")
    if rep_gamma == 1.0:
        return logits

    counts = _prefix_token_counts(
        prefix=prefix,
        vocab_size=logits.size(-1),
        excluded_token_ids=excluded_token_ids,
        dtype=logits.dtype,
    )
    log_gamma = logits.new_tensor(rep_gamma).log()
    return logits - counts * log_gamma


# -----------------------------------------------------------------------------
# Stage-III teacher top-p target generation
# -----------------------------------------------------------------------------


def _phase_value(
    step: int,
    early_steps: int,
    early_value: float,
    late_value: float,
) -> float:
    """Two-phase early/late schedule."""
    return float(early_value if step <= early_steps else late_value)


@torch.no_grad()
def _sample_top_p(
    adjusted_logits: torch.Tensor,
    top_p: float,
    temperature: float,
) -> torch.Tensor:
    """
    Sample from the smallest top-p nucleus.

    The nucleus is defined from the repetition-adjusted logits. Temperature is
    applied only after non-nucleus logits are masked.
    """
    if not 0.0 < top_p <= 1.0:
        raise ValueError(f"top_p must be in (0, 1], got {top_p}")
    if temperature <= 0.0:
        raise ValueError(f"temperature must be > 0, got {temperature}")

    nucleus_probs = F.softmax(adjusted_logits, dim=-1)
    sorted_probs, sorted_idx = torch.sort(
        nucleus_probs,
        dim=-1,
        descending=True,
    )
    cdf = torch.cumsum(sorted_probs, dim=-1)

    # Keep the token that crosses the threshold, so this is the smallest
    # nucleus whose cumulative mass reaches/exceeds top_p.
    sorted_remove = cdf > top_p
    sorted_remove[..., 1:] = sorted_remove[..., :-1].clone()
    sorted_remove[..., 0] = False
    keep_sorted = ~sorted_remove

    keep_vocab = torch.zeros_like(keep_sorted, dtype=torch.bool)
    keep_vocab.scatter_(1, sorted_idx, keep_sorted)

    filtered_logits = adjusted_logits.masked_fill(
        ~keep_vocab,
        -torch.inf,
    )
    sample_probs = F.softmax(
        filtered_logits / temperature,
        dim=-1,
    )
    return torch.multinomial(sample_probs, num_samples=1).squeeze(1)


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
    temperature: float = 1.0,
    length_cap_ratio: float = 0.15,
    min_cap: int = 2,
    rep_gamma: float = 1.3,
    explore_steps: int = 4,
    top_p_start: float = 0.97,
    temperature_start: float = 1.30,
    rep_gamma_start: float = 1.03,
) -> torch.Tensor:
    """
    Generate Stage-III EMA-teacher targets with stochastic top-p decoding.

    Repetition control is identical to Stage II:

        adjusted_logit(v) = logit(v) - c(v) * log(rep_gamma)

    Early decoding uses the exploratory settings. Later decoding uses
    ``top_p``, ``temperature``, and ``rep_gamma``.

    EOS may occur naturally before the cap and is forced at the cap.

    Returns:
        (B, 1, max_len), containing BOS ... EOS followed by PAD.
    """
    if xa.dim() != 3:
        raise ValueError(f"xa must be (B, T, D), got {tuple(xa.shape)}")
    if cond_lens.dim() != 1 or cond_lens.numel() != xa.size(0):
        raise ValueError("cond_lens must contain one length per sample")
    if max_len < 2:
        raise ValueError("max_len must be >= 2")
    if min_cap < 1:
        raise ValueError("min_cap must be >= 1")
    if length_cap_ratio <= 0.0:
        raise ValueError("length_cap_ratio must be > 0")

    device = xa.device
    batch_size = xa.size(0)

    cap = (length_cap_ratio * cond_lens.float()).long()
    cap = torch.clamp(cap, min=min_cap, max=max_len - 1)

    seq = torch.full(
        (batch_size, 1),
        bos_idx,
        dtype=torch.long,
        device=device,
    )
    finished = torch.zeros(
        batch_size,
        dtype=torch.bool,
        device=device,
    )

    excluded = [bos_idx, eos_idx, pad_idx]
    if mask_idx is not None:
        excluded.append(mask_idx)

    for step in range(1, max_len):
        logits = decoder(x=seq, xa=xa)[:, -1, :].clone()

        top_p_now = _phase_value(
            step,
            explore_steps,
            top_p_start,
            top_p,
        )
        temperature_now = _phase_value(
            step,
            explore_steps,
            temperature_start,
            temperature,
        )
        rep_gamma_now = _phase_value(
            step,
            explore_steps,
            rep_gamma_start,
            rep_gamma,
        )

        adjusted_logits = _apply_logit_repetition_penalty(
            logits=logits,
            prefix=seq,
            rep_gamma=rep_gamma_now,
            excluded_token_ids=excluded,
        )

        # PAD is unavailable until a row has finished.
        adjusted_logits[~finished, pad_idx] = -torch.inf

        # Force EOS at the sample-specific cap.
        must_eos = (~finished) & (step >= cap)
        if must_eos.any():
            adjusted_logits[must_eos] = -torch.inf
            adjusted_logits[must_eos, eos_idx] = 0.0

        # Finished rows emit PAD only.
        if finished.any():
            adjusted_logits[finished] = -torch.inf
            adjusted_logits[finished, pad_idx] = 0.0

        next_token = _sample_top_p(
            adjusted_logits=adjusted_logits,
            top_p=top_p_now,
            temperature=temperature_now,
        )

        seq = torch.cat((seq, next_token.unsqueeze(1)), dim=1)
        finished |= next_token.eq(eos_idx)

        if finished.all():
            break

    if seq.size(1) < max_len:
        pad = torch.full(
            (batch_size, max_len - seq.size(1)),
            pad_idx,
            dtype=torch.long,
            device=device,
        )
        seq = torch.cat((seq, pad), dim=1)
    else:
        seq = seq[:, :max_len]

    for row in range(batch_size):
        eos_pos = seq[row].eq(eos_idx).nonzero(as_tuple=True)[0]
        if eos_pos.numel() > 0:
            end = int(eos_pos[0].item()) + 1
            if end < max_len:
                seq[row, end:] = pad_idx

    return seq.unsqueeze(1)


# -----------------------------------------------------------------------------
# Evaluation beam search
# -----------------------------------------------------------------------------


def _beam_rank(
    score: float,
    decoded_length: int,
    alpha_len: float,
) -> float:
    """Optional length-normalized beam ranking."""
    if alpha_len <= 0.0:
        return score
    return score / (max(1, decoded_length) ** alpha_len)


@torch.no_grad()
def beam_search_teacher_targets_single(
    decoder,
    xa: torch.Tensor,
    cond_lens: torch.Tensor,
    bos_idx: int,
    eos_idx: int,
    pad_idx: int,
    mask_idx: Optional[int] = None,
    max_len: int = 60,
    beam_size: int = 4,
    length_cap_ratio: float = 0.15,
    min_cap: int = 2,
    rep_gamma: float = 1.3,
    length_norm_alpha: float = 0.0,
) -> torch.Tensor:
    """
    Deterministic beam search using the same repetition rule as Stage II and
    Stage-III teacher top-p decoding:

        adjusted_logit(v) = logit(v) - c(v) * log(rep_gamma)

    EOS may occur naturally before the cap and is forced at the cap.
    Raw beam scores remain cumulative log-probabilities; optional length
    normalization affects ranking only.

    Returns:
        (B, K, max_len), with beam 0 arranged as the selected hypothesis.
    """
    if xa.dim() != 3:
        raise ValueError(f"xa must be (B, T, D), got {tuple(xa.shape)}")
    if cond_lens.dim() != 1 or cond_lens.numel() != xa.size(0):
        raise ValueError("cond_lens must contain one length per sample")
    if beam_size < 1:
        raise ValueError("beam_size must be >= 1")
    if max_len < 2:
        raise ValueError("max_len must be >= 2")
    if min_cap < 1:
        raise ValueError("min_cap must be >= 1")
    if length_cap_ratio <= 0.0:
        raise ValueError("length_cap_ratio must be > 0")
    if rep_gamma < 1.0:
        raise ValueError("rep_gamma must be >= 1.0")
    if length_norm_alpha < 0.0:
        raise ValueError("length_norm_alpha must be >= 0")

    device = xa.device
    batch_size = xa.size(0)

    excluded = [bos_idx, eos_idx, pad_idx]
    if mask_idx is not None:
        excluded.append(mask_idx)

    all_beams: List[
        List[Tuple[List[int], float, bool]]
    ] = []

    for row in range(batch_size):
        cond_len = int(cond_lens[row].item())
        if cond_len < 1:
            raise ValueError("Every conditioning length must be >= 1")

        cap = int(math.floor(length_cap_ratio * cond_len))
        cap = max(min_cap, cap)
        cap = min(max_len - 1, cap)

        xa_row = xa[row : row + 1, :cond_len]

        # (tokens including BOS, raw cumulative log-probability, finished)
        beams: List[Tuple[List[int], float, bool]] = [
            ([int(bos_idx)], 0.0, False)
        ]

        for step in range(1, cap + 1):
            candidates: List[Tuple[List[int], float, bool]] = []

            for tokens, raw_score, finished in beams:
                if finished:
                    candidates.append((tokens, raw_score, True))
                    continue

                prefix = torch.tensor(
                    tokens,
                    dtype=torch.long,
                    device=device,
                ).unsqueeze(0)

                logits = decoder(
                    x=prefix,
                    xa=xa_row,
                )[:, -1, :]

                adjusted_logits = _apply_logit_repetition_penalty(
                    logits=logits,
                    prefix=prefix,
                    rep_gamma=rep_gamma,
                    excluded_token_ids=excluded,
                )

                adjusted_logits[:, pad_idx] = -torch.inf

                if step == cap:
                    forced = torch.full_like(
                        adjusted_logits,
                        -torch.inf,
                    )
                    forced[:, eos_idx] = adjusted_logits[:, eos_idx]
                    adjusted_logits = forced

                log_probs = F.log_softmax(
                    adjusted_logits,
                    dim=-1,
                ).squeeze(0)

                local_k = min(beam_size, log_probs.numel())
                token_scores, token_ids = torch.topk(
                    log_probs,
                    k=local_k,
                )

                for token_score, token_id in zip(
                    token_scores.tolist(),
                    token_ids.tolist(),
                ):
                    token_id = int(token_id)
                    candidates.append(
                        (
                            tokens + [token_id],
                            raw_score + float(token_score),
                            token_id == int(eos_idx),
                        )
                    )

            if not candidates:
                break

            candidates.sort(
                key=lambda item: _beam_rank(
                    item[1],
                    decoded_length=len(item[0]) - 1,
                    alpha_len=length_norm_alpha,
                ),
                reverse=True,
            )
            beams = candidates[:beam_size]

            if all(finished for _, _, finished in beams):
                break

        completed = [beam for beam in beams if beam[2]]
        selection_pool = completed if completed else beams
        selection_pool = sorted(
            selection_pool,
            key=lambda item: _beam_rank(
                item[1],
                decoded_length=len(item[0]) - 1,
                alpha_len=length_norm_alpha,
            ),
            reverse=True,
        )
        selected = selection_pool[0]

        selected_index = next(
            idx for idx, beam in enumerate(beams) if beam is selected
        )
        remaining = beams[:selected_index] + beams[selected_index + 1 :]
        remaining.sort(
            key=lambda item: _beam_rank(
                item[1],
                decoded_length=len(item[0]) - 1,
                alpha_len=length_norm_alpha,
            ),
            reverse=True,
        )

        ranked = [selected] + remaining
        while len(ranked) < beam_size:
            ranked.append(ranked[-1])

        all_beams.append(ranked[:beam_size])

    output = torch.full(
        (batch_size, beam_size, max_len),
        pad_idx,
        dtype=torch.long,
        device=device,
    )

    for row, beams in enumerate(all_beams):
        for beam_idx, (tokens, _, _) in enumerate(beams):
            tokens = tokens[:max_len]
            output[row, beam_idx, : len(tokens)] = torch.tensor(
                tokens,
                dtype=torch.long,
                device=device,
            )

    return output


# -----------------------------------------------------------------------------
# Training losses
# -----------------------------------------------------------------------------


def _per_sequence_mean_nll(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pad_idx: int,
) -> torch.Tensor:
    """Mean NLL per sequence over non-PAD positions; PAD is ignored in CE."""
    batch_size, seq_len, vocab_size = logits.shape

    nll = F.cross_entropy(
        logits.reshape(-1, vocab_size),
        targets.reshape(-1),
        reduction="none",
        ignore_index=pad_idx,
        label_smoothing=0.0,
    ).view(batch_size, seq_len)

    mask = targets.ne(pad_idx).float()
    denom = mask.sum(dim=1).clamp_min(1.0)
    return (nll * mask).sum(dim=1) / denom


def _build_likelihood_matrix(
    decoder,
    reps: torch.Tensor,
    tokens: torch.Tensor,
    pad_idx: int,
) -> torch.Tensor:
    """
    Build the clean pairwise conditional log-likelihood matrix.

    ``S[i, j]`` is the mean log-likelihood of canonical token sequence ``j``
    conditioned on representation ``i``.

    The decoder input is always the untouched teacher-forced prefix
    ``tokens[:, :-1]``. No prefix corruption or masking is performed here.
    Decoder self-attention dropout is controlled by the caller.
    """
    batch_size, time_steps, dim = reps.shape
    if tokens.size(0) != batch_size:
        raise ValueError(
            "Representations and tokens must have the same batch size"
        )

    tokens = tokens.to(reps.device)
    token_len = tokens.size(1)

    decoder_input = tokens[:, :-1]
    targets = tokens[:, 1:]

    input_big = (
        decoder_input.unsqueeze(0)
        .expand(batch_size, batch_size, token_len - 1)
        .reshape(batch_size * batch_size, token_len - 1)
    )
    target_big = (
        targets.unsqueeze(0)
        .expand(batch_size, batch_size, token_len - 1)
        .reshape(batch_size * batch_size, token_len - 1)
    )
    reps_big = (
        reps.unsqueeze(1)
        .expand(batch_size, batch_size, time_steps, dim)
        .reshape(batch_size * batch_size, time_steps, dim)
    )

    logits = decoder(x=input_big, xa=reps_big)
    nll = _per_sequence_mean_nll(
        logits,
        target_big,
        pad_idx=pad_idx,
    )
    return (-nll).view(batch_size, batch_size)


def _sequence_mean_ce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pad_idx: int,
) -> torch.Tensor:
    """Cross-entropy with equal sequence weighting."""
    batch_size, seq_len, vocab_size = logits.shape

    ce = F.cross_entropy(
        logits.reshape(-1, vocab_size),
        targets.reshape(-1),
        reduction="none",
        ignore_index=pad_idx,
    ).view(batch_size, seq_len)

    mask = targets.ne(pad_idx).float()
    denom = mask.sum(dim=1).clamp_min(1.0)
    return ((ce * mask).sum(dim=1) / denom).mean()


def _entropy_regularizer(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pad_idx: int,
    eos_idx: int,
    beta: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Encourage per-step entropy over non-PAD, non-EOS target positions."""
    log_probs = F.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    entropy = -(probs * log_probs).sum(dim=-1)

    mask = targets.ne(pad_idx) & targets.ne(eos_idx)
    denom = mask.sum().clamp_min(1)
    mean_entropy = (entropy * mask).sum() / denom
    loss = -beta * mean_entropy
    return loss, mean_entropy.detach()


def _infonce(
    scores: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Row-wise diagonal InfoNCE over a square score matrix."""
    labels = torch.arange(scores.size(0), device=scores.device)
    return F.cross_entropy(scores / temperature, labels)


def _linear_schedule(
    start: float,
    end: float,
    progress: float,
) -> float:
    progress = float(max(0.0, min(1.0, progress)))
    return float(start + progress * (end - start))


# -----------------------------------------------------------------------------
# Stage-III PairAlign
# -----------------------------------------------------------------------------


class AudioTokenizerAR_FINAL(L.LightningModule):
    """Stage-III PairAlign self-alignment model."""

    def __init__(self, config):
        super().__init__()
        self.config = config

        checkpoints = self.config.get("checkpoints", {})
        stage2_checkpoint = (
            checkpoints.get("stage2")
            or checkpoints.get("ar_prior")
        )
        if not stage2_checkpoint:
            raise ValueError(
                "AudioTokenizerAR_FINAL requires checkpoints.stage2 "
                "(or checkpoints.ar_prior)."
            )

        # Config-driven checkpoint loading; no machine- or user-specific paths.
        prior = AudioTokenizerARPrior.load_from_checkpoint(
            stage2_checkpoint,
            map_location="gpu" if torch.cuda.is_available() else "cpu",
            strict=False,
        )
        prior.eval()

        self.encoder_student = copy.deepcopy(prior.encoder)
        self.encoder_teacher = copy.deepcopy(prior.encoder)

        self.decoder_student = TextDecoder(
            n_vocab=self.config["num_codewords"],
            n_ctx=512,
            n_state=512,
            n_head=8,
            n_layer=6,
            n_cond_positions=900,
        )
        self.decoder_student.load_state_dict(prior.decoder.state_dict())

        self.decoder_teacher = copy.deepcopy(self.decoder_student)
        del prior

        self.encoder_teacher.eval()
        self.decoder_teacher.eval()
        self.decoder_teacher.set_self_attn_drop_p(0.0)

        for parameter in self.encoder_teacher.parameters():
            parameter.requires_grad = False
        for parameter in self.decoder_teacher.parameters():
            parameter.requires_grad = False
        for parameter in self.encoder_student.parameters():
            parameter.requires_grad = True
        for parameter in self.decoder_student.parameters():
            parameter.requires_grad = True

        # EMA teacher.
        self.teacher_decay = float(
            self.config.get("teacher_decay", 0.995)
        )
        self.ema_update_every = int(
            self.config.get("ema_update_every", 500)
        )

        # Prefix-corruption / self-attention-drop schedule.
        self.steps_per_epoch = int(
            self.config.get("steps_per_epoch", 15792)
        )
        self.warmup_epochs = int(
            self.config.get("corruption_warmup_epochs", 0)
        )
        self.anneal_epochs = int(
            self.config.get("corruption_anneal_epochs", 8)
        )

        self.target_frac_start = float(
            self.config.get("target_frac_start", 0.30)
        )
        self.target_frac_end = float(
            self.config.get("target_frac_end", 0.10)
        )
        self.sa_drop_start = float(
            self.config.get("sa_drop_start", 0.20)
        )
        self.sa_drop_end = float(
            self.config.get("sa_drop_end", 0.10)
        )

        self.target_frac = self.target_frac_start
        self.sa_drop_p = self.sa_drop_start
        self.decoder_student.set_self_attn_drop_p(self.sa_drop_p)

        # Loss settings.
        self.nce_temperature = float(
            self.config.get("nce_temperature", 0.5)
        )
        self.entropy_beta = float(
            self.config.get("entropy_beta", 1e-3)
        )

        # EMA-teacher sampling settings.
        self.teacher_top_p = float(
            self.config.get("teacher_top_p", 0.88)
        )
        self.teacher_temperature = float(
            self.config.get("teacher_temperature", 1.0)
        )
        self.teacher_rep_gamma = float(
            self.config.get("teacher_rep_gamma", 1.3)
        )
        self.teacher_explore_steps = int(
            self.config.get("teacher_explore_steps", 4)
        )
        self.teacher_top_p_start = float(
            self.config.get("teacher_top_p_start", 0.97)
        )
        self.teacher_temperature_start = float(
            self.config.get("teacher_temperature_start", 1.30)
        )
        self.teacher_rep_gamma_start = float(
            self.config.get("teacher_rep_gamma_start", 1.03)
        )

        self.optim = self.config["optimizer"]
        self.lr = float(self.config["lr"])
        self.wt_decay = float(self.config["wt_decay"])
        self.use_lr_scheduler = bool(self.config["lr_scheduler"])

        self.val_codes_cache: List[List[int]] = []

        # Guard EMA against repeated train-batch hooks observing the same
        # optimizer step under gradient accumulation.
        self._last_ema_update_step = -1

        self._validate_hyperparameters()
        self.save_hyperparameters()

    def _validate_hyperparameters(self) -> None:
        """Fail early on invalid training and decoding settings."""
        if not 0.0 < self.teacher_top_p <= 1.0:
            raise ValueError("teacher_top_p must be in (0, 1]")
        if not 0.0 < self.teacher_top_p_start <= 1.0:
            raise ValueError("teacher_top_p_start must be in (0, 1]")
        if self.teacher_temperature <= 0.0:
            raise ValueError("teacher_temperature must be > 0")
        if self.teacher_temperature_start <= 0.0:
            raise ValueError("teacher_temperature_start must be > 0")
        if self.teacher_rep_gamma < 1.0:
            raise ValueError("teacher_rep_gamma must be >= 1.0")
        if self.teacher_rep_gamma_start < 1.0:
            raise ValueError("teacher_rep_gamma_start must be >= 1.0")
        if not 0.0 <= self.teacher_decay < 1.0:
            raise ValueError("teacher_decay must be in [0, 1)")
        if self.ema_update_every <= 0:
            raise ValueError("ema_update_every must be > 0")
        if self.steps_per_epoch <= 0:
            raise ValueError("steps_per_epoch must be > 0")
        if self.warmup_epochs < 0 or self.anneal_epochs < 0:
            raise ValueError("corruption schedule epochs must be >= 0")
        if self.nce_temperature <= 0.0:
            raise ValueError("nce_temperature must be > 0")
        if self.entropy_beta < 0.0:
            raise ValueError("entropy_beta must be >= 0")
        if not 0.0 <= self.target_frac_start <= 1.0:
            raise ValueError("target_frac_start must be in [0, 1]")
        if not 0.0 <= self.target_frac_end <= 1.0:
            raise ValueError("target_frac_end must be in [0, 1]")
        if not 0.0 <= self.sa_drop_start <= 1.0:
            raise ValueError("sa_drop_start must be in [0, 1]")
        if not 0.0 <= self.sa_drop_end <= 1.0:
            raise ValueError("sa_drop_end must be in [0, 1]")

    def _generate_training_targets(
        self,
        z_ctx_anc: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, cond_len, _ = z_ctx_anc.shape
        cond_lens = torch.full(
            (batch_size,),
            cond_len,
            dtype=torch.long,
            device=z_ctx_anc.device,
        )

        targets = top_p_teacher_targets_single(
            decoder=self.decoder_teacher,
            xa=z_ctx_anc,
            cond_lens=cond_lens,
            bos_idx=self.decoder_teacher.BOS_IDX,
            eos_idx=self.decoder_teacher.EOS_IDX,
            pad_idx=self.decoder_teacher.PAD_IDX,
            mask_idx=getattr(self.decoder_teacher, "MASK_IDX", None),
            max_len=60,
            top_p=self.teacher_top_p,
            temperature=self.teacher_temperature,
            rep_gamma=self.teacher_rep_gamma,
            explore_steps=self.teacher_explore_steps,
            top_p_start=self.teacher_top_p_start,
            temperature_start=self.teacher_temperature_start,
            rep_gamma_start=self.teacher_rep_gamma_start,
        )
        return targets[:, 0, :]

    @torch.no_grad()
    def generate_teacher_beam_tokens(
        self,
        z_ctx: torch.Tensor,
        max_len: int = 60,
        beam_size: int = 4,
        length_cap_ratio: float = 0.15,
        min_cap: int = 2,
        rep_gamma: float = 1.3,
        length_norm_alpha: float = 0.0,
    ) -> torch.Tensor:
        """
        Deterministically decode Stage-III teacher tokens from encoder states.

        This is the public evaluation/artifact API. It uses the same beam search
        and logit-space repetition penalty as ``tokenize``.
        """
        self.decoder_teacher.eval()
        self.decoder_teacher.set_self_attn_drop_p(0.0)

        batch_size, cond_len, _ = z_ctx.shape
        cond_lens = torch.full(
            (batch_size,),
            cond_len,
            dtype=torch.long,
            device=z_ctx.device,
        )

        beams = beam_search_teacher_targets_single(
            decoder=self.decoder_teacher,
            xa=z_ctx,
            cond_lens=cond_lens,
            bos_idx=self.decoder_teacher.BOS_IDX,
            eos_idx=self.decoder_teacher.EOS_IDX,
            pad_idx=self.decoder_teacher.PAD_IDX,
            mask_idx=getattr(self.decoder_teacher, "MASK_IDX", None),
            max_len=max_len,
            beam_size=beam_size,
            length_cap_ratio=length_cap_ratio,
            min_cap=min_cap,
            rep_gamma=rep_gamma,
            length_norm_alpha=length_norm_alpha,
        )
        return beams[:, 0, :]

    def _compute_clean_likelihood_scores(
        self,
        z_ctx_anc: torch.Tensor,
        z_ctx_pos: torch.Tensor,
        canonical_tokens: torch.Tensor,
        pad_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the two clean B x B likelihood matrices used by InfoNCE.

        This scoring path deliberately removes the stochastic regularizers used
        by the ordinary autoregressive training forward:

          * prefix corruption is disabled by passing the untouched canonical
            teacher tokens directly to ``_build_likelihood_matrix``;
          * decoder self-attention dropout is temporarily set to 0.0.

        The current scheduled self-attention dropout is restored immediately
        after both matrices are computed, including when scoring raises.
        """
        self.decoder_student.set_self_attn_drop_p(0.0)
        try:
            score_anc = _build_likelihood_matrix(
                decoder=self.decoder_student,
                reps=z_ctx_anc,
                tokens=canonical_tokens,
                pad_idx=pad_idx,
            )
            score_pos = _build_likelihood_matrix(
                decoder=self.decoder_student,
                reps=z_ctx_pos,
                tokens=canonical_tokens,
                pad_idx=pad_idx,
            )
        finally:
            self.decoder_student.set_self_attn_drop_p(self.sa_drop_p)

        return score_anc, score_pos

    def _shared_step(self, batch, mode: str):
        if mode not in {"train", "valid"}:
            raise ValueError(f"Unsupported mode: {mode}")

        anc = batch[0].squeeze(1).permute(0, 2, 1)
        pos = batch[1].squeeze(1).permute(0, 2, 1)

        bos_idx = self.decoder_student.BOS_IDX
        eos_idx = self.decoder_student.EOS_IDX
        pad_idx = self.decoder_student.PAD_IDX
        mask_idx = self.decoder_student.MASK_IDX

        log_values: Dict[str, float] = {}

        self.encoder_teacher.eval()
        self.decoder_teacher.eval()
        self.decoder_teacher.set_self_attn_drop_p(0.0)

        if mode == "train":
            self.encoder_student.train()
            self.decoder_student.train()

            with torch.no_grad():
                z_teacher_anc = self.encoder_teacher(anc)
                indices_anc = self._generate_training_targets(z_teacher_anc)

            z_ctx_anc = self.encoder_student(anc)
            z_ctx_pos = self.encoder_student(pos)

            decoder_input = indices_anc[:, :-1]
            target = indices_anc[:, 1:]

            decoder_input = _token_dropout_with_forced_prefix(
                decoder_input,
                pad_idx=pad_idx,
                bos_idx=bos_idx,
                eos_idx=eos_idx,
                mask_idx=mask_idx,
                target_frac=self.target_frac,
                min_keep=2,
            )

            content_positions = (
                indices_anc[:, :-1].ne(pad_idx)
                & indices_anc[:, :-1].ne(bos_idx)
                & indices_anc[:, :-1].ne(eos_idx)
            )
            masked_positions = (
                decoder_input.eq(mask_idx)
                & content_positions
            )
            mask_pct = (
                100.0
                * masked_positions.sum(dim=1).float()
                / content_positions.sum(dim=1).clamp_min(1).float()
            ).mean()
            log_values["train_mask_pct"] = mask_pct.item()

            logits_anc = self.decoder_student(
                x=decoder_input,
                xa=z_ctx_anc,
            )
            logits_pos = self.decoder_student(
                x=decoder_input,
                xa=z_ctx_pos,
            )

            _assert_eos_exists(target, eos_idx)

            # Contrastive likelihood scoring is intentionally clean:
            # no prefix corruption and no decoder self-attention dropout.
            score_anc, score_pos = self._compute_clean_likelihood_scores(
                z_ctx_anc=z_ctx_anc,
                z_ctx_pos=z_ctx_pos,
                canonical_tokens=indices_anc,
                pad_idx=pad_idx,
            )

            ce_loss = (
                _sequence_mean_ce(logits_anc, target, pad_idx)
                + _sequence_mean_ce(logits_pos, target, pad_idx)
            )

            entropy_loss_anc, entropy_anc = _entropy_regularizer(
                logits_anc,
                target,
                pad_idx=pad_idx,
                eos_idx=eos_idx,
                beta=self.entropy_beta,
            )
            entropy_loss_pos, entropy_pos = _entropy_regularizer(
                logits_pos,
                target,
                pad_idx=pad_idx,
                eos_idx=eos_idx,
                beta=self.entropy_beta,
            )

            nce_anc = _infonce(score_anc, self.nce_temperature)
            nce_pos = _infonce(score_pos, self.nce_temperature)

            loss = (
                0.5 * ce_loss
                + 0.5 * (entropy_loss_anc + entropy_loss_pos)
                + nce_anc
                + nce_pos
            )

            diagonal = torch.diag(score_anc).mean()
            if score_anc.size(0) > 1:
                off_diagonal = (
                    score_anc.sum() - torch.diag(score_anc).sum()
                ) / (
                    score_anc.numel() - score_anc.size(0)
                )
                score_gap = diagonal - off_diagonal
            else:
                score_gap = diagonal.new_zeros(())

            codes_pred_anc = _clean_codes(
                logits_anc.argmax(dim=-1),
                pad_idx,
                bos_idx,
                eos_idx,
                mask_idx,
            )
            codes_pred_pos = _clean_codes(
                logits_pos.argmax(dim=-1),
                pad_idx,
                bos_idx,
                eos_idx,
                mask_idx,
            )

            log_values.update(
                {
                    "train_loss": loss.item(),
                    "train_autoreg_loss": 0.5 * ce_loss.item(),
                    "train_entropy_nats": 0.5
                    * (entropy_anc.item() + entropy_pos.item()),
                    "train_entropy_reg": 0.5
                    * (entropy_loss_anc + entropy_loss_pos).item(),
                    "train_nce_anc": nce_anc.item(),
                    "train_nce_pos": nce_pos.item(),
                    "train_Sgap_anc": score_gap.item(),
                }
            )

        else:
            self.encoder_student.eval()
            self.decoder_student.eval()

            with torch.no_grad():
                z_ctx_anc = self.encoder_teacher(anc)
                z_ctx_pos = self.encoder_teacher(pos)
                indices_anc = self.generate_teacher_beam_tokens(z_ctx_anc)
                indices_pos = self.generate_teacher_beam_tokens(z_ctx_pos)

            codes_pred_anc = _clean_codes(
                indices_anc,
                pad_idx,
                bos_idx,
                eos_idx,
                mask_idx,
            )
            codes_pred_pos = _clean_codes(
                indices_pos,
                pad_idx,
                bos_idx,
                eos_idx,
                mask_idx,
            )

            keep_ids = [
                idx
                for idx, seq in enumerate(codes_pred_anc)
                if seq.numel() > 0
            ]

            if not keep_ids:
                loss = torch.zeros((), device=self.device)
                return {"loss": loss, "z": z_ctx_anc[:1]}

            keep_tensor = torch.tensor(
                keep_ids,
                dtype=torch.long,
                device=z_ctx_anc.device,
            )
            z_ctx_anc = z_ctx_anc[keep_tensor]
            z_ctx_pos = z_ctx_pos[keep_tensor]
            codes_pred_anc = [codes_pred_anc[idx] for idx in keep_ids]
            codes_pred_pos = [codes_pred_pos[idx] for idx in keep_ids]

            for seq in codes_pred_anc:
                self.val_codes_cache.append(
                    seq.detach().cpu().tolist()
                )

            loss = torch.zeros((), device=self.device)

        similarity = self._get_normalized_edit_similarity(
            codes_pred_anc,
            codes_pred_pos,
        )
        log_values[f"{mode}_student_acc"] = similarity

        self.z_anc = z_ctx_anc[0]
        self.z_pos = z_ctx_pos[0]
        self.z_anc_quant = z_ctx_anc[0]
        self.z_pos_quant = z_ctx_pos[0]

        self.log_dict(
            log_values,
            on_epoch=True,
            on_step=True,
            logger=True,
            prog_bar=True,
        )
        return {"loss": loss, "z": z_ctx_anc}

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, mode="train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, mode="valid")

    @torch.no_grad()
    def on_train_batch_start(self, batch, batch_idx):
        self._update_corruption_params_stepwise()
        self.decoder_student.set_self_attn_drop_p(self.sa_drop_p)

        self.log(
            "sched_target_frac",
            float(self.target_frac),
            on_step=True,
            on_epoch=False,
            prog_bar=True,
        )
        self.log(
            "sched_sa_drop_p",
            float(self.sa_drop_p),
            on_step=True,
            on_epoch=False,
            prog_bar=True,
        )

    @torch.no_grad()
    def on_train_batch_end(self, outputs, batch, batch_idx):
        step = int(self.global_step)
        if step == 0:
            return
        if step % self.ema_update_every != 0:
            return
        if step == self._last_ema_update_step:
            return

        for student_param, teacher_param in zip(
            self.encoder_student.parameters(),
            self.encoder_teacher.parameters(),
        ):
            teacher_param.data.mul_(self.teacher_decay).add_(
                student_param.data,
                alpha=1.0 - self.teacher_decay,
            )

        for student_param, teacher_param in zip(
            self.decoder_student.parameters(),
            self.decoder_teacher.parameters(),
        ):
            teacher_param.data.mul_(self.teacher_decay).add_(
                student_param.data,
                alpha=1.0 - self.teacher_decay,
            )

        self._last_ema_update_step = step

    @torch.no_grad()
    def on_validation_epoch_end(self):
        seqs = self.val_codes_cache
        count = len(seqs)

        if count == 0:
            self.val_codes_cache.clear()
            return

        sequence_counter = Counter(tuple(seq) for seq in seqs)
        collision_rate = (
            count - len(sequence_counter)
        ) / count
        top_sequence_fraction = (
            sequence_counter.most_common(1)[0][1] / count
        )

        token_counter = Counter()
        total_tokens = 0
        for seq in seqs:
            token_counter.update(seq)
            total_tokens += len(seq)

        used_tokens = len(token_counter)
        top_token_fraction = (
            token_counter.most_common(1)[0][1] / total_tokens
            if total_tokens > 0
            else 0.0
        )

        self.log(
            "val_collision_rate",
            collision_rate,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "val_seq_top1_frac",
            top_sequence_fraction,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "val_tok_used",
            float(used_tokens),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "val_tok_top1_frac",
            float(top_token_fraction),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        self.val_codes_cache.clear()

    def _update_corruption_params_stepwise(self) -> None:
        warmup_steps = self.warmup_epochs * self.steps_per_epoch
        anneal_steps = self.anneal_epochs * self.steps_per_epoch
        step = int(self.global_step)

        if step < warmup_steps:
            self.target_frac = self.target_frac_start
            self.sa_drop_p = self.sa_drop_start
            return

        step_after_warmup = step - warmup_steps
        if step_after_warmup < anneal_steps:
            progress = step_after_warmup / max(1, anneal_steps)
            self.target_frac = _linear_schedule(
                self.target_frac_start,
                self.target_frac_end,
                progress,
            )
            self.sa_drop_p = _linear_schedule(
                self.sa_drop_start,
                self.sa_drop_end,
                progress,
            )
            return

        self.target_frac = self.target_frac_end
        self.sa_drop_p = self.sa_drop_end

    @torch.no_grad()
    def predict_step(self, batch, batch_idx):
        self.encoder_teacher.eval()
        self.decoder_teacher.eval()
        self.decoder_teacher.set_self_attn_drop_p(0.0)

        z_ctx = self.encoder_teacher(batch)
        indices = self.generate_teacher_beam_tokens(z_ctx)
        return z_ctx, z_ctx, indices

    def configure_optimizers(self):
        param_groups = [
            {
                "params": [
                    p
                    for p in self.decoder_student.parameters()
                    if p.requires_grad
                ],
                "lr": self.lr,
            },
            {
                "params": [
                    p
                    for p in self.encoder_student.parameters()
                    if p.requires_grad
                ],
                "lr": self.lr * 0.1,
            },
        ]

        if self.optim == "adam":
            optimizer = torch.optim.Adam(param_groups)
        elif self.optim == "adamw":
            if self.wt_decay <= 0:
                raise ValueError("AdamW requires positive weight decay")
            optimizer = torch.optim.AdamW(
                param_groups,
                weight_decay=self.wt_decay,
            )
        else:
            raise NotImplementedError(
                f"Unsupported optimizer: {self.optim}"
            )

        if not self.use_lr_scheduler:
            return {"optimizer": optimizer}

        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(self.config.get("scheduler_t_max", 70000)),
            eta_min=float(
                self.config.get("scheduler_eta_min", 5e-7)
            ),
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": self.config.get(
                    "lr_scheduler_interval",
                    "epoch",
                ),
            },
        }

    @classmethod
    def load_from_pretrained(
        cls,
        checkpoint,
        gpu_index=0,
    ):
        model = cls.load_from_checkpoint(
            checkpoint,
            map_location="cpu",
            strict=False,
        )
        model.eval()

        if torch.cuda.is_available():
            model = model.to(f"cuda:{gpu_index}")

        return model

    @torch.no_grad()
    def _prepare_inference_input(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert common evaluation layouts to encoder layout.

        Accepted:
          * (B, 1, F, T)
          * (B, F, T)
          * (B, T, F)
          * (B, 1, T) for encoders that accept waveform-like input.
        """
        if not torch.is_tensor(x):
            raise TypeError(f"Expected torch.Tensor, got {type(x)}")

        if x.dim() == 4:
            if x.size(1) != 1:
                raise ValueError(
                    "4D input must have shape (B, 1, F, T)"
                )
            return x.squeeze(1).permute(0, 2, 1).contiguous()

        if x.dim() != 3:
            raise ValueError(
                f"Expected a 3D or 4D input, got {tuple(x.shape)}"
            )

        if x.size(1) == 1:
            return x.contiguous()

        n_mels = self.config.get("n_mels")
        if n_mels is not None:
            n_mels = int(n_mels)

            if x.size(1) == n_mels and x.size(2) != n_mels:
                return x.permute(0, 2, 1).contiguous()

            if x.size(2) == n_mels:
                return x.contiguous()

        return x.permute(0, 2, 1).contiguous()

    @torch.no_grad()
    def tokenize(
        self,
        x: torch.Tensor,
        max_len: int = 60,
        beam_size: int = 4,
        length_cap_ratio: float = 0.15,
        min_cap: int = 2,
        rep_gamma: float = 1.3,
        length_norm_alpha: float = 0.0,
        strip_special_tokens: bool = True,
    ) -> List[torch.Tensor]:
        """
        Teacher-only deterministic Stage-III tokenization for evaluation.

        Returns a ragged list of one-dimensional LongTensors.
        """
        self.encoder_teacher.eval()
        self.decoder_teacher.eval()
        self.decoder_teacher.set_self_attn_drop_p(0.0)

        x = x.to(self.device)
        encoder_input = self._prepare_inference_input(x)
        z_ctx = self.encoder_teacher(encoder_input)

        indices = self.generate_teacher_beam_tokens(
            z_ctx,
            max_len=max_len,
            beam_size=beam_size,
            length_cap_ratio=length_cap_ratio,
            min_cap=min_cap,
            rep_gamma=rep_gamma,
            length_norm_alpha=length_norm_alpha,
        )

        pad_idx = self.decoder_teacher.PAD_IDX
        bos_idx = self.decoder_teacher.BOS_IDX
        eos_idx = self.decoder_teacher.EOS_IDX
        mask_idx = getattr(
            self.decoder_teacher,
            "MASK_IDX",
            None,
        )

        output: List[torch.Tensor] = []
        for seq in indices:
            if strip_special_tokens:
                keep = (
                    seq.ne(pad_idx)
                    & seq.ne(bos_idx)
                    & seq.ne(eos_idx)
                )
                if mask_idx is not None:
                    keep &= seq.ne(mask_idx)
                seq = seq[keep]
            else:
                eos_pos = seq.eq(eos_idx).nonzero(as_tuple=True)[0]
                if eos_pos.numel() > 0:
                    seq = seq[: int(eos_pos[0].item()) + 1]
                else:
                    pad_pos = seq.eq(pad_idx).nonzero(as_tuple=True)[0]
                    if pad_pos.numel() > 0:
                        seq = seq[: int(pad_pos[0].item())]

            output.append(seq)

        return output

    @torch.no_grad()
    def extract_tokens(
        self,
        audio_data,
        hop,
        gpu_index="0",
    ):
        """Backwards-compatible long-audio Stage-III token extraction."""
        if gpu_index is None:
            raise RuntimeError("Inference requires a GPU index")

        feature_extractor = torch.nn.Sequential(
            T.MelSpectrogram(
                sample_rate=self.config["fs"],
                n_fft=int(
                    self.config["win_size"] * self.config["fs"]
                ),
                hop_length=int(
                    self.config["win_hop"] * self.config["fs"]
                ),
                n_mels=self.config["n_mels"],
                center=True,
            ).to(self.device),
            T.AmplitudeToDB().to(self.device),
        )

        device = f"cuda:{gpu_index}"
        chunk_size = int(
            self.config["seglen"] * self.config["fs"]
        )
        hop_size = int(hop * self.config["fs"])

        starts = np.arange(
            0,
            len(audio_data) - chunk_size + 1,
            hop_size,
        )
        if len(starts) == 0:
            return torch.empty((0, 0), dtype=torch.int16)

        audio_chunks = torch.stack(
            [
                audio_data[start : start + chunk_size]
                for start in starts
            ],
            dim=0,
        )
        audio_features = feature_extractor(
            audio_chunks.to(device)
        ).permute(0, 2, 1)

        _, _, indices = self.get_predictions(
            audio_features,
            device,
        )
        return indices.detach().cpu().to(torch.int16)

    @torch.no_grad()
    def get_predictions(
        self,
        features,
        device,
    ):
        """Chunked Stage-III teacher decoding for long evaluation batches."""
        chunk_size = 256

        if len(features) <= chunk_size:
            return self.predict_step(features.to(device), 0)

        z_parts = []
        index_parts = []

        for start in range(0, len(features), chunk_size):
            z_ctx, _, indices = self.predict_step(
                features[start : start + chunk_size].to(device),
                0,
            )
            z_parts.append(z_ctx.detach().cpu())
            index_parts.append(indices.detach().cpu())

        z_ctx = torch.cat(z_parts, dim=0)
        indices = torch.cat(index_parts, dim=0)
        return z_ctx, z_ctx, indices

    @staticmethod
    def _edit_distance(seq1, seq2) -> int:
        """Standard Levenshtein distance."""
        len1 = len(seq1)
        len2 = len(seq2)

        if len1 == 0:
            return len2
        if len2 == 0:
            return len1

        previous = list(range(len2 + 1))

        for idx1 in range(1, len1 + 1):
            current = [idx1] + [0] * len2

            for idx2 in range(1, len2 + 1):
                substitution_cost = (
                    0 if seq1[idx1 - 1] == seq2[idx2 - 1] else 1
                )
                current[idx2] = min(
                    previous[idx2] + 1,
                    current[idx2 - 1] + 1,
                    previous[idx2 - 1] + substitution_cost,
                )

            previous = current

        return previous[len2]

    @classmethod
    def _get_normalized_edit_similarity(
        cls,
        codes_anc,
        codes_pos,
    ) -> float:
        """
        Mean 1 - normalized Levenshtein distance over paired token sequences.
        """
        similarities = []

        for anc, pos in zip(codes_anc, codes_pos):
            if isinstance(anc, torch.Tensor):
                anc = anc.detach().cpu().tolist()
            if isinstance(pos, torch.Tensor):
                pos = pos.detach().cpu().tolist()

            denominator = max(len(anc), len(pos))
            if denominator == 0:
                similarities.append(1.0)
                continue

            distance = cls._edit_distance(anc, pos)
            similarities.append(1.0 - distance / denominator)

        return (
            float(np.mean(similarities))
            if similarities
            else 0.0
        )
