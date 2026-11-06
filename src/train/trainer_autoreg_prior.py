import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

import lightning as L
import transformers

from train.trainer_stage1 import AudioTokenizer as Stage1AudioTokenizer
from models import TextDecoder


def _load_stage1_backbone(config):
    """Load the frozen Stage-I model used to define Stage-II targets."""
    checkpoints = config.get("checkpoints", {})

    if not isinstance(checkpoints, dict):
        raise TypeError(
            "config['checkpoints'] must be a mapping containing 'stage1'."
        )

    checkpoint_path = checkpoints.get("stage1")
    if not checkpoint_path:
        raise ValueError(
            "AudioTokenizerARPrior requires checkpoints.stage1. "
            "Stage II starts from the Stage-I tokenizer."
        )

    model = Stage1AudioTokenizer.load_from_checkpoint(
        checkpoint_path,
        map_location="cpu",
        strict=False,
    )
    model.eval()
    model.requires_grad_(False)

    return model, "stage1", checkpoint_path


def add_bos_eos(seq: torch.Tensor, bos_idx: int, eos_idx: int) -> torch.Tensor:
    """Add BOS/EOS while preserving the input tensor's device and dtype."""
    bos = seq.new_tensor([bos_idx])
    eos = seq.new_tensor([eos_idx])
    return torch.cat((bos, seq, eos), dim=0)


def linear_anneal(start: float, end: float, t: float) -> float:
    """Linear interpolation for t in [0, 1]."""
    return start + t * (end - start)


def cosine_anneal(start: float, end: float, t: float) -> float:
    """Cosine interpolation for t in [0, 1]."""
    return end + 0.5 * (start - end) * (1.0 + math.cos(math.pi * t))


def token_acc_from_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    pad_idx: int,
) -> float:
    """Token accuracy excluding PAD positions."""
    pred = logits.argmax(dim=-1)
    mask = target.ne(pad_idx)
    correct = pred.eq(target) & mask
    denom = mask.sum().clamp_min(1)
    return (correct.sum().float() / denom.float()).item()


def token_dropout_to_fraction(
    x: torch.Tensor,
    pad_idx: int,
    bos_idx: int,
    eos_idx: int,
    mask_idx: int,
    target_frac: float,
    min_keep: int = 1,
) -> torch.Tensor:
    """
    Replace approximately ``target_frac`` of valid prefix tokens with MASK.

    BOS/EOS/PAD are never masked. At least ``min_keep`` valid content tokens
    are left unmasked in every sequence.
    """
    if not 0.0 <= target_frac <= 1.0:
        raise ValueError(f"target_frac must be in [0, 1], got {target_frac}")

    batch_size, seq_len = x.shape
    out = x.clone()

    valid = x.ne(pad_idx) & x.ne(bos_idx) & x.ne(eos_idx)
    valid_count = valid.sum(dim=1)

    drop_count = torch.round(valid_count.float() * target_frac).long()
    max_drop = (valid_count - min_keep).clamp_min(0)
    drop_count = torch.minimum(drop_count, max_drop)

    if drop_count.max().item() == 0:
        return out

    scores = torch.rand((batch_size, seq_len), device=x.device)
    scores.masked_fill_(~valid, 2.0)
    sorted_idx = scores.argsort(dim=1)

    cols = torch.arange(seq_len, device=x.device).unsqueeze(0)
    take = cols < drop_count.unsqueeze(1)
    rows = torch.arange(batch_size, device=x.device).unsqueeze(1).expand_as(sorted_idx)

    out[rows[take], sorted_idx[take]] = mask_idx
    return out


def _prefix_token_counts(
    prefix: torch.Tensor,
    vocab_size: int,
    excluded_token_ids: Optional[Sequence[int]] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """
    Count token occurrences in a batched prefix.

    Args:
        prefix: (B, L) token ids.
        vocab_size: decoder vocabulary size.
        excluded_token_ids: special tokens whose counts are forced to zero.
        dtype: dtype of the returned count tensor.

    Returns:
        counts: (B, V), on the same device as ``prefix``.
    """
    if prefix.dim() != 2:
        raise ValueError(
            f"Expected prefix with shape (B, L), got {tuple(prefix.shape)}"
        )

    if dtype is None:
        dtype = torch.float32

    batch_size = prefix.size(0)
    counts = torch.zeros(
        (batch_size, vocab_size),
        dtype=dtype,
        device=prefix.device,
    )
    counts.scatter_add_(1, prefix, torch.ones_like(prefix, dtype=dtype))

    if excluded_token_ids:
        valid_ids = [
            int(token_id)
            for token_id in excluded_token_ids
            if token_id is not None and 0 <= int(token_id) < vocab_size
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

    The same helper is used by top-p sampling and beam search.

    Special symbols are excluded by setting their effective prefix counts to
    zero. ``rep_gamma=1`` disables the penalty.
    """
    if logits.dim() != 2:
        raise ValueError(
            f"Expected logits with shape (B, V), got {tuple(logits.shape)}"
        )
    if prefix.dim() != 2:
        raise ValueError(
            f"Expected prefix with shape (B, L), got {tuple(prefix.shape)}"
        )
    if logits.size(0) != prefix.size(0):
        raise ValueError(
            "logits and prefix must have the same batch dimension: "
            f"{logits.size(0)} != {prefix.size(0)}"
        )
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


@torch.no_grad()
def nucleus_multinomial_sample(
    logits: torch.Tensor,
    top_p: float = 0.9,
    temperature: float = 1.0,
    min_tokens_to_keep: int = 1,
    prefix: Optional[torch.Tensor] = None,
    rep_gamma: float = 1.0,
    excluded_token_ids: Optional[Sequence[int]] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Multinomial top-p sampler used only as a Stage-II collapse diagnostic.

    Repetition control uses the same logit-space rule as beam search:

        adjusted_logit(v) = logit(v) - c(v) * log(rep_gamma)

    Top-p support is computed from the repetition-adjusted logits. Temperature
    is then applied to the retained logits immediately before multinomial
    sampling.

    Args:
        logits: (B, V) next-token logits.
        top_p: nucleus threshold.
        temperature: sampling temperature.
        min_tokens_to_keep: minimum retained support.
        prefix: (B, L) generated prefix used for repetition counts.
        rep_gamma: compound repetition factor; 1.0 disables the penalty.
        excluded_token_ids: special symbols excluded from repetition control.

    Returns:
        next_tokens: (B,)
        stats: diagnostic distribution statistics.
    """
    if logits.dim() != 2:
        raise ValueError(f"Expected (B, V) logits, got {tuple(logits.shape)}")
    if not 0.0 < top_p <= 1.0:
        raise ValueError(f"top_p must be in (0, 1], got {top_p}")
    if temperature <= 0.0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    if min_tokens_to_keep < 1:
        raise ValueError(
            f"min_tokens_to_keep must be >= 1, got {min_tokens_to_keep}"
        )
    if rep_gamma < 1.0:
        raise ValueError(f"rep_gamma must be >= 1.0, got {rep_gamma}")

    _, vocab_size = logits.shape

    adjusted_logits = logits
    if prefix is not None:
        adjusted_logits = _apply_logit_repetition_penalty(
            logits=adjusted_logits,
            prefix=prefix,
            rep_gamma=rep_gamma,
            excluded_token_ids=excluded_token_ids,
        )

    # Build the nucleus from the repetition-adjusted next-token distribution.
    nucleus_probs = F.softmax(adjusted_logits, dim=-1)
    sorted_probs, sorted_idx = torch.sort(
        nucleus_probs,
        dim=-1,
        descending=True,
    )
    cdf = torch.cumsum(sorted_probs, dim=-1)

    # Retain the smallest nucleus whose cumulative mass reaches/exceeds top_p.
    # The token that crosses the threshold must therefore be kept.
    sorted_remove = cdf > top_p
    sorted_remove[..., 1:] = sorted_remove[..., :-1].clone()
    sorted_remove[..., 0] = False

    min_keep = min(max(1, min_tokens_to_keep), vocab_size)
    sorted_remove[..., :min_keep] = False

    keep_sorted = ~sorted_remove

    # Convert the keep mask back to vocabulary order.
    keep_vocab = torch.zeros_like(keep_sorted, dtype=torch.bool)
    keep_vocab.scatter_(1, sorted_idx, keep_sorted)

    # Mask non-nucleus logits, then apply temperature for the actual sample.
    filtered_logits = adjusted_logits.masked_fill(~keep_vocab, -torch.inf)
    sample_probs = F.softmax(filtered_logits / temperature, dim=-1)

    next_tokens = torch.multinomial(
        sample_probs,
        num_samples=1,
    ).squeeze(1)

    # Diagnostics reflect the actual final sampling distribution.
    entropy = -(
        sample_probs.clamp_min(1e-12)
        * sample_probs.clamp_min(1e-12).log()
    ).sum(dim=-1)

    sorted_sample_probs, _ = torch.sort(
        sample_probs,
        dim=-1,
        descending=True,
    )
    p1 = sorted_sample_probs[:, 0]
    p2 = (
        sorted_sample_probs[:, 1]
        if vocab_size > 1
        else torch.zeros_like(p1)
    )
    top12_gap = p1 - p2
    nucleus_size = keep_sorted.sum(dim=-1).float()
    effective_support = torch.exp(entropy)

    stats = {
        "entropy": entropy.detach(),
        "top1_prob": p1.detach(),
        "top12_gap": top12_gap.detach(),
        "nucleus_size": nucleus_size.detach(),
        "nucleus_eff_support": effective_support.detach(),
    }
    return next_tokens, stats


@torch.no_grad()
def decode_with_top_p_multinomial(
    decoder,
    xa: torch.Tensor,
    bos_idx: int,
    eos_idx: int,
    pad_idx: int,
    max_len: int = 32,
    top_p: float = 0.9,
    temperature: float = 1.0,
    rep_gamma: float = 1.3,
    mask_idx: Optional[int] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Stage-II stochastic AR diagnostic used to check for decoder collapse.

    This remains a diagnostic path. General tokenization/evaluation uses
    ``beam_search_teacher_targets_single`` through
    ``AudioTokenizerARPrior.tokenize``.

    Repetition control is count-compounded and applied to logits before
    nucleus construction.
    """
    if max_len < 2:
        raise ValueError(
            f"max_len must be >= 2 for autoregressive diagnostics, got {max_len}"
        )

    device = xa.device
    batch_size = xa.size(0)

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

    entropy_steps: List[torch.Tensor] = []
    top1_steps: List[torch.Tensor] = []
    gap_steps: List[torch.Tensor] = []
    nucleus_steps: List[torch.Tensor] = []
    support_steps: List[torch.Tensor] = []

    excluded = [bos_idx, eos_idx, pad_idx]
    if mask_idx is not None:
        excluded.append(mask_idx)

    for _ in range(1, max_len):
        logits = decoder(x=seq, xa=xa)
        step_logits = logits[:, -1, :].clone()

        # Finished rows emit PAD only.
        step_logits[finished] = -torch.inf
        step_logits[finished, pad_idx] = 0.0

        next_tok, stats = nucleus_multinomial_sample(
            logits=step_logits,
            top_p=top_p,
            temperature=temperature,
            min_tokens_to_keep=1,
            prefix=seq,
            rep_gamma=rep_gamma,
            excluded_token_ids=excluded,
        )

        entropy_steps.append(stats["entropy"])
        top1_steps.append(stats["top1_prob"])
        gap_steps.append(stats["top12_gap"])
        nucleus_steps.append(stats["nucleus_size"])
        support_steps.append(stats["nucleus_eff_support"])

        seq = torch.cat(
            (seq, next_tok.unsqueeze(1)),
            dim=1,
        )
        finished |= next_tok.eq(eos_idx)

        if finished.all():
            break

    # Preserve the diagnostic API: output is padded to max_len.
    if seq.size(1) < max_len:
        pad = torch.full(
            (batch_size, max_len - seq.size(1)),
            pad_idx,
            dtype=torch.long,
            device=device,
        )
        seq = torch.cat((seq, pad), dim=1)

    out = seq[:, :max_len].clone()

    for batch_idx in range(batch_size):
        eos_pos = out[batch_idx].eq(eos_idx).nonzero(as_tuple=True)[0]
        if eos_pos.numel() > 0:
            end = int(eos_pos[0].item()) + 1
            if end < out.size(1):
                out[batch_idx, end:] = pad_idx

    entropy = torch.stack(entropy_steps, dim=1)
    p1 = torch.stack(top1_steps, dim=1)
    gap = torch.stack(gap_steps, dim=1)
    nucleus = torch.stack(nucleus_steps, dim=1)
    support = torch.stack(support_steps, dim=1)

    content = out[:, 1:]
    valid = content.ne(pad_idx) & content.ne(eos_idx)
    adjacent_repeat = (
        content[:, 1:].eq(content[:, :-1])
        & valid[:, 1:]
        & valid[:, :-1]
    )
    repeat_rate = (
        adjacent_repeat.float().sum(dim=1)
        / valid[:, 1:].float().sum(dim=1).clamp_min(1.0)
    )

    diagnostics = {
        "mean_entropy_per_step": entropy.mean(dim=1),
        "mean_top1_prob_per_step": p1.mean(dim=1),
        "mean_top12_gap_per_step": gap.mean(dim=1),
        "mean_nucleus_size": nucleus.mean(dim=1),
        "mean_nucleus_eff_support": support.mean(dim=1),
        "adjacent_repeat_rate": repeat_rate,
    }
    return out, diagnostics


def _beam_rank(
    score: float,
    decoded_length: int,
    alpha_len: float,
) -> float:
    """Optional length-normalized ranking score."""
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
    alpha_len: float = 0.0,
) -> torch.Tensor:
    r"""
    Inference-time beam search with compound repetition control.

    For every active beam:

        adjusted_logit(v) = logit(v) - c(v) * log(rep_gamma)

    The exact same logit-space helper used by top-p sampling is used here.
    BOS/EOS/PAD/MASK are excluded from repetition counts.

    PAD is disallowed for unfinished beams. EOS is available at every ordinary
    step and is forced at the final rate-capped step.

    Beam scores are cumulative token log-probabilities. Optional length
    normalization changes only ranking and final selection.

    The rate cap is ``floor(length_cap_ratio * T_x)``, bounded by ``min_cap``
    and ``max_len`` for evaluation convenience.

    Returns:
        LongTensor of shape (B, K, L), padded with PAD. Beam 0 is the final
        selected hypothesis.
    """
    if xa.dim() != 3:
        raise ValueError(
            f"xa must have shape (B, T, D), got {tuple(xa.shape)}"
        )
    if cond_lens.dim() != 1 or cond_lens.numel() != xa.size(0):
        raise ValueError(
            "cond_lens must be a 1D tensor with one valid length per sample"
        )
    if beam_size < 1:
        raise ValueError(
            f"beam_size must be >= 1, got {beam_size}"
        )
    if max_len < 1:
        raise ValueError(
            f"max_len must be >= 1, got {max_len}"
        )
    if min_cap < 1:
        raise ValueError(
            f"min_cap must be >= 1, got {min_cap}"
        )
    if length_cap_ratio <= 0.0:
        raise ValueError(
            f"length_cap_ratio must be > 0, got {length_cap_ratio}"
        )
    if rep_gamma < 1.0:
        raise ValueError(
            f"rep_gamma must be >= 1.0, got {rep_gamma}"
        )
    if alpha_len < 0.0:
        raise ValueError(
            f"alpha_len must be >= 0, got {alpha_len}"
        )

    device = xa.device
    batch_size = xa.size(0)

    excluded = [int(bos_idx), int(eos_idx), int(pad_idx)]
    if mask_idx is not None:
        excluded.append(int(mask_idx))

    all_ranked_beams: List[List[Tuple[List[int], float, bool]]] = []

    for batch_idx in range(batch_size):
        valid_cond_len = int(cond_lens[batch_idx].item())
        if valid_cond_len < 1:
            raise ValueError(
                f"cond_lens[{batch_idx}] must be >= 1, "
                f"got {valid_cond_len}"
            )

        length_cap = int(
            math.floor(length_cap_ratio * valid_cond_len)
        )
        length_cap = max(min_cap, length_cap)
        length_cap = min(max_len, length_cap)

        xa_b = xa[
            batch_idx : batch_idx + 1,
            :valid_cond_len,
        ]

        # (token sequence including BOS, raw cumulative score, finished)
        beams: List[Tuple[List[int], float, bool]] = [
            ([int(bos_idx)], 0.0, False)
        ]

        for step in range(1, length_cap + 1):
            candidates: List[
                Tuple[List[int], float, bool]
            ] = []

            for seq_tokens, raw_score, finished in beams:
                if finished:
                    candidates.append(
                        (seq_tokens, raw_score, True)
                    )
                    continue

                seq_tensor = torch.tensor(
                    seq_tokens,
                    dtype=torch.long,
                    device=device,
                ).unsqueeze(0)

                logits = decoder(
                    x=seq_tensor,
                    xa=xa_b,
                )[:, -1, :]

                adjusted_logits = _apply_logit_repetition_penalty(
                    logits=logits,
                    prefix=seq_tensor,
                    rep_gamma=rep_gamma,
                    excluded_token_ids=excluded,
                )

                # PAD cannot extend an unfinished beam.
                adjusted_logits[:, pad_idx] = -torch.inf

                # At the rate cap, EOS is forced.
                if step == length_cap:
                    forced_logits = torch.full_like(
                        adjusted_logits,
                        -torch.inf,
                    )
                    forced_logits[:, eos_idx] = (
                        adjusted_logits[:, eos_idx]
                    )
                    adjusted_logits = forced_logits

                log_probs = F.log_softmax(
                    adjusted_logits,
                    dim=-1,
                ).squeeze(0)

                local_k = min(
                    beam_size,
                    log_probs.numel(),
                )
                token_log_probs, token_ids = torch.topk(
                    log_probs,
                    k=local_k,
                )

                for token_log_prob, token_id in zip(
                    token_log_probs.tolist(),
                    token_ids.tolist(),
                ):
                    new_tokens = (
                        seq_tokens + [int(token_id)]
                    )
                    new_score = (
                        raw_score + float(token_log_prob)
                    )
                    new_finished = (
                        int(token_id) == int(eos_idx)
                    )
                    candidates.append(
                        (
                            new_tokens,
                            new_score,
                            new_finished,
                        )
                    )

            if not candidates:
                break

            candidates.sort(
                key=lambda item: _beam_rank(
                    item[1],
                    decoded_length=len(item[0]) - 1,
                    alpha_len=alpha_len,
                ),
                reverse=True,
            )
            beams = candidates[:beam_size]

            if all(
                finished
                for _, _, finished in beams
            ):
                break

        completed = [
            beam for beam in beams if beam[2]
        ]

        selection_pool = (
            completed if completed else beams
        )
        selection_pool = sorted(
            selection_pool,
            key=lambda item: _beam_rank(
                item[1],
                decoded_length=len(item[0]) - 1,
                alpha_len=alpha_len,
            ),
            reverse=True,
        )
        selected = selection_pool[0]

        # Place the selected sequence at beam index 0.
        selected_index = beams.index(selected)
        remaining = (
            beams[:selected_index]
            + beams[selected_index + 1 :]
        )
        remaining.sort(
            key=lambda item: _beam_rank(
                item[1],
                decoded_length=len(item[0]) - 1,
                alpha_len=alpha_len,
            ),
            reverse=True,
        )

        ranked = [selected] + remaining

        while len(ranked) < beam_size:
            ranked.append(ranked[-1])

        all_ranked_beams.append(
            ranked[:beam_size]
        )

    max_output_len = max(
        len(seq_tokens)
        for sample_beams in all_ranked_beams
        for seq_tokens, _, _ in sample_beams
    )

    output = torch.full(
        (
            batch_size,
            beam_size,
            max_output_len,
        ),
        pad_idx,
        dtype=torch.long,
        device=device,
    )

    for batch_idx, sample_beams in enumerate(
        all_ranked_beams
    ):
        for beam_idx, (
            seq_tokens,
            _,
            _,
        ) in enumerate(sample_beams):
            seq_tensor = torch.tensor(
                seq_tokens,
                dtype=torch.long,
                device=device,
            )
            output[
                batch_idx,
                beam_idx,
                : seq_tensor.numel(),
            ] = seq_tensor

    return output


class AudioTokenizerARPrior(L.LightningModule):
    """Stage-II autoregressive prior for PairAlign."""

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Stage II is initialized from the Stage-I tokenizer.
        (
            stage1_model,
            self.stage1_source,
            self.stage1_checkpoint,
        ) = _load_stage1_backbone(self.config)

        source_config = getattr(stage1_model, "config", {})
        source_vocab_size = (
            source_config.get("num_codewords")
            if isinstance(source_config, dict)
            else None
        )
        if (
            source_vocab_size is not None
            and int(source_vocab_size)
            != int(self.config["num_codewords"])
        ):
            raise ValueError(
                "Stage-II num_codewords does not match the selected "
                f"{self.stage1_source} checkpoint: "
                f"{self.config['num_codewords']} != {source_vocab_size}"
            )

        # The Stage-I encoder and VQ define the frozen canonical targets and
        # conditioning representations used by Stage II.
        self.encoder = stage1_model.encoder
        self.vq_layer = stage1_model.vq_layer
        self.encoder.eval()
        self.vq_layer.eval()

        for parameter in self.encoder.parameters():
            parameter.requires_grad = False
        for parameter in self.vq_layer.parameters():
            parameter.requires_grad = False

        # Do not register the full Stage-I model as a nested submodule. The
        # Stage-II checkpoint only needs the frozen encoder/VQ references.
        del stage1_model

        # Prefix-corruption schedule.
        self.target_frac_start = float(
            self.config.get("target_frac_start", 0.6)
        )
        self.target_frac_end = float(
            self.config.get("target_frac_end", 0.3)
        )
        self.target_frac_warmup_epochs = int(
            self.config.get("target_frac_warmup_epochs", 0)
        )
        self.target_frac_schedule = self.config.get(
            "target_frac_schedule", "linear"
        )
        self.target_frac = self.target_frac_start

        # Decoder self-attention dropout schedule.
        self.sa_drop_start = float(
            self.config.get("sa_drop_start", 0.6)
        )
        self.sa_drop_end = float(
            self.config.get("sa_drop_end", 0.2)
        )
        self.sa_drop_warmup_epochs = int(
            self.config.get("sa_drop_warmup_epochs", 0)
        )
        self.sa_drop_schedule = self.config.get(
            "sa_drop_schedule", "linear"
        )
        self.sa_drop_p = self.sa_drop_start

        self.decoder = TextDecoder(
            n_vocab=self.config["num_codewords"],
            n_ctx=512,
            n_state=512,
            n_head=8,
            n_layer=6,
        )

        self.optim = self.config["optimizer"]
        self.lr = self.config["lr"]
        self.wt_decay = self.config["wt_decay"]
        self.lr_scheduler = bool(
            self.config["lr_scheduler"]
        )

        self._validate_configuration()
        self.save_hyperparameters()

    def _validate_configuration(self) -> None:
        """Validate submission-critical Stage-II hyperparameters."""
        if int(self.config["num_codewords"]) <= 0:
            raise ValueError("num_codewords must be positive")

        if not 0.0 <= self.target_frac_start <= 1.0:
            raise ValueError("target_frac_start must be in [0, 1]")
        if not 0.0 <= self.target_frac_end <= 1.0:
            raise ValueError("target_frac_end must be in [0, 1]")

        if not 0.0 <= self.sa_drop_start < 1.0:
            raise ValueError("sa_drop_start must be in [0, 1)")
        if not 0.0 <= self.sa_drop_end < 1.0:
            raise ValueError("sa_drop_end must be in [0, 1)")

        if self.target_frac_schedule not in {"linear", "cosine"}:
            raise ValueError(
                "target_frac_schedule must be 'linear' or 'cosine'"
            )
        if self.sa_drop_schedule not in {"linear", "cosine"}:
            raise ValueError(
                "sa_drop_schedule must be 'linear' or 'cosine'"
            )

        if self.target_frac_warmup_epochs < 0:
            raise ValueError("target_frac_warmup_epochs must be >= 0")
        if self.sa_drop_warmup_epochs < 0:
            raise ValueError("sa_drop_warmup_epochs must be >= 0")

        if self.lr <= 0.0:
            raise ValueError("lr must be > 0")
        if self.wt_decay < 0.0:
            raise ValueError("wt_decay must be >= 0")

        diagnostic_rep_gamma = float(
            self.config.get("diagnostic_rep_gamma", 1.3)
        )
        if diagnostic_rep_gamma < 1.0:
            raise ValueError("diagnostic_rep_gamma must be >= 1.0")

    @staticmethod
    def deduplicate(
        seq: torch.Tensor,
    ) -> torch.Tensor:
        """Collapse consecutive duplicate token ids."""
        if seq.numel() <= 1:
            return seq

        keep = torch.ones_like(
            seq,
            dtype=torch.bool,
        )
        keep[1:] = seq[1:].ne(seq[:-1])
        return seq[keep]

    def _prepare_inference_input(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert common evaluation layouts to encoder layout (B, T, F).

        Supported:
          * (B, 1, F, T)
          * (B, F, T)
          * (B, T, F)
        """
        if x.dim() == 4:
            if x.size(1) != 1:
                raise ValueError(
                    "4D inference input must have shape "
                    "(B, 1, F, T); "
                    f"got {tuple(x.shape)}"
                )
            return (
                x.squeeze(1)
                .permute(0, 2, 1)
                .contiguous()
            )

        if x.dim() != 3:
            raise ValueError(
                "Inference input must be "
                "(B,1,F,T), (B,F,T), or (B,T,F); "
                f"got {tuple(x.shape)}"
            )

        n_mels = self.config.get("n_mels")
        if (
            n_mels is not None
            and x.size(1) == int(n_mels)
            and x.size(2) != int(n_mels)
        ):
            return (
                x.permute(0, 2, 1)
                .contiguous()
            )

        return x

    def _shared_step(
        self,
        batch,
        mode: str,
    ):
        if mode not in {"train", "valid"}:
            raise ValueError(
                f"Unsupported mode: {mode}"
            )

        log_values: Dict[str, float] = {}

        self.encoder.eval()
        self.vq_layer.eval()

        if mode == "train":
            self.decoder.train()
            self.decoder.set_self_attn_drop_p(self.sa_drop_p)
        else:
            self.decoder.eval()
            self.decoder.set_self_attn_drop_p(0.0)

        anc = (
            batch[0]
            .squeeze(1)
            .permute(0, 2, 1)
        )
        pos = (
            batch[1]
            .squeeze(1)
            .permute(0, 2, 1)
        )

        with torch.no_grad():
            z_ctx_anc = self.encoder(anc)
            z_ctx_pos = self.encoder(pos)

            (
                _,
                indices_anc,
                _,
            ) = self.vq_layer(z_ctx_anc)

            if indices_anc.min().item() < 0:
                raise RuntimeError(
                    "Stage-I VQ produced a negative token id"
                )

            if (
                indices_anc.max().item()
                >= self.config["num_codewords"]
            ):
                raise RuntimeError(
                    "Stage-I VQ token id exceeds "
                    "decoder vocabulary: "
                    f"min={indices_anc.min().item()}, "
                    f"max={indices_anc.max().item()}, "
                    f"vocab={self.config['num_codewords']}"
                )

        bos_idx = self.decoder.BOS_IDX
        eos_idx = self.decoder.EOS_IDX
        pad_idx = self.decoder.PAD_IDX
        mask_idx = self.decoder.MASK_IDX

        # Shared canonical Stage-II target:
        # anchor and positive conditioning paths both predict
        # the deterministic deduplicated Stage-I anchor target.
        anchor_targets = [
            add_bos_eos(
                self.deduplicate(seq),
                bos_idx,
                eos_idx,
            )
            for seq in indices_anc
        ]

        indices_anc = pad_sequence(
            anchor_targets,
            batch_first=True,
            padding_value=pad_idx,
        ).to(z_ctx_anc.device)

        decoder_input = indices_anc[:, :-1]
        target = indices_anc[:, 1:]

        if mode == "train":
            decoder_input = token_dropout_to_fraction(
                decoder_input,
                pad_idx=pad_idx,
                bos_idx=bos_idx,
                eos_idx=eos_idx,
                mask_idx=mask_idx,
                target_frac=self.target_frac,
                min_keep=2,
            )

        if mode == "valid":
            decoder_input_clean = (
                indices_anc[:, :-1]
            )
            target_clean = (
                indices_anc[:, 1:]
            )

            logits_clean = self.decoder(
                x=decoder_input_clean,
                xa=z_ctx_anc,
            )
            loss_clean = F.cross_entropy(
                logits_clean.reshape(
                    -1,
                    logits_clean.size(-1),
                ),
                target_clean.reshape(-1),
                ignore_index=pad_idx,
            )
            acc_clean = token_acc_from_logits(
                logits_clean,
                target_clean,
                pad_idx,
            )

            perm = torch.randperm(
                z_ctx_anc.size(0),
                device=z_ctx_anc.device,
            )
            logits_shuf = self.decoder(
                x=decoder_input_clean,
                xa=z_ctx_anc[perm],
            )
            loss_shuf = F.cross_entropy(
                logits_shuf.reshape(
                    -1,
                    logits_shuf.size(-1),
                ),
                target_clean.reshape(-1),
                ignore_index=pad_idx,
            )
            acc_shuf = token_acc_from_logits(
                logits_shuf,
                target_clean,
                pad_idx,
            )

            log_values.update(
                {
                    "valid_clean_loss": loss_clean.item(),
                    "valid_shuf_loss": loss_shuf.item(),
                    "valid_shuf_gap": (
                        loss_shuf - loss_clean
                    ).item(),
                    "valid_clean_tok_acc": acc_clean,
                    "valid_shuf_tok_acc": acc_shuf,
                }
            )

            # Diagnostic only: stochastic generation checks
            # whether the Stage-II prior has collapsed.
            _, diag = decode_with_top_p_multinomial(
                decoder=self.decoder,
                xa=z_ctx_anc[:16],
                bos_idx=bos_idx,
                eos_idx=eos_idx,
                pad_idx=pad_idx,
                mask_idx=mask_idx,
                max_len=32,
                top_p=0.9,
                temperature=1.0,
                rep_gamma=float(
                    self.config.get(
                        "diagnostic_rep_gamma",
                        1.3,
                    )
                ),
            )

            log_values.update(
                {
                    "entropy": diag[
                        "mean_entropy_per_step"
                    ].mean().item(),
                    "top1_prob": diag[
                        "mean_top1_prob_per_step"
                    ].mean().item(),
                    "top12_gap": diag[
                        "mean_top12_gap_per_step"
                    ].mean().item(),
                    "nucleus_size": diag[
                        "mean_nucleus_size"
                    ].mean().item(),
                    "eff_support": diag[
                        "mean_nucleus_eff_support"
                    ].mean().item(),
                    "repeat_rate": diag[
                        "adjacent_repeat_rate"
                    ].mean().item(),
                }
            )

        logits_pred_anc = self.decoder(
            x=decoder_input,
            xa=z_ctx_anc,
        )
        logits_pred_pos = self.decoder(
            x=decoder_input,
            xa=z_ctx_pos,
        )

        loss_anc = F.cross_entropy(
            logits_pred_anc.reshape(
                -1,
                logits_pred_anc.size(-1),
            ),
            target.reshape(-1),
            ignore_index=pad_idx,
        )
        loss_pos = F.cross_entropy(
            logits_pred_pos.reshape(
                -1,
                logits_pred_pos.size(-1),
            ),
            target.reshape(-1),
            ignore_index=pad_idx,
        )

        loss = loss_anc + loss_pos
        log_values[
            f"{mode}_autoreg_loss"
        ] = loss.item()

        # Lightweight compatibility attributes for callbacks.
        self.z_anc = z_ctx_anc[0]
        self.z_pos = z_ctx_pos[0]
        target_codes = []
        pred_codes = []

        pred = logits_pred_pos.argmax(
            dim=-1
        )

        for batch_idx in range(
            indices_anc.size(0)
        ):
            target_seq = (
                indices_anc[batch_idx]
            )
            pred_seq = pred[batch_idx]

            target_keep = (
                target_seq.ne(pad_idx)
                & target_seq.ne(bos_idx)
                & target_seq.ne(eos_idx)
                & target_seq.ne(mask_idx)
            )
            pred_keep = (
                pred_seq.ne(pad_idx)
                & pred_seq.ne(bos_idx)
                & pred_seq.ne(eos_idx)
                & pred_seq.ne(mask_idx)
            )

            target_codes.append(
                target_seq[target_keep]
            )
            pred_codes.append(
                pred_seq[pred_keep]
            )

        log_values[
            f"{mode}_acc"
        ] = self._get_word_accuracy(
            target_codes,
            pred_codes,
        )

        self.log_dict(
            log_values,
            on_epoch=True,
            on_step=True,
            logger=True,
            prog_bar=True,
        )

        return {
            "loss": loss,
            "z": z_ctx_anc,
        }

    def training_step(
        self,
        batch,
        batch_idx,
    ):
        return self._shared_step(
            batch,
            mode="train",
        )

    def validation_step(
        self,
        batch,
        batch_idx,
    ):
        return self._shared_step(
            batch,
            mode="valid",
        )

    @torch.no_grad()
    def on_train_batch_end(
        self,
        outputs,
        batch,
        batch_idx,
    ):
        """Update prefix-mask and self-attention-drop schedules."""
        step = int(self.global_step)

        max_steps = int(
            getattr(
                self.trainer,
                "estimated_stepping_batches",
                0,
            )
            or 0
        )
        if max_steps <= 1:
            return

        max_epochs = max(
            1,
            int(
                getattr(
                    self.trainer,
                    "max_epochs",
                    1,
                )
                or 1
            ),
        )
        optimizer_steps_per_epoch = max(
            1,
            max_steps // max_epochs,
        )

        target_warmup_steps = (
            int(self.target_frac_warmup_epochs)
            * optimizer_steps_per_epoch
        )

        self.target_frac = (
            self._scheduled_value(
                step=step,
                max_steps=max_steps,
                warmup_steps=(
                    target_warmup_steps
                ),
                start=(
                    self.target_frac_start
                ),
                end=(
                    self.target_frac_end
                ),
                schedule=(
                    self.target_frac_schedule
                ),
            )
        )

        sa_warmup_steps = (
            int(self.sa_drop_warmup_epochs)
            * optimizer_steps_per_epoch
        )

        self.sa_drop_p = (
            self._scheduled_value(
                step=step,
                max_steps=max_steps,
                warmup_steps=(
                    sa_warmup_steps
                ),
                start=self.sa_drop_start,
                end=self.sa_drop_end,
                schedule=(
                    self.sa_drop_schedule
                ),
            )
        )

        self.decoder.set_self_attn_drop_p(
            self.sa_drop_p
        )

        self.log(
            "train_target_frac",
            self.target_frac,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "train_sa_drop_p",
            self.sa_drop_p,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
        )

    @staticmethod
    def _scheduled_value(
        step: int,
        max_steps: int,
        warmup_steps: int,
        start: float,
        end: float,
        schedule: str,
    ) -> float:
        if step < warmup_steps:
            return float(start)

        denom = max(
            1,
            max_steps - 1 - warmup_steps,
        )
        t = (
            step - warmup_steps
        ) / denom
        t = float(
            max(
                0.0,
                min(1.0, t),
            )
        )

        if schedule == "linear":
            return float(
                linear_anneal(
                    start,
                    end,
                    t,
                )
            )

        if schedule == "cosine":
            return float(
                cosine_anneal(
                    start,
                    end,
                    t,
                )
            )

        raise ValueError(
            f"Unknown schedule: {schedule}"
        )

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
        Stage-II deterministic inference for evaluation scripts.

        Returns a ragged list of 1D LongTensors, one per sample.

        If ``strip_special_tokens=True``, BOS/EOS/PAD/MASK are removed.
        Otherwise the sequence is returned through the first EOS, or through
        the first PAD when EOS is absent.
        """
        self.encoder.eval()
        self.decoder.eval()
        self.decoder.set_self_attn_drop_p(
            0.0
        )

        x = x.to(self.device)
        anc = self._prepare_inference_input(
            x
        )
        z_ctx_anc = self.encoder(anc)

        batch_size, t_max, _ = (
            z_ctx_anc.shape
        )

        cond_lens = torch.full(
            (batch_size,),
            t_max,
            dtype=torch.long,
            device=z_ctx_anc.device,
        )

        indices_beams = (
            beam_search_teacher_targets_single(
                decoder=self.decoder,
                xa=z_ctx_anc,
                cond_lens=cond_lens,
                bos_idx=(
                    self.decoder.BOS_IDX
                ),
                eos_idx=(
                    self.decoder.EOS_IDX
                ),
                pad_idx=(
                    self.decoder.PAD_IDX
                ),
                mask_idx=getattr(
                    self.decoder,
                    "MASK_IDX",
                    None,
                ),
                max_len=max_len,
                beam_size=beam_size,
                length_cap_ratio=(
                    length_cap_ratio
                ),
                min_cap=min_cap,
                rep_gamma=rep_gamma,
                alpha_len=(
                    length_norm_alpha
                ),
            )
        )

        # Beam 0 is the final selected sequence.
        indices = indices_beams[
            :, 0, :
        ]

        pad_idx = self.decoder.PAD_IDX
        bos_idx = self.decoder.BOS_IDX
        eos_idx = self.decoder.EOS_IDX
        mask_idx = getattr(
            self.decoder,
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
                eos_pos = (
                    seq.eq(eos_idx)
                    .nonzero(as_tuple=True)[0]
                )

                if eos_pos.numel() > 0:
                    seq = seq[
                        : int(
                            eos_pos[0].item()
                        )
                        + 1
                    ]
                else:
                    pad_pos = (
                        seq.eq(pad_idx)
                        .nonzero(
                            as_tuple=True
                        )[0]
                    )
                    if pad_pos.numel() > 0:
                        seq = seq[
                            : int(
                                pad_pos[0].item()
                            )
                        ]

            output.append(seq)

        return output

    def configure_optimizers(self):
        trainable_params = [
            p
            for p in self.parameters()
            if p.requires_grad
        ]

        if self.optim == "adam":
            optimizer = torch.optim.Adam(
                trainable_params,
                lr=self.lr,
            )

        elif self.optim == "adamw":
            if self.wt_decay <= 0:
                raise ValueError(
                    "AdamW requires a positive "
                    "weight decay"
                )

            optimizer = torch.optim.AdamW(
                trainable_params,
                lr=self.lr,
                weight_decay=self.wt_decay,
            )

        else:
            raise NotImplementedError(
                f"Unsupported optimizer: "
                f"{self.optim}"
            )

        if not self.lr_scheduler:
            return {
                "optimizer": optimizer
            }

        scheduler = (
            transformers
            .get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=int(
                    self.config.get(
                        "num_warmup_steps",
                        10000,
                    )
                ),
                num_training_steps=int(
                    self.config.get(
                        "num_training_steps",
                        300000,
                    )
                ),
            )
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": self.config.get(
                    "lr_scheduler_interval",
                    "step",
                ),
            },
        }

    @staticmethod
    def _get_word_accuracy(
        codes_anc: Sequence[
            torch.Tensor
        ],
        codes_pos: Sequence[
            torch.Tensor
        ],
    ) -> float:
        """Mean set-Jaccard overlap used only as a training log."""
        scores: List[float] = []

        for anc, pos in zip(
            codes_anc,
            codes_pos,
        ):
            anc_set = set(
                anc.detach()
                .cpu()
                .tolist()
            )
            pos_set = set(
                pos.detach()
                .cpu()
                .tolist()
            )

            union = anc_set | pos_set

            scores.append(
                1.0
                if not union
                else (
                    len(
                        anc_set
                        & pos_set
                    )
                    / len(union)
                )
            )

        return (
            float(sum(scores) / len(scores))
            if scores
            else 0.0
        )

    @classmethod
    def load_from_pretrained(
        cls,
        checkpoint,
        gpu_index=0,
    ):
        """Load a Stage-II checkpoint without assuming CUDA is available."""
        model = cls.load_from_checkpoint(
            checkpoint,
            map_location="cpu",
            strict=False,
        )
        model.eval()
        model.decoder.set_self_attn_drop_p(0.0)

        if torch.cuda.is_available():
            model = model.to(f"cuda:{gpu_index}")

        return model

