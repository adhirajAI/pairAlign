from typing import Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn


try:
    from torch.nn.functional import scaled_dot_product_attention

    SDPA_AVAILABLE = True
except (ImportError, RuntimeError, OSError):
    scaled_dot_product_attention = None
    SDPA_AVAILABLE = False


class LayerNorm(nn.LayerNorm):
    """LayerNorm computed in float32 and cast back to the input dtype."""

    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x.float()).to(x.dtype)


class Linear(nn.Linear):
    """Linear layer that follows the activation dtype."""

    def forward(self, x: Tensor) -> Tensor:
        return F.linear(
            x,
            self.weight.to(x.dtype),
            None if self.bias is None else self.bias.to(x.dtype),
        )


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention with an optional attention-observation path.

    ``return_attn=False`` is the ordinary training/evaluation path.

    ``return_attn=True`` does not introduce a second attention implementation
    for the model output. The hidden-state output is still produced by the same
    SDPA/fallback path used normally. Attention weights are recomputed only for
    observation and returned detached.
    """

    use_sdpa = True

    def __init__(self, n_state: int, n_head: int):
        super().__init__()

        if n_state % n_head != 0:
            raise ValueError(
                f"n_state ({n_state}) must be divisible by n_head ({n_head})"
            )

        self.n_head = int(n_head)
        self.query = Linear(n_state, n_state)
        self.key = Linear(n_state, n_state, bias=False)
        self.value = Linear(n_state, n_state)
        self.out = Linear(n_state, n_state)

    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        xa_kpm: Optional[Tensor] = None,
        kv_cache: Optional[dict] = None,
        return_attn: bool = False,
    ):
        q = self.query(x)

        if kv_cache is None or xa is None or self.key not in kv_cache:
            source = x if xa is None else xa
            k = self.key(source)
            v = self.value(source)
        else:
            k = kv_cache[self.key]
            v = kv_cache[self.value]

        attn_out, qk, attn = self.qkv_attention(
            q=q,
            k=k,
            v=v,
            mask=mask,
            xa_kpm=xa_kpm,
            return_attn=return_attn,
        )
        projected = self.out(attn_out)

        if return_attn:
            return projected, qk, attn

        return projected, qk

    @staticmethod
    def _additive_key_padding_mask(
        xa_kpm: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Convert (B,S) True=PAD mask to additive SDPA mask."""
        additive = torch.zeros(
            xa_kpm.size(0),
            1,
            1,
            xa_kpm.size(1),
            dtype=dtype,
            device=xa_kpm.device,
        )
        return additive.masked_fill(
            xa_kpm[:, None, None, :],
            float("-inf"),
        )

    @staticmethod
    def _explicit_attention_weights(
        q: Tensor,
        k: Tensor,
        scale: float,
        mask: Optional[Tensor],
        xa_kpm: Optional[Tensor],
    ) -> Tuple[Tensor, Tensor]:
        """
        Recompute attention logits/weights for observation only.

        q: (B,H,Tq,Dh)
        k: (B,H,Tk,Dh)
        """
        qk = (q * scale) @ (k * scale).transpose(-1, -2)

        if mask is not None:
            qk = qk + mask[: qk.size(-2), : qk.size(-1)]

        if xa_kpm is not None:
            if xa_kpm.shape != (qk.size(0), qk.size(-1)):
                raise ValueError(
                    "xa_kpm must have shape (B,Tkey); "
                    f"got {tuple(xa_kpm.shape)} for attention "
                    f"{tuple(qk.shape)}"
                )
            qk = qk.masked_fill(
                xa_kpm[:, None, None, :],
                float("-inf"),
            )

        weights = F.softmax(
            qk.float(),
            dim=-1,
        ).to(q.dtype)

        return qk.detach(), weights.detach()

    def qkv_attention(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        mask: Optional[Tensor] = None,
        xa_kpm: Optional[Tensor] = None,
        return_attn: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
        n_batch, n_ctx, n_state = q.shape
        head_dim = n_state // self.n_head
        scale = head_dim ** -0.25

        q = q.view(
            n_batch,
            n_ctx,
            self.n_head,
            head_dim,
        ).permute(0, 2, 1, 3)

        k = k.view(
            k.size(0),
            k.size(1),
            self.n_head,
            head_dim,
        ).permute(0, 2, 1, 3)

        v = v.view(
            v.size(0),
            v.size(1),
            self.n_head,
            head_dim,
        ).permute(0, 2, 1, 3)

        # Self-attention receives the causal mask. Cross-attention does not.
        is_causal = (
            mask is not None
            and xa_kpm is None
            and n_ctx > 1
        )

        qk_observed: Optional[Tensor] = None
        attn_observed: Optional[Tensor] = None

        if SDPA_AVAILABLE and MultiHeadAttention.use_sdpa:
            attn_mask = None
            if xa_kpm is not None:
                attn_mask = self._additive_key_padding_mask(
                    xa_kpm=xa_kpm,
                    dtype=q.dtype,
                )

            output = scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                is_causal=is_causal,
            )

            if return_attn:
                qk_observed, attn_observed = (
                    self._explicit_attention_weights(
                        q=q,
                        k=k,
                        scale=scale,
                        mask=mask,
                        xa_kpm=xa_kpm,
                    )
                )

        else:
            qk = (
                (q * scale)
                @ (k * scale).transpose(-1, -2)
            )

            if mask is not None:
                qk = qk + mask[
                    : qk.size(-2),
                    : qk.size(-1),
                ]

            if xa_kpm is not None:
                qk = qk.masked_fill(
                    xa_kpm[:, None, None, :],
                    float("-inf"),
                )

            weights = F.softmax(
                qk.float(),
                dim=-1,
            ).to(q.dtype)

            output = weights @ v
            qk_observed = qk.detach()

            if return_attn:
                attn_observed = weights.detach()

        output = (
            output.permute(0, 2, 1, 3)
            .flatten(start_dim=2)
        )

        return (
            output,
            qk_observed,
            attn_observed,
        )


class ResidualAttentionBlock(nn.Module):
    """Transformer decoder block with cross-attention."""

    def __init__(
        self,
        n_state: int,
        n_head: int,
        cross_attention: bool = False,
        self_attn_drop_p: float = 0.0,
    ):
        super().__init__()

        self.attn = MultiHeadAttention(
            n_state,
            n_head,
        )
        self.attn_ln = LayerNorm(
            n_state
        )

        self.cross_attn = (
            MultiHeadAttention(
                n_state,
                n_head,
            )
            if cross_attention
            else None
        )
        self.cross_attn_ln = (
            LayerNorm(n_state)
            if cross_attention
            else None
        )

        n_mlp = n_state * 4
        self.mlp = nn.Sequential(
            Linear(n_state, n_mlp),
            nn.GELU(),
            Linear(n_mlp, n_state),
        )
        self.mlp_ln = LayerNorm(
            n_state
        )

        self.dropout = nn.Dropout(0.1)
        self.self_attn_drop_p = float(
            self_attn_drop_p
        )

    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        xa_kpm: Optional[Tensor] = None,
        kv_cache: Optional[dict] = None,
        return_cross_attn: bool = False,
    ):
        self_attn_out = self.attn(
            self.attn_ln(x),
            mask=mask,
            kv_cache=kv_cache,
        )[0]
        self_attn_out = self.dropout(
            self_attn_out
        )

        if (
            self.training
            and self.self_attn_drop_p > 0.0
        ):
            keep = (
                torch.rand(
                    x.size(0),
                    1,
                    1,
                    device=x.device,
                )
                >= self.self_attn_drop_p
            ).to(self_attn_out.dtype)
            self_attn_out = (
                self_attn_out * keep
            )

        x = x + self_attn_out

        cross_weights = None

        if self.cross_attn is not None:
            if return_cross_attn:
                (
                    cross_attn_out,
                    _,
                    cross_weights,
                ) = self.cross_attn(
                    self.cross_attn_ln(x),
                    xa,
                    xa_kpm=xa_kpm,
                    kv_cache=kv_cache,
                    return_attn=True,
                )
            else:
                cross_attn_out = self.cross_attn(
                    self.cross_attn_ln(x),
                    xa,
                    xa_kpm=xa_kpm,
                    kv_cache=kv_cache,
                )[0]

            x = x + self.dropout(
                cross_attn_out
            )

        x = x + self.dropout(
            self.mlp(
                self.mlp_ln(x)
            )
        )

        if return_cross_attn:
            return x, cross_weights

        return x


class TextDecoder(nn.Module):
    """
    Autoregressive text/token decoder used by Stage II and Stage III.

    The default forward API is unchanged:

        logits = decoder(x=tokens, xa=encoder_states)

    Cross-attention can be observed without replacing the decoder:

        logits, cross_attn = decoder(
            x=tokens,
            xa=encoder_states,
            return_cross_attn=True,
        )

    ``cross_attn`` has shape (B,Tdec,Tenc) and is averaged over decoder
    layers and attention heads.

    Attention exposure adds no trainable parameters and does not change the
    normal hidden-state path. Historical checkpoints remain state-dict
    compatible.

    ``cond_gate`` is retained for historical checkpoint compatibility even
    though the current forward does not use it.
    """

    def __init__(
        self,
        n_vocab: int,
        n_ctx: int,
        n_state: int,
        n_head: int,
        n_layer: int,
        self_attn_drop_p: float = 0.0,
        n_cond_positions: int = 1,
    ):
        super().__init__()

        if n_vocab <= 0:
            raise ValueError(
                "n_vocab must be positive"
            )
        if n_ctx <= 0:
            raise ValueError(
                "n_ctx must be positive"
            )
        if n_state % n_head != 0:
            raise ValueError(
                "n_state must be divisible by n_head"
            )

        self.token_embedding = nn.Embedding(
            n_vocab + 4,
            n_state,
        )

        self.BOS_IDX = n_vocab
        self.EOS_IDX = n_vocab + 1
        self.PAD_IDX = n_vocab + 2
        self.MASK_IDX = n_vocab + 3

        self.positional_embedding = (
            nn.Parameter(
                torch.empty(
                    n_ctx,
                    n_state,
                )
            )
        )

        self.blocks: Iterable[
            ResidualAttentionBlock
        ] = nn.ModuleList(
            [
                ResidualAttentionBlock(
                    n_state=n_state,
                    n_head=n_head,
                    cross_attention=True,
                    self_attn_drop_p=(
                        self_attn_drop_p
                    ),
                )
                for _ in range(n_layer)
            ]
        )

        self.ln = LayerNorm(n_state)
        self.dropout = nn.Dropout(0.1)

        causal_mask = torch.full(
            (n_ctx, n_ctx),
            float("-inf"),
        ).triu_(1)
        self.register_buffer(
            "mask",
            causal_mask,
            persistent=False,
        )

        self.n_cond_positions = int(
            max(
                0,
                n_cond_positions,
            )
        )

        self.cond_proj = Linear(
            n_state,
            n_state,
        )

        # Retained to preserve historical checkpoint keys.
        self.cond_gate = Linear(
            2 * n_state,
            1,
        )

        nn.init.zeros_(
            self.cond_proj.weight
        )
        nn.init.zeros_(
            self.cond_proj.bias
        )

        nn.init.zeros_(
            self.cond_gate.weight
        )
        nn.init.constant_(
            self.cond_gate.bias,
            -2.0,
        )

    def set_self_attn_drop_p(
        self,
        p: float,
    ) -> None:
        p = float(
            max(
                0.0,
                min(
                    1.0,
                    p,
                ),
            )
        )
        for block in self.blocks:
            block.self_attn_drop_p = p

    @staticmethod
    def _pool_encoder(
        xa: Tensor,
        xa_kpm: Optional[
            Tensor
        ] = None,
    ) -> Tensor:
        if xa_kpm is None:
            return xa.mean(dim=1)

        if xa_kpm.shape != xa.shape[:2]:
            raise ValueError(
                "xa_kpm must have shape (B,Tenc); "
                f"got {tuple(xa_kpm.shape)} for xa "
                f"{tuple(xa.shape)}"
            )

        keep = (
            ~xa_kpm
        ).to(xa.dtype)

        denom = (
            keep.sum(
                dim=1,
                keepdim=True,
            )
            .clamp_min(1.0)
        )

        return (
            (
                xa
                * keep.unsqueeze(-1)
            )
            .sum(dim=1)
            / denom
        )

    def forward(
        self,
        x: Tensor,
        xa: Tensor,
        xa_kpm: Optional[
            Tensor
        ] = None,
        kv_cache: Optional[
            dict
        ] = None,
        return_cross_attn: bool = False,
    ):
        if x.dim() != 2:
            raise ValueError(
                "x must have shape (B,Tdec), "
                f"got {tuple(x.shape)}"
            )
        if xa.dim() != 3:
            raise ValueError(
                "xa must have shape (B,Tenc,D), "
                f"got {tuple(xa.shape)}"
            )
        if x.size(0) != xa.size(0):
            raise ValueError(
                "x and xa must have the same batch size"
            )

        offset = (
            next(
                iter(
                    kv_cache.values()
                )
            ).shape[1]
            if kv_cache
            else 0
        )

        if (
            offset + x.size(1)
            > self.positional_embedding.size(0)
        ):
            raise ValueError(
                "Decoder sequence exceeds positional "
                "embedding context."
            )

        tok = self.token_embedding(x)

        # Preserve the training implementation exactly: the encoder-summary
        # bias is injected only into the first n_cond_positions positions.
        if (
            self.n_cond_positions > 0
            and tok.size(1) > 0
        ):
            abs_start = offset
            abs_end = (
                offset + tok.size(1)
            )

            cond_start = max(
                abs_start,
                0,
            )
            cond_end = min(
                abs_end,
                self.n_cond_positions,
            )

            if cond_end > cond_start:
                local_start = (
                    cond_start
                    - abs_start
                )
                local_end = (
                    cond_end
                    - abs_start
                )

                encoder_summary = (
                    self._pool_encoder(
                        xa,
                        xa_kpm=xa_kpm,
                    )
                )
                encoder_summary = (
                    F.layer_norm(
                        encoder_summary,
                        (
                            encoder_summary
                            .shape[-1],
                        ),
                    )
                )
                cond_vec = self.cond_proj(
                    encoder_summary
                )

                tok[
                    :,
                    local_start:local_end,
                    :,
                ] = (
                    tok[
                        :,
                        local_start:local_end,
                        :,
                    ]
                    + cond_vec.unsqueeze(1)
                )

        hidden = (
            tok
            + self.positional_embedding[
                offset :
                offset + x.size(1)
            ]
        )
        hidden = self.dropout(
            hidden
        ).to(xa.dtype)

        per_layer_cross_attn: List[
            Tensor
        ] = []

        for block in self.blocks:
            if return_cross_attn:
                (
                    hidden,
                    cross_weights,
                ) = block(
                    hidden,
                    xa,
                    mask=self.mask,
                    xa_kpm=xa_kpm,
                    kv_cache=kv_cache,
                    return_cross_attn=True,
                )
                if cross_weights is not None:
                    per_layer_cross_attn.append(
                        cross_weights
                    )
            else:
                hidden = block(
                    hidden,
                    xa,
                    mask=self.mask,
                    xa_kpm=xa_kpm,
                    kv_cache=kv_cache,
                    return_cross_attn=False,
                )

        hidden = self.ln(
            hidden
        )

        logits = (
            hidden
            @ self.token_embedding.weight
            .to(hidden.dtype)
            .transpose(0, 1)
        ).float()

        if not return_cross_attn:
            return logits

        if not per_layer_cross_attn:
            cross_attn = hidden.new_zeros(
                (
                    x.size(0),
                    x.size(1),
                    xa.size(1),
                )
            )
        else:
            # (L,B,H,Tdec,Tenc) -> (B,Tdec,Tenc)
            cross_attn = (
                torch.stack(
                    per_layer_cross_attn,
                    dim=0,
                )
                .mean(dim=0)
                .mean(dim=1)
            )

        return logits, cross_attn
