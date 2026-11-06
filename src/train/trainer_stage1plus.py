from collections import Counter
from copy import deepcopy
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
import transformers
from tslearn.metrics import dtw_path_from_metric
from vector_quantize_pytorch import VectorQuantize

from models import MambaEncoder, ModelArgs
from utils import ctc_loss


def attach_vq_distance_exposer(
    vq_layer: VectorQuantize,
    *,
    detach: bool = False,
):
    """
    Expose the codebook score tensor produced inside ``VectorQuantize``.

    The custom no-blank CTC objective needs the per-frame codebook scores, but
    ``VectorQuantize.forward`` does not return them. A forward hook on its
    codebook stores the third codebook output on ``vq_layer.last_distances``.

    ``detach=False`` is required during Stage-I+ training so CTC gradients can
    flow through the score tensor.
    """
    if not hasattr(vq_layer, "_codebook"):
        raise AttributeError(
            "VectorQuantize does not expose the expected '_codebook' module. "
            "The Stage-I+ CTC hook must be updated for this library version."
        )

    vq_layer.last_distances = None

    def _capture_distances(module, inputs, outputs):
        del module, inputs

        if not isinstance(outputs, (tuple, list)) or len(outputs) < 3:
            raise RuntimeError(
                "Expected the VQ codebook forward to return at least three "
                "values: (quantized, indices, distances)."
            )

        distances = outputs[2]
        vq_layer.last_distances = (
            distances.detach()
            if detach
            else distances
        )

    return vq_layer._codebook.register_forward_hook(
        _capture_distances
    )


def vq_distances_to_no_blank_ctc_log_probs(
    distances: torch.Tensor,
    blank_id: int,
) -> torch.Tensor:
    """
    Convert VQ codebook scores to custom-CTC log probabilities.

    Args:
        distances:
            Score tensor with shape (B,T,K), or (1,B,T,K) for a single
            codebook head.
        blank_id:
            Blank symbol index in the returned vocabulary. For Stage I+, this
            is normally K, so the blank is appended after the K real codes.

    Returns:
        Log probabilities with shape (T,B,V). The blank column is assigned the
        minimum finite value of the dtype, making blank emission effectively
        impossible while remaining numerically compatible with ``utils.ctc_loss``.

    Note:
        This function intentionally preserves the score-to-log-probability
        convention used by the original Stage-I+ implementation:
        ``log_softmax(distances, dim=-1)``.
    """
    if distances is None:
        raise RuntimeError(
            "No VQ distance tensor was captured. Ensure the distance hook is "
            "attached before calling the VQ layer."
        )

    if distances.dim() == 4:
        if distances.size(0) != 1:
            raise ValueError(
                "Multiple VQ heads are not supported by the Stage-I+ CTC "
                f"conversion; got shape {tuple(distances.shape)}."
            )
        distances = distances[0]

    if distances.dim() != 3:
        raise ValueError(
            "Expected VQ distances with shape (B,T,K) or (1,B,T,K), "
            f"got {tuple(distances.shape)}."
        )

    batch_size, time_steps, codebook_size = distances.shape

    if blank_id < 0:
        raise ValueError(
            f"blank_id must be non-negative, got {blank_id}"
        )

    real_log_probs = F.log_softmax(
        distances,
        dim=-1,
    ).transpose(0, 1).contiguous()

    # Use a finite sentinel rather than -inf because the project-specific CTC
    # dynamic program is the authoritative implementation for Stage I+.
    minimum_value = torch.finfo(real_log_probs.dtype).min
    blank_column = real_log_probs.new_full(
        (time_steps, batch_size, 1),
        minimum_value,
    )

    if blank_id == codebook_size:
        return torch.cat(
            (real_log_probs, blank_column),
            dim=-1,
        )

    if blank_id < codebook_size:
        return torch.cat(
            (
                real_log_probs[..., :blank_id],
                blank_column,
                real_log_probs[..., blank_id:],
            ),
            dim=-1,
        )

    padding = real_log_probs.new_full(
        (
            time_steps,
            batch_size,
            blank_id - codebook_size,
        ),
        minimum_value,
    )

    return torch.cat(
        (
            real_log_probs,
            padding,
            blank_column,
        ),
        dim=-1,
    )


class AudioTokenizer(L.LightningModule):
    """
    Stage-I geometric audio tokenizer with optional Stage-I+ training.

    Stage I optimizes the original VQ + DTW-aligned frame-level contrastive
    objective.

    Stage I+ keeps the same architecture and the same Stage-I losses, and adds
    a symmetric no-blank CTC consistency objective. Stage I+ is required to be
    initialized from a pretrained Stage-I checkpoint using
    ``initialize_stage1_plus`` or ``from_training_config``.

    No DTW framewise pairwise-prediction auxiliary loss is implemented here.
    """

    @staticmethod
    def _checkpoint_safe_config(
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Remove filesystem checkpoint paths before storing hyperparameters.

        The live ``self.config`` remains unchanged. This only sanitizes the
        configuration serialized into Lightning checkpoints.
        """
        safe_config = deepcopy(config)

        checkpoints = safe_config.get("checkpoints")
        if isinstance(checkpoints, dict):
            safe_config["checkpoints"] = {
                key: value
                for key, value in checkpoints.items()
                if key not in {
                    "stage1",
                    "stage1_plus",
                }
            }

        stage1_plus_options = safe_config.get(
            "stage1_plus"
        )
        if isinstance(stage1_plus_options, dict):
            stage1_plus_options.pop(
                "pretrained_checkpoint",
                None,
            )

        return safe_config

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config

        mamba_args = ModelArgs(
            d_model=config["d_model"],
            d_project=config["d_project"],
            d_state=config["d_state"],
            d_conv=config["d_conv"],
            expand=config["expand"],
            n_layer=config["n_layer"],
            headdim=config["headdim"],
            block_type=config["block_type"],
        )
        self.encoder = MambaEncoder(mamba_args)

        self.vq_layer = VectorQuantize(
            dim=config["codeword_dims"],
            codebook_size=config["num_codewords"],
            decay=config["ema_decay"],
            commitment_weight=config["beta"],
            kmeans_init=True,
            kmeans_iters=10,
            use_cosine_sim=True,
            threshold_ema_dead_code=2,
            orthogonal_reg_weight=config["ortho_weight"],
            orthogonal_reg_active_codes_only=False,
        )

        self.temperature = float(config["temperature"])
        self.cost_factors = config["cost_factors"]
        self.optim = config["optimizer"]
        self.lr = float(config["lr"])
        self.wt_decay = float(config["wt_decay"])
        self.lr_scheduler = bool(config["lr_scheduler"])

        # Defaults preserve the original Stage-I training behavior.
        self.max_contrastive_items = int(
            config.get("max_contrastive_items", 30)
        )
        self.num_negative_frames = int(
            config.get("num_negative_frames", 30)
        )

        stage1_plus_config = config.get("stage1_plus", {})
        if stage1_plus_config is None:
            stage1_plus_config = {}
        if not isinstance(stage1_plus_config, dict):
            raise TypeError("config['stage1_plus'] must be a dictionary")

        self.use_stage1_plus = bool(
            stage1_plus_config.get("enabled", False)
        )

        # A fresh Stage-I+ object is intentionally NOT trainable until its
        # weights have been initialized from a pretrained Stage-I checkpoint.
        # This flag is not a model parameter and therefore does not alter
        # Stage-I / Stage-I+ state-dict compatibility.
        self._stage1_plus_initialized_from_stage1 = (
            not self.use_stage1_plus
        )

        self.ctc_balance_alpha = float(
            stage1_plus_config.get("ctc_balance_alpha", 0.5)
        )
        self.ctc_eps = float(
            stage1_plus_config.get("ctc_eps", 1e-8)
        )

        self.num_codewords = int(config["num_codewords"])
        self.ctc_blank_id = self.num_codewords

        self._validate_configuration()

        # The hook has no parameters and therefore does not alter checkpoint
        # compatibility between Stage I and Stage I+.
        self._vq_distance_hook_handle = None
        if self.use_stage1_plus:
            self._vq_distance_hook_handle = attach_vq_distance_exposer(
                self.vq_layer,
                detach=False,
            )

        self.val_codes_cache: List[List[int]] = []

        self.save_hyperparameters(
            {
                "config": self._checkpoint_safe_config(
                    config
                )
            }
        )

    @staticmethod
    def _resolve_stage1_checkpoint(
        config: Dict[str, Any],
    ) -> str:
        """
        Resolve the pretrained Stage-I checkpoint required by Stage I+.

        Accepted configuration locations, in precedence order:
          1. stage1_plus.pretrained_checkpoint
          2. checkpoints.stage1

        Stage I+ cannot be initialized without an existing Stage-I checkpoint.
        """
        stage1_plus_config = config.get("stage1_plus", {}) or {}
        checkpoints = config.get("checkpoints", {}) or {}

        if not isinstance(stage1_plus_config, dict):
            raise TypeError(
                "config['stage1_plus'] must be a dictionary"
            )
        if not isinstance(checkpoints, dict):
            raise TypeError(
                "config['checkpoints'] must be a dictionary"
            )

        checkpoint_path = (
            stage1_plus_config.get("pretrained_checkpoint")
            or checkpoints.get("stage1")
        )

        if not checkpoint_path:
            raise ValueError(
                "Stage I+ requires a pretrained Stage-I checkpoint. "
                "Set either stage1_plus.pretrained_checkpoint or "
                "checkpoints.stage1."
            )

        checkpoint_path = os.path.expanduser(
            str(checkpoint_path)
        )

        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(
                "Stage I+ requires an existing pretrained Stage-I checkpoint, "
                f"but no file was found at: {checkpoint_path}"
            )

        return checkpoint_path

    @staticmethod
    def _checkpoint_was_stage1_plus(
        checkpoint: Dict[str, Any],
    ) -> bool:
        """
        Determine whether a Lightning checkpoint represents a trained
        Stage-I+ model.

        New checkpoints carry an explicit provenance marker. The
        hyperparameter fallback supports checkpoints produced before that
        marker was added.
        """
        if bool(
            checkpoint.get(
                "stage1_plus_initialized_from_stage1",
                False,
            )
        ):
            return True

        hyper_parameters = checkpoint.get(
            "hyper_parameters",
            {},
        )
        if not isinstance(hyper_parameters, dict):
            return False

        saved_config = hyper_parameters.get(
            "config",
            {},
        )
        if not isinstance(saved_config, dict):
            return False

        stage1_plus_config = saved_config.get(
            "stage1_plus",
            {},
        ) or {}

        return (
            isinstance(stage1_plus_config, dict)
            and bool(
                stage1_plus_config.get(
                    "enabled",
                    False,
                )
            )
        )

    def _assert_stage1_plus_training_ready(self) -> None:
        """
        Prevent Stage I+ from being trained from random initialization.

        Ordinary Stage I is unaffected. Stage I+ must either:
          * be created through ``initialize_stage1_plus`` /
            ``from_training_config`` using a pretrained Stage-I checkpoint, or
          * be resumed from an already-trained Stage-I+ checkpoint.
        """
        if (
            self.use_stage1_plus
            and not self._stage1_plus_initialized_from_stage1
        ):
            raise RuntimeError(
                "Stage I+ cannot be trained from random initialization. "
                "Initialize it from a pretrained Stage-I checkpoint with "
                "AudioTokenizer.from_training_config(...) or "
                "AudioTokenizer.initialize_stage1_plus(...)."
            )

    def on_save_checkpoint(
        self,
        checkpoint: Dict[str, Any],
    ) -> None:
        """
        Persist Stage-I+ initialization provenance without storing the source
        checkpoint path.
        """
        checkpoint[
            "stage1_plus_initialized_from_stage1"
        ] = bool(
            self.use_stage1_plus
            and self._stage1_plus_initialized_from_stage1
        )

    def on_load_checkpoint(
        self,
        checkpoint: Dict[str, Any],
    ) -> None:
        """
        Permit normal resume/evaluation of an already-trained Stage-I+
        checkpoint.
        """
        if (
            self.use_stage1_plus
            and self._checkpoint_was_stage1_plus(
                checkpoint
            )
        ):
            self._stage1_plus_initialized_from_stage1 = True

    def _validate_configuration(self) -> None:
        if self.temperature <= 0.0:
            raise ValueError(
                f"temperature must be > 0, got {self.temperature}"
            )
        if self.max_contrastive_items < 1:
            raise ValueError(
                "max_contrastive_items must be >= 1"
            )
        if self.num_negative_frames < 1:
            raise ValueError(
                "num_negative_frames must be >= 1"
            )
        if self.num_codewords < 1:
            raise ValueError(
                "num_codewords must be >= 1"
            )
        if self.use_stage1_plus and self.ctc_balance_alpha < 0.0:
            raise ValueError(
                "stage1_plus.ctc_balance_alpha must be >= 0"
            )
        if self.ctc_eps <= 0.0:
            raise ValueError(
                "stage1_plus.ctc_eps must be > 0"
            )

        for required_cost in ("vq_loss", "triplet_loss"):
            if required_cost not in self.cost_factors:
                raise KeyError(
                    f"cost_factors is missing '{required_cost}'"
                )

    @staticmethod
    def deduplicate(seq: torch.Tensor) -> torch.Tensor:
        """Collapse consecutive duplicate token IDs."""
        if seq.numel() <= 1:
            return seq

        keep = torch.ones_like(
            seq,
            dtype=torch.bool,
        )
        keep[1:] = seq[1:].ne(seq[:-1])
        return seq[keep]

    def _encode_and_quantize(
        self,
        features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if features.dim() != 3:
            raise ValueError(
                "Expected encoder input with shape (B,T,F), "
                f"got {tuple(features.shape)}"
            )

        z_ctx = self.encoder(features)
        z_ctx_quant, indices, _ = self.vq_layer(z_ctx)

        # In inference/evaluation utilities the captured VQ scores are not
        # needed. Clear the reference so Stage-I+ analysis calls do not retain
        # an unnecessary score tensor between batches.
        if self.use_stage1_plus:
            self.vq_layer.last_distances = None

        return z_ctx, z_ctx_quant, indices

    def _prepare_feature_input(
        self,
        features: torch.Tensor,
    ) -> torch.Tensor:
        """Convert common feature layouts to encoder layout (B,T,F)."""
        if not torch.is_tensor(features):
            raise TypeError(
                f"features must be a torch.Tensor, got {type(features)}"
            )

        n_mels = int(self.config["n_mels"])

        if features.dim() == 4:
            if features.size(1) != 1:
                raise ValueError(
                    "Expected 4D input with shape (B,1,F,T), "
                    f"got {tuple(features.shape)}"
                )
            features = features.squeeze(1)

        if features.dim() == 2:
            features = features.unsqueeze(0)

        if features.dim() != 3:
            raise ValueError(
                "Expected (B,F,T), (B,T,F), (F,T), (T,F), or "
                f"(B,1,F,T); got {tuple(features.shape)}"
            )

        if (
            features.size(1) == n_mels
            and features.size(2) != n_mels
        ):
            return features.permute(0, 2, 1).contiguous()

        if features.size(2) == n_mels:
            return features.contiguous()

        raise ValueError(
            "Could not identify the mel-feature dimension. "
            f"Expected n_mels={n_mels}, got shape {tuple(features.shape)}"
        )

    def _consume_vq_distances(self) -> torch.Tensor:
        """Read and clear the most recently captured VQ score tensor."""
        distances = getattr(
            self.vq_layer,
            "last_distances",
            None,
        )

        if distances is None:
            raise RuntimeError(
                "Stage-I+ expected VQ distances, but none were captured."
            )

        self.vq_layer.last_distances = None
        return distances

    def _symmetric_no_blank_ctc_loss(
        self,
        anchor_log_probs_btv: torch.Tensor,
        positive_log_probs_btv: torch.Tensor,
        anchor_indices: torch.LongTensor,
        positive_indices: torch.LongTensor,
        max_items: int,
    ) -> Tuple[torch.Tensor, int]:
        """
        Compute the Stage-I+ symmetric cross-view no-blank CTC loss.

        Anchor emissions predict the deduplicated positive token sequence, and
        positive emissions predict the deduplicated anchor token sequence.
        The project-specific ``utils.ctc_loss`` implementation is used for both
        directions.
        """
        if anchor_log_probs_btv.dim() != 3:
            raise ValueError(
                "anchor_log_probs_btv must have shape (B,T,V)"
            )
        if positive_log_probs_btv.dim() != 3:
            raise ValueError(
                "positive_log_probs_btv must have shape (B,T,V)"
            )

        batch_size = anchor_log_probs_btv.size(0)
        items_to_score = min(
            batch_size,
            max_items,
        )

        ctc_sum = anchor_log_probs_btv.new_zeros(())
        valid_items = 0

        for sample_index in range(items_to_score):
            anchor_log_probs = anchor_log_probs_btv[
                sample_index
            ].unsqueeze(1)
            positive_log_probs = positive_log_probs_btv[
                sample_index
            ].unsqueeze(1)

            anchor_target = self.deduplicate(
                anchor_indices[sample_index]
            ).unsqueeze(0).to(
                device=positive_log_probs.device,
                dtype=torch.long,
            )
            positive_target = self.deduplicate(
                positive_indices[sample_index]
            ).unsqueeze(0).to(
                device=anchor_log_probs.device,
                dtype=torch.long,
            )

            anchor_input_lengths = torch.tensor(
                [anchor_log_probs.size(0)],
                device=anchor_log_probs.device,
                dtype=torch.long,
            )
            positive_input_lengths = torch.tensor(
                [positive_log_probs.size(0)],
                device=positive_log_probs.device,
                dtype=torch.long,
            )

            positive_target_lengths = torch.tensor(
                [positive_target.size(1)],
                device=anchor_log_probs.device,
                dtype=torch.long,
            )
            anchor_target_lengths = torch.tensor(
                [anchor_target.size(1)],
                device=positive_log_probs.device,
                dtype=torch.long,
            )

            # Anchor emissions -> positive symbolic target.
            ctc_anchor = ctc_loss(
                log_probs=anchor_log_probs,
                targets=positive_target,
                input_lengths=anchor_input_lengths,
                target_lengths=positive_target_lengths,
                blank=self.ctc_blank_id,
                reduction="none",
            ).mean()

            # Positive emissions -> anchor symbolic target.
            ctc_positive = ctc_loss(
                log_probs=positive_log_probs,
                targets=anchor_target,
                input_lengths=positive_input_lengths,
                target_lengths=anchor_target_lengths,
                blank=self.ctc_blank_id,
                reduction="none",
            ).mean()

            pair_ctc = 0.5 * (
                ctc_anchor
                + ctc_positive
            )

            if torch.isfinite(pair_ctc):
                ctc_sum = ctc_sum + pair_ctc
                valid_items += 1

        if valid_items == 0:
            return ctc_sum, 0

        return ctc_sum / valid_items, valid_items

    def _shared_step(
        self,
        batch,
        mode: str,
    ):
        if mode not in {"train", "valid"}:
            raise ValueError(
                f"Unsupported mode: {mode}"
            )

        if mode == "train":
            self._assert_stage1_plus_training_ready()

        if len(batch) < 2:
            raise ValueError(
                "Stage-I expects paired anchor/positive inputs."
            )

        anchor = (
            batch[0]
            .squeeze(1)
            .permute(0, 2, 1)
            .contiguous()
        )
        positive = (
            batch[1]
            .squeeze(1)
            .permute(0, 2, 1)
            .contiguous()
        )

        if anchor.size(0) != positive.size(0):
            raise ValueError(
                "Anchor and positive batches must have the same size."
            )

        batch_size = anchor.size(0)

        z_anchor = self.encoder(anchor)
        z_positive = self.encoder(positive)

        (
            z_anchor_quant,
            anchor_indices,
            anchor_vq_loss,
        ) = self.vq_layer(z_anchor)

        anchor_distances = None
        if self.use_stage1_plus and mode == "train":
            anchor_distances = self._consume_vq_distances()

        (
            z_positive_quant,
            positive_indices,
            positive_vq_loss,
        ) = self.vq_layer(z_positive)

        positive_distances = None
        if self.use_stage1_plus and mode == "train":
            positive_distances = self._consume_vq_distances()

        log_values: Dict[str, Any] = {}

        codebook_loss = (
            0.5
            * (
                anchor_vq_loss
                + positive_vq_loss
            )
            * float(
                self.cost_factors["vq_loss"]
            )
        )
        loss = codebook_loss
        log_values[
            f"{mode}_codebook_loss"
        ] = codebook_loss

        if mode == "train":
            if batch_size < 2:
                raise ValueError(
                    "Stage-I contrastive training requires batch_size >= 2."
                )

            contrastive_sum = z_anchor.new_zeros(())
            total_aligned_frames = 0
            token_agreements: List[float] = []

            items_to_score = min(
                batch_size,
                self.max_contrastive_items,
            )

            for sample_index in range(items_to_score):
                negative_index = self._sample_negative_index(
                    batch_size,
                    sample_index,
                )

                triplets = self._get_triplet_indices(
                    anchor[sample_index],
                    positive[sample_index],
                    positive[negative_index],
                    num_negative_frames=self.num_negative_frames,
                )

                if triplets.numel() == 0:
                    continue

                anchor_tokens = anchor_indices[
                    sample_index,
                    triplets[:, 0],
                ]
                positive_tokens = positive_indices[
                    sample_index,
                    triplets[:, 1],
                ]

                token_agreements.append(
                    anchor_tokens.eq(
                        positive_tokens
                    ).float().mean().item()
                )

                anchor_repr = z_anchor[sample_index]
                positive_repr = z_positive[sample_index]
                negative_repr = z_positive[negative_index]

                positive_similarity = torch.einsum(
                    "nd,nd->n",
                    anchor_repr[triplets[:, 0]],
                    positive_repr[triplets[:, 1]],
                )
                negative_similarity = torch.einsum(
                    "nd,nkd->nk",
                    anchor_repr[triplets[:, 0]],
                    negative_repr[triplets[:, 2:]],
                )

                logits = torch.cat(
                    (
                        positive_similarity.unsqueeze(1),
                        negative_similarity,
                    ),
                    dim=1,
                ) / self.temperature

                contrastive_sum = (
                    contrastive_sum
                    + F.cross_entropy(
                        logits,
                        torch.zeros(
                            logits.size(0),
                            dtype=torch.long,
                            device=logits.device,
                        ),
                        reduction="sum",
                    )
                )
                total_aligned_frames += int(
                    triplets.size(0)
                )

            if total_aligned_frames == 0:
                raise RuntimeError(
                    "No aligned frames were produced for the Stage-I "
                    "contrastive objective."
                )

            contrastive_loss = (
                float(
                    self.cost_factors["triplet_loss"]
                )
                * contrastive_sum
                / total_aligned_frames
            )
            loss = loss + contrastive_loss

            log_values[
                f"{mode}_anc_pos_token_acc"
            ] = (
                float(np.mean(token_agreements))
                if token_agreements
                else 0.0
            )
            log_values[
                f"{mode}_frames"
            ] = float(total_aligned_frames)
            log_values[
                f"{mode}_contra_loss"
            ] = contrastive_loss

            if self.use_stage1_plus:
                anchor_ctc_log_probs = (
                    vq_distances_to_no_blank_ctc_log_probs(
                        anchor_distances,
                        blank_id=self.ctc_blank_id,
                    )
                    .transpose(0, 1)
                    .contiguous()
                )
                positive_ctc_log_probs = (
                    vq_distances_to_no_blank_ctc_log_probs(
                        positive_distances,
                        blank_id=self.ctc_blank_id,
                    )
                    .transpose(0, 1)
                    .contiguous()
                )

                raw_ctc_loss, valid_ctc_items = (
                    self._symmetric_no_blank_ctc_loss(
                        anchor_log_probs_btv=anchor_ctc_log_probs,
                        positive_log_probs_btv=positive_ctc_log_probs,
                        anchor_indices=anchor_indices,
                        positive_indices=positive_indices,
                        max_items=items_to_score,
                    )
                )

                if valid_ctc_items > 0:
                    ctc_scale = (
                        self.ctc_balance_alpha
                        * (
                            contrastive_loss.detach()
                            / (
                                raw_ctc_loss.detach()
                                + self.ctc_eps
                            )
                        )
                    )
                    ctc_auxiliary_loss = (
                        ctc_scale
                        * raw_ctc_loss
                    )
                    loss = loss + ctc_auxiliary_loss
                else:
                    ctc_scale = raw_ctc_loss.new_zeros(())
                    ctc_auxiliary_loss = raw_ctc_loss.new_zeros(())

                log_values[
                    f"{mode}_ctc_raw_loss"
                ] = raw_ctc_loss
                log_values[
                    f"{mode}_ctc_scale"
                ] = ctc_scale
                log_values[
                    f"{mode}_ctc_loss"
                ] = ctc_auxiliary_loss
                log_values[
                    f"{mode}_ctc_valid_items"
                ] = float(valid_ctc_items)

        # Compatibility attributes used by existing callbacks/analysis code.
        self.z_anc = z_anchor[0]
        self.z_pos = z_positive[0]
        self.z_anc_quant = z_anchor_quant[0]
        self.z_pos_quant = z_positive_quant[0]

        anchor_codes = [
            self.deduplicate(seq)
            for seq in anchor_indices
        ]
        positive_codes = [
            self.deduplicate(seq)
            for seq in positive_indices
        ]

        log_values[
            f"{mode}_acc"
        ] = self._get_normalized_edit_similarity(
            anchor_codes,
            positive_codes,
        )

        if mode == "valid":
            self.val_codes_cache.extend(
                seq.detach().cpu().tolist()
                for seq in anchor_codes
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
            "z": z_anchor,
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
    def on_validation_epoch_end(self) -> None:
        sequences = self.val_codes_cache
        num_sequences = len(sequences)

        if num_sequences == 0:
            self.val_codes_cache.clear()
            return

        sequence_counts = Counter(
            tuple(sequence)
            for sequence in sequences
        )
        num_unique = len(sequence_counts)
        collision_rate = (
            num_sequences - num_unique
        ) / num_sequences
        top_sequence_fraction = (
            sequence_counts.most_common(1)[0][1]
            / num_sequences
        )

        token_counts = Counter()
        total_tokens = 0
        for sequence in sequences:
            token_counts.update(sequence)
            total_tokens += len(sequence)

        used_tokens = len(token_counts)
        top_token_fraction = (
            token_counts.most_common(1)[0][1]
            / total_tokens
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

    @staticmethod
    def _sample_negative_index(
        batch_size: int,
        excluded_index: int,
    ) -> int:
        if batch_size < 2:
            raise ValueError(
                "Negative sampling requires batch_size >= 2."
            )
        if not 0 <= excluded_index < batch_size:
            raise ValueError(
                f"excluded_index={excluded_index} is outside "
                f"batch_size={batch_size}"
            )

        while True:
            sampled = int(
                torch.randint(
                    batch_size,
                    (1,),
                ).item()
            )
            if sampled != excluded_index:
                return sampled

    @staticmethod
    def _get_triplet_indices(
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
        num_negative_frames: int,
    ) -> torch.LongTensor:
        """
        DTW-align anchor/positive frames and attach K random negative-frame
        indices from another positive example.
        """
        if (
            anchor.dim() != 2
            or positive.dim() != 2
            or negative.dim() != 2
        ):
            raise ValueError(
                "anchor, positive, and negative must each have shape (T,D)."
            )
        if num_negative_frames < 1:
            raise ValueError(
                "num_negative_frames must be >= 1"
            )
        if negative.size(0) < 1:
            raise ValueError(
                "negative sequence contains no frames"
            )

        anchor_norm = (
            anchor
            / torch.linalg.norm(
                anchor,
                dim=1,
                keepdim=True,
            ).clamp_min(1e-12)
        )
        positive_norm = (
            positive
            / torch.linalg.norm(
                positive,
                dim=1,
                keepdim=True,
            ).clamp_min(1e-12)
        )

        path, _ = dtw_path_from_metric(
            anchor_norm.detach().cpu().numpy(),
            positive_norm.detach().cpu().numpy(),
            metric="cosine",
        )

        alignment: Dict[int, List[int]] = {}
        for anchor_index, positive_index in path:
            alignment.setdefault(
                int(anchor_index),
                [],
            ).append(
                int(positive_index)
            )

        rows = []
        for (
            anchor_index,
            candidate_positive_indices,
        ) in alignment.items():
            candidates = torch.tensor(
                candidate_positive_indices,
                dtype=torch.long,
                device=positive_norm.device,
            )
            similarities = (
                anchor_norm[anchor_index]
                .unsqueeze(0)
                @ positive_norm[candidates].T
            ).squeeze(0)
            positive_index = int(
                candidates[
                    similarities.argmax()
                ].item()
            )

            negative_indices = np.random.choice(
                negative.size(0),
                num_negative_frames,
                replace=True,
            )
            rows.append(
                [
                    anchor_index,
                    positive_index,
                    *negative_indices.tolist(),
                ]
            )

        if not rows:
            return torch.empty(
                (0, 2 + num_negative_frames),
                dtype=torch.long,
                device=anchor.device,
            )

        return torch.tensor(
            rows,
            dtype=torch.long,
            device=anchor.device,
        )

    @staticmethod
    def _edit_distance(
        first: Sequence[int],
        second: Sequence[int],
    ) -> int:
        """Standard Levenshtein distance."""
        n = len(first)
        m = len(second)

        if n == 0:
            return m
        if m == 0:
            return n

        previous = list(range(m + 1))

        for i in range(1, n + 1):
            current = [i] + [0] * m

            for j in range(1, m + 1):
                substitution_cost = (
                    0
                    if first[i - 1] == second[j - 1]
                    else 1
                )
                current[j] = min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + substitution_cost,
                )

            previous = current

        return previous[m]

    @classmethod
    def _get_normalized_edit_similarity(
        cls,
        anchor_codes: Sequence[torch.Tensor],
        positive_codes: Sequence[torch.Tensor],
    ) -> float:
        if len(anchor_codes) != len(positive_codes):
            raise ValueError(
                "anchor_codes and positive_codes must have the same length."
            )

        similarities = []

        for anchor_seq, positive_seq in zip(
            anchor_codes,
            positive_codes,
        ):
            if isinstance(anchor_seq, torch.Tensor):
                anchor_seq = (
                    anchor_seq.detach().cpu().tolist()
                )
            if isinstance(positive_seq, torch.Tensor):
                positive_seq = (
                    positive_seq.detach().cpu().tolist()
                )

            denominator = max(
                len(anchor_seq),
                len(positive_seq),
            )

            if denominator == 0:
                similarities.append(1.0)
                continue

            distance = cls._edit_distance(
                anchor_seq,
                positive_seq,
            )
            similarities.append(
                1.0 - distance / denominator
            )

        return (
            float(np.mean(similarities))
            if similarities
            else 0.0
        )

    @torch.no_grad()
    def tokenize(
        self,
        features: torch.Tensor,
        deduplicate: bool = True,
        return_tensors: bool = False,
        device: Optional[torch.device] = None,
    ):
        """Tokenize acoustic features using the Stage-I encoder and VQ."""
        self.eval()

        if device is None:
            device = next(
                self.parameters()
            ).device
        else:
            device = torch.device(device)

        encoder_input = (
            self._prepare_feature_input(
                features
            )
            .to(device)
        )
        _, _, indices = self._encode_and_quantize(
            encoder_input
        )

        output = []
        for sequence in indices.detach().cpu():
            sequence = sequence.reshape(-1)

            if deduplicate:
                sequence = self.deduplicate(
                    sequence
                )

            if return_tensors:
                output.append(
                    sequence.clone()
                )
            else:
                output.append(
                    [
                        int(token)
                        for token in sequence.tolist()
                    ]
                )

        return output

    @torch.no_grad()
    def predict_step(
        self,
        batch,
        batch_idx,
    ):
        """Lightning prediction hook; batch must be in (B,T,F) layout."""
        return self._encode_and_quantize(
            batch
        )

    def configure_optimizers(self):
        self._assert_stage1_plus_training_ready()

        if self.optim == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.lr,
            )

        elif self.optim == "adamw":
            if self.wt_decay <= 0.0:
                raise ValueError(
                    "AdamW requires a positive weight decay."
                )

            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.lr,
                weight_decay=self.wt_decay,
            )

        else:
            raise NotImplementedError(
                f"Unsupported optimizer: {self.optim}"
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
                        10,
                    )
                ),
                num_training_steps=int(
                    self.config.get(
                        "num_training_steps",
                        300,
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
                    "epoch",
                ),
            },
        }

    @classmethod
    def initialize_stage1_plus(
        cls,
        config: Dict[str, Any],
        stage1_checkpoint: Optional[str] = None,
        gpu_index: Optional[int] = None,
    ):
        """
        Initialize Stage I+ strictly from a pretrained Stage-I checkpoint.

        Stage I+ is never allowed to begin from random initialization.

        The source checkpoint must be a Stage-I checkpoint, not an already
        Stage-I+ checkpoint. Only model weights are transferred; optimizer and
        scheduler state are intentionally not inherited.

        If ``stage1_checkpoint`` is omitted, the path is resolved from:
          1. stage1_plus.pretrained_checkpoint
          2. checkpoints.stage1
        """
        if stage1_checkpoint is None:
            stage1_checkpoint = cls._resolve_stage1_checkpoint(
                config
            )
        else:
            stage1_checkpoint = os.path.expanduser(
                str(stage1_checkpoint)
            )
            if not os.path.isfile(stage1_checkpoint):
                raise FileNotFoundError(
                    "Stage I+ requires an existing pretrained Stage-I "
                    f"checkpoint, but no file was found at: {stage1_checkpoint}"
                )

        checkpoint = torch.load(
            stage1_checkpoint,
            map_location="cpu",
        )

        if not isinstance(checkpoint, dict):
            raise TypeError(
                "Expected a Lightning Stage-I checkpoint dictionary."
            )

        if cls._checkpoint_was_stage1_plus(checkpoint):
            raise ValueError(
                "Stage I+ initialization requires a Stage-I checkpoint, "
                "but the supplied checkpoint is marked as Stage I+."
            )

        state_dict = checkpoint.get(
            "state_dict",
        )
        if state_dict is None:
            raise KeyError(
                "The supplied Stage-I checkpoint has no 'state_dict'."
            )

        stage1_plus_config = deepcopy(config)
        stage1_plus_options = stage1_plus_config.setdefault(
            "stage1_plus",
            {},
        )
        if not isinstance(stage1_plus_options, dict):
            raise TypeError(
                "config['stage1_plus'] must be a dictionary"
            )

        stage1_plus_options["enabled"] = True

        # Keep the path available in the live training configuration if the
        # caller supplied it there, but provenance stored in the checkpoint is
        # only a boolean marker and never records this source path.
        model = cls(stage1_plus_config)

        # The CTC hook adds no state-dict entries, so strict loading verifies
        # that Stage I and Stage I+ share exactly the same trainable
        # architecture.
        model.load_state_dict(
            state_dict,
            strict=True,
        )
        model._stage1_plus_initialized_from_stage1 = True
        model.train()

        if (
            gpu_index is not None
            and torch.cuda.is_available()
        ):
            model = model.to(
                torch.device(
                    f"cuda:{int(gpu_index)}"
                )
            )

        return model

    @classmethod
    def from_training_config(
        cls,
        config: Dict[str, Any],
        gpu_index: Optional[int] = None,
    ):
        """
        Build Stage I or Stage I+ from one training configuration.

        Stage I:
            stage1_plus.enabled: false

        Stage I+:
            stage1_plus.enabled: true

        When Stage I+ is enabled, a pretrained Stage-I checkpoint is mandatory
        and must exist at either:
            stage1_plus.pretrained_checkpoint
        or:
            checkpoints.stage1

        Stage I+ is then initialized strictly from that Stage-I checkpoint
        before any training step is permitted.
        """
        stage1_plus_config = config.get(
            "stage1_plus",
            {},
        ) or {}

        if not isinstance(stage1_plus_config, dict):
            raise TypeError(
                "config['stage1_plus'] must be a dictionary"
            )

        if not bool(
            stage1_plus_config.get(
                "enabled",
                False,
            )
        ):
            model = cls(config)

            if (
                gpu_index is not None
                and torch.cuda.is_available()
            ):
                model = model.to(
                    torch.device(
                        f"cuda:{int(gpu_index)}"
                    )
                )

            return model

        stage1_checkpoint = cls._resolve_stage1_checkpoint(
            config
        )

        return cls.initialize_stage1_plus(
            config=config,
            stage1_checkpoint=stage1_checkpoint,
            gpu_index=gpu_index,
        )

    @classmethod
    def load_from_pretrained(
        cls,
        checkpoint,
        gpu_index: Optional[int] = None,
    ):
        """Load a trained Stage-I or Stage-I+ checkpoint for evaluation."""
        model = cls.load_from_checkpoint(
            checkpoint,
            map_location="cpu",
        )
        model.eval()

        if (
            gpu_index is not None
            and torch.cuda.is_available()
        ):
            model = model.to(
                torch.device(
                    f"cuda:{int(gpu_index)}"
                )
            )

        return model
