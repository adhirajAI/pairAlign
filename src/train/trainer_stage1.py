from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
import transformers
from tslearn.metrics import dtw_path_from_metric
from vector_quantize_pytorch import VectorQuantize

from models import MambaEncoder, ModelArgs


class AudioTokenizer(L.LightningModule):
    """Stage-I geometric audio tokenizer."""

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
        self.max_contrastive_items = int(config.get("max_contrastive_items", 30))
        self.num_negative_frames = int(config.get("num_negative_frames", 30))

        if self.temperature <= 0.0:
            raise ValueError(f"temperature must be > 0, got {self.temperature}")
        if self.max_contrastive_items < 1:
            raise ValueError("max_contrastive_items must be >= 1")
        if self.num_negative_frames < 1:
            raise ValueError("num_negative_frames must be >= 1")

        self.val_codes_cache: List[List[int]] = []
        self.save_hyperparameters()

    @staticmethod
    def deduplicate(seq: torch.Tensor) -> torch.Tensor:
        """Collapse consecutive duplicate token IDs."""
        if seq.numel() <= 1:
            return seq
        keep = torch.ones_like(seq, dtype=torch.bool)
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
        return z_ctx, z_ctx_quant, indices

    def _prepare_feature_input(self, features: torch.Tensor) -> torch.Tensor:
        """Convert common feature layouts to encoder layout (B,T,F)."""
        if not torch.is_tensor(features):
            raise TypeError(f"features must be a torch.Tensor, got {type(features)}")

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
                "Expected (B,F,T), (B,T,F), (F,T), (T,F), or (B,1,F,T); "
                f"got {tuple(features.shape)}"
            )

        if features.size(1) == n_mels and features.size(2) != n_mels:
            return features.permute(0, 2, 1).contiguous()

        if features.size(2) == n_mels:
            return features.contiguous()

        raise ValueError(
            "Could not identify the mel-feature dimension. "
            f"Expected n_mels={n_mels}, got shape {tuple(features.shape)}"
        )

    def _shared_step(self, batch, mode: str):
        if mode not in {"train", "valid"}:
            raise ValueError(f"Unsupported mode: {mode}")
        if len(batch) < 2:
            raise ValueError("Stage-I expects paired anchor/positive inputs.")

        anchor = batch[0].squeeze(1).permute(0, 2, 1).contiguous()
        positive = batch[1].squeeze(1).permute(0, 2, 1).contiguous()

        if anchor.size(0) != positive.size(0):
            raise ValueError("Anchor and positive batches must have the same size.")

        batch_size = anchor.size(0)

        z_anchor = self.encoder(anchor)
        z_positive = self.encoder(positive)

        z_anchor_quant, anchor_indices, anchor_vq_loss = self.vq_layer(z_anchor)
        z_positive_quant, positive_indices, positive_vq_loss = self.vq_layer(z_positive)

        log_values: Dict[str, Any] = {}

        codebook_loss = (
            0.5
            * (anchor_vq_loss + positive_vq_loss)
            * float(self.cost_factors["vq_loss"])
        )
        loss = codebook_loss
        log_values[f"{mode}_codebook_loss"] = codebook_loss

        if mode == "train":
            if batch_size < 2:
                raise ValueError("Stage-I contrastive training requires batch_size >= 2.")

            contrastive_sum = z_anchor.new_zeros(())
            total_aligned_frames = 0
            token_agreements: List[float] = []

            for sample_index in range(min(batch_size, self.max_contrastive_items)):
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

                anchor_tokens = anchor_indices[sample_index, triplets[:, 0]]
                positive_tokens = positive_indices[sample_index, triplets[:, 1]]
                token_agreements.append(
                    anchor_tokens.eq(positive_tokens).float().mean().item()
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
                    (positive_similarity.unsqueeze(1), negative_similarity),
                    dim=1,
                ) / self.temperature

                contrastive_sum = contrastive_sum + F.cross_entropy(
                    logits,
                    torch.zeros(
                        logits.size(0),
                        dtype=torch.long,
                        device=logits.device,
                    ),
                    reduction="sum",
                )
                total_aligned_frames += int(triplets.size(0))

            if total_aligned_frames == 0:
                raise RuntimeError(
                    "No aligned frames were produced for the Stage-I contrastive objective."
                )

            contrastive_loss = (
                float(self.cost_factors["triplet_loss"])
                * contrastive_sum
                / total_aligned_frames
            )
            loss = loss + contrastive_loss

            log_values[f"{mode}_anc_pos_token_acc"] = (
                float(np.mean(token_agreements)) if token_agreements else 0.0
            )
            log_values[f"{mode}_frames"] = float(total_aligned_frames)
            log_values[f"{mode}_contra_loss"] = contrastive_loss

        # Compatibility attributes used by existing callbacks/analysis code.
        self.z_anc = z_anchor[0]
        self.z_pos = z_positive[0]
        self.z_anc_quant = z_anchor_quant[0]
        self.z_pos_quant = z_positive_quant[0]

        anchor_codes = [self.deduplicate(seq) for seq in anchor_indices]
        positive_codes = [self.deduplicate(seq) for seq in positive_indices]

        log_values[f"{mode}_acc"] = self._get_normalized_edit_similarity(
            anchor_codes,
            positive_codes,
        )

        if mode == "valid":
            # Diagnostics are computed on the symbolic sequences after
            # consecutive-duplicate collapse.
            self.val_codes_cache.extend(
                seq.detach().cpu().tolist() for seq in anchor_codes
            )

        self.log_dict(
            log_values,
            on_epoch=True,
            on_step=True,
            logger=True,
            prog_bar=True,
        )

        return {"loss": loss, "z": z_anchor}

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, mode="train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, mode="valid")

    @torch.no_grad()
    def on_validation_epoch_end(self) -> None:
        sequences = self.val_codes_cache
        num_sequences = len(sequences)

        if num_sequences == 0:
            self.val_codes_cache.clear()
            return

        sequence_counts = Counter(tuple(sequence) for sequence in sequences)
        num_unique = len(sequence_counts)
        collision_rate = (num_sequences - num_unique) / num_sequences
        top_sequence_fraction = sequence_counts.most_common(1)[0][1] / num_sequences

        token_counts = Counter()
        total_tokens = 0
        for sequence in sequences:
            token_counts.update(sequence)
            total_tokens += len(sequence)

        used_tokens = len(token_counts)
        top_token_fraction = (
            token_counts.most_common(1)[0][1] / total_tokens
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
            raise ValueError("Negative sampling requires batch_size >= 2.")
        if not 0 <= excluded_index < batch_size:
            raise ValueError(
                f"excluded_index={excluded_index} is outside batch_size={batch_size}"
            )

        # Preserve the original uniform rejection-sampling behavior without
        # recursion. Keeping torch.randint on CPU also preserves the original
        # RNG source used by the training code.
        while True:
            sampled = int(torch.randint(batch_size, (1,)).item())
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
        if anchor.dim() != 2 or positive.dim() != 2 or negative.dim() != 2:
            raise ValueError(
                "anchor, positive, and negative must each have shape (T,D)."
            )
        if num_negative_frames < 1:
            raise ValueError("num_negative_frames must be >= 1")
        if negative.size(0) < 1:
            raise ValueError("negative sequence contains no frames")

        anchor_norm = anchor / torch.linalg.norm(
            anchor,
            dim=1,
            keepdim=True,
        ).clamp_min(1e-12)
        positive_norm = positive / torch.linalg.norm(
            positive,
            dim=1,
            keepdim=True,
        ).clamp_min(1e-12)

        path, _ = dtw_path_from_metric(
            anchor_norm.detach().cpu().numpy(),
            positive_norm.detach().cpu().numpy(),
            metric="cosine",
        )

        alignment: Dict[int, List[int]] = {}
        for anchor_index, positive_index in path:
            alignment.setdefault(int(anchor_index), []).append(int(positive_index))

        rows = []
        for anchor_index, candidate_positive_indices in alignment.items():
            candidates = torch.tensor(
                candidate_positive_indices,
                dtype=torch.long,
                device=positive_norm.device,
            )
            similarities = (
                anchor_norm[anchor_index].unsqueeze(0)
                @ positive_norm[candidates].T
            ).squeeze(0)
            positive_index = int(candidates[similarities.argmax()].item())

            negative_indices = np.random.choice(
                negative.size(0),
                num_negative_frames,
                replace=True,
            )
            rows.append(
                [anchor_index, positive_index, *negative_indices.tolist()]
            )

        if not rows:
            return torch.empty(
                (0, 2 + num_negative_frames),
                dtype=torch.long,
                device=anchor.device,
            )

        return torch.tensor(rows, dtype=torch.long, device=anchor.device)

    @staticmethod
    def _edit_distance(first: Sequence[int], second: Sequence[int]) -> int:
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
                substitution_cost = 0 if first[i - 1] == second[j - 1] else 1
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
        for anchor_seq, positive_seq in zip(anchor_codes, positive_codes):
            if isinstance(anchor_seq, torch.Tensor):
                anchor_seq = anchor_seq.detach().cpu().tolist()
            if isinstance(positive_seq, torch.Tensor):
                positive_seq = positive_seq.detach().cpu().tolist()

            denominator = max(len(anchor_seq), len(positive_seq))
            if denominator == 0:
                similarities.append(1.0)
                continue

            distance = cls._edit_distance(anchor_seq, positive_seq)
            similarities.append(1.0 - distance / denominator)

        return float(np.mean(similarities)) if similarities else 0.0

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
            device = next(self.parameters()).device
        else:
            device = torch.device(device)

        encoder_input = self._prepare_feature_input(features).to(device)
        _, _, indices = self._encode_and_quantize(encoder_input)

        output = []
        for sequence in indices.detach().cpu():
            sequence = sequence.reshape(-1)
            if deduplicate:
                sequence = self.deduplicate(sequence)

            if return_tensors:
                output.append(sequence.clone())
            else:
                output.append([int(token) for token in sequence.tolist()])

        return output

    @torch.no_grad()
    def predict_step(self, batch, batch_idx):
        """Lightning prediction hook; batch must be in (B,T,F) layout."""
        return self._encode_and_quantize(batch)

    def configure_optimizers(self):
        if self.optim == "adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        elif self.optim == "adamw":
            if self.wt_decay <= 0.0:
                raise ValueError("AdamW requires a positive weight decay.")
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.lr,
                weight_decay=self.wt_decay,
            )
        else:
            raise NotImplementedError(f"Unsupported optimizer: {self.optim}")

        if not self.lr_scheduler:
            return {"optimizer": optimizer}

        # Defaults preserve the original epoch-stepped schedule.
        scheduler = transformers.get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(self.config.get("num_warmup_steps", 10)),
            num_training_steps=int(self.config.get("num_training_steps", 300)),
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": self.config.get("lr_scheduler_interval", "epoch"),
            },
        }

    @classmethod
    def load_from_pretrained(
        cls,
        checkpoint,
        gpu_index: Optional[int] = None,
    ):
        """Load a Stage-I checkpoint without assuming CUDA is available."""
        model = cls.load_from_checkpoint(
            checkpoint,
            map_location="cpu",
        )
        model.eval()

        if gpu_index is not None and torch.cuda.is_available():
            model = model.to(torch.device(f"cuda:{int(gpu_index)}"))

        return model
