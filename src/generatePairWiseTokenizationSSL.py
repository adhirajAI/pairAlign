import os
import csv
import yaml
import argparse
import warnings
from dataclasses import dataclass
from typing import Any, List, Tuple

import torch
import torchaudio
import joblib
import numpy as np

from torch.utils.data import DataLoader
from lightning.pytorch import seed_everything
from tqdm.auto import tqdm
from huggingface_hub import hf_hub_download

from speechbrain.integrations.huggingface.hubert import HuBERT
from speechbrain.integrations.huggingface.wav2vec2 import Wav2Vec2
from speechbrain.integrations.huggingface.wavlm import WavLM

from utils import AudioPairAugDataset


# ---------------------------------------------------------------------
# SSL Quantizer registry
# ---------------------------------------------------------------------

SSL_MODEL_REGISTRY = {
    "hubert": {
        "class": HuBERT,
        "source": "facebook/hubert-base-ls960",
        "kmeans_name": "hubert",
    },
    "wav2vec2": {
        "class": Wav2Vec2,
        "source": "facebook/wav2vec2-base",
        "kmeans_name": "wav2vec",
    },
    # IMPORTANT:
    # chaanks/Kmeans LibriSpeech_wavlm_k512_L7.pt has 1024-dim centers,
    # so use wavlm-large, not wavlm-base.
    "wavlm": {
        "class": WavLM,
        "source": "microsoft/wavlm-large",
        "kmeans_name": "wavlm",
    },
}

def deduplicate_consecutive_tokens(tokens: torch.Tensor) -> torch.Tensor:
    """
    Removes consecutive duplicate tokens.

    Example:
        [1, 1, 1, 4, 4, 2, 2, 7] -> [1, 4, 2, 7]
    """
    tokens = tokens.detach().cpu().reshape(-1)

    if tokens.numel() == 0:
        return tokens

    keep = torch.ones(tokens.numel(), dtype=torch.bool)
    keep[1:] = tokens[1:] != tokens[:-1]

    return tokens[keep]

@dataclass
class SSLKMeansConfig:
    ssl_model: str
    k: int = 512
    layer_num: int = 7
    dataset: str = "LibriSpeech"
    kmeans_repo_id: str = "chaanks/Kmeans"
    save_root: str = "pretrained_ssl"


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def path_to_file_id(path: str) -> str:
    return os.path.splitext(os.path.basename(str(path)))[0]


def serialize_tokens(tokens: Any) -> str:
    """
    Converts token output to CSV-friendly string.
    Handles tensor/list/nested list outputs.
    """
    if isinstance(tokens, torch.Tensor):
        tokens = tokens.detach().cpu()

        if tokens.ndim == 0:
            return str(int(tokens.item()))

        if tokens.ndim == 1:
            return " ".join(str(int(x)) for x in tokens.tolist())

        if tokens.ndim == 2 and tokens.size(0) == 1:
            return " ".join(str(int(x)) for x in tokens[0].tolist())

        return " ".join(str(int(x)) for x in tokens.reshape(-1).tolist())

    if isinstance(tokens, (list, tuple)):
        if len(tokens) == 0:
            return ""

        if isinstance(tokens[0], torch.Tensor):
            flat = []
            for t in tokens:
                t = t.detach().cpu().reshape(-1).tolist()
                flat.extend(t)
            return " ".join(str(int(x)) for x in flat)

        if isinstance(tokens[0], (list, tuple)):
            if len(tokens) == 1:
                return " ".join(str(int(x)) for x in tokens[0])

            flat = []
            for row in tokens:
                flat.extend(row)
            return " ".join(str(int(x)) for x in flat)

        return " ".join(str(int(x)) for x in tokens)

    return str(tokens)


def maybe_stack_or_pad(xs: List[torch.Tensor]) -> torch.Tensor:
    """
    Tries to stack fixed-length waveform tensors.
    If lengths differ, right-pads to max length.

    Expected input item shape:
      [T] or [1, T] or [C, T]
    Output:
      [B, T]
    """
    cleaned = []

    for x in xs:
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Expected tensor waveform, got {type(x)}")

        # [C, T] -> [T]
        if x.ndim == 2:
            x = x.mean(dim=0)

        # scalar/other invalid
        if x.ndim != 1:
            raise ValueError(f"Expected waveform shape [T] or [C, T], got {x.shape}")

        cleaned.append(x)

    lengths = [x.numel() for x in cleaned]

    if len(set(lengths)) == 1:
        return torch.stack(cleaned, dim=0)

    max_len = max(lengths)
    batch = torch.zeros(len(cleaned), max_len, dtype=cleaned[0].dtype)

    for i, x in enumerate(cleaned):
        batch[i, : x.numel()] = x

    return batch


def load_kmeans_centers(path: str) -> torch.Tensor:
    """
    chaanks/Kmeans files are sklearn/joblib-style serialized KMeans objects,
    not torch.load checkpoints.
    """
    obj = joblib.load(path)

    print("Loaded KMeans object type:", type(obj))

    if hasattr(obj, "cluster_centers_"):
        return torch.tensor(obj.cluster_centers_, dtype=torch.float32)

    if isinstance(obj, dict):
        print("KMeans dict keys:", obj.keys())

        for key in [
            "cluster_centers_",
            "cluster_centers",
            "centroids",
            "centers",
            "kmeans",
            "model",
        ]:
            if key not in obj:
                continue

            value = obj[key]

            if hasattr(value, "cluster_centers_"):
                return torch.tensor(value.cluster_centers_, dtype=torch.float32)

            if isinstance(value, np.ndarray):
                return torch.tensor(value, dtype=torch.float32)

            if torch.is_tensor(value):
                return value.float()

    if isinstance(obj, np.ndarray):
        return torch.tensor(obj, dtype=torch.float32)

    raise RuntimeError(f"Could not extract KMeans centers from: {type(obj)}")


# ---------------------------------------------------------------------
# SSL KMeans tokenizer
# ---------------------------------------------------------------------

class SSLKMeansTokenizer(torch.nn.Module):
    def __init__(self, config: SSLKMeansConfig, device: torch.device):
        super().__init__()

        self.config = config
        self.device = device

        ssl_model = config.ssl_model.lower()

        if ssl_model not in SSL_MODEL_REGISTRY:
            raise ValueError(
                f"Unknown ssl_model={ssl_model}. "
                f"Choose from {list(SSL_MODEL_REGISTRY.keys())}."
            )

        model_info = SSL_MODEL_REGISTRY[ssl_model]
        model_class = model_info["class"]
        source = model_info["source"]
        kmeans_name = model_info["kmeans_name"]

        self.kmeans_filename = (
            f"{config.dataset}_{kmeans_name}_k{config.k}_L{config.layer_num}.pt"
        )

        print(f"SSL model:       {ssl_model}")
        print(f"HF source:       {source}")
        print(f"KMeans repo:     {config.kmeans_repo_id}")
        print(f"KMeans file:     {self.kmeans_filename}")
        print(f"SSL layer:       {config.layer_num}")

        self.ssl = model_class(
            source=source,
            save_path=os.path.join(config.save_root, ssl_model),
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

        self.register_buffer("centers", centers.to(device), persistent=False)

        print("KMeans centers:", self.centers.shape)

    @torch.no_grad()
    def tokenize(self, wavs: torch.Tensor) -> List[torch.Tensor]:
        """
        wavs:
            [B, T], sampled at 16 kHz.

        returns:
            List[Tensor], each tensor has shape [T_ssl]
        """
        wavs = wavs.to(self.device)

        hidden_states = self.ssl(wavs)

        # Depending on SpeechBrain version, this should usually be:
        # [num_layers + 1, B, T_ssl, D]
        if isinstance(hidden_states, (tuple, list)):
            hidden_states = torch.stack(list(hidden_states), dim=0)

        if hidden_states.ndim != 4:
            raise RuntimeError(
                f"Expected hidden states with shape [L, B, T, D], "
                f"got {hidden_states.shape}"
            )

        feats = hidden_states[self.config.layer_num]  # [B, T_ssl, D]

        if feats.shape[-1] != self.centers.shape[-1]:
            raise RuntimeError(
                f"Feature dim mismatch. SSL features have dim {feats.shape[-1]}, "
                f"but KMeans centers have dim {self.centers.shape[-1]}. "
                f"Check ssl_model/source and KMeans file."
            )

        tokens = []

        for b in range(feats.shape[0]):
            dists = torch.cdist(
                feats[b].float(),
                self.centers.float(),
            )  # [T_ssl, K]

            tok = dists.argmin(dim=-1).detach().cpu()  # [T_ssl]
            tok = deduplicate_consecutive_tokens(tok)
            tokens.append(tok)

        return tokens


# ---------------------------------------------------------------------
# Collate / batch parsing
# ---------------------------------------------------------------------

def pair_collate(batch):
    """
    Supports common AudioPairAugDataset return formats.

    Expected useful formats:
      1. (anc_wav, pos_wav)
      2. (anc_wav, pos_wav, path)
      3. (anc_wav, pos_wav, file_id)
      4. dict with keys like anc/pos/path

    Returns:
      anc_wavs: Tensor [B, T]
      pos_wavs: Tensor [B, T]
      file_ids: List[str]
    """
    first = batch[0]

    if isinstance(first, dict):
        anc_keys = ["anc", "anchor", "anchor_wav", "wav", "x"]
        pos_keys = ["pos", "positive", "positive_wav", "wav_aug", "x_aug"]
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
                f"Could not infer anchor/positive keys from dict keys: {first.keys()}"
            )

        anc_wavs = maybe_stack_or_pad([b[anc_key] for b in batch])
        pos_wavs = maybe_stack_or_pad([b[pos_key] for b in batch])

        if id_key is not None:
            raw_ids = [b[id_key] for b in batch]
            file_ids = [
                path_to_file_id(x) if "/" in str(x) or "." in str(x) else str(x)
                for x in raw_ids
            ]
        else:
            file_ids = ["unknown"] * len(batch)

        return anc_wavs, pos_wavs, file_ids

    if isinstance(first, (tuple, list)):
        if len(first) < 2:
            raise ValueError(f"Dataset item must contain at least anc and pos. Got: {first}")

        anc_wavs = maybe_stack_or_pad([b[0] for b in batch])
        pos_wavs = maybe_stack_or_pad([b[1] for b in batch])

        if len(first) >= 3:
            raw_ids = [b[2] for b in batch]
            file_ids = [
                path_to_file_id(x) if "/" in str(x) or "." in str(x) else str(x)
                for x in raw_ids
            ]
        else:
            file_ids = ["unknown"] * len(batch)

        return anc_wavs, pos_wavs, file_ids

    raise ValueError(f"Unsupported dataset item type: {type(first)}")


# ---------------------------------------------------------------------
# Main generation loop
# ---------------------------------------------------------------------

@torch.no_grad()
def generate_pair_tokenizations(
    tokenizer: SSLKMeansTokenizer,
    loader: DataLoader,
    output_csv: str,
    device: torch.device,
):
    tokenizer.eval()
    tokenizer.to(device)

    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)

    global_index = 0

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["file_id", "anc_tokenization", "pos_tokenization"])

        pbar = tqdm(loader, desc="Generating SSL pair tokenizations", unit="batch")

        for anc_wavs, pos_wavs, file_ids in pbar:
            anc_wavs = anc_wavs.to(device, non_blocking=True)
            pos_wavs = pos_wavs.to(device, non_blocking=True)

            anc_tokens = tokenizer.tokenize(anc_wavs)
            pos_tokens = tokenizer.tokenize(pos_wavs)

            if len(anc_tokens) != len(pos_tokens):
                raise RuntimeError(
                    f"Anchor/positive token batch mismatch: "
                    f"{len(anc_tokens)} vs {len(pos_tokens)}"
                )

            for fid, anc_tok, pos_tok in zip(file_ids, anc_tokens, pos_tokens):
                if fid == "unknown":
                    fid = f"sample_{global_index:09d}"

                writer.writerow(
                    [
                        fid,
                        serialize_tokens(anc_tok),
                        serialize_tokens(pos_tok),
                    ]
                )

                global_index += 1

            pbar.set_postfix(written=global_index)

    print(f"Saved pair tokenizations to: {output_csv}")


# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root_dir",
        type=str,
        default=os.path.dirname(os.getcwd()),
        help="Parent working directory",
    )

    parser.add_argument(
        "--config",
        type=str,
        default="mainFull.yaml",
        help="Config file name inside root_dir/config/",
    )

    parser.add_argument(
        "--split",
        type=str,
        default="valid",
        choices=["train", "valid", "test"],
        help="Which config split to tokenize",
    )

    parser.add_argument(
        "--output_csv",
        type=str,
        required=True,
        help="Output CSV path",
    )

    parser.add_argument(
        "--ssl_model",
        type=str,
        required=True,
        choices=["hubert", "wav2vec2", "wavlm"],
        help="SSL quantizer to use.",
    )

    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="CUDA device id",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Override config batch size",
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Override config num_workers",
    )

    parser.add_argument(
        "--layer_num",
        type=int,
        default=7,
        help="SSL layer used for KMeans. For chaanks LibriSpeech files, use 7.",
    )

    parser.add_argument(
        "--save_root",
        type=str,
        default="pretrained_ssl",
        help="Where to cache SSL models.",
    )

    parser.add_argument(
        "--add_noise",
        action="store_true",
        help="Use augmentation/noise for positive pair.",
    )

    parser.add_argument(
        "--no_add_noise",
        action="store_true",
        help="Disable augmentation/noise.",
    )
    parser.add_argument(
        "--use_time_compression",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Slightly compress only the positive view and right-pad it back to 3 seconds.",
    )

    parser.add_argument(
        "--feats_type",
        type=str,
        default="raw",
        choices=["raw"],
        help="SSL tokenizers operate on raw waveform pairs.",
    )

    args = parser.parse_args()

    warnings.filterwarnings(
        "default",
        category=UserWarning,
    )

    config_path = os.path.join(args.root_dir, "config", args.config)

    with open(config_path, "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    seed_everything(cfg.get("seed", 13), workers=True)

    if args.split not in cfg:
        raise ValueError(f"Split `{args.split}` not found in config.")

    split_cfg = cfg[args.split]
    dataset_csv = split_cfg["dataset_csv"]

    batch_size = args.batch_size if args.batch_size is not None else int(cfg["batchsize"])
    num_workers = args.num_workers if args.num_workers is not None else int(cfg["load_workers"])

    if args.no_add_noise:
        add_noise = False
    else:
        add_noise = True if args.add_noise else True

    print(f"Reading config:  {config_path}")
    print(f"Using split:     {args.split}")
    print(f"Dataset CSV:     {dataset_csv}")
    print(f"SSL model:       {args.ssl_model}")
    print(f"Output CSV:      {args.output_csv}")
    print(f"add_noise:       {add_noise}")
    print(f"time_compress:   {args.use_time_compression}")
    print(f"feats_type:      {args.feats_type}")

    # IMPORTANT:
    # SSL tokenizers need raw waveform, not mel spectrogram.
    # This assumes your AudioPairAugDataset supports feats_type="waveform".
    # If your dataset uses another name, pass --feats_type raw/audio/etc.
    dataset = AudioPairAugDataset(
        audio_csv=dataset_csv,
        fs=cfg["fs"],
        context_sec=cfg["seglen"],
        add_noise=add_noise,
        use_time_compression=args.use_time_compression,
        return_path=True,
        feats_type=args.feats_type,
        rf_size=int(round(cfg["win_size"] * cfg["fs"])),
        stride=int(round(cfg["win_hop"] * cfg["fs"])),
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

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    ssl_cfg = SSLKMeansConfig(
        ssl_model=args.ssl_model,
        k=512,
        layer_num=args.layer_num,
        dataset="LibriSpeech",
        kmeans_repo_id="chaanks/Kmeans",
        save_root=args.save_root,
    )

    tokenizer = SSLKMeansTokenizer(
        config=ssl_cfg,
        device=device,
    )

    generate_pair_tokenizations(
        tokenizer=tokenizer,
        loader=loader,
        output_csv=args.output_csv,
        device=device,
    )


if __name__ == "__main__":
    main()