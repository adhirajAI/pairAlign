import os
import csv
import yaml
import argparse
from typing import Any, List, Tuple

import torch
from torch.utils.data import DataLoader
from lightning.pytorch import seed_everything
from tqdm.auto import tqdm

from train import AudioTokenizer, AudioTokenizerARPrior, AudioTokenizerAR_FINAL
from utils import AudioPairAugDataset


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


def build_model(cfg, ckpt_path: str, model_type: str, device_id: int):
    """
    Uses your existing load_from_pretrained style.
    Adjust here if your checkpoint loader is different.
    """
    if model_type == "AudioTokenizer":
        model = AudioTokenizer.load_from_pretrained(ckpt_path, device_id)
    elif model_type == "AudioTokenizerARPrior":
        model = AudioTokenizerARPrior.load_from_pretrained(ckpt_path, device_id)
    elif model_type == "AudioTokenizerAR_FINAL":
        model = AudioTokenizerAR_FINAL.load_from_pretrained(ckpt_path, device_id)
        # model = AudioTokenizerAR_FINAL(cfg)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    return model


# ---------------------------------------------------------------------
# Collate / batch parsing
# ---------------------------------------------------------------------

def pair_collate(batch):
    """
    Supports common AudioPairAugDataset return formats.

    Expected useful formats:
      1. (anc, pos)
      2. (anc, pos, path)
      3. (anc, pos, file_id)
      4. dict with keys like anc/pos/path

    Returns:
      anc_feats: Tensor
      pos_feats: Tensor
      file_ids: List[str]
    """

    first = batch[0]

    # Case 1: dictionary dataset output
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
                f"Could not infer anchor/positive keys from dict keys: {first.keys()}"
            )

        anc_feats = torch.stack([b[anc_key] for b in batch], dim=0)
        pos_feats = torch.stack([b[pos_key] for b in batch], dim=0)

        if id_key is not None:
            raw_ids = [b[id_key] for b in batch]
            file_ids = [
                path_to_file_id(x) if "/" in str(x) or "." in str(x) else str(x)
                for x in raw_ids
            ]
        else:
            file_ids = ["unknown"] * len(batch)

        return anc_feats, pos_feats, file_ids

    # Case 2: tuple/list dataset output
    if isinstance(first, (tuple, list)):
        if len(first) < 2:
            raise ValueError(f"Dataset item must contain at least anc and pos. Got: {first}")

        anc_feats = torch.stack([b[0] for b in batch], dim=0)
        pos_feats = torch.stack([b[1] for b in batch], dim=0)

        if len(first) >= 3:
            raw_ids = [b[2] for b in batch]
            file_ids = [
                path_to_file_id(x) if "/" in str(x) or "." in str(x) else str(x)
                for x in raw_ids
            ]
        else:
            file_ids = ["unknown"] * len(batch)

        return anc_feats, pos_feats, file_ids

    raise ValueError(f"Unsupported dataset item type: {type(first)}")


# ---------------------------------------------------------------------
# Main generation loop
# ---------------------------------------------------------------------

@torch.no_grad()
def generate_pair_tokenizations(
    model,
    loader,
    output_csv: str,
    device: torch.device,
    max_len: int = 301,
    beam_size: int = 4,
):
    model.eval()
    model.to(device)

    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)

    global_index = 0

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["file_id", "anc_tokenization", "pos_tokenization"])

        pbar = tqdm(loader, desc="Generating pair tokenizations", unit="batch")

        for anc_feats, pos_feats, file_ids in pbar:
            anc_feats = anc_feats.to(device, non_blocking=True)
            pos_feats = pos_feats.to(device, non_blocking=True)

            anc_tokens = model.tokenize(
                anc_feats,
                max_len=max_len,
                beam_size=beam_size,
                strip_special_tokens=True,
            )

            pos_tokens = model.tokenize(
                pos_feats,
                max_len=max_len,
                beam_size=beam_size,
                strip_special_tokens=True,
            )

            if len(anc_tokens) != len(pos_tokens):
                raise RuntimeError(
                    f"Anchor/positive token batch mismatch: "
                    f"{len(anc_tokens)} vs {len(pos_tokens)}"
                )

            for fid, anc_tok, pos_tok in zip(file_ids, anc_tokens, pos_tokens):
                if fid == "unknown":
                    fid = f"sample_{global_index:09d}"

                writer.writerow([
                    fid,
                    serialize_tokens(anc_tok),
                    serialize_tokens(pos_tok),
                ])

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
        "--ckpt",
        type=str,
        required=True,
        help="Checkpoint path",
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
        "--model_type",
        type=str,
        default="AudioTokenizerAR_FINAL",
        choices=["AudioTokenizer", "AudioTokenizerARPrior", "AudioTokenizerAR_FINAL"],
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
        "--add_noise",
        action="store_true",
        help="Use augmentation/noise for positive pair",
    )

    parser.add_argument(
        "--no_add_noise",
        action="store_true",
        help="Disable augmentation/noise",
    )
    parser.add_argument(
        "--use_time_compression",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Slightly compress only the positive view and right-pad it back to 3 seconds.",
    )

    args = parser.parse_args()

    config_path = os.path.join(args.root_dir, "config", args.config)

    with open(config_path, "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    seed_everything(cfg.get("seed", 13), workers=True) # 13 # 97 # 313 -> 1, 2, 3 

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

    print(f"Reading config: {config_path}")
    print(f"Using split: {args.split}")
    print(f"Dataset CSV: {dataset_csv}")
    print(f"Checkpoint: {args.ckpt}")
    print(f"Output CSV: {args.output_csv}")
    print(f"add_noise: {add_noise}")
    print(f"use_time_compression: {args.use_time_compression}")

    dataset = AudioPairAugDataset(
        audio_csv=dataset_csv,
        fs=cfg["fs"],
        context_sec=cfg["seglen"],
        add_noise=add_noise,
        use_time_compression=args.use_time_compression,
        return_path=True,   # important for file_id if your dataset supports it
        feats_type="melspectrogram",
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

    model = build_model(
        cfg=cfg,
        ckpt_path=args.ckpt,
        model_type=args.model_type,
        device_id=args.device,
    )

    generate_pair_tokenizations(
        model=model,
        loader=loader,
        output_csv=args.output_csv,
        device=device,
        max_len=args.max_len,
        beam_size=args.beam_size,
    )


if __name__ == "__main__":
    main()