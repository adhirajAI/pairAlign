import argparse
import os
import csv
from dataclasses import dataclass
from typing import List, Tuple, Optional, Any, Dict
from tqdm.auto import tqdm 
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchaudio
import torchaudio.transforms as T
import torchaudio.functional as AF
from train import AudioTokenizer, AudioTokenizerARPrior, AudioTokenizerAR_FINAL


# ============================================================
# Basic audio utils
# ============================================================

def _load_audio_mono(path: str) -> Tuple[torch.Tensor, int]:
    """
    Returns:
        wav: (1, N) float32
        sr: int
    """
    wav, sr = torchaudio.load(path)
    if wav.dtype != torch.float32:
        wav = wav.float()
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    return wav, sr


def _ensure_sr(wav: torch.Tensor, sr: int, target_sr: int) -> Tuple[torch.Tensor, int]:
    if sr == target_sr:
        return wav, sr
    wav = AF.resample(wav, orig_freq=sr, new_freq=target_sr)
    return wav, target_sr


def _read_paths_from_csv(csv_path: str, path_col: str = "path") -> List[str]:
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    paths: List[str] = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        if path_col not in reader.fieldnames:
            raise ValueError(
                f"CSV must contain column `{path_col}`. Found: {reader.fieldnames}"
            )
        for row in reader:
            p = row[path_col].strip()
            if p:
                paths.append(p)

    return paths


def _crop_or_pad_at_start(wav: torch.Tensor, start: int, target_len: int) -> torch.Tensor:
    """
    Deterministic crop for continuous sweep.

    wav: (1, N)
    start: starting sample index
    returns: (1, target_len)

    If the requested segment exceeds the file end, it is zero-padded.
    """
    n = wav.size(-1)
    if start < 0:
        raise ValueError(f"start must be >= 0, got {start}")

    end = start + target_len
    if start >= n:
        return torch.zeros((1, target_len), dtype=wav.dtype)

    seg = wav[..., start:min(end, n)]
    if seg.size(-1) < target_len:
        seg = F.pad(seg, (0, target_len - seg.size(-1)))
    return seg


def _path_to_file_id(path: str) -> str:
    """
    Example:
      /.../1089-134686-0000.flac -> 1089-134686-0000
    """
    return os.path.splitext(os.path.basename(path))[0]


def _serialize_tokens(tokens: Any) -> str:
    """
    Converts output of model.tokenize(...) to a CSV-friendly string.
    Assumes tokenize returns one of:
      - torch.Tensor
      - list[int]
      - nested list / tensor with batch dim
    """
    if isinstance(tokens, torch.Tensor):
        tokens = tokens.detach().cpu()

        # Common cases:
        # (L,), (1, L), or even scalar
        if tokens.ndim == 0:
            return str(int(tokens.item()))
        if tokens.ndim == 1:
            return " ".join(str(int(x)) for x in tokens.tolist())
        if tokens.ndim == 2 and tokens.size(0) == 1:
            return " ".join(str(int(x)) for x in tokens[0].tolist())

        # Fallback: flatten
        return " ".join(str(int(x)) for x in tokens.reshape(-1).tolist())

    if isinstance(tokens, (list, tuple)):
        if len(tokens) == 0:
            return ""
        if isinstance(tokens[0], (list, tuple)):
            if len(tokens) == 1:
                return " ".join(str(int(x)) for x in tokens[0])
            flat = []
            for row in tokens:
                flat.extend(row)
            return " ".join(str(int(x)) for x in flat)
        return " ".join(str(int(x)) for x in tokens)

    return str(tokens)


# ============================================================
# Feature extraction config
# ============================================================

@dataclass
class SweepConfig:
    fs: int = 16000
    context_sec: float = 3.0
    feats_type: str = "melspectrogram"   # "raw", "spectrogram", "melspectrogram"
    n_mels: int = 96
    win_size_sec: float = 0.025
    win_hop_sec: float = 0.010

    # continuous sweep hop
    sweep_hop_sec: float = 0.10

    # batching
    batch_size: int = 16
    num_workers: int = 0

    # CSV
    input_csv: str = ""
    output_csv: str = ""
    path_col: str = "path"

    # include trailing padded final window if audio length is not perfectly aligned
    include_last_partial_window: bool = True


# ============================================================
# Continuous sweep dataset
# ============================================================

class ContinuousSweepTokenizationDataset(Dataset):
    """
    Reuses the same audio loading and feature extraction interface as training,
    but:
      - no augmentation
      - deterministic continuous sweep windows
      - each item corresponds to one (file, window)

    Output per item:
      features, file_id, window_id
    """

    def __init__(
        self,
        audio_csv: str,
        fs: int = 16000,
        context_sec: float = 3.0,
        feats_type: str = "raw",
        rf_size: int = 400,
        stride: int = 160,
        n_mels: int = 80,
        sweep_hop_sec: float = 0.10,
        path_col: str = "path",
        include_last_partial_window: bool = True,
    ):
        super().__init__()

        self.audio_files = _read_paths_from_csv(audio_csv, path_col=path_col)
        self.audio_files = [p for p in self.audio_files if os.path.isfile(p)]

        if len(self.audio_files) == 0:
            raise ValueError("No valid audio files found in CSV.")

        self.fs = fs
        self.context_sec = float(context_sec)
        self.target_len = int(round(self.context_sec * self.fs))
        self.sweep_hop_sec = float(sweep_hop_sec)
        self.sweep_hop_len = int(round(self.sweep_hop_sec * self.fs))
        self.include_last_partial_window = bool(include_last_partial_window)

        if self.sweep_hop_len <= 0:
            raise ValueError(f"sweep_hop_sec too small: {sweep_hop_sec}")

        self.feats_type = feats_type
        self.rf_size = rf_size
        self.stride = stride
        self.n_mels = n_mels

        if feats_type == "melspectrogram":
            self.featextract = torch.nn.Sequential(
                T.MelSpectrogram(
                    sample_rate=self.fs,
                    n_fft=self.rf_size,
                    hop_length=self.stride,
                    n_mels=self.n_mels,
                    center=True,
                ),
                T.AmplitudeToDB(),
            )
        elif feats_type == "spectrogram":
            self.featextract = torch.nn.Sequential(
                T.Spectrogram(
                    n_fft=self.rf_size,
                    hop_length=self.stride,
                    center=True,
                    power=2.0,
                ),
                T.AmplitudeToDB(),
            )
        elif feats_type == "raw":
            self.featextract = None
        else:
            raise ValueError(
                "feats_type must be one of {'raw', 'spectrogram', 'melspectrogram'}"
            )

        # Build a flat index over all (file, window) pairs
        # Each entry: (path, file_id, window_id, start_sample)
        self.index: List[Tuple[str, str, int, int]] = []
        self._build_index()

    def _maybe_extract_feat(self, wav: torch.Tensor) -> torch.Tensor:
        if self.featextract is None:
            return wav
        return self.featextract(wav)

    def _compute_window_starts(self, num_samples: int) -> List[int]:
        """
        Continuous sweep starts:
          0, hop, 2*hop, ...

        If include_last_partial_window=True, ensures the tail is covered with
        one final padded window if necessary.
        """
        if num_samples <= 0:
            return [0]

        if num_samples <= self.target_len:
            return [0]

        starts = list(range(0, num_samples - self.target_len + 1, self.sweep_hop_len))

        if self.include_last_partial_window:
            last_regular_start = starts[-1]
            tail_uncovered = (last_regular_start + self.target_len) < num_samples
            if tail_uncovered:
                starts.append(num_samples - self.target_len if num_samples >= self.target_len else 0)

                # remove duplicates while keeping order
                deduped = []
                seen = set()
                for s in starts:
                    if s not in seen:
                        deduped.append(s)
                        seen.add(s)
                starts = deduped

        return starts

    def _build_index(self) -> None:
        for path in self.audio_files:
            info = torchaudio.info(path)
            file_num_samples = info.num_frames
            file_sr = info.sample_rate

            # Convert expected sample count after resampling if needed
            if file_sr != self.fs:
                duration_sec = file_num_samples / float(file_sr)
                num_samples = int(round(duration_sec * self.fs))
            else:
                num_samples = file_num_samples

            file_id = _path_to_file_id(path)
            starts = self._compute_window_starts(num_samples)

            for window_id, start in enumerate(starts):
                self.index.append((path, file_id, window_id, start))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int):
        path, file_id, window_id, start = self.index[idx]

        wav, sr = _load_audio_mono(path)
        wav, _ = _ensure_sr(wav, sr, self.fs)

        seg = _crop_or_pad_at_start(wav, start=start, target_len=self.target_len)
        feat = self._maybe_extract_feat(seg)

        return feat, file_id, window_id


def sweep_collate(batch):
    """
    batch items:
      feat, file_id, window_id
    """
    feats = torch.stack([x[0] for x in batch], dim=0)
    file_ids = [x[1] for x in batch]
    window_ids = [x[2] for x in batch]
    return feats, file_ids, window_ids


# ============================================================
# Main tokenization runner
# ============================================================
@torch.no_grad()
def run_continuous_sweep_tokenization(
    model,
    cfg: SweepConfig,
    device: Optional[torch.device] = None,
) -> None:
    """
    model.tokenize(...) must return ragged list[Tensor] or list[list[int]].
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rf_size = int(round(cfg.win_size_sec * cfg.fs))
    stride = int(round(cfg.win_hop_sec * cfg.fs))

    dataset = ContinuousSweepTokenizationDataset(
        audio_csv=cfg.input_csv,
        fs=cfg.fs,
        context_sec=cfg.context_sec,
        feats_type=cfg.feats_type,
        rf_size=rf_size,
        stride=stride,
        n_mels=cfg.n_mels,
        sweep_hop_sec=cfg.sweep_hop_sec,
        path_col=cfg.path_col,
        include_last_partial_window=cfg.include_last_partial_window,
    )

    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=sweep_collate,
    )

    model.eval()
    model.to(device)

    os.makedirs(os.path.dirname(cfg.output_csv) or ".", exist_ok=True)

    total_windows = len(dataset)

    with open(cfg.output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["file_id", "window_id", "tokenization"])

        pbar = tqdm(
            loader,
            total=len(loader),
            desc="Tokenizing windows",
            unit="batch",
        )

        written = 0

        for feats, file_ids, window_ids in pbar:
            feats = feats.to(device)

            batch_tokens = model.tokenize(
                feats,
                max_len=60,
                beam_size=4,
                strip_special_tokens=True,
            )

            print(batch_tokens[0])

            if len(batch_tokens) != len(file_ids):
                raise RuntimeError(
                    f"Batch size mismatch: got {len(batch_tokens)} outputs for {len(file_ids)} inputs."
                )

            for file_id, window_id, tok in zip(file_ids, window_ids, batch_tokens):
                tok_str = _serialize_tokens(tok)
                writer.writerow([file_id, window_id, tok_str])
                written += 1

            pbar.set_postfix(windows=f"{written}/{total_windows}")

    print(f"Saved tokenizations to: {cfg.output_csv}")

# ============================================================
# CLI
# ============================================================


def main():
    parser = argparse.ArgumentParser(description="Generate fixed-context sweep tokenizations.")
    parser.add_argument("--ckpt", required=True, help="Tokenizer checkpoint path.")
    parser.add_argument(
        "--model_type",
        default="AudioTokenizerAR_FINAL",
        choices=["AudioTokenizer", "AudioTokenizerARPrior", "AudioTokenizerAR_FINAL"],
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--path_col", default="path")
    parser.add_argument("--sweep_hop_sec", type=float, default=0.10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--include_last_partial_window",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()

    model_classes = {
        "AudioTokenizer": AudioTokenizer,
        "AudioTokenizerARPrior": AudioTokenizerARPrior,
        "AudioTokenizerAR_FINAL": AudioTokenizerAR_FINAL,
    }
    model = model_classes[args.model_type].load_from_pretrained(args.ckpt, args.device)

    cfg = SweepConfig(
        fs=16000,
        context_sec=3.0,
        feats_type="melspectrogram",
        n_mels=96,
        win_size_sec=0.025,
        win_hop_sec=0.010,
        sweep_hop_sec=args.sweep_hop_sec,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        input_csv=args.input_csv,
        output_csv=args.output_csv,
        path_col=args.path_col,
        include_last_partial_window=args.include_last_partial_window,
    )

    run_continuous_sweep_tokenization(
        model=model,
        cfg=cfg,
        device=torch.device(
            f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
        ),
    )


if __name__ == "__main__":
    main()
