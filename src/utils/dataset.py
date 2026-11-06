#!/usr/bin/env python3

import csv
import math
import os
import random
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

import torchaudio
import torchaudio.functional as AF
import torchaudio.transforms as T


FIXED_CONTEXT_SEC = 3.0


# -----------------------------------------------------------------------------
# Basic audio utilities
# -----------------------------------------------------------------------------

def _frame_rms(
    wav: torch.Tensor,
    frame_len: int,
    hop_len: int,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Frame-wise RMS for a mono waveform shaped (1, N)."""
    if wav.dim() != 2 or wav.size(0) != 1:
        raise ValueError(
            f"Expected wav shape (1, N), got {tuple(wav.shape)}"
        )

    signal = wav.squeeze(0)

    if signal.numel() < frame_len:
        signal = F.pad(
            signal,
            (0, frame_len - signal.numel()),
        )

    frames = signal.unfold(
        0,
        frame_len,
        hop_len,
    )

    return torch.sqrt(
        torch.mean(frames * frames, dim=-1) + eps
    )


def _active_ratio_from_rms(
    wav: torch.Tensor,
    frame_len: int,
    hop_len: int,
    rms_threshold: float,
) -> float:
    """Fraction of frames whose RMS is above ``rms_threshold``."""
    rms = _frame_rms(
        wav,
        frame_len=frame_len,
        hop_len=hop_len,
    )

    if rms.numel() == 0:
        return 0.0

    return float(
        (rms >= rms_threshold).float().mean().item()
    )


def _random_crop_or_pad_non_silent(
    wav: torch.Tensor,
    target_len: int,
    rng: Optional[random.Random] = None,
    num_tries: int = 12,
    frame_ms: float = 25.0,
    hop_ms: float = 10.0,
    sr: int = 16000,
    min_active_ratio: float = 0.30,
    rms_threshold: float = 0.01,
) -> Tuple[torch.Tensor, int]:
    """
    Return one fixed-length crop and its source start sample.

    If the source is shorter than ``target_len``, it is right-padded.
    """
    num_samples = wav.size(-1)

    if num_samples == target_len:
        return wav, 0

    if num_samples < target_len:
        return (
            F.pad(
                wav,
                (0, target_len - num_samples),
            ),
            0,
        )

    rr = rng if rng is not None else random

    frame_len = max(
        1,
        int(sr * frame_ms / 1000.0),
    )
    hop_len = max(
        1,
        int(sr * hop_ms / 1000.0),
    )

    max_start = num_samples - target_len

    best_crop = None
    best_start = 0
    best_ratio = -1.0

    for _ in range(num_tries):
        start = rr.randint(
            0,
            max_start,
        )
        crop = wav[
            ...,
            start : start + target_len,
        ]

        active_ratio = _active_ratio_from_rms(
            crop,
            frame_len=frame_len,
            hop_len=hop_len,
            rms_threshold=rms_threshold,
        )

        if active_ratio > best_ratio:
            best_crop = crop
            best_start = start
            best_ratio = active_ratio

        if active_ratio >= min_active_ratio:
            return crop, int(start)

    if best_crop is None:
        raise RuntimeError(
            "Non-silent crop search did not produce a crop."
        )

    return best_crop, int(best_start)


def _random_crop_or_pad(
    wav: torch.Tensor,
    target_len: int,
    rng: Optional[random.Random] = None,
) -> Tuple[torch.Tensor, int]:
    """Return one random fixed-length crop and its source start sample."""
    num_samples = wav.size(-1)

    if num_samples == target_len:
        return wav, 0

    if num_samples < target_len:
        return (
            F.pad(
                wav,
                (0, target_len - num_samples),
            ),
            0,
        )

    rr = rng if rng is not None else random

    start = rr.randint(
        0,
        num_samples - target_len,
    )

    return (
        wav[
            ...,
            start : start + target_len,
        ],
        int(start),
    )


def _load_audio_mono(
    path: str,
) -> Tuple[torch.Tensor, int]:
    """Load audio as float32 mono waveform shaped (1, N)."""
    wav, sample_rate = torchaudio.load(path)

    if wav.dtype != torch.float32:
        wav = wav.float()

    if wav.size(0) > 1:
        wav = wav.mean(
            dim=0,
            keepdim=True,
        )

    return wav, int(sample_rate)


def _ensure_sr(
    wav: torch.Tensor,
    sample_rate: int,
    target_sample_rate: int,
) -> Tuple[torch.Tensor, int]:
    """Resample to ``target_sample_rate`` when necessary."""
    if sample_rate == target_sample_rate:
        return wav, sample_rate

    return (
        AF.resample(
            wav,
            orig_freq=sample_rate,
            new_freq=target_sample_rate,
        ),
        target_sample_rate,
    )


def _rms(
    x: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    return torch.sqrt(
        torch.mean(x * x) + eps
    )


def _match_rms(
    x: torch.Tensor,
    reference: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Match global RMS while preserving zero-valued right padding."""
    x_rms = _rms(x, eps)
    reference_rms = _rms(
        reference,
        eps,
    )

    return x * (
        reference_rms
        / x_rms.clamp_min(eps)
    )


def _clip(
    x: torch.Tensor,
    limit: float = 1.0,
) -> torch.Tensor:
    return torch.clamp(
        x,
        -limit,
        limit,
    )


def _read_paths_from_csv(
    csv_path: str,
    path_col: str = "path",
) -> List[str]:
    """Read non-empty paths from a CSV column."""
    if not csv_path or not os.path.isfile(csv_path):
        raise ValueError(
            f"CSV not found: {csv_path}"
        )

    paths: List[str] = []

    with open(
        csv_path,
        "r",
        newline="",
    ) as handle:
        reader = csv.DictReader(handle)

        if (
            reader.fieldnames is None
            or path_col not in reader.fieldnames
        ):
            raise ValueError(
                f"CSV must contain column '{path_col}'. "
                f"Found: {reader.fieldnames}"
            )

        for row in reader:
            path = row[path_col].strip()
            if path:
                paths.append(path)

    return paths


def _recursive_find_audio_files(
    root_dir: str,
    exts: Tuple[str, ...] = (".flac",),
) -> List[str]:
    """Recursively collect files with the requested extensions."""
    if not root_dir or not os.path.isdir(root_dir):
        return []

    paths: List[str] = []

    for current_root, _, files in os.walk(
        root_dir
    ):
        for filename in files:
            if filename.lower().endswith(exts):
                paths.append(
                    os.path.join(
                        current_root,
                        filename,
                    )
                )

    return sorted(paths)


def _collect_audio_files_from_dirs(
    directories: List[str],
    exts: Tuple[str, ...] = (
        ".wav",
        ".flac",
    ),
) -> List[str]:
    """Recursively collect audio files from multiple directories."""
    paths: List[str] = []

    for directory in directories:
        if directory and os.path.isdir(directory):
            paths.extend(
                _recursive_find_audio_files(
                    directory,
                    exts=exts,
                )
            )

    return sorted(
        set(paths)
    )


# -----------------------------------------------------------------------------
# Positive-view augmentation
# -----------------------------------------------------------------------------

@dataclass
class AugConfig:
    """Mild positive-view augmentation configuration."""

    p_gain: float = 0.5
    gain_db_min: float = -3.0
    gain_db_max: float = 3.0

    p_noise: float = 0.5
    snr_db_min: float = 15.0
    snr_db_max: float = 30.0

    p_reverb: float = 0.15
    reverb_ir_max_sec: float = 0.25

    p_real_rir: float = 0.25

    p_bandpass: float = 0.15
    bp_low_min_hz: float = 80.0
    bp_low_max_hz: float = 250.0
    bp_high_min_hz: float = 3000.0
    bp_high_max_hz: float = 7600.0

    p_time_mask: float = 0.05
    time_mask_max_frac: float = 0.05

    # Optional temporal transformation.
    #
    # When enabled by the dataset, the positive is ALWAYS compressed slightly.
    # The rate is constrained to > 1, so temporal expansion is impossible.
    time_compress_rate_min: float = 1.02
    time_compress_rate_max: float = 1.08
    time_compress_n_fft: int = 512
    time_compress_hop_length: int = 128

    # Number of non-temporal augmentations. Time compression, when enabled,
    # is applied separately and is not part of this budget.
    max_ops_per_sample: int = 2

    peak_clip: float = 0.99


class AudioAugmenter:
    """
    Create a positive view from an already-extracted 3-second anchor.

    The anchor is never modified.

    If time compression is enabled:
      1. non-temporal augmentations are applied to a copy of the anchor;
      2. the positive is compressed with a phase vocoder using rate > 1;
      3. the compressed content is right-padded with zeros back to exactly
         the original 3-second sample count.

    Therefore no augmentation can import content from outside the anchor crop.
    """

    def __init__(
        self,
        sr: int,
        cfg: AugConfig,
        noise_files: Optional[List[str]] = None,
        rir_files: Optional[List[str]] = None,
        use_time_compression: bool = False,
        audio_cache_max: int = 64,
    ):
        self.sr = int(sr)
        self.cfg = cfg
        self.noise_files = noise_files or []
        self.rir_files = rir_files or []
        self.use_time_compression = bool(
            use_time_compression
        )

        self._audio_cache: Dict[
            str,
            torch.Tensor,
        ] = {}
        self._audio_cache_max = int(
            audio_cache_max
        )

        self._validate_config()

    def _validate_probability(
        self,
        name: str,
        value: float,
    ) -> None:
        if not 0.0 <= value <= 1.0:
            raise ValueError(
                f"{name} must be in [0, 1], got {value}"
            )

    def _validate_config(self) -> None:
        for name in (
            "p_gain",
            "p_noise",
            "p_reverb",
            "p_real_rir",
            "p_bandpass",
            "p_time_mask",
        ):
            self._validate_probability(
                name,
                float(
                    getattr(
                        self.cfg,
                        name,
                    )
                ),
            )

        if self.cfg.max_ops_per_sample < 0:
            raise ValueError(
                "max_ops_per_sample must be >= 0"
            )

        # Compression must be strict. rate == 1 would leave duration unchanged;
        # rate < 1 would expand the 3-second crop and is prohibited.
        if (
            self.cfg.time_compress_rate_min
            <= 1.0
        ):
            raise ValueError(
                "time_compress_rate_min must be > 1.0. "
                "Only positive-view time compression is supported."
            )

        if (
            self.cfg.time_compress_rate_max
            < self.cfg.time_compress_rate_min
        ):
            raise ValueError(
                "time_compress_rate_max must be >= "
                "time_compress_rate_min"
            )

        if (
            self.cfg.time_compress_n_fft < 2
            or self.cfg.time_compress_hop_length < 1
        ):
            raise ValueError(
                "Invalid STFT configuration for time compression."
            )

        if (
            self.cfg.time_compress_hop_length
            > self.cfg.time_compress_n_fft
        ):
            raise ValueError(
                "time_compress_hop_length must be <= "
                "time_compress_n_fft"
            )

    def _get_cached_audio(
        self,
        path: str,
    ) -> torch.Tensor:
        if path in self._audio_cache:
            return self._audio_cache[path]

        wav, sample_rate = _load_audio_mono(
            path
        )
        wav, _ = _ensure_sr(
            wav,
            sample_rate,
            self.sr,
        )

        if (
            len(self._audio_cache)
            >= self._audio_cache_max
        ):
            self._audio_cache.pop(
                next(
                    iter(
                        self._audio_cache
                    )
                )
            )

        self._audio_cache[path] = wav
        return wav

    def _get_noise(
        self,
        target_len: int,
        rng: Optional[random.Random] = None,
    ) -> Optional[torch.Tensor]:
        if not self.noise_files:
            return None

        rr = (
            rng
            if rng is not None
            else random
        )

        path = rr.choice(
            self.noise_files
        )
        noise = self._get_cached_audio(
            path
        )
        noise, _ = _random_crop_or_pad(
            noise,
            target_len,
            rng=rr,
        )

        return noise

    def _get_rir(
        self,
        rng: Optional[random.Random] = None,
    ) -> Optional[torch.Tensor]:
        if not self.rir_files:
            return None

        rr = (
            rng
            if rng is not None
            else random
        )

        path = rr.choice(
            self.rir_files
        )
        rir = self._get_cached_audio(
            path
        )

        return rir[
            ...,
            : int(0.4 * self.sr),
        ]

    def _add_noise_snr(
        self,
        x: torch.Tensor,
        rng: Optional[random.Random] = None,
    ) -> torch.Tensor:
        rr = (
            rng
            if rng is not None
            else random
        )

        noise = self._get_noise(
            x.size(-1),
            rng=rr,
        )

        if noise is None:
            noise = (
                torch.randn_like(x)
                * 0.01
            )

        snr_db = rr.uniform(
            self.cfg.snr_db_min,
            self.cfg.snr_db_max,
        )

        target_noise_rms = (
            _rms(x)
            / (
                10.0
                ** (
                    snr_db
                    / 20.0
                )
            )
        )

        noise = noise * (
            target_noise_rms
            / _rms(noise).clamp_min(1e-8)
        )

        return x + noise

    def _gain(
        self,
        x: torch.Tensor,
        rng: Optional[random.Random] = None,
    ) -> torch.Tensor:
        rr = (
            rng
            if rng is not None
            else random
        )

        gain_db = rr.uniform(
            self.cfg.gain_db_min,
            self.cfg.gain_db_max,
        )

        return x * (
            10.0
            ** (
                gain_db
                / 20.0
            )
        )

    def _bandpass(
        self,
        x: torch.Tensor,
        rng: Optional[random.Random] = None,
    ) -> torch.Tensor:
        rr = (
            rng
            if rng is not None
            else random
        )

        low = rr.uniform(
            self.cfg.bp_low_min_hz,
            self.cfg.bp_low_max_hz,
        )
        high = rr.uniform(
            self.cfg.bp_high_min_hz,
            self.cfg.bp_high_max_hz,
        )

        if high <= low + 100.0:
            high = low + 100.0

        y = AF.highpass_biquad(
            x,
            sample_rate=self.sr,
            cutoff_freq=low,
        )

        return AF.lowpass_biquad(
            y,
            sample_rate=self.sr,
            cutoff_freq=high,
        )

    def _synthetic_reverb(
        self,
        x: torch.Tensor,
        rng: Optional[random.Random] = None,
    ) -> torch.Tensor:
        rr = (
            rng
            if rng is not None
            else random
        )

        ir_len = max(
            16,
            int(
                self.cfg.reverb_ir_max_sec
                * self.sr
                * rr.uniform(
                    0.3,
                    1.0,
                )
            ),
        )

        time = torch.linspace(
            0,
            1,
            steps=ir_len,
            device=x.device,
            dtype=x.dtype,
        )
        decay = torch.exp(
            -6.0 * time
        )

        impulse_response = (
            torch.randn(
                1,
                ir_len,
                device=x.device,
                dtype=x.dtype,
            )
            * decay
        )
        impulse_response = (
            impulse_response
            / _rms(
                impulse_response
            ).clamp_min(1e-8)
        )

        y = F.conv1d(
            x.unsqueeze(0),
            impulse_response.unsqueeze(1),
            padding=ir_len // 2,
        ).squeeze(0)

        y = y[
            ...,
            : x.size(-1),
        ]

        wet = rr.uniform(
            0.03,
            0.10,
        )

        return (
            (1.0 - wet) * x
            + wet * y
        )

    def _real_rir_reverb(
        self,
        x: torch.Tensor,
        rng: Optional[random.Random] = None,
    ) -> torch.Tensor:
        rr = (
            rng
            if rng is not None
            else random
        )

        rir = self._get_rir(
            rng=rr
        )

        if (
            rir is None
            or rir.numel() == 0
        ):
            return self._synthetic_reverb(
                x,
                rng=rr,
            )

        rir = rir - rir.mean(
            dim=-1,
            keepdim=True,
        )
        rir = (
            rir
            / _rms(rir).clamp_min(1e-8)
        )

        y = F.conv1d(
            x.unsqueeze(0),
            rir.unsqueeze(1),
            padding=rir.size(-1) // 2,
        ).squeeze(0)

        y = y[
            ...,
            : x.size(-1),
        ]

        wet = rr.uniform(
            0.03,
            0.12,
        )

        return (
            (1.0 - wet) * x
            + wet * y
        )

    def _time_mask(
        self,
        x: torch.Tensor,
        rng: Optional[random.Random] = None,
    ) -> torch.Tensor:
        rr = (
            rng
            if rng is not None
            else random
        )

        num_samples = x.size(-1)
        max_width = int(
            self.cfg.time_mask_max_frac
            * num_samples
        )

        if max_width <= 1:
            return x

        width = rr.randint(
            1,
            max_width,
        )
        start = rr.randint(
            0,
            num_samples - width,
        )

        y = x.clone()
        y[
            ...,
            start : start + width,
        ] = 0.0

        return y

    def _time_compress(
        self,
        x: torch.Tensor,
        rng: Optional[random.Random] = None,
    ) -> Tuple[torch.Tensor, float, int]:
        """
        Slightly compress ``x`` in time and right-pad back to its input length.

        A phase vocoder is used so this is a time-stretch/compression operation
        rather than sample-rate conversion.

        Returns:
            padded:
                waveform with exactly the same sample count as ``x``.
            rate:
                compression rate (> 1).
            content_num_samples:
                number of non-padding samples before right padding.
        """
        if x.dim() != 2 or x.size(0) != 1:
            raise ValueError(
                "Time compression expects mono waveform shape (1, N)."
            )

        rr = (
            rng
            if rng is not None
            else random
        )

        rate = rr.uniform(
            self.cfg.time_compress_rate_min,
            self.cfg.time_compress_rate_max,
        )

        if rate <= 1.0:
            raise RuntimeError(
                "Internal error: positive-view time-compression rate "
                f"must be > 1, got {rate}."
            )

        target_len = int(
            x.size(-1)
        )
        content_len = max(
            1,
            int(
                round(
                    target_len
                    / rate
                )
            ),
        )

        # Since rate > 1, the content must become strictly shorter.
        content_len = min(
            content_len,
            target_len - 1,
        )

        n_fft = int(
            self.cfg.time_compress_n_fft
        )
        hop_length = int(
            self.cfg.time_compress_hop_length
        )

        window = torch.hann_window(
            n_fft,
            device=x.device,
            dtype=x.dtype,
        )

        spectrum = torch.stft(
            x.squeeze(0),
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=n_fft,
            window=window,
            center=True,
            return_complex=True,
        )

        phase_advance = torch.linspace(
            0,
            math.pi * hop_length,
            spectrum.size(-2),
            device=spectrum.device,
            dtype=x.dtype,
        ).unsqueeze(-1)

        compressed_spectrum = AF.phase_vocoder(
            spectrum,
            rate=rate,
            phase_advance=phase_advance,
        )

        compressed = torch.istft(
            compressed_spectrum,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=n_fft,
            window=window,
            center=True,
            length=content_len,
        ).unsqueeze(0)

        if compressed.size(-1) >= target_len:
            raise RuntimeError(
                "Time compression did not shorten the positive view."
            )

        right_pad = (
            target_len
            - compressed.size(-1)
        )

        padded = F.pad(
            compressed,
            (0, right_pad),
        )

        if padded.size(-1) != target_len:
            raise RuntimeError(
                "Positive-view padding failed to restore fixed context length."
            )

        return (
            padded,
            float(rate),
            int(compressed.size(-1)),
        )

    def __call__(
        self,
        anchor: torch.Tensor,
        rng: Optional[random.Random] = None,
        return_info: bool = False,
    ) -> Union[
        torch.Tensor,
        Tuple[
            torch.Tensor,
            Dict[str, Union[bool, float, int]],
        ],
    ]:
        """
        Build the positive view from the fixed-length anchor.

        When ``use_time_compression=True``, time compression is mandatory for
        every positive view. It is applied last and followed only by RMS
        matching and clipping, so the final right-padding remains zero.
        """
        if anchor.dim() != 2 or anchor.size(0) != 1:
            raise ValueError(
                "anchor must have shape (1, N)"
            )

        rr = (
            rng
            if rng is not None
            else random
        )

        # Work on a clone so no augmentation can alter the anchor tensor.
        positive = anchor.clone()

        operations = []

        if rr.random() < self.cfg.p_gain:
            operations.append(
                self._gain
            )
        if rr.random() < self.cfg.p_bandpass:
            operations.append(
                self._bandpass
            )
        if rr.random() < self.cfg.p_reverb:
            operations.append(
                self._synthetic_reverb
            )
        if rr.random() < self.cfg.p_real_rir:
            operations.append(
                self._real_rir_reverb
            )
        if rr.random() < self.cfg.p_noise:
            operations.append(
                self._add_noise_snr
            )
        if rr.random() < self.cfg.p_time_mask:
            operations.append(
                self._time_mask
            )

        rr.shuffle(
            operations
        )

        operations = operations[
            : self.cfg.max_ops_per_sample
        ]

        for operation in operations:
            positive = operation(
                positive,
                rng=rr,
            )

        time_compression_applied = False
        time_compression_rate = 1.0
        positive_content_num_samples = int(
            positive.size(-1)
        )

        if self.use_time_compression:
            (
                positive,
                time_compression_rate,
                positive_content_num_samples,
            ) = self._time_compress(
                positive,
                rng=rr,
            )
            time_compression_applied = True

        # Scaling/clipping preserve the zero right-padding created above.
        positive = _match_rms(
            positive,
            anchor,
        )
        positive = _clip(
            positive,
            self.cfg.peak_clip,
        )

        if positive.size(-1) != anchor.size(-1):
            raise RuntimeError(
                "Positive and anchor must have identical fixed context length."
            )

        info: Dict[
            str,
            Union[bool, float, int],
        ] = {
            "time_compression_applied": (
                time_compression_applied
            ),
            "time_compression_rate": float(
                time_compression_rate
            ),
            "positive_content_num_samples": int(
                positive_content_num_samples
            ),
            "positive_right_pad_samples": int(
                anchor.size(-1)
                - positive_content_num_samples
            ),
        }

        if return_info:
            return positive, info

        return positive


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------

class AudioPairAugDataset(Dataset):
    """
    Paired fixed-context dataset.

    Invariant:
        anchor length   = exactly 3.0 seconds
        positive length = exactly 3.0 seconds

    The 3-second anchor is cropped first. The positive is created exclusively
    from that already-cropped anchor.

    If ``use_time_compression=True``:
        * the anchor remains untouched;
        * every positive view is mildly compressed with rate > 1;
        * the shortened positive is right-padded with zeros back to 3 seconds.

    Validation uses a deterministic per-item RNG so its crop, augmentation,
    and compression rate remain fixed across epochs.
    """

    def __init__(
        self,
        audio_files: Optional[List[str]] = None,
        audio_csv: Optional[str] = None,
        audio_root_dir: Optional[str] = None,
        fs: int = 16000,
        context_sec: float = FIXED_CONTEXT_SEC,
        aug_cfg: Optional[AugConfig] = None,
        use_time_compression: bool = False,
        add_noise: bool = True,
        noise_dir: Optional[str] = None,
        noise_dirs: Optional[List[str]] = None,
        noise_csv: Optional[str] = None,
        rir_dir: Optional[str] = None,
        rir_dirs: Optional[List[str]] = None,
        rir_csv: Optional[str] = None,
        distortion_root: Optional[str] = None,
        use_default_distortions: bool = False,
        return_path: bool = False,
        feats_type: str = "raw",
        rf_size: int = 400,
        stride: int = 160,
        n_mels: int = 80,
        split: str = "train",
        val_seed: int = 1234,
        use_non_silent_crop: bool = True,
        non_silent_num_tries: int = 20,
        non_silent_frame_ms: float = 25.0,
        non_silent_hop_ms: float = 10.0,
        non_silent_min_active_ratio: float = 0.50,
        non_silent_rms_threshold: float = 0.008,
        return_metadata: bool = False,
    ):
        super().__init__()

        if split not in {
            "train",
            "val",
        }:
            raise ValueError(
                "split must be 'train' or 'val'"
            )

        if not math.isclose(
            float(context_sec),
            FIXED_CONTEXT_SEC,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError(
                "This dataset is fixed to a 3.0-second context. "
                f"Received context_sec={context_sec}."
            )

        if fs <= 0:
            raise ValueError(
                "fs must be positive"
            )

        self.split = split
        self.is_train = (
            split == "train"
        )
        self.val_seed = int(
            val_seed
        )
        self.return_metadata = bool(
            return_metadata
        )
        self.return_path = bool(
            return_path
        )

        self.fs = int(fs)
        self.context_sec = FIXED_CONTEXT_SEC
        self.target_len = int(
            round(
                FIXED_CONTEXT_SEC
                * self.fs
            )
        )

        collected_audio: List[str] = []

        if audio_files is not None:
            collected_audio.extend(
                path
                for path in audio_files
                if (
                    isinstance(
                        path,
                        str,
                    )
                    and path
                )
            )

        if audio_csv is not None:
            collected_audio.extend(
                _read_paths_from_csv(
                    audio_csv,
                    path_col="path",
                )
            )

        if audio_root_dir is not None:
            collected_audio.extend(
                _recursive_find_audio_files(
                    audio_root_dir,
                    exts=(".flac",),
                )
            )

        self.audio_files = sorted(
            {
                path
                for path in collected_audio
                if os.path.isfile(path)
            }
        )

        if not self.audio_files:
            raise ValueError(
                "No valid audio files were found."
            )

        config = (
            aug_cfg
            if aug_cfg is not None
            else AugConfig()
        )

        # ``add_noise=False`` truly disables the noise operation, including
        # the synthetic fallback.
        if not add_noise:
            config = replace(
                config,
                p_noise=0.0,
            )

        if (
            use_default_distortions
            and distortion_root
        ):
            if noise_dirs is None:
                noise_dirs = [
                    os.path.join(
                        distortion_root,
                        "noise_16k",
                    ),
                    os.path.join(
                        distortion_root,
                        "pointsource_noises",
                    ),
                    os.path.join(
                        distortion_root,
                        "real_rirs_isotropic_noises",
                    ),
                ]

            if rir_dirs is None:
                rir_dirs = [
                    os.path.join(
                        distortion_root,
                        "rir_16k",
                    )
                ]

        collected_noise: List[str] = []
        collected_rirs: List[str] = []

        if (
            add_noise
            and noise_dir
            and os.path.isdir(noise_dir)
        ):
            collected_noise.extend(
                _recursive_find_audio_files(
                    noise_dir,
                    exts=(
                        ".wav",
                        ".flac",
                    ),
                )
            )

        if (
            add_noise
            and noise_dirs is not None
        ):
            collected_noise.extend(
                _collect_audio_files_from_dirs(
                    noise_dirs,
                    exts=(
                        ".wav",
                        ".flac",
                    ),
                )
            )

        if (
            add_noise
            and noise_csv is not None
        ):
            collected_noise.extend(
                _read_paths_from_csv(
                    noise_csv,
                    path_col="path",
                )
            )

        if (
            rir_dir
            and os.path.isdir(rir_dir)
        ):
            collected_rirs.extend(
                _recursive_find_audio_files(
                    rir_dir,
                    exts=(
                        ".wav",
                        ".flac",
                    ),
                )
            )

        if rir_dirs is not None:
            collected_rirs.extend(
                _collect_audio_files_from_dirs(
                    rir_dirs,
                    exts=(
                        ".wav",
                        ".flac",
                    ),
                )
            )

        if rir_csv is not None:
            collected_rirs.extend(
                _read_paths_from_csv(
                    rir_csv,
                    path_col="path",
                )
            )

        self.noise_files = sorted(
            {
                path
                for path in collected_noise
                if os.path.isfile(path)
            }
        )
        self.rir_files = sorted(
            {
                path
                for path in collected_rirs
                if os.path.isfile(path)
            }
        )

        self.augment = AudioAugmenter(
            sr=self.fs,
            cfg=config,
            noise_files=self.noise_files,
            rir_files=self.rir_files,
            use_time_compression=(
                use_time_compression
            ),
        )
        self.use_time_compression = bool(
            use_time_compression
        )

        self.feats_type = feats_type
        self.rf_size = int(
            rf_size
        )
        self.stride = int(
            stride
        )
        self.n_mels = int(
            n_mels
        )

        if feats_type == "melspectrogram":
            if self.n_mels <= 0:
                raise ValueError(
                    "n_mels must be positive"
                )

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
                "feats_type must be 'melspectrogram', "
                "'spectrogram', or 'raw'."
            )

        self.use_non_silent_crop = bool(
            use_non_silent_crop
        )
        self.non_silent_num_tries = int(
            non_silent_num_tries
        )
        self.non_silent_frame_ms = float(
            non_silent_frame_ms
        )
        self.non_silent_hop_ms = float(
            non_silent_hop_ms
        )
        self.non_silent_min_active_ratio = float(
            non_silent_min_active_ratio
        )
        self.non_silent_rms_threshold = float(
            non_silent_rms_threshold
        )

    def __len__(self) -> int:
        return len(
            self.audio_files
        )

    def _maybe_extract_feat(
        self,
        wav: torch.Tensor,
    ) -> torch.Tensor:
        if self.featextract is None:
            return wav

        return self.featextract(
            wav
        )

    def _get_rng_for_idx(
        self,
        idx: int,
    ) -> random.Random:
        """Fixed validation RNG for reproducible paired views."""
        return random.Random(
            self.val_seed
            + int(idx)
        )

    def _sample_crop(
        self,
        wav: torch.Tensor,
        rng: Optional[random.Random] = None,
    ) -> Tuple[torch.Tensor, int]:
        if self.use_non_silent_crop:
            crop, start = (
                _random_crop_or_pad_non_silent(
                    wav,
                    target_len=self.target_len,
                    rng=rng,
                    num_tries=self.non_silent_num_tries,
                    frame_ms=self.non_silent_frame_ms,
                    hop_ms=self.non_silent_hop_ms,
                    sr=self.fs,
                    min_active_ratio=(
                        self.non_silent_min_active_ratio
                    ),
                    rms_threshold=(
                        self.non_silent_rms_threshold
                    ),
                )
            )
        else:
            crop, start = (
                _random_crop_or_pad(
                    wav,
                    self.target_len,
                    rng=rng,
                )
            )

        if crop.size(-1) != self.target_len:
            raise RuntimeError(
                "Anchor crop is not exactly 3 seconds."
            )

        return crop, start

    def __getitem__(
        self,
        idx: int,
    ):
        path = self.audio_files[
            idx
        ]

        wav, source_sr = _load_audio_mono(
            path
        )
        wav, model_sr = _ensure_sr(
            wav,
            source_sr,
            self.fs,
        )

        stem, _ = os.path.splitext(
            path
        )
        phn_path = next(
            (
                candidate
                for candidate in (
                    stem + ".PHN",
                    stem + ".phn",
                )
                if os.path.isfile(
                    candidate
                )
            ),
            None,
        )

        if (
            phn_path is not None
            and source_sr != model_sr
        ):
            raise ValueError(
                "PHN alignment requires the source and model "
                "sample timebases to match."
            )

        rng = (
            None
            if self.is_train
            else self._get_rng_for_idx(
                idx
            )
        )

        # IMPORTANT:
        # The 3-second anchor is extracted first. The positive is constructed
        # only from this fixed crop; no later operation sees the source audio.
        anchor, crop_start_sample = (
            self._sample_crop(
                wav,
                rng=rng,
            )
        )

        anchor_reference = anchor.clone()

        positive, augmentation_info = (
            self.augment(
                anchor,
                rng=rng,
                return_info=True,
            )
        )

        # Hard invariants used by all stages of PairAlign training.
        if anchor.size(-1) != self.target_len:
            raise RuntimeError(
                "Anchor context changed after cropping."
            )
        if positive.size(-1) != self.target_len:
            raise RuntimeError(
                "Positive context is not exactly 3 seconds."
            )
        if not torch.equal(
            anchor,
            anchor_reference,
        ):
            raise RuntimeError(
                "Positive-view augmentation modified the anchor in place."
            )

        if self.use_time_compression:
            if not bool(
                augmentation_info[
                    "time_compression_applied"
                ]
            ):
                raise RuntimeError(
                    "Time compression was enabled but was not applied."
                )

            compression_rate = float(
                augmentation_info[
                    "time_compression_rate"
                ]
            )

            if compression_rate <= 1.0:
                raise RuntimeError(
                    "Positive view must be compressed, never expanded."
                )

            right_pad_samples = int(
                augmentation_info[
                    "positive_right_pad_samples"
                ]
            )

            if right_pad_samples <= 0:
                raise RuntimeError(
                    "Compressed positive must contain non-empty right padding."
                )

        anchor_out = self._maybe_extract_feat(
            anchor
        )
        positive_out = self._maybe_extract_feat(
            positive
        )

        metadata = {
            "audio_path": path,
            "phn_path": phn_path,
            "crop_start_sample": int(
                crop_start_sample
            ),
            "anc_crop_start_sample": int(
                crop_start_sample
            ),
            "pos_crop_start_sample": int(
                crop_start_sample
            ),
            "segment_num_samples": int(
                self.target_len
            ),
            "anchor_num_samples": int(
                anchor.size(-1)
            ),
            "positive_num_samples": int(
                positive.size(-1)
            ),
            "sample_rate": int(
                model_sr
            ),
            "phn_sample_rate": int(
                source_sr
            ),
            "context_sec": float(
                FIXED_CONTEXT_SEC
            ),
            "time_compression_applied": bool(
                augmentation_info[
                    "time_compression_applied"
                ]
            ),
            "time_compression_rate": float(
                augmentation_info[
                    "time_compression_rate"
                ]
            ),
            "positive_content_num_samples": int(
                augmentation_info[
                    "positive_content_num_samples"
                ]
            ),
            "positive_right_pad_samples": int(
                augmentation_info[
                    "positive_right_pad_samples"
                ]
            ),
            # A compressed positive no longer shares the anchor's linear time
            # axis, so exact PHN-frame alignment must not be assumed for it.
            "pos_time_preserving": (
                not bool(
                    augmentation_info[
                        "time_compression_applied"
                    ]
                )
            ),
        }

        if self.return_metadata:
            return (
                anchor_out,
                positive_out,
                path,
                metadata,
            )

        if self.return_path:
            return (
                anchor_out,
                positive_out,
                path,
            )

        return (
            anchor_out,
            positive_out,
        )
