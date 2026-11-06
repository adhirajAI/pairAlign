import argparse
import os
import pickle
from pathlib import Path

import torch
import yaml
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichModelSummary,
    RichProgressBar,
)
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from train import AudioTokenizer, AudioTokenizerARPrior, AudioTokenizerAR_FINAL
from utils.dataset import AudioPairAugDataset


torch.set_float32_matmul_precision("medium")

MODEL_REGISTRY = {
    "AudioTokenizer": AudioTokenizer,
    "AudioTokenizerARPrior": AudioTokenizerARPrior,
    "AudioTokenizerAR_FINAL": AudioTokenizerAR_FINAL,
}

DEFAULT_MONITOR = {
    "AudioTokenizer": ("valid_acc_epoch", "max"),
    "AudioTokenizerARPrior": ("valid_acc_epoch", "max"),
    "AudioTokenizerAR_FINAL": ("valid_student_acc_epoch", "max"),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Train one of the supported audio tokenizers.")
    parser.add_argument(
        "--root_dir",
        type=str,
        default=".",
        help="Project root used for checkpoints and config lookup.",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default=None,
        help="Checkpoint run name. Overrides config 'run_name' when provided.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="mainFull.yaml",
        help="Config path or config filename.",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=sorted(MODEL_REGISTRY),
        default=None,
        help="Optional tokenizer override. Otherwise uses config 'model_name'.",
    )
    parser.add_argument(
        "-c",
        "--train_checkpoint",
        type=str,
        default=None,
        help="Lightning checkpoint used to resume the selected training stage.",
    )
    parser.add_argument(
        "--device",
        type=int,
        nargs="+",
        default=[0],
        help="GPU device indices passed to Lightning.",
    )
    return parser.parse_args()


def resolve_config_path(root_dir: str, config_arg: str) -> Path:
    config_path = Path(config_arg).expanduser()
    candidates = [
        config_path,
        Path(root_dir).expanduser() / config_arg,
        Path(root_dir).expanduser() / "config" / config_arg,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    checked = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Could not find config file. Checked: {checked}")


def load_config(root_dir: str, config_arg: str):
    config_path = resolve_config_path(root_dir, config_arg)
    with config_path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config must contain a YAML mapping: {config_path}")
    return cfg, config_path


def build_dataset(cfg, split: str):
    split_cfg = cfg[split]
    data_cfg = cfg.get("data", {})
    noise_dir = split_cfg.get("noise_dir_path")

    return AudioPairAugDataset(
        audio_csv=split_cfg["dataset_csv"],
        fs=cfg["fs"],
        context_sec=cfg["seglen"],
        add_noise=bool(data_cfg.get("add_noise", True)),
        use_time_compression=bool(data_cfg.get("use_time_compression", False)),
        return_path=False,
        feats_type=cfg.get("feats_type", "melspectrogram"),
        rf_size=int(round(cfg["win_size"] * cfg["fs"])),
        stride=int(round(cfg["win_hop"] * cfg["fs"])),
        n_mels=cfg["n_mels"],
        split="train" if split == "train" else "val",
        distortion_root=noise_dir,
        use_default_distortions=noise_dir is not None,
        val_seed=int(cfg.get("validation_seed", 1234)),
    )


def build_model(cfg, cli_model=None):
    model_name = cli_model or cfg.get("model_name", "AudioTokenizerAR_FINAL")
    if model_name not in MODEL_REGISTRY:
        valid = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(f"Unknown model_name '{model_name}'. Choose one of: {valid}")
    cfg["model_name"] = model_name
    return MODEL_REGISTRY[model_name](cfg), model_name


def build_checkpoint_callback(cfg, model_name: str):
    trainer_cfg = cfg.get("trainer", {})
    default_monitor, default_mode = DEFAULT_MONITOR[model_name]
    monitor = trainer_cfg.get("checkpoint_monitor") or default_monitor
    mode = trainer_cfg.get("checkpoint_mode") or default_mode
    return ModelCheckpoint(
        filename="{epoch:04d}-{step}",
        monitor=monitor,
        mode=mode,
        save_top_k=int(trainer_cfg.get("save_top_k", 1)),
        save_last=True,
    )


def main():
    args = parse_args()
    cfg, config_path = load_config(args.root_dir, args.config)

    if args.model is not None:
        cfg["model_name"] = args.model
    if args.train_checkpoint is not None:
        cfg["checkpoint"] = args.train_checkpoint

    seed_everything(int(cfg.get("seed", 240)), workers=True)

    run_name = args.ckpt_dir or cfg.get("run_name") or cfg.get("model_name", "tokenizer")
    checkpoint_root = Path(args.root_dir).expanduser() / "checkpoints"
    run_root = checkpoint_root / run_name
    run_root.mkdir(parents=True, exist_ok=True)

    with (run_root / "params.pkl").open("wb") as handle:
        pickle.dump(cfg, handle)

    print(f"Reading config: {config_path}")
    print(f"Selected tokenizer: {cfg.get('model_name', 'AudioTokenizerAR_FINAL')}")

    train_dataset = build_dataset(cfg, "train")
    valid_dataset = build_dataset(cfg, "valid")

    print(
        "Dataset sizes: "
        f"train={len(train_dataset)}, valid={len(valid_dataset)}"
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=int(cfg["batchsize"]),
        shuffle=True,
        drop_last=True,
        num_workers=int(cfg["load_workers"]),
        pin_memory=True,
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=int(cfg["batchsize"]),
        shuffle=False,
        drop_last=True,
        num_workers=int(cfg["load_workers"]),
        pin_memory=True,
    )

    model, model_name = build_model(cfg, cli_model=args.model)

    lr_monitor = LearningRateMonitor(logging_interval="step")
    checkpoint_callback = build_checkpoint_callback(cfg, model_name)
    progress_bar = RichProgressBar(
        refresh_rate=1,
        theme=RichProgressBarTheme(
            description="#0066ff",
            progress_bar="#ccff33",
            progress_bar_finished="#ccff33",
            progress_bar_pulse="#ccff33",
            batch_progress="#ff4d4d",
            time="#ff4d4d",
            processing_speed="#ff4d4d",
            metrics="white",
        ),
    )

    version = "_".join(
        [
            f"bsz:{cfg['batchsize']}",
            f"lr:{cfg['lr']}",
            f"seg:{cfg['seglen']}",
        ]
    )
    logger = TensorBoardLogger(
        save_dir=str(checkpoint_root),
        version=version,
        name=run_name,
        default_hp_metric=False,
    )

    trainer_cfg = cfg.get("trainer", {})
    trainer = Trainer(
        devices=args.device,
        accelerator=trainer_cfg.get("accelerator", "gpu"),
        callbacks=[
            lr_monitor,
            checkpoint_callback,
            progress_bar,
            RichModelSummary(),
        ],
        logger=logger,
        check_val_every_n_epoch=int(trainer_cfg.get("check_val_every_n_epoch", 1)),
        max_epochs=int(trainer_cfg.get("max_epochs", -1)),
        deterministic=bool(trainer_cfg.get("deterministic", False)),
        default_root_dir=str(checkpoint_root),
    )

    resume_checkpoint = args.train_checkpoint or cfg.get("checkpoint")
    if resume_checkpoint:
        print(f"Resuming training from: {resume_checkpoint}")
    else:
        print("Training from scratch for the selected stage.")

    trainer.fit(
        model,
        train_dataloader,
        valid_dataloader,
        ckpt_path=resume_checkpoint,
    )


if __name__ == "__main__":
    main()
