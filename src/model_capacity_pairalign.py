#!/usr/bin/env python3

import argparse

from transformers import HubertModel, WavLMModel

from train import AudioTokenizerAR_FINAL


def count_params(model) -> int:
    """Count every parameter registered in a module."""
    return sum(parameter.numel() for parameter in model.parameters())


def format_params(count: int) -> str:
    if count >= 1_000_000_000:
        return f"{count / 1_000_000_000:.3f} B"
    if count >= 1_000_000:
        return f"{count / 1_000_000:.3f} M"
    if count >= 1_000:
        return f"{count / 1_000:.3f} K"
    return str(count)


def print_row(name: str, count: int) -> None:
    print(
        f"{name:<44}"
        f"{count:>18,}"
        f"{format_params(count):>15}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Report parameter counts for HuBERT Base, WavLM Large, "
            "and the PairAlign teacher encoder/decoder used at inference."
        )
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the PairAlign checkpoint.",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help=(
            "GPU index passed to "
            "AudioTokenizerAR_FINAL.load_from_pretrained."
        ),
    )

    args = parser.parse_args()

    print("\nLoading HuBERT Base...")
    hubert = HubertModel.from_pretrained(
        "facebook/hubert-base-ls960"
    )
    hubert_params = count_params(hubert)
    del hubert

    print("Loading WavLM Large...")
    wavlm = WavLMModel.from_pretrained(
        "microsoft/wavlm-large"
    )
    wavlm_params = count_params(wavlm)
    del wavlm

    print("Loading PairAlign tokenizer...")
    model = AudioTokenizerAR_FINAL.load_from_pretrained(
        args.checkpoint,
        args.device,
    )

    # Count every parameter in the teacher encoder and decoder used by
    # PairAlign inference. No parameters or submodules are excluded.
    pairalign_encoder_params = count_params(
        model.encoder_teacher
    )
    pairalign_decoder_params = count_params(
        model.decoder_teacher
    )
    pairalign_encoder_decoder_params = (
        pairalign_encoder_params
        + pairalign_decoder_params
    )

    print()
    print("=" * 79)
    print("MODEL CAPACITY REPORT")
    print("=" * 79)

    print(
        f"{'Model / Component':<44}"
        f"{'Exact Parameters':>18}"
        f"{'Capacity':>15}"
    )

    print("-" * 79)

    print_row("HuBERT Base", hubert_params)
    print_row("WavLM Large", wavlm_params)

    print("-" * 79)

    print_row(
        "PairAlign Teacher Encoder",
        pairalign_encoder_params,
    )
    print_row(
        "PairAlign Teacher Decoder",
        pairalign_decoder_params,
    )
    print_row(
        "PairAlign Encoder + Decoder",
        pairalign_encoder_decoder_params,
    )

    print("=" * 79)

    print("\nPaper-ready values:\n")

    print(
        f"HuBERT Base                  : "
        f"{format_params(hubert_params)}"
    )
    print(
        f"WavLM Large                  : "
        f"{format_params(wavlm_params)}"
    )
    print(
        f"PairAlign Encoder            : "
        f"{format_params(pairalign_encoder_params)}"
    )
    print(
        f"PairAlign Decoder            : "
        f"{format_params(pairalign_decoder_params)}"
    )
    print(
        f"PairAlign Encoder + Decoder  : "
        f"{format_params(pairalign_encoder_decoder_params)}"
    )


if __name__ == "__main__":
    main()
