"""Lightweight command-line parsing for the embedding workflow."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

BACKEND_ALIASES = {"csd": "csd", "sd": "siglip-dinov3"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate style/content embeddings and open the cluster dashboard"
    )
    parser.add_argument(
        "--backend",
        choices=("csd", "siglip-dinov3"),
        default=None,
        help="Embedding backend (default: csd)",
    )
    parser.add_argument(
        "--text-mode",
        choices=("image-only", "caption-guided"),
        default="image-only",
        help="Caption semantics are used only when explicitly selected",
    )
    parser.add_argument(
        "--style-model-config",
        type=Path,
        default=None,
        help="SigLIP2-DINOv3 model configuration",
    )
    parser.add_argument(
        "--style-model-checkpoint",
        type=Path,
        default=None,
        help="Override the configured alignment checkpoint",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Publish a new immutable build instead of reusing a compatible one",
    )

    parser.add_argument(
        "--train-data-dir",
        "--train_data_dir",
        dest="train_data_dir",
        type=Path,
        default=Path("datasets"),
    )
    parser.add_argument(
        "--dataset-path",
        "--dataset_path",
        dest="dataset_path",
        type=Path,
        default=Path("datasets.lance"),
    )
    parser.add_argument(
        "--embeddings-path",
        "--embeddings_path",
        dest="embeddings_path",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--model-name",
        "--model_name",
        default="yuxi-liu-wired/CSD",
    )
    parser.add_argument(
        "--processor-name",
        "--processor_name",
        default="openai/clip-vit-large-patch14",
    )
    parser.add_argument("--batch-size", "--batch_size", type=int, default=12)
    parser.add_argument("--num-workers", "--num_workers", type=int, default=0)
    parser.add_argument("--k-clusters", "--k_clusters", type=int, default=40)
    parser.add_argument(
        "--min-cluster-size", "--min_cluster_size", type=int, default=10
    )
    parser.add_argument(
        "--output-dir", "--output_dir", type=Path, default=Path("output")
    )
    parser.add_argument("--symlink", action="store_true")
    parser.add_argument(
        "--precision", choices=("auto", "fp32", "fp16", "bf16"), default="auto"
    )
    parser.add_argument(
        "--finch-partition-index",
        "--finch_partition_index",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--model_type",
        dest="legacy_model_type",
        choices=("csd", "sd"),
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--sd_config",
        dest="legacy_style_model_config",
        type=Path,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--sd_checkpoint",
        dest="legacy_style_model_checkpoint",
        type=Path,
        default=None,
        help=argparse.SUPPRESS,
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)

    legacy_backend = (
        BACKEND_ALIASES[args.legacy_model_type]
        if args.legacy_model_type is not None
        else None
    )
    if args.backend is not None and legacy_backend not in {None, args.backend}:
        parser.error("--backend conflicts with the legacy --model_type option")
    args.backend = args.backend or legacy_backend or "csd"

    if (
        args.style_model_config is not None
        and args.legacy_style_model_config is not None
        and args.style_model_config != args.legacy_style_model_config
    ):
        parser.error(
            "--style-model-config conflicts with the legacy --sd_config option"
        )
    args.style_model_config = (
        args.style_model_config
        or args.legacy_style_model_config
        or Path("configs/siglip_dinov3.yaml")
    )

    if (
        args.style_model_checkpoint is not None
        and args.legacy_style_model_checkpoint is not None
        and args.style_model_checkpoint != args.legacy_style_model_checkpoint
    ):
        parser.error(
            "--style-model-checkpoint conflicts with the legacy --sd_checkpoint option"
        )
    args.style_model_checkpoint = (
        args.style_model_checkpoint or args.legacy_style_model_checkpoint
    )
    del args.legacy_model_type
    del args.legacy_style_model_config
    del args.legacy_style_model_checkpoint
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    from .workflow import run

    result = run(args)
    return 0 if result is None else int(result)
