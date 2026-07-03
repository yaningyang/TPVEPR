from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl
import torch

from tpvepr.config import TPVEPRConfig
from tpvepr.dataset import GroupedRankingDataset, make_loader
from tpvepr.features import prepare_candidate_features
from tpvepr.io import read_table, write_table
from tpvepr.training import build_model, resolve_device, score_dataset


def load_checkpoint(
    checkpoint_path: Path,
    device_name: str | None = None,
) -> tuple[TPVEPRConfig, dict, torch.nn.Module, torch.device]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    required = {
        "model_state",
        "feature_state",
        "config",
        "relation_indices",
        "prior_indices",
    }
    missing = sorted(required - set(checkpoint))
    if missing:
        raise ValueError(f"Checkpoint is missing fields: {missing}")

    config = TPVEPRConfig(**checkpoint["config"])
    device = resolve_device(device_name or config.device)
    model = build_model(
        config,
        checkpoint["feature_state"],
        checkpoint["relation_indices"],
        checkpoint["prior_indices"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return config, checkpoint, model, device


def predict(
    input_path: Path,
    checkpoint_path: Path,
    output_path: Path,
    *,
    device_name: str | None = None,
    promoter_upstream: int | None = None,
    promoter_downstream: int | None = None,
) -> pl.DataFrame:
    config, checkpoint, model, device = load_checkpoint(checkpoint_path, device_name)
    preprocessing = checkpoint.get("preprocessing", {})
    upstream = (
        promoter_upstream
        if promoter_upstream is not None
        else int(preprocessing.get("promoter_upstream", 2000))
    )
    downstream = (
        promoter_downstream
        if promoter_downstream is not None
        else int(preprocessing.get("promoter_downstream", 500))
    )

    raw = read_table(input_path).with_row_index("_input_row")
    original_columns = [column for column in raw.columns if column != "_input_row"]
    frame = prepare_candidate_features(
        raw,
        upstream=upstream,
        downstream=downstream,
        require_promoter_window=True,
    )
    missing_features = sorted(
        (
            set(config.numeric_features)
            | set(config.categorical_features)
            | {"rank_group_id", "variant_id", "variant_pos"}
        )
        - set(frame.columns)
    )
    if missing_features:
        raise ValueError(f"Prediction input cannot produce configured features: {missing_features}")

    group_sizes = frame.group_by("rank_group_id").agg(pl.len().alias("_group_size"))
    singleton_count = group_sizes.filter(pl.col("_group_size") == 1).height
    if singleton_count:
        print(
            f"Warning: {singleton_count} groups contain one candidate; "
            "their within-group rank is necessarily 1."
        )

    dataset = GroupedRankingDataset(
        frame,
        checkpoint["feature_state"],
        config,
        split=None,
    )
    loader = make_loader(dataset, config, shuffle=False)
    scored = score_dataset(model, dataset, loader, device).sort("_input_row")

    output_columns = list(original_columns)
    for column in ("variant_id", "rank_group_id"):
        if column not in output_columns:
            output_columns.append(column)
    output_columns.extend(["tpvepr_score", "tpvepr_rank"])
    result = scored.select(output_columns)
    write_table(result, output_path)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rank new promoter candidate sets with TPVEPR.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default=None, help="auto, cpu, cuda, or cuda:0")
    parser.add_argument("--promoter-upstream", type=int, default=None)
    parser.add_argument("--promoter-downstream", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    result = predict(
        input_path=args.input,
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        device_name=args.device,
        promoter_upstream=args.promoter_upstream,
        promoter_downstream=args.promoter_downstream,
    )
    print(
        f"Wrote {args.output}: {result.height:,} variants in "
        f"{result['rank_group_id'].n_unique():,} tissue-gene groups"
    )


if __name__ == "__main__":
    main()
