from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from tpvepr.features import (
    add_chromosome_splits,
    add_training_targets,
    prepare_candidate_features,
)
from tpvepr.io import read_table, write_json, write_table


def prepare_training_dataset(
    input_path: Path,
    output_path: Path,
    *,
    promoter_upstream: int = 2000,
    promoter_downstream: int = 500,
    min_group_size: int = 2,
    cap_neg_log10_p: float = 300.0,
    use_existing_split: bool = False,
) -> pl.DataFrame:
    if min_group_size < 2:
        raise ValueError("min_group_size must be at least 2 for a ranking dataset.")
    frame = read_table(input_path).with_row_index("_input_row")
    frame = prepare_candidate_features(
        frame,
        upstream=promoter_upstream,
        downstream=promoter_downstream,
        require_promoter_window=True,
    )
    frame = add_training_targets(frame, cap_neg_log10_p=cap_neg_log10_p)

    if use_existing_split:
        if "split" not in frame.columns:
            raise ValueError("--use-existing-split requires an input column named 'split'.")
        invalid = sorted(
            set(frame["split"].drop_nulls().unique().to_list())
            - {"train", "validation", "test"}
        )
        if invalid:
            raise ValueError(f"Invalid split labels: {invalid}")
    else:
        frame = add_chromosome_splits(frame)

    duplicate_count = (
        frame.group_by(["rank_group_id", "variant_id"])
        .agg(pl.len().alias("n"))
        .filter(pl.col("n") > 1)
        .height
    )
    if duplicate_count:
        raise ValueError(
            f"Found {duplicate_count} duplicated rank_group_id/variant_id combinations."
        )

    group_summary = frame.group_by("rank_group_id").agg(
        pl.len().alias("rank_group_size"),
        pl.col("split").n_unique().alias("_split_count"),
    )
    mixed_split_groups = group_summary.filter(pl.col("_split_count") != 1).height
    if mixed_split_groups:
        raise ValueError(
            f"Found {mixed_split_groups} tissue-gene groups spanning multiple data splits."
        )
    group_summary = group_summary.drop("_split_count")
    frame = (
        frame.join(group_summary, on="rank_group_id", how="left")
        .filter(pl.col("rank_group_size") >= min_group_size)
        .sort(["rank_group_id", "variant_pos", "variant_id"])
        .drop("_input_row")
    )
    if frame.height == 0:
        raise ValueError("No rows remain after candidate-group filtering.")
    missing_splits = sorted(
        {"train", "validation", "test"} - set(frame["split"].unique().to_list())
    )
    if missing_splits:
        raise ValueError(
            "Prepared data must contain train, validation, and test rows; "
            f"missing: {missing_splits}"
        )

    write_table(frame, output_path)
    split_summary = (
        frame.group_by("split")
        .agg(
            pl.len().alias("rows"),
            pl.col("rank_group_id").n_unique().alias("groups"),
        )
        .sort("split")
    )
    write_json(
        output_path.with_suffix(".summary.json"),
        {
            "input": str(input_path),
            "output": str(output_path),
            "rows": frame.height,
            "groups": frame["rank_group_id"].n_unique(),
            "min_group_size": min_group_size,
            "promoter_window": [-promoter_upstream, promoter_downstream],
            "cap_neg_log10_p": cap_neg_log10_p,
            "splits": split_summary.to_dicts(),
        },
    )
    return frame


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a chromosome-held-out TPVEPR ranking dataset."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--promoter-upstream", type=int, default=2000)
    parser.add_argument("--promoter-downstream", type=int, default=500)
    parser.add_argument("--min-group-size", type=int, default=2)
    parser.add_argument("--cap-neg-log10-p", type=float, default=300.0)
    parser.add_argument("--use-existing-split", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    frame = prepare_training_dataset(
        input_path=args.input,
        output_path=args.output,
        promoter_upstream=args.promoter_upstream,
        promoter_downstream=args.promoter_downstream,
        min_group_size=args.min_group_size,
        cap_neg_log10_p=args.cap_neg_log10_p,
        use_existing_split=args.use_existing_split,
    )
    print(
        f"Wrote {args.output}: {frame.height:,} rows, "
        f"{frame['rank_group_id'].n_unique():,} groups"
    )


if __name__ == "__main__":
    main()
