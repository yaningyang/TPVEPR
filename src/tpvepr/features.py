from __future__ import annotations

import numpy as np
import polars as pl

from tpvepr.io import normalize_chromosome_expression, require_columns


BASE_REQUIRED_COLUMNS = {
    "tissue",
    "gene_id",
    "chr",
    "variant_pos",
    "ref",
    "alt",
    "strand",
    "gene_start",
    "gene_end",
}


def add_missing_input_columns(frame: pl.DataFrame) -> pl.DataFrame:
    if "variant_id" not in frame.columns:
        frame = frame.with_columns(
            (
                pl.col("chr").cast(pl.Utf8)
                + pl.lit("_")
                + pl.col("variant_pos").cast(pl.Utf8)
                + pl.lit("_")
                + pl.col("ref").cast(pl.Utf8)
                + pl.lit("_")
                + pl.col("alt").cast(pl.Utf8)
            ).alias("variant_id")
        )
    defaults: dict[str, object] = {
        "gene_chr": None,
        "biotype": "unknown",
        "af": 0.0,
        "ma_samples": 0,
        "ma_count": 0,
    }
    for column, value in defaults.items():
        if column not in frame.columns:
            if column == "gene_chr":
                frame = frame.with_columns(pl.col("chr").alias("gene_chr"))
            else:
                frame = frame.with_columns(pl.lit(value).alias(column))
    return frame


def add_promoter_coordinates(
    frame: pl.DataFrame,
    upstream: int = 2000,
    downstream: int = 500,
) -> pl.DataFrame:
    expressions: list[pl.Expr] = []
    if "rel_to_tss" not in frame.columns:
        expressions.append(
            pl.when(pl.col("strand") == "+")
            .then(pl.col("variant_pos") - pl.col("gene_start"))
            .when(pl.col("strand") == "-")
            .then(pl.col("gene_end") - pl.col("variant_pos"))
            .otherwise(None)
            .alias("rel_to_tss")
        )
    if "promoter_start" not in frame.columns:
        expressions.append(
            pl.when(pl.col("strand") == "+")
            .then(pl.col("gene_start") - upstream)
            .when(pl.col("strand") == "-")
            .then(pl.col("gene_end") - downstream)
            .otherwise(None)
            .alias("promoter_start")
        )
    if "promoter_end" not in frame.columns:
        expressions.append(
            pl.when(pl.col("strand") == "+")
            .then(pl.col("gene_start") + downstream)
            .when(pl.col("strand") == "-")
            .then(pl.col("gene_end") + upstream)
            .otherwise(None)
            .alias("promoter_end")
        )
    return frame.with_columns(*expressions) if expressions else frame


def validate_candidate_rows(
    frame: pl.DataFrame,
    upstream: int = 2000,
    downstream: int = 500,
    require_promoter_window: bool = True,
) -> None:
    require_columns(frame, BASE_REQUIRED_COLUMNS, "Candidate table")
    invalid_strand = frame.filter(~pl.col("strand").is_in(["+", "-"])).height
    if invalid_strand:
        raise ValueError(f"Found {invalid_strand} rows with strand other than '+' or '-'.")
    invalid_allele = frame.filter(
        (~pl.col("ref").is_in(["A", "C", "G", "T"]))
        | (~pl.col("alt").is_in(["A", "C", "G", "T"]))
        | (pl.col("ref") == pl.col("alt"))
    ).height
    if invalid_allele:
        raise ValueError(
            f"Found {invalid_allele} non-SNV or invalid allele rows. "
            "TPVEPR accepts single-nucleotide substitutions with ref != alt."
        )
    if require_promoter_window:
        outside = frame.filter(
            (pl.col("rel_to_tss") < -upstream) | (pl.col("rel_to_tss") > downstream)
        ).height
        if outside:
            raise ValueError(
                f"Found {outside} rows outside the promoter window "
                f"[-{upstream}, +{downstream}] relative to the TSS."
            )


def add_engineered_features(frame: pl.DataFrame) -> pl.DataFrame:
    require_columns(
        frame,
        {
            "variant_pos",
            "promoter_start",
            "promoter_end",
            "rel_to_tss",
            "gene_start",
            "gene_end",
            "ref",
            "alt",
        },
        "Candidate table",
    )
    transition = (
        ((pl.col("ref") == "A") & (pl.col("alt") == "G"))
        | ((pl.col("ref") == "G") & (pl.col("alt") == "A"))
        | ((pl.col("ref") == "C") & (pl.col("alt") == "T"))
        | ((pl.col("ref") == "T") & (pl.col("alt") == "C"))
    )
    frame = frame.with_columns(
        pl.col("ref").cast(pl.Utf8).str.to_uppercase(),
        pl.col("alt").cast(pl.Utf8).str.to_uppercase(),
        pl.col("af").cast(pl.Float64, strict=False).fill_null(0.0).fill_nan(0.0),
        pl.col("ma_samples").cast(pl.Float64, strict=False).fill_null(0.0).fill_nan(0.0),
        pl.col("ma_count").cast(pl.Float64, strict=False).fill_null(0.0).fill_nan(0.0),
    )
    return frame.with_columns(
        pl.col("rel_to_tss").abs().cast(pl.Float64).alias("abs_tss_distance"),
        pl.col("rel_to_tss").cast(pl.Float64).alias("signed_tss_distance"),
        (pl.col("gene_end") - pl.col("gene_start") + 1)
        .cast(pl.Float64)
        .alias("gene_length"),
        (pl.col("promoter_end") - pl.col("promoter_start") + 1)
        .cast(pl.Float64)
        .alias("promoter_length"),
        (
            (pl.col("variant_pos") - pl.col("promoter_start"))
            / (pl.col("promoter_end") - pl.col("promoter_start") + 1)
        )
        .cast(pl.Float64)
        .alias("promoter_position_fraction"),
        pl.col("ref").is_in(["G", "C"]).cast(pl.Int8).alias("ref_is_gc"),
        pl.col("alt").is_in(["G", "C"]).cast(pl.Int8).alias("alt_is_gc"),
        transition.cast(pl.Int8).alias("is_transition"),
        (~transition).cast(pl.Int8).alias("is_transversion"),
    )


def add_task_features(frame: pl.DataFrame) -> pl.DataFrame:
    epsilon = 1e-6
    return frame.with_columns(
        pl.col("abs_tss_distance").log1p().alias("log1p_abs_tss_distance"),
        pl.col("ma_samples").clip(0, None).log1p().alias("log1p_ma_samples"),
        pl.col("ma_count").clip(0, None).log1p().alias("log1p_ma_count"),
        (2.0 * pl.col("af").clip(0.0, 1.0) * (1.0 - pl.col("af").clip(0.0, 1.0))).alias(
            "allele_heterozygosity"
        ),
        (-pl.col("abs_tss_distance") / 500.0).exp().alias("tss_proximity_500"),
        (-pl.col("abs_tss_distance") / 2000.0).exp().alias("tss_proximity_2000"),
        (
            pl.col("signed_tss_distance") / (pl.col("abs_tss_distance") + epsilon)
        ).alias("tss_direction"),
    )


def prepare_candidate_features(
    frame: pl.DataFrame,
    upstream: int = 2000,
    downstream: int = 500,
    require_promoter_window: bool = True,
) -> pl.DataFrame:
    require_columns(frame, BASE_REQUIRED_COLUMNS, "Candidate table")
    frame = frame.with_columns(
        normalize_chromosome_expression("chr"),
        pl.col("variant_pos").cast(pl.Int64),
        pl.col("gene_start").cast(pl.Int64),
        pl.col("gene_end").cast(pl.Int64),
        pl.col("strand").cast(pl.Utf8),
        pl.col("ref").cast(pl.Utf8).str.to_uppercase(),
        pl.col("alt").cast(pl.Utf8).str.to_uppercase(),
    )
    frame = add_missing_input_columns(frame)
    frame = add_promoter_coordinates(frame, upstream=upstream, downstream=downstream)
    validate_candidate_rows(
        frame,
        upstream=upstream,
        downstream=downstream,
        require_promoter_window=require_promoter_window,
    )
    frame = add_engineered_features(frame)
    frame = add_task_features(frame)
    if "rank_group_id" not in frame.columns:
        frame = frame.with_columns(
            (pl.col("tissue") + pl.lit("::") + pl.col("gene_id")).alias("rank_group_id")
        )
    return frame


def add_chromosome_splits(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.with_columns(
        pl.when(pl.col("chr").is_in(["chr17", "chr18"]))
        .then(pl.lit("validation"))
        .when(pl.col("chr").is_in(["chr19", "chr20", "chr21", "chr22", "chrX", "chrY"]))
        .then(pl.lit("test"))
        .otherwise(pl.lit("train"))
        .alias("split")
    )


def add_training_targets(
    frame: pl.DataFrame,
    cap_neg_log10_p: float = 300.0,
) -> pl.DataFrame:
    require_columns(frame, {"pval_nominal", "rank_group_id"}, "Training table")
    frame = frame.with_columns(
        pl.col("pval_nominal").cast(pl.Float64, strict=False).alias("pval_nominal")
    )
    invalid = frame.filter(
        pl.col("pval_nominal").is_null()
        | pl.col("pval_nominal").is_nan()
        | (pl.col("pval_nominal") < 0)
        | (pl.col("pval_nominal") > 1)
    ).height
    if invalid:
        raise ValueError(f"Found {invalid} invalid pval_nominal values; expected values in [0, 1].")
    frame = frame.with_columns(
        pl.when(pl.col("pval_nominal") > 0)
        .then((-pl.col("pval_nominal").log10()).clip(0.0, cap_neg_log10_p))
        .otherwise(cap_neg_log10_p)
        .alias("relevance")
    )
    return frame.with_columns(
        pl.col("relevance")
        .rank(method="dense", descending=True)
        .over("rank_group_id")
        .cast(pl.Int64)
        .alias("true_rank")
    )


def relative_features(
    numeric: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
) -> np.ndarray:
    relative = np.zeros((numeric.shape[0], numeric.shape[1] * 2), dtype=np.float32)
    for start, end in zip(starts, ends):
        values = numeric[start:end]
        group_size = end - start
        mean = values.mean(axis=0, keepdims=True)
        std = values.std(axis=0, keepdims=True)
        relative[start:end, : numeric.shape[1]] = (values - mean) / np.maximum(std, 1e-5)
        if group_size == 1:
            continue
        ranks = np.empty_like(values, dtype=np.float32)
        for feature_index in range(values.shape[1]):
            order = np.argsort(values[:, feature_index], kind="stable")
            sorted_values = values[order, feature_index]
            run_starts = np.r_[0, np.flatnonzero(sorted_values[1:] != sorted_values[:-1]) + 1]
            run_ends = np.r_[run_starts[1:], group_size]
            for run_start, run_end in zip(run_starts, run_ends):
                average_rank = (run_start + run_end - 1) / 2.0
                ranks[order[run_start:run_end], feature_index] = average_rank
        relative[start:end, numeric.shape[1] :] = ranks / float(group_size - 1)
    return relative
