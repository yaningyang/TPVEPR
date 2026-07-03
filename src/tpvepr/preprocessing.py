from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import polars as pl

from tpvepr.io import ensure_dir, require_columns, write_json


EGENE_SUFFIX = ".v11.eGenes.txt.gz"
SIGNIFICANT_PAIR_SUFFIXES = (
    ".v11.eQTLs.signif_pairs.parquet",
    ".v11.signif_variant_gene_pairs.txt.gz",
    ".v11.eQTLs.signif_pairs.txt.gz",
)


@dataclass(frozen=True)
class PreprocessConfig:
    input_root: str
    output_root: str
    promoter_upstream: int = 2000
    promoter_downstream: int = 500
    qvalue_threshold: float = 0.05
    biotypes: tuple[str, ...] = ("protein_coding", "lncRNA")


@dataclass(frozen=True)
class TissueFiles:
    tissue: str
    egene_path: Path
    significant_pairs_path: Path | None


def resolve_data_directory(input_root: Path) -> Path:
    root = input_root.resolve()
    if not root.exists():
        raise FileNotFoundError(f"GTEx input directory does not exist: {root}")
    if list(root.glob(f"*{EGENE_SUFFIX}")):
        return root
    candidates = [
        path
        for path in root.iterdir()
        if path.is_dir() and list(path.glob(f"*{EGENE_SUFFIX}"))
    ]
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(
            f"No files matching '*{EGENE_SUFFIX}' were found in {root} or one level below it."
        )
    raise ValueError(f"Multiple possible GTEx directories found: {candidates}")


def discover_tissues(input_root: Path) -> list[TissueFiles]:
    data_directory = resolve_data_directory(input_root)
    tissues: list[TissueFiles] = []
    for egene_path in sorted(data_directory.glob(f"*{EGENE_SUFFIX}")):
        tissue = egene_path.name.removesuffix(EGENE_SUFFIX)
        significant_pairs_path = next(
            (
                data_directory / f"{tissue}{suffix}"
                for suffix in SIGNIFICANT_PAIR_SUFFIXES
                if (data_directory / f"{tissue}{suffix}").exists()
            ),
            None,
        )
        tissues.append(TissueFiles(tissue, egene_path, significant_pairs_path))
    if not tissues:
        raise ValueError("No GTEx tissues were discovered.")
    return tissues


def read_egenes(path: Path, tissue: str) -> pl.DataFrame:
    frame = pl.read_csv(
        path,
        separator="\t",
        infer_schema_length=5000,
        null_values=["NA", "NaN", ""],
    )
    require_columns(
        frame,
        {
            "gene_id",
            "gene_name",
            "biotype",
            "gene_chr",
            "gene_start",
            "gene_end",
            "strand",
            "qval",
        },
        f"eGenes file {path.name}",
    )
    return frame.with_columns(
        pl.lit(tissue).alias("tissue"),
        pl.col("gene_start").cast(pl.Int64),
        pl.col("gene_end").cast(pl.Int64),
        pl.col("qval").cast(pl.Float64),
        pl.when(pl.col("strand") == "+")
        .then(pl.col("gene_start"))
        .when(pl.col("strand") == "-")
        .then(pl.col("gene_end"))
        .otherwise(None)
        .alias("tss_pos"),
    )


def scan_significant_pairs(path: Path) -> pl.LazyFrame:
    if path.suffix.lower() == ".parquet":
        frame = pl.scan_parquet(path)
    else:
        frame = pl.scan_csv(
            path,
            separator="\t",
            infer_schema_length=5000,
            null_values=["NA", "NaN", ""],
        )
    names = set(frame.collect_schema().names())
    rename: dict[str, str] = {}
    if "gene_id" not in names and "phenotype_id" in names:
        rename["phenotype_id"] = "gene_id"
    if "tss_distance" not in names and "start_distance" in names:
        rename["start_distance"] = "tss_distance"
    return frame.rename(rename) if rename else frame


def parse_variant_columns(frame: pl.LazyFrame) -> pl.LazyFrame:
    names = set(frame.collect_schema().names())
    if {"chr", "variant_pos", "ref", "alt"}.issubset(names):
        return frame.with_columns(
            pl.col("chr").cast(pl.Utf8),
            pl.col("variant_pos").cast(pl.Int64),
            pl.col("ref").cast(pl.Utf8).str.to_uppercase(),
            pl.col("alt").cast(pl.Utf8).str.to_uppercase(),
        )
    if "variant_id" not in names:
        raise ValueError("Significant-pairs file must contain variant_id or chr/variant_pos/ref/alt.")
    parts = pl.col("variant_id").str.split_exact("_", 4)
    return frame.with_columns(
        parts.struct.field("field_0").alias("chr"),
        parts.struct.field("field_1").cast(pl.Int64).alias("variant_pos"),
        parts.struct.field("field_2").str.to_uppercase().alias("ref"),
        parts.struct.field("field_3").str.to_uppercase().alias("alt"),
    )


def _gene_annotations(
    tissues: Iterable[TissueFiles],
    config: PreprocessConfig,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    all_genes = pl.concat(
        [read_egenes(item.egene_path, item.tissue) for item in tissues],
        how="vertical_relaxed",
    )
    if config.biotypes:
        all_genes = all_genes.filter(pl.col("biotype").is_in(list(config.biotypes)))
    significant = all_genes.filter(pl.col("qval") <= config.qvalue_threshold)
    return all_genes, significant


def _promoter_pairs_for_tissue(
    item: TissueFiles,
    annotations: pl.LazyFrame,
    upstream: int,
    downstream: int,
) -> pl.DataFrame | None:
    if item.significant_pairs_path is None:
        return None
    pairs = parse_variant_columns(scan_significant_pairs(item.significant_pairs_path))
    names = set(pairs.collect_schema().names())
    required = {"gene_id", "variant_id", "pval_nominal", "chr", "variant_pos", "ref", "alt"}
    missing = sorted(required - names)
    if missing:
        raise ValueError(
            f"{item.significant_pairs_path.name} is missing required columns: {', '.join(missing)}"
        )
    joined = pairs.with_columns(pl.lit(item.tissue).alias("tissue")).join(
        annotations,
        on=["tissue", "gene_id"],
        how="inner",
    )
    return (
        joined.with_columns(
            pl.when(pl.col("strand") == "+")
            .then(pl.col("variant_pos") - pl.col("gene_start"))
            .when(pl.col("strand") == "-")
            .then(pl.col("gene_end") - pl.col("variant_pos"))
            .otherwise(None)
            .alias("rel_to_tss"),
            pl.when(pl.col("strand") == "+")
            .then(pl.col("gene_start") - upstream)
            .otherwise(pl.col("gene_end") - downstream)
            .alias("promoter_start"),
            pl.when(pl.col("strand") == "+")
            .then(pl.col("gene_start") + downstream)
            .otherwise(pl.col("gene_end") + upstream)
            .alias("promoter_end"),
        )
        .filter(
            (pl.col("rel_to_tss") >= -upstream)
            & (pl.col("rel_to_tss") <= downstream)
            & pl.col("ref").is_in(["A", "C", "G", "T"])
            & pl.col("alt").is_in(["A", "C", "G", "T"])
            & (pl.col("ref") != pl.col("alt"))
        )
        .with_columns(
            pl.lit(True).alias("is_promoter"),
            pl.col("qval").alias("egene_qval"),
        )
        .drop("qval")
        .collect()
    )


def preprocess_gtex_v11(config: PreprocessConfig) -> Path:
    if config.promoter_upstream < 0 or config.promoter_downstream < 0:
        raise ValueError("Promoter upstream/downstream distances must be non-negative.")
    if not 0 <= config.qvalue_threshold <= 1:
        raise ValueError("qvalue_threshold must be in [0, 1].")

    output_root = ensure_dir(Path(config.output_root))
    tissues = discover_tissues(Path(config.input_root))
    all_genes, significant_genes = _gene_annotations(tissues, config)

    annotations = significant_genes.select(
        [
            "tissue",
            "gene_id",
            "gene_name",
            "biotype",
            "gene_chr",
            "gene_start",
            "gene_end",
            "strand",
            "tss_pos",
            "qval",
        ]
    ).lazy()
    promoter_frames: list[pl.DataFrame] = []
    summary_rows: list[dict[str, object]] = []
    for item in tissues:
        promoter = _promoter_pairs_for_tissue(
            item,
            annotations,
            upstream=config.promoter_upstream,
            downstream=config.promoter_downstream,
        )
        promoter_count = 0 if promoter is None else promoter.height
        if promoter is not None:
            promoter_frames.append(promoter)
        summary_rows.append(
            {
                "tissue": item.tissue,
                "egene_file": str(item.egene_path),
                "significant_pairs_file": (
                    str(item.significant_pairs_path) if item.significant_pairs_path else ""
                ),
                "has_significant_pairs": item.significant_pairs_path is not None,
                "tested_gene_rows": all_genes.filter(pl.col("tissue") == item.tissue).height,
                "significant_egene_rows": significant_genes.filter(
                    pl.col("tissue") == item.tissue
                ).height,
                "promoter_pair_rows": promoter_count,
            }
        )
    if not promoter_frames:
        raise ValueError("No promoter SNV pairs were produced from the supplied GTEx files.")

    promoter_pairs = pl.concat(promoter_frames, how="diagonal_relaxed")
    output_path = output_root / "promoter_significant_pairs.parquet"
    promoter_pairs.write_parquet(output_path)
    pl.DataFrame(summary_rows).write_csv(output_root / "preprocess_summary.tsv", separator="\t")
    write_json(output_root / "preprocess_config.json", asdict(config))
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Filter GTEx v11 significant cis-eQTL pairs to strand-aware promoter SNVs."
    )
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--promoter-upstream", type=int, default=2000)
    parser.add_argument("--promoter-downstream", type=int, default=500)
    parser.add_argument("--qvalue-threshold", type=float, default=0.05)
    parser.add_argument(
        "--biotypes",
        nargs="+",
        default=["protein_coding", "lncRNA"],
        help="Gene biotypes to retain.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    output_path = preprocess_gtex_v11(
        PreprocessConfig(
            input_root=str(args.input_root),
            output_root=str(args.output_root),
            promoter_upstream=args.promoter_upstream,
            promoter_downstream=args.promoter_downstream,
            qvalue_threshold=args.qvalue_threshold,
            biotypes=tuple(args.biotypes),
        )
    )
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
