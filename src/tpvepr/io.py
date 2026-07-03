from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import polars as pl


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    if is_dataclass(payload):
        payload = asdict(payload)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def read_table(path: Path) -> pl.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pl.read_parquet(path)
    if suffix in {".tsv", ".txt", ".gz"}:
        return pl.read_csv(
            path,
            separator="\t",
            infer_schema_length=10_000,
            null_values=["NA", "NaN", "nan", ""],
        )
    if suffix == ".csv":
        return pl.read_csv(
            path,
            infer_schema_length=10_000,
            null_values=["NA", "NaN", "nan", ""],
        )
    raise ValueError(f"Unsupported table type: {path}. Use .parquet, .tsv, .txt, .txt.gz, or .csv.")


def write_table(frame: pl.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        frame.write_parquet(path)
    elif suffix in {".tsv", ".txt"}:
        frame.write_csv(path, separator="\t")
    elif suffix == ".csv":
        frame.write_csv(path)
    else:
        raise ValueError(f"Unsupported output type: {path}. Use .parquet, .tsv, .txt, or .csv.")


def require_columns(frame: pl.DataFrame, required: set[str], context: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{context} is missing required columns: {', '.join(missing)}")


def normalize_chromosome_expression(column: str = "chr") -> pl.Expr:
    value = pl.col(column).cast(pl.Utf8)
    return (
        pl.when(value.str.starts_with("chr"))
        .then(value)
        .otherwise(pl.lit("chr") + value)
        .alias(column)
    )
