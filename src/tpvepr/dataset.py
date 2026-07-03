from __future__ import annotations

import math
from typing import Any, Iterator

import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from tpvepr.config import TPVEPRConfig
from tpvepr.features import relative_features


def fit_feature_state(
    frame: pl.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
) -> dict[str, Any]:
    if "split" not in frame.columns:
        raise ValueError("Training data must contain a split column.")
    train = frame.filter(pl.col("split") == "train")
    if train.height == 0:
        raise ValueError("Training split is empty.")
    missing = sorted(
        (set(numeric_features) | set(categorical_features)) - set(frame.columns)
    )
    if missing:
        raise ValueError(f"Training data is missing configured features: {missing}")

    state: dict[str, Any] = {"numeric": {}, "categorical": {}}
    for feature in numeric_features:
        values = train[feature].cast(pl.Float64, strict=False).to_numpy()
        finite = values[np.isfinite(values)]
        mean = float(finite.mean()) if finite.size else 0.0
        std = float(finite.std(ddof=1)) if finite.size > 1 else 1.0
        if not np.isfinite(std) or std <= 0:
            std = 1.0
        state["numeric"][feature] = {"mean": mean, "std": std}

    for feature in categorical_features:
        categories = (
            train[feature]
            .cast(pl.Utf8)
            .fill_null("NA")
            .unique()
            .sort()
            .to_list()
        )
        state["categorical"][feature] = {
            "<UNK>": 0,
            **{str(value): index + 1 for index, value in enumerate(categories)},
        }
    return state


def numeric_matrix(
    frame: pl.DataFrame,
    state: dict[str, Any],
    features: list[str],
) -> np.ndarray:
    columns: list[np.ndarray] = []
    for feature in features:
        stats = state["numeric"][feature]
        values = frame[feature].cast(pl.Float64, strict=False).to_numpy()
        values = np.nan_to_num(
            values,
            nan=stats["mean"],
            posinf=stats["mean"],
            neginf=stats["mean"],
        )
        columns.append((values - stats["mean"]) / stats["std"])
    return np.column_stack(columns).astype(np.float32, copy=False)


def categorical_matrix(
    frame: pl.DataFrame,
    state: dict[str, Any],
    features: list[str],
) -> np.ndarray:
    columns: list[np.ndarray] = []
    for feature in features:
        vocabulary = state["categorical"][feature]
        values = frame[feature].cast(pl.Utf8).fill_null("NA").to_list()
        columns.append(
            np.fromiter(
                (vocabulary.get(str(value), 0) for value in values),
                dtype=np.int64,
                count=len(values),
            )
        )
    if not columns:
        return np.zeros((frame.height, 0), dtype=np.int64)
    return np.column_stack(columns)


class GroupedRankingDataset(Dataset):
    """Flat feature storage with variable-length tissue-gene group views."""

    def __init__(
        self,
        frame: pl.DataFrame,
        state: dict[str, Any],
        config: TPVEPRConfig,
        split: str | None = None,
    ) -> None:
        if split is not None:
            frame = frame.filter(pl.col("split") == split)
        if frame.height == 0:
            label = f"split {split!r}" if split is not None else "input"
            raise ValueError(f"No rows found for {label}.")
        frame = frame.with_row_index("_dataset_source_row").sort(
            ["rank_group_id", "variant_pos", "variant_id"]
        )
        group_ids = np.asarray(frame["rank_group_id"].to_list(), dtype=object)
        starts = np.r_[0, np.flatnonzero(group_ids[1:] != group_ids[:-1]) + 1]
        ends = np.r_[starts[1:], frame.height]
        lengths = ends - starts
        observed_maximum = int(lengths.max(initial=0))
        if observed_maximum > config.max_group_size:
            raise ValueError(
                f"Observed group size {observed_maximum} exceeds "
                f"max_group_size={config.max_group_size}. Increase the configured limit."
            )

        self.frame = frame
        self.numeric = numeric_matrix(frame, state, config.numeric_features)
        self.categorical = categorical_matrix(frame, state, config.categorical_features)
        self.relevance = (
            frame["relevance"].cast(pl.Float32).to_numpy()
            if "relevance" in frame.columns
            else np.zeros(frame.height, dtype=np.float32)
        )
        self.starts = starts.astype(np.int64)
        self.ends = ends.astype(np.int64)
        self.lengths = lengths.astype(np.int64)
        self.relative = relative_features(self.numeric, self.starts, self.ends)

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, index: int) -> dict[str, np.ndarray]:
        start = self.starts[index]
        end = self.ends[index]
        return {
            "numeric": self.numeric[start:end],
            "relative": self.relative[start:end],
            "categorical": self.categorical[start:end],
            "relevance": self.relevance[start:end],
            "row_indices": np.arange(start, end, dtype=np.int64),
        }


class PairBudgetBatchSampler(Sampler[list[int]]):
    """Batch similarly sized groups under a padded pairwise-attention budget."""

    def __init__(
        self,
        lengths: np.ndarray,
        max_groups: int,
        max_pair_tokens: int,
        shuffle: bool,
        seed: int,
    ) -> None:
        self.lengths = lengths
        self.max_groups = max_groups
        self.max_pair_tokens = max_pair_tokens
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        order = np.argsort(self.lengths, kind="stable")
        buckets = [order[index : index + 2048].copy() for index in range(0, len(order), 2048)]
        if self.shuffle:
            for bucket in buckets:
                rng.shuffle(bucket)
            rng.shuffle(buckets)

        batch: list[int] = []
        maximum_length = 0
        for bucket in buckets:
            for index in bucket:
                candidate_maximum = max(maximum_length, int(self.lengths[index]))
                candidate_size = len(batch) + 1
                pair_tokens = candidate_size * candidate_maximum * candidate_maximum
                if batch and (
                    candidate_size > self.max_groups
                    or pair_tokens > self.max_pair_tokens
                ):
                    yield batch
                    batch = []
                    maximum_length = 0
                batch.append(int(index))
                maximum_length = max(maximum_length, int(self.lengths[index]))
        if batch:
            yield batch

    def __len__(self) -> int:
        return math.ceil(len(self.lengths) / self.max_groups)


def collate_groups(batch: list[dict[str, np.ndarray]]) -> dict[str, torch.Tensor]:
    batch_size = len(batch)
    maximum_length = max(item["numeric"].shape[0] for item in batch)
    numeric_dim = batch[0]["numeric"].shape[1]
    categorical_dim = batch[0]["categorical"].shape[1]

    numeric = np.zeros((batch_size, maximum_length, numeric_dim), dtype=np.float32)
    relative = np.zeros((batch_size, maximum_length, numeric_dim * 2), dtype=np.float32)
    categorical = np.zeros((batch_size, maximum_length, categorical_dim), dtype=np.int64)
    relevance = np.zeros((batch_size, maximum_length), dtype=np.float32)
    mask = np.zeros((batch_size, maximum_length), dtype=bool)
    row_indices = np.full((batch_size, maximum_length), -1, dtype=np.int64)

    for batch_index, item in enumerate(batch):
        length = item["numeric"].shape[0]
        numeric[batch_index, :length] = item["numeric"]
        relative[batch_index, :length] = item["relative"]
        categorical[batch_index, :length] = item["categorical"]
        relevance[batch_index, :length] = item["relevance"]
        row_indices[batch_index, :length] = item["row_indices"]
        mask[batch_index, :length] = True

    return {
        "numeric": torch.from_numpy(numeric),
        "relative": torch.from_numpy(relative),
        "categorical": torch.from_numpy(categorical),
        "relevance": torch.from_numpy(relevance),
        "mask": torch.from_numpy(mask),
        "row_indices": torch.from_numpy(row_indices),
    }


def make_loader(
    dataset: GroupedRankingDataset,
    config: TPVEPRConfig,
    shuffle: bool,
) -> DataLoader:
    sampler = PairBudgetBatchSampler(
        dataset.lengths,
        max_groups=config.max_batch_groups,
        max_pair_tokens=config.max_pair_tokens,
        shuffle=shuffle,
        seed=config.seed,
    )
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=collate_groups,
        num_workers=0,
    )
