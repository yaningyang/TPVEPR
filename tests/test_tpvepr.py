from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import torch

from tpvepr.config import TPVEPRConfig
from tpvepr.features import relative_features
from tpvepr.model import TPVEPR
from tpvepr.predict import predict
from tpvepr.prepare import prepare_training_dataset
from tpvepr.training import train


ROOT = Path(__file__).resolve().parents[1]


def test_relative_features_are_permutation_equivariant_with_ties() -> None:
    numeric = np.array(
        [[0.0, 1.0], [0.0, 1.0], [1.0, 1.0], [1.0, 2.0]],
        dtype=np.float32,
    )
    starts = np.array([0])
    ends = np.array([4])
    relative = relative_features(numeric, starts, ends)
    permutation = np.array([2, 0, 3, 1])
    permuted = relative_features(numeric[permutation], starts, ends)
    np.testing.assert_allclose(relative, permuted[np.argsort(permutation)])
    np.testing.assert_allclose(relative[0, 2:], relative[1, 2:])


def test_model_is_permutation_equivariant() -> None:
    torch.manual_seed(7)
    numeric = torch.tensor([[[0.0, 1.0], [0.0, 1.0], [1.0, 1.0], [1.0, 2.0]]])
    relative = torch.from_numpy(
        relative_features(numeric[0].numpy(), np.array([0]), np.array([4]))
    ).unsqueeze(0)
    categorical = torch.tensor([[[1], [1], [2], [2]]])
    mask = torch.ones((1, 4), dtype=torch.bool)
    model = TPVEPR(
        num_numeric=2,
        categorical_cardinalities=[3],
        relation_numeric_indices=[0, 1],
        prior_numeric_indices=[0, 1],
        d_model=16,
        num_heads=2,
        num_layers=1,
        dropout=0.0,
    )
    model.eval()
    permutation = torch.tensor([2, 0, 3, 1])
    inverse = torch.argsort(permutation)
    with torch.no_grad():
        scores = model(numeric, relative, categorical, mask)
        permuted_scores = model(
            numeric[:, permutation],
            relative[:, permutation],
            categorical[:, permutation],
            mask[:, permutation],
        )[:, inverse]
    torch.testing.assert_close(scores, permuted_scores, atol=1e-6, rtol=1e-6)


def test_small_end_to_end_workflow(tmp_path: Path) -> None:
    input_path = ROOT / "examples" / "example_training_input.tsv"
    dataset_path = tmp_path / "ranking.parquet"
    prepared = prepare_training_dataset(input_path, dataset_path)
    assert prepared.height == 12
    assert set(prepared["split"].unique()) == {"train", "validation", "test"}

    config = TPVEPRConfig.from_json(ROOT / "configs" / "smoke_test.json")
    output_root = tmp_path / "run"
    metrics = train(dataset_path, output_root, config)
    assert metrics.height == 2
    checkpoint = output_root / "model" / "tpvepr.pt"
    assert checkpoint.exists()

    prediction_path = tmp_path / "predictions.tsv"
    predictions = predict(
        input_path=input_path,
        checkpoint_path=checkpoint,
        output_path=prediction_path,
        device_name="cpu",
    )
    assert prediction_path.exists()
    assert predictions.height == 12
    assert predictions["tpvepr_score"].is_null().sum() == 0
    maximum_ranks = (
        predictions.group_by("rank_group_id")
        .agg(pl.col("tpvepr_rank").max().alias("maximum_rank"))
        ["maximum_rank"]
        .to_list()
    )
    assert maximum_ranks == [4, 4, 4]
