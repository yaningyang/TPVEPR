from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import polars as pl
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from tpvepr import __version__
from tpvepr.config import TPVEPRConfig
from tpvepr.dataset import (
    GroupedRankingDataset,
    fit_feature_state,
    make_loader,
)
from tpvepr.io import ensure_dir, read_table, write_json
from tpvepr.model import TPVEPR


def resolve_device(requested: str) -> torch.device:
    requested = requested.lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")
    return torch.device(requested)


def normalized_relevance(relevance: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    minimum = relevance.masked_fill(~mask, float("inf")).amin(dim=1, keepdim=True)
    maximum = relevance.masked_fill(~mask, -float("inf")).amax(dim=1, keepdim=True)
    normalized = (relevance - minimum) / (maximum - minimum).clamp_min(1e-6)
    return normalized.masked_fill(~mask, 0.0)


def lambda_pairwise_loss(
    scores: torch.Tensor,
    relevance: torch.Tensor,
    normalized: torch.Tensor,
    mask: torch.Tensor,
    k: int,
) -> torch.Tensor:
    batch_size, group_size = scores.shape
    predicted_order = torch.argsort(scores.masked_fill(~mask, -1e4).detach(), dim=1, descending=True)
    predicted_rank = torch.empty_like(predicted_order)
    rank_values = torch.arange(group_size, device=scores.device).expand(batch_size, -1)
    predicted_rank.scatter_(1, predicted_order, rank_values)
    discount = torch.where(
        predicted_rank < k,
        1.0 / torch.log2(predicted_rank.float() + 2.0),
        torch.zeros_like(predicted_rank, dtype=torch.float32),
    )

    gains = torch.pow(2.0, normalized) - 1.0
    ideal_order = torch.argsort(relevance.masked_fill(~mask, -1e4), dim=1, descending=True)
    ideal_gains = torch.gather(gains, 1, ideal_order)
    position = torch.arange(group_size, device=scores.device)
    ideal_discount = torch.where(
        position < k,
        1.0 / torch.log2(position.float() + 2.0),
        torch.zeros(group_size, device=scores.device),
    )
    ideal_dcg = (ideal_gains * ideal_discount).sum(dim=1).clamp_min(1e-6)

    relevance_difference = relevance.unsqueeze(2) - relevance.unsqueeze(1)
    score_difference = scores.unsqueeze(2) - scores.unsqueeze(1)
    valid_pairs = mask.unsqueeze(2) & mask.unsqueeze(1) & (relevance_difference > 0)
    gain_difference = (gains.unsqueeze(2) - gains.unsqueeze(1)).abs()
    discount_difference = (discount.unsqueeze(2) - discount.unsqueeze(1)).abs()
    weights = gain_difference * discount_difference / ideal_dcg[:, None, None]
    weighted_loss = F.softplus(-score_difference) * weights * valid_pairs
    return weighted_loss.sum() / weights.mul(valid_pairs).sum().clamp_min(1e-6)


def ranking_loss(
    scores: torch.Tensor,
    relevance: torch.Tensor,
    mask: torch.Tensor,
    config: TPVEPRConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    normalized = normalized_relevance(relevance, mask)
    target_distribution = torch.softmax(
        (normalized / config.list_temperature).masked_fill(~mask, -1e4),
        dim=1,
    )
    log_prediction = F.log_softmax(scores.masked_fill(~mask, -1e4), dim=1)
    listwise = -(target_distribution * log_prediction).sum(dim=1).mean()

    maximum = relevance.masked_fill(~mask, -1e4).amax(dim=1, keepdim=True)
    top_target = (relevance == maximum) & mask
    top_distribution = top_target.float() / top_target.sum(dim=1, keepdim=True).clamp_min(1)
    top1 = -(top_distribution * log_prediction).sum(dim=1).mean()
    pairwise = lambda_pairwise_loss(
        scores,
        relevance,
        normalized,
        mask,
        k=config.ndcg_loss_k,
    )

    centered_score = scores - (
        scores.masked_fill(~mask, 0).sum(dim=1, keepdim=True)
        / mask.sum(dim=1, keepdim=True).clamp_min(1)
    )
    centered_target = normalized - (
        normalized.sum(dim=1, keepdim=True)
        / mask.sum(dim=1, keepdim=True).clamp_min(1)
    )
    calibration = (((centered_score - centered_target) ** 2) * mask).sum() / mask.sum().clamp_min(1)

    total = (
        config.listwise_weight * listwise
        + config.pairwise_weight * pairwise
        + config.top1_weight * top1
        + config.calibration_weight * calibration
    )
    return total, {
        "loss": float(total.detach().cpu()),
        "listwise": float(listwise.detach().cpu()),
        "pairwise": float(pairwise.detach().cpu()),
        "top1": float(top1.detach().cpu()),
        "calibration": float(calibration.detach().cpu()),
    }


def group_ranking_metrics(
    relevance: np.ndarray,
    scores: np.ndarray,
) -> dict[str, float]:
    minimum = float(np.min(relevance))
    maximum = float(np.max(relevance))
    graded = (
        (relevance - minimum) / (maximum - minimum)
        if maximum > minimum
        else np.zeros_like(relevance)
    )
    gains = np.exp2(graded) - 1.0
    predicted_order = np.argsort(-scores, kind="stable")
    ideal_order = np.argsort(-relevance, kind="stable")
    metrics: dict[str, float] = {}
    for k in (1, 3, 5):
        cutoff = min(k, len(relevance))
        discounts = 1.0 / np.log2(np.arange(2, cutoff + 2))
        dcg = float(np.sum(gains[predicted_order[:cutoff]] * discounts))
        idcg = float(np.sum(gains[ideal_order[:cutoff]] * discounts))
        metrics[f"ndcg@{k}"] = dcg / idcg if idcg > 0 else 1.0
    metrics["hit@1"] = float(relevance[predicted_order[0]] == maximum)

    relevance_difference = relevance[:, None] - relevance[None, :]
    score_difference = scores[:, None] - scores[None, :]
    comparable = np.triu(relevance_difference != 0, k=1)
    metrics["pairwise_accuracy"] = (
        float(np.mean((relevance_difference[comparable] * score_difference[comparable]) > 0))
        if comparable.any()
        else 0.5
    )
    return metrics


@torch.no_grad()
def evaluate(
    model: TPVEPR,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    rows: list[dict[str, float]] = []
    for batch in loader:
        scores = model(
            batch["numeric"].to(device),
            batch["relative"].to(device),
            batch["categorical"].to(device),
            batch["mask"].to(device),
        ).cpu().numpy()
        relevance = batch["relevance"].numpy()
        masks = batch["mask"].numpy()
        for batch_index in range(scores.shape[0]):
            valid = masks[batch_index]
            row = group_ranking_metrics(
                relevance[batch_index][valid],
                scores[batch_index][valid],
            )
            row["group_size"] = float(valid.sum())
            rows.append(row)
    metric_names = ("ndcg@1", "ndcg@3", "ndcg@5", "hit@1", "pairwise_accuracy")
    return {
        "groups": float(len(rows)),
        "mean_group_size": float(np.mean([row["group_size"] for row in rows])),
        **{
            metric: float(np.mean([row[metric] for row in rows]))
            for metric in metric_names
        },
    }


def selection_score(metrics: dict[str, float]) -> float:
    return (
        0.30 * metrics["ndcg@1"]
        + 0.25 * metrics["ndcg@3"]
        + 0.15 * metrics["ndcg@5"]
        + 0.20 * metrics["hit@1"]
        + 0.10 * metrics["pairwise_accuracy"]
    )


def build_model(
    config: TPVEPRConfig,
    feature_state: dict,
    relation_indices: list[int],
    prior_indices: list[int],
) -> TPVEPR:
    cardinalities = [
        len(feature_state["categorical"][feature])
        for feature in config.categorical_features
    ]
    return TPVEPR(
        num_numeric=len(config.numeric_features),
        categorical_cardinalities=cardinalities,
        relation_numeric_indices=relation_indices,
        prior_numeric_indices=prior_indices,
        embedding_dim=config.embedding_dim,
        d_model=config.d_model,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        ffn_multiplier=config.ffn_multiplier,
        dropout=config.dropout,
    )


@torch.no_grad()
def score_dataset(
    model: TPVEPR,
    dataset: GroupedRankingDataset,
    loader: DataLoader,
    device: torch.device,
) -> pl.DataFrame:
    model.eval()
    flat_scores = np.full(dataset.frame.height, np.nan, dtype=np.float32)
    for batch in loader:
        scores = model(
            batch["numeric"].to(device),
            batch["relative"].to(device),
            batch["categorical"].to(device),
            batch["mask"].to(device),
        ).cpu().numpy()
        row_indices = batch["row_indices"].numpy()
        masks = batch["mask"].numpy()
        for batch_index in range(scores.shape[0]):
            valid = masks[batch_index]
            flat_scores[row_indices[batch_index][valid]] = scores[batch_index][valid]
    return (
        dataset.frame.with_columns(pl.Series("tpvepr_score", flat_scores))
        .with_columns(
            pl.col("tpvepr_score")
            .rank(method="dense", descending=True)
            .over("rank_group_id")
            .cast(pl.Int64)
            .alias("tpvepr_rank")
        )
        .sort("_dataset_source_row")
        .drop("_dataset_source_row")
    )


def train(
    dataset_path: Path,
    output_root: Path,
    config: TPVEPRConfig,
    *,
    promoter_upstream: int = 2000,
    promoter_downstream: int = 500,
) -> pl.DataFrame:
    config.validate()
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    torch.set_num_threads(max(1, min(torch.get_num_threads(), 12)))
    device = resolve_device(config.device)

    output_root = ensure_dir(output_root)
    model_directory = ensure_dir(output_root / "model")
    metrics_directory = ensure_dir(output_root / "metrics")
    predictions_directory = ensure_dir(output_root / "predictions")
    write_json(output_root / "run_config.json", config.to_dict())

    frame = read_table(dataset_path)
    required = {
        "rank_group_id",
        "split",
        "relevance",
        "variant_id",
        "variant_pos",
        *config.numeric_features,
        *config.categorical_features,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Prepared training dataset is missing columns: {missing}")

    feature_state = fit_feature_state(
        frame,
        config.numeric_features,
        config.categorical_features,
    )
    feature_to_index = {
        feature: index for index, feature in enumerate(config.numeric_features)
    }
    relation_indices = [feature_to_index[name] for name in config.relation_features]
    prior_indices = [feature_to_index[name] for name in config.prior_features]

    train_dataset = GroupedRankingDataset(frame, feature_state, config, split="train")
    validation_dataset = GroupedRankingDataset(frame, feature_state, config, split="validation")
    test_dataset = GroupedRankingDataset(frame, feature_state, config, split="test")
    train_loader = make_loader(train_dataset, config, shuffle=True)
    validation_loader = make_loader(validation_dataset, config, shuffle=False)
    test_loader = make_loader(test_dataset, config, shuffle=False)

    model = build_model(
        config,
        feature_state,
        relation_indices,
        prior_indices,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    checkpoint_path = model_directory / "tpvepr.pt"
    history: list[dict[str, float | int]] = []
    best_score = -float("inf")
    epochs_without_improvement = 0
    for epoch in range(1, config.epochs + 1):
        model.train()
        components: dict[str, list[float]] = {
            name: []
            for name in ("loss", "listwise", "pairwise", "top1", "calibration")
        }
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            scores = model(
                batch["numeric"].to(device),
                batch["relative"].to(device),
                batch["categorical"].to(device),
                batch["mask"].to(device),
            )
            loss, batch_components = ranking_loss(
                scores,
                batch["relevance"].to(device),
                batch["mask"].to(device),
                config,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            for name, value in batch_components.items():
                components[name].append(value)

        validation_metrics = evaluate(model, validation_loader, device)
        current_score = selection_score(validation_metrics)
        row: dict[str, float | int] = {
            "epoch": epoch,
            **{
                f"train_{name}": float(np.mean(values))
                for name, values in components.items()
            },
            **{
                f"validation_{name}": value
                for name, value in validation_metrics.items()
            },
            "selection_score": current_score,
        }
        history.append(row)
        print(
            f"epoch={epoch} loss={row['train_loss']:.6f} "
            f"val_ndcg@3={validation_metrics['ndcg@3']:.6f} "
            f"selection={current_score:.6f}"
        )

        if current_score > best_score:
            best_score = current_score
            epochs_without_improvement = 0
            torch.save(
                {
                    "format_version": 1,
                    "tpvepr_version": __version__,
                    "model_state": model.state_dict(),
                    "feature_state": feature_state,
                    "config": config.to_dict(),
                    "relation_indices": relation_indices,
                    "prior_indices": prior_indices,
                    "preprocessing": {
                        "promoter_upstream": promoter_upstream,
                        "promoter_downstream": promoter_downstream,
                    },
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                break

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    metric_rows = []
    for split, loader in (("validation", validation_loader), ("test", test_loader)):
        metric_rows.append(
            {
                "method": "TPVEPR",
                "split": split,
                **evaluate(model, loader, device),
            }
        )
    metrics = pl.DataFrame(metric_rows)
    metrics.write_csv(metrics_directory / "ranking_metrics.tsv", separator="\t")
    pl.DataFrame(history).write_csv(metrics_directory / "training_history.tsv", separator="\t")
    score_dataset(model, test_dataset, test_loader, device).write_csv(
        predictions_directory / "test_predictions.tsv",
        separator="\t",
    )
    write_json(metrics_directory / "feature_state.json", feature_state)
    print(f"Saved best checkpoint to {checkpoint_path}")
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the TPVEPR relation-aware ranker.")
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--device", default=None, help="Override config device: auto, cpu, cuda, cuda:0.")
    parser.add_argument("--promoter-upstream", type=int, default=2000)
    parser.add_argument("--promoter-downstream", type=int, default=500)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = TPVEPRConfig.from_json(args.config)
    if args.device is not None:
        config.device = args.device
    metrics = train(
        dataset_path=args.dataset,
        output_root=args.output_root,
        config=config,
        promoter_upstream=args.promoter_upstream,
        promoter_downstream=args.promoter_downstream,
    )
    print(metrics)


if __name__ == "__main__":
    main()
