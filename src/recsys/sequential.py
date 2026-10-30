"""Reload-only inference для checkpoint RePlay SASRec."""

from __future__ import annotations

from dataclasses import dataclass
import importlib.metadata
import json
from pathlib import Path
import tempfile
from typing import Any
import zipfile

import numpy as np
import pandas as pd

from .data import IdMappings, interaction_matrix
from .submission import build_submission, recommendations_from_scores


@dataclass(frozen=True)
class SASRecConfig:
    """Полная схема config из сохранённого bundle, включая training metadata."""

    max_sequence_length: int = 100
    window_stride: int = 50
    embedding_dim: int = 64
    num_heads: int = 2
    num_blocks: int = 2
    dropout: float = 0.2
    batch_size: int = 128
    prediction_batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    gradient_clip: float = 1.0
    max_epochs: int = 50
    early_stopping_patience: int = 8
    seed: int = 42


def encoded_histories(events: pd.DataFrame, mappings: IdMappings) -> dict[str, list[int]]:
    """Кодирует полные хронологические истории в item indices из bundle."""
    item_to_index = mappings.item_to_index
    ordered = events.sort_values(["user_id", "timestamp"], kind="stable")
    histories: dict[str, list[int]] = {}
    for user_id, item_id in ordered[["user_id", "item_id"]].itertuples(index=False):
        item_id = str(item_id)
        if item_id not in item_to_index:
            raise ValueError(f"Неизвестный для checkpoint item_id: {item_id}")
        histories.setdefault(str(user_id), []).append(item_to_index[item_id])
    return histories


def version_info() -> dict[str, str]:
    versions = {}
    for package in ("replay-rec", "torch", "lightning", "numpy", "pandas"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def load_checkpoint_bundle(
    bundle_path: str | Path,
    output_dir: str | Path | None = None,
) -> tuple[Path, IdMappings, SASRecConfig, dict[str, Any]]:
    """Извлекает checkpoint и восстанавливает mappings/config/manifest."""
    bundle_path = Path(bundle_path)
    if not bundle_path.is_file():
        raise FileNotFoundError(bundle_path)
    target = Path(output_dir) if output_dir is not None else Path(
        tempfile.mkdtemp(prefix="sasrec_reload_")
    )
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(bundle_path) as archive:
        required = {"model.ckpt", "manifest.json"}
        if not required <= set(archive.namelist()):
            raise ValueError("Bundle должен содержать model.ckpt и manifest.json")
        checkpoint_path = target / "model.ckpt"
        checkpoint_path.write_bytes(archive.read("model.ckpt"))
        manifest = json.loads(archive.read("manifest.json"))
    if manifest.get("format_version") != 1:
        raise ValueError(f"Неподдерживаемый format_version: {manifest.get('format_version')}")
    mappings = IdMappings.from_dict(manifest["mappings"])
    config = SASRecConfig(**manifest["config"])
    return checkpoint_path, mappings, config, manifest


def create_replay_network(config: SASRecConfig, item_count: int) -> Any:
    """Создаёт архитектуру, совместимую с full checkpoint из standalone notebook."""
    try:
        from replay.data import FeatureHint, FeatureSource, FeatureType
        from replay.data.nn import TensorFeatureInfo, TensorFeatureSource, TensorSchema
        from replay.nn.sequential import SasRec
    except ImportError as error:
        raise RuntimeError("Для inference установите зависимости: pip install -e '.[sasrec]'") from error
    schema = TensorSchema([TensorFeatureInfo(
        name="item_id",
        feature_type=FeatureType.CATEGORICAL,
        is_seq=True,
        feature_hint=FeatureHint.ITEM_ID,
        cardinality=item_count,
        padding_value=item_count,
        embedding_dim=config.embedding_dim,
        feature_sources=[TensorFeatureSource(FeatureSource.INTERACTIONS, "item_id")],
    )])
    return SasRec.from_params(
        schema=schema,
        embedding_dim=config.embedding_dim,
        num_heads=config.num_heads,
        num_blocks=config.num_blocks,
        max_sequence_length=config.max_sequence_length,
        dropout=config.dropout,
    )


def restore_replay_model(
    checkpoint_path: str | Path,
    config: SASRecConfig,
    item_count: int,
    device: str = "cpu",
) -> Any:
    """Загружает full-refit weights; обучение и optimizer не создаются."""
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("Для SASRec inference нужен PyTorch") from error
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if int(payload.get("item_count", -1)) != item_count:
        raise ValueError("Размер item vocabulary в checkpoint не совпадает с mappings")
    network = create_replay_network(config, item_count)
    network.load_state_dict(payload["model_state"])
    network.to(device)
    network.eval()
    return network


def trained_items(histories: dict[str, list[int]], item_count: int) -> np.ndarray:
    """Исключает cold items со случайными embeddings из выдачи."""
    mask = np.zeros(item_count, dtype=bool)
    for sequence in histories.values():
        mask[np.asarray(sequence, dtype=int)] = True
    return mask


def predict_scores(
    model: Any,
    histories: dict[str, list[int]],
    users: list[str],
    item_count: int,
    config: SASRecConfig,
    device: str = "cpu",
    trained_item_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Вычисляет full-catalog scores батчами без gradients."""
    import torch

    model.eval()
    batches = []
    for start in range(0, len(users), config.prediction_batch_size):
        batch_users = users[start:start + config.prediction_batch_size]
        inputs = np.full((len(batch_users), config.max_sequence_length), item_count, dtype=np.int64)
        for row, user_id in enumerate(batch_users):
            tail = histories.get(user_id, [])[-config.max_sequence_length:]
            if tail:
                inputs[row, -len(tail):] = tail
        tokens = torch.as_tensor(inputs, device=device)
        with torch.inference_mode():
            scores = model(
                feature_tensors={"item_id": tokens},
                padding_mask=tokens.ne(item_count),
            )["logits"]
        values = scores.float().cpu().numpy()
        if trained_item_mask is not None:
            values[:, ~np.asarray(trained_item_mask, dtype=bool)] = -np.inf
        batches.append(values)
    return np.concatenate(batches) if batches else np.empty((0, item_count), dtype=np.float32)


def submission_from_bundle(
    bundle_path: str | Path,
    events: pd.DataFrame,
    sample: pd.DataFrame,
    *,
    device: str = "cpu",
    extract_dir: str | Path | None = None,
    k: int = 10,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Полный reload-only путь: bundle + known events → submission."""
    checkpoint, mappings, config, manifest = load_checkpoint_bundle(bundle_path, extract_dir)
    users = sample["user_id"].astype(str).tolist()
    unknown_users = set(users) - set(mappings.user_ids)
    if unknown_users:
        raise ValueError(f"Sample содержит неизвестные checkpoint users: {len(unknown_users)}")
    histories = encoded_histories(events, mappings)
    seen = interaction_matrix(events, mappings)
    popularity = np.asarray(seen.sum(axis=0)).ravel()
    mask = trained_items(histories, len(mappings.item_ids))
    model = restore_replay_model(checkpoint, config, len(mappings.item_ids), device=device)
    scores = predict_scores(
        model, histories, users, len(mappings.item_ids), config,
        device=device, trained_item_mask=mask,
    )
    recommendations = recommendations_from_scores(
        scores, users, mappings.item_ids, seen, mappings.user_to_index, k, popularity,
    )
    submission = build_submission(sample, recommendations, k)
    seen_by_user = events.groupby("user_id")["item_id"].agg(set).to_dict()
    if any(set(recommendations[user]) & seen_by_user.get(user, set()) for user in users):
        raise AssertionError("SASRec inference рекомендовал seen item")
    return submission, manifest
