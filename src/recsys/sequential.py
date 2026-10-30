"""RePlay SASRec: окна, causal next-item labels и переносимый checkpoint."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import importlib.metadata
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any
import zipfile

import numpy as np
import pandas as pd

from .data import IdMappings


@dataclass(frozen=True)
class SASRecConfig:
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
    item_to_index = mappings.item_to_index
    ordered = events.sort_values(["user_id", "timestamp"], kind="stable")
    histories: dict[str, list[int]] = {}
    for user_id, item_id in ordered[["user_id", "item_id"]].itertuples(index=False):
        if item_id not in item_to_index:
            raise ValueError(f"Неизвестный item_id: {item_id}")
        histories.setdefault(str(user_id), []).append(item_to_index[str(item_id)])
    return histories


def sliding_windows(sequence: list[int], max_sequence_length: int = 100, stride: int = 50) -> list[list[int]]:
    """Окна длины <= max_len+1: последний элемент нужен как next-item label."""
    if max_sequence_length < 1 or stride < 1:
        raise ValueError("Длины окна и шага должны быть положительными")
    window_size = max_sequence_length + 1
    if len(sequence) <= window_size:
        return [sequence.copy()] if len(sequence) >= 2 else []
    starts = list(range(0, len(sequence) - window_size + 1, stride))
    last_start = len(sequence) - window_size
    if starts[-1] != last_start:
        starts.append(last_start)
    return [sequence[start:start + window_size] for start in starts]


def autoregressive_examples(sequence: list[int], max_sequence_length: int = 100,
                            stride: int = 50, padding_value: int = -1) -> tuple[np.ndarray, np.ndarray]:
    """Правое выравнивание input/next-item labels; padding labels равны -1."""
    windows = sliding_windows(sequence, max_sequence_length, stride)
    inputs = np.full((len(windows), max_sequence_length), padding_value, dtype=np.int64)
    labels = np.full_like(inputs, -1)
    for row, window in enumerate(windows):
        source, target = window[:-1], window[1:]
        inputs[row, -len(source):] = source
        labels[row, -len(target):] = target
    return inputs, labels


def causal_attention_mask(length: int) -> np.ndarray:
    """True выше диагонали: будущие позиции недоступны."""
    return np.triu(np.ones((length, length), dtype=bool), k=1)


def replay_training_frame(histories: dict[str, list[int]], mappings: IdMappings,
                          config: SASRecConfig) -> pd.DataFrame:
    """Одна строка на перекрывающееся окно, как ожидает ParquetModule RePlay."""
    user_to_index = mappings.user_to_index
    rows = [
        {"user_id": user_to_index[user_id], "item_id": window}
        for user_id, sequence in histories.items()
        for window in sliding_windows(sequence, config.max_sequence_length, config.window_stride)
    ]
    if not rows:
        raise ValueError("Нет последовательностей длины хотя бы два")
    return pd.DataFrame(rows)


def replay_prediction_frame(histories: dict[str, list[int]], mappings: IdMappings) -> pd.DataFrame:
    return pd.DataFrame({
        "user_id": [mappings.user_to_index[user] for user in histories],
        "item_id": [sequence for sequence in histories.values()],
        "seen_ids": [sequence for sequence in histories.values()],
    })


def version_info() -> dict[str, str]:
    versions = {}
    for package in ("replay-rec", "torch", "lightning", "numpy", "pandas"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def save_checkpoint_bundle(bundle_path: str | Path, checkpoint_path: str | Path,
                           mappings: IdMappings, config: SASRecConfig,
                           metrics: dict[str, Any]) -> Path:
    """ZIP содержит веса, mappings, config, metrics и версии для inference без fit."""
    bundle_path, checkpoint_path = Path(bundle_path), Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {"format_version": 1, "checkpoint": "model.ckpt", "mappings": mappings.to_dict(),
                "config": asdict(config), "metrics": metrics, "versions": version_info()}
    with zipfile.ZipFile(bundle_path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.write(checkpoint_path, "model.ckpt")
        archive.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
    return bundle_path


def load_checkpoint_bundle(bundle_path: str | Path, output_dir: str | Path | None = None
                           ) -> tuple[Path, IdMappings, SASRecConfig, dict[str, Any]]:
    """Распаковывает bundle и возвращает всё необходимое для reload-only inference."""
    target = Path(output_dir) if output_dir is not None else Path(tempfile.mkdtemp(prefix="sasrec_reload_"))
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(bundle_path) as archive:
        if not {"model.ckpt", "manifest.json"} <= set(archive.namelist()):
            raise ValueError("Некорректный checkpoint bundle")
        archive.extract("model.ckpt", target)
        manifest = json.loads(archive.read("manifest.json"))
    mappings = IdMappings.from_dict(manifest["mappings"])
    config = SASRecConfig(**manifest["config"])
    return target / "model.ckpt", mappings, config, manifest


def replay_components(config: SASRecConfig, item_count: int) -> dict[str, Any]:
    """Создаёт pinned RePlay 0.22.0 модель; import отложен для CPU-бейзлайнов."""
    try:
        from replay.data import FeatureHint, FeatureType
        from replay.data.nn import TensorFeatureInfo, TensorSchema
        from replay.nn.lightning import LightningModule
        from replay.nn.lightning.optimizer import OptimizerFactory
        from replay.nn.sequential import SasRec
    except ImportError as error:
        raise RuntimeError("Установите pip install -e '.[sasrec]'") from error
    schema = TensorSchema([TensorFeatureInfo(
        name="item_id", is_seq=True, padding_value=item_count, cardinality=item_count,
        embedding_dim=config.embedding_dim, feature_type=FeatureType.CATEGORICAL,
        feature_hint=FeatureHint.ITEM_ID,
    )])
    network = SasRec.from_params(schema=schema, embedding_dim=config.embedding_dim,
                                 max_sequence_length=config.max_sequence_length,
                                 num_heads=config.num_heads, num_blocks=config.num_blocks,
                                 dropout=config.dropout)
    module = LightningModule(network, optimizer_factory=OptimizerFactory(
        optimizer="adam", learning_rate=config.learning_rate))
    return {"schema": schema, "model": module}


def create_replay_network(config: SASRecConfig, item_count: int) -> Any:
    """Создаёт именно `replay.nn.sequential.SasRec` для собственного цикла."""
    try:
        from replay.data import FeatureHint, FeatureSource, FeatureType
        from replay.data.nn import TensorFeatureInfo, TensorFeatureSource, TensorSchema
        from replay.nn.sequential import SasRec
    except ImportError as error:
        raise RuntimeError("Установите pip install -e '.[sasrec]'") from error
    schema = TensorSchema([TensorFeatureInfo(
        name="item_id", feature_type=FeatureType.CATEGORICAL, is_seq=True,
        feature_hint=FeatureHint.ITEM_ID, cardinality=item_count, padding_value=item_count,
        embedding_dim=config.embedding_dim,
        feature_sources=[TensorFeatureSource(FeatureSource.INTERACTIONS, "item_id")],
    )])
    return SasRec.from_params(
        schema=schema, embedding_dim=config.embedding_dim, num_heads=config.num_heads,
        num_blocks=config.num_blocks, max_sequence_length=config.max_sequence_length,
        dropout=config.dropout,
    )


def fit_sasrec(histories: dict[str, list[int]], item_count: int, config: SASRecConfig,
               checkpoint_path: str | Path, validation_fn: Any | None = None,
               epochs: int | None = None, device: str = "cuda") -> tuple[Any, pd.DataFrame, int]:
    """Обучает autoregressive next-item objective и сохраняет лучшую validation-эпоху."""
    try:
        import torch
        from torch.utils.data import DataLoader, Dataset
    except ImportError as error:
        raise RuntimeError("Для SASRec нужен PyTorch") from error
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Включите GPU в Colab или явно передайте device='cpu' для smoke test")
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    padding = item_count

    class WindowDataset(Dataset):
        def __init__(self) -> None:
            self.windows = [window for sequence in histories.values() for window in
                            sliding_windows(sequence, config.max_sequence_length, config.window_stride)]

        def __len__(self) -> int:
            return len(self.windows)

        def __getitem__(self, index: int) -> tuple[Any, Any, Any]:
            window = self.windows[index]
            tokens = np.full(config.max_sequence_length + 1, padding, dtype=np.int64)
            tokens[-len(window):] = window
            inputs, labels = tokens[:-1], tokens[1:]
            mask = (inputs != padding) & (labels != padding)
            return torch.from_numpy(inputs), torch.from_numpy(labels), torch.from_numpy(mask)

    dataset = WindowDataset()
    if not len(dataset):
        raise ValueError("Нет последовательностей длины хотя бы два")
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(config.seed))
    model = create_replay_network(config, item_count).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    history, best_score, best_epoch, stale = [], -np.inf, 0, 0
    for epoch in range(1, (epochs or config.max_epochs) + 1):
        model.train()
        total_loss, total_targets = 0.0, 0
        for tokens, labels, mask in loader:
            tokens, labels, mask = tokens.to(device), labels.to(device), mask.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = model(feature_tensors={"item_id": tokens}, padding_mask=tokens.ne(padding),
                         positive_labels=labels.unsqueeze(-1),
                         target_padding_mask=mask.unsqueeze(-1))["loss"]
            if not torch.isfinite(loss):
                raise FloatingPointError("SASRec loss не является конечным")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            optimizer.step()
            count = int(mask.sum())
            total_loss += float(loss.detach()) * count
            total_targets += count
        score = None if validation_fn is None else float(validation_fn(model))
        history.append({"epoch": epoch, "train_loss": total_loss / total_targets,
                        "local_recall_at_10": score})
        improved = validation_fn is None or score > best_score
        if improved:
            best_score = best_score if score is None else score
            best_epoch, stale = epoch, 0
            torch.save({"model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                        "config": asdict(config), "item_count": item_count, "epoch": epoch}, checkpoint_path)
        else:
            stale += 1
        if validation_fn is not None and stale >= config.early_stopping_patience:
            break
    restored = restore_replay_model(checkpoint_path, config, item_count, device=device)
    return restored, pd.DataFrame(history), best_epoch


def predict_scores(model: Any, histories: dict[str, list[int]], users: list[str], item_count: int,
                   config: SASRecConfig, device: str = "cuda",
                   trained_item_mask: np.ndarray | None = None) -> np.ndarray:
    """Возвращает full-catalog score; seen filtering выполняется общим submission-модулем."""
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
            scores = model(feature_tensors={"item_id": tokens}, padding_mask=tokens.ne(item_count))["logits"]
        values = scores.float().cpu().numpy()
        if trained_item_mask is not None:
            values[:, ~np.asarray(trained_item_mask, dtype=bool)] = -np.inf
        batches.append(values)
    return np.concatenate(batches) if batches else np.empty((0, item_count), dtype=np.float32)


def trained_items(histories: dict[str, list[int]], item_count: int) -> np.ndarray:
    """Маска items с обучающим сигналом; cold embeddings нельзя рекомендовать."""
    mask = np.zeros(item_count, dtype=bool)
    for sequence in histories.values():
        mask[np.asarray(sequence, dtype=int)] = True
    return mask


def restore_replay_model(checkpoint_path: str | Path, config: SASRecConfig, item_count: int,
                         device: str = "cpu") -> Any:
    """Восстанавливает RePlay SasRec без обучения."""
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("Для SASRec нужен PyTorch") from error
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    restored = create_replay_network(config, item_count)
    restored.load_state_dict(payload["model_state"])
    restored.to(device)
    restored.eval()
    return restored


def copy_best_checkpoint(source: str | Path, destination: str | Path) -> Path:
    """Явная операция для сохранения checkpoint лучшей validation-эпохи."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return destination
