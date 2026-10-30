"""Загрузка, EDA, хронологический split и отображения ID."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

FILENAMES = ("events.csv", "user_features.csv", "item_features.csv", "submission_sample.csv")


@dataclass(frozen=True)
class IdMappings:
    user_ids: tuple[str, ...]
    item_ids: tuple[str, ...]

    @property
    def user_to_index(self) -> dict[str, int]:
        return {value: index for index, value in enumerate(self.user_ids)}

    @property
    def item_to_index(self) -> dict[str, int]:
        return {value: index for index, value in enumerate(self.item_ids)}

    def to_dict(self) -> dict[str, list[str]]:
        return {"user_ids": list(self.user_ids), "item_ids": list(self.item_ids)}

    @classmethod
    def from_dict(cls, value: dict[str, list[str]]) -> "IdMappings":
        return cls(tuple(value["user_ids"]), tuple(value["item_ids"]))


def resolve_data_dir(explicit: str | Path | None = None) -> Path:
    """Находит единственную папку, содержащую все четыре входных CSV."""
    candidates = [Path(explicit)] if explicit is not None else [Path("data"), Path("."), Path("/content/data")]
    matches = [path.resolve() for path in candidates if all((path / name).is_file() for name in FILENAMES)]
    if len(matches) != 1:
        raise FileNotFoundError(f"Ожидалась одна папка с {FILENAMES}; найдено: {matches}")
    return matches[0]


def load_data(data_dir: str | Path) -> dict[str, pd.DataFrame]:
    path = Path(data_dir)
    events = pd.read_csv(path / "events.csv", dtype={"user_id": str, "item_id": str})
    users = pd.read_csv(path / "user_features.csv", dtype={"user_id": str})
    items = pd.read_csv(path / "item_features.csv", dtype={"item_id": str})
    sample = pd.read_csv(path / "submission_sample.csv", dtype={"user_id": str, "item_id": str})
    if set(events) != {"user_id", "item_id", "rating", "timestamp"}:
        raise ValueError("events.csv имеет неожиданную схему")
    if list(sample) != ["user_id", "item_id"]:
        raise ValueError("submission_sample.csv должен содержать user_id, item_id")
    if events.isna().any().any() or not events["rating"].between(1, 5).all():
        raise ValueError("В событиях есть пропуски или rating вне [1, 5]")
    if events.duplicated(["user_id", "item_id"]).any():
        raise ValueError("Повторные пары user-item требуют отдельной политики")
    return {"events": events, "user_features": users, "item_features": items, "sample": sample}


def eda_summary(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    return pd.DataFrame([
        {"table": name, "rows": len(frame), "columns": len(frame.columns),
         "missing": int(frame.isna().sum().sum()), "duplicates": int(frame.duplicated().sum())}
        for name, frame in tables.items()
    ])


def chronological_user_split(events: pd.DataFrame, validation_fraction: float = 0.20) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Скрывает ceil(20%) последних событий каждого пользователя."""
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction должен лежать между 0 и 1")
    ordered = events.sort_values(["user_id", "timestamp"], kind="stable").copy()
    sizes = ordered.groupby("user_id")["item_id"].transform("size")
    if (sizes < 2).any():
        raise ValueError("Для split нужно минимум два события у каждого пользователя")
    tail = np.ceil(sizes * validation_fraction).astype(int).clip(lower=1)
    tail = np.minimum(tail, sizes - 1)
    positions = ordered.groupby("user_id").cumcount()
    valid_mask = positions >= sizes - tail
    train, valid = ordered.loc[~valid_mask].copy(), ordered.loc[valid_mask].copy()
    if not (train.groupby("user_id")["timestamp"].max() < valid.groupby("user_id")["timestamp"].min()).all():
        raise AssertionError("Нарушена хронология: проверьте одинаковые timestamp")
    if not set(train.index).isdisjoint(valid.index):
        raise AssertionError("Train и validation пересекаются")
    return train, valid


def make_mappings(user_features: pd.DataFrame, item_features: pd.DataFrame) -> IdMappings:
    return IdMappings(tuple(sorted(user_features["user_id"].astype(str).unique())),
                      tuple(sorted(item_features["item_id"].astype(str).unique())))


def interaction_matrix(events: pd.DataFrame, mappings: IdMappings) -> sparse.csr_matrix:
    rows = events["user_id"].map(mappings.user_to_index)
    cols = events["item_id"].map(mappings.item_to_index)
    if rows.isna().any() or cols.isna().any():
        raise ValueError("Событие содержит ID вне mappings")
    matrix = sparse.csr_matrix((np.ones(len(events), dtype=np.float32), (rows.astype(int), cols.astype(int))),
                               shape=(len(mappings.user_ids), len(mappings.item_ids)), dtype=np.float32)
    matrix.sum_duplicates()
    matrix.data[:] = 1.0
    return matrix


def ground_truth(events: pd.DataFrame) -> dict[str, set[str]]:
    return events.groupby("user_id")["item_id"].agg(set).to_dict()
