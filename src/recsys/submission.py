"""Seen filtering, top-10 и экспорт submission.csv."""

from __future__ import annotations

from pathlib import Path
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import sparse


def top_k_unseen(scores: np.ndarray, item_ids: Sequence[str], seen_indices: Sequence[int], k: int = 10,
                 popularity: np.ndarray | None = None) -> list[str]:
    values = np.asarray(scores, dtype=float).copy()
    values[np.asarray(seen_indices, dtype=int)] = -np.inf
    allowed = np.flatnonzero(np.isfinite(values))
    if len(allowed) < k:
        raise ValueError("Недостаточно unseen items для top-k")
    secondary = np.zeros_like(values) if popularity is None else np.asarray(popularity)
    order = np.lexsort((allowed, -secondary[allowed], -values[allowed]))
    return [str(item_ids[index]) for index in allowed[order[:k]]]


def recommendations_from_scores(scores: np.ndarray, users: Sequence[str], item_ids: Sequence[str],
                                seen: sparse.csr_matrix, user_to_index: Mapping[str, int], k: int = 10,
                                popularity: np.ndarray | None = None) -> dict[str, list[str]]:
    result = {}
    for offset, user_id in enumerate(users):
        row = user_to_index[user_id]
        seen_indices = seen.indices[seen.indptr[row]:seen.indptr[row + 1]]
        result[user_id] = top_k_unseen(scores[offset], item_ids, seen_indices, k, popularity)
    return result


def build_submission(sample: pd.DataFrame, recommendations: Mapping[str, Sequence[str]], k: int = 10) -> pd.DataFrame:
    output = sample[["user_id"]].copy()
    output["item_id"] = output["user_id"].map(lambda user: " ".join(map(str, recommendations[user])))
    if output["item_id"].str.split().map(len).ne(k).any():
        raise ValueError(f"Каждому пользователю нужно ровно {k} рекомендаций")
    if output["item_id"].str.split().map(lambda values: len(set(values))).ne(k).any():
        raise ValueError("В рекомендациях есть дубликаты")
    return output


def validate_submission(
    submission: pd.DataFrame,
    sample: pd.DataFrame,
    events: pd.DataFrame,
    catalog: Sequence[str],
    k: int = 10,
) -> None:
    """Проверяет schema, sample order, IDs, uniqueness и full-history seen filtering."""
    if list(submission.columns) != ["user_id", "item_id"]:
        raise ValueError("Submission должен содержать user_id, item_id")
    if submission["user_id"].astype(str).tolist() != sample["user_id"].astype(str).tolist():
        raise ValueError("Порядок user_id отличается от submission_sample.csv")
    rows = submission["item_id"].astype(str).str.split()
    if rows.map(len).ne(k).any() or rows.map(lambda values: len(set(values))).ne(k).any():
        raise ValueError(f"Каждая строка должна содержать {k} уникальных items")
    catalog_set = set(map(str, catalog))
    if rows.map(lambda values: set(values) <= catalog_set).eq(False).any():
        raise ValueError("Submission содержит item_id вне каталога")
    seen = events.groupby("user_id")["item_id"].agg(set).to_dict()
    for user_id, values in zip(submission["user_id"].astype(str), rows, strict=True):
        if set(values) & seen.get(user_id, set()):
            raise ValueError(f"Seen item в рекомендациях пользователя {user_id}")


def write_submission(path: str | Path, frame: pd.DataFrame) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
