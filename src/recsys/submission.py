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


def write_submission(path: str | Path, frame: pd.DataFrame) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
