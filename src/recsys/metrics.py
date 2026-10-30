"""Метрики локальной валидации."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Hashable


def recall_at_k(
    recommendations: Mapping[Hashable, Sequence[Hashable]],
    ground_truth: Mapping[Hashable, Sequence[Hashable] | set[Hashable]],
    k: int = 10,
) -> float:
    """Macro Recall@k по пользователям с непустым target.

    Дубликаты рекомендаций считаются ошибкой, а не дополнительными попаданиями.
    """
    if k <= 0:
        raise ValueError("k должен быть положительным")

    recalls: list[float] = []
    for user, relevant_items in ground_truth.items():
        relevant = set(relevant_items)
        if not relevant:
            continue
        predicted = list(recommendations.get(user, ()))[:k]
        if len(predicted) != len(set(predicted)):
            raise ValueError(f"Повторные рекомендации для {user}")
        recalls.append(len(relevant.intersection(predicted)) / len(relevant))
    if not recalls:
        raise ValueError("Нет пользователей с непустым target")
    return float(sum(recalls) / len(recalls))
