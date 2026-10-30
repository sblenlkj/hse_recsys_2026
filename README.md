# HSE RecSys Challenge 2026

Компактное воспроизводимое решение, которое строит 10 рекомендаций фильмов на пользователя. Быстрые этапы EDA → Popularity → ItemKNN → EASE находятся в основной тетрадке; GPU-обучение RePlay SASRec вынесено в отдельный автономный Colab. Все модели используют одинаковый хронологический holdout: последние 20% истории каждого пользователя.

## Структура

```text
final_solution.ipynb   # быстрые EDA и классические модели
sasrec_colab_standalone.ipynb  # автономное GPU-обучение и inference
hypotheses.md          # история гипотез и подтверждённых результатов
src/recsys/data.py     # загрузка, split, mappings
src/recsys/metrics.py  # Recall@10
src/recsys/classical.py
src/recsys/sequential.py
src/recsys/submission.py
```

Ratings не порогуются: любое событие участвует в бинарной истории и любое validation-событие является target. Уже встречавшиеся items исключаются. Строковые item IDs сохраняются без преобразования в итоговом CSV.

## Быстрые этапы

`final_solution.ipynb` выполняет EDA, 80/20 split, Popularity, ItemKNN и EASE. Обычный Run All не импортирует RePlay/PyTorch, не требует GPU и не запускает SASRec.

## SASRec в чистом Colab

1. Загрузите только `sasrec_colab_standalone.ipynb` в Google Colab и включите **T4 GPU**.
2. Нажмите **Runtime → Run all**.
3. В upload-диалоге выберите `events.csv`, `user_features.csv`, `item_features.csv`, `submission_sample.csv`.
4. После validation и full-data refit скачайте `submission_sasrec.csv` и `sasrec_outputs.zip`. Архив содержит оба checkpoints, learning curves, mappings/config/version manifests и checkpoint bundles.

Standalone notebook не требует GitHub, Google Drive, `src/`, `README.md`, `pyproject.toml` или `pip install -e .`. Для inference из готового bundle установите в ней `RELOAD_ONLY=True`, загрузите `sasrec_full_bundle.zip` и те же CSV: bundle содержит модель, но не пользовательские истории.

Для локального окружения:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[sasrec,dev]'
jupyter lab final_solution.ipynb
```

Зависимости закреплены в `pyproject.toml` и `uv.lock`; `replay-rec==0.22.0`, Lightning и TorchMetrics зафиксированы, PyTorch ограничен совместимой веткой 2.x. `uv.lock` можно использовать через `uv sync --extra sasrec --extra dev`.

## Checkpoint и inference без обучения

После validation сохраняется лучший checkpoint. Финальный bundle содержит `model.ckpt`, mappings исходных ID, SASRec config, метрики и версии библиотек:

```python
from recsys.sequential import load_checkpoint_bundle, restore_replay_model

checkpoint, mappings, config, manifest = load_checkpoint_bundle("outputs/sasrec_bundle.zip")
model = restore_replay_model(checkpoint, config, len(mappings.item_ids))
```

Reload-only раздел автономной тетрадки загружает bundle и формирует рекомендации без вызова fit. Submission сохраняется в порядке пользователей из `submission_sample.csv`, имеет колонки `user_id,item_id` и ровно 10 уникальных unseen items в каждой строке.

## Ограничения

- Исторические метрики в тетрадке подписаны как исторические и не выдаются за новый запуск.
- Полный SASRec требует GPU; EASE требует плотную item-item матрицу в памяти.
- Способ формирования скрытого Kaggle test и официальный relevance threshold неизвестны.
- User/item features показаны в EDA, но Popularity, ItemKNN, EASE и SASRec их не используют.
