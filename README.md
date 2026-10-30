# HSE RecSys Challenge 2026

Финальное решение рекомендует 10 unseen-фильмов каждому пользователю. Быстрые классические модели и обучение SASRec разделены: обычный локальный запуск не требует GPU, PyTorch или RePlay.

## Структура

```text
notebooks/final_solution.ipynb            # EDA, 80/20, Popularity, ItemKNN, EASE, submissions
notebooks/sasrec_colab_standalone.ipynb   # единственное место обучения SASRec
src/recsys/                               # data, metrics, classical, submission, SASRec inference
sasrec_outputs/sasrec_full_bundle.zip     # готовая full-refit модель + mappings/config/manifest
sasrec_outputs/submission_sasrec.csv      # отправленный Kaggle CSV
submissions/                              # нумерованные результаты этапов
```

Исходные CSV читаются из локальной `data/`: `events.csv`, `user_features.csv`, `item_features.csv`, `submission_sample.csv`. Не публикуйте данные соревнования и новые веса без проверки правил и размера файлов.

## Результаты и submissions

| Файл | Эксперимент | Local Recall@10 | Kaggle Recall@10 |
|---|---|---:|---:|
| `1_submission_itemknn.csv` | ItemKNN, текущий 80/20; full-data refit | 0.055767* | 0.0594* |
| `2_submission_itemknn_genre.csv` | отдельная ранняя линия 90/10: implicit ItemKNN + genre, α=0.25 | 0.057316* | неизвестен |
| `3_submission_ease.csv` | EASE, λ=1000; full-data refit | 0.065038* | 0.0860* |
| `4_submission_sasrec.csv` | RePlay SASRec, текущий full-refit checkpoint | **0.141219**, epoch 49 | **0.2842** |

Звёздочкой отмечены исторические показатели соответствующих экспериментов; новый полный validation run при финальной очистке не выполнялся. Score SASRec 0.2842 подтверждён пользователем по Kaggle для неизменённого `sasrec_outputs/submission_sasrec.csv` от 30.10.2026. Нумерованная копия №4 побайтно совпадает с ним.

## Быстрый локальный запуск

```bash
uv sync --extra dev
uv run --with jupyterlab jupyter lab notebooks/final_solution.ipynb
```

Run All выполняет EDA, chronological 80/20, local Popularity/ItemKNN/EASE, затем full-data refit ItemKNN и выбранного EASE. Он создаёт submissions №1 и №3, проверяет восстановленный исторический №2 и побайтно копирует сохранённый SASRec в №4. SASRec не обучается.

## Обучение SASRec

Откройте `notebooks/sasrec_colab_standalone.ipynb` в чистом Google Colab, включите T4 GPU, загрузите четыре CSV и выполните Run All. Тетрадка автономна и содержит sliding windows, next-item loss, validation checkpoint selection и отдельный full-data refit. Существующую модель ради inference переобучать не нужно.

## Inference из готового bundle

Bundle уже доступен по пути `sasrec_outputs/sasrec_full_bundle.zip`; он содержит `model.ckpt`, исходные string-ID mappings, config, версии и learning curves metadata. История interactions в bundle не входит.

```bash
uv sync --extra sasrec
```

```python
import pandas as pd
from recsys.data import load_data
from recsys.sequential import submission_from_bundle

tables = load_data("data")
submission, manifest = submission_from_bundle(
    "sasrec_outputs/sasrec_full_bundle.zip",
    tables["events"],
    tables["sample"],
    device="cpu",  # или "cuda"
)
submission.to_csv("submissions/4_submission_sasrec_inferred.csv", index=False)
```

В конце основной тетрадки тот же путь доступен под флагом `RUN_SASREC_INFERENCE=False`. При `False` быстрый Run All не устанавливает тяжёлые зависимости.

## Ограничения

- Hidden-test relevance policy неизвестна; локальный holdout — прокси.
- EASE использует плотную item-item матрицу.
- Текущий checkpoint был фактически обучен в Colab, но его обучение не воспроизводилось в рамках финальной очистки.
- ALS, BERT4Rec и CatBoost не входят в финальную линию.
