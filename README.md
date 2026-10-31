# HSE RecSys Challenge 2026

Учебное исследование рекомендательных систем: от базовой популярности и item-based collaborative filtering до EASE и последовательного трансформера SASRec. Задача — рекомендовать каждому пользователю **10 ранее не встречавшихся фильмов**, метрика — **Recall@10**.

**Состав команды:** Александр Кауров (Alexander Kaurow), Артём Ляхов, Дмитрий Каневский (blenlkj), Елизавета Бичурина (Elizaveta Bichurina), Виктория Малашко, Илья Конзафаров (Konzafarov Ilia).

Работа выполнялась совместно: участники независимо исследовали различные подходы к рекомендательным системам, после чего результаты экспериментов были обсуждены и объединены. В данном репозитории представлены реализация моделей, история гипотез, результаты экспериментов и сохранённые веса SASRec для повторного inference без обучения.

**Лучший результат команды на Kaggle — Recall@10 = 0.2842.** Отправка submissions выполнялась через аккаунт **blenlkj**.

## Результаты

| Модель | Kaggle Recall@10 | Артефакт |
|---|---:|---|
| ItemKNN | 0.0647 | `submissions/1_submission_itemknn.csv` |
| EASE | 0.0860 | `submissions/2_submission_ease.csv` |
| **SASRec** | **0.2842** | `submissions/3_submission_sasrec.csv` |

Результаты подтверждены Kaggle submissions от 30.10.2026. Локальная оценка использует chronological holdout последних 20% истории каждого пользователя; она не тождественна скрытой проверке Kaggle.

Подробнее:
- [Архитектура и воспроизведение](docs/architecture.md) — структура `src`, назначение двух тетрадок, обучение и восстановление весов.
- [Гипотезы и результаты](docs/hypotheses.md) — исследовательская последовательность, local/Kaggle метрики, выводы.

## Структура

```text
src/recsys/                           # data, metrics, классические модели, SASRec inference
notebooks/final_solution.ipynb       # EDA, Popularity, ItemKNN, EASE
notebooks/sasrec_colab_standalone.ipynb  # автономное обучение SASRec в Colab
submissions/                         # проверенные Kaggle CSV
sasrec_outputs/                      # checkpoint, bundle, learning curves, оригинальный CSV
docs/architecture.md
docs/hypotheses.md
data/                                # events.csv, user_features.csv, item_features.csv, submission_sample.csv
```

## Быстрый запуск

Четыре CSV (`events.csv`, `user_features.csv`, `item_features.csv`, `submission_sample.csv`) размещаются в `data/`.

```bash
uv sync
uv run --with jupyterlab jupyter lab notebooks/final_solution.ipynb
```

Основная тетрадка проводит EDA, chronological split, local evaluation и создаёт нумерованные submissions. Обучение SASRec при обычном запуске **не выполняется**.

## Обучение и использование SASRec

Для обучения откройте `notebooks/sasrec_colab_standalone.ipynb` в Google Colab с GPU, загрузите четыре CSV и запустите все ячейки. Тетрадка самодостаточна относительно кода проекта: ей не нужен `src/recsys`, но нужны RePlay/PyTorch из установочных ячеек.

Уже обученная модель сохранена в `sasrec_outputs/sasrec_full_bundle.zip`. Для повторного inference без fit используется `src/recsys/sequential.py` и прежняя история interactions:

```python
from recsys.data import load_data
from recsys.sequential import submission_from_bundle

tables = load_data("data")
submission, manifest = submission_from_bundle(
    "sasrec_outputs/sasrec_full_bundle.zip",
    tables["events"],
    tables["sample"],
)
```

Для этого требуется среда с SASRec-зависимостями (`uv sync --extra sasrec`). 
