# Архитектура решения HSE RecSys Challenge 2026
## 1. Общая идея

Мы отделили **переиспользуемую логику** от **запуска экспериментов**, а дорогое обучение SASRec — от повседневного запуска классических моделей.

```text
data/*.csv
   │
   ├── notebooks/final_solution.ipynb
   │     ├── EDA и chronological validation
   │     ├── Popularity → ItemKNN → EASE
   │     ├── local Recall@10
   │     └── submissions/1_*, 2_* + копия готового 3_*
   │
   └── notebooks/sasrec_colab_standalone.ipynb
         ├── самостоятельный запуск в Google Colab
         ├── 80/20 validation и SASRec training
         ├── best checkpoint → full-data refit
         └── sasrec_outputs/ (веса, bundle, submission)
                               │
                               ▼
                         src/recsys/sequential.py
                         reload-only inference
```

## 2. Структура репозитория

```text
README.md
docs/
  architecture.md             # этот документ
  hypotheses.md               # исследовательская история
notebooks/
  final_solution.ipynb
  sasrec_colab_standalone.ipynb
src/recsys/
  __init__.py
  data.py
  metrics.py
  classical.py
  submission.py
  sequential.py
submissions/
  1_submission_itemknn.csv
  2_submission_ease.csv
  3_submission_sasrec.csv
sasrec_outputs/
  sasrec_full_bundle.zip
  sasrec_full.ckpt
  sasrec_validation_best.ckpt
  sasrec_validation_bundle.zip
  submission_sasrec.csv
  validation_learning_curve.csv
  full_learning_curve.csv
pyproject.toml
uv.lock
```

`data/` содержит локальные CSV соревнования.

## 3. Ответственность модулей src

| Модуль | Что находится внутри | Где используется |
|---|---|---|
| `data.py` | Проверка и загрузка CSV, EDA, chronological per-user split, `IdMappings`, user–item CSR matrix, validation ground truth | Обе линии |
| `metrics.py` | Macro Recall@K по пользователям с непустым ground truth | Локальная оценка |
| `classical.py` | Popularity, ItemKNN, EASE и вычисление matrix scores | Основная тетрадка |
| `submission.py` | Фильтрация seen items, top-K, сохранение CSV, проверки схемы и уникальности | Обе линии, inference |
| `sequential.py` | `SASRecConfig`, чтение checkpoint bundle, создание совместимой сети RePlay, восстановление весов, batched scores, submission из bundle | Только SASRec inference |

Обучающего цикла SASRec в `src/recsys/sequential.py` больше нет. Он находится в автономной тетрадке. При этом `src` по-прежнему нужен для поддерживаемого локального восстановления уже обученной модели.

### Ключевая граница

`src/recsys` — источник переиспользуемых операций и inference; notebooks — экспериментальные сценарии.

Полностью автономная Colab-тетрадка **встраивает нужную реализацию в свои ячейки**. Поэтому при запуске на чистом Colab она не импортирует `recsys`, не требует Git clone и `pip install -e .`. Для самой модели SASRec внешняя библиотека RePlay и PyTorch всё же необходимы.

## 4. notebooks/final_solution.ipynb — быстрые эксперименты

Основная тетрадка делает:

1. Загрузка четырёх CSV и EDA.
2. Chronological 80/20 split **внутри истории каждого пользователя**.
3. Оценка Popularity, затем ItemKNN с cosine similarity и shrinkage.
4. Проверка EASE при нескольких коэффициентах регуляризации.
5. Сравнение local Recall@10.
6. Full-data refit классических моделей и экспорт CSV.
7. Подключение сохранённого SASRec submission **без нового обучения**.

По умолчанию Run All не обучает SASRec и не требует GPU. Восстановление checkpoint для самостоятельного inference — дополнительный режим, который может потребовать установки RePlay/PyTorch.

## 5. notebooks/sasrec_colab_standalone.ipynb — GPU training

Тетрадка содержит полный цикл, отдельно от `src`:

- загрузка тех же четырёх CSV через Colab;
- построение последовательностей и overlapping sliding windows;
- RePlay SASRec с causal next-item objective;
- validation Recall@10 после эпох;
- сохранение best checkpoint;
- обучение новой модели на **всех** известных взаимодействиях с выбранным числом эпох;
- создание `submission_sasrec.csv`;
- упаковка checkpoint, конфигурации, ID mappings и метаданных в ZIP;
- дополнительный `RELOAD_ONLY` сценарий без fit.

Фактический сохранённый запуск: **best epoch 49**, **local Recall@10 = 0.141219**, **Kaggle Recall@10 = 0.2842**.

Важно: checkpoint, полученный на train80, используется для выбора эпохи, а для Kaggle создаётся другая модель, обученная на всех событиях.

## 6. Инференс и переносимость весов

При inference требуются:

1. `sasrec_full_bundle.zip` (веса, model config, mappings);
2. `events.csv` (контекст последовательностей и полный список seen items);
3. `submission_sample.csv` (список и порядок пользователей);
4. установленная совместимая среда RePlay/PyTorch.

В `src/recsys/sequential.py` функция `submission_from_bundle(...)` восстанавливает сеть, переводит исходные IDs в индексы, считает оценки кандидатов, исключает известные items и формирует top-10 для каждого пользователя.

Bundle **не содержит историю пользовательских взаимодействий**, и поэтому одного ZIP без данных для нового inference недостаточно.

## 7. Результаты, файлы и проверка качества

| Этап | Файл в submissions/ | Kaggle Recall@10 |
|---|---|---:|
| ItemKNN | `1_submission_itemknn.csv` | 0.0647 |
| EASE | `2_submission_ease.csv` | 0.0860 |
| SASRec | `3_submission_sasrec.csv` | 0.2842 |

Третий файл — нумерованный вариант уже проверенного `sasrec_outputs/submission_sasrec.csv`, не результат повторного обучения в основной тетрадке.

Контроль корректности: исходные строковые IDs, 10 различных фильмов на пользователя, порядок строк по sample, исключение seen items и отсутствие утечки validation events при локальной оценке.

## 8. Почему такое разделение

- **Простота**: базовые эксперименты запускаются быстро, без GPU и большого ML-стека.
- **Воспроизводимость**: методы, разбиение и метрика вынесены в отдельные переиспользуемые функции.
- **Автономность Colab**: обучение SASRec не зависит от структуры Git-репозитория или абсолютных путей.
- **Повторное использование**: готовые веса можно загрузить локально, не повторяя обучение.
- **Проверяемая история**: numbered CSV отражают шаги исследования.
