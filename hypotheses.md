# Гипотезы и результаты

Основная финальная линия использует chronological holdout последних 20% событий каждого пользователя. Все ratings считаются взаимодействиями и validation targets; уже встречавшиеся items исключаются. Отдельный genre-эксперимент ниже использовал более ранний протокол 90/10 и поэтому не сравнивается с 80/20 как строго тот же эксперимент.

| Этап | Гипотеза → эксперимент | Local Recall@10 | Kaggle | Вывод |
|---|---|---:|---:|---|
| Popularity | Глобальная популярность unseen items даст sanity baseline | 0.036167* | — | Pipeline работает, персонализации недостаточно |
| ItemKNN | Совместные взаимодействия отражают близость фильмов; cosine, shrinkage, top-100 | 0.055767* | 0.0594* | Collaborative signal заметно улучшает baseline |
| ItemKNN + genre | В отдельной линии 90/10 rerank 100 implicit-KNN candidates по positive genre affinity, α=0.25 | 0.057316* | неизвестен | Небольшой дополнительный signal; не часть линии `recsys_start.ipynb` |
| EASE | Совместно обученные item-item веса лучше KNN; λ ∈ {100, 300, 1000, 3000, 10000} | **0.065038***, λ=1000 | **0.0860*** | Лучший классический результат |
| SASRec | Порядок событий важен; RePlay 0.22.0, length 100, stride 50, autoregressive next-item | **0.141219**, epoch 49 | **0.2842** | Последовательная модель дала основной прирост |

`*` — историческая метрика, не пересчитанная при финальной очистке. Для текущего SASRec local score и epoch взяты из `sasrec_full_bundle.zip`; Kaggle 0.2842 подтверждён пользователем для сохранённого `submission_sasrec.csv` 30.10.2026. Более ранняя сессия исходной линии показывала 0.141546 на epoch 48 — это другой запуск, и числа не смешиваются.

Нумерованные артефакты фиксируют путь исследования:

1. `submissions/1_submission_itemknn.csv` — full-data ItemKNN.
2. `submissions/2_submission_itemknn_genre.csv` — точный исторический CSV отдельной genre-гипотезы, восстановленный из Git history.
3. `submissions/3_submission_ease.csv` — full-data EASE с λ=1000.
4. `submissions/4_submission_sasrec.csv` — побайтная копия подтверждённого Kaggle submission.

Обучение SASRec полностью остаётся в автономной Colab-тетрадке. `src/recsys/sequential.py` содержит только совместимое восстановление сети, bundle/mappings, batched prediction и submission inference; optimizer, loss loop и sliding-window dataset там отсутствуют. ALS, BERT4Rec и CatBoost намеренно исключены.
