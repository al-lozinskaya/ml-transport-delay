# ML Transport Delay

Учебный ML-проект для прогнозирования задержек общественного транспорта. Главная цель проекта - показать автоматизированный ML-пайплайн: загрузку данных, preprocessing, feature selection, подбор модели, оценку качества, MLflow Tracking и запуск через Docker.

## Бизнес-задача

Оператору общественного транспорта важно заранее понимать риск задержки рейса. Такой прогноз помогает перераспределять ресурсы, информировать пассажиров и оценивать влияние погоды, трафика, часов пик и событий в городе.

## ML-задача

Задача формулируется как binary classification. Целевая переменная - `delayed`.

Главные метрики:

- `f1` - баланс precision и recall;
- `roc_auc` - качество ранжирования вероятностей задержки.

## Датасет

 Датасет взят с [kaggle - Public Transport Delays](https://www.kaggle.com/datasets/khushikyad001/public-transport-delays-with-weather-and-events) в учебных целях.

Файл данных находится в `data/public_transport_delays.csv`. В датасете есть признаки маршрута, станций, расписания, погоды, трафика, праздников, часов пик, сезона и целевая колонка `delayed`.

## ETL

Extract: `src.load_data.load_data` читает CSV через `pandas`.

Transform: `src.load_data.drop_leakage_columns` удаляет технические и опасные признаки, а `src.preprocessing.prepare_features` исправляет признаки событий и удаляет сырые временные колонки.

Load: подготовленные `X` и `y` передаются в sklearn pipeline, где preprocessing и обучение выполняются единым воспроизводимым процессом.

## Исключенные признаки

Из обучения удаляются:

- `trip_id` - технический идентификатор поездки;
- `actual_departure_delay_min` - фактическая задержка отправления;
- `actual_arrival_delay_min` - фактическая задержка прибытия.

Колонки `actual_departure_delay_min` и `actual_arrival_delay_min` исключаются из-за data leakage: это фактические значения задержек, которые становятся известны после или во время поездки, поэтому их нельзя использовать для прогноза до поездки.

## Preprocessing

Preprocessing реализован через `sklearn.pipeline.Pipeline` и `ColumnTransformer`.

Текущая обработка признаков:

1. Из обучающих данных удаляются признаки, которые нельзя использовать для честного прогноза:
   - `trip_id` — технический идентификатор поездки;
   - `actual_departure_delay_min` — фактическая задержка отправления, известна после события;
   - `actual_arrival_delay_min` — фактическая задержка прибытия, напрямую связана с `delayed`;
   - `delayed` хранится отдельно в `y` и не попадает в `X`.

2. Сырые временные колонки удаляются и не используются в обучении:
   - `date`;
   - `time`;
   - `scheduled_departure`;
   - `scheduled_arrival`.

3. Для событий исправляется логическая ошибка датасета: если `event_type == "None"`, то `event_attendance_est` принудительно становится `0`.

4. Дальше признаки делятся по типам:
   - числовые признаки проходят `SimpleImputer(strategy="median")` и `StandardScaler`;
   - категориальные признаки проходят `SimpleImputer(strategy="constant", fill_value="unknown")` и `OneHotEncoder(handle_unknown="ignore", sparse_output=False)`.

5. Все преобразования обучаются только внутри sklearn `Pipeline`, поэтому preprocessing не подглядывает в test-часть.

В модели используются такие признаки:

- маршрут и остановки: `transport_type`, `route_id`, `origin_station`, `destination_station`;
- погода: `weather_condition`, `temperature_C`, `humidity_percent`, `wind_speed_kmh`, `precipitation_mm`;
- события: `event_type`, `event_attendance_est`;
- дорожная и календарная ситуация: `traffic_congestion_index`, `holiday`, `peak_hour`, `weekday`, `season`.

Не используются в `X`:

- `delayed` — целевая переменная, хранится отдельно в `y`;
- `trip_id`, `actual_departure_delay_min`, `actual_arrival_delay_min` — исключены из-за leakage или технического характера;
- `date`, `time`, `scheduled_departure`, `scheduled_arrival` — удаляются без создания новых признаков.

После preprocessing сравниваются два режима:

- без отбора признаков: `selector="passthrough"`;
- с отбором признаков: `SelectKBest(score_func=f_classif)` с разными `k`.

В MLflow сохраняется artifact `preprocessing/preprocessing_summary.json`. В нем можно посмотреть фактические числовые и категориальные признаки, шаги preprocessing, выбранный selector и финальную модель.

## Ablation test

Ablation test нужен, чтобы проверить, какие группы признаков действительно помогают модели, а какие могут давать шум или переобучение. В проекте сравниваются несколько наборов признаков из `FEATURE_SETS` в `src/config.py`.

Используемые наборы:

- `all_features` — все доступные честные признаки после удаления leakage и сырых временных колонок:
  `transport_type`, `route_id`, `origin_station`, `destination_station`, погодные признаки, события, трафик, праздник, час пик, `weekday`, `season`.
- `without_stations` — без `origin_station` и `destination_station`, но с `route_id`.
- `without_route_and_stations` — без `route_id`, `origin_station`, `destination_station`.
- `weather_time_events_only` — только погода, события, трафик и календарные признаки из исходного датасета.

Ablation запускается как отдельный эксперимент через общий entrypoint `src.main`:

```bash
python -m src.main --mode ablation
```

Через Docker:

```bash
docker compose run --rm ablation
```

Результат сохраняется в:

```text
reports/figures/ablation_results.csv
```

В таблице ablation для каждого набора признаков сохраняются:

- название набора `feature_set`;
- количество признаков `feature_count`;
- список признаков `features`;
- лучшая модель `best_model`;
- лучшие параметры `best_params`;
- метрики `accuracy`, `balanced_accuracy`, `precision`, `recall`, `f1`, `f1_macro`, `roc_auc`.

## Forward Selection

Forward Selection — отдельный эксперимент для пошагового выбора исходных признаков. Он начинает с пустого набора, на каждом шаге пробует добавить один доступный признак, оценивает качество по cross-validation и оставляет признак с лучшим приростом.

В отличие от `SelectKBest`, этот эксперимент выбирает именно исходные колонки датасета, например `traffic_congestion_index` или `weather_condition`, а не отдельные one-hot признаки после кодирования.

Запуск локально:

```bash
python -m src.main --mode forward-selection
```

Через Docker:

```bash
docker compose run --rm forward-selection
```

Результат сохраняется в:

```text
reports/figures/forward_selection_results.csv
```

## Автоматизация ML-пайплайна

Пайплайн включает:

- preprocessing для числовых и категориальных признаков;
- `SelectKBest` для автоматического отбора признаков;
- подбор `k` для `SelectKBest`;
- сравнение `LogisticRegression`, `RandomForestClassifier`, `GradientBoostingClassifier`;
- сохранение test-метрик для лучшей конфигурации каждого семейства моделей;
- подбор гиперпараметров через `GridSearchCV`;
- автоматическую валидацию качества по порогам из `src/config.py`;
- мониторинг CPU и RAM во время обучения через `psutil`;
- логирование эксперимента в MLflow.

## Модули `src`

- `src/main.py` - общая точка входа: готовит данные, делает train/test split и выбирает эксперимент по `--mode`.
- `src/experiments/` - пакет экспериментов: общий интерфейс, registry и отдельные классы для `train`, `ablation`, `forward-selection`.
- `src/load_data.py` - загрузка CSV, проверка `delayed`, удаление leakage columns и разделение на `X`/`y`.
- `src/preprocessing.py` - удаление запрещенных/сырых временных колонок, исправление событий и preprocessing pipeline.
- `src/train.py` - training pipeline, `SelectKBest`, сетки гиперпараметров и `GridSearchCV`.
- `src/experiments/ablation_experiment.py` - ablation test по наборам признаков из `FEATURE_SETS`.
- `src/experiments/forward_selection_experiment.py` - пошаговый выбор исходных признаков через Forward Selection.
- `src/evaluate.py` - метрики, confusion matrix, classification report и графики.
- `src/monitor.py` - quality warnings, системные метрики CPU/RAM и MLflow logging.
- `src/config.py` - пути, целевая колонка, leakage columns и пороги качества.

## Метрики и отчеты

После обучения считаются accuracy, precision, recall, f1, roc_auc, confusion matrix и classification report.

Также во время запуска обучения собираются системные метрики:

- `cpu_percent_mean`;
- `cpu_percent_max`;
- `memory_percent_mean`;
- `memory_percent_max`.

Графики и отчеты сохраняются в `reports/figures/`:

- `confusion_matrix.png`;
- `metrics.png`;
- `model_comparison.csv`;
- `model_comparison.png`;
- `model_comparison_lines.png`;
- `metrics.json`;

Для удобного сравнения в MLflow каждая модель из текущего запуска дополнительно логируется как nested run с одинаковыми именами метрик (`accuracy`, `f1`, `roc_auc` и т.д.). Поэтому в UI MLflow можно открыть график одной метрики и сравнить линии/точки разных моделей в рамках одного запуска.
- `classification_report.txt`;
- `feature_importance.png`, если лучшая модель поддерживает feature importance.

## Docker и MLflow

Проект содержит `Dockerfile` для запуска обучения и `docker-compose.yml` с сервисами `mlflow`, `trainer`, `ablation` и `forward-selection`. Все ML-сервисы используют один Docker image, собранный из `Dockerfile`, но запускают `src.main` с разными `--mode`.

Docker нужен для воспроизводимого запуска: контейнер изолирует Python-зависимости, читает данные из `data/` и сохраняет результаты в понятные локальные папки. Docker image содержит код проекта и зависимости. Контейнер `trainer` запускает обучение, а результаты сохраняются не внутри одноразового контейнера, а в локальные папки проекта через bind mounts:

- `./data:/app/data:ro` - исходные данные доступны контейнеру только для чтения;
- `./reports:/app/reports` - графики, метрики и текстовые отчеты;
- `./models:/app/models` - обученная модель;
- `./mlruns:/app/mlruns` - MLflow-логи и артефакты экспериментов.

После завершения контейнера файлы остаются на локальной машине в `reports/`, `models/` и `mlruns/`. Генерируемые артефакты не коммитятся в GitHub; исключение можно сделать только для специально добавленных демонстрационных скриншотов в `docs/screenshots/`.

MLflow UI после запуска Docker Compose доступен по адресу:

```text
http://localhost:5000
```

## Команды запуска локально

```bash
python -m venv .venv
pip install -r requirements.txt
python -m src.main --mode train
python -m src.main --mode ablation
python -m src.main --mode forward-selection
pytest
mlflow ui
```

## Команды запуска через Docker

```bash
docker compose up --build
```

Если запускать через Docker Desktop кнопками, можно запускать отдельные сервисы:

- `trainer` — основное обучение и сравнение моделей;
- `ablation` — ablation test по наборам признаков;
- `forward-selection` — пошаговый выбор признаков;
- `mlflow` — MLflow UI.

Из терминала эти же режимы можно запускать так:

```bash
docker compose run --rm trainer
docker compose run --rm ablation
docker compose run --rm forward-selection
```

Посмотреть логи обучения и использование ресурсов можно командами:

```bash
docker compose logs trainer
docker stats
```

После запуска:

- MLflow UI доступен на `http://localhost:5000`;
- отчеты появляются в `reports/`;
- модель появляется в `models/`.

## CI/CD

В проекте есть GitHub Actions workflow `.github/workflows/ci.yml`. Он запускается на `push` и `pull_request`.

CI выполняет базовые проверки готовности проекта:

- checkout репозитория;
- установка Python 3.11;
- установка зависимостей из `requirements.txt`;
- проверка импорта пакета;
- запуск `pytest`;
- сборка Docker image.

Такой workflow имитирует непрерывную интеграцию: при отправке изменений в GitHub автоматически проверяются зависимости, тесты и возможность собрать контейнер для воспроизводимого запуска.

## Соответствие критериям задания

- Описание бизнес-задачи - выполнено.
- Схема ML-пайплайна - выполнено.
- ETL: Extract, Transform, Load - выполнено.
- Архитектура ML-модели - выполнено.
- AutoML/автоматизация - выполнено через сравнение моделей, `SelectKBest` и `GridSearchCV`.
- Метрики модели - выполнено: `accuracy`, `precision`, `recall`, `f1`, `roc_auc`.
- Визуализации и отчеты - выполнено, результаты сохраняются в `reports/figures/`.
- Тестирование - выполнено через `pytest`.
- Docker - выполнено через `Dockerfile` и `docker-compose.yml`.
- CI/CD - выполнено через GitHub Actions.
- Мониторинг качества и инфраструктуры - выполнено через пороги `f1`/`roc_auc`, MLflow и CPU/RAM-метрики через `psutil`.
- GitHub-репозиторий - выполнено.
- Материалы для сдачи - добавлены `docs/screenshots/README.md` и `docs/presentation_outline.md`.

## Структура проекта

```text
ml-transport-delay/
├── data/
│   └── public_transport_delays.csv
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── main.py
│   ├── load_data.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   └── monitor.py
├── tests/
│   ├── test_data.py
│   └── test_features.py
├── models/
│   └── .gitkeep
├── reports/
│   └── figures/
├── docs/
│   ├── presentation_outline.md
│   └── screenshots/
│       └── README.md
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── README.md
├── pytest.ini
├── .gitignore
└── .github/
    └── workflows/
        └── ci.yml
```

## Выводы для бизнеса

Проект показывает, как перейти от сырого CSV к воспроизводимому процессу обучения. Такой пайплайн проще защищать, проверять и развивать: можно добавлять новые признаки, сравнивать модели в MLflow и контролировать минимальное качество через автоматические проверки.
