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

Transform: `src.load_data.drop_leakage_columns` удаляет технические и опасные признаки, а `src.predprocessing.prepare_features` извлекает признаки из date/time колонок.

Load: подготовленные `X` и `y` передаются в sklearn pipeline, где preprocessing и обучение выполняются единым воспроизводимым процессом.

## Исключенные признаки

Из обучения удаляются:

- `trip_id` - технический идентификатор поездки;
- `actual_departure_delay_min` - фактическая задержка отправления;
- `actual_arrival_delay_min` - фактическая задержка прибытия.

Колонки `actual_departure_delay_min` и `actual_arrival_delay_min` исключаются из-за data leakage: это фактические значения задержек, которые становятся известны после или во время поездки, поэтому их нельзя использовать для прогноза до поездки.

## Preprocessing

Preprocessing реализован через `sklearn.pipeline.Pipeline` и `ColumnTransformer`.

Для числовых признаков используются `SimpleImputer(strategy="median")` и `StandardScaler`.

Для категориальных признаков используются `SimpleImputer(strategy="constant", fill_value="unknown")` и `OneHotEncoder(handle_unknown="ignore", sparse_output=False)`.

Date/time признаки обрабатываются в коде: из колонок вроде `date` и `time` извлекаются `day_of_week`, `month` и `hour`, после чего исходные строки даты и времени удаляются.

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

- `src/main.py` - основная точка входа, собирает весь пайплайн end-to-end.
- `src/load_data.py` - загрузка CSV, проверка `delayed`, удаление leakage columns и разделение на `X`/`y`.
- `src/predprocessing.py` - date/time features и preprocessing pipeline.
- `src/train.py` - training pipeline, `SelectKBest`, сетки гиперпараметров и `GridSearchCV`.
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

Проект содержит `Dockerfile` для запуска обучения и `docker-compose.yml` с сервисами `mlflow` и `trainer`.

MLflow UI после запуска Docker Compose доступен по адресу:

```text
http://localhost:5000
```

## Команды запуска локально

```bash
python -m venv .venv
pip install -r requirements.txt
python -m src.main
pytest
mlflow ui
```

## Команды запуска через Docker

```bash
docker compose up --build
```

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
│   ├── predprocessing.py
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
