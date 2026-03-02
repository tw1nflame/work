from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss

def data_split(
    df,
    year_col="year",
    train_year_max=2022,
    val_year=2023,
    test_year=2024,
    target_col="target",
):
    """
    df: DataFrame с MultiIndex, где один из уровней индекса = year_col
    year_col: имя уровня индекса с годом
    train_year_max: обучаем на годах <= этому
    val_year: год для валидации
    test_year: год для теста (обычно последний)
    target_col: имя целевой колонки (0/1)
    """
    X = df.drop(columns=[target_col], errors="ignore")
    y = df[target_col].astype(int)

    years = df.index.get_level_values(year_col)

    X_train, y_train = X[years <= train_year_max], y[years <= train_year_max]
    X_val, y_val     = X[years == val_year],       y[years == val_year]
    X_test, y_test   = X[years >= test_year],      y[years >= test_year]

    return X_train, y_train, X_val, y_val, X_test, y_test


def train_catboost(
    X_train,
    y_train,
    X_val,
    y_val,
    iterations_num=10_000,
    eval_metric="AUC",
    learning_rate=0.01,
    random_seed=42,
    verbose=200,
    early_stopping_rounds=200,
    class_weights=None,  # можно "Balanced"
):
    """
    X_train, y_train: train-выборка
    X_val, y_val: val-выборка для early stopping
    iterations_num: максимум итераций
    eval_metric: метрика CatBoost (например 'AUC' или 'Logloss')
    learning_rate: шаг обучения
    random_seed: сид
    verbose: частота логов
    early_stopping_rounds: patience для early stopping
    class_weights: None или 'Balanced'
    """
    if len(X_val) == 0:
        raise ValueError("Validation split is empty (check val_year).")

    # зафиксируем набор фичей по train и подгоним val под него
    X_val = X_val.reindex(columns=X_train.columns)

    cat_features = list(X_train.select_dtypes(include=["object"]).columns)

    train_pool = Pool(X_train, y_train, cat_features=cat_features)
    val_pool   = Pool(X_val, y_val, cat_features=cat_features)

    model = CatBoostClassifier(
        iterations=iterations_num,
        learning_rate=learning_rate,
        eval_metric=eval_metric,
        random_seed=random_seed,
        verbose=verbose,
        early_stopping_rounds=early_stopping_rounds,
        class_weights=class_weights,
        allow_writing_files=False,
    )

    model.fit(train_pool, eval_set=val_pool, use_best_model=True)
    return model


def calculate_metrics(model, X_test, y_test):
    """
    model: обученная модель с predict_proba
    X_test, y_test: тестовые данные
    """
    p = model.predict_proba(X_test)[:, 1]

    metrics = {
        "auc": roc_auc_score(y_test, p),
        "brier": brier_score_loss(y_test, p),
        "logloss": log_loss(y_test, p),
    }

    print("AUC:", metrics["auc"])
    print("Brier:", metrics["brier"])
    print("LogLoss:", metrics["logloss"])
    return metrics
