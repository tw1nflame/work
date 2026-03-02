from pathlib import Path
from datetime import datetime
import json
import pandas as pd

def save_exp(
    X_train, y_train, X_val, y_val, X_test, y_test,
    model,
    metrics: dict,
    fi_df: pd.DataFrame,
    calibrated_model=None,
    exp_root="exp",
    name=None,
    save_data=True,
    save_models=True,
    save_fi=True,
    save_meta=True,
):
    """
    Сохраняет эксперимент в папку exp/<timestamp>_<name>/.

    Параметры:
      X_train, y_train, X_val, y_val, X_test, y_test: сплиты
      model: базовая модель (CatBoost)
      metrics: словарь метрик (например, {"auc":..., "brier":..., "logloss":...})
      fi_df: DataFrame важности фич (например, columns: ["feature","importance"])
      calibrated_model: калиброванная модель (опционально)
      exp_root: корневая папка для экспериментов
      name: суффикс имени эксперимента (опционально)
      save_data/save_models/save_fi/save_meta: что сохранять
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = Path(exp_root) / (f"{ts}_{name}" if name else ts)
    exp_dir.mkdir(parents=True, exist_ok=True)

    # 1) DATA
    if save_data:
        data_dir = exp_dir / "data"
        data_dir.mkdir(exist_ok=True)

        def _save_split(X, y, split_name):
            df = X.copy()
            df["target"] = y.astype(int).values
            df.to_parquet(data_dir / f"{split_name}.parquet", index=True)

        _save_split(X_train, y_train, "train")
        _save_split(X_val,   y_val,   "val")
        _save_split(X_test,  y_test,  "test")

    # 2) MODELS
    if save_models:
        model_dir = exp_dir / "models"
        model_dir.mkdir(exist_ok=True)

        # CatBoost умеет save_model
        model.save_model(str(model_dir / "model.cbm"))
        if calibrated_model is not None:
            # CalibratedClassifierCV обычно sklearn-объект → через joblib
            import joblib
            joblib.dump(calibrated_model, model_dir / "calibrated_model.joblib")

    # 3) METRICS
    (exp_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    # 4) FEATURE IMPORTANCE
    if save_fi and fi_df is not None:
        fi_df.to_csv(exp_dir / "feature_importance.csv", index=False)

    # 5) META (параметры обучения/модели)
    if save_meta:
        meta = {}
        try:
            meta["catboost_params"] = model.get_params()
        except Exception:
            meta["catboost_params"] = None
        try:
            meta["catboost_all_params"] = model.get_all_params()
        except Exception:
            meta["catboost_all_params"] = None
        try:
            meta["best_iteration"] = int(model.get_best_iteration())
        except Exception:
            meta["best_iteration"] = None
        try:
            meta["best_score"] = model.get_best_score()
        except Exception:
            meta["best_score"] = None

        (exp_dir / "meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8"
        )

    return str(exp_dir)
