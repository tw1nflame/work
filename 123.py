from catboost import CatBoostClassifier


def load_exp(exp_path: str):
    """
    Загружает эксперимент и возвращает:
    X_train, y_train, X_val, y_val, X_test, y_test, model, calibrated_model

    Параметры:
      exp_path — путь к папке эксперимента (например: "exp/20260302_120501_basic")
    """

    exp_dir = Path(exp_path)

    # ========= 1) DATA =========
    data_dir = exp_dir / "data"

    def _load_split(name):
        df = pd.read_parquet(data_dir / f"{name}.parquet")
        y = df["target"].copy()
        X = df.drop(columns=["target"])
        return X, y

    X_train, y_train = _load_split("train")
    X_val,   y_val   = _load_split("val")
    X_test,  y_test  = _load_split("test")

    # ========= 2) MODELS =========
    model_dir = exp_dir / "models"

    # базовая модель CatBoost
    model = CatBoostClassifier()
    model.load_model(str(model_dir / "model.cbm"))

    # калиброванная модель (может отсутствовать)
    cal_path = model_dir / "calibrated_model.joblib"
    calibrated_model = joblib.load(cal_path) if cal_path.exists() else None

    return (
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
        model,
        calibrated_model,
    )
