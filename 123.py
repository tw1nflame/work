import pandas as pd

def make_lags_dataset(
    df: pd.DataFrame,
    lags=(1, 2, 3),
    target_col: str = "target",
    keep_only_with_all_lags: bool = False,
):
    """
    Добавляет несколько лагов (t-1, t-2, ...) к фичам и (опционально) фильтрует строки.

    df: DataFrame с MultiIndex (id, year) + колонка target_col
    lags: iterable лагов, например (1,2,3)
    target_col: имя таргета
    keep_only_with_all_lags:
        False -> оставлять строки, где есть ХОТЯ БЫ ОДИН лаг (по всем лаговым колонкам не все NaN)
        True  -> оставлять строки, где есть ВСЕ лаги (для каждого lag есть хотя бы одно не-NaN значение)
    """
    if not isinstance(df.index, pd.MultiIndex):
        raise ValueError("df должен иметь MultiIndex (id, year)")

    lags = sorted(set(int(l) for l in lags))
    if any(l <= 0 for l in lags):
        raise ValueError("lags должны быть положительными целыми")

    X_now = df.drop(columns=[target_col], errors="ignore")

    lag_blocks = []
    masks = []  # маски наличия каждого лага по строкам
    for lag in lags:
        X_lag = X_now.groupby(level=0).shift(lag)
        X_lag.columns = [f"{c}_lag{lag}" for c in X_lag.columns]
        lag_blocks.append(X_lag)

        # для этого lag: есть хотя бы одна не-NaN лаговая фича
        masks.append(~X_lag.isna().all(axis=1))

    df_lag = pd.concat([X_now, *lag_blocks, df[target_col]], axis=1)

    # фильтрация строк
    if keep_only_with_all_lags:
        keep_mask = pd.concat(masks, axis=1).all(axis=1)
    else:
        keep_mask = pd.concat(masks, axis=1).any(axis=1)

    return df_lag[keep_mask].copy()
