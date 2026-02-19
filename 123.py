import numpy as np
import pandas as pd


def build_dataset_with_gaps(
    df: pd.DataFrame,
    max_lag: int = 5,
    drop_cols=('reg_date', 'tax_regime'),
    id_col='vat_num',
    year_col='year',
    dflt_col='dflt_year',
) -> pd.DataFrame:
    """
    Для "сжатого" датасета (пустые годы удалены):
    - target = dflt_year в календарном (t+1)
    - фичи берём из последней доступной отчётности <= t (t-1, если нет — t-2, ... t-n)
    - is_missing_report = 1, если report_lag >= 1 (нет отчётности за t, используем старую)
    - report_lag = t - report_year (сколько лет "просрочена" отчётность)
    - можно ограничить max_lag (не брать слишком старую отчётность)
    """

    df = df.copy()
    df = df.drop(columns=list(drop_cols), errors='ignore')

    # ---- приведение типов ключей (важно для merge_asof)
    df[id_col] = df[id_col].astype(str)  # безопасно, если id бывает смешанных типов
    df[year_col] = pd.to_numeric(df[year_col], errors='coerce')
    df[dflt_col] = pd.to_numeric(df[dflt_col], errors='coerce')

    df = df.dropna(subset=[id_col, year_col, dflt_col]).copy()
    df[year_col] = df[year_col].astype(int)
    df[dflt_col] = df[dflt_col].astype(int)

    # ---- fin cols
    service_cols = [year_col, id_col, dflt_col]
    fin_cols = [c for c in df.columns if c not in service_cols]

    # ---- нормализуем '0' как числа там, где это возможно
    for c in fin_cols:
        if df[c].dtype == 'object':
            df[c] = pd.to_numeric(df[c], errors='ignore')

    # ---- сортировка (не обязательно для последующих шагов, но полезно)
    df = df.sort_values([id_col, year_col], kind='mergesort')

    # ---- удаляем строки, где все финфичи равны 0 / '0' (если вдруг ещё есть)
    if len(fin_cols) > 0:
        all_zeros_mask = df[fin_cols].isin([0, '0']).all(axis=1)
        df = df.loc[~all_zeros_mask].copy()

    # ============================================================
    # 1) target = dflt_year в календарном (year+1)
    # ============================================================
    base = df[[id_col, year_col, dflt_col]].drop_duplicates()

    base_next = base.rename(columns={dflt_col: 'target'}).copy()
    base_next[year_col] = base_next[year_col] - 1  # join по (id, year=t) -> target за t+1

    df = df.merge(base_next, on=[id_col, year_col], how='left')
    df = df.dropna(subset=['target']).copy()
    df['target'] = df['target'].astype(int)

    # ============================================================
    # 2) обрезаем после первого дефолта (включительно)
    # ============================================================
    df = df.sort_values([id_col, year_col], kind='mergesort')
    df['ever_defaulted'] = df.groupby(id_col)[dflt_col].cumsum()
    df = df[df['ever_defaulted'] <= 1].copy()

    # Берём только "здоровые" годы как точки наблюдения (как у тебя)
    obs = df[df[dflt_col] == 0][[id_col, year_col, 'target']].copy()

    # ============================================================
    # 3) Находим последнюю доступную отчётность <= t через merge_asof
    #    ВАЖНО: сортировка для merge_asof должна быть по on-key ГЛОБАЛЬНО:
    #           left:  [year, id], right: [report_year, id]
    # ============================================================
    rep = df[[id_col, year_col] + fin_cols].copy()
    rep = rep.rename(columns={year_col: 'report_year'})
    rep = rep.dropna(subset=[id_col, 'report_year']).copy()
    rep['report_year'] = pd.to_numeric(rep['report_year'], errors='coerce')
    rep = rep.dropna(subset=['report_year']).copy()
    rep['report_year'] = rep['report_year'].astype(int)

    obs = obs.dropna(subset=[id_col, year_col]).copy()
    obs[year_col] = pd.to_numeric(obs[year_col], errors='coerce')
    obs = obs.dropna(subset=[year_col]).copy()
    obs[year_col] = obs[year_col].astype(int)

    # КРИТИЧЕСКИ важно для устранения "left keys must be sorted"
    obs = obs.sort_values([year_col, id_col], kind='mergesort')
    rep = rep.sort_values(['report_year', id_col], kind='mergesort')

    merged = pd.merge_asof(
        obs,
        rep,
        left_on=year_col,
        right_on='report_year',
        by=id_col,
        direction='backward',
        allow_exact_matches=True
    )

    # если по компании нет ни одного отчётного года до t -> fin_cols будут NaN
    # такие наблюдения не пригодны для модели
    if len(fin_cols) > 0:
        merged = merged.dropna(subset=fin_cols).copy()

    # ============================================================
    # 4) report_lag + is_missing_report + ограничение max_lag
    # ============================================================
    merged['report_lag'] = merged[year_col] - merged['report_year']
    merged['is_missing_report'] = (merged['report_lag'] >= 1).astype(int)

    merged = merged[(merged['report_lag'] >= 0) & (merged['report_lag'] <= max_lag)].copy()

    # ============================================================
    # 5) финальные колонки/индекс
    # ============================================================
    merged = merged.drop(columns=['report_year'], errors='ignore')
    merged = merged.set_index([id_col, year_col]).sort_index()

    return merged


# ===== пример использования =====
# df = pd.read_csv('trading_small_df.csv', index_col=0)
# df_final = build_dataset_with_gaps(df, max_lag=5)
# print(df_final.shape)
# print(df_final['is_missing_report'].value_counts(dropna=False))
# df_final.head()
