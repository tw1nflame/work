import numpy as np
import pandas as pd

def build_dataset_with_gaps(
    df: pd.DataFrame,
    max_lag: int = 5,
    drop_cols=('reg_date', 'tax_regime'),
    id_col='vat_num',
    year_col='year',
    dflt_col='dflt_year',
):
    df = df.copy()
    df = df.drop(columns=list(drop_cols), errors='ignore')

    # ---- базовая нормализация типов (очень важно для merge_asof)
    df[year_col] = pd.to_numeric(df[year_col], errors='coerce').astype('Int64')
    df = df.dropna(subset=[id_col, year_col])  # выкинуть NaN ключи
    df[year_col] = df[year_col].astype(int)

    # ---- fin cols
    service_cols = [year_col, id_col, dflt_col]
    fin_cols = [c for c in df.columns if c not in service_cols]

    # если где-то строки '0' – нормализуем
    # (если у тебя уже всё float/int — ок, просто не повредит)
    for c in fin_cols:
        if df[c].dtype == 'object':
            df[c] = pd.to_numeric(df[c], errors='ignore')

    # ---- сортировка
    df = df.sort_values([id_col, year_col], kind='mergesort')  # стабильная сортировка

    # ---- удалить строки, где все финфичи нулевые (если они ещё есть)
    all_zeros_mask = df[fin_cols].isin([0, '0']).all(axis=1)
    df = df.loc[~all_zeros_mask].copy()

    # ============================================================
    # 1) target = dflt_year в календарном (year+1)
    # ============================================================
    base = df[[id_col, year_col, dflt_col]].drop_duplicates()
    base_next = base.rename(columns={dflt_col: 'target'}).copy()
    base_next[year_col] = base_next[year_col] - 1  # чтобы join по (id, year=t) дал target за t+1

    df = df.merge(base_next, on=[id_col, year_col], how='left')

    # оставляем только строки, где известен следующий год (т.е. target не NaN)
    df = df.dropna(subset=['target']).copy()
    df['target'] = df['target'].astype(int)

    # ============================================================
    # 2) обрезать после первого дефолта и оставить только "здоровые" годы как точки наблюдения
    # ============================================================
    df['ever_defaulted'] = df.groupby(id_col)[dflt_col].cumsum()
    df = df[df['ever_defaulted'] <= 1].copy()

    obs = df[df[dflt_col] == 0][[id_col, year_col, 'target']].copy()

    # ============================================================
    # 3) last observation carried backward: берём последнюю отчётность <= year
    #    делаем репорт-таблицу и merge_asof
    # ============================================================
    rep = df[[id_col, year_col] + fin_cols].copy()
    rep = rep.rename(columns={year_col: 'report_year'})
    rep = rep.dropna(subset=[id_col, 'report_year']).copy()
    rep['report_year'] = pd.to_numeric(rep['report_year'], errors='coerce').astype(int)

    # сортировки, которые требуются merge_asof
    obs = obs.dropna(subset=[id_col, year_col]).copy()
    obs[year_col] = pd.to_numeric(obs[year_col], errors='coerce').astype(int)

    obs = obs.sort_values([id_col, year_col], kind='mergesort')
    rep = rep.sort_values([id_col, 'report_year'], kind='mergesort')

    merged = pd.merge_asof(
        obs,
        rep,
        left_on=year_col,
        right_on='report_year',
        by=id_col,
        direction='backward',
        allow_exact_matches=True
    )

    # если по компании нет ни одного отчётного года до obs-year -> fin_cols будут NaN
    # такие строки выкидываем (иначе модель не применима)
    merged = merged.dropna(subset=fin_cols).copy()

    # ============================================================
    # 4) лаг и флаги пропуска
    # ============================================================
    merged['report_lag'] = merged[year_col] - merged['report_year']
    merged['is_missing_report'] = (merged['report_lag'] >= 1).astype(int)

    # ограничение "t-n"
    merged = merged[(merged['report_lag'] >= 0) & (merged['report_lag'] <= max_lag)].copy()

    # техколонки
    merged = merged.drop(columns=['report_year'], errors='ignore')

    # индекс как раньше
    merged = merged.set_index([id_col, year_col]).sort_index()

    return merged
