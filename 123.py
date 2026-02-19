import numpy as np
import pandas as pd

def build_dataset_with_gaps(
    df: pd.DataFrame,
    max_lag: int = 5,             # насколько далеко назад разрешаем брать отчётность
    drop_cols=('reg_date', 'tax_regime'),
    id_col='vat_num',
    year_col='year',
    dflt_col='dflt_year',
):
    """
    Строит датасет для задачи "дефолт в следующем году".
    Для каждого года t, где есть таргет (t+1 существует), берём фичи из последнего доступного года <= t.
    Если последняя отчётность не за t (lag>0), ставим is_missing_report=1.
    Можно ограничить max_lag, чтобы не тянуть слишком старую отчётность.
    """

    df = df.copy()

    # 0) чистка
    df = df.drop(columns=list(drop_cols), errors='ignore')

    # служебные/фичи
    service_cols = [year_col, id_col, dflt_col]
    fin_cols = [c for c in df.columns if c not in service_cols]

    # привести "0" строки к числам где возможно (опционально)
    # если у тебя уже числовые — можно убрать
    for c in fin_cols:
        if df[c].dtype == 'object':
            df[c] = pd.to_numeric(df[c], errors='ignore')

    # 1) сортировка
    df = df.sort_values([id_col, year_col])

    # 2) выкинем строки, где ВСЕ финфичи нули (если такие ещё бывают)
    all_zeros_mask = df[fin_cols].isin([0, '0']).all(axis=1)
    df = df.loc[~all_zeros_mask].copy()

    # 3) строим таргет: дефолт в следующем наблюдаемом году (t -> t+1 по календарю)
    # Нам нужно именно календарный t+1, а не "следующая строка".
    # Поэтому создаём таблицу (id, year) -> dflt_year и джойним year+1.
    base = df[[id_col, year_col, dflt_col]].drop_duplicates()
    base_next = base.copy()
    base_next[year_col] = base_next[year_col] - 1  # чтобы при merge получить dflt_year(year+1)
    base_next = base_next.rename(columns={dflt_col: 'target'})

    df = df.merge(base_next, on=[id_col, year_col], how='left')

    # target определён только если в данных есть строка за year+1 (календарно)
    # иначе target = NaN и такие строки нам не нужны для обучения
    # (это соответствует твоему прежнему dropna(target))
    df = df[~df['target'].isna()].copy()
    df['target'] = df['target'].astype(int)

    # 4) Обрезаем после первого дефолта (оставляем до первого дефолта включительно как в старом пайплайне)
    df['ever_defaulted'] = df.groupby(id_col)[dflt_col].cumsum()
    df = df[df['ever_defaulted'] <= 1].copy()

    # 5) Берём только "здоровые" года как точки наблюдения (dflt_year==0),
    #    предсказываем переход 0->1
    df_obs = df[df[dflt_col] == 0].copy()

    # 6) Теперь ключевая часть: для каждой точки наблюдения (id, year=t),
    #    найти последний год с отчётностью <= t и взять оттуда fin_cols.
    #    Это делается merge_asof по year внутри каждой компании.
    rep = df[[id_col, year_col] + fin_cols].copy()
    rep = rep.sort_values([id_col, year_col])

    obs = df_obs[[id_col, year_col, 'target']].copy()
    obs = obs.sort_values([id_col, year_col])

    # merge_asof работает "по времени", но с ключом by=id_col
    merged = pd.merge_asof(
        obs,
        rep,
        on=year_col,
        by=id_col,
        direction='backward',
        allow_exact_matches=True
    )

    # last_report_year = year_col из репорта, который смержился (он же merged[year_col] уже t),
    # поэтому сохраним отдельно до merge: добавим в rep колонку report_year
    rep2 = df[[id_col, year_col] + fin_cols].copy()
    rep2 = rep2.rename(columns={year_col: 'report_year'}).sort_values([id_col, 'report_year'])

    merged = pd.merge_asof(
        obs,
        rep2,
        left_on=year_col,
        right_on='report_year',
        by=id_col,
        direction='backward',
        allow_exact_matches=True
    )

    # 7) считаем лаг и флаг пропуска отчётности
    merged['report_lag'] = merged[year_col] - merged['report_year']
    merged['is_missing_report'] = (merged['report_lag'] >= 1).astype(int)  # 1 = нет отчётности за t (используем старую)

    # 8) ограничиваем максимальный лаг (t-n)
    # если отчётность слишком старая — выкидываем такие наблюдения
    merged = merged[merged['report_lag'] <= max_lag].copy()

    # 9) финальная чистка: убираем техколонки
    merged = merged.drop(columns=['report_year'], errors='ignore')

    # индекс как раньше
    merged = merged.set_index([id_col, year_col]).sort_index()

    return merged


# пример использования:
# df_raw = pd.read_csv('trading_small_df.csv', index_col=0)
# df_final = build_dataset_with_gaps(df_raw, max_lag=5)
# df_final.head()
