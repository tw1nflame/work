import pandas as pd
import numpy as np

# === 0) LOAD + ВАША ПРЕДОБРАБОТКА (как на скриншоте) ===
df = pd.read_csv('../trading_df.csv')

df = df.drop(columns=['cred_limit', 'fin_cond_index', 'tax_regime', 'reg_date'], errors='ignore')
df = df.sort_values(['vat_num', 'year']).copy()
df = df.drop(columns=['Unnamed: 0'], errors='ignore')

base_cols = ['year', 'vat_num', 'dflt_year']
fin_cols = [c for c in df.columns if c not in base_cols]

# 1) режем после первого дефолта
df['ever_defaulted'] = df.groupby('vat_num')['dflt_year'].cumsum()
df = df[df['ever_defaulted'] <= 1].copy()
df = df.drop(columns=['ever_defaulted'])

# 2) удаляем строки без отчетности (all-zero по фин.колонкам)
all_zero_mask = df[fin_cols].isin([0, '0']).all(axis=1)
df = df[~all_zero_mask].copy()

# target
df['target'] = df['dflt_year']
df = df.drop(columns=['dflt_year'])

# df_final как у вас
df_final = df.set_index(['vat_num', 'year']).sort_index()

# === 1) ДОБАВЛЯЕМ LAG-1 ПРИЗНАКИ ===
# Берем только фичи (без таргета) и сдвигаем по группе vat_num на 1 год назад
X_now = df_final.drop(columns=['target'], errors='ignore')
X_lag1 = X_now.groupby(level=0).shift(1)
X_lag1.columns = [f"{c}_lag1" for c in X_lag1.columns]

# Итоговый датасет: текущие фичи + lag1 + target
df_lag = pd.concat([X_now, X_lag1, df_final['target']], axis=1)

# === 2) СКОЛЬКО vat_num ИМЕЮТ ДОСТУПНЫЙ LAG-1 ПОСЛЕ ПРЕДОБРАБОТКИ ===
# Считаем, что lag-1 "есть", если для строки не все lag1-фичи NaN
lag1_cols = X_lag1.columns
has_lag1_row = ~df_lag[lag1_cols].isna().all(axis=1)

# на уровне компании: есть хотя бы одна строка с lag1
has_lag1_company = has_lag1_row.groupby(level=0).any()

n_companies_total = df_lag.index.get_level_values(0).nunique()
n_companies_with_lag1 = int(has_lag1_company.sum())
pct_with_lag1 = 100 * n_companies_with_lag1 / n_companies_total if n_companies_total else 0.0

print("=== LAG-1 COVERAGE (after preprocessing) ===")
print("Total vat_num:", n_companies_total)
print("vat_num with at least one row having lag-1:", n_companies_with_lag1)
print("Share with lag-1 (%):", round(pct_with_lag1, 2))

# (опционально) сколько строк в df_lag имеют lag1
print("\nRows total:", len(df_lag))
print("Rows with lag-1 available:", int(has_lag1_row.sum()), f"({has_lag1_row.mean():.2%})")

# === 3) (опционально) СДЕЛАТЬ "ЧИСТЫЙ" ДАТАСЕТ ДЛЯ ОБУЧЕНИЯ С LAG-1 ===
# Если хочешь учить только на строках, где lag1 реально есть:
df_lag_ready = df_lag[has_lag1_row].copy()

print("\n=== df_lag_ready shape (only rows with lag-1) ===")
print(df_lag_ready.shape)

# df_lag  -> содержит и строки без lag1 (NaN в lag1 признаках)
# df_lag_ready -> только строки, где лаг-1 доступен
