import pandas as pd

# === LOAD ===
df = pd.read_csv("trading_df.csv")

# --- базовые колонки ---
base_cols = ['year', 'vat_num', 'dflt_year']
fin_cols = [c for c in df.columns if c not in base_cols]

# --- сортировка по времени ---
df = df.sort_values(['vat_num', 'year'])

# === 1. находим первый дефолт у каждой компании ===
df['cum_default'] = df.groupby('vat_num')['dflt_year'].cumsum()

first_default_mask = (df['dflt_year'] == 1) & (df['cum_default'] == 1)
df_first_default = df[first_default_mask].copy()

# === 2. проверяем нулевые фин показатели ===
all_zero_mask = df_first_default[fin_cols].isin([0, '0']).all(axis=1)

n_total = len(df_first_default)
n_zero = all_zero_mask.sum()
n_nonzero = n_total - n_zero

print("=== FIRST DEFAULT DIAGNOSTICS ===")
print(f"Total first defaults: {n_total}")
print(f"With ALL-ZERO financials: {n_zero} ({n_zero / n_total:.2%})")
print(f"With NON-ZERO financials: {n_nonzero} ({n_nonzero / n_total:.2%})")

# === optional: посмотреть примеры ===
print("\nSample ZERO-report defaults:")
display(df_first_default[all_zero_mask].head())

print("\nSample NON-ZERO-report defaults:")
display(df_first_default[~all_zero_mask].head())
