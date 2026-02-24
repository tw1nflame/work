import pandas as pd

df = pd.read_csv('trading_df.csv')
df = df.drop(columns=['cred_limit', 'fin_cond_index', 'tax_regime'], errors='ignore')
df = df.sort_values(['vat_num', 'year']).copy()

base_cols = ['year', 'vat_num', 'dflt_year']
fin_cols = [c for c in df.columns if c not in base_cols]

# 1) сначала режем после первого дефолта
df['ever_defaulted'] = df.groupby('vat_num')['dflt_year'].cumsum()
df = df[df['ever_defaulted'] <= 1].copy()
df = df.drop(columns=['ever_defaulted'])

# 2) потом удаляем строки без отчетности
all_zero_mask = df[fin_cols].isin([0, '0']).all(axis=1)
df = df[~all_zero_mask].copy()

print(len(df))
print(df['dflt_year'].value_counts())
