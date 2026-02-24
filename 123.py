import pandas as pd

# 1) Берем 2024 (year может быть в индексе или в колонке)
if 'year' in df_final.columns:
    df_2024 = df_final[df_final['year'] == 2024].copy()
else:
    df_2024 = df_final[df_final.index.get_level_values('year') == 2024].copy()

# 2) Исключаем служебные колонки (если вдруг есть)
service_cols = {'target', 'dflt_year', 'year', 'vat_num'}
fin_cols = [c for c in df_2024.columns if c not in service_cols]

# 3) Считаем нули (0 и '0') по колонкам
zero_mask = df_2024[fin_cols].isin([0, '0'])

res_2024_zeros = pd.DataFrame({
    'zero_count': zero_mask.sum(),
    'rows_2024': len(df_2024)
})
res_2024_zeros['zero_pct'] = (100 * res_2024_zeros['zero_count'] / res_2024_zeros['rows_2024']).round(2)

# 4) Сортировка: где больше всего нулей
res_2024_zeros = res_2024_zeros.sort_values(['zero_count', 'zero_pct'], ascending=False)

print(f"Rows in 2024: {len(df_2024)}")
print(res_2024_zeros)

# (опционально) только колонки, где есть хотя бы один ноль
# print(res_2024_zeros[res_2024_zeros['zero_count'] > 0])
