# default_year в исходных данных: минимальный год, где dflt_year==1
default_year = df.loc[df['dflt_year'] == 1].groupby('vat_num')['year'].min()

# есть ли строка за год до дефолта
years_per_company = df.groupby('vat_num')['year'].apply(set)
has_prev_row = default_year.index.to_series().apply(
    lambda vat: (default_year[vat] - 1) in years_per_company[vat]
)

print("Доля дефолтов, у которых есть строка за t-1:", has_prev_row.mean())
print("Нет строки t-1 (кол-во):", (~has_prev_row).sum())
