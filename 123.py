default_year = df.loc[df['dflt_year']==1].groupby('vat_num')['year'].min()

last_report = (
    df[df['dflt_year']==0]
    .groupby('vat_num')['year']
    .max()
)

lag = (default_year - last_report).dropna()

print(lag.value_counts().sort_index())
print(lag.describe())
