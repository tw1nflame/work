# year уже есть и в индексе, и в колонке -> берем явно уровень индекса
y = final_df.index.get_level_values('year')
t = final_df['target']

res = (
    pd.DataFrame({'year': y, 'target': t})
    .groupby('year')['target']
    .agg(total='size', ones='sum')
)

res['zeros'] = res['total'] - res['ones']
res['target_1_pct'] = (100 * res['ones'] / res['total']).round(2)
res['target_0_pct'] = (100 * res['zeros'] / res['total']).round(2)

print(res[['zeros', 'ones', 'target_0_pct', 'target_1_pct']])
