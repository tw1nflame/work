y_year = df_final.index.get_level_values('year').to_numpy()
t = df_final['target'].to_numpy()

res = (pd.DataFrame({'year': y_year, 'target': t})
       .groupby('year', as_index=True)['target']
       .agg(total='size', ones='sum'))

res['zeros'] = res['total'] - res['ones']
res['target_1_pct'] = (100 * res['ones'] / res['total']).round(2)
res['target_0_pct'] = (100 * res['zeros'] / res['total']).round(2)

print(res[['zeros', 'ones', 'target_0_pct', 'target_1_pct']])
