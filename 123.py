# если year в индексе:
tmp = final_df.assign(year=final_df.index.get_level_values('year'))

# количество 0/1 и % target=1 по годам
res = tmp.groupby('year')['target'].agg(total='size', ones='sum')
res['zeros'] = res['total'] - res['ones']
res['target_1_pct'] = (res['ones'] / res['total'] * 100).round(2)
res['target_0_pct'] = (res['zeros'] / res['total'] * 100).round(2)

print(res[['zeros', 'ones', 'target_0_pct', 'target_1_pct']])
