tmp = df_final.reset_index()

duplicates = (
    tmp[tmp['target'] == 1]
    .groupby('vat_num')
    .size()
)

print("Компаний с >1 строкой target=1:", (duplicates > 1).sum())
