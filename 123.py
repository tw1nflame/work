# === 8) ZERO / "0" COUNTS BY ROW (OLD vs NEW) ===
print("\n=== 8) ZERO / '0' COUNTS BY ROW ===")

# Берем общее пространство фичей (как в вашем сравнении)
feat_cols = [c for c in common_feature_cols if c in df_old.columns and c in df_new.columns]

# OLD
old_zero_mask = df_old[feat_cols].isin([0, '0'])
old_zero_cnt = old_zero_mask.sum(axis=1)

print("\nOLD:")
print("rows:", len(df_old))
print("feature cols checked:", len(feat_cols))
print("all-zero rows:", int((old_zero_cnt == len(feat_cols)).sum()), f"({(old_zero_cnt == len(feat_cols)).mean():.2%})")
print("zero-count per row (describe):")
print(old_zero_cnt.describe())
print("\nOLD zero-count distribution (tail 15):")
print(old_zero_cnt.value_counts().sort_index().tail(15))

# NEW
new_zero_mask = df_new[feat_cols].isin([0, '0'])
new_zero_cnt = new_zero_mask.sum(axis=1)

print("\nNEW:")
print("rows:", len(df_new))
print("feature cols checked:", len(feat_cols))
print("all-zero rows:", int((new_zero_cnt == len(feat_cols)).sum()), f"({(new_zero_cnt == len(feat_cols)).mean():.2%})")
print("zero-count per row (describe):")
print(new_zero_cnt.describe())
print("\nNEW zero-count distribution (tail 15):")
print(new_zero_cnt.value_counts().sort_index().tail(15))

# Дополнительно: доля нулей по всем ячейкам (по общим фичам)
old_zero_cells = int(old_zero_mask.to_numpy().sum())
new_zero_cells = int(new_zero_mask.to_numpy().sum())
old_total_cells = len(df_old) * len(feat_cols)
new_total_cells = len(df_new) * len(feat_cols)

print("\nZero-cell share in common feature space:")
print(f"OLD: {old_zero_cells}/{old_total_cells} = {old_zero_cells/old_total_cells:.2%}")
print(f"NEW: {new_zero_cells}/{new_total_cells} = {new_zero_cells/new_total_cells:.2%}")
