# X_now: текущие фичи
X_now = df_final.drop(columns=['target'], errors='ignore')

# lag-1
X_lag1 = X_now.groupby(level=0).shift(1)
X_lag1.columns = [f"{c}_lag1" for c in X_lag1.columns]

# lag-2
X_lag2 = X_now.groupby(level=0).shift(2)
X_lag2.columns = [f"{c}_lag2" for c in X_lag2.columns]

# итоговый датасет: текущие + lag1 + lag2 + target
df_lag = pd.concat([X_now, X_lag1, X_lag2, df_final['target']], axis=1)

# строки, где lag1 и lag2 реально доступны (не все NaN по соответствующим группам колонок)
lag1_cols = X_lag1.columns
lag2_cols = X_lag2.columns

has_lag1_row = ~df_lag[lag1_cols].isna().all(axis=1)
has_lag2_row = ~df_lag[lag2_cols].isna().all(axis=1)

# если хочешь учить только на строках, где есть и lag1, и lag2:
df_lag_ready = df_lag[has_lag1_row & has_lag2_row].copy()

print("Rows total:", len(df_lag))
print("Rows with lag1:", int(has_lag1_row.sum()), f"({has_lag1_row.mean():.2%})")
print("Rows with lag2:", int(has_lag2_row.sum()), f"({has_lag2_row.mean():.2%})")
print("Rows with lag1 & lag2:", len(df_lag_ready), f"({(has_lag1_row & has_lag2_row).mean():.2%})")
print("df_lag_ready shape:", df_lag_ready.shape)
