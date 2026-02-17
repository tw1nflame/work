import pandas as pd
import matplotlib.pyplot as plt

# --- 1. топ-20 фич с наибольшей долей нулей ---
top20_zero_cols = zero_share_df.head(20).index.tolist()

# --- 2. получаем feature importance ---
fi_df = model.get_feature_importance(prettified=True)

fi_df = fi_df.rename(columns={
    "Feature Id": "feature",
    "Importances": "importance"
})

# --- 3. фильтруем только нужные фичи ---
fi_top_zero = (
    fi_df[fi_df["feature"].isin(top20_zero_cols)]
    .sort_values("importance", ascending=False)
    .reset_index(drop=True)
)

print("Feature importance for top 20 zero-share features:")
print(fi_top_zero)

# --- 4. график ---
plt.figure(figsize=(8,6))
plt.barh(fi_top_zero["feature"], fi_top_zero["importance"])
plt.gca().invert_yaxis()
plt.title("Feature importance for top 20 zero-share features")
plt.xlabel("Importance")
plt.show()
