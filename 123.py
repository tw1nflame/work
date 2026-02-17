# === FEATURE IMPORTANCE ДЛЯ ВСЕХ ФИЧ ===

import pandas as pd
import matplotlib.pyplot as plt

# получаем importance из CatBoost
fi_df = model.get_feature_importance(prettified=True)

# приводим к удобным названиям
fi_df = fi_df.rename(columns={
    "Feature Id": "feature",
    "Importances": "importance"
})

# сортируем
fi_df = fi_df.sort_values("importance", ascending=False).reset_index(drop=True)

print("\nTop 30 features by importance:")
print(fi_df.head(30))

# --- график топ-20 ---
plt.figure(figsize=(10,6))
plt.barh(fi_df["feature"].head(20)[::-1], fi_df["importance"].head(20)[::-1])
plt.title("Top 20 Feature Importance")
plt.xlabel("Importance")
plt.show()
