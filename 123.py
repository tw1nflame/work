import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import expit  # sigmoid
from catboost import Pool

# возьмем любой объект из test (первый)
i = 0
x1 = X_test.iloc[[i]]  # DataFrame 1xN

# SHAP для CatBoost (log-odds / raw)
# CatBoost возвращает (n_objects, n_features + 1), последний столбец = expected_value (base, raw)
shap_vals = model.get_feature_importance(
    Pool(x1, cat_features=cat_features),
    type="ShapValues"
)
phi = shap_vals[0, :-1]   # SHAP по фичам (raw)
base = shap_vals[0, -1]   # base value (raw)

feat_names = x1.columns.to_list()

# --- порядок фичей в waterfall: по убыванию |SHAP| ---
order = np.argsort(-np.abs(phi))
phi_ord = phi[order]
names_ord = [feat_names[j] for j in order]

top_k = 20
phi_k = phi_ord[:top_k]
names_k = names_ord[:top_k]

# --- проверка: итоговая вероятность из raw ---
raw_final = base + phi.sum()
pd_from_shap = float(expit(raw_final))
pd_model = float(model.predict_proba(x1)[:, 1])

print("PD model:", pd_model)
print("PD from SHAP (raw -> sigmoid):", pd_from_shap)

# --- waterfall в RAW (log-odds) со знаками (+ красный, - синий) ---
# Начальные точки (left) для каждого бара: base + сумма предыдущих вкладов
raw_left = base + np.r_[0.0, np.cumsum(phi_k[:-1])]
raw_width = phi_k

# Цвета по знаку SHAP
colors = np.where(raw_width >= 0, "crimson", "steelblue")

# Для красивого водопада: бар рисуем от min(left, left+width), ширина = abs(width)
raw_start = np.minimum(raw_left, raw_left + raw_width)
raw_w = np.abs(raw_width)

# Для отображения сверху вниз (как в классических waterfall)
y = np.arange(top_k)[::-1]

plt.figure(figsize=(11, 7))
plt.barh(y, raw_w[::-1], left=raw_start[::-1], color=colors[::-1])
plt.yticks(y, names_k[::-1])
plt.xlabel("Δ raw (log-odds)")
plt.axvline(base, linestyle="--", linewidth=1)  # base line
plt.axvline(raw_final, linestyle=":", linewidth=1)  # final raw line

plt.title(
    f"Waterfall (RAW / log-odds, top {top_k})\n"
    f"base PD={expit(base):.4f} → final PD={expit(raw_final):.4f}"
)
plt.tight_layout()
plt.show()

# --- 'прочие' фичи одним блоком (raw и в PD) ---
other_raw = phi_ord[top_k:].sum()
other_pd = expit(raw_final) - expit(base + phi_k.sum())  # вклад остальных в PD (нелинеен, просто справочно)

print(f"Other features total Δraw: {other_raw:+.6f}")
print(f"Other features net ΔPD (reference): {other_pd:+.6f}")
