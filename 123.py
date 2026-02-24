import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import expit  # sigmoid

# возьмем любой объект из test (первый)
i = 0
x1 = X_test.iloc[[i]]  # DataFrame 1xN

# SHAP для CatBoost:
# shap_vals: shape (1, n_features) для бинарной классификации
# expected_value: base value (в raw/логит шкале)
shap_vals = model.get_feature_importance(Pool(x1, cat_features=cat_features), type="ShapValues")
# CatBoost возвращает (n_objects, n_features + 1), последний столбец = expected_value (base)
phi = shap_vals[0, :-1]
base = shap_vals[0, -1]

# --- порядок фичей в waterfall: по убыванию |SHAP| ---
feat_names = x1.columns.to_list()
order = np.argsort(-np.abs(phi))
phi_ord = phi[order]
names_ord = [feat_names[j] for j in order]

# --- строим "пошаговую" PD траекторию ---
raw_steps = np.r_[base, base + np.cumsum(phi_ord)]
pd_steps = expit(raw_steps)                     # PD после каждого шага
pd_delta = np.diff(pd_steps)                    # вклад фичи уже в PD-шкале

# --- итоговая проверка (должно совпасть с predict_proba) ---
pd_model = float(model.predict_proba(x1)[:, 1])
pd_from_shap = float(pd_steps[-1])
print("PD model:", pd_model)
print("PD from SHAP path:", pd_from_shap)

# --- рисуем waterfall в PD ---
top_k = 20  # если фичей много, покажем топ-20
pd_delta_k = pd_delta[:top_k]
names_k = names_ord[:top_k]

start_pd = pd_steps[0]
end_pd = pd_steps[-1]

# позиции для накопления
cum = start_pd
lefts = []
widths = []
for d in pd_delta_k:
    lefts.append(min(cum, cum + d))
    widths.append(abs(d))
    cum += d

y = np.arange(len(names_k))[::-1]

plt.figure(figsize=(10, 6))
plt.barh(y, widths[::-1], left=np.array(lefts)[::-1])
plt.yticks(y, names_k[::-1])
plt.xlabel("Δ PD")
plt.title(f"Waterfall in PD space (top {top_k} features)\nstart PD={start_pd:.4f} → end PD={end_pd:.4f}")
plt.axvline(0, linewidth=1)
plt.tight_layout()
plt.show()

# если хочешь вывести еще "прочие" (все кроме top_k) одним блоком:
other = pd_delta[top_k:].sum()
print(f"Other features total ΔPD: {other:+.6f}")
