import numpy as np
import matplotlib.pyplot as plt
from catboost import Pool

# ========= 1. объект =========
i = 0
x1 = X_test.iloc[[i]]

shap_vals = model.get_feature_importance(
    Pool(x1, cat_features=cat_features),
    type="ShapValues"
)

phi = shap_vals[0, :-1].astype(float)   # SHAP (log-odds)
base = float(shap_vals[0, -1])          # E[f(X)]

feature_names = x1.columns.tolist()

# ========= 2. сортировка =========
order = np.argsort(-np.abs(phi))
phi = phi[order]
feature_names = [feature_names[j] for j in order]

top_k = 15
phi_main = phi[:top_k]
names_main = feature_names[:top_k]

phi_other = phi[top_k:].sum()

# ========= 3. шаги =========
vals = [base]
for v in phi_main:
    vals.append(vals[-1] + v)

vals = np.array(vals)

# ========= 4. отрисовка =========
fig, ax = plt.subplots(figsize=(8, 10))

y = np.arange(len(phi_main) + 1)

# функции для стрелок
def draw_bar(x0, x1, y, color):
    h = 0.6
    ax.fill_between(
        [x0, x1],
        [y - h/2, y - h/2],
        [y + h/2, y + h/2],
        color=color,
        alpha=0.9
    )

# ========= 5. рисуем =========
current = base

for idx, (name, val) in enumerate(zip(names_main, phi_main)):
    new_val = current + val
    color = "#ff0051" if val > 0 else "#1e88e5"

    draw_bar(current, new_val, idx, color)

    ax.text(
        new_val + 0.02*np.sign(val),
        idx,
        f"{val:+.2f}",
        va="center",
        fontsize=10,
        color=color
    )

    current = new_val

# ========= other features =========
new_val = current + phi_other
color = "#ff0051" if phi_other > 0 else "#1e88e5"
draw_bar(current, new_val, len(phi_main), color)

ax.text(
    new_val,
    len(phi_main),
    f"{phi_other:+.2f}",
    va="center",
    fontsize=10,
    color=color
)

names_plot = names_main + [f"{len(phi) - top_k} other features"]

# ========= подписи =========
ax.set_yticks(np.arange(len(names_plot)))
ax.set_yticklabels(names_plot)

ax.axvline(base, linestyle="--", color="gray")

final_value = base + phi.sum()

ax.set_title(
    f"Waterfall (log-odds)\n"
    f"f(x) = {final_value:.3f}",
    fontsize=14
)

ax.text(
    base,
    len(names_plot)+0.3,
    f"E[f(X)] = {base:.3f}",
    ha="center",
    fontsize=11,
    color="gray"
)

ax.invert_yaxis()
ax.set_xlabel("Model output (log-odds)")
plt.tight_layout()
plt.show()
