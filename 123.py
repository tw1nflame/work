import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from catboost import Pool

# ========= 1) берем любой объект из test =========
i = 0
x1 = X_test.iloc[[i]]

shap_vals = model.get_feature_importance(
    Pool(x1, cat_features=cat_features),
    type="ShapValues"
)

phi = shap_vals[0, :-1].astype(float)   # SHAP (log-odds)
base = float(shap_vals[0, -1])          # E[f(X)]
feat_names = x1.columns.tolist()

# ========= 2) сортируем по |SHAP| =========
order = np.argsort(-np.abs(phi))
phi = phi[order]
feat_names = [feat_names[j] for j in order]

top_k = 15
phi_main = phi[:top_k]
names_main = feat_names[:top_k]
phi_other = phi[top_k:].sum()
n_other = len(phi) - top_k

# ========= 3) строим шаги =========
current = base
starts, ends = [], []
for v in phi_main:
    starts.append(current)
    current = current + v
    ends.append(current)

final_value = base + phi.sum()

# ========= 4) рисуем =========
fig, ax = plt.subplots(figsize=(8, 10))

# цвета (как в shap примерно)
pos_color = "#ff0051"   # красный
neg_color = "#1e88e5"   # синий

h = 0.6

def draw_segment(x0, x1, y, color):
    ax.barh(
        y=y,
        width=(x1 - x0),
        left=x0,
        height=h,
        color=color,
        alpha=0.9
    )

# подписи: черные + белая обводка
txt_effect = [pe.withStroke(linewidth=3, foreground="white")]

# рисуем top_k
for idx, (name, v, x0, x1_) in enumerate(zip(names_main, phi_main, starts, ends)):
    color = pos_color if v > 0 else neg_color
    draw_segment(x0, x1_, idx, color)

    # подпись всегда СНАРУЖИ бара
    pad = 0.02 * (ax.get_xlim()[1] - ax.get_xlim()[0]) if ax.has_data() else 0.05
    x_text = x1_ + (pad if v > 0 else -pad)
    ha = "left" if v > 0 else "right"

    ax.text(
        x_text, idx, f"{v:+.2f}",
        va="center", ha=ha,
        fontsize=10, color="black",
        path_effects=txt_effect
    )

# other features (внизу)
y_other = len(phi_main)
x0 = ends[-1] if len(ends) else base
x1_ = x0 + phi_other
color = pos_color if phi_other > 0 else neg_color
draw_segment(x0, x1_, y_other, color)

pad = 0.02 * (ax.get_xlim()[1] - ax.get_xlim()[0]) if ax.has_data() else 0.05
x_text = x1_ + (pad if phi_other > 0 else -pad)
ha = "left" if phi_other > 0 else "right"
ax.text(
    x_text, y_other, f"{phi_other:+.2f}",
    va="center", ha=ha,
    fontsize=10, color="black",
    path_effects=txt_effect
)

names_plot = names_main + [f"{n_other} other features"]
ax.set_yticks(np.arange(len(names_plot)))
ax.set_yticklabels(names_plot)

# оси/вид
ax.invert_yaxis()
ax.set_xlabel("Model output (log-odds)")
ax.set_title(f"Waterfall (log-odds)\n f(x) = {final_value:.3f}", fontsize=14)

# базовая вертикальная линия
ax.axvline(base, linestyle="--", color="gray", linewidth=1)

# легкая сетка по X
ax.grid(axis="x", linestyle=":", alpha=0.4)

# ====== фикс: E[f(X)] рисуем в координатах ОСИ (не данных) ======
# x = base в data coords, y чуть ниже оси: используем blended transform
ax.annotate(
    f"E[f(X)] = {base:.3f}",
    xy=(base, 0), xycoords=ax.get_xaxis_transform(),   # x в data, y в axes
    xytext=(0, -28), textcoords="offset points",
    ha="center", va="top",
    fontsize=11, color="gray"
)

plt.tight_layout()
plt.show()
