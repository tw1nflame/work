import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit
from catboost import Pool

# any test object
i = 0
x1 = X_test.iloc[[i]]

# SHAP from CatBoost: (n_features + 1), last column is base (raw/logit)
shap = model.get_feature_importance(Pool(x1, cat_features=cat_features), type="ShapValues")[0]
phi, base = shap[:-1], shap[-1]

# order by |SHAP| (in raw space)
names = x1.columns.to_numpy()
order = np.argsort(-np.abs(phi))

top_k = 20
idx = order[:top_k]
phi_k = phi[idx]
names_k = names[idx]

# PD path and per-feature ΔPD (same order)
raw_steps = np.r_[base, base + np.cumsum(phi_k)]
pd_steps = expit(raw_steps)
d_pd = np.diff(pd_steps)

start_pd, end_pd = pd_steps[0], pd_steps[-1]
print("PD model:", float(model.predict_proba(x1)[:, 1]))
print("PD from SHAP path (top_k only):", float(end_pd))

# waterfall geometry
left = pd_steps[:-1]
y = np.arange(len(names_k))[::-1]

colors = np.where(d_pd >= 0, "red", "blue")
plt.figure(figsize=(10, 6))
plt.barh(y, np.abs(d_pd)[::-1], left=np.minimum(left, left + d_pd)[::-1], color=colors[::-1])
plt.yticks(y, names_k[::-1])
plt.xlabel("Δ PD")
plt.title(f"Waterfall in PD space (top {top_k})\nstart PD={start_pd:.4f} → end PD={end_pd:.4f}")
plt.axvline(0, linewidth=1)
plt.tight_layout()
plt.show()

# remaining features collapsed
phi_rest = phi[order[top_k:]]
other = expit(base + phi_k.sum() + phi_rest.sum()) - expit(base + phi_k.sum())
print(f"Other features total ΔPD: {other:+.6f}")
