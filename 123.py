import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from scipy.special import expit  # sigmoid
from catboost import Pool

# =========================
# 0) Выбери объект из test
# =========================
i = 0
x1 = X_test.iloc[[i]]  # DataFrame 1xN (одна строка)

# =========================
# 1) Получаем SHAP от CatBoost (RAW / log-odds)
# =========================
pool1 = Pool(x1, cat_features=cat_features)

# shap_vals: (n_objects, n_features + 1), последний столбец = base_value (raw)
shap_vals = model.get_feature_importance(pool1, type="ShapValues")

phi = shap_vals[0, :-1].astype(float)  # SHAP values по фичам (raw/log-odds)
base = float(shap_vals[0, -1])         # base_value (raw/log-odds)

feat_names = x1.columns.tolist()
data_values = x1.iloc[0].values        # значения фичей для подписей на графике

# =========================
# 2) Собираем shap.Explanation
# =========================
explanation = shap.Explanation(
    values=phi,
    base_values=base,
    data=data_values,
    feature_names=feat_names
)

# =========================
# 3) Проверки согласованности
# =========================
raw_final = base + phi.sum()
pd_from_shap = float(expit(raw_final))

# Важно: predict_proba у CatBoost обычно уже в вероятностях.
# Если у тебя дальше используется калибратор (cal), то сравнивай отдельно с cal.predict_proba
pd_model = float(model.predict_proba(x1)[:, 1])

print("=== CONSISTENCY CHECK ===")
print("raw base:", base)
print("raw final (base + sum(phi)):", raw_final)
print("PD from SHAP (sigmoid(raw_final)):", pd_from_shap)
print("PD from model.predict_proba:", pd_model)

# =========================
# 4) Waterfall (как в shap.plots.waterfall)
# =========================
max_display = 15  # сколько фичей показывать, остальное уйдет в "other features"

plt.figure(figsize=(10, 10))
shap.plots.waterfall(explanation, max_display=max_display, show=False)

plt.title(
    "Waterfall (RAW / log-odds)\n"
    f"base raw={base:.4f} (PD={expit(base):.4f}) → "
    f"final raw={raw_final:.4f} (PD={pd_from_shap:.4f})",
    fontsize=12
)
plt.tight_layout()
plt.show()

# =========================
# 5) Если у тебя есть калиброванная модель cal (CalibratedClassifierCV),
#    то waterfall все равно строится по raw-скорингу БАЗОВОЙ модели,
#    а калиброванный PD можно просто вывести рядом:
# =========================
try:
    pd_cal = float(cal.predict_proba(x1)[:, 1])  # если cal существует
    print("\nPD from calibrated model (cal.predict_proba):", pd_cal)
except NameError:
    pass
