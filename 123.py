import numpy as np
import pandas as pd

from autogluon.tabular import TabularPredictor
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss
from sklearn.isotonic import IsotonicRegression


# ============================================================
# 0) Подготовка данных (как у тебя)
# ============================================================

df_ag = df_final.reset_index()  # vat_num, year как колонки

X = df_ag.drop(columns=['target', 'fin_cond_index', 'cred_limit'], errors='ignore')
y = df_ag['target'].astype(int)
years = df_ag['year']

train_year_max = 2021
val_year = 2022
test_year = 2023

# year/vat_num только для сплита, в фичи не даём
X_model = X.drop(columns=['vat_num', 'year'], errors='ignore')

X_train, y_train = X_model[years <= train_year_max], y[years <= train_year_max]
X_val,   y_val   = X_model[years == val_year],      y[years == val_year]
X_test,  y_test  = X_model[years >= test_year],     y[years >= test_year]

train_data = X_train.copy()
train_data['target'] = y_train.values

val_data = X_val.copy()
val_data['target'] = y_val.values


# ============================================================
# 1) Обучение AutoGluon (аналог model.fit(train, eval_set=val))
# ============================================================

predictor = TabularPredictor(
    label='target',
    problem_type='binary',
    eval_metric='roc_auc',
    verbosity=2,
).fit(
    train_data=train_data,
    tuning_data=val_data,
    presets='medium_quality',   # можно 'best_quality'
)


# ============================================================
# 2) Оценка ДО калибровки (как у тебя)
# ============================================================

p_val = predictor.predict_proba(X_val)[1].values
p_test = predictor.predict_proba(X_test)[1].values

print("Before calibration:")
print(" AUC test:   ", roc_auc_score(y_test, p_test))
print(" Brier test: ", brier_score_loss(y_test, p_test))
print(" LogLoss test:", log_loss(y_test, p_test))


# ============================================================
# 3) Калибровка isotonic "втупую" по предсказанным вероятностям
#    (то же самое, что делает isotonic-калибратор, но без CalibratedClassifierCV)
# ============================================================

iso = IsotonicRegression(out_of_bounds='clip')
iso.fit(p_val, y_val.values)

p_test_cal = iso.transform(p_test)

print("\nAfter calibration (isotonic on val probs):")
print(" AUC test:   ", roc_auc_score(y_test, p_test_cal))
print(" Brier test: ", brier_score_loss(y_test, p_test_cal))
print(" LogLoss test:", log_loss(y_test, p_test_cal))
