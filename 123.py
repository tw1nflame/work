# Обучение CatBoost только на топ-20 признаках по feature importance
# Предполагается, что уже есть: df_final (с индексом vat_num, year), обученная model (базовая) ИЛИ feat_importances,
# и что целевая колонка называется 'target'.

from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score
import numpy as np

# --- 1) Получаем топ-20 фич по importance из уже обученной модели ---
fi_df = model.get_feature_importance(prettified=True).rename(
    columns={"Feature Id": "feature", "Importances": "importance"}
).sort_values("importance", ascending=False)

top20_features = fi_df["feature"].head(20).tolist()
print("Top-20 features:", top20_features)

# --- 2) Готовим X/y только по этим фичам ---
X = df_final[top20_features]
y = df_final["target"].astype(int)

years = df_final.index.get_level_values("year")
train_year_max = 2021
val_year = 2022
test_year = 2023

X_train, y_train = X[years <= train_year_max], y[years <= train_year_max]
X_val, y_val     = X[years == val_year],      y[years == val_year]
X_test, y_test   = X[years >= test_year],     y[years >= test_year]

cat_features = list(X_train.select_dtypes(include=["object"]).columns)

train_pool = Pool(X_train, y_train, cat_features=cat_features)
val_pool   = Pool(X_val,   y_val,   cat_features=cat_features)
test_pool  = Pool(X_test,  y_test,  cat_features=cat_features)

# --- 3) Обучаем модель заново на топ-20 ---
model_top20 = CatBoostClassifier(
    iterations=3000,
    learning_rate=0.03,
    eval_metric="AUC",
    random_seed=42,
    verbose=200,
    early_stopping_rounds=200,
    auto_class_weights="Balanced",
    allow_writing_files=False
)

model_top20.fit(train_pool, eval_set=val_pool, use_best_model=True)

# --- 4) Быстрая оценка AUC на test ---
p_test = model_top20.predict_proba(X_test)[:, 1]
print("AUC test (top-20 features):", roc_auc_score(y_test, p_test))
