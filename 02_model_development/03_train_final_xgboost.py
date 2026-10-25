"""Fit the selected XGBoost configuration on the complete training set."""

import json

import joblib

from model_config import FINAL_PARAMETERS, RESULT_DIR, build_pipeline, load_records, records_to_xy


X_train, y_train = records_to_xy(load_records("train_set.npy"))
model = build_pipeline("XGBoost")
model.fit(X_train, y_train)

joblib.dump(model, RESULT_DIR / "final_xgboost_model.joblib")
with open(RESULT_DIR / "final_xgboost_parameters.json", "w", encoding="utf-8") as file:
    json.dump(FINAL_PARAMETERS["XGBoost"], file, indent=2)

print("Final XGBoost model saved.")
