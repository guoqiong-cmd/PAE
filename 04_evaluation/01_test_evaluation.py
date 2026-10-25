"""Evaluate the saved final XGBoost pipeline on the fixed test set."""

import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "02_model_development"))
from model_config import records_to_xy, load_records


X_test, y_test = records_to_xy(load_records("test_set.npy"))
model = joblib.load(ROOT / "03_results" / "final_xgboost_model.joblib")
prediction = np.asarray(model.predict(X_test)).ravel()

print(f"R2 = {r2_score(y_test, prediction):.6f}")
print(f"RMSE = {np.sqrt(mean_squared_error(y_test, prediction)):.6f}")
print(f"MAE = {mean_absolute_error(y_test, prediction):.6f}")
