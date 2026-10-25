"""Nested XGBoost validation at the experimental-condition level."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, RandomizedSearchCV


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "02_model_development"))
from model_config import build_pipeline, load_all_records, prefixed_search_space, records_to_xy


records = load_all_records()
X, y = records_to_xy(records)
groups = records["condition_group_ID"].astype(int)

if len(y) != 701:
    raise ValueError(f"Expected 701 observations, found {len(y)}")
if len(np.unique(groups)) != 122:
    raise ValueError(
        f"Expected 122 experimental-condition groups, found {len(np.unique(groups))}"
    )

prediction = np.full(len(y), np.nan)

for fold, (train_index, test_index) in enumerate(
    GroupKFold(n_splits=5).split(X, y, groups), start=1
):
    search = RandomizedSearchCV(
        build_pipeline("XGBoost", use_final_parameters=False, grouped=True),
        prefixed_search_space("XGBoost"), n_iter=80, scoring="r2",
        cv=GroupKFold(n_splits=3), n_jobs=-1, random_state=42,
        refit=True, error_score="raise",
    )
    search.fit(X.iloc[train_index], y[train_index], groups=groups[train_index])
    prediction[test_index] = search.predict(X.iloc[test_index])
    print(f"Experimental-condition-level validation: fold {fold}/5 complete")

summary = pd.DataFrame([{
    "validation": "Experimental-condition-level nested five-fold CV",
    "n": len(y),
    "groups": len(np.unique(groups)),
    "R2": r2_score(y, prediction),
    "RMSE": np.sqrt(mean_squared_error(y, prediction)),
    "MAE": mean_absolute_error(y, prediction),
}])

summary.to_csv(
    ROOT / "03_results" / "grouped_validation_results.csv",
    index=False,
)
print(summary.to_string(index=False))
