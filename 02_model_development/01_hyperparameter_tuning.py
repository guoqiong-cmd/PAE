"""Three-fold hyperparameter optimization using the training set only."""

import argparse
import json

import pandas as pd
from sklearn.model_selection import GridSearchCV, KFold, RandomizedSearchCV

from model_config import (
    MODEL_ORDER, RESULT_DIR, SEARCH_ITERATIONS, build_pipeline,
    load_records, prefixed_search_space, records_to_xy,
)


parser = argparse.ArgumentParser()
parser.add_argument("--model", choices=MODEL_ORDER + ["all"], default="all")
args = parser.parse_args()

X_train, y_train = records_to_xy(load_records("train_set.npy"))
cv = KFold(n_splits=3, shuffle=True, random_state=42)

for name in MODEL_ORDER if args.model == "all" else [args.model]:
    pipeline = build_pipeline(name, use_final_parameters=False)
    if name == "RF":
        search = GridSearchCV(pipeline, prefixed_search_space(name), scoring="r2", cv=cv, n_jobs=-1)
    else:
        search = RandomizedSearchCV(
            pipeline, prefixed_search_space(name), n_iter=SEARCH_ITERATIONS[name],
            scoring="r2", cv=cv, n_jobs=-1, random_state=42, refit=True,
        )
    search.fit(X_train, y_train)
    pd.DataFrame(search.cv_results_).to_csv(RESULT_DIR / f"{name}_tuning_results.csv", index=False)
    with open(RESULT_DIR / f"{name}_best_parameters.json", "w", encoding="utf-8") as file:
        json.dump(search.best_params_, file, indent=2)
    print(name, search.best_score_, search.best_params_)

