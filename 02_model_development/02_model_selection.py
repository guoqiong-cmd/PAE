"""Five-fold training-set assessment followed by one common test evaluation."""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score

from model_config import MODEL_ORDER, RESULT_DIR, build_pipeline, load_records, records_to_xy


def metrics(observed, predicted):
    prediction = np.asarray(predicted).ravel()
    return {
        "test_R2": r2_score(observed, prediction),
        "test_RMSE": np.sqrt(mean_squared_error(observed, prediction)),
        "test_MAE": mean_absolute_error(observed, prediction),
    }


X_train, y_train = records_to_xy(load_records("train_set.npy"))
X_test, y_test = records_to_xy(load_records("test_set.npy"))
cv = KFold(n_splits=5, shuffle=True, random_state=42)
rows = []

for name in MODEL_ORDER:
    model = build_pipeline(name)
    score = cross_val_score(model, X_train, y_train, cv=cv, scoring="r2", n_jobs=-1)
    model.fit(X_train, y_train)
    rows.append({
        "model": name,
        "validation_5fold_R2_mean": score.mean(),
        "validation_5fold_R2_sd": score.std(),
        **metrics(y_test, model.predict(X_test)),
    })

result = pd.DataFrame(rows)
result.to_csv(RESULT_DIR / "model_performance.csv", index=False)
print(result.to_string(index=False))
