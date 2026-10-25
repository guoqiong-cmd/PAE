"""Refit XGBoost on all 701 observations and aggregate SHAP by original feature."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import shap


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "02_model_development"))
from model_config import FEATURES, build_pipeline, load_all_records, records_to_xy


records = load_all_records()
X, y = records_to_xy(records)
model = build_pipeline("XGBoost")
model.fit(X, y)

encoded = model.named_steps["preprocess"].transform(X)
encoded_names = model.named_steps["preprocess"].get_feature_names_out()
values = shap.TreeExplainer(model.named_steps["model"]).shap_values(encoded)
if isinstance(values, list):
    values = values[0]
encoded_shap = pd.DataFrame(values, columns=encoded_names)


def original_name(name):
    if name.startswith("cat__Particle_type_"):
        return "Particle_type"
    if name.startswith("cat__Light_"):
        return "Light"
    return name.replace("num__", "")


mapping = {name: original_name(name) for name in encoded_names}
aggregated = pd.DataFrame({
    feature: encoded_shap[[name for name in encoded_names if mapping[name] == feature]].sum(axis=1)
    for feature in FEATURES
})
aggregated["ln_RC"] = y
aggregated["PAE_label"] = records["PAE"]
aggregated["Particle_type_label"] = records["Particle_type"]
aggregated["Light_label"] = records["Light"]
aggregated.to_csv(ROOT / "04_evaluation" / "shap_values_by_feature.csv", index=False)

importance = aggregated[FEATURES].abs().mean().sort_values(ascending=False).rename("mean_abs_SHAP")
importance.to_csv(ROOT / "04_evaluation" / "shap_global_importance.csv")
print(importance)
