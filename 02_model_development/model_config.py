"""Shared data loader, pipelines, search spaces, and final parameters."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "01_data"
RESULT_DIR = ROOT / "03_results"

SEED = 42
FEATURES = [
    "Particle_type", "MP_density", "Light", "Particle_size", "Exposure time",
    "PH", "Temperature", "Salinities", "logKow",
]
CATEGORICAL = ["Particle_type", "Light"]
CONTINUOUS = [name for name in FEATURES if name not in CATEGORICAL]
MODEL_ORDER = ["PLSR", "SVR", "MLP", "RF", "XGBoost"]

FINAL_PARAMETERS = {
    "PLSR": {"n_components": 12, "tol": 1e-5, "max_iter": 1000},
    "SVR": {"kernel": "rbf", "C": 30, "gamma": 0.03, "epsilon": 0.1},
    "MLP": {
        "hidden_layer_sizes": (128,), "activation": "relu", "alpha": 0.01,
        "learning_rate_init": 0.003, "batch_size": 16,
    },
    "RF": {
        "n_estimators": 500, "max_depth": None, "min_samples_split": 5,
        "min_samples_leaf": 1, "max_features": 0.5,
    },
    "XGBoost": {
        "n_estimators": 300, "max_depth": 6, "learning_rate": 0.03,
        "subsample": 0.7, "colsample_bytree": 0.7, "min_child_weight": 3,
        "reg_alpha": 0, "reg_lambda": 1,
    },
}

SEARCH_SPACES = {
    "PLSR": {
        "n_components": [1, 2, 3, 4, 5, 6, 8, 10, 12],
        "tol": [1e-6, 1e-5, 1e-4], "max_iter": [500, 1000],
    },
    "SVR": {
        "kernel": ["rbf"], "C": [0.1, 0.3, 1, 3, 10, 30, 100],
        "gamma": ["scale", 0.001, 0.003, 0.01, 0.03, 0.1],
        "epsilon": [0.01, 0.03, 0.05, 0.1, 0.2],
    },
    "MLP": {
        "hidden_layer_sizes": [(32,), (64,), (128,), (64, 32), (128, 64), (128, 64, 32)],
        "activation": ["relu", "tanh"], "alpha": [1e-5, 1e-4, 1e-3, 1e-2],
        "learning_rate_init": [1e-4, 3e-4, 1e-3, 3e-3], "batch_size": [16, 32, 64],
    },
    "RF": {
        "n_estimators": [300, 500, 800], "max_depth": [None, 5, 10],
        "min_samples_split": [2, 5, 10], "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", 0.5, 1.0],
    },
    "XGBoost": {
        "n_estimators": [200, 300, 500, 800, 1000], "max_depth": [2, 3, 4, 5, 6],
        "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
        "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
        "min_child_weight": [1, 3, 5, 7], "reg_alpha": [0, 0.001, 0.01, 0.1],
        "reg_lambda": [0.5, 1, 2, 5],
    },
}

SEARCH_ITERATIONS = {"PLSR": 30, "SVR": 60, "MLP": 60, "XGBoost": 80}


def load_records(filename):
    return np.load(DATA_DIR / filename, allow_pickle=False)


def load_all_records():
    """Reconstruct the complete processed dataset from the fixed split."""
    records = np.concatenate([
        load_records("train_set.npy"),
        load_records("test_set.npy"),
    ])
    return np.sort(records, order="row_ID")


def records_to_xy(records):
    X = pd.DataFrame({name: records[name] for name in FEATURES})
    y = records["ln_RC"].astype(float)
    return X, y


def build_pipeline(model_name, use_final_parameters=True, grouped=False):
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    numeric_step = StandardScaler() if model_name in {"PLSR", "SVR", "MLP"} else "passthrough"
    preprocess = ColumnTransformer(
        [("cat", encoder, CATEGORICAL), ("num", numeric_step, CONTINUOUS)],
        remainder="drop",
    )
    fixed = {
        "PLSR": {"scale": False},
        "SVR": {},
        "MLP": {
            "random_state": SEED, "max_iter": 3000, "early_stopping": True,
            "validation_fraction": 0.15, "n_iter_no_change": 30,
        },
        "RF": {"random_state": SEED, "n_jobs": -1},
        "XGBoost": {
            "objective": "reg:squarederror", "eval_metric": "rmse",
            "random_state": SEED, "n_jobs": 1 if grouped else -1, "tree_method": "hist",
        },
    }[model_name]
    if use_final_parameters:
        fixed.update(FINAL_PARAMETERS[model_name])
    estimator = {
        "PLSR": PLSRegression, "SVR": SVR, "MLP": MLPRegressor,
        "RF": RandomForestRegressor, "XGBoost": XGBRegressor,
    }[model_name](**fixed)
    return Pipeline([("preprocess", preprocess), ("model", estimator)])


def prefixed_search_space(model_name):
    return {f"model__{key}": value for key, value in SEARCH_SPACES[model_name].items()}
