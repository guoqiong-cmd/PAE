# MetaPAEs model analysis package

This package contains the code, trained model, model outputs, and documentation associated with the reported model-development and evaluation workflow. PAE identity is retained only as metadata and is not used as a model predictor.

## Contents

- `02_model_development`: preprocessing, three-fold hyperparameter optimization, five-fold training-set assessment, and final XGBoost fitting.
- `03_results`: final XGBoost model, final parameters, five-model performance, and the experimental-condition-level validation result.
- `04_evaluation`: held-out test evaluation, nested grouped validation, and SHAP analysis.
- `05_documentation`: data dictionary and package requirements.

## Model-development sequence

The dataset consisted of 701 processed observations and was divided once into an 80% training set (n = 560) and a 20% held-out test set (n = 141) using `random_state = 42`. Preprocessing and three-fold hyperparameter optimization were conducted within the training set for each candidate algorithm. The selected configuration of each algorithm was assessed by five-fold cross-validation within the training set, refitted using all training observations, and evaluated once on the common held-out test set. XGBoost was retained as the final model.

The 122-group nested validation was conducted separately and was not used for model selection. Records sharing the same experimental condition were kept in the same outer fold.

## Basic use

Install the required Python packages using:

```bash
pip install -r "05_documentation/requirements.txt"
```

The provided scripts correspond to the model-development and evaluation procedures described in the study.
