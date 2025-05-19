import os
import sys
import json
import mlflow
import logging
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def load_params(params_path="params.yaml"):
    with open(params_path, "r") as f:
        return yaml.safe_load(f)

def main():
    # Load config
    config = load_params()

    # Setup MLflow
    mlflow_cfg = config.get("mlflow", {})
    mlflow.set_tracking_uri(mlflow_cfg.get("tracking_uri", "http://localhost:5000"))
    mlflow.set_experiment(mlflow_cfg.get("experiment_name", "model_training"))

    # Load data
    train_path = config["paths"]["train_data"]
    test_path = config["paths"]["test_data"]
    # Use the correct location for target_column
    target_col = config.get("target_column", "Survived")
    if "data" in config and "target_column" in config["data"]:
        target_col = config["data"]["target_column"]

    log.info(f"Loading data from {train_path} and {test_path}")
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    # Select features from params.yaml
    numeric_features = config["features"]["numeric_features"]
    categorical_features = config["features"]["categorical_features"]
    all_features = numeric_features + categorical_features

    X_train = train_data[all_features]
    y_train = train_data[target_col]
    if target_col in test_data.columns:
        X_test = test_data[all_features]
        y_test = test_data[target_col]
    else:
        X_test = test_data[all_features]
        y_test = None

    # Encode categorical features
    X_train = pd.get_dummies(X_train, columns=categorical_features)
    X_test = pd.get_dummies(X_test, columns=categorical_features)

    # Align columns
    X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

    log.info(f"Training data shape: {X_train.shape}")
    log.info(f"Test data shape: {X_test.shape}")

    # Model selection
    model_type = config["model"]["type"]
    log.info(f"Creating {model_type} model")

    if model_type == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        base_model = RandomForestClassifier(**config["model"]["params"])
        grid_search_params = config.get("grid_search", {}).get("random_forest", {})
    elif model_type == "xgboost":
        import xgboost as xgb
        base_model = xgb.XGBClassifier(**config["model"]["params"])
        grid_search_params = config.get("grid_search", {}).get("xgboost", {})
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # Grid search
    log.info("Setting up grid search")
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=grid_search_params,
        cv=config.get("train", {}).get("cv_folds", 5),
        scoring=config.get("train", {}).get("scoring_metric", "f1_weighted"),
        n_jobs=config.get("train", {}).get("n_jobs", -1),
        verbose=2
    )

    # MLflow run
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        log.info(f"MLflow run ID: {run_id}")

        # Log initial parameters
        mlflow.log_params(config["model"]["params"])

        # Train model
        log.info("Starting grid search training")
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        log.info(f"Best parameters: {best_params}")

        # Only log best_params that are not already logged or have changed
        # Only log best_params that are not already logged or have changed, and skip if both are None
        for k, v in best_params.items():
            orig_val = config["model"]["params"].get(k)
            # Skip if both are None or 'None'
            if (orig_val is None or str(orig_val).lower() == "none") and (v is None or str(v).lower() == "none"):
                continue
            if str(orig_val) != str(v):
                try:
                    mlflow.log_param(k, v)
                except mlflow.exceptions.MlflowException as e:
                    log.warning(f"Could not log param {k}: {e}")

        # Evaluate
        log.info("Evaluating model")
        train_preds = best_model.predict(X_train)
        test_preds = best_model.predict(X_test)

        train_metrics = {
            "train_accuracy": accuracy_score(y_train, train_preds),
            "train_precision": precision_score(y_train, train_preds, average='weighted'),
            "train_recall": recall_score(y_train, train_preds, average='weighted'),
            "train_f1": f1_score(y_train, train_preds, average='weighted')
        }
        test_metrics = {
            "test_accuracy": accuracy_score(y_test, test_preds) if y_test is not None else None,
            "test_precision": precision_score(y_test, test_preds, average='weighted') if y_test is not None else None,
            "test_recall": recall_score(y_test, test_preds, average='weighted') if y_test is not None else None,
            "test_f1": f1_score(y_test, test_preds, average='weighted') if y_test is not None else None
        }
        log.info(f"Train metrics: {train_metrics}")
        log.info(f"Test metrics: {test_metrics}")

        for name, value in {**train_metrics, **test_metrics}.items():
            if value is not None:
                mlflow.log_metric(name, value)

        # Log model
        log.info("Logging model to MLflow")
        if model_type == "random_forest":
            mlflow.sklearn.log_model(best_model, "model")
        elif model_type == "xgboost":
            mlflow.xgboost.log_model(best_model, "model")

        # Save model
        model_dir = config["paths"]["output_dir"]
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f"{model_type}_model.joblib")
        joblib.dump(best_model, model_path)
        log.info(f"Model saved to {model_path}")

        # Save metrics for DVC
        metrics_dir = "metrics"
        os.makedirs(metrics_dir, exist_ok=True)
        metrics_path = os.path.join(metrics_dir, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump({
                **test_metrics,
                "best_run_id": run_id,
                "model_path": model_path
            }, f)
        log.info(f"Metrics saved to {metrics_path}")

        # Register model if configured
        if mlflow_cfg.get("register_model", False):
            model_name = f"{model_type}_model"
            log.info(f"Registering model as {model_name}")
            if model_type == "random_forest":
                mlflow.sklearn.log_model(
                    best_model,
                    "registered_model",
                    registered_model_name=model_name
                )
            elif model_type == "xgboost":
                mlflow.xgboost.log_model(
                    best_model,
                    "registered_model",
                    registered_model_name=model_name
                )

    log.info("Training completed successfully")
    return test_metrics

if __name__ == "__main__":
    main()