import yaml
import pandas as pd
from src.data.make_dataset import load_data
from src.data.preprocess import prepare_train_data, prepare_test_data
from src.features.build_features import create_feature_pipeline
from src.models.train_model import train_model

def main():
    # Load parameters
    with open("params.yaml") as f:
        params = yaml.safe_load(f)

    # Load data
    train_df, test_df = load_data(
        train_path=params["paths"]["train_data"],
        test_path=params["paths"]["test_data"]
    )

    # Preprocess
    X_train, y_train = prepare_train_data(
        train_df,
        drop_columns=params["data"]["preprocessing"]["drop_columns"],
        imputation_strategies=params["data"]["preprocessing"]["imputation"]
    )

    # Build features
    feature_pipeline = create_feature_pipeline(
        numeric_features=params["features"]["numeric_features"],
        categorical_features=params["features"]["categorical_features"],
        numeric_transformer_params=params["features"]["transformers"]["numeric"],
        categorical_transformer_params=params["features"]["transformers"]["categorical"]
    )

    # Train model
    model = train_model(
        X_train=X_train,
        y_train=y_train,
        feature_pipeline=feature_pipeline,
        model_type=params["model"]["type"],
        model_params=params["model"]["params"],
        output_dir=params["paths"]["output_dir"],
        visualize=False,
        visualization_dir=None
    )

if __name__ == "__main__":
    main()