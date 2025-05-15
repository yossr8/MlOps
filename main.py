import logging
import os
from typing import Dict, Any

import hydra
import pandas as pd
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split

from src.data.make_dataset import load_data
from src.data.preprocess import prepare_train_data, prepare_test_data
from src.features.build_features import create_feature_pipeline
from src.models.train_model import train_model
from src.models.predict_model import predict

logger = logging.getLogger(__name__)

@hydra.main(config_path="conf", config_name="config", version_base="1.1")
def main(cfg: DictConfig) -> None:
    """
    Main entry point for the Titanic classification pipeline.
    
    Args:
        cfg: Hydra configuration
    """
    logger.info("Configuration:")
    logger.info(cfg)
    
    logger.info("Starting Titanic classification pipeline")
    
    try:
        # Step 1: Load data
        logger.info("Loading data")
        train_df, test_df = load_data(
            train_path=cfg.paths.train_data,
            test_path=cfg.paths.test_data
        )
        
        # Step 2: Preprocess data
        logger.info("Preprocessing data")
        imputation_strategies = {
            'Age': {'strategy': cfg.data.preprocessing.imputation.age.strategy},
            'Embarked': {'strategy': cfg.data.preprocessing.imputation.embarked.strategy},
            'Fare': {'strategy': cfg.data.preprocessing.imputation.fare.strategy}
        }
        
        X_train, y_train = prepare_train_data(
            train_df,
            drop_columns=cfg.data.preprocessing.drop_columns,
            imputation_strategies=imputation_strategies
        )
        
        X_test = prepare_test_data(
            test_df,
            drop_columns=cfg.data.preprocessing.drop_columns,
            imputation_strategies=imputation_strategies
        )
        
        # Step 3: Build features
        logger.info("Building features")
        feature_pipeline = create_feature_pipeline(
        numeric_features=list(cfg.features.numeric_features),
        categorical_features=list(cfg.features.categorical_features),
        numeric_transformer_params=cfg.features.transformers.numeric,
        categorical_transformer_params=cfg.features.transformers.categorical
        )
        # Step 4: Train model
        logger.info("Training model")
        visualization_dir = cfg.paths.output_dir + "/visualizations" if cfg.visualization.enabled else None
        
        model = train_model(
            X_train=X_train,
            y_train=y_train,
            feature_pipeline=feature_pipeline,
            model_type=cfg.model.type,
            model_params=dict(cfg.model.params),
            output_dir=cfg.paths.output_dir,
            visualize=cfg.visualization.enabled,
            visualization_dir=visualization_dir
        )
        
        # Step 5: Make predictions
        logger.info("Making predictions on test data")
        test_predictions = predict(model, X_test)
        
        # Save predictions
        submission = pd.DataFrame({
            'PassengerId': test_df['PassengerId'],
            'Survived': test_predictions
        })
        
        os.makedirs(cfg.paths.output_dir, exist_ok=True)
        submission_path = f"{cfg.paths.output_dir}/submission.csv"
        submission.to_csv(submission_path, index=False)
        logger.info(f"Predictions saved to {submission_path}")
        
        logger.info("Pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"Error in pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    main()