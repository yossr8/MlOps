import logging
import pickle
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score
import os

logger = logging.getLogger(__name__)

def get_classifier(model_type: str, model_params: Dict[str, Any]) -> BaseEstimator:
    """
    Get a classifier based on the model type and parameters.
    
    Args:
        model_type: Type of model to use (e.g., 'random_forest')
        model_params: Parameters for the model
        
    Returns:
        A scikit-learn classifier
    """
    logger.info(f"Using {model_type} classifier")
    
    if model_type.lower() == "random_forest":
        return RandomForestClassifier(**model_params)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    feature_pipeline: ColumnTransformer,
    model_type: str = "random_forest",
    model_params: Optional[Dict[str, Any]] = None,
    output_dir: str = "models",
    visualize: bool = False,
    visualization_dir: Optional[str] = None
) -> BaseEstimator:
    """
    Train a classification model using the provided data and feature pipeline.
    
    Args:
        X_train: Training features
        y_train: Training targets
        feature_pipeline: Feature engineering pipeline
        model_type: Type of model to use
        model_params: Parameters for the model
        output_dir: Directory to save model artifacts
        visualize: Whether to generate visualizations
        visualization_dir: Directory to save visualizations
        
    Returns:
        Trained classifier
    """
    logger.info("Training classifier")
    
    if model_params is None:
        model_params = {}
    
    # Get classifier
    classifier = get_classifier(model_type, model_params)
    
    # Create full pipeline with feature engineering and model
    model = Pipeline([
        ('features', feature_pipeline),
        ('classifier', classifier)
    ])
    
    logger.info("Fitting model to training data")
    try:
        # Fit the model
        model.fit(X_train, y_train)
        
        # Evaluate on training data
        y_pred = model.predict(X_train)
        accuracy = accuracy_score(y_train, y_pred)
        logger.info(f"Training accuracy: {accuracy:.4f}")
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)
        logger.info(f"Cross-validation scores: {cv_scores}")
        logger.info(f"Mean CV accuracy: {cv_scores.mean():.4f}")
        
        # Feature importance if available
        if hasattr(classifier, 'feature_importances_'):
            # Get feature names after transformation
            feature_names = feature_pipeline.get_feature_names_out()
            feature_importances = classifier.feature_importances_
            
            # Create DataFrame for feature importance
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': feature_importances
            }).sort_values('Importance', ascending=False)
            
            logger.info("Top 10 feature importances:")
            logger.info(importance_df.head(10))
            
            if visualize and visualization_dir:
                os.makedirs(visualization_dir, exist_ok=True)
                plt.figure(figsize=(10, 6))
                sns.barplot(x='Importance', y='Feature', data=importance_df.head(15))
                plt.title('Feature Importance')
                plt.tight_layout()
                plt.savefig(f"{visualization_dir}/feature_importance.png")
                plt.close()
        
        # Save model
        os.makedirs(output_dir, exist_ok=True)
        with open(f"{output_dir}/model.pkl", "wb") as f:
            pickle.dump(model, f)
        logger.info(f"Model saved to {output_dir}/model.pkl")
        
        return model
        
    except Exception as e:
        logger.error(f"Error during model training: {str(e)}")
        raise