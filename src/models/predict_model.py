import logging
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

logger = logging.getLogger(__name__)

def predict(model: BaseEstimator, X: pd.DataFrame) -> np.ndarray:
    """
    Make predictions using a trained model.
    
    Args:
        model: Trained classification model
        X: Feature dataframe for prediction
        
    Returns:
        Array of predictions
    """
    logger.info(f"Making predictions on data with shape {X.shape}")
    
    try:
        predictions = model.predict(X)
        logger.info(f"Generated {len(predictions)} predictions")
        return predictions
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise