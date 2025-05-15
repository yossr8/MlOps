import logging
import pandas as pd
from typing import Tuple

logger = logging.getLogger(__name__)

def load_data(train_path: str, test_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load training and test data from CSV files.
    
    Args:
        train_path: Path to the training data CSV
        test_path: Path to the test data CSV
        
    Returns:
        Tuple of (train_df, test_df)
    """
    # Load training data
    logger.info(f"Loading training data from {train_path}")
    train_df = pd.read_csv(train_path)
    logger.info(f"Training data shape: {train_df.shape}")
    
    # Load test data
    logger.info(f"Loading test data from {test_path}")
    test_df = pd.read_csv(test_path)
    logger.info(f"Test data shape: {test_df.shape}")
    
    return train_df, test_df