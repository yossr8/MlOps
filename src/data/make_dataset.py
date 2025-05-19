import logging
import pandas as pd
import os
from typing import Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data(train_path: str, test_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load training and test data from CSV files.
    """
    logger.info(f"Loading training data from {train_path}")
    train_df = pd.read_csv(train_path)
    logger.info(f"Training data shape: {train_df.shape}")

    logger.info(f"Loading test data from {test_path}")
    test_df = pd.read_csv(test_path)
    logger.info(f"Test data shape: {test_df.shape}")

    return train_df, test_df

def preprocess_data(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Basic preprocessing (placeholder). Modify as needed.
    """
    # Example: Fill missing values with median for numeric columns
    for col in train_df.select_dtypes(include='number').columns:
        median = train_df[col].median()
        train_df[col] = train_df[col].fillna(median)
        if col in test_df.columns:
            test_df[col] = test_df[col].fillna(median)
    # You can add more preprocessing steps here
    return train_df, test_df

def save_data(train_df: pd.DataFrame, test_df: pd.DataFrame, train_out: str, test_out: str):
    """
    Save processed train and test data to CSV.
    """
    os.makedirs(os.path.dirname(train_out), exist_ok=True)
    train_df.to_csv(train_out, index=False)
    test_df.to_csv(test_out, index=False)
    logger.info(f"Saved processed train data to {train_out}")
    logger.info(f"Saved processed test data to {test_out}")

def main():
    # Define paths (relative to project root)
    train_path = "data/raw/train.csv"
    test_path = "data/raw/test.csv"
    train_out = "data/processed/train.csv"
    test_out = "data/processed/test.csv"

    train_df, test_df = load_data(train_path, test_path)
    train_df, test_df = preprocess_data(train_df, test_df)
    save_data(train_df, test_df, train_out, test_out)

if __name__ == "__main__":
    main()