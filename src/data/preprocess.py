import logging
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional

logger = logging.getLogger(__name__)

def preprocess_data(
    df: pd.DataFrame, 
    drop_columns: List[str] = None,
    imputation_strategies: Optional[Dict[str, Dict[str, str]]] = None
) -> pd.DataFrame:
    """
    Preprocess the dataframe by handling missing values and dropping unnecessary columns.
    
    Args:
        df: Input dataframe
        drop_columns: List of columns to drop
        imputation_strategies: Dictionary of imputation strategies for specific columns
        
    Returns:
        Preprocessed dataframe
    """
    logger.info("Missing values before preprocessing:")
    logger.info(df.isnull().sum())
    
    logger.info("Performing basic data cleaning")
    
    # Create a copy to avoid modification warnings
    df_processed = df.copy()
    
    # Handle missing values according to specified strategies
    if imputation_strategies:
        for column, strategy_info in imputation_strategies.items():
            if column in df_processed.columns and df_processed[column].isnull().any():
                strategy = strategy_info.get("strategy", "median")
                
                if column == 'Age' and strategy == 'median':
                    logger.info("Handling missing Age values")
                    median_age = df_processed['Age'].median()
                    df_processed['Age'] = df_processed['Age'].fillna(median_age)
                
                elif column == 'Embarked' and strategy == 'most_frequent':
                    logger.info("Handling missing Embarked values")
                    most_common = df_processed['Embarked'].mode()[0]
                    df_processed['Embarked'] = df_processed['Embarked'].fillna(most_common)
                
                elif column == 'Fare' and strategy == 'median':
                    logger.info("Handling missing Fare values")
                    median_fare = df_processed['Fare'].median()
                    df_processed['Fare'] = df_processed['Fare'].fillna(median_fare)
    
    # Drop unnecessary columns
    if drop_columns:
        logger.info(f"Dropping columns: {drop_columns}")
        df_processed = df_processed.drop(columns=[col for col in drop_columns if col in df_processed.columns])
    
    logger.info(f"Shape after preprocessing: {df_processed.shape}")
    return df_processed

def prepare_train_data(
    df: pd.DataFrame, 
    target_column: str = 'Survived',
    drop_columns: List[str] = None,
    imputation_strategies: Optional[Dict[str, Dict[str, str]]] = None
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare training data by preprocessing and splitting into features and target.
    
    Args:
        df: Input dataframe
        target_column: Name of the target column
        drop_columns: List of columns to drop
        imputation_strategies: Dictionary of imputation strategies for specific columns
        
    Returns:
        Tuple of (X, y) where X is features dataframe and y is target series
    """
    df_processed = preprocess_data(df, drop_columns, imputation_strategies)
    
    # Split into features and target
    if target_column in df_processed.columns:
        X = df_processed.drop(columns=[target_column])
        y = df_processed[target_column]
        logger.info(f"Features shape: {X.shape}")
        logger.info(f"Target shape: {y.shape}")
        return X, y
    else:
        logger.warning(f"Target column '{target_column}' not found in dataframe")
        return df_processed, None

def prepare_test_data(
    df: pd.DataFrame,
    drop_columns: List[str] = None,
    imputation_strategies: Optional[Dict[str, Dict[str, str]]] = None
) -> pd.DataFrame:
    """
    Prepare test data by preprocessing.
    
    Args:
        df: Input dataframe
        drop_columns: List of columns to drop
        imputation_strategies: Dictionary of imputation strategies for specific columns
        
    Returns:
        Preprocessed dataframe
    """
    return preprocess_data(df, drop_columns, imputation_strategies)