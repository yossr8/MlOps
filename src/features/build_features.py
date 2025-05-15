import logging
from typing import Dict, List, Tuple, Any, Optional

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)

def create_feature_pipeline(
    numeric_features: List[str],
    categorical_features: List[str],
    numeric_transformer_params: Optional[Dict[str, Any]] = None,
    categorical_transformer_params: Optional[Dict[str, Any]] = None
) -> ColumnTransformer:
    """
    Create a feature engineering pipeline for processing numeric and categorical features.
    
    Args:
        numeric_features: List of numeric feature names
        categorical_features: List of categorical feature names
        numeric_transformer_params: Parameters for numeric transformer
        categorical_transformer_params: Parameters for categorical transformer
        
    Returns:
        A ColumnTransformer pipeline for feature processing
    """
    logger.info("Setting up feature engineering pipeline")
    
    # Validate inputs
    if not isinstance(numeric_features, list) or not all(isinstance(x, str) for x in numeric_features):
        raise ValueError("numeric_features must be a list of strings")
    
    if not isinstance(categorical_features, list) or not all(isinstance(x, str) for x in categorical_features):
        raise ValueError("categorical_features must be a list of strings")
    
    logger.info(f"Numeric features: {numeric_features}")
    logger.info(f"Categorical features: {categorical_features}")
    
    # Use default params if none provided
    if numeric_transformer_params is None:
        numeric_transformer_params = {"imputer": {"strategy": "median"}}
    
    if categorical_transformer_params is None:
        categorical_transformer_params = {
            "imputer": {"strategy": "most_frequent"},
            "encoder": {"handle_unknown": "ignore"}
        }
    
    # Create numeric transformer
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(**numeric_transformer_params["imputer"])),
        ('scaler', StandardScaler())
    ])
    
    # Create categorical transformer
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy=categorical_transformer_params["imputer"]["strategy"], fill_value='missing')),
        ('encoder', OneHotEncoder(**categorical_transformer_params["encoder"]))
    ])
    
    # Create preprocessor with properly specified columns
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        verbose_feature_names_out=False,  # Simplify output feature names
        remainder='drop'  # Drop any columns not explicitly included
    )
    
    logger.info("Feature engineering pipeline created")
    return preprocessor