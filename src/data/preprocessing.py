# ============================================================================
# FILE: src/data/preprocessing.py
# ============================================================================
"""Data preprocessing pipeline"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import logging
from typing import Tuple, List
import joblib
from pathlib import Path

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Preprocess data for model training"""
    
    def __init__(self, config: dict, feature_config: dict):
        self.config = config
        self.feature_config = feature_config
        self.preprocessor = None
        self.feature_names = None
    
    def create_preprocessing_pipeline(
        self,
        numeric_features: List[str],
        categorical_features: List[str]
    ) -> ColumnTransformer:
        """Create sklearn preprocessing pipeline"""
        logger.info("Creating preprocessing pipeline...")
        
        # Numeric transformer
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        # Categorical transformer
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        # Combine transformers
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ],
            remainder='drop'
        )
        
        logger.info(f"Pipeline created with {len(numeric_features)} numeric and "
                   f"{len(categorical_features)} categorical features")
        
        return preprocessor
    
    def prepare_features(
        self,
        df: pd.DataFrame,
        target_col: str
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Separate features and target"""
        logger.info(f"Preparing features with target: {target_col}")
        
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataframe")
        
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        logger.info(f"Features shape: {X.shape}, Target shape: {y.shape}")
        
        return X, y
    
    def split_data(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Split data into train and test sets"""
        logger.info("Splitting data into train and test sets...")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.config['data']['test_size'],
            random_state=self.config['data']['random_state']
        )
        
        logger.info(f"Train set: {X_train.shape[0]} samples")
        logger.info(f"Test set: {X_test.shape[0]} samples")
        
        return X_train, X_test, y_train, y_test
    
    def fit_transform(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        numeric_features: List[str],
        categorical_features: List[str]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Fit preprocessor on train and transform both train and test"""
        logger.info("Fitting and transforming data...")
        
        # Create pipeline
        self.preprocessor = self.create_preprocessing_pipeline(
            numeric_features,
            categorical_features
        )
        
        # Fit and transform
        X_train_processed = self.preprocessor.fit_transform(X_train)
        X_test_processed = self.preprocessor.transform(X_test)
        
        # Get feature names after transformation
        self.feature_names = self._get_feature_names_out(
            numeric_features,
            categorical_features,
            X_train
        )
        
        logger.info(f"Processed training data shape: {X_train_processed.shape}")
        logger.info(f"Processed test data shape: {X_test_processed.shape}")
        
        return X_train_processed, X_test_processed
    
    def _get_feature_names_out(
        self,
        numeric_features: List[str],
        categorical_features: List[str],
        X: pd.DataFrame
    ) -> List[str]:
        """Get feature names after one-hot encoding"""
        feature_names = []
        
        # Add numeric feature names
        feature_names.extend(numeric_features)
        
        # Add one-hot encoded feature names
        for cat_feat in categorical_features:
            if cat_feat in X.columns:
                unique_values = X[cat_feat].unique()
                for val in unique_values:
                    feature_names.append(f"{cat_feat}_{val}")
        
        return feature_names
    
    def save_preprocessor(self, filepath: str = "models/preprocessor.joblib"):
        """Save fitted preprocessor"""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.preprocessor, filepath)
        logger.info(f"Preprocessor saved to: {filepath}")
    
    def load_preprocessor(self, filepath: str = "models/preprocessor.joblib"):
        """Load fitted preprocessor"""
        self.preprocessor = joblib.load(filepath)
        logger.info(f"Preprocessor loaded from: {filepath}")
