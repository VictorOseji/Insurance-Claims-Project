# ============================================================================
# FILE: src/data/data_loader.py
# ============================================================================
"""Data loading and preprocessing utilities"""
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


class DataLoader:
    """Handle data loading and preprocessing"""
    
    def __init__(self, config: dict):
        self.config = config
        self.data_config = config['data']
    
    def create_sample_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Create sample regression dataset"""
        logger.info("Creating sample regression dataset...")
        
        X, y = make_regression(
            n_samples=self.data_config['n_samples'],
            n_features=self.data_config['n_features'],
            n_informative=self.data_config['n_informative'],
            noise=self.data_config['noise'],
            random_state=self.data_config['random_state']
        )
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.data_config['test_size'],
            random_state=self.data_config['random_state']
        )
        
        logger.info(f"Dataset created: {X_train.shape[0]} train, {X_test.shape[0]} test samples")
        return X_train, X_test, y_train, y_test
