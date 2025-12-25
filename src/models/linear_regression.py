# ============================================================================
# FILE: src/models/linear_regression.py
# ============================================================================
"""Linear Regression model implementation"""
from sklearn.linear_model import LinearRegression
from .base_model import BaseModel
import numpy as np
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class LinearRegressionModel(BaseModel):
    """Linear Regression model wrapper"""
    
    def __init__(self, config: dict):
        super().__init__(config)
        self.model_name = "Linear Regression"
        self.model = LinearRegression()
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train linear regression model"""
        logger.info(f"Training {self.model_name}...")
        self.model.fit(X_train, y_train)
        return self.model
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        return self.model.predict(X)
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters"""
        return {}
