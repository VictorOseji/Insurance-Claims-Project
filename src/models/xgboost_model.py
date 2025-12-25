# ============================================================================
# FILE: src/models/xgboost_model.py
# ============================================================================
"""XGBoost model implementation"""
from xgboost import XGBRegressor
from .base_model import BaseModel
import numpy as np
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class XGBoostModel(BaseModel):
    """XGBoost model wrapper"""
    
    def __init__(self, config: dict):
        super().__init__(config)
        self.model_name = "XGBoost"
        self.model = XGBRegressor(
            random_state=config['data']['random_state'],
            objective='reg:squarederror'
        )
        self.best_params = None
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train XGBoost model"""
        logger.info(f"Training {self.model_name}...")
        self.model.fit(X_train, y_train)
        return self.model
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        return self.model.predict(X)
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters"""
        if self.best_params:
            return self.best_params
        return self.model.get_params()
    
    def set_params(self, params: Dict[str, Any]):
        """Set model parameters"""
        self.model.set_params(**params)
        self.best_params = params

