# ============================================================================
# FILE: src/vetiver_utils/model_wrapper.py
# ============================================================================
"""Vetiver model wrapping utilities"""
from vetiver import VetiverModel
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class VetiverModelWrapper:
    """Wrap models in Vetiver format"""
    
    @staticmethod
    def create_vetiver_model(model, model_name: str, X_train: np.ndarray) -> VetiverModel:
        """Create a Vetiver model with metadata"""
        logger.info(f"Creating Vetiver model: {model_name}")
        
        # Create vetiver model with prototype data
        v_model = VetiverModel(
            model=model,
            model_name=model_name,
            prototype_data=pd.DataFrame(X_train[:5])
        )
        
        return v_model

