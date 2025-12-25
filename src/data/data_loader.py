# ============================================================================
# FILE: src/data/data_loader.py
# ============================================================================
"""Data loading utilities for insurance claims"""
import pandas as pd
import logging
from pathlib import Path
from typing import Tuple

logger = logging.getLogger(__name__)


class InsuranceDataLoader:
    """Load and merge insurance claims data"""
    
    def __init__(self, config: dict):
        self.config = config
        self.data_config = config['data']
    
    def load_raw_data(self) -> pd.DataFrame:
        """Load and merge claims and policy data"""
        logger.info("Loading raw data files...")
        
        # Load datasets
        claims_path = self.data_config['claims_file']
        policy_path = self.data_config['policy_file']
        
        logger.info(f"Loading claims data from: {claims_path}")
        claims = pd.read_csv(claims_path)
        
        logger.info(f"Loading policy data from: {policy_path}")
        policy = pd.read_csv(policy_path)
        
        # Merge datasets
        logger.info("Merging claims and policy data...")
        claims_data = pd.merge(
            claims, 
            policy, 
            on=['Policy_ID', 'Customer_ID']
        )
        
        logger.info(f"Merged data shape: {claims_data.shape}")
        logger.info(f"Columns: {list(claims_data.columns)}")
        
        return claims_data
    
    def save_processed_data(self, df: pd.DataFrame, filename: str = "processed_claims.csv"):
        """Save processed data"""
        output_path = Path(self.data_config['processed_path']) / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        df.to_csv(output_path, index=False)
        logger.info(f"Processed data saved to: {output_path}")
    
    def load_processed_data(self, filename: str = "processed_claims.csv") -> pd.DataFrame:
        """Load previously processed data"""
        input_path = Path(self.data_config['processed_path']) / filename
        
        if not input_path.exists():
            raise FileNotFoundError(f"Processed data not found at: {input_path}")
        
        logger.info(f"Loading processed data from: {input_path}")
        return pd.read_csv(input_path)
