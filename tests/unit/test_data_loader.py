# ============================================================================
# FILE: tests/unit/test_data_loader.py
# ============================================================================
"""Unit tests for data loader"""
import pytest
from src.data.data_loader import DataLoader


def test_create_sample_data(sample_config):
    """Test sample data creation"""
    loader = DataLoader(sample_config)
    X_train, X_test, y_train, y_test = loader.create_sample_data()
    
    assert X_train.shape[0] > 0
    assert X_test.shape[0] > 0
    assert len(y_train) == X_train.shape[0]
    assert len(y_test) == X_test.shape[0]
