# ============================================================================
# FILE: tests/unit/test_models.py
# ============================================================================
"""Unit tests for model classes"""
import pytest
import numpy as np
from src.models.linear_regression import LinearRegressionModel
from src.models.random_forest import RandomForestModel


def test_linear_regression_train(sample_config, sample_data):
    """Test linear regression training"""
    X, y = sample_data
    model = LinearRegressionModel(sample_config)
    
    trained_model = model.train(X, y)
    assert trained_model is not None
    assert hasattr(trained_model, 'coef_')


def test_linear_regression_predict(sample_config, sample_data):
    """Test linear regression prediction"""
    X, y = sample_data
    model = LinearRegressionModel(sample_config)
    model.train(X, y)
    
    predictions = model.predict(X)
    assert len(predictions) == len(y)
    assert isinstance(predictions, np.ndarray)


def test_random_forest_train(sample_config, sample_data):
    """Test random forest training"""
    X, y = sample_data
    model = RandomForestModel(sample_config)
    
    trained_model = model.train(X, y)
    assert trained_model is not None
    assert hasattr(trained_model, 'estimators_')
