# ============================================================================
# FILE: tests/conftest.py
# ============================================================================
"""Pytest configuration and fixtures"""
import pytest
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def sample_data():
    """Create sample data for testing"""
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = np.random.randn(100)
    return X, y


@pytest.fixture
def sample_config():
    """Create sample configuration"""
    return {
        'data': {
            'test_size': 0.2,
            'random_state': 42,
            'n_samples': 100,
            'n_features': 5,
            'n_informative': 4,
            'noise': 1
        },
        'mlflow': {
            'tracking_uri': 'test_mlruns',
            'experiment_name': 'test_experiment'
        },
        'pins': {
            'board_path': 'test_pins_board',
            'board_type': 'folder'
        },
        'training': {
            'cv_folds': 2,
            'n_jobs': 1,
            'verbose': 0,
            'scoring': 'r2'
        }
    }


@pytest.fixture
def sample_model_params():
    """Create sample model parameters"""
    return {
        'linear_regression': {
            'enabled': True,
            'params': {}
        },
        'random_forest': {
            'enabled': True,
            'param_grid': {
                'n_estimators': [10, 50],
                'max_depth': [5, 10]
            }
        }
    }
