# ============================================================================
# FILE: src/mlflow_utils/experiment_tracker.py
# ============================================================================
"""MLflow experiment tracking utilities"""
import mlflow
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


class ExperimentTracker:
    """Manage MLflow experiment tracking"""
    
    def __init__(self, config: dict):
        self.config = config['mlflow']
        self._setup_mlflow()
    
    def _setup_mlflow(self):
        """Initialize MLflow tracking"""
        mlflow.set_tracking_uri(self.config['tracking_uri'])
        mlflow.set_experiment(self.config['experiment_name'])
        logger.info(f"MLflow tracking initialized: {self.config['experiment_name']}")
    
    def log_metrics(self, metrics: Dict[str, float]):
        """Log metrics to MLflow"""
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)
    
    def log_params(self, params: Dict[str, Any]):
        """Log parameters to MLflow"""
        for param_name, param_value in params.items():
            mlflow.log_param(param_name, param_value)
    
    def log_model(self, model, model_type: str = "sklearn"):
        """Log model to MLflow"""
        if model_type == "sklearn":
            mlflow.sklearn.log_model(model, "model")
        elif model_type == "xgboost":
            mlflow.xgboost.log_model(model, "model")
    
    def set_tags(self, tags: Dict[str, str]):
        """Set tags for the run"""
        for tag_name, tag_value in tags.items():
            mlflow.set_tag(tag_name, tag_value)

