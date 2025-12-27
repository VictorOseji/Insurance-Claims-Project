# ============================================================================
# FILE: src/training/trainer.py
# ============================================================================
"""Main training orchestration"""
import mlflow
# import numpy as np
# from typing import Dict, Any
import logging
from ..models.linear_regression import LinearRegressionModel
from ..models.random_forest import RandomForestModel
from ..models.gradient_boosting import GradientBoostingModel
from ..models.xgboost_model import XGBoostModel
from ..evaluation.metrics import ModelEvaluator
from ..mlflow_utils.experiment_tracker import ExperimentTracker
from ..vetiver_utils.model_wrapper import VetiverModelWrapper
from ..pins_utils.board_manager import PinsBoardManager
from .hyperparameter_tuner import HyperparameterTuner

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Orchestrate model training pipeline"""
    
    def __init__(self, config: dict, model_params: dict):
        self.config = config
        self.model_params = model_params
        self.experiment_tracker = ExperimentTracker(config)
        self.pins_manager = PinsBoardManager(config)
        self.tuner = HyperparameterTuner(config)
        self.evaluator = ModelEvaluator()
        self.vetiver_wrapper = VetiverModelWrapper()
    
    def train_linear_regression(self, X_train, X_test, y_train, y_test):
        """Train Linear Regression model"""
        with mlflow.start_run(run_name="Linear_Regression") as run:
            # Initialize model
            lr_model = LinearRegressionModel(self.config)
            
            # Train
            model = lr_model.train(X_train, y_train)
            
            # Predict and evaluate
            y_pred = lr_model.predict(X_test)
            metrics = self.evaluator.evaluate_model(y_test, y_pred)
            
            # Log to MLflow
            self.experiment_tracker.set_tags({
                "model_type": lr_model.model_name,
                "run_id": run.info.run_id
            })
            self.experiment_tracker.log_metrics(metrics)
            self.experiment_tracker.log_model(model)  # ✅ Unified method
            
            # Get feature names from preprocessor (if available)
            feature_names = getattr(self, 'feature_names', None)
            
            # Create Vetiver model and pin
            v_model = self.vetiver_wrapper.create_vetiver_model(
                model, "linear_regression", X_train, feature_names=feature_names
            )
            pin_name = self.pins_manager.pin_model(
                v_model, lr_model.model_name, metrics
            )
            
            mlflow.set_tag("vetiver_pin_name", pin_name)
            
            logger.info(f"{lr_model.model_name} - R²: {metrics['r2']:.4f}, "
                       f"RMSE: {metrics['rmse']:.4f}")
            
            return metrics, run.info.run_id
    
    def train_random_forest(self, X_train, X_test, y_train, y_test):
        """Train Random Forest with tuning"""
        with mlflow.start_run(run_name="Random_Forest_Tuned") as run:
            # Initialize model
            rf_model = RandomForestModel(self.config)
            
            # Get param grid
            param_grid = self.model_params['random_forest']['param_grid']
            
            # Tune hyperparameters
            best_model, best_params = self.tuner.tune(
                rf_model.model, param_grid, X_train, y_train
            )
            rf_model.model = best_model
            rf_model.set_params(best_params)
            
            # Predict and evaluate
            y_pred = rf_model.predict(X_test)
            metrics = self.evaluator.evaluate_model(y_test, y_pred)
            
            # Log to MLflow
            self.experiment_tracker.set_tags({
                "model_type": rf_model.model_name,
                "run_id": run.info.run_id
            })
            self.experiment_tracker.log_params(best_params)
            self.experiment_tracker.log_metrics(metrics)
            self.experiment_tracker.log_model(best_model)  # ✅ Unified method
            
            # Get feature names from preprocessor (if available)
            feature_names = getattr(self, 'feature_names', None)
            
            # Create Vetiver model and pin
            v_model = self.vetiver_wrapper.create_vetiver_model(
                best_model, "random_forest", X_train, feature_names=feature_names
            )
            pin_name = self.pins_manager.pin_model(
                v_model, rf_model.model_name, metrics, best_params
            )
            
            mlflow.set_tag("vetiver_pin_name", pin_name)
            
            logger.info(f"{rf_model.model_name} - R²: {metrics['r2']:.4f}, "
                       f"RMSE: {metrics['rmse']:.4f}")
            logger.info(f"Best params: {best_params}")
            
            return metrics, run.info.run_id
    
    def train_gradient_boosting(self, X_train, X_test, y_train, y_test):
        """Train Gradient Boosting with tuning"""
        with mlflow.start_run(run_name="Gradient_Boosting_Tuned") as run:
            # Initialize model
            gbm_model = GradientBoostingModel(self.config)
            
            # Get param grid
            param_grid = self.model_params['gradient_boosting']['param_grid']
            
            # Tune hyperparameters
            best_model, best_params = self.tuner.tune(
                gbm_model.model, param_grid, X_train, y_train
            )
            gbm_model.model = best_model
            gbm_model.set_params(best_params)
            
            # Predict and evaluate
            y_pred = gbm_model.predict(X_test)
            metrics = self.evaluator.evaluate_model(y_test, y_pred)
            
            # Log to MLflow
            self.experiment_tracker.set_tags({
                "model_type": gbm_model.model_name,
                "run_id": run.info.run_id
            })
            self.experiment_tracker.log_params(best_params)
            self.experiment_tracker.log_metrics(metrics)
            self.experiment_tracker.log_model(best_model)  # ✅ Unified method
            
            # Get feature names from preprocessor (if available)
            feature_names = getattr(self, 'feature_names', None)
            
            # Create Vetiver model and pin
            v_model = self.vetiver_wrapper.create_vetiver_model(
                best_model, "gradient_boosting", X_train, feature_names=feature_names
            )
            pin_name = self.pins_manager.pin_model(
                v_model, gbm_model.model_name, metrics, best_params
            )
            
            mlflow.set_tag("vetiver_pin_name", pin_name)
            
            logger.info(f"{gbm_model.model_name} - R²: {metrics['r2']:.4f}, "
                       f"RMSE: {metrics['rmse']:.4f}")
            logger.info(f"Best params: {best_params}")
            
            return metrics, run.info.run_id
    
    def train_xgboost(self, X_train, X_test, y_train, y_test):
        """Train XGBoost with tuning"""
        with mlflow.start_run(run_name="XGBoost_Tuned") as run:
            # Initialize model
            xgb_model = XGBoostModel(self.config)
            
            # Get param grid
            param_grid = self.model_params['xgboost']['param_grid']
            
            # Tune hyperparameters
            best_model, best_params = self.tuner.tune(
                xgb_model.model, param_grid, X_train, y_train
            )
            xgb_model.model = best_model
            xgb_model.set_params(best_params)
            
            # Predict and evaluate
            y_pred = xgb_model.predict(X_test)
            metrics = self.evaluator.evaluate_model(y_test, y_pred)
            
            # Log to MLflow
            self.experiment_tracker.set_tags({
                "model_type": xgb_model.model_name,
                "run_id": run.info.run_id
            })
            self.experiment_tracker.log_params(best_params)
            self.experiment_tracker.log_metrics(metrics)
            self.experiment_tracker.log_model(best_model)  # ✅ Unified method handles XGBoost
            
            # Get feature names from preprocessor (if available)
            feature_names = getattr(self, 'feature_names', None)
            
            # Create Vetiver model and pin
            v_model = self.vetiver_wrapper.create_vetiver_model(
                best_model, "xgboost", X_train, feature_names=feature_names
            )
            pin_name = self.pins_manager.pin_model(
                v_model, xgb_model.model_name, metrics, best_params
            )
            
            mlflow.set_tag("vetiver_pin_name", pin_name)
            
            logger.info(f"{xgb_model.model_name} - R²: {metrics['r2']:.4f}, "
                       f"RMSE: {metrics['rmse']:.4f}")
            logger.info(f"Best params: {best_params}")
            
            return metrics, run.info.run_id

