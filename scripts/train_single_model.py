# ============================================================================
# FILE: scripts/train_single_model.py
# ============================================================================

"""
Train a single model (called by Snakemake for parallel execution)
Usage: python train_single_model.py <model_name>
Example: python train_single_model.py random_forest
"""

import sys
from pathlib import Path
import numpy as np
import json
import mlflow
import joblib

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config_loader import ConfigLoader
from src.utils.logger import setup_logger
from src.models.linear_regression import LinearRegressionModel
from src.models.random_forest import RandomForestModel
from src.models.gradient_boosting import GradientBoostingModel
from src.models.xgboost_model import XGBoostModel
from src.evaluation.metrics import ModelEvaluator
from src.mlflow_utils.experiment_tracker import ExperimentTracker
from src.vetiver_utils.model_wrapper import VetiverModelWrapper
from src.pins_utils.board_manager import PinsBoardManager
from src.training.hyperparameter_tuner import HyperparameterTuner

def main():
    if len(sys.argv) != 2:
        print("Usage: python train_single_model.py <model_name>")
        print("Available models: linear_regression, random_forest, gradient_boosting, xgboost")
        sys.exit(1)
    
    model_name = sys.argv[1]
    logger = setup_logger(f"train_{model_name}", f"logs/train_{model_name}.log")
    
    logger.info("="*60)
    logger.info(f"TRAINING: {model_name.upper().replace('_', ' ')}")
    logger.info("="*60)
    
    # Load configuration
    logger.info("Loading configuration...")
    config_loader = ConfigLoader()
    config = config_loader.get_main_config()
    model_params = config_loader.get_model_params()
    
    # Load preprocessed data
    logger.info("Loading preprocessed data...")
    X_train = np.load("data/interim/X_train.npy")
    X_test = np.load("data/interim/X_test.npy")
    y_train = np.load("data/interim/y_train.npy")
    y_test = np.load("data/interim/y_test.npy")
    
    with open("data/interim/feature_names.json", "r") as f:
        feature_names = json.load(f)
    
    logger.info(f"X_train: {X_train.shape}")
    logger.info(f"X_test: {X_test.shape}")
    logger.info(f"Features: {len(feature_names)}")
    
    # Initialize components
    evaluator = ModelEvaluator()
    tracker = ExperimentTracker(config)
    vetiver_wrapper = VetiverModelWrapper()
    pins_manager = PinsBoardManager(config)
    tuner = HyperparameterTuner(config)
    
    # Create output directory
    output_dir = Path(f"results/models/{model_name}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Train model
    logger.info("Starting model training...")
    
    with mlflow.start_run(run_name=f"{model_name.replace('_', ' ').title()}_Snakemake"):
        
        if model_name == "linear_regression":
            logger.info("Training Linear Regression...")
            model_instance = LinearRegressionModel(config)
            trained_model = model_instance.train(X_train, y_train)
            best_params = {}
            
        elif model_name == "random_forest":
            logger.info("Training Random Forest with hyperparameter tuning...")
            model_instance = RandomForestModel(config)
            param_grid = model_params['random_forest']['param_grid']
            logger.info(f"Parameter grid: {param_grid}")
            trained_model, best_params = tuner.tune(
                model_instance.model, param_grid, X_train, y_train
            )
            model_instance.model = trained_model
            logger.info(f"Best parameters: {best_params}")
            
        elif model_name == "gradient_boosting":
            logger.info("Training Gradient Boosting with hyperparameter tuning...")
            model_instance = GradientBoostingModel(config)
            param_grid = model_params['gradient_boosting']['param_grid']
            logger.info(f"Parameter grid: {param_grid}")
            trained_model, best_params = tuner.tune(
                model_instance.model, param_grid, X_train, y_train
            )
            model_instance.model = trained_model
            logger.info(f"Best parameters: {best_params}")
            
        elif model_name == "xgboost":
            logger.info("Training XGBoost with hyperparameter tuning...")
            model_instance = XGBoostModel(config)
            param_grid = model_params['xgboost']['param_grid']
            logger.info(f"Parameter grid: {param_grid}")
            trained_model, best_params = tuner.tune(
                model_instance.model, param_grid, X_train, y_train
            )
            model_instance.model = trained_model
            logger.info(f"Best parameters: {best_params}")
        
        else:
            logger.error(f"Unknown model: {model_name}")
            logger.error("Available models: linear_regression, random_forest, gradient_boosting, xgboost")
            sys.exit(1)
        
        # Evaluate
        logger.info("Evaluating model on test set...")
        y_pred = model_instance.predict(X_test)
        metrics = evaluator.evaluate_model(y_test, y_pred)
        
        logger.info("Model Performance:")
        logger.info(f"  R² Score: {metrics['r2']:.4f}")
        logger.info(f"  RMSE: {metrics['rmse']:.2f}")
        logger.info(f"  MAE: {metrics['mae']:.2f}")
        logger.info(f"  MSE: {metrics['mse']:.2f}")
        
        # Log to MLflow
        logger.info("Logging to MLflow...")
        tracker.set_tags({
            "model_type": model_instance.model_name,
            "training_mode": "snakemake_parallel"
        })
        if best_params:
            tracker.log_params(best_params)
        tracker.log_metrics(metrics)
        tracker.log_model(trained_model)
        
        # Create Vetiver model and pin
        logger.info("Creating Vetiver model and pinning...")
        v_model = vetiver_wrapper.create_vetiver_model(
            trained_model, model_name, X_train, feature_names=feature_names
        )
        pin_name = pins_manager.pin_model(
            v_model, model_instance.model_name, metrics, best_params
        )
        
        mlflow.set_tag("vetiver_pin_name", pin_name)
        logger.info(f"Model pinned as: {pin_name}")
        
        # Save model locally
        logger.info("Saving model to local storage...")
        model_path = output_dir / "model.pkl"
        joblib.dump(trained_model, model_path)
        
        # Save metrics
        logger.info("Saving metrics...")
        metrics_with_info = {
            "model_name": model_name,
            "model_type": model_instance.model_name,
            "r2": metrics['r2'],
            "rmse": metrics['rmse'],
            "mae": metrics['mae'],
            "mse": metrics['mse'],
            "best_params": best_params,
            "pin_name": pin_name,
            "num_features": len(feature_names)
        }
        
        metrics_path = output_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics_with_info, f, indent=2)
        
        logger.info("="*60)
        logger.info(f"✓ {model_instance.model_name.upper()} TRAINING COMPLETE!")
        logger.info("="*60)
        logger.info("Outputs:")
        logger.info(f"  - Model: {model_path}")
        logger.info(f"  - Metrics: {metrics_path}")
        logger.info(f"  - MLflow Run ID: {mlflow.active_run().info.run_id}")
        logger.info(f"  - Vetiver Pin: {pin_name}")
        logger.info("="*60)

if __name__ == "__main__":
    main()
