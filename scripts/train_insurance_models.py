# ============================================================================
# FILE: scripts/train_insurance_models.py
# ============================================================================
"""Main training script for insurance claims prediction models"""
import sys
from pathlib import Path
import numpy as np
from vetiver import vetiverModel

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config_loader import ConfigLoader
from src.utils.logger import setup_logger
from src.data.data_loader import InsuranceDataLoader
from src.data.preprocessing import DataPreprocessor
from src.training.trainer import ModelTrainer
from src.evaluation.model_comparison import ModelComparator
from src.pins_utils.board_manager import PinsBoardManager


def main():
    """Main training pipeline for insurance claims"""
    # Setup logger
    logger = setup_logger("insurance_training", "logs/training.log")
    
    logger.info("="*80)
    logger.info("Starting Insurance Claims Prediction Pipeline")
    logger.info("="*80)
    
    # Load configuration
    logger.info("\n1. Loading configuration...")
    config_loader = ConfigLoader()
    config = config_loader.get_main_config()
    model_params = config_loader.get_model_params()
    feature_config = config_loader.load_config("feature_config.yaml")
    
    # Load processed data
    logger.info("\n2. Loading processed data...")
    data_loader = InsuranceDataLoader(config)
    
    try:
        processed_data = data_loader.load_processed_data()
    except FileNotFoundError:
        logger.error("Processed data not found. Please run prepare_data.py first.")
        return
    
    # Prepare features and target
    logger.info("\n3. Preparing features and target...")
    preprocessor = DataPreprocessor(config, feature_config)
    
    target_col = config['models']['target_variable']
    
    # Select relevant columns for modeling
    numeric_features = feature_config['numeric_features']
    categorical_features = feature_config['categorical_features']
    
    # Filter to only include features that exist in the dataframe
    available_numeric = [f for f in numeric_features if f in processed_data.columns]
    available_categorical = [f for f in categorical_features if f in processed_data.columns]
    
    all_features = available_numeric + available_categorical + [target_col]
    modeling_data = processed_data[all_features].copy()
    
    # Remove rows with missing target
    modeling_data = modeling_data.dropna(subset=[target_col])
    
    logger.info(f"Modeling dataset shape: {modeling_data.shape}")
    logger.info(f"Numeric features: {len(available_numeric)}")
    logger.info(f"Categorical features: {len(available_categorical)}")
    
    # Prepare X and y
    X, y = preprocessor.prepare_features(modeling_data, target_col)
    
    # Split data
    X_train, X_test, y_train, y_test = preprocessor.split_data(X, y)
    
    # Fit and transform
    X_train_processed, X_test_processed = preprocessor.fit_transform(
        X_train,
        X_test,
        available_numeric,
        available_categorical
    )
    
    # Get feature names after preprocessing
    feature_names = preprocessor.get_feature_names()
    logger.info(f"Feature names after preprocessing: {len(feature_names)} features")
    
    # Save preprocessor
    preprocessor.save_preprocessor()
    
    # Initialize trainer
    logger.info("\n4. Initializing model trainer...")
    trainer = ModelTrainer(config, model_params)
    
    # Store feature names in trainer for Vetiver
    trainer.feature_names = feature_names
    
    # Train all models
    logger.info("\n5. Training models with hyperparameter tuning...")
    logger.info("-"*80)
    
    results = {}
    
    if model_params['linear_regression']['enabled']:
        metrics, run_id = trainer.train_linear_regression(
            X_train_processed, X_test_processed, y_train, y_test
        )
        results['linear_regression'] = {'metrics': metrics, 'run_id': run_id}
    
    if model_params['random_forest']['enabled']:
        metrics, run_id = trainer.train_random_forest(
            X_train_processed, X_test_processed, y_train, y_test
        )
        results['random_forest'] = {'metrics': metrics, 'run_id': run_id}
    
    if model_params['gradient_boosting']['enabled']:
        metrics, run_id = trainer.train_gradient_boosting(
            X_train_processed, X_test_processed, y_train, y_test
        )
        results['gradient_boosting'] = {'metrics': metrics, 'run_id': run_id}
    
    if model_params['xgboost']['enabled']:
        metrics, run_id = trainer.train_xgboost(
            X_train_processed, X_test_processed, y_train, y_test
        )
        results['xgboost'] = {'metrics': metrics, 'run_id': run_id}
    
    # List pinned models
    logger.info("\n6. Reviewing pinned models...")
    pins_manager = PinsBoardManager(config)
    
    # Compare models
    logger.info("\n7. Comparing all models...")
    comparator = ModelComparator(config['mlflow']['experiment_name'])
    best_pin_name = comparator.compare_models()
    
    # Demonstrate loading best model
    logger.info("\n8. Loading best model from pins...")
    if best_pin_name and best_pin_name != 'N/A':
        loaded_model = pins_manager.load_model(best_pin_name)
        if loaded_model:
            # Make a sample prediction
            sample_pred = loaded_model.model.predict(X_test_processed[:3])
            logger.info(f"Sample predictions: {sample_pred}")
            logger.info(f"Actual values: {y_test.iloc[:3].values}")
    
    logger.info("\n" + "="*80)
    logger.info("✓ TRAINING PIPELINE COMPLETE!")
    logger.info("="*80)
    logger.info("\n📊 View MLflow UI: mlflow ui")
    logger.info("Then open http://localhost:5000")
    logger.info(f"\n📌 Pins board location: {config['pins']['board_path']}")
    logger.info("Models are versioned and ready for deployment")


if __name__ == "__main__":
    main()