# ============================================================================
# FILE: src/evaluation/model_comparison.py
# ============================================================================
"""Model comparison utilities"""
import mlflow
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class ModelComparator:
    """Compare and analyze model performance"""
    
    def __init__(self, experiment_name: str):
        self.experiment_name = experiment_name
    
    def compare_models(self) -> Optional[str]:
        """Compare all models and identify the best one"""
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        
        if not experiment:
            logger.error(f"Experiment {self.experiment_name} not found")
            return None
        
        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
        
        if runs.empty:
            logger.warning("No runs found in experiment")
            return None
        
        # Sort by R² score
        runs_sorted = runs.sort_values('metrics.r2', ascending=False)
        
        # Print comparison
        print("\n" + "="*80)
        print("MODEL COMPARISON SUMMARY")
        print("="*80)
        print(f"{'Model':<25} {'R²':<10} {'RMSE':<10} {'MAE':<10} {'Pin Name':<20}")
        print("-"*80)
        
        for _, run in runs_sorted.iterrows():
            model_name = run.get('tags.model_type', 'Unknown')
            r2 = run.get('metrics.r2', 0)
            rmse = run.get('metrics.rmse', 0)
            mae = run.get('metrics.mae', 0)
            pin_name = run.get('tags.vetiver_pin_name', 'N/A')
            print(f"{model_name:<25} {r2:<10.4f} {rmse:<10.4f} {mae:<10.4f} {pin_name:<20}")

            # Handle None values with default 0
            model_name = model_name if model_name is not None else 'Unknown'
            r2 = r2 if r2 is not None else 0
            rmse = rmse if rmse is not None else 0
            mae = mae if mae is not None else 0
            pin_name = pin_name if pin_name is not None else 'N/A'

            print(f"{model_name:<25} {r2:<10.4f} {rmse:<10.4f} {mae:<10.4f} {pin_name:<20}")

        
        print("-"*80)
        best_run = runs_sorted.iloc[0]
        best_r2 = best_run.get('metrics.r2', 0)
        best_r2 = best_r2 if best_r2 is not None else 0
        
        best_model_name = best_run.get('tags.model_type', 'Unknown')
        best_model_name = best_model_name if best_model_name is not None else 'Unknown'
        
        best_pin_name = best_run.get('tags.vetiver_pin_name', 'N/A')
        best_pin_name = best_pin_name if best_pin_name is not None else 'N/A'
        
        best_run_id = best_run.get('run_id', 'N/A')
        best_run_id = best_run_id if best_run_id is not None else 'N/A'

        print(f"\n🏆 Best Model: {best_run.get('tags.model_type', 'Unknown')}")
        print(f"   MLflow Run ID: {best_run['run_id']}")
        print(f"   Vetiver Pin: {best_run.get('tags.vetiver_pin_name', 'N/A')}")
        print(f"   R² Score: {best_r2:.2f}")
        print("="*80 + "\n")
        
        return best_run.get('tags.vetiver_pin_name')