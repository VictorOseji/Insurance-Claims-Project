# Insurance Claims Prediction Pipeline

A production-ready machine learning pipeline for predicting insurance claim amounts using MLflow for experiment tracking, Vetiver for model deployment, and Pins for model versioning.

## Project Overview

This project predicts the `Ultimate_Claim_Amount` for insurance claims using various machine learning models. It includes comprehensive data processing, feature engineering, model training, and deployment capabilities.

## Features

- **Data Processing Pipeline**: Automated data loading, validation, and feature engineering
- **Feature Engineering**: 10+ derived features including risk indicators and temporal features
- **Multi-Model Training**: Linear Regression, Random Forest, Gradient Boosting, XGBoost
- **Hyperparameter Tuning**: Automated grid search with cross-validation
- **MLflow Integration**: Complete experiment tracking and model registry
- **Vetiver**: Standardized model deployment format
- **Pins**: Model versioning and storage
- **Data Validation**: Comprehensive quality checks

## Project Structure

```
insurance-claims-prediction/
├── config/
│   ├── config.yaml              # Main configuration
│   ├── model_params.yaml        # Model hyperparameters
│   └── feature_config.yaml      # Feature engineering config
├── data/
│   ├── raw/
│   │   ├── claims_table.csv     # Claims data
│   │   └── policyholder_table.csv # Policy data
│   └── processed/               # Processed data
├── src/
│   ├── data/
│   │   ├── data_loader.py       # Load and merge data
│   │   ├── data_validator.py    # Data quality validation
│   │   ├── feature_engineering.py # Feature creation
│   │   └── preprocessing.py     # Data preprocessing pipeline
│   ├── models/                  # Model implementations
│   ├── training/                # Training orchestration
│   ├── evaluation/              # Model evaluation
│   ├── mlflow_utils/            # MLflow integration
│   ├── vetiver_utils/           # Vetiver integration
│   ├── pins_utils/              # Pins integration
│   └── utils/                   # Utility functions
├── scripts/
│   ├── prepare_data.py          # Data preparation script
│   └── train_insurance_models.py # Main training script
└── models/                      # Saved models and preprocessors
```

## Installation

### Prerequisites
- Python 3.9 or higher
- pip package manager

### Setup

1. **Clone the repository**:
```bash
git clone <repository-url>
cd insurance-claims-prediction
```

2. **Create virtual environment**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
pip install -e .
```

4. **Create necessary directories**:
```bash
mkdir -p data/raw data/processed logs mlruns model_pins_board models
```

5. **Add your data**:
Place your CSV files in the `data/raw/` directory:
- `claims_table.csv`
- `policyholder_table.csv`

## Usage

### Step 1: Data Preparation

Run the data preparation pipeline to load, validate, and engineer features:

```bash
python scripts/prepare_data.py
```

This script will:
- Load and merge claims and policy data
- Validate data quality
- Handle missing values
- Create derived features
- Save processed data

### Step 2: Train Models

Run the model training pipeline:

```bash
python scripts/train_insurance_models.py
```

This script will:
- Load processed data
- Split into train/test sets
- Apply preprocessing transformations
- Train all enabled models with hyperparameter tuning
- Log results to MLflow
- Version models with Pins
- Compare model performance

### Step 3: View Results

Start the MLflow UI to view experiment results:

```bash
mlflow ui
```

Then open http://localhost:5000 in your browser.

## Feature Engineering

The pipeline creates the following derived features:

### Temporal Features
- **FNOL_Delay**: Days between accident and FNOL reporting
- **Settlement_Delay**: Days between FNOL and settlement
- **wk_days**: Day of week when accident occurred

### Demographic Features
- **Driver_Age**: Driver age at time of accident (years)
- **License_Age**: Years since license issue
- **Vehicle_Age**: Vehicle age at time of accident (years)

### Risk Indicators
- **High_Risk_Driver**: Flag for drivers < 25 or > 70 years old
- **Inexperienced_Driver**: Flag for drivers with < 2 years license
- **Old_Vehicle**: Flag for vehicles > 10 years old
- **Early_FNOL**: Flag for FNOL reported within 1 day

## Configuration

### Main Configuration (`config/config.yaml`)

```yaml
project:
  name: "insurance-claims-prediction"
  version: "0.1.0"

data:
  claims_file: "data/raw/claims_table.csv"
  policy_file: "data/raw/policyholder_table.csv"
  test_size: 0.2
  random_state: 42

models:
  target_variable: "Ultimate_Claim_Amount"
  enabled:
    - linear_regression
    - random_forest
    - gradient_boosting
    - xgboost
```

### Model Parameters (`config/model_params.yaml`)

Customize hyperparameter search grids for each model.

### Feature Configuration (`config/feature_config.yaml`)

Define feature engineering rules, imputation strategies, and feature lists.

## Models

### Available Models

1. **Linear Regression**: Baseline model
2. **Random Forest**: Ensemble method with bagging
3. **Gradient Boosting**: Sequential ensemble method
4. **XGBoost**: Optimized gradient boosting

### Evaluation Metrics

All models are evaluated using:
- **MSE**: Mean Squared Error
- **RMSE**: Root Mean Squared Error
- **MAE**: Mean Absolute Error
- **R²**: R-squared score

## Data Requirements

### Claims Table (`claims_table.csv`)
Required columns:
- Policy_ID
- Customer_ID
- Claim_ID
- Accident_Date
- FNOL_Date
- Settlement_Date
- Estimated_Claim_Amount
- Ultimate_Claim_Amount
- Traffic_Condition
- Weather_Condition

### Policyholder Table (`policyholder_table.csv`)
Required columns:
- Policy_ID
- Customer_ID
- Date_of_Birth
- Full_License_Issue_Date
- Vehicle_Year
- (Other relevant policy details)

## Loading Trained Models

### From Pins Board

```python
from src.pins_utils.board_manager import PinsBoardManager
from src.utils.config_loader import ConfigLoader

# Load configuration
config = ConfigLoader().get_main_config()

# Initialize pins manager
pins_manager = PinsBoardManager(config)

# List available models
models = pins_manager.list_models()
print(models)

# Load a specific model
model = pins_manager.load_model("xgboost")

# Load preprocessor
import joblib
preprocessor = joblib.load("models/preprocessor.joblib")

# Make predictions
X_new_processed = preprocessor.transform(X_new)
predictions = model.model.predict(X_new_processed)
```

### From MLflow

```python
import mlflow

# Load model by run ID
model_uri = f"runs:/<run_id>/model"
model = mlflow.sklearn.load_model(model_uri)

# Make predictions
predictions = model.predict(X_new_processed)
```

## Pipeline Workflow

```
┌─────────────────────┐
│   Raw Data Files    │
│  (CSV files in      │
│   data/raw/)        │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  prepare_data.py    │
│  • Load & merge     │
│  • Validate         │
│  • Feature engineer │
│  • Save processed   │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Processed Data     │
│  (data/processed/)  │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ train_insurance_    │
│     models.py       │
│  • Load processed   │
│  • Preprocess       │
│  • Train models     │
│  • Hyperparameter   │
│    tuning           │
│  • Log to MLflow    │
│  • Version with     │
│    Pins             │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Trained Models     │
│  • MLflow registry  │
│  • Pins board       │
│  • Preprocessor     │
└─────────────────────┘
```

## Testing

Run tests:
```bash
pytest tests/ -v
```

Run with coverage:
```bash
pytest tests/ --cov=src --cov-report=html
```

## Troubleshooting

### Common Issues

**Issue**: `FileNotFoundError` when running training script
```
Solution: Run prepare_data.py first to create processed data
```

**Issue**: Missing columns in data
```
Solution: Ensure your CSV files contain all required columns
```

**Issue**: Memory error during training
```
Solution: Reduce parameter grid size in config/model_params.yaml
```

## Best Practices

1. **Always run data preparation first**: `prepare_data.py` before `train_insurance_models.py`
2. **Check data validation results**: Review logs for data quality issues
3. **Monitor MLflow experiments**: Track performance across runs
4. **Version your configurations**: Keep track of config changes
5. **Save preprocessor**: Always save the fitted preprocessor for deployment

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a pull request

## License

MIT License

## Support

For issues and questions:
- Check the logs in `logs/` directory
- Review MLflow UI for experiment details
- Open an issue on GitHub

## Authors

Your Team Name

## Acknowledgments

- MLflow for experiment tracking
- Vetiver for model deployment
- Pins for model versioning
- scikit-learn for machine learning algorithms