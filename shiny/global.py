# global.py

# --- Library Imports ---

# Core Shiny
from shiny import ui
import pandas as pd  # Replaces: dplyr, readr, readxl
import numpy as np

# Visualization
import plotly.graph_objects as go  # Replaces: ggplot2, plotly

# Data Manipulation
from datetime import datetime  # Replaces: lubridate

# Machine Learning
# Replaces: randomForest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import xgboost  # Replaces: xgboost
import shap     # Replaces: SHAPforxgboost, iml (partial)
import lime     # Replaces: lime
import lime.lime_tabular

# MLOps
import mlflow
from mlflow.tracking import MlflowClient  # Replaces: mlflow::mlflow_client

# Note: 'DT' and 'shinycssloaders' are R-specific packages.
# - DataTables: Handled by Pandas and ui.output_data_frame in Python Shiny.
# - CSS Loaders (Spinners): Handled by Shiny Python's built-in reactive rendering or custom CSS.

# --- Global Options ---

# R: options(shiny.maxRequestSize = 100 * 1024^2)
# In Shiny for Python, file upload limits are typically handled in the App() constructor 
# or server configuration. We define it here as a constant for reference.
MAX_REQUEST_SIZE = 100 * 1024**2  # 100MB

# R: options(spinner.type = 4)
# Spinner configuration is usually handled via CSS classes or specific UI functions in Python.
SPINNER_TYPE = 4

# --- Initialize MLflow Connection ---

# R: mlflow_client <- mlflow::mlflow_client(tracking_uri = "http://localhost:5000")
try:
    mlflow_client = MlflowClient(tracking_uri="http://localhost:5000")
except Exception as e:
    print(f"Warning: Could not connect to MLflow at localhost:5000. Error: {e}")
    mlflow_client = None

# --- Define Global Constants ---

# R: CLAIM_TYPES <- c(...)
CLAIM_TYPES = ["Third-Party Damage", "Windshield Damage", "Collision", "Theft"]

# R: VEHICLE_TYPES <- c(...)
VEHICLE_TYPES = ["SUV", "Sedan", "Van", "EV"]

# R: TRAFFIC_CONDITIONS <- c(...)
TRAFFIC_CONDITIONS = ["Light", "Moderate", "Heavy"]

# R: WEATHER_CONDITIONS <- c(...)
WEATHER_CONDITIONS = ["Clear", "Rainy", "Foggy", "Snowy"]