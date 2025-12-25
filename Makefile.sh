# ============================================================================
# FILE: Makefile (Optional but helpful)
# ============================================================================
"""
.PHONY: install test clean train lint format

install:
\tpip install -r requirements.txt
\tpip install -e .

test:
\tpytest tests/ -v --cov=src

clean:
\trm -rf __pycache__ .pytest_cache .coverage htmlcov
\tfind . -type d -name __pycache__ -exec rm -rf {} +
\tfind . -type f -name "*.pyc" -delete

train:
\tpython scripts/train_all_models.py

lint:
\tflake8 src/

format:
\tblack src/ tests/ scripts/

mlflow-ui:
\tmlflow ui

setup:
\tmkdir -p data/raw data/processed logs mlruns model_pins_board
\tcp .env.example .env
"""
