from setuptools import setup, find_packages

setup(
    name="ml-model-pipeline",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "xgboost>=2.0.0",
        "mlflow>=2.9.0",
        "vetiver>=0.2.0",
        "pins>=0.8.0",
    ],
    extras_require={
        "dev": ["pytest", "black", "flake8", "mypy"],
        "api": ["fastapi", "uvicorn"],
    },
    python_requires=">=3.9",
)