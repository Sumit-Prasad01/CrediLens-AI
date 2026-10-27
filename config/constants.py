from pathlib import Path

# Project Root Directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Configuration Path
CONFIG_FILE_PATH = PROJECT_ROOT / "config" / "config.yaml"

# Artifact Directories
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
RAW_DATA_DIR = ARTIFACTS_DIR / "raw"
PROCESSED_DATA_DIR = ARTIFACTS_DIR / "processed"
MODELS_DIR = ARTIFACTS_DIR / "models"
VISUALS_DIR = ARTIFACTS_DIR / "visuals"

# File Paths
RAW_DATA_PATH = RAW_DATA_DIR / "credit_risk_dataset.csv"
CLEANED_DATA_PATH = PROCESSED_DATA_DIR / "cleaned_credit_risk.csv"
MODEL_PATH = MODELS_DIR / "credit_risk_model.pkl"
BEST_THRESHOLD_PATH = MODELS_DIR / "best_threshold.pkl"
METRICS_PATH = MODELS_DIR / "model_metrics.json"
ERROR_ANALYSIS_PATH = PROCESSED_DATA_DIR / "error_analysis.csv"
