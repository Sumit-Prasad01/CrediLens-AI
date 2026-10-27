from pathlib import Path
from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config.constants import CONFIG_FILE_PATH, CLEANED_DATA_PATH, PROCESSED_DATA_DIR
from utils.custom_exception import CustomException
from utils.helpers import read_yaml, ensure_dir, save_object
from utils.logger import get_logger

logger = get_logger(__name__)


class DataTransformation:
    """Prepare train/test splits, handle class imbalance, and create preprocessing pipelines."""

    def __init__(self, config_path: Path = CONFIG_FILE_PATH) -> None:
        try:
            self.config = read_yaml(config_path) if Path(config_path).exists() else {}
            trans_cfg = self.config.get("data_transformation", {})
            self.target_column = trans_cfg.get("target_column", "loan_status")
            self.test_size = trans_cfg.get("test_size", 0.2)
            self.random_state = trans_cfg.get("random_state", 42)

            self.numeric_cols = trans_cfg.get("numeric_columns", [
                "person_age", "person_income", "person_emp_length", "loan_amnt",
                "loan_int_rate", "loan_percent_income", "cb_person_cred_hist_length"
            ])
            self.categorical_cols = trans_cfg.get("categorical_columns", [
                "person_home_ownership", "loan_intent", "loan_grade", "cb_person_default_on_file"
            ])

            self.processed_dir = Path(PROCESSED_DATA_DIR)
            ensure_dir(self.processed_dir)
        except Exception as e:
            logger.error("Initialization error in DataTransformation")
            raise CustomException("Failed to initialize DataTransformation", e)

    def get_preprocessor(self, include_scaler: bool = False) -> ColumnTransformer:
        """Create sklearn ColumnTransformer for numeric and categorical pipelines."""
        try:
            if include_scaler:
                num_pipeline = Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ])
            else:
                num_pipeline = Pipeline([
                    ("imputer", SimpleImputer(strategy="median"))
                ])

            cat_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
                ("onehot", OneHotEncoder(handle_unknown="ignore"))
            ])

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", num_pipeline, self.numeric_cols),
                    ("cat", cat_pipeline, self.categorical_cols)
                ]
            )
            return preprocessor
        except Exception as e:
            logger.error("Failed to construct ColumnTransformer preprocessor")
            raise CustomException("Preprocessor construction failed", e)

    def split_data(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, float]:
        """Perform stratified train-test split and calculate imbalance scale_pos_weight."""
        try:
            if self.target_column not in df.columns:
                raise KeyError(f"Target column '{self.target_column}' not found in dataset columns.")

            X = df.drop(columns=[self.target_column])
            y = df[self.target_column]

            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=self.test_size,
                random_state=self.random_state,
                stratify=y
            )

            neg, pos = np.bincount(y_train)
            scale_pos_weight = float(neg / pos)

            logger.info(
                f"Data Split complete: Train shape = {X_train.shape}, Test shape = {X_test.shape}"
            )
            logger.info(
                f"Class distribution in train - Non-Default (0): {neg}, Default (1): {pos} "
                f"(scale_pos_weight = {scale_pos_weight:.3f})"
            )

            return X_train, X_test, y_train, y_test, scale_pos_weight

        except Exception as e:
            logger.error("Error during train-test splitting and imbalance weighting")
            raise CustomException("Failed to split data", e)

    def run(
        self, df: pd.DataFrame | None = None, cleaned_csv_path: Path = CLEANED_DATA_PATH
    ) -> Dict[str, Any]:
        """Execute transformation pipeline."""
        try:
            if df is None:
                if not Path(cleaned_csv_path).exists():
                    raise FileNotFoundError(f"Cleaned dataset not found at {cleaned_csv_path}")
                df = pd.read_csv(cleaned_csv_path)

            X_train, X_test, y_train, y_test, scale_pos_weight = self.split_data(df)

            preprocessor_tree = self.get_preprocessor(include_scaler=False)
            preprocessor_linear = self.get_preprocessor(include_scaler=True)

            return {
                "X_train": X_train,
                "X_test": X_test,
                "y_train": y_train,
                "y_test": y_test,
                "scale_pos_weight": scale_pos_weight,
                "preprocessor_tree": preprocessor_tree,
                "preprocessor_linear": preprocessor_linear,
                "numeric_cols": self.numeric_cols,
                "categorical_cols": self.categorical_cols,
            }

        except Exception as e:
            logger.error("Data Transformation run failed")
            raise CustomException("Data Transformation failed", e)


if __name__ == "__main__":
    transformer = DataTransformation()
    result = transformer.run()
    logger.info("Data Transformation test run successful.")
