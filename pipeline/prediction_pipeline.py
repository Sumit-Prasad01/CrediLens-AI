from pathlib import Path
from typing import Dict, Any, Union
import numpy as np
import pandas as pd

from config.constants import MODEL_PATH, BEST_THRESHOLD_PATH
from utils.custom_exception import CustomException
from utils.helpers import load_object
from utils.logger import get_logger

logger = get_logger(__name__)


class PredictPipeline:
    """Production prediction pipeline that loads the champion calibrated model and best threshold."""

    def __init__(
        self,
        model_path: Path = MODEL_PATH,
        threshold_path: Path = BEST_THRESHOLD_PATH
    ) -> None:
        try:
            self.model_path = Path(model_path)
            self.threshold_path = Path(threshold_path)

            if not self.model_path.exists():
                raise FileNotFoundError(f"Model file not found at: {self.model_path}")
            if not self.threshold_path.exists():
                raise FileNotFoundError(f"Threshold file not found at: {self.threshold_path}")

            self.model = load_object(self.model_path)
            self.threshold = float(load_object(self.threshold_path))
            logger.info(f"Loaded calibrated model and optimal threshold ({self.threshold:.4f})")

        except Exception as e:
            logger.error("Initialization error in PredictPipeline")
            raise CustomException("Failed to initialize PredictPipeline", e)

    def predict(self, features: pd.DataFrame) -> Dict[str, Any]:
        """Predict loan default probabilities, decisions, and risk levels."""
        try:
            probabilities = self.model.predict_proba(features)[:, 1]
            predictions = (probabilities >= self.threshold).astype(int)

            risk_categories = []
            for prob in probabilities:
                if prob < 0.25:
                    risk_categories.append("Low Risk (Approved)")
                elif prob < self.threshold:
                    risk_categories.append("Moderate Risk (Under Review / Approved with Conditions)")
                else:
                    risk_categories.append("High Risk (Default Likely / Rejected)")

            results = {
                "default_probability": [round(float(p), 4) for p in probabilities],
                "prediction": [int(p) for p in predictions],
                "decision": ["Default Risk" if p == 1 else "Approved" for p in predictions],
                "risk_category": risk_categories,
                "applied_threshold": round(self.threshold, 4),
            }
            return results

        except Exception as e:
            logger.error("Error during prediction execution")
            raise CustomException("Prediction execution failed", e)


class CustomData:
    """Helper class to map single applicant input features into a Pandas DataFrame."""

    def __init__(
        self,
        person_age: int,
        person_income: float,
        person_home_ownership: str,
        person_emp_length: float,
        loan_intent: str,
        loan_grade: str,
        loan_amnt: float,
        loan_int_rate: float,
        loan_percent_income: float,
        cb_person_default_on_file: str,
        cb_person_cred_hist_length: int,
    ) -> None:
        self.person_age = person_age
        self.person_income = person_income
        self.person_home_ownership = person_home_ownership
        self.person_emp_length = person_emp_length
        self.loan_intent = loan_intent
        self.loan_grade = loan_grade
        self.loan_amnt = loan_amnt
        self.loan_int_rate = loan_int_rate
        self.loan_percent_income = loan_percent_income
        self.cb_person_default_on_file = cb_person_default_on_file
        self.cb_person_cred_hist_length = cb_person_cred_hist_length

    def get_data_as_data_frame(self) -> pd.DataFrame:
        """Convert applicant attributes to a single-row DataFrame."""
        try:
            data_dict = {
                "person_age": [self.person_age],
                "person_income": [self.person_income],
                "person_home_ownership": [self.person_home_ownership],
                "person_emp_length": [self.person_emp_length],
                "loan_intent": [self.loan_intent],
                "loan_grade": [self.loan_grade],
                "loan_amnt": [self.loan_amnt],
                "loan_int_rate": [self.loan_int_rate],
                "loan_percent_income": [self.loan_percent_income],
                "cb_person_default_on_file": [self.cb_person_default_on_file],
                "cb_person_cred_hist_length": [self.cb_person_cred_hist_length],
            }
            return pd.DataFrame(data_dict)
        except Exception as e:
            logger.error("Failed to convert applicant data to DataFrame")
            raise CustomException("Failed to construct applicant DataFrame", e)


if __name__ == "__main__":
    # Smoke test with sample applicant
    sample = CustomData(
        person_age=25,
        person_income=45000.0,
        person_home_ownership="RENT",
        person_emp_length=3.0,
        loan_intent="PERSONAL",
        loan_grade="B",
        loan_amnt=8000.0,
        loan_int_rate=11.5,
        loan_percent_income=0.18,
        cb_person_default_on_file="N",
        cb_person_cred_hist_length=3,
    )
    df = sample.get_data_as_data_frame()
    pipeline = PredictPipeline()
    res = pipeline.predict(df)
    logger.info(f"Sample Prediction Result: {res}")
