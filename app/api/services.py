from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd

from config.constants import MODEL_PATH, BEST_THRESHOLD_PATH
from pipeline.prediction_pipeline import PredictPipeline
from app.api.schemas import LoanApplicationRequest, PredictionResponse
from utils.custom_exception import CustomException
from utils.logger import get_logger

logger = get_logger(__name__)


class ModelService:
    """Singleton service to manage model lifecycle and inference in FastAPI."""

    _instance: Optional["ModelService"] = None

    def __init__(self) -> None:
        self.model_path = Path(MODEL_PATH)
        self.threshold_path = Path(BEST_THRESHOLD_PATH)
        self.pipeline: Optional[PredictPipeline] = None
        self.load_model()

    @classmethod
    def get_instance(cls) -> "ModelService":
        """Get or create singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def load_model(self) -> None:
        """Load the model and threshold from artifacts."""
        try:
            if not self.model_path.exists():
                logger.warning(f"Model file not found at {self.model_path}. Train the pipeline first.")
                self.pipeline = None
                return

            self.pipeline = PredictPipeline(
                model_path=self.model_path,
                threshold_path=self.threshold_path
            )
            logger.info("ModelService: Calibrated champion model loaded successfully.")
        except Exception as e:
            logger.error(f"ModelService: Failed to load model artifacts: {e}")
            self.pipeline = None
            raise CustomException("Failed to load model service", e)

    def is_ready(self) -> bool:
        """Check if model pipeline is ready to serve requests."""
        return self.pipeline is not None and self.pipeline.model is not None

    def predict(self, request: LoanApplicationRequest) -> PredictionResponse:
        """Execute risk inference on validated application data."""
        if not self.is_ready():
            raise RuntimeError("Model is not initialized. Ensure credit_risk_model.pkl exists.")

        try:
            # Convert Pydantic request to pandas DataFrame
            input_dict = request.model_dump()
            input_df = pd.DataFrame([input_dict])

            # Run inference via PredictPipeline
            result = self.pipeline.predict(input_df)

            prob = float(result["default_probability"][0])
            pred = int(result["prediction"][0])
            threshold = float(result["applied_threshold"])
            risk_cat = str(result["risk_category"][0])
            decision = str(result["decision"][0])
            result_str = "High Risk" if pred == 1 else "Low Risk"

            return PredictionResponse(
                default_probability=round(prob, 4),
                default_prediction=pred,
                threshold=round(threshold, 4),
                Result=result_str,
                risk_category=risk_cat,
                decision=decision
            )
        except Exception as e:
            logger.error(f"Error during model inference in ModelService: {e}")
            raise CustomException("Inference error in ModelService", e)


model_service = ModelService.get_instance()
