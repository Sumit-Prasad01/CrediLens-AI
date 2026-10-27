from pathlib import Path
from typing import Dict, Any

from config.constants import CONFIG_FILE_PATH
from src.data_ingestion import DataIngestion
from src.data_validation import DataValidation
from src.data_transformation import DataTransformation
from src.model_trainer import ModelTrainer
from src.model_evaluation import ModelEvaluation
from src.model_explainability import ModelExplainability
from utils.custom_exception import CustomException
from utils.helpers import read_yaml
from utils.logger import get_logger

logger = get_logger(__name__)


class TrainingPipeline:
    """Orchestrates the entire end-to-end ML training, calibration, evaluation, and explainability workflow."""

    def __init__(self, config_path: Path = CONFIG_FILE_PATH) -> None:
        try:
            self.config_path = config_path
            self.config = read_yaml(config_path) if Path(config_path).exists() else {}
            logger.info("Initialized Training Pipeline.")
        except Exception as e:
            logger.error("Initialization error in TrainingPipeline")
            raise CustomException("Failed to initialize TrainingPipeline", e)

    def run(self) -> Dict[str, Any]:
        """Execute the complete training pipeline."""
        try:
            logger.info("==================================================")
            logger.info("           STARTING TRAINING PIPELINE             ")
            logger.info("==================================================")

            # Step 1: Data Ingestion
            logger.info(">>> STEP 1: Data Ingestion <<<")
            ingestion = DataIngestion(config_path=self.config_path)
            raw_csv_path = ingestion.run()
            logger.info(f"Raw data ready at: {raw_csv_path}")

            # Step 2: Data Validation & EDA Visuals
            logger.info(">>> STEP 2: Data Validation & EDA Visuals <<<")
            validator = DataValidation(config_path=self.config_path)
            cleaned_df, validation_report = validator.run(raw_csv_path=raw_csv_path)
            logger.info(f"Data validation complete. Cleaned shape: {cleaned_df.shape}")

            # Step 3: Data Transformation & Pipelines
            logger.info(">>> STEP 3: Data Transformation <<<")
            transformer = DataTransformation(config_path=self.config_path)
            transform_data = transformer.run(df=cleaned_df)
            logger.info(f"Transformation complete. Scale pos weight: {transform_data['scale_pos_weight']:.3f}")

            # Step 4: Model Training, Tuning, Calibration & Saving
            logger.info(">>> STEP 4: Model Training, Tuning & Calibration <<<")
            trainer = ModelTrainer(config_path=self.config_path)
            trainer_results = trainer.run(transform_data)
            logger.info(f"Model training and calibration complete. Best threshold: {trainer_results['best_threshold']:.4f}")

            # Step 5: Model Evaluation & Visual Diagrams
            logger.info(">>> STEP 5: Model Evaluation & Visual Diagrams <<<")
            evaluator = ModelEvaluation(config_path=self.config_path)
            evaluation_metrics = evaluator.run(trainer_results, transform_data)
            logger.info("Evaluation metrics and evaluation diagrams saved.")

            # Step 6: SHAP Explainability & Visual Diagrams
            logger.info(">>> STEP 6: SHAP Explainability & Visuals <<<")
            explainer = ModelExplainability(config_path=self.config_path)
            explainability_results = explainer.run(
                champion_pipeline=trainer_results["champion_pipeline"],
                X_test=transform_data["X_test"],
                sample_row_index=5
            )
            logger.info("SHAP global and local explainability visuals saved.")

            logger.info("==================================================")
            logger.info("      TRAINING PIPELINE COMPLETED SUCCESSFULLY    ")
            logger.info("==================================================")

            return {
                "validation_report": validation_report,
                "trainer_results": trainer_results,
                "evaluation_metrics": evaluation_metrics,
                "explainability_results": explainability_results,
            }

        except Exception as e:
            logger.error("Training Pipeline failed during execution")
            raise CustomException("Training Pipeline execution failed", e)


if __name__ == "__main__":
    pipeline = TrainingPipeline()
    pipeline.run()
