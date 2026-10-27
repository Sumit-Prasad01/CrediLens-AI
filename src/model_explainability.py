from pathlib import Path
from typing import Any, Dict, Tuple, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from config.constants import CONFIG_FILE_PATH, VISUALS_DIR
from utils.custom_exception import CustomException
from utils.helpers import read_yaml, ensure_dir
from utils.logger import get_logger

logger = get_logger(__name__)


class ModelExplainability:
    """SHAP (SHapley Additive exPlanations) Global and Local interpretability generator."""

    def __init__(self, config_path: Path = CONFIG_FILE_PATH) -> None:
        try:
            self.config = read_yaml(config_path) if Path(config_path).exists() else {}
            self.visuals_dir = Path(VISUALS_DIR)
            ensure_dir(self.visuals_dir)
        except Exception as e:
            logger.error("Initialization error in ModelExplainability")
            raise CustomException("Failed to initialize ModelExplainability", e)

    def extract_tree_model_and_features(
        self, champion_pipeline: Any, X_test: pd.DataFrame
    ) -> Tuple[Any, pd.DataFrame, list]:
        """Extract the trained classifier, transformed test features, and column names."""
        try:
            # Handle Pipeline or CalibratedClassifierCV
            if hasattr(champion_pipeline, "named_steps"):
                preprocessor = champion_pipeline.named_steps["preprocessor"]
                classifier = champion_pipeline.named_steps["classifier"]
            elif hasattr(champion_pipeline, "estimator") and hasattr(champion_pipeline.estimator, "named_steps"):
                preprocessor = champion_pipeline.estimator.named_steps["preprocessor"]
                classifier = champion_pipeline.estimator.named_steps["classifier"]
            elif hasattr(champion_pipeline, "calibrated_classifiers_"):
                base_est = champion_pipeline.calibrated_classifiers_[0].estimator
                preprocessor = base_est.named_steps["preprocessor"]
                classifier = base_est.named_steps["classifier"]
            else:
                raise ValueError("Could not extract preprocessor and classifier from the champion model.")

            X_test_transformed = preprocessor.transform(X_test)
            if hasattr(X_test_transformed, "toarray"):
                X_test_transformed = X_test_transformed.toarray()

            feature_names = preprocessor.get_feature_names_out().tolist()
            # Clean up feature names for cleaner plots (e.g. "num__person_age" -> "person_age", "cat__" -> "")
            clean_names = [
                name.replace("num__", "").replace("cat__", "") for name in feature_names
            ]

            X_test_df = pd.DataFrame(X_test_transformed, columns=clean_names, index=X_test.index)
            return classifier, X_test_df, clean_names

        except Exception as e:
            logger.error("Error extracting tree model and features for SHAP")
            raise CustomException("Feature extraction for SHAP failed", e)

    def generate_global_shap_visuals(
        self, explainer: Any, shap_values: np.ndarray, X_test_df: pd.DataFrame
    ) -> None:
        """Generate and save SHAP summary beeswarm and feature importance bar plots."""
        try:
            logger.info("Generating SHAP global explanation plots...")

            # 1. Summary Beeswarm Plot
            plt.figure(figsize=(10, 7))
            shap.summary_plot(shap_values, X_test_df, show=False)
            plt.title("SHAP Global Feature Impact (Beeswarm)", fontsize=13, weight="bold")
            plt.tight_layout()
            summary_path = self.visuals_dir / "shap_summary_plot.png"
            plt.savefig(summary_path, dpi=300, bbox_inches="tight")
            plt.close()
            logger.info(f"Saved SHAP summary plot to: {summary_path}")

            # 2. Feature Importance Bar Plot
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, X_test_df, plot_type="bar", show=False)
            plt.title("SHAP Mean |Importance| Across Features", fontsize=13, weight="bold")
            plt.tight_layout()
            importance_path = self.visuals_dir / "shap_importance_bar.png"
            plt.savefig(importance_path, dpi=300, bbox_inches="tight")
            plt.close()
            logger.info(f"Saved SHAP feature importance bar plot to: {importance_path}")

        except Exception as e:
            logger.error("Failed to generate global SHAP plots")
            raise CustomException("Global SHAP plotting failed", e)

    def generate_local_waterfall_plot(
        self,
        explainer: Any,
        shap_values: np.ndarray,
        X_test_df: pd.DataFrame,
        feature_names: list,
        row_index: int = 5
    ) -> None:
        """Generate and save SHAP waterfall plot for a specific applicant sample."""
        try:
            logger.info(f"Generating SHAP local waterfall plot for applicant sample at index {row_index}...")

            expected_value = explainer.expected_value
            if isinstance(expected_value, (list, np.ndarray)):
                expected_value = expected_value[1] if len(expected_value) > 1 else expected_value[0]

            values_row = shap_values[row_index]

            explanation = shap.Explanation(
                values=values_row,
                base_values=expected_value,
                data=X_test_df.iloc[row_index].values,
                feature_names=feature_names
            )

            plt.figure(figsize=(10, 7))
            shap.plots.waterfall(explanation, show=False)
            plt.title(f"SHAP Local Explanation (Applicant #{row_index})", fontsize=13, weight="bold")
            plt.tight_layout()
            waterfall_path = self.visuals_dir / "shap_waterfall_sample.png"
            plt.savefig(waterfall_path, dpi=300, bbox_inches="tight")
            plt.close()
            logger.info(f"Saved SHAP waterfall diagram to: {waterfall_path}")

        except Exception as e:
            logger.error("Failed to generate local SHAP waterfall diagram")
            raise CustomException("Local SHAP waterfall failed", e)

    def run(
        self, champion_pipeline: Any, X_test: pd.DataFrame, sample_row_index: int = 5
    ) -> Dict[str, Any]:
        """Execute SHAP explainability workflow and save all diagrams."""
        try:
            classifier, X_test_df, feature_names = self.extract_tree_model_and_features(
                champion_pipeline, X_test
            )

            logger.info("Computing SHAP values using TreeExplainer...")
            explainer = shap.TreeExplainer(classifier)
            shap_values = explainer.shap_values(X_test_df)

            # Generate global diagrams
            self.generate_global_shap_visuals(explainer, shap_values, X_test_df)

            # Generate local waterfall diagram
            self.generate_local_waterfall_plot(
                explainer, shap_values, X_test_df, feature_names, row_index=sample_row_index
            )

            # Compute top 5 influential features globally
            mean_abs_shap = np.abs(shap_values).mean(axis=0)
            feature_imp_df = pd.DataFrame({
                "feature": feature_names,
                "importance": mean_abs_shap
            }).sort_values(by="importance", ascending=False)

            top_features = feature_imp_df.head(5).to_dict(orient="records")
            logger.info(f"Top 5 SHAP influential features: {top_features}")

            return {
                "top_features": top_features,
                "feature_importance": feature_imp_df.to_dict(orient="records"),
            }

        except Exception as e:
            logger.error("Model Explainability workflow failed")
            raise CustomException("Model Explainability failed", e)


if __name__ == "__main__":
    logger.info("ModelExplainability module loaded.")
