from pathlib import Path
from typing import Dict, Any, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    classification_report,
)

from config.constants import (
    CONFIG_FILE_PATH,
    VISUALS_DIR,
    METRICS_PATH,
    ERROR_ANALYSIS_PATH,
)
from utils.custom_exception import CustomException
from utils.helpers import read_yaml, ensure_dir, save_json
from utils.logger import get_logger

logger = get_logger(__name__)


class ModelEvaluation:
    """Evaluate candidate and calibrated models, compute metrics, perform error analysis, and save visual diagrams."""

    def __init__(self, config_path: Path = CONFIG_FILE_PATH) -> None:
        try:
            self.config = read_yaml(config_path) if Path(config_path).exists() else {}
            self.visuals_dir = Path(VISUALS_DIR)
            self.metrics_path = Path(METRICS_PATH)
            self.error_analysis_path = Path(ERROR_ANALYSIS_PATH)

            ensure_dir(self.visuals_dir)
            ensure_dir(self.metrics_path.parent)
            ensure_dir(self.error_analysis_path.parent)
        except Exception as e:
            logger.error("Initialization error in ModelEvaluation")
            raise CustomException("Failed to initialize ModelEvaluation", e)

    def evaluate_model_performance(
        self,
        name: str,
        model: Any,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        threshold: float = 0.5
    ) -> Dict[str, Any]:
        """Compute metrics for a model at a specified threshold."""
        try:
            proba = model.predict_proba(X_test)[:, 1]
            preds = (proba >= threshold).astype(int)

            acc = float(accuracy_score(y_test, preds))
            prec = float(precision_score(y_test, preds, zero_division=0))
            rec = float(recall_score(y_test, preds, zero_division=0))
            f1 = float(f1_score(y_test, preds, zero_division=0))
            roc_auc = float(roc_auc_score(y_test, proba))
            ap = float(average_precision_score(y_test, proba))
            brier = float(brier_score_loss(y_test, proba))
            cm = confusion_matrix(y_test, preds).tolist()

            logger.info(
                f"[{name}] (Threshold = {threshold:.2f}) -> "
                f"AP: {ap:.4f} | ROC-AUC: {roc_auc:.4f} | F1: {f1:.4f} | "
                f"Prec: {prec:.4f} | Rec: {rec:.4f} | Brier: {brier:.4f}"
            )

            return {
                "threshold": threshold,
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "roc_auc": roc_auc,
                "average_precision": ap,
                "brier_score": brier,
                "confusion_matrix": cm,
            }
        except Exception as e:
            logger.error(f"Failed to evaluate model {name}")
            raise CustomException(f"Evaluation failed for {name}", e)

    def save_confusion_matrix_plot(
        self, y_test: pd.Series, y_pred: np.ndarray, model_name: str = "Calibrated XGBoost"
    ) -> None:
        """Generate and save annotated confusion matrix plot."""
        try:
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(6, 5))
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=["Approved (0)", "Default Risk (1)"],
                yticklabels=["Actual 0", "Actual 1"]
            )
            plt.title(f"{model_name} Confusion Matrix", fontsize=12, weight="bold")
            plt.xlabel("Predicted Label", fontsize=10)
            plt.ylabel("True Label", fontsize=10)
            plt.tight_layout()

            out_path = self.visuals_dir / "confusion_matrix.png"
            plt.savefig(out_path, dpi=300)
            plt.close()
            logger.info(f"Saved confusion matrix diagram to: {out_path}")
        except Exception as e:
            logger.error("Failed to save confusion matrix diagram")
            raise CustomException("Failed to save confusion matrix", e)

    def save_precision_recall_curve_plot(
        self, y_test: pd.Series, y_proba: np.ndarray, threshold: float
    ) -> None:
        """Generate and save Precision-Recall curve with operating point."""
        try:
            precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
            ap = average_precision_score(y_test, y_proba)

            plt.figure(figsize=(7, 6))
            plt.plot(recalls, precisions, color="#2980b9", lw=2.5, label=f"PR Curve (AP = {ap:.3f})")
            plt.axhline(y_test.mean(), color="gray", linestyle="--", label=f"Baseline ({y_test.mean():.2f})")

            # Find precision and recall at selected threshold
            pred = (y_proba >= threshold).astype(int)
            pt_prec = precision_score(y_test, pred, zero_division=0)
            pt_rec = recall_score(y_test, pred, zero_division=0)
            plt.plot(pt_rec, pt_prec, "ro", markersize=8, label=f"Optimal Thr ({threshold:.2f}): P={pt_prec:.2f}, R={pt_rec:.2f}")

            plt.xlabel("Recall", fontsize=11)
            plt.ylabel("Precision", fontsize=11)
            plt.title("Precision-Recall Curve (Calibrated Champion Model)", fontsize=13, weight="bold")
            plt.legend(loc="lower left")
            plt.grid(alpha=0.3)
            plt.tight_layout()

            out_path = self.visuals_dir / "precision_recall_curve.png"
            plt.savefig(out_path, dpi=300)
            plt.close()
            logger.info(f"Saved Precision-Recall curve to: {out_path}")
        except Exception as e:
            logger.error("Failed to save Precision-Recall curve")
            raise CustomException("Failed to save PR curve", e)

    def save_roc_curve_plot(self, y_test: pd.Series, y_proba: np.ndarray) -> None:
        """Generate and save ROC-AUC curve."""
        try:
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            auc_val = roc_auc_score(y_test, y_proba)

            plt.figure(figsize=(7, 6))
            plt.plot(fpr, tpr, color="#8e44ad", lw=2.5, label=f"ROC Curve (AUC = {auc_val:.3f})")
            plt.plot([0, 1], [0, 1], color="gray", linestyle="--", label="Random Classifier")

            plt.xlabel("False Positive Rate", fontsize=11)
            plt.ylabel("True Positive Rate", fontsize=11)
            plt.title("Receiver Operating Characteristic (ROC) Curve", fontsize=13, weight="bold")
            plt.legend(loc="lower right")
            plt.grid(alpha=0.3)
            plt.tight_layout()

            out_path = self.visuals_dir / "roc_curve.png"
            plt.savefig(out_path, dpi=300)
            plt.close()
            logger.info(f"Saved ROC curve to: {out_path}")
        except Exception as e:
            logger.error("Failed to save ROC curve")
            raise CustomException("Failed to save ROC curve", e)

    def save_model_comparison_pr_curves(
        self, models_dict: Dict[str, Any], X_test: pd.DataFrame, y_test: pd.Series
    ) -> None:
        """Generate side-by-side PR curves comparison across models."""
        try:
            plt.figure(figsize=(8, 6))
            colors = {"Logistic Regression": "#95a5a6", "LightGBM": "#e67e22", "XGBoost": "#27ae60"}

            for name, model in models_dict.items():
                proba = model.predict_proba(X_test)[:, 1]
                ap = average_precision_score(y_test, proba)
                p, r, _ = precision_recall_curve(y_test, proba)
                color = colors.get(name, None)
                plt.plot(r, p, lw=2, color=color, label=f"{name} (AP = {ap:.3f})")

            plt.axhline(y_test.mean(), color="black", linestyle=":", label=f"No-Skill Baseline ({y_test.mean():.2f})")
            plt.xlabel("Recall", fontsize=11)
            plt.ylabel("Precision", fontsize=11)
            plt.title("Model Comparison: Precision-Recall Curves (Test Set)", fontsize=13, weight="bold")
            plt.legend(loc="lower left")
            plt.grid(alpha=0.3)
            plt.tight_layout()

            out_path = self.visuals_dir / "model_comparison_pr_curve.png"
            plt.savefig(out_path, dpi=300)
            plt.close()
            logger.info(f"Saved model comparison PR curves to: {out_path}")
        except Exception as e:
            logger.error("Failed to save model comparison PR curves")
            raise CustomException("Failed to save comparison PR curves", e)

    def save_calibration_curves(
        self, raw_model: Any, calibrated_model: Any, X_test: pd.DataFrame, y_test: pd.Series
    ) -> None:
        """Generate calibration curve comparing raw uncalibrated vs calibrated probabilities."""
        try:
            raw_proba = raw_model.predict_proba(X_test)[:, 1]
            cal_proba = calibrated_model.predict_proba(X_test)[:, 1]

            frac_pos_raw, mean_raw = calibration_curve(y_test, raw_proba, n_bins=10, strategy="quantile")
            frac_pos_cal, mean_cal = calibration_curve(y_test, cal_proba, n_bins=10, strategy="quantile")

            plt.figure(figsize=(7, 6))
            plt.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
            plt.plot(mean_raw, frac_pos_raw, "s-", color="#e74c3c", label=f"Raw XGBoost (Brier={brier_score_loss(y_test, raw_proba):.4f})")
            plt.plot(mean_cal, frac_pos_cal, "o-", color="#2ecc71", label=f"Calibrated XGBoost (Brier={brier_score_loss(y_test, cal_proba):.4f})")

            plt.xlabel("Mean Predicted Probability", fontsize=11)
            plt.ylabel("Fraction of Positives (True Defaults)", fontsize=11)
            plt.title("Probability Calibration Reliability Curve", fontsize=13, weight="bold")
            plt.legend(loc="upper left")
            plt.grid(alpha=0.3)
            plt.tight_layout()

            out_path = self.visuals_dir / "calibration_curve.png"
            plt.savefig(out_path, dpi=300)
            plt.close()
            logger.info(f"Saved calibration curve to: {out_path}")
        except Exception as e:
            logger.error("Failed to save calibration curve")
            raise CustomException("Failed to save calibration curve", e)

    def perform_error_analysis(
        self,
        calibrated_model: Any,
        best_threshold: float,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> Dict[str, Any]:
        """Perform error analysis, isolate False Positives / False Negatives, and save audit CSV."""
        try:
            probas = calibrated_model.predict_proba(X_test)[:, 1]
            preds = (probas >= best_threshold).astype(int)

            result_df = X_test.copy()
            result_df["actual"] = y_test.values
            result_df["predicted_probability"] = np.round(probas, 4)
            result_df["predicted_label"] = preds

            false_positives = result_df[(result_df["actual"] == 0) & (result_df["predicted_label"] == 1)]
            false_negatives = result_df[(result_df["actual"] == 1) & (result_df["predicted_label"] == 0)]

            logger.info(
                f"Error Analysis at Threshold {best_threshold:.2f} -> "
                f"Total Test: {len(result_df)} | False Positives: {len(false_positives)} | "
                f"False Negatives: {len(false_negatives)}"
            )

            # Save full error analysis DataFrame to processed artifacts
            result_df.to_csv(self.error_analysis_path, index=False)
            logger.info(f"Saved error analysis audit records to: {self.error_analysis_path}")

            return {
                "total_test_samples": len(result_df),
                "false_positives_count": len(false_positives),
                "false_negatives_count": len(false_negatives),
                "fp_rate": len(false_positives) / (result_df["actual"] == 0).sum(),
                "fn_rate": len(false_negatives) / (result_df["actual"] == 1).sum(),
            }
        except Exception as e:
            logger.error("Error analysis failed")
            raise CustomException("Failed to perform error analysis", e)

    def run(
        self,
        trainer_results: Dict[str, Any],
        transform_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute complete evaluation workflow, save metrics and visuals."""
        try:
            X_test = transform_data["X_test"]
            y_test = transform_data["y_test"]

            calibrated_model = trainer_results["calibrated_model"]
            best_threshold = trainer_results["best_threshold"]
            champion_pipeline = trainer_results["champion_pipeline"]
            fitted_pipelines = trainer_results["fitted_pipelines"]

            # 1. Evaluate candidate models
            candidate_metrics = {}
            for name, pipe in fitted_pipelines.items():
                candidate_metrics[name] = self.evaluate_model_performance(name, pipe, X_test, y_test, threshold=0.5)

            # 2. Evaluate Calibrated Champion Model at 0.5 and best threshold
            calibrated_default_metrics = self.evaluate_model_performance(
                "Calibrated Champion (Threshold 0.50)", calibrated_model, X_test, y_test, threshold=0.5
            )
            calibrated_opt_metrics = self.evaluate_model_performance(
                f"Calibrated Champion (Threshold {best_threshold:.2f})", calibrated_model, X_test, y_test, threshold=best_threshold
            )

            # 3. Save Visualizations
            cal_proba = calibrated_model.predict_proba(X_test)[:, 1]
            cal_preds = (cal_proba >= best_threshold).astype(int)

            self.save_confusion_matrix_plot(y_test, cal_preds, model_name="Calibrated XGBoost")
            self.save_precision_recall_curve_plot(y_test, cal_proba, threshold=best_threshold)
            self.save_roc_curve_plot(y_test, cal_proba)
            self.save_model_comparison_pr_curves(fitted_pipelines, X_test, y_test)
            self.save_calibration_curves(champion_pipeline, calibrated_model, X_test, y_test)

            # 4. Error Analysis
            error_summary = self.perform_error_analysis(calibrated_model, best_threshold, X_test, y_test)

            # 5. Compile and save all metrics to JSON
            all_metrics = {
                "cv_results": trainer_results.get("cv_results", {}),
                "candidate_models_test": candidate_metrics,
                "calibrated_champion_default_thr": calibrated_default_metrics,
                "calibrated_champion_optimal_thr": calibrated_opt_metrics,
                "error_analysis_summary": error_summary,
            }
            save_json(self.metrics_path, all_metrics)

            return all_metrics

        except Exception as e:
            logger.error("Model Evaluation workflow failed")
            raise CustomException("Model Evaluation failed", e)


if __name__ == "__main__":
    logger.info("ModelEvaluation module loaded.")
