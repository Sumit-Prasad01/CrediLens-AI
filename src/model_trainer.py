from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from scipy.stats import randint, uniform
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_validate, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from config.constants import (
    CONFIG_FILE_PATH,
    MODEL_PATH,
    BEST_THRESHOLD_PATH,
    MODELS_DIR,
)
from utils.custom_exception import CustomException
from utils.helpers import read_yaml, ensure_dir, save_object
from utils.logger import get_logger

logger = get_logger(__name__)


def find_optimal_threshold(y_true: pd.Series | np.ndarray, y_proba: np.ndarray) -> Tuple[float, float, float, float]:
    """Calculate the decision threshold that maximizes the F1-score."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    # Align arrays with thresholds
    p = precisions[:-1]
    r = recalls[:-1]
    f1 = 2 * (p * r) / (p + r + 1e-9)
    best_idx = int(np.argmax(f1))
    best_th = float(thresholds[best_idx])
    return best_th, float(f1[best_idx]), float(p[best_idx]), float(r[best_idx])


def threshold_for_target_recall(
    y_true: pd.Series | np.ndarray, y_proba: np.ndarray, target_recall: float = 0.80
) -> Tuple[float, float, float]:
    """Find threshold that maximizes precision subject to recall >= target_recall."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    p = precisions[:-1]
    r = recalls[:-1]
    eligible = np.where(r >= target_recall)[0]
    if len(eligible) == 0:
        idx = int(np.argmax(r))
    else:
        idx = eligible[np.argmax(p[eligible])]
    return float(thresholds[idx]), float(p[idx]), float(r[idx])


class ModelTrainer:
    """Train baseline and ensemble models, perform CV/tuning, calibrate probabilities, and save the champion model."""

    def __init__(self, config_path: Path = CONFIG_FILE_PATH) -> None:
        try:
            self.config = read_yaml(config_path) if Path(config_path).exists() else {}
            train_cfg = self.config.get("model_training", {})
            self.cv_splits = train_cfg.get("cv_splits", 5)
            self.random_state = train_cfg.get("random_state", 42)
            self.scoring_metric = train_cfg.get("scoring", "average_precision")
            self.target_recall = train_cfg.get("target_recall", 0.80)
            self.tune_hyperparameters = train_cfg.get("tune_hyperparameters", False)
            self.n_iter_search = train_cfg.get("n_iter_search", 25)

            self.xgb_params = train_cfg.get("xgboost_params", {})
            self.lgb_params = train_cfg.get("lightgbm_params", {})

            self.calibration_method = train_cfg.get("calibration_method", "sigmoid")
            self.calibration_cv = train_cfg.get("calibration_cv", 5)

            self.model_path = Path(MODEL_PATH)
            self.threshold_path = Path(BEST_THRESHOLD_PATH)
            ensure_dir(self.model_path.parent)

            self.cv = StratifiedKFold(
                n_splits=self.cv_splits,
                shuffle=True,
                random_state=self.random_state
            )
        except Exception as e:
            logger.error("Initialization error in ModelTrainer")
            raise CustomException("Failed to initialize ModelTrainer", e)

    def build_candidate_pipelines(
        self,
        preprocessor_tree: Any,
        preprocessor_linear: Any,
        scale_pos_weight: float
    ) -> Dict[str, Pipeline]:
        """Build candidate pipelines for Logistic Regression, XGBoost, and LightGBM."""
        try:
            lr_pipe = Pipeline([
                ("preprocessor", preprocessor_linear),
                ("classifier", LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=self.random_state
                ))
            ])

            xgb_clf = XGBClassifier(
                n_estimators=self.xgb_params.get("n_estimators", 557),
                max_depth=self.xgb_params.get("max_depth", 5),
                learning_rate=self.xgb_params.get("learning_rate", 0.07147),
                subsample=self.xgb_params.get("subsample", 0.70525),
                colsample_bytree=self.xgb_params.get("colsample_bytree", 0.98479),
                min_child_weight=self.xgb_params.get("min_child_weight", 5),
                gamma=self.xgb_params.get("gamma", 2.68013),
                scale_pos_weight=scale_pos_weight,
                random_state=self.random_state,
                eval_metric="logloss"
            )
            xgb_pipe = Pipeline([
                ("preprocessor", preprocessor_tree),
                ("classifier", xgb_clf)
            ])

            lgb_clf = LGBMClassifier(
                n_estimators=self.lgb_params.get("n_estimators", 408),
                num_leaves=self.lgb_params.get("num_leaves", 40),
                max_depth=self.lgb_params.get("max_depth", 8),
                learning_rate=self.lgb_params.get("learning_rate", 0.05365),
                subsample=self.lgb_params.get("subsample", 0.81944),
                colsample_bytree=self.lgb_params.get("colsample_bytree", 0.90711),
                min_child_weight=self.lgb_params.get("min_child_weight", 4),
                min_split_gain=self.lgb_params.get("min_split_gain", 1.6032),
                reg_alpha=self.lgb_params.get("reg_alpha", 0.36923),
                reg_lambda=self.lgb_params.get("reg_lambda", 2.27134),
                scale_pos_weight=scale_pos_weight,
                random_state=self.random_state,
                verbosity=-1
            )
            lgb_pipe = Pipeline([
                ("preprocessor", preprocessor_tree),
                ("classifier", lgb_clf)
            ])

            return {
                "Logistic Regression": lr_pipe,
                "XGBoost": xgb_pipe,
                "LightGBM": lgb_pipe
            }

        except Exception as e:
            logger.error("Failed to build candidate model pipelines")
            raise CustomException("Pipeline building failed", e)

    def cross_validate_models(
        self, pipelines: Dict[str, Pipeline], X_train: pd.DataFrame, y_train: pd.Series
    ) -> Dict[str, Dict[str, float]]:
        """Run 5-fold cross-validation on all candidate pipelines."""
        try:
            logger.info("Running 5-fold Stratified Cross-Validation on candidate models...")
            scoring = {
                "roc_auc": "roc_auc",
                "accuracy": "accuracy",
                "precision": "precision",
                "recall": "recall",
                "f1": "f1",
                "average_precision": "average_precision"
            }
            results = {}

            for name, pipe in pipelines.items():
                logger.info(f"Cross-validating: {name}")
                cv_res = cross_validate(
                    pipe, X_train, y_train, cv=self.cv, scoring=scoring, n_jobs=-1
                )
                summary = {
                    "roc_auc": float(cv_res["test_roc_auc"].mean()),
                    "accuracy": float(cv_res["test_accuracy"].mean()),
                    "precision": float(cv_res["test_precision"].mean()),
                    "recall": float(cv_res["test_recall"].mean()),
                    "f1": float(cv_res["test_f1"].mean()),
                    "average_precision": float(cv_res["test_average_precision"].mean())
                }
                results[name] = summary
                logger.info(
                    f"{name} CV Results -> AP: {summary['average_precision']:.4f} | "
                    f"ROC-AUC: {summary['roc_auc']:.4f} | F1: {summary['f1']:.4f} | "
                    f"Prec: {summary['precision']:.4f} | Rec: {summary['recall']:.4f}"
                )

            return results
        except Exception as e:
            logger.error("Cross-validation failed")
            raise CustomException("Model cross-validation error", e)

    def tune_xgboost(
        self, preprocessor_tree: Any, scale_pos_weight: float, X_train: pd.DataFrame, y_train: pd.Series
    ) -> Pipeline:
        """Run hyperparameter search on XGBoost."""
        try:
            logger.info(f"Executing RandomizedSearchCV for XGBoost ({self.n_iter_search} iterations)...")
            base_pipe = Pipeline([
                ("preprocessor", preprocessor_tree),
                ("classifier", XGBClassifier(
                    scale_pos_weight=scale_pos_weight,
                    random_state=self.random_state,
                    eval_metric="logloss"
                ))
            ])
            param_grid = {
                "classifier__n_estimators": randint(150, 600),
                "classifier__max_depth": randint(3, 9),
                "classifier__learning_rate": uniform(0.01, 0.5),
                "classifier__subsample": uniform(0.6, 0.4),
                "classifier__colsample_bytree": uniform(0.6, 0.4),
                "classifier__min_child_weight": randint(1, 10),
                "classifier__gamma": uniform(0, 5)
            }
            search = RandomizedSearchCV(
                base_pipe,
                param_distributions=param_grid,
                n_iter=self.n_iter_search,
                cv=self.cv,
                scoring=self.scoring_metric,
                n_jobs=-1,
                random_state=self.random_state,
                verbose=1
            )
            search.fit(X_train, y_train)
            logger.info(f"Tuning complete. Best score ({self.scoring_metric}): {search.best_score_:.4f}")
            logger.info(f"Best parameters: {search.best_params_}")
            return search.best_estimator_
        except Exception as e:
            logger.error("Hyperparameter tuning for XGBoost failed")
            raise CustomException("XGBoost tuning failed", e)

    def train_and_calibrate(
        self,
        pipelines: Dict[str, Pipeline],
        X_train: pd.DataFrame,
        y_train: pd.Series,
        preprocessor_tree: Any,
        scale_pos_weight: float
    ) -> Tuple[Any, float, Pipeline, Dict[str, Pipeline]]:
        """Fit all models, select the champion (XGBoost), calibrate it, and calculate optimal threshold."""
        try:
            fitted_pipelines: Dict[str, Pipeline] = {}

            # Fit all candidate models on full training split
            for name, pipe in pipelines.items():
                logger.info(f"Fitting {name} on full train dataset...")
                pipe.fit(X_train, y_train)
                fitted_pipelines[name] = pipe

            # If user requested hyperparameter tuning, tune XGBoost
            if self.tune_hyperparameters:
                champion_pipeline = self.tune_xgboost(preprocessor_tree, scale_pos_weight, X_train, y_train)
                fitted_pipelines["XGBoost"] = champion_pipeline
            else:
                champion_pipeline = fitted_pipelines["XGBoost"]

            # Probability Calibration using CalibratedClassifierCV
            logger.info(f"Calibrating champion model using {self.calibration_method} method with {self.calibration_cv}-fold CV...")
            calibrated_model = CalibratedClassifierCV(
                estimator=champion_pipeline,
                method=self.calibration_method,
                cv=self.calibration_cv
            )
            calibrated_model.fit(X_train, y_train)

            # Determine optimal decision threshold on train probabilities
            train_probas = calibrated_model.predict_proba(X_train)[:, 1]
            opt_th, f1_val, p_val, r_val = find_optimal_threshold(y_train, train_probas)
            rec_th, rec_p, rec_r = threshold_for_target_recall(y_train, train_probas, self.target_recall)

            logger.info(
                f"Optimal F1 Threshold: {opt_th:.4f} (Train F1: {f1_val:.4f}, Prec: {p_val:.4f}, Rec: {r_val:.4f})"
            )
            logger.info(
                f"Target Recall ({self.target_recall}) Threshold: {rec_th:.4f} (Prec: {rec_p:.4f}, Rec: {rec_r:.4f})"
            )

            # Save champion calibrated model and best threshold
            save_object(self.model_path, calibrated_model)
            save_object(self.threshold_path, opt_th)

            return calibrated_model, opt_th, champion_pipeline, fitted_pipelines

        except Exception as e:
            logger.error("Failed during model training, calibration, or saving")
            raise CustomException("Model training and calibration failed", e)

    def run(self, transform_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute complete model training workflow."""
        try:
            X_train = transform_data["X_train"]
            y_train = transform_data["y_train"]
            preprocessor_tree = transform_data["preprocessor_tree"]
            preprocessor_linear = transform_data["preprocessor_linear"]
            scale_pos_weight = transform_data["scale_pos_weight"]

            # 1. Build pipelines
            candidate_pipelines = self.build_candidate_pipelines(
                preprocessor_tree, preprocessor_linear, scale_pos_weight
            )

            # 2. Run 5-fold CV
            cv_results = self.cross_validate_models(candidate_pipelines, X_train, y_train)

            # 3. Fit, Calibrate, Optimize Threshold, and Save
            calibrated_model, best_threshold, champion_pipeline, fitted_pipelines = self.train_and_calibrate(
                candidate_pipelines, X_train, y_train, preprocessor_tree, scale_pos_weight
            )

            return {
                "calibrated_model": calibrated_model,
                "best_threshold": best_threshold,
                "champion_pipeline": champion_pipeline,
                "fitted_pipelines": fitted_pipelines,
                "cv_results": cv_results
            }

        except Exception as e:
            logger.error("Model Trainer run failed")
            raise CustomException("Model Trainer failed", e)


if __name__ == "__main__":
    from src.data_transformation import DataTransformation
    dt = DataTransformation()
    data = dt.run()
    trainer = ModelTrainer()
    trainer_res = trainer.run(data)
    logger.info("Model Trainer executed successfully.")
