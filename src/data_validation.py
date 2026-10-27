import os
from pathlib import Path
from typing import Dict, Any, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from config.constants import (
    CONFIG_FILE_PATH,
    RAW_DATA_PATH,
    CLEANED_DATA_PATH,
    VISUALS_DIR,
    PROCESSED_DATA_DIR,
)
from utils.custom_exception import CustomException
from utils.helpers import read_yaml, ensure_dir
from utils.logger import get_logger

logger = get_logger(__name__)


class DataValidation:
    """Validate raw credit risk data, generate EDA visuals, and clean outliers."""

    def __init__(self, config_path: Path = CONFIG_FILE_PATH) -> None:
        try:
            self.config = read_yaml(config_path) if Path(config_path).exists() else {}
            val_cfg = self.config.get("data_validation", {})
            self.min_age = val_cfg.get("min_age", 18)
            self.max_age = val_cfg.get("max_age", 100)
            self.max_emp_length = val_cfg.get("max_emp_length", 60)
            self.min_loan_amnt = val_cfg.get("min_loan_amnt", 1)

            self.visuals_dir = Path(self.config.get("artifacts", {}).get("visuals_dir", VISUALS_DIR))
            self.cleaned_data_path = Path(CLEANED_DATA_PATH)
            ensure_dir(self.visuals_dir)
            ensure_dir(self.cleaned_data_path.parent)
        except Exception as e:
            logger.error("Initialization error in DataValidation")
            raise CustomException("Failed to initialize DataValidation", e)

    def generate_eda_visuals(self, df: pd.DataFrame) -> None:
        """Generate and save EDA plots into artifacts/visuals/."""
        try:
            logger.info("Generating EDA visual artifacts...")
            sns.set_theme(style="whitegrid")

            # 1. Target Balance Plot
            plt.figure(figsize=(7, 5))
            ax = sns.countplot(
                x="loan_status",
                data=df,
                palette=["#2ecc71", "#e74c3c"],
                hue="loan_status",
                legend=False
            )
            plt.title("Loan Status Distribution (0 = Non-Default, 1 = Default)", fontsize=13, weight="bold")
            plt.xlabel("Loan Status", fontsize=11)
            plt.ylabel("Applicant Count", fontsize=11)

            total = len(df)
            for p in ax.patches:
                count = int(p.get_height())
                pct = 100 * count / total
                ax.annotate(
                    f"{count:,}\n({pct:.1f}%)",
                    (p.get_x() + p.get_width() / 2.0, count / 2),
                    ha="center",
                    va="center",
                    color="white",
                    weight="bold",
                    fontsize=11
                )
            target_plot_path = self.visuals_dir / "target_distribution.png"
            plt.tight_layout()
            plt.savefig(target_plot_path, dpi=300)
            plt.close()
            logger.info(f"Saved target distribution diagram to {target_plot_path}")

            # 2. Numerical Features Boxplots (Outlier Check)
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if "loan_status" in numeric_cols:
                numeric_cols.remove("loan_status")

            n_cols = len(numeric_cols)
            fig, axes = plt.subplots(nrows=(n_cols + 2) // 3, ncols=3, figsize=(15, 4 * ((n_cols + 2) // 3)))
            axes = axes.flatten()

            for idx, col in enumerate(numeric_cols):
                sns.boxplot(x=df[col], ax=axes[idx], color="#3498db")
                axes[idx].set_title(f"Distribution: {col}", fontsize=11, weight="bold")
                axes[idx].set_xlabel("")

            # Hide any unused subplots
            for j in range(idx + 1, len(axes)):
                fig.delaxes(axes[j])

            plt.tight_layout()
            outlier_plot_path = self.visuals_dir / "numerical_outliers_boxplot.png"
            plt.savefig(outlier_plot_path, dpi=300)
            plt.close()
            logger.info(f"Saved numerical outliers boxplots to {outlier_plot_path}")

            # 3. Correlation Heatmap
            plt.figure(figsize=(10, 8))
            corr = df.corr(numeric_only=True)
            sns.heatmap(
                corr,
                annot=True,
                fmt=".2f",
                cmap="coolwarm",
                vmin=-1,
                vmax=1,
                linewidths=0.5,
                square=True,
                cbar_kws={"shrink": 0.8}
            )
            plt.title("Correlation Heatmap of Numerical Features", fontsize=14, weight="bold")
            plt.tight_layout()
            corr_plot_path = self.visuals_dir / "correlation_heatmap.png"
            plt.savefig(corr_plot_path, dpi=300)
            plt.close()
            logger.info(f"Saved correlation heatmap to {corr_plot_path}")

        except Exception as e:
            logger.error("Failed to generate EDA visual artifacts")
            raise CustomException("Failed to generate EDA visuals", e)

    def clean_and_validate(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Clean dataset and handle business logic outliers."""
        try:
            initial_rows = len(df)
            duplicates_count = int(df.duplicated().sum())

            # 1. Remove duplicate rows
            cleaned_df = df.drop_duplicates().copy()

            # 2. Age boundaries [min_age, max_age]
            cleaned_df = cleaned_df[
                (cleaned_df["person_age"] >= self.min_age) & (cleaned_df["person_age"] <= self.max_age)
            ]

            # 3. Employment length validation
            # Keep rows where emp_length <= age and emp_length <= max_emp_length
            # If emp_length is null, retain for median imputation during preprocessing or filter invalid
            cleaned_df = cleaned_df[
                cleaned_df["person_emp_length"].isna()
                | (
                    (cleaned_df["person_emp_length"] <= cleaned_df["person_age"])
                    & (cleaned_df["person_emp_length"] <= self.max_emp_length)
                )
            ]

            # 4. Positive loan amount
            cleaned_df = cleaned_df[cleaned_df["loan_amnt"] >= self.min_loan_amnt]

            final_rows = len(cleaned_df)
            removed_rows = initial_rows - final_rows

            validation_summary = {
                "initial_row_count": initial_rows,
                "duplicates_removed": duplicates_count,
                "total_rows_removed": removed_rows,
                "cleaned_row_count": final_rows,
                "columns_count": cleaned_df.shape[1],
                "missing_values": cleaned_df.isnull().sum().to_dict(),
            }

            logger.info(
                f"Data Cleaning Completed: Initial {initial_rows} rows -> Cleaned {final_rows} rows "
                f"({removed_rows} outliers/duplicates removed)."
            )

            # Save cleaned dataset
            cleaned_df.to_csv(self.cleaned_data_path, index=False)
            logger.info(f"Cleaned dataset saved to: {self.cleaned_data_path.resolve()}")

            return cleaned_df, validation_summary

        except Exception as e:
            logger.error("Error during data validation and cleaning")
            raise CustomException("Failed to validate and clean data", e)

    def run(self, raw_csv_path: Path = RAW_DATA_PATH) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Execute validation and visual generation pipeline."""
        try:
            logger.info(f"Starting data validation from: {raw_csv_path}")
            df = pd.read_csv(raw_csv_path)

            # Generate EDA Visuals on raw data
            self.generate_eda_visuals(df)

            # Clean and validate
            cleaned_df, report = self.clean_and_validate(df)
            return cleaned_df, report

        except Exception as e:
            logger.error("Data Validation run failed")
            raise CustomException("Data Validation failed", e)


if __name__ == "__main__":
    validator = DataValidation()
    validator.run()
