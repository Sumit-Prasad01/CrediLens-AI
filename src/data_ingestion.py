import shutil
from pathlib import Path
import pandas as pd
import kagglehub

from config.constants import CONFIG_FILE_PATH, RAW_DATA_PATH
from utils.custom_exception import CustomException
from utils.helpers import read_yaml, ensure_dir
from utils.logger import get_logger

logger = get_logger(__name__)


class DataIngestion:
    """Download and store the raw credit risk dataset."""

    def __init__(self, config_path: Path = CONFIG_FILE_PATH) -> None:
        try:
            self.config = read_yaml(config_path) if Path(config_path).exists() else {}
            ingestion_cfg = self.config.get("data_ingestion", {})
            self.dataset_id = ingestion_cfg.get("dataset_id", "laotse/credit-risk-dataset")
            self.output_dir = Path(ingestion_cfg.get("raw_dir", "artifacts/raw"))
            self.file_name = ingestion_cfg.get("file_name", "credit_risk_dataset.csv")
            self.destination_path = self.output_dir / self.file_name
        except Exception as e:
            logger.error("Failed to initialize DataIngestion component")
            raise CustomException("Initialization error in DataIngestion", e)

    def download_dataset(self) -> Path:
        """Download dataset from Kaggle if not already present."""
        try:
            ensure_dir(self.output_dir)

            if self.destination_path.exists() and self.destination_path.stat().st_size > 0:
                logger.info(f"Dataset already exists at: {self.destination_path.resolve()}. Skipping download.")
                return self.destination_path

            logger.info(f"Downloading dataset from Kaggle: {self.dataset_id}")
            downloaded_path = Path(kagglehub.dataset_download(self.dataset_id))

            csv_files = list(downloaded_path.rglob("*.csv"))
            if not csv_files:
                raise FileNotFoundError(f"No CSV files found in {downloaded_path}")

            source_path = csv_files[0]
            shutil.copy2(source_path, self.destination_path)
            logger.info(f"Dataset copied to: {self.destination_path.resolve()}")

            return self.destination_path
        except Exception as e:
            logger.error("Error during dataset download")
            raise CustomException("Failed to download dataset", e)

    def validate_dataset(self, csv_path: Path) -> pd.DataFrame:
        """Load and perform basic validation of the raw dataset."""
        try:
            df = pd.read_csv(csv_path)
            if df.empty:
                raise ValueError("The downloaded CSV file is empty.")

            logger.info(f"Dataset successfully loaded. Shape: {df.shape}")
            logger.info(f"Columns present: {df.columns.tolist()}")
            return df
        except Exception as e:
            logger.error("Error validating downloaded raw dataset")
            raise CustomException("Failed to validate raw dataset", e)

    def run(self) -> Path:
        """Run the complete ingestion pipeline and return path to raw CSV."""
        try:
            csv_path = self.download_dataset()
            self.validate_dataset(csv_path)
            return csv_path
        except Exception as e:
            logger.error("Data Ingestion execution failed")
            raise CustomException("Data Ingestion failed", e)


if __name__ == "__main__":
    ingestion = DataIngestion()
    ingestion.run()
