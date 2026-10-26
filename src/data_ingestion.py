from pathlib import Path
import shutil

import kagglehub
import pandas as pd

from utils.custom_exception import CustomException
from utils.logger import get_logger

logger = get_logger(__name__)

class DataIngestion:
    """Download and store the raw credit risk dataset."""

    def __init__(
        self,
        dataset_id: str = "laotse/credit-risk-dataset",
        output_dir: str = "artifacts/raw",
    ) -> None:
        self.dataset_id = dataset_id
        self.output_dir = Path(output_dir)
        self.file_name = "credit_risk_dataset.csv"

    def download_dataset(self) -> Path:
        """Download dataset files from Kaggle."""

        logger.info(f"Downloading dataset: {self.dataset_id}")

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Download dataset files to KaggleHub's cache
        downloaded_path = Path(
            kagglehub.dataset_download(self.dataset_id)
        )

        # Find CSV files in the downloaded directory
        csv_files = list(downloaded_path.rglob("*.csv"))

        if not csv_files:
            raise FileNotFoundError(
                f"No CSV files found in {downloaded_path}"
            )

        if len(csv_files) > 1:
            logger.info("Available CSV files:")
            for index, file in enumerate(csv_files, start=1):
                logger.info(f"{index}. {file.name}")

            raise ValueError(
                "Multiple CSV files found. Select the required CSV "
                "file explicitly."
            )

        source_path = csv_files[0]
        destination_path = self.output_dir / self.file_name

        # Copy the CSV into the project artifacts directory
        shutil.copy2(source_path, destination_path)

        logger.info(f"Dataset saved to: {destination_path.resolve()}")

        return destination_path

    def validate_dataset(self, csv_path: Path) -> pd.DataFrame:
        """Load and perform basic validation of the raw dataset."""

        df = pd.read_csv(csv_path)

        if df.empty:
            raise ValueError("The downloaded CSV is empty.")

        logger.info(f"\nDataset shape: {df.shape}")
        logger.info(f"Columns: {df.columns.tolist()}")
        logger.info("\nFirst 5 records:")
        logger.info(df.head())

        return df

    def run(self) -> pd.DataFrame:
        """Run the complete ingestion pipeline."""

        csv_path = self.download_dataset()
        return self.validate_dataset(csv_path)


if __name__ == "__main__":
    ingestion = DataIngestion()
    ingestion.run()
