import os
import sys
from zipfile import ZipFile
from hate.logger import logging
from hate.exception import CustomException
from hate.configuration.gcloud_syncer import GcloudSync
from hate.constants import DATA_DIR, DATA_INGESTION_IMBALANCE_DATA, DATA_INGESTION_RAW_DATA
from hate.entity.config_entity import DataIngestionConfig
from hate.entity.artifact_entity import DataIngestionArtifacts


class DataIngestion:
    def __init__(self, data_ingestion_config : DataIngestionConfig):
        self.data_ingestion_config = data_ingestion_config
        self.gcloud = GcloudSync()

    def _local_dataset_paths(self):
        dataset_dir = os.path.join(os.getcwd(), DATA_DIR, "dataset")
        imbalance_data_file_path = os.path.join(dataset_dir, DATA_INGESTION_IMBALANCE_DATA)
        raw_data_file_path = os.path.join(dataset_dir, DATA_INGESTION_RAW_DATA)

        if os.path.exists(imbalance_data_file_path) and os.path.exists(raw_data_file_path):
            return imbalance_data_file_path, raw_data_file_path

        return None

    def get_data_from_gcloud(self) -> None:
        try:
            logging.info("Entered the get_data_from_gcloud method of DataIngestion class")
            os.makedirs(self.data_ingestion_config.DATA_INGESTION_ARTIFACTS_DIR, exist_ok=True)
            self.gcloud.sync_folder_from_gcloud(self.data_ingestion_config.BUCKET_NAME,
                                                self.data_ingestion_config.ZIP_FILE_NAME,
                                                self.data_ingestion_config.DATA_INGESTION_ARTIFACTS_DIR
                                                )
            logging.info("Exit the get_data_from_gcloud method of DataIngestion class")
        except Exception as e:
                                                raise CustomException(e, sys) from e

    def unzip_and_clean(self):
        logging.info("Entered the unzip_and_clean method of DataIngestion class")
        try:
            with ZipFile(self.data_ingestion_config.ZIP_FILE_PATH, 'r') as zip_ref:
                zip_ref.extractall(self.data_ingestion_config.ZIP_FILE_DIR)
                logging.info("Exiting the unzip_and_clean method from DataIngestion class")
                return self.data_ingestion_config.DATA_ARTIFACTS_DIR, self.data_ingestion_config.NEW_DATA_ARTIFACTS_DIR
        except Exception as e:
            raise CustomException(e, sys) from e
    
    def initiate_data_ingestion(self) -> DataIngestionArtifacts:
        logging.info("Entered the initiate_data_ingestion method of DataIngestio class")

        try:
            try:
                self.get_data_from_gcloud()
                logging.info("Fetched data from the gcloud bucket")
                imbalance_data_file_path, raw_data_file_path = self.unzip_and_clean()
            except Exception as gcloud_error:
                logging.warning(
                    "Gcloud ingestion failed, falling back to local dataset files: %s",
                    gcloud_error,
                )
                local_dataset_paths = self._local_dataset_paths()
                if local_dataset_paths is None:
                    raise

                logging.info("Using local dataset files bundled with the repository")
                imbalance_data_file_path, raw_data_file_path = local_dataset_paths

            data_ingestion_artifacts = DataIngestionArtifacts(
                imbalance_data_file_path,
                raw_data_file_path
            )

            logging.info("Exit the intitiate_data_ingestion method from the DataIngestion class")
            logging.info(f"Data ingestion artifact: {data_ingestion_artifacts}")

            return data_ingestion_artifacts

        except Exception as e:
            raise CustomException(e, sys) from e