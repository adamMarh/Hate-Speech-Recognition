import os
import sys
import io
import shutil
import keras
import pickle
import pandas as pd
from PIL import Image
from hate.logger import logging
from hate.exception import CustomException
from hate.constants import *
from keras.utils import pad_sequences
from keras.preprocessing.text import Tokenizer
from hate.components.data_transformation import DataTransformation
from hate.entity.config_entity import DataTransformationConfig
from hate.entity.artifact_entity import DataIngestionArtifacts
from hate.configuration.gcloud_syncer import GcloudSync


class PredictionPipeline:
    def __init__(self):
        self.bucket_name = BUCKET_NAME
        self.model_name = MODEL_NAME
        self.model_path = os.path.join("artifacts", "PredictModel")
        self.local_model_path = os.path.join(self.model_path, self.model_name)
        self.tokenizer_path = "tokenizer.pickle"
        self.gcloud = GcloudSync()
        self.data_transformation = DataTransformation(data_transformation_config= DataTransformationConfig, data_ingestion_artifacts=DataIngestionArtifacts)

    def get_local_model_path(self) -> str:
        if os.path.exists(self.local_model_path):
            return self.local_model_path

        candidate_paths = []
        artifacts_root = "artifacts"
        if os.path.isdir(artifacts_root):
            for root, _, files in os.walk(artifacts_root):
                if self.model_name in files and root.endswith("ModelTrainerArtifacts"):
                    candidate_paths.append(os.path.join(root, self.model_name))

        if candidate_paths:
            candidate_paths.sort(key=os.path.getmtime, reverse=True)
            os.makedirs(self.model_path, exist_ok=True)
            shutil.copy2(candidate_paths[0], self.local_model_path)
            return self.local_model_path

        return ""

    def get_local_tokenizer_path(self) -> str:
        if os.path.exists(self.tokenizer_path):
            return self.tokenizer_path

        candidate_paths = []
        artifacts_root = "artifacts"
        if os.path.isdir(artifacts_root):
            for root, _, files in os.walk(artifacts_root):
                if "x_train.csv" in files and root.endswith("ModelTrainerArtifacts"):
                    candidate_paths.append(os.path.join(root, "x_train.csv"))

        if not candidate_paths:
            return ""

        candidate_paths.sort(key=os.path.getmtime, reverse=True)
        training_frame = pd.read_csv(candidate_paths[0], index_col=False)
        if "tweet" in training_frame.columns:
            texts = training_frame["tweet"].fillna("").astype(str)
        else:
            texts = training_frame.iloc[:, -1].fillna("").astype(str)

        tokenizer = Tokenizer(num_words=MAX_WORDS)
        tokenizer.fit_on_texts(texts)
        with open(self.tokenizer_path, "wb") as handle:
            pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return self.tokenizer_path
    
    def get_model_from_gcloud(self) -> str:
        logging.info("Entered the get_model_from_gcloud method in the PredictionPipeline")
        try:
            os.makedirs(self.model_path, exist_ok=True)
            self.gcloud.sync_folder_from_gcloud(self.bucket_name, self.model_name, self.model_path)
            best_model_path = self.local_model_path
            logging.info("Exited the get_model_from_gcloud method in the PredictionPipeline class")
            return best_model_path
        except Exception as e:
            raise CustomException(e, sys) from e
        
    def resolve_model_path(self) -> str:
        try:
            return self.get_model_from_gcloud()
        except Exception as gcloud_error:
            logging.warning(
                "Falling back to local prediction model because GCS fetch failed: %s",
                gcloud_error,
            )
            local_model_path = self.get_local_model_path()
            if local_model_path:
                return local_model_path

            raise CustomException(
                FileNotFoundError("No prediction model is available in GCS or locally."),
                sys,
            ) from gcloud_error

    def predict(self, text):
        logging.info("Running the prediction function")
        try:
            best_model_path: str = self.resolve_model_path()
            load_model = keras.models.load_model(best_model_path)
            tokenizer_path = self.get_local_tokenizer_path()
            if not tokenizer_path:
                raise FileNotFoundError("No tokenizer.pickle file or training data was found to rebuild one.")

            with open(tokenizer_path, 'rb') as handle:
                load_tokenizer = pickle.load(handle)
            
            text = self.data_transformation.concat_data_cleaning(text)
            text = [text]
            print(text)
            seq = load_tokenizer.texts_to_sequences(text)
            padded = pad_sequences(seq, maxlen=300)
            print(seq)
            pred = load_model.predict(padded)
            print("pred", pred)
            if pred > 0.3:
                print("hate and abusive")
                return "hate and abusive"
            else:
                print("no hate")
                return "no hate"
        except Exception as e:
            raise CustomException(e, sys) from e
        
    def run_pipeline(self, text):
        logging.info("Entered the run_pipeline method of PredictionPipeline class")
        try:
            predicted_text = self.predict(text)
            logging.info("Exited the run pipeline method from PredictionPipeline class")
            return predicted_text
        except Exception as e:
            raise CustomException(e, sys) from e
        