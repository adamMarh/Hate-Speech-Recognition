import os
import sys
import io
import keras
import pickle
from PIL import Image
from hate.logger import logging
from hate.exception import CustomException
from hate.constants import *
from keras.utils import pad_sequences
from hate.components.data_transformation import DataTransformation
from hate.entity.config_entity import DataTransformationConfig
from hate.entity.artifact_entity import DataIngestionArtifacts
from hate.configuration.gcloud_syncer import GcloudSync


class PredictionPipeline:
    def __init__(self):
        self.bucket_name = BUCKET_NAME
        self.model_name = MODEL_NAME
        self.model_path = os.path.join("artifacts", "PredictModel")
        self.gcloud = GcloudSync()
        self.data_transformation = DataTransformation(data_transformation_config= DataTransformationConfig, data_ingestion_artifacts=DataIngestionArtifacts)
    
    def get_model_from_gcloud(self) -> str:
        logging.info("Entered the get_model_from_gcloud method in the PredictionPipeline")
        try:
            os.makedirs(self.model_path, exist_ok=True)
            self.gcloud.sync_folder_from_gcloud(self.bucket_name, self.model_name, self.model_path)
            best_model_path = os.path.join(self.model_path, self.model_name)
            logging.info("Exited the get_model_from_gcloud method in the PredictionPipeline class")
            return best_model_path
        except Exception as e:
            raise CustomException(e, sys) from e
        
    def predict(self, best_model_path, text):
        logging.info("Running the prediction function")
        try:
            best_model_path: str = self.get_model_from_gcloud()
            load_model = keras.models.load_model(best_model_path)
            with open('tokenize.pickle', 'rb') as handle:
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
            best_model_path: str = self.get_model_from_gcloud()
            predicted_text = self.predict(best_model_path, text)
            logging.info("Exited the run pipeline method from PredictionPipeline class")
            return predicted_text
        except Exception as e:
            raise CustomException(e, sys) from e
        