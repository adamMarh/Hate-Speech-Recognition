import os
import sys
import keras
import pickle
import numpy as np
import pandas as pd

from hate.logger import logging
from hate.exception import CustomException

from keras.utils import pad_sequences

from hate.constants import *
from hate.ml.model import ModelArchitecture
from hate.configuration.gcloud_syncer import GcloudSync
from keras.preprocessing.text import Tokenizer
from sklearn.metrics import confusion_matrix
from hate.entity.config_entity import ModelEvaluationConfig
from hate.entity.artifact_entity import ModelEvaluationArtifacts, ModelTrainerArtifacts, DataTransformationArtifacts



class ModelEvaluation:
    def __init__(self, model_evaluation_config: ModelEvaluationConfig,
                 model_trainer_artifacts: ModelTrainerArtifacts,
                 data_transformation_artifacts: DataTransformationArtifacts):
        self.model_evaluation_config = model_evaluation_config
        self.model_trainer_artifacts = model_trainer_artifacts
        self.data_transformation_artifacts = data_transformation_artifacts
        self.gcloud = GcloudSync()
        
    def get_best_model_from_gcloud(self) -> str:
        try:
            logging.info("Enteres the get_best_model_from_gcloud method from ModelEvaluation")
            os.makedirs(self.model_evaluation_config.BEST_MODEL_DIR_PATH, exist_ok= True)
            
            self.gcloud.sync_folder_from_gcloud(self.model_evaluation_config.BUCKET_NAME,
                                                self.model_evaluation_config.MODEL_NAME,
                                                self.model_evaluation_config.BEST_MODEL_DIR_PATH)
            
            best_model_path = os.path.join(self.model_evaluation_config.BEST_MODEL_DIR_PATH,
                                           self.model_evaluation_config.MODEL_NAME)
            
            logging.info("Exited the get_best_model_from_gcloud method")
            return best_model_path
        except Exception as e:
            raise CustomException(e, sys) from e
    
    def evaluate(self):
        try:
            logging.info("Entering the evaluate method of the ModelEvaluation class")
            print(self.model_trainer_artifacts.x_test_path)
            
            x_test = pd.read_csv(self.model_trainer_artifacts.x_test_path, index_col=0)
            print(x_test)
            y_test = pd.read_csv(self.model_trainer_artifacts.y_test_path, index_col=0)
            
            with open('tokenizer.picke', 'rb') as handle:
                tokenizer = pickle.load(handle)
            
            load_model = keras.models.load_model(self.model_trainer_artifacts.trained_model_path)
            
            x_test = x_test['tweet'].astype(str)
            
            x_test = x_test.squeeze()
            y_test = y_test.squeeze()
            
            test_sequences = tokenizer.texts_to_sequences(x_test)
            test_sequences_matrix = pad_sequences(test_sequences, maxlen=MAX_LEN)
            print(f"------------------{test_sequences_matrix}------------------")
            print(f"------------------{x_test.shape}------------------")
            print(f"------------------{y_test.shape}------------------")
            
            accuracy = load_model.evaluate(test_sequences_matrix, y_test)
            logging.info(f"the test accuracy is {accuracy}")
            lstm_prediction = load_model.predict(test_sequences_matrix)
            res = []
            for prediction in lstm_prediction:
                if prediction[0] < 0.5:
                    res.append()
                else:
                    res.append(1)
            print(confusion_matrix(y_test, res))
            logging.info(f"The confusion matrix is : {confusion_matrix(y_test, res)}")
            return accuracy
        except Exception as e:
            raise CustomException(e, sys) from e
    
    def initiate_model_evaluation(self) -> ModelEvaluationArtifacts:
        try:
            logging.info("Initiating the model evaluation. Loading currently trained model")
            trained_model=keras.models.load_model(self.model_trainer_artifacts.trained_model_path)
            with open('tokenizer.pickle', 'rb') as handle:
                load_tokenizer = pickle.load(handle)
            
            trained_model_accuracy = self.evaluate()
            
            logging.info("Fetch best model from gcloud storage")
            best_model_path = self.get_best_model_from_gcloud()
            
            logging.info("Checking if the best model is present in gcloud")
            if os.path.isfile(best_model_path) is False:
                is_model_accepted = True
                logging.info("Gcloud model is no longer the best model, the one currently trained is.")
            
            else:
                logging.info("Load best model fetched from gcloud storage")
                best_model = keras.models.load_model(best_model_path)
                best_model_accuracy = self.evaluate()
                
                logging.info("Comparing loss between best_model_loss and trained_model_loss")
                if best_model_accuracy > trained_model_accuracy:
                    is_best_model_accepted = True
                    logging.info("Trained model not accepted")
                else:
                    is_best_model_accepted = False
                    logging.info("Trained model accepted")
                
            model_evaluation_artifacts = ModelEvaluationArtifacts(is_model_accepted)
            logging.info("Returning the ModelEvaluationArtifacts")
            return model_evaluation_artifacts
        except Exception as e:
            raise CustomException(e, sys) from e
    
    