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
        self.tokenizer_path = 'tokenizer.pickle'
        self.model_name = self.model_evaluation_config.MODEL_NAME
        self.best_model_dir = self.model_evaluation_config.BEST_MODEL_DIR_PATH

    def get_local_best_model_path(self) -> str:
        candidate_paths = []
        artifacts_root = 'artifacts'
        if os.path.isdir(artifacts_root):
            for root, _, files in os.walk(artifacts_root):
                if self.model_name in files and (root.endswith('best_model') or root.endswith('ModelTrainerArtifacts')):
                    candidate_paths.append(os.path.join(root, self.model_name))

        if candidate_paths:
            candidate_paths.sort(key=os.path.getmtime, reverse=True)
            return candidate_paths[0]

        return ''

    def get_local_tokenizer_path(self) -> str:
        if os.path.exists(self.tokenizer_path):
            return self.tokenizer_path

        candidate_paths = []
        artifacts_root = 'artifacts'
        if os.path.isdir(artifacts_root):
            for root, _, files in os.walk(artifacts_root):
                if 'x_train.csv' in files and root.endswith('ModelTrainerArtifacts'):
                    candidate_paths.append(os.path.join(root, 'x_train.csv'))

        if not candidate_paths:
            return ''

        candidate_paths.sort(key=os.path.getmtime, reverse=True)
        training_frame = pd.read_csv(candidate_paths[0], index_col=False)
        if 'tweet' in training_frame.columns:
            texts = training_frame['tweet'].fillna('').astype(str)
        else:
            texts = training_frame.iloc[:, -1].fillna('').astype(str)

        tokenizer = Tokenizer(num_words=MAX_WORDS)
        tokenizer.fit_on_texts(texts)
        with open(self.tokenizer_path, 'wb') as handle:
            pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return self.tokenizer_path
        
    def get_best_model_from_gcloud(self) -> str:
        try:
            logging.info("Enteres the get_best_model_from_gcloud method from ModelEvaluation")
            os.makedirs(self.best_model_dir, exist_ok= True)
            
            self.gcloud.sync_folder_from_gcloud(self.model_evaluation_config.BUCKET_NAME,
                                                self.model_evaluation_config.MODEL_NAME,
                                                self.best_model_dir)
            
            best_model_path = os.path.join(self.best_model_dir, self.model_evaluation_config.MODEL_NAME)
            
            logging.info("Exited the get_best_model_from_gcloud method")
            return best_model_path
        except Exception as e:
            raise CustomException(e, sys) from e

    def resolve_best_model_path(self) -> str:
        try:
            return self.get_best_model_from_gcloud()
        except Exception as gcloud_error:
            logging.warning(
                "Falling back to local best model because GCS fetch failed: %s",
                gcloud_error,
            )
            local_best_model_path = self.get_local_best_model_path()
            if local_best_model_path:
                return local_best_model_path

            raise CustomException(
                FileNotFoundError("No best model is available in GCS or locally."),
                sys,
            ) from gcloud_error
    
    def evaluate(self, model_path: str):
        try:
            logging.info("Entering the evaluate method of the ModelEvaluation class")
            print(self.model_trainer_artifacts.x_test_path)
            
            x_test = pd.read_csv(self.model_trainer_artifacts.x_test_path, index_col=0)
            print(x_test)
            y_test = pd.read_csv(self.model_trainer_artifacts.y_test_path, index_col=0)
            
            tokenizer_path = self.get_local_tokenizer_path()
            if not tokenizer_path:
                raise FileNotFoundError("No tokenizer.pickle file or training data was found to rebuild one.")

            with open(tokenizer_path, 'rb') as handle:
                tokenizer = pickle.load(handle)
            
            load_model = keras.models.load_model(model_path)
            
            x_test = x_test['tweet'].astype(str)
            
            x_test = x_test.squeeze()
            y_test = y_test.squeeze()
            
            test_sequences = tokenizer.texts_to_sequences(x_test)
            test_sequences_matrix = pad_sequences(test_sequences, maxlen=MAX_LEN)
            print(f"------------------{test_sequences_matrix}------------------")
            print(f"------------------{x_test.shape}------------------")
            print(f"------------------{y_test.shape}------------------")
            
            accuracy = load_model.evaluate(test_sequences_matrix, y_test, verbose=0)
            if isinstance(accuracy, (list, tuple)):
                accuracy = accuracy[1]
            logging.info(f"the test accuracy is {accuracy}")
            lstm_prediction = load_model.predict(test_sequences_matrix)
            res = []
            for prediction in lstm_prediction:
                if prediction[0] < 0.5:
                    res.append(0)
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
            trained_model_accuracy = self.evaluate(self.model_trainer_artifacts.trained_model_path)
            
            logging.info("Fetch best model from gcloud storage")
            best_model_path = self.resolve_best_model_path()
            
            logging.info("Checking if the best model is present in gcloud or locally")
            if os.path.isfile(best_model_path) is False:
                is_model_accepted = True
                logging.info("No best model was available, so the trained model is accepted.")
            else:
                best_model_accuracy = self.evaluate(best_model_path)
                logging.info("Comparing best_model_accuracy and trained_model_accuracy")
                if trained_model_accuracy >= best_model_accuracy:
                    is_model_accepted = True
                    logging.info("Trained model accepted")
                else:
                    is_model_accepted = False
                    logging.info("Trained model not accepted")

            model_evaluation_artifacts = ModelEvaluationArtifacts(is_model_accepted=is_model_accepted)
            logging.info("Returning the ModelEvaluationArtifacts")
            return model_evaluation_artifacts
        except Exception as e:
            raise CustomException(e, sys) from e
    
    