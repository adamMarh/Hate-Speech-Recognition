import os
import sys
import pickle

import pandas as pd
from hate.logger import logging
from hate.constants import *
from hate.exception import CustomException

from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from hate.entity.config_entity import ModelTrainerConfig
from hate.entity.artifact_entity import ModelTrainerArtifacts, DataTransformationArtifacts
from hate.ml.model import ModelArchitecture


class ModelTrainer:
    def __init__(self, data_transformation_artifacts: DataTransformationArtifacts, model_trainer_config: ModelTrainerConfig):
        self.data_transformation_artifacts = data_transformation_artifacts
        self.model_trainer_config = model_trainer_config
    
    def splitting_data(self, csv_path):
        try:
            logging.info("Entering the splitting_data function of ModelTrainer")
            logging.info("Reading the data")
            df = pd.read_csv(csv_path, index_col=False)
            logging.info("Splitting the data  into x and y")
            x = df[TWEET].fillna("").astype(str)
            y = df[LABEL]
            logging.info("Applying train_test_split on the data")
            x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3, random_state=RANDOM_STATE)
            print(len(x_train), len(y_train))
            print(len(x_test), len(y_test))
            print(type(x_train), type(y_train))
            logging.info("Exiting the splitting_data function")
            return x_train, x_test, y_train, y_test
        except Exception as e:
            raise CustomException(e, sys) from e
    
    def tokeninzing(self, x_train):
        try:
            logging.info("Entering the tokeninzing function from the ModelTrainer")
            tokenizer = Tokenizer(num_words=self.model_trainer_config.MAX_WORDS)
            tokenizer.fit_on_texts(x_train)
            sequences = tokenizer.texts_to_sequences(x_train)
            logging.info(f"Converting text to sequences: {sequences}")
            sequences_matrix = pad_sequences(sequences, maxlen=self.model_trainer_config.MAX_LEN)
            logging.info(f"The sequence matrix is: {sequences_matrix}")
            return sequences_matrix, tokenizer
        except Exception as e:
            raise CustomException(e, sys) from e
    
    def initiate_model_trainer(self) -> ModelTrainerArtifacts:
        try:
            logging.info("Entering the initiate_model_trainer function")
            x_train, x_test, y_train, y_test = self.splitting_data(self.data_transformation_artifacts.transformed_data_path)
            model_architecture = ModelArchitecture()
            
            model = model_architecture.get_model()
            
            logging.info(f"X_train size is {len(x_train)}")
            logging.info(f"X_test size is {len(x_test)}")
            
            sequences_matrix, tokenizer = self.tokeninzing(x_train=x_train)
            
            logging.info("Entering the model training")
            model.fit(sequences_matrix, y_train, batch_size=self.model_trainer_config.BATCH_SIZE,
                      epochs=self.model_trainer_config.EPOCH, validation_split=self.model_trainer_config.VALIDATION_SPLIT)
            logging.info("Model training finished")
            
            with open('tokenizer.pickle', 'wb') as handle:
                pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
            os.makedirs(self.model_trainer_config.TRAINED_MODEL_DIR, exist_ok=True)
            
            logging.info("Saving the model")
            model.save(self.model_trainer_config.TRAINED_MODEL_PATH)
            x_test.to_csv(self.model_trainer_config.X_TEST_DATA_PATH)
            x_train.to_csv(self.model_trainer_config.X_TRAIN_DATA_PATH)
            y_test.to_csv(self.model_trainer_config.Y_TEST_DATA_PATH)
            
            model_trainer_artifacts = ModelTrainerArtifacts(
                trained_model_path=self.model_trainer_config.TRAINED_MODEL_PATH,
                x_test_path= self.model_trainer_config.X_TEST_DATA_PATH,
                y_test_path= self.model_trainer_config.Y_TEST_DATA_PATH,
            )
            logging.info("Returning the ModelTrainerArtifacts")
            return model_trainer_artifacts
        except Exception as e:
            raise CustomException(e, sys) from e
