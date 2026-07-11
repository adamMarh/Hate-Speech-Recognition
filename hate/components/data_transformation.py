import os
import re
import sys
import string
import pandas as pd
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from sklearn.model_selection import train_test_split
from hate.logger import logging
from hate.exception import CustomException
from hate.entity.config_entity import DataTransformationConfig
from hate.entity.artifact_entity import DataIngestionArtifacts, DataTransformationArtifacts


class DataTransformation:
    def __init__(self, data_transformation_config: DataTransformationConfig, data_ingestion_artifacts: DataIngestionArtifacts):
        self.data_transformation_config = data_transformation_config
        self.data_ingestion_artifacts = data_ingestion_artifacts
    
    def imbalance_data_training(self):
        try:
            logging.info("Entered into the imbalance_data_training function")
            imbalance_data = pd.read_csv(self.data_ingestion_artifacts.imbalance_data_file_path)
            imbalance_data.drop(self.data_transformation_config.ID, axis=self.data_transformation_config.AXIS, inplace=self.data_transformation_config.INPLACE)
            inplace = self.data_transformation_config.INPLACE
            logging.info(f"Exiting the imbalance_data_training function and returned imbalance data {imbalance_data}")
            return imbalance_data
        except Exception as e:
            raise CustomException(e, sys) from e
        
    def raw_data_cleaning(self):
        try:
            logging.info("Entering raw_data_cleaning function")
            raw_data = pd.read_csv(self.data_ingestion_artifacts.raw_data_file_path)
            raw_data.drop(self.data_transformation_config.DROP_COLUMNS, axis=self.data_transformation_config.AXIS, inplace=self.data_transformation_config.INPLACE)
            inplace = self.data_transformation_config.INPLACE

            #replace the value of 0 to 1
            raw_data[self.data_transformation_config.CLASS].replace({0:1}, inplace=True)

            #replace the value of 2 to 0
            raw_data[self.data_transformation_config.CLASS].replace({2:0}, inplace=True)

            raw_data.rename(columns={self.data_transformation_config.CLASS: self.data_transformation_config.LABEL}, inplace=True)

            logging.info(f"Exiting the raw_data_cleaning function and returned raw data: {raw_data}")
            return raw_data
        except Exception as e:
            raise CustomException(e, sys) from e
    
    def concat_dataframe(self):
        try:
            logging.info("Entering concat_dataframe function")
            frame = [self.raw_data_cleaning(), self.imbalance_data_training()]
            df = pd.concat(frame)
            print(df.head())
            logging.info(f"Exiting concat_dataframe function and returning concatenated dataframe: {df}")
            return df
        
        except Exception as e:
            raise CustomException(e, sys) from e
    
    def concat_data_cleaning(self, words):
        try:
            logging.info(f"Entered into the concat_data_cleaning function with param: {words}")
            stemmer = nltk.SnowballStemmer("english")
            stopword = set(stopwords.words('english'))
            words = str(words).lower()
            words = re.sub(r'\[.*?\]', '', words)
            words = re.sub(r'https?://\S+|www\.\S+', '', words)
            words = re.sub(r'<.*?>+', '', words)
            words = re.sub(r'[%s]' % re.escape(string.punctuation), '', words)
            words = re.sub(r'\n', '', words)
            words = re.sub(r'\w*\d\w*', '', words)
            words = [word for word in words.split(' ') if word not in stopword]
            words = ' '.join(words)
            words = [stemmer.stem(word) for word in words.split(' ')]
            words = ' '.join(words)
            logging.info(f"Exiting concat_data_cleaning function ")
            return words
        except Exception as e:
            raise CustomException(e, sys) from e
    
    def initiate_data_transformation(self) -> DataTransformationArtifacts:
        try:
            logging.info("Entered initiate_data_transformation function of DataTransformation class")
            df = self.concat_dataframe()
            df[self.data_transformation_config.TWEET]=df[self.data_transformation_config.TWEET].apply(self.concat_data_cleaning)
            
            os.makedirs(self.data_transformation_config.DATA_TRANSFORMATION_ARTIFACTS_DIR, exist_ok=True)
            df.to_csv(self.data_transformation_config.TRANSFORMED_FILE_PATH, index=False, header=True)

            data_transformation_artifacts = DataTransformationArtifacts(
                transformed_data_path=self.data_transformation_config.TRANSFORMED_FILE_PATH
            )
            logging.info("returning the DataTransformationArtifacts from initiate_data_transformation of DataTransformation class")
            return data_transformation_artifacts
        except Exception as e:
            raise CustomException(e,sys) from e
    
