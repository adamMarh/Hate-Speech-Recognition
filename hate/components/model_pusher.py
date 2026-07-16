import sys
from hate.logger import logging
from hate.exception import CustomException
from hate.configuration.gcloud_syncer import GcloudSync
from hate.entity.config_entity import ModelPusherConfig
from hate.entity.artifact_entity import  ModelPusherArtifacts


class ModelPusher:
    def __init__(self, model_pusher_config: ModelPusherConfig):
        self.model_pusher_config = model_pusher_config
        self.gcloud = GcloudSync()
    
    
    def initiate_model_pusher(self) -> ModelPusherArtifacts:
        logging.info("Entered the initiate_model_pusher from ModelPusher class")
        try:
            try:
                self.gcloud.sync_folder_to_gcloud(self.model_pusher_config.BUCKET_NAME,
                                                  self.model_pusher_config.TRAINED_MODEL_PATH,
                                                  self.model_pusher_config.MODEL_NAME)
                logging.info("Uploaded best model to gcloud storage")
            except Exception as sync_error:
                logging.warning("Skipping model upload because GCS sync failed: %s", sync_error)
            
            #Saving the model pusher artifacts
            model_pusher_artifacts = ModelPusherArtifacts(self.model_pusher_config.BUCKET_NAME)
            logging.info("Exited the intiate_model_pusher method and returned model_pusher_artifacts")
            return model_pusher_artifacts
        except Exception as e:
            raise CustomException(e, sys) from e