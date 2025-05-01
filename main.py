from AutoSummaryAI.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from AutoSummaryAI.pipeline.stage_02_data_validation import DataValidationTrainingPipeline
from AutoSummaryAI.pipeline.stage_03_data_transformation import DataTransformationTrainingPipeline
from AutoSummaryAI.pipeline.stage_04_model_trainer import ModelTrainerTrainingPipeline
from AutoSummaryAI.pipeline.stage_05_model_evaluation import ModelEvaluationTrainingPipeline
from AutoSummaryAI.logging import logger

STAGE_NAME = "Data Ingestion stage"
try:
   logger.info(f">>>>>> stage {STAGE_NAME} has initialized <<<<<<") 
   data_ingestion = DataIngestionTrainingPipeline()
   data_ingestion.main()
   logger.info(f">>>>>> stage {STAGE_NAME} has completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e

STAGE_NAME = "Data Validation stage"
try:
   logger.info(f">>>>>> stage {STAGE_NAME} has initialized <<<<<<") 
   data_validation = DataValidationTrainingPipeline()
   data_validation.main()
   logger.info(f">>>>>> stage {STAGE_NAME} has completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e

STAGE_NAME = "Data Transformation stage"
try:
   logger.info(f">>>>>> stage {STAGE_NAME} has initialized <<<<<<") 
   data_transformation = DataTransformationTrainingPipeline()
   data_transformation.main()
   logger.info(f">>>>>> stage {STAGE_NAME} has completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e

STAGE_NAME = "Model Training stage"
try:
   logger.info(f">>>>>> stage {STAGE_NAME} has initialized <<<<<<") 
   model_training = ModelTrainerTrainingPipeline()
   model_training.main()
   logger.info(f">>>>>> stage {STAGE_NAME} has completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e

STAGE_NAME = "Model Evaluation stage"
try:
   logger.info(f">>>>>> stage {STAGE_NAME} has initialized <<<<<<") 
   model_evaluation = ModelEvaluationTrainingPipeline()
   model_evaluation.main()
   logger.info(f">>>>>> stage {STAGE_NAME} has completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e