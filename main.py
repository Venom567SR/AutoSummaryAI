from AutoSummaryAI.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
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