from AutoSummaryAI.config.configuration import ConfigurationManager
from AutoSummaryAI.components.model_evaluation import ModelEvaluation
from AutoSummaryAI.logging import logger




class ModelEvaluationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        model_evaluation_config = config.get_model_evaluation_config()
        model_evaluation = ModelEvaluation(config=model_evaluation_config)
        model_evaluation.evaluate()