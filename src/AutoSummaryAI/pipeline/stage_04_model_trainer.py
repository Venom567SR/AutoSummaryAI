from AutoSummaryAI.config.configuration import ConfigurationManager
from AutoSummaryAI.components.model_trainer import ModelTrainer
from AutoSummaryAI.logging import logger
import torch
import gc
import os


class ModelTrainerTrainingPipeline:
    def __init__(self):
        pass

    def main(self):    
        # First make sure the rouge-score is installed
        try:
            import rouge_score
        except ImportError:
            import subprocess
            import sys
            print("Installing rouge-score...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "rouge-score"])
            
        # Force aggressive garbage collection and empty CUDA cache
        gc.collect()
        torch.cuda.empty_cache()
            
        # Set memory efficient options
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:32"
            
        # Check available GPU memory
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
            print(f"Currently allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
            print(f"Max allocated: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
            
        # Fix potential deadlocks in dataloader
        torch.multiprocessing.set_sharing_strategy('file_system')
            
        # Get configuration and create trainer
        config = ConfigurationManager()
        model_trainer_config = config.get_model_trainer_config()
        model_trainer = ModelTrainer(config=model_trainer_config)
            
        # Train with all optimizations enabled
        model_trainer.train()