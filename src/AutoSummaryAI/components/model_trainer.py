from transformers import TrainingArguments, Trainer
from transformers import DataCollatorForSeq2Seq
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_dataset, load_from_disk
from AutoSummaryAI.entity import ModelTrainerConfig
import torch
import os

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train(self):
        """
        Train the model with extreme GPU memory optimization techniques for 4GB GPUs
        using Parameter-Efficient Fine-Tuning (PEFT) with LoRA and ROUGE metrics
        """
        import torch
        import gc
        import numpy as np
        from transformers import BitsAndBytesConfig
        from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
        
        # First, install required packages if not already installed
        try:
            import peft
            import rouge_score
        except ImportError:
            import subprocess
            import sys
            print("Installing required packages...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "peft", "rouge-score"])
            import peft
            import rouge_score
            
        # Force aggressive garbage collection and memory cleanup
        gc.collect()
        torch.cuda.empty_cache()
        
        # Report initial memory state
        if torch.cuda.is_available():
            print(f"Initial GPU memory: {torch.cuda.memory_allocated()/1024**2:.2f} MB allocated")
        
        # Set device to CUDA if available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        
        # Load tokenizer first (uses less memory)
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_ckpt)
        
        # Setup 4-bit quantization configuration (more aggressive than 8-bit)
        print("Configuring 4-bit quantization...")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,              # Enable 4-bit quantization (more memory efficient)
            bnb_4bit_use_double_quant=True, # Use nested quantization for 4-bit weights
            bnb_4bit_quant_type="nf4",      # Use normalized float 4 format
            bnb_4bit_compute_dtype=torch.bfloat16  # Use bfloat16 for computation with 4-bit
        )
        
        # Load model with extreme memory optimization
        print("Loading model with 4-bit quantization...")
        model_pegasus = AutoModelForSeq2SeqLM.from_pretrained(
            self.config.model_ckpt,
            quantization_config=quantization_config,
            device_map="auto",
            low_cpu_mem_usage=True,
            torch_dtype=torch.bfloat16
        )
        
        # Immediately prepare model for k-bit training
        print("Preparing model for training with PEFT/LoRA...")
        model_pegasus = prepare_model_for_kbit_training(model_pegasus)
        
        # Define LoRA configuration with memory-efficient parameters
        print("Configuring LoRA with reduced parameters...")
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
            # Apply to fewer modules for memory efficiency
            target_modules=["q_proj", "v_proj"],
            bias="none"
        )
        
        # Apply LoRA adapter
        print("Applying LoRA adapter to the model...")
        model_pegasus = get_peft_model(model_pegasus, lora_config)
        
        # Enable gradient checkpointing (trades computation for memory)
        print("Enabling gradient checkpointing...")
        model_pegasus.gradient_checkpointing_enable()
        
        # Print trainable parameters information
        self.print_trainable_parameters(model_pegasus)
        
        # Set extreme small batch sizes for 4GB GPU
        train_batch_size = 1
        eval_batch_size = 1
        
        # Use higher gradient accumulation
        gradient_accumulation_steps = 16
        
        print(f"GPU memory after loading model: {torch.cuda.memory_allocated()/1024**2:.2f} MB allocated")
        
        # Data collator that will handle dynamic padding
        print("Creating data collator...")
        seq2seq_data_collator = DataCollatorForSeq2Seq(tokenizer, model=model_pegasus, padding="longest")
        
        # Load dataset
        print("Loading dataset...")
        dataset_samsum_pt = load_from_disk(self.config.data_path)
        
        # Use a smaller subset of training and validation data for testing
        # This helps with memory usage and speeds up initial validation
        train_subset_size = min(1000, len(dataset_samsum_pt["train"]))
        print(f"Using {train_subset_size} examples for training (out of {len(dataset_samsum_pt['train'])})")
        train_dataset = dataset_samsum_pt["train"].select(range(train_subset_size))
        
        # Use very small validation set
        if "validation" in dataset_samsum_pt:
            val_subset_size = min(50, len(dataset_samsum_pt["validation"]))
            print(f"Using {val_subset_size} examples for validation (out of {len(dataset_samsum_pt['validation'])})")
            val_dataset = dataset_samsum_pt["validation"].select(range(val_subset_size))
        else:
            val_dataset = None
        
        # Define compute_metrics function for ROUGE evaluation
        def compute_metrics(pred):
            """
            Compute ROUGE metrics for summarization model evaluation
            with minimal memory usage.
            """
            from rouge_score import rouge_scorer
            
            # Process predictions in small batches to save memory
            labels = []
            predictions = []
            
            # Process in batches of 8 to avoid memory spikes
            batch_size = 8
            
            for i in range(0, len(pred.predictions), batch_size):
                # Get batch
                pred_batch = pred.predictions[i:i + batch_size]
                label_batch = pred.label_ids[i:i + batch_size]
                
                # Decode predictions
                pred_texts = tokenizer.batch_decode(
                    pred_batch, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                predictions.extend(pred_texts)
                
                # Process labels - replace -100 padding
                proc_labels = []
                for label in label_batch:
                    # Replace -100 with pad token ID
                    label = np.where(label != -100, label, tokenizer.pad_token_id)
                    proc_labels.append(label)
                
                # Decode labels
                label_texts = tokenizer.batch_decode(
                    proc_labels, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                labels.extend(label_texts)
                
                # Force garbage collection after each batch
                if i % (batch_size * 4) == 0:
                    gc.collect()
                    torch.cuda.empty_cache()
            
            # Calculate ROUGE scores
            scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
            
            # Calculate scores for each prediction and reference pair
            rouge1_scores = []
            rouge2_scores = []
            rougeL_scores = []
            
            # Process in batches to avoid memory spikes
            for i in range(len(predictions)):
                # Handle empty predictions or references gracefully
                if not predictions[i].strip() or not labels[i].strip():
                    rouge1_scores.append(0)
                    rouge2_scores.append(0)
                    rougeL_scores.append(0)
                    continue
                    
                # Calculate scores
                try:
                    scores = scorer.score(labels[i], predictions[i])
                    rouge1_scores.append(scores['rouge1'].fmeasure)
                    rouge2_scores.append(scores['rouge2'].fmeasure)
                    rougeL_scores.append(scores['rougeL'].fmeasure)
                except Exception as e:
                    print(f"Error calculating ROUGE: {e}")
                    rouge1_scores.append(0)
                    rouge2_scores.append(0)
                    rougeL_scores.append(0)
                
                # Force garbage collection periodically
                if i % 20 == 0:
                    gc.collect()
            
            # Return the average scores
            results = {
                'rouge1': float(np.mean(rouge1_scores)),
                'rouge2': float(np.mean(rouge2_scores)),
                'rougeL': float(np.mean(rougeL_scores))
            }
            print(f"ROUGE Scores: {results}")
            return results
            
        # Set up training arguments with extreme memory optimization
        print(f"Setting up trainer with batch size: {train_batch_size}, grad_accum: {gradient_accumulation_steps}")
        
        # Set shorter training length for testing
        num_epochs = 1  # Start with 1 epoch to verify setup
        
        trainer_args = TrainingArguments(
            output_dir=self.config.root_dir,
            num_train_epochs=num_epochs,
            warmup_steps=self.config.warmup_steps,
            per_device_train_batch_size=train_batch_size,
            per_device_eval_batch_size=eval_batch_size,
            weight_decay=self.config.weight_decay,
            logging_steps=25,  # More frequent logging
            eval_strategy="steps" if val_dataset else "no",
            eval_steps=100,    # More frequent evaluation
            save_steps=500,    # Save less frequently to reduce disk I/O
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=float(self.config.learning_rate),
            load_best_model_at_end=True if val_dataset else False,
            metric_for_best_model="rouge1",  # Use ROUGE1 for model selection
            greater_is_better=True,          # Higher ROUGE is better
            fp16=True,                      # Use mixed precision
            fp16_full_eval=True,            # Use mixed precision for eval too
            gradient_checkpointing=True,    # Enable gradient checkpointing
            optim="adamw_torch",            # Use AdamW optimizer
            max_grad_norm=0.3,              # Lower gradient clipping threshold
            ddp_find_unused_parameters=False,
            dataloader_pin_memory=False,    # Save CPU memory
            report_to="none",               # Disable reporting
            # These settings help with memory issues during training
            dataloader_num_workers=0,       # Don't use multiprocessing
            group_by_length=True,           # Group similar length sequences
            lr_scheduler_type="cosine",     # Cosine learning rate schedule
            # The following are to avoid "deadlocks"
            use_cpu=False,                  # Don't use CPU for training
            seed=42,                        # Fixed seed for reproducibility
            # Debug settings - uncomment to help diagnose issues
            # debug="underflow_overflow",
        )
        
        # Create trainer with compute_metrics
        print("Creating trainer with ROUGE metrics...")
        trainer = Trainer(
            model=model_pegasus,
            args=trainer_args,
            data_collator=seq2seq_data_collator,
            train_dataset=train_dataset,  # Using the subset
            eval_dataset=val_dataset,     # Using the subset
            compute_metrics=compute_metrics  # Add the ROUGE metrics calculation
        )
        
        # Train model
        print("Starting training...")
        try:
            trainer.train()
            
            # Save model and tokenizer
            print("Saving model and tokenizer...")
            # Save the LoRA adapter only to save space
            model_pegasus.save_pretrained(os.path.join(self.config.root_dir, "pegasus-samsum-model-lora"))
            tokenizer.save_pretrained(os.path.join(self.config.root_dir, "tokenizer"))
            print("Training complete!")
            
        except Exception as e:
            print(f"Error during training: {e}")
            import traceback
            traceback.print_exc()
            
            # Try to save checkpoint even if training fails
            try:
                print("Attempting to save checkpoint after error...")
                model_pegasus.save_pretrained(os.path.join(self.config.root_dir, "checkpoint-error"))
                tokenizer.save_pretrained(os.path.join(self.config.root_dir, "tokenizer-error"))
            except:
                print("Could not save checkpoint after error.")
    
    def print_trainable_parameters(self, model):
        """
        Prints the number of trainable parameters in the model.
        """
        trainable_params = 0
        all_params = 0
        for _, param in model.named_parameters():
            all_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(
            f"trainable params: {trainable_params:,} || all params: {all_params:,} || trainable%: {100 * trainable_params / all_params:.2f}%"
        )