from itertools import combinations
import logging
import os
import random
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import datasets
import transformers
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForSequenceClassification,
    LlamaForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
    set_seed,
)
from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy
from transformers import Trainer, TrainingArguments as BaseTrainingArguments
from transformers.trainer_utils import get_last_checkpoint

import json
from typing import Any, Dict
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List
import functools
from torch.utils.checkpoint import checkpoint
checkpoint_use_reentrant = False

from BinaryLoss import BinaryLoss
from DataLoader import load_datasets, LabelFilter
from Arguments import ScriptArguments, TrainingArguments
from Trainer import PreferenceTrainer

logger = logging.getLogger(__name__)

class MTDEvalClassifier(nn.Module):
    def __init__(self, input_dim, num_dimensions):
        super(MTDEvalClassifier, self).__init__()
        self.num_dimensions = num_dimensions
        
        self.mean_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, 2048, bias=False), nn.SiLU(), nn.Dropout(0.1),
                nn.Linear(2048, 1024, bias=False), nn.SiLU(),
                nn.Linear(1024, 1024, bias=False), nn.SiLU(),
                nn.Linear(1024, 1024, bias=False), nn.SiLU(), nn.Dropout(0.1),
                nn.Linear(1024, 1, bias=False) 
            ) for _ in range(num_dimensions)
        ])

        self.var_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, 2048, bias=False), nn.SiLU(), nn.Dropout(0.1),
                nn.Linear(2048, 1024, bias=False), nn.SiLU(),
                nn.Linear(1024, 1024, bias=False), nn.SiLU(),
                nn.Linear(1024, 1024, bias=False), nn.SiLU(), nn.Dropout(0.1),
                nn.Linear(1024, 1, bias=False)  
            ) for _ in range(num_dimensions)
        ])

    def forward(self, x):
        means = torch.cat([layer(x) for layer in self.mean_layers], dim=-1)  # [batch_size, num_dimensions]
        vars = torch.cat([layer(x) for layer in self.var_layers], dim=-1)    # [batch_size, num_dimensions]
        return torch.cat([means, vars], dim=-1)  # We provide var as output, but we use a fixed constant for variance actually

class LlamaForMDQRwardModel(AutoModelForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        self.score = MTDEvalClassifier(config.hidden_size, config.num_labels//2)

class DataCollator:
    def __init__(self, args, training_args, tokenizer):
        self.args = args
        self.training_args = training_args
        self.tokenizer = tokenizer
        self.tokenizer.padding_side = "left"
        self.pad_token_id = self.tokenizer.pad_token_id
        self.max_length = self.args.max_length

    @torch.no_grad()
    def __call__(self, features: Any) -> Dict[str, Any]:
        text_field = ["dialogue_a", "dialogue_b"]
        batch = self.tokenizer(
            sum([[item[text] for text in text_field] for item in features], []),
            add_special_tokens=False, truncation=True, return_tensors="pt", 
            padding=True, max_length=self.max_length
        )
        
        labels = {}
        for label_name in self.args.label_field:
            if label_name in features[0]:
                labels[label_name] = torch.tensor([item[label_name] for item in features], dtype=torch.long)
        
        return dict(
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask,
            labels=labels
        )

def confidence_mask(labels, confidence):        
    return (labels - 0.5).abs() >= confidence / 2  

class ConfidenceFilter:         
    def __init__(self, label_field, confidence):
        self.label_field = label_field
        self.confidence = confidence

    def __call__(self, example):
        labels = torch.tensor([example[label] for label in self.label_field])
        return confidence_mask(labels[labels != -1], self.confidence).any().item()     

def bce_with_temperature(probs, labels, temperature = 2.0):     
    probs = probs.clamp(min=0.0, max=1.0)                       
    labels = labels.clamp(min=0.0, max=1.0)

    if temperature != 1.0:
        labels = (labels.logit() / temperature).sigmoid()

    return torch.nn.functional.binary_cross_entropy(probs, labels)

def main():
    parser = HfArgumentParser((ScriptArguments, TrainingArguments))
    args, training_args = parser.parse_args_into_dataclasses()
    
    os.makedirs(training_args.output_dir, exist_ok=True)
    
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")
    logger.info(f"Additional arguments {args}")
    
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        try:
            last_checkpoint = get_last_checkpoint(training_args.output_dir)
            if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
                trainer_state_path = os.path.join(last_checkpoint, "trainer_state.json")
                if os.path.exists(trainer_state_path):
                    logger.info(
                        f"Checkpoint detected, resuming training at {last_checkpoint}."
                    )
                    training_args.resume_from_checkpoint = last_checkpoint
                else:
                    logger.warning(
                        f"Checkpoint directory {last_checkpoint} found but trainer_state.json is missing. "
                        f"Starting training from scratch."
                    )
                    last_checkpoint = None
        except Exception as e:
            logger.warning(f"Error detecting checkpoint: {e}. Starting training from scratch.")
            last_checkpoint = None

    # Set seed before initializing model. 
    set_seed(training_args.seed)
    print(">>>>>>", args.tokenizer_name)
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name,
        cache_dir=args.cache_dir,
        use_fast=args.use_fast_tokenizer,
        revision=args.model_revision,
        use_auth_token=True if args.use_auth_token else None,
    )       

    config = AutoConfig.from_pretrained(
        args.config_name,
        cache_dir=args.cache_dir,
        revision=args.model_revision,
        use_auth_token=True if args.use_auth_token else None
    )       

    if args.config_overrides:
        logger.info(f"Overriding config: {args.config_overrides}")
        config.update_from_string(args.config_overrides)
        logger.info(f"New config: {config}")

    config.num_labels = 2 * (len(args.label_field))
    tokenizer.pad_token_id = 0
    config.pad_token_id = 0

    if args.model_name_or_path:
        half_dtype = (torch.bfloat16 if training_args.bf16 else (torch.float16 if training_args.fp16 else None))
        device_map = {"":int(os.environ.get("LOCAL_RANK") or 0)}
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name_or_path,
            cache_dir=args.cache_dir,
            revision=args.model_revision,
            use_auth_token=True if args.use_auth_token else None,
            torch_dtype=half_dtype,
            use_flash_attention_2=training_args.use_flash_attention_2,
        )

        for param in model.model.parameters():
            param.requires_grad = False
        
        in_features, out_features = model.score.in_features, model.score.out_features
        model.score = MTDEvalClassifier(input_dim=in_features, num_dimensions=len(args.label_field))
        for param in model.score.parameters():
            param.requires_grad = True
    else:
        model = AutoModelForSequenceClassification.from_config(config)

    if args.lora or args.lora_path:
        from peft import PeftModel, get_peft_model, LoraConfig, TaskType
        if args.lora_path:
            logger.info(f">>>>>> Loading LoRA model from {args.lora_path}")
            model = PeftModel.from_pretrained(model, args.lora_path)
        else:
            lora_target_modules = args.lora_target_modules.split(',')
            peft_config = LoraConfig(
                task_type=TaskType.SEQ_CLS,
                inference_mode=False,
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                target_modules=lora_target_modules,
                modules_to_save=args.lora_modules_to_save,
            )
            model = get_peft_model(model, peft_config)
        
        model.print_trainable_parameters()

    logger.info(f"Model: {model}")
    
    train_paths = Path(args.train_datasets_dir)
    train_files = [str(f) for f in train_paths.glob("**/*.json")]
    
    if not train_files:
        raise ValueError(f"No JSON files found in {args.train_datasets_dir}. Check your directory path.")
    
    print(f"Found {len(train_files)} training files: {train_files}")
    
    train_dataset_dict = load_datasets(
        tokenizer,
        train_files,
        args.label_field,
        training_args.dataloader_num_workers,
        cache_dir=args.cache_dir
    )

    if not train_dataset_dict or all(len(ds) == 0 for ds in train_dataset_dict.values()):
        raise ValueError("Training dataset is empty after processing. Check your data and filtering criteria.")
    
    train_dataset = datasets.concatenate_datasets(list(train_dataset_dict.values()))
    
    print(f"Training dataset size: {len(train_dataset)}")
    logger.warning(f"Training sequence number: {len(train_dataset):,}")

    collator = DataCollator(args, training_args, tokenizer)

    trainer = PreferenceTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=None,
        tokenizer=tokenizer,
        data_collator=collator,
        label_field=args.label_field,
        dimension_weights=args.dimension_weights,  
    )

    if trainer.is_fsdp_enabled:
        def layer_policy_fn(module):
            return "layer" in module.__class__.__name__.lower()

        auto_wrap_policy = functools.partial(lambda_auto_wrap_policy,
                                             lambda_fn=layer_policy_fn)
        trainer.accelerator.state.fsdp_plugin.auto_wrap_policy = auto_wrap_policy

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
            print(f"Checkpoint detected, resuming training at {checkpoint}")
        
        logger.info("Start training")
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # save model and tokenizer

        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        
        logger.info("Training completed")
        logger.info(f"Final training metrics: {metrics}")
    else:
        logger.info("Training mode not enabled, skipping training")

if __name__ == "__main__":
    main()

