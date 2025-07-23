from typing import Dict, List
from transformers import AutoTokenizer, AutoModelForSequenceClassification, LlamaForSequenceClassification, AutoModel, AutoConfig
import torch
import torch.nn as nn
import logging
import numpy as np

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

    def forward(self, x):
        means = torch.cat([layer(x) for layer in self.mean_layers], dim=-1)  # [batch_size, num_dimensions]
        return means 

class LlamaForMDQRwardModel(AutoModelForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        if hasattr(config, 'num_labels') and config.num_labels >= 1: 
            num_dimensions = config.num_labels
        else:
            num_dimensions = 10 
            logger.warning(f"Config num_labels is {getattr(config, 'num_labels', 'None')}, using default {num_dimensions} dimensions for MTDEvalClassifier")
        
        if hasattr(config, 'hidden_size'):
            input_dim = config.hidden_size
        else:
            input_dim = getattr(config, 'd_model', 8192) 
            logger.warning(f"Config hidden_size not found, using {input_dim}")
        
        self.score = MTDEvalClassifier(input_dim, num_dimensions)
        logger.info(f"Initialized MTDEvalClassifier with input_dim={input_dim}, num_dimensions={num_dimensions}")

class MTDEvalPipeline:
    dimensions = ["Accuracy", "Logicality", "Fluency", "Relevance", "Personalization", "Creativity", "Interactivity", "Emotionality", "Knowledge", "Safety"] 
    
    def __init__(self, model_id, device_map="auto", torch_dtype=torch.bfloat16, truncation=True, trust_remote_code=False, max_length=8192):
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=trust_remote_code)
        
        expected_num_labels = len(self.dimensions) 
        if not hasattr(config, 'num_labels') or config.num_labels != expected_num_labels:
            logger.warning(f"Updating config.num_labels from {getattr(config, 'num_labels', 'None')} to {expected_num_labels}")
            config.num_labels = expected_num_labels
        
        self.model = LlamaForMDQRwardModel.from_pretrained(
            model_id,
            config=config,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            use_fast=True,
        )
        self.tokenizer.pad_token_id = 0
        self.truncation = truncation
        self.device = self.model.device
        self.max_length = max_length

    def _get_normalized_scores(self, logits):
        num_dims = len(self.dimensions)
        if logits.size(-1) != num_dims:
            raise ValueError(f"Output dimension mismatch: got {logits.size(-1)}, expected {num_dims}")
        return torch.sigmoid(logits.squeeze().cpu()).numpy()

    def score_dialog(self, messages: List[Dict[str, str]]):
        inputs = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        input_ids = self.tokenizer(inputs, return_tensors="pt", padding=True, truncation=self.truncation, max_length=self.max_length).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**input_ids)
            return self._get_normalized_scores(outputs.logits) 

    def __call__(self, messages: List[Dict[str, str]]) -> Dict[str, float]:
        mean_scores_normalized = self.score_dialog(messages)
        dimensional_scores = {}
        
        for i, dim in enumerate(self.dimensions):
            dimensional_scores[dim] = float(mean_scores_normalized[i])
        
        overall_score = sum(dimensional_scores.values()) / len(dimensional_scores)
        
        return {
            "dimensional_scores": dimensional_scores,
            "overall_score": overall_score,
            "evaluation_dims": self.dimensions,
        }

    def compare_responses(self, dialog_A, dialog_B, fair_threshold=0.05):
        def _get_winner(score_A, score_B, threshold):
            diff = score_A - score_B
            return "A" if diff > threshold else ("B" if diff < -threshold else "Fair")

        mean_A = self.score_dialog(dialog_A)
        mean_B = self.score_dialog(dialog_B)
        
        dimension_comparisons = {
            dim: {
                "A_score": float(mean_A[i]),
                "B_score": float(mean_B[i]),
                "winner": _get_winner(mean_A[i], mean_B[i], fair_threshold/2),  
                "score_diff": float(mean_A[i] - mean_B[i])
            } for i, dim in enumerate(self.dimensions)
        }

        overall_winner = _get_winner(
            mean_A.mean(), mean_B.mean(), fair_threshold
        )
        
        return {
            "dimension_comparisons": dimension_comparisons,
            "overall_winner": overall_winner,
            "A_scores": dict(zip(self.dimensions, mean_A)),
            "B_scores": dict(zip(self.dimensions, mean_B))
        }