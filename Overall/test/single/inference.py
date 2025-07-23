from typing import Dict, List
from transformers import AutoTokenizer, AutoModelForSequenceClassification, LlamaForSequenceClassification, AutoConfig
import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)

class MTDEvalClassifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MTDEvalClassifier, self).__init__()
        self.mean_layer = nn.Sequential(
            nn.Linear(input_dim, 2048, bias=False), nn.SiLU(), nn.Dropout(0.1),
            nn.Linear(2048, 1024, bias=False), nn.SiLU(),
            nn.Linear(1024, 1024, bias=False), nn.SiLU(),
            nn.Linear(1024, 1024, bias=False), nn.SiLU(), nn.Dropout(0.1),
            nn.Linear(1024, output_dim, bias=False)
        )

        self.var_layer = nn.Sequential(
            nn.Linear(input_dim, 2048, bias=False), nn.SiLU(), nn.Dropout(0.1),
            nn.Linear(2048, 1024, bias=False), nn.SiLU(),
            nn.Linear(1024, 1024, bias=False), nn.SiLU(),
            nn.Linear(1024, 1024, bias=False), nn.SiLU(), nn.Dropout(0.1),
            nn.Linear(1024, output_dim, bias=False)
        )

    def forward(self, x):
        mean = self.mean_layer(x)
        var = self.var_layer(x)
        return torch.cat([mean, var], dim=-1)

class LlamaForMDQRwardModel(LlamaForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        self.score = MTDEvalClassifier(config.hidden_size, config.num_labels)

class MTDEvalPipeline:
    def __init__(self, model_id, device_map="auto", torch_dtype=torch.bfloat16, truncation=True, trust_remote_code=False, max_length=8192, dimensions: list[str] | None = None,):
        self.dimensions = dimensions or ["Overall"]
        print(f"pipeline has dimensions: {self.dimensions}")
        expected_labels = len(self.dimensions)

        cfg = AutoConfig.from_pretrained(model_id)
        cfg.num_labels = expected_labels
        self.model = LlamaForMDQRwardModel.from_pretrained(
            model_id,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            use_fast=True,
        )
        self.truncation = truncation
        self.device = self.model.device
        self.max_length = max_length

    def __call__(self, messages: List[Dict[str, str]]) -> Dict[str, float]:
        """
        messages: OpenAI chat messages to be scored, i.e., [{'role': 'user', 'content': '...'}, {'role': 'assistant', 'content': '...'}]
        return: a dictionary of results
        """

        print(messages)

        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        input_ids = self.tokenizer(
            inputs,
            return_tensors="pt",
            padding=True,
            truncation=self.truncation,
            max_length=self.max_length
        ).to(self.device)
        
        # inference
        with torch.no_grad():
            try:
                outputs = self.model(**input_ids)
                logits = outputs.logits

                mean, _ = torch.split(logits, logits.size(-1)//2, dim=-1)

                overall = mean[0][0].item()
                # Apply sigmoid to bring the score into the range [0, 1]
                overall_score = torch.sigmoid(torch.tensor(overall)).item()
                
            except Exception as e:
                logger.error(e)
                raise e
                
        return {
            "dimensional_score": {"Overall": overall_score},
            "overall_score": overall_score,
            "evaluation_dim": ["Overall"],
            "Overall": overall_score,
        }
    
