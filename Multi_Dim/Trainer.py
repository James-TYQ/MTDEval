import os
import json
import torch
import torch.nn as nn
import logging
from transformers import Trainer

from BinaryLoss import BinaryLoss

logger = logging.getLogger(__name__)

def is_sagemaker_mp_enabled():
    smp_options = os.getenv("SM_HP_MP_PARAMETERS", "{}")
    try:
        smp_options = json.loads(smp_options)
        if "partitions" not in smp_options:
            return False
    except json.JSONDecodeError:
        return False

    mpi_options = os.getenv("SM_FRAMEWORK_PARAMS", "{}")
    try:
        mpi_options = json.loads(mpi_options)
        if not mpi_options.get("sagemaker_mpi_enabled", False):
            return False
    except json.JSONDecodeError:
        return False

    return _smdistributed_available

class PreferenceTrainer(Trainer):
    def __init__(self, *args, label_field=None, dimension_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.binary_loss = BinaryLoss()
        self.label_field = ["Accuracy", "Logicality", "Fluency", "Relevance", "Personalization", "Creativity", "Interactivity", "Emotionality", "Knowledge", "Safety"]
        self.dimension_weights = dimension_weights if dimension_weights else [1.0] * len(self.label_field)
        
        if len(self.dimension_weights) != len(self.label_field):
            logger.warning(f"Dimension weights number ({len(self.dimension_weights)}) does not match the number of label fields ({len(self.label_field)}), using default weights")
            self.dimension_weights = [1.0] * len(self.label_field)
        
        num_dimensions = len(self.label_field)
        
        init_sens = torch.logit(torch.full((num_dimensions, 5), self.args.initial_sensitivity, device=self.args.device))
        init_spec = torch.logit(torch.full((num_dimensions, 5), self.args.initial_specificity, device=self.args.device))

        self.sensitivity = nn.Parameter(init_sens, requires_grad=True)  
        self.specificity = nn.Parameter(init_spec, requires_grad=True) 
        
        logger.info(f"Initialized {num_dimensions} dimensions sensitivity and specificity parameters")
        
        self.sensitivity.data = self.sensitivity.data.to(self.args.device)
        self.specificity.data = self.specificity.data.to(self.args.device)
    
    def create_optimizer(self):
        """create optimizer containing sensitivity and specificity parameters"""
        if self.optimizer is None:
            trainable = [p for p in self.model.parameters() if p.requires_grad]
            optimizer_grouped_parameters = [
                {"params": trainable, "weight_decay": self.args.weight_decay},
                {"params": [self.sensitivity, self.specificity],
                "weight_decay": 0.0,
                "lr": self.args.ss_learning_rate},
            ]

            optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(self.args)
            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
        
        return self.optimizer
        
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # get labels and model outputs
        labels = inputs.pop("labels")
        outputs = model(**inputs, use_cache=False)
        
        # handle different types of outputs
        if isinstance(outputs, tuple):
            logits = outputs[0]
        else:
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs
            num_dimensions = len(self.label_field)
            logits_mean, _ = torch.chunk(logits, 2, dim=-1)  # [batch_size, num_dimensions], [batch_size, num_dimensions]
        
        metrics = {}
        valid_losses = []
        valid_weights = []  
        
        # compute loss for each dimension
        for dim_idx, conclusion_dim in enumerate(self.label_field):
            if conclusion_dim in labels:
                dim_labels = labels[conclusion_dim]
                
                # check if all labels are Fair (-1)
                all_fair = all(dim_labels[0][i].item() == -1 for i in range(5))
                if all_fair:
                    logger.debug(f"All labels for {conclusion_dim} are Fair, skipping")
                    continue
                
                # get logits for current dimension
                dim_logits_mean = logits_mean[:, dim_idx]  # [batch_size]
                score_diff = dim_logits_mean[0::2] - dim_logits_mean[1::2]  # [batch_size//2]
                
                # calculate probability
                # score_var = dim_logits_var[0::2]**2 + dim_logits_var[1::2]**2 + 1e-4
                score_var = torch.ones_like(score_diff)     # keep var as constant
                p = 0.5 * (1 + torch.erf(-score_diff / torch.sqrt(2 * score_var)))
                p_scalar = p.squeeze()  
                row = torch.stack([1 - p_scalar, p_scalar])  # [2]
                p_expand = row.unsqueeze(0).repeat(5, 1)  # [5,2]
                
                _sensitivity = torch.sigmoid(self.sensitivity[dim_idx])  # [5]
                _specificity = torch.sigmoid(self.specificity[dim_idx])  # [5]
                
                alpha = torch.stack((dim_labels[0] + (-1)**dim_labels[0] * _sensitivity, (1-dim_labels[0]) + (-1)**(1-dim_labels[0]) * _sensitivity)).T.to(p.device)
                beta  = torch.stack(((1-dim_labels[0]) + (-1)**(1-dim_labels[0]) *_specificity, dim_labels[0] + (-1)**dim_labels[0] * _specificity)).T.to(p.device)

                new_labels = []
                for i in range(5):
                    if dim_labels[0][i].item() == -1:
                        new_labels.append(torch.tensor([-1, -1], device=dim_labels.device))
                    elif dim_labels[0][i].item() == 1:
                        new_labels.append(torch.tensor([0, 1], device=dim_labels.device))
                    elif dim_labels[0][i].item() == 0:
                        new_labels.append(torch.tensor([1, 0], device=dim_labels.device))
                    else:
                        raise ValueError(f"Unexpected value: {dim_labels[0][i].item()}")

                new_labels = torch.stack(new_labels, dim=0)  

                # calculate loss and accuracy
                loss, sample_losses = self.binary_loss(p_expand, new_labels, alpha=alpha, beta=beta)
                
                # calculate accuracy
                valid_mask = (dim_labels == 0) | (dim_labels == 1)
                if valid_mask.sum() > 0:
                    pred = (p > 0.5).float()
                    accuracy = (pred == dim_labels.float()).float()[valid_mask].mean()
                else:
                    accuracy = torch.tensor(0.0, device=p.device)
                
                # record the loss and weight for this dimension
                valid_losses.append(loss)
                valid_weights.append(self.dimension_weights[dim_idx])
                
                # record the metrics for each dimension
                metrics[f"{conclusion_dim}_loss"] = loss.item()
                metrics[f"{conclusion_dim}_acc"] = accuracy.item()
        
        if not valid_losses:
            final_loss = torch.tensor(0.0, device=self.args.device, requires_grad=True)
        else:
            weighted_losses = []
            for loss, weight in zip(valid_losses, valid_weights):
                weighted_losses.append(loss * weight)
            
            total_weighted_loss = torch.stack(weighted_losses).sum()
            total_weights = sum(valid_weights)
            final_loss = total_weighted_loss / total_weights
        
        # record the overall metrics
        metrics["final_loss"] = final_loss.item()
        metrics["valid_dimensions"] = len(valid_losses)
        metrics["total_weights"] = sum(valid_weights) if valid_weights else 0.0
        metrics["sensitivity"] = torch.sigmoid(self.sensitivity).detach().cpu().numpy().tolist()
        metrics["specificity"] = torch.sigmoid(self.specificity).detach().cpu().numpy().tolist()
        
        self.log(metrics)
        
        return final_loss

    def save_model(self, output_dir=None, _internal_call=False):
        super().save_model(output_dir=output_dir, _internal_call=_internal_call)
        
        # save sensitivity and specificity parameters
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # calculate current sensitivity and specificity values
        sensitivity_value = torch.sigmoid(self.sensitivity).detach()
        specificity_value = torch.sigmoid(self.specificity).detach()
        
        # create dictionary containing parameters and values
        ss_dict = {
            "sensitivity": self.sensitivity,
            "specificity": self.specificity,
            "sensitivity_value": sensitivity_value,
            "specificity_value": specificity_value
        }
        
        # save parameters
        torch.save(ss_dict, os.path.join(output_dir, "sensitivity_specificity.pt"))
        logger.info(f"Save sensitivity: {sensitivity_value}, specificity: {specificity_value}")
        
    def _load_from_checkpoint(self, resume_from_checkpoint):
        super()._load_from_checkpoint(resume_from_checkpoint)
        
        # load sensitivity and specificity parameters
        ss_path = os.path.join(resume_from_checkpoint, "sensitivity_specificity.pt")
        if os.path.exists(ss_path):
            ss_dict = torch.load(ss_path, map_location="cpu")

            self.sensitivity.data.copy_(ss_dict["sensitivity"].data)
            self.specificity.data.copy_(ss_dict["specificity"].data)

            self.sensitivity.data = self.sensitivity.data.to(self.args.device)
            self.specificity.data = self.specificity.data.to(self.args.device)

            logger.info("Load sensitivity: %.4f, specificity: %.4f",
                        torch.sigmoid(self.sensitivity).item(),
                        torch.sigmoid(self.specificity).item())
        else:   
            logger.warning(f"Sensitivity and specificity parameters file not found: {ss_path}")
