import torch
import torch.nn as nn

class BinaryLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, probs, labels, alpha, beta):
        valid_mask = (labels == 0) | (labels == 1)

        if valid_mask.sum() == 0:
            raise ValueError("No valid labels, cannot calculate loss")

        valid_probs = torch.zeros_like(probs)
        valid_labels = torch.zeros_like(labels)
        valid_alpha = torch.zeros_like(alpha)
        valid_beta = torch.zeros_like(beta)
        
        valid_probs[valid_mask] = probs[valid_mask]
        valid_labels[valid_mask] = labels[valid_mask]
        valid_alpha[valid_mask] = alpha[valid_mask]
        valid_beta[valid_mask] = beta[valid_mask]
        
        valid_labels = torch.clamp(valid_labels, 0, 1)  # Ensure labels are in [0,1] range

        p = valid_probs
        g = valid_labels
        print("p:", p)
        print("g:", g)
        print("alpha:", valid_alpha)
        print("beta:", valid_beta)
        
        log_a = g * torch.log(valid_alpha) + (1 - g) * torch.log(1 - valid_alpha)
        log_a = torch.sum(log_a, dim=1, keepdim=True)
        a = torch.exp(log_a)

        log_b = (1 - g) * torch.log(valid_beta) + g * torch.log(1 - valid_beta)
        log_b = torch.sum(log_b, dim=1, keepdim=True)
        b = torch.exp(log_b)

        loss_val = torch.log(a * p + b * (1 - p)) / g.size()[1]    

        if valid_mask.sum() > 0:
            mean_loss = -torch.mean(loss_val[valid_mask] * valid_labels[valid_mask]) * 2.0
        
        accuracy = ((valid_probs > 0.5).float() == valid_labels).float().mean() if valid_mask.sum() > 0 else torch.tensor(0.0, device=probs.device)

        return mean_loss, loss_val  
    
