import torch
import torch.nn as nn
import torch.nn.functional as F

class MultinomialCrossEntropyLoss(nn.Module):
    def __init__(self, weights=None):
        super().__init__()
        self.weights = weights
        
    def forward(self, logits, targets):
        """
        logits: (B, Q, H, W)
        targets: (B, Q, H, W) soft encoded labels
        """
        log_probs = F.log_softmax(logits, dim=1)
        loss_map = - (targets * log_probs)
        if self.weights is not None:
            # Ensure weights are on the correct device
            if self.weights.device != logits.device:
                self.weights = self.weights.to(logits.device)
            w_broadcast = self.weights.view(1, -1, 1, 1)
            loss_map = loss_map * w_broadcast
        loss = loss_map.sum(dim=1).mean() 
        return loss
