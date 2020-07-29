import numpy as np
import torch
import torch.nn as nn

class NoisyCrossEntropyLoss(nn.Module):
    def __init__(self, alpha : float):
        super(NoisyCrossEntropyLoss, self).__init__()
        self.alpha = alpha
        self.cce = nn.CrossEntropyLoss()
        return

    def forward(self, pred, target, target_modified):
        loss = self.cce(pred, target)
        if target_modified is not None:
            loss_modified = self.cce(pred, target_modified)
            return (1 - self.alpha)*loss + self.alpha*loss_modified
        else: 
            return loss

    def set_alpha(self, alpha : float):
        self.alpha = alpha