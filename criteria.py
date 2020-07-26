import numpy as np
import torch
import torch.nn as nn

class NoisyCrossEntropyLoss(nn.Module):
    def __init__(self, alpha):
        super(NoisyCrossEntropyLoss, self).__init__()
        self.alpha = alpha
        self.cce = nn.CrossEntropyLoss()
        return

    def forward(self, pred, target, target_modified):
        loss = self.cce(pred, target)
        if target_modified is not None:
            loss_modified = self.cce(pred, target_modified)
            return self.alpha*loss + self.alpha*loss_modified
        else: 
            return loss