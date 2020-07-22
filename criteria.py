import numpy as np
import torch
import torch.nn as nn

class NoisyCrossEntropy(nn.Module):
    def __init__(self, alpha):
        super(NoisyCrossEntropy, self).__init__()
        self.alpha = alpha
        self.cce = nn.CrossEntropyLoss()
        return

    def forward(self, pred, target, target_modified):
        loss = self.cce(pred, target)
        loss_modified = self.cce(pred, target_modified)
        return self.alpha*loss + self.alpha*loss_modified