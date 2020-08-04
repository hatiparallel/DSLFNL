import numpy as np
import torch
import time

import logger

def train(train_loader, model, criterion, optimizer, corrector):
    
    result = logger.Result()

    for i, (input, target) in enumerate(train_loader):
        target_modified = None


        input, target = input.cuda(), target.cuda()

        if corrector is not None:
            model.eval()
            features, _ = model(input)
            target_modified = corrector.get_modified_labels(features.detach().cpu().numpy()).cuda()

        # compute pred
        model.train()
        _, pred = model(input)

        loss = criterion(pred, target, target_modified)

        optimizer.zero_grad()
        
        loss.backward()  # compute gradient and do SGD step

        optimizer.step()

        # measure accuracy and record loss
        target = target.cpu().detach().numpy()
        pred = pred.cpu().detach().numpy()
        loss = loss.cpu().detach().item()
        result.update(target, pred, loss)

    result.calculate()
    return result