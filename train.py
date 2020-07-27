import numpy as np
import torch
import time

import logger

def train(train_loader, model, criterion, optimizer, corrector):
    
    # switch to train mode
    model.train()

    end = time.time()
    
    result = logger.Result()

    for i, (input, target) in enumerate(train_loader):
        target_modified = None


        input, target = input.cuda(), target.cuda()
        data_time = time.time() - end

        # compute pred
        end = time.time()
        features, pred = model(input)

        if corrector is not None:
            target_modified = corrector.get_modified_labels(features.detach().cpu().numpy()).cuda()

        loss = criterion(pred, target, target_modified)

        optimizer.zero_grad()
        
        loss.backward()  # compute gradient and do SGD step

        optimizer.step()

        gpu_time = time.time() - end

        # measure accuracy and record loss
        target = target.cpu().detach().numpy()
        pred = pred.cpu().detach().numpy()
        loss = loss.cpu().detach().item()
        result.update(target, pred, loss)
        end = time.time()

    result.calculate()
    return result