import numpy as np
import torch
import torch.nn as nn
import torchvision
import argparse
import os
import torch.optim.lr_scheduler as lr_scheduler

import misc
import train
import test
import logger
from loader import Food101, Food101n, Clothing1M
import criteria
import models

from correct import LabelCorrector

model_names = ['resnet50']
loss_names = ['ccenoisy']

data_names = ['Clothing1M']

parser = argparse.ArgumentParser(description='UTKFace Training')

# fundamental arguments
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                    ' | '.join(model_names) +
                    ' (default: resnet50)')
parser.add_argument('--data', metavar='DATA', default='Food101n',
                    choices=data_names,
                    help='dataset: ' +
                    ' | '.join(data_names) +
                    ' (default: Food101n)')

# arguments for the epochs, batchsize, loss
parser.add_argument('--epochs', default=15, type=int, metavar='N',
                    help='number of epoch to load the checkpoint (default: 15)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    help='mini-batch size (default: 128)')
parser.add_argument('-c', '--criterion', metavar='LOSS', default='cce',
                    choices=loss_names,
                    help='loss function: ' +
                    ' | '.join(loss_names) +
                    ' (default: cce)')

# arguments for the optimizer
parser.add_argument('--optimizer', dest='optimizer', default='SGD', type=str,
                    help='Set optimizer.')
parser.add_argument('-r', '--lr', default=0.002, type=float,
                    metavar='LR', help='initial learning rate (default 0.002)')
parser.add_argument('-m', '--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '-w', default=5e-3, type=float,
                    metavar='W', help='weight decay (default: 5e-3)')

# arguments for others
parser.add_argument('-j', '--workers', default=10, type=int, metavar='N',
                    help='number of data loading workers (default: 10)')


args = parser.parse_args()

print(args)

def main():
    output_directory = misc.get_output_directory(args)

    checkpoint_filename = os.path.join(output_directory, 'checkpoint-' + str(args.epochs - 1) + '.pth.tar')

    assert os.path.isfile(checkpoint_filename), '=> no checkpoint found at "{}"'.format(args.resume)
    print('=> loading checkpoint "{}"'.format(checkpoint_filename))
    checkpoint = torch.load(checkpoint_filename)
    model = checkpoint['model']
    print('=> loaded checkpoint (epoch {})'.format(checkpoint['epoch']))


    if args.data == 'Clothing1M':
        train_set = Clothing1M(True, random = True)
        test_set = Clothing1M(False, random = True)
    else:
        raise RuntimeError('Dataset not found.' +
                           'The dataset must be either of Food101 or Clothing1M.')

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers)

    test_loader = torch.utils.data.DataLoader(test_set,
                                             batch_size=1, shuffle=False, num_workers=args.workers)

    corrector = LabelCorrector(1280, 8)

    corrector.save_prototypes(train_set, model)

    corresponded = 0

    for i, (input, target) in enumerate(test_loader):
        target_modified = None

        input = input.cuda()

        model.eval()
        features, _ = model(input)
        target_modified = corrector.get_modified_labels(features.detach().cpu().numpy())

        if target.item() == target_modified.item():
            corresponded += 1

    print("label accuracy : {}".format(corresponded / len(test_loader)))
        
if __name__ == '__main__':
    main()