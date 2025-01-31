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
loss_names = ['cce', 'ccenoisy']

data_names = ['Food101n', 'Clothing1M']

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
parser.add_argument('--epochs', default=30, type=int, metavar='N',
                    help='number of total epochs to run (default: 30)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--change-epoch', default = 0, type = int, metavar = 'N',
                    help = 'epoch number to change alpha')
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
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')

args = parser.parse_args()

print(args)




def main():
    # random.seed(0)
    torch.manual_seed(0)
    torch.random.manual_seed(0)
    # np.random.seed(0) 
    # torch.cuda.manual_seed(0)
    # torch.backends.cudnn.deterministic = True

    # create results folder, if not already exists
    output_directory = misc.get_output_directory(args)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    train_csv = os.path.join(output_directory, 'train.csv')
    test_csv = os.path.join(output_directory, 'test.csv')
    best_txt = os.path.join(output_directory, 'best.txt')
    
    print('=> creating data loaders ...')
    if args.data == 'Food101n':
        train_set = Food101n(True, random = True)
        test_set = Food101(False, random = True)
    elif args.data == 'Clothing1M':
        train_set = Clothing1M(True, random = True)
        test_set = Clothing1M(False, random = True)
    else:
        raise RuntimeError('Dataset not found.' +
                           'The dataset must be either of Food101 or Clothing1M.')

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers)
    # set batch size to be 1 for validation
    test_loader = torch.utils.data.DataLoader(test_set,
                                             batch_size=1, shuffle=False, num_workers=args.workers)

    print('=> data loaders created.')

    # optionally resume from a checkpoint
    if args.start_epoch != 0:
        assert os.path.isfile(args.resume), \
            '=> no checkpoint found at "{}"'.format(args.resume)
        print('=> loading checkpoint "{}"'.format(args.resume))
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch'] + 1
        best_result = checkpoint['best_result']
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']
        print('=> loaded checkpoint (epoch {})'.format(checkpoint['epoch']))

    # create new model
    else:
        # define model
        print('=> creating Model ({}) ...'.format(args.arch))

        if args.arch == 'resnet50':
            if args.data == 'Food101n':
                model = models.ResNet(50, 101)
            else:
                model = models.ResNet(50, 14)
        else:
            raise RuntimeError('model not found')

        print('=> model created.')

        
    # define loss function (criterion) and optimizer
    if args.criterion == 'cce' or args.criterion == 'ccenoisy':
        criterion = criteria.NoisyCrossEntropyLoss(0.).cuda()
    else:
        raise RuntimeError('loss function not found')


    if  args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    else:
        raise RuntimeError('optimizer not defined')

    optimizer_scheduler = lr_scheduler.StepLR(optimizer, args.epochs//3)

    model = model.cuda()
    print(model)
    print('=> model transferred to GPU.')

    train_logger, test_logger = None, None

    for epoch in range(args.start_epoch, args.epochs):
        corrector = None

        if args.criterion == 'ccenoisy' and epoch >= args.change_epoch:
            criterion.set_alpha(0.5)

            corrector = LabelCorrector(1280, 8)

            corrector.save_prototypes(train_set, model)

        train_result = train.train(train_loader, model, criterion, optimizer, corrector)

        if epoch == 0:
            train_logger = logger.Logger(output_directory, train_result, train_csv)
        else:
            train_logger.append(train_result)

        optimizer_scheduler.step()
        # evaluate on validation set
        test_result = test.validate(test_loader, model, criterion, optimizer)

        if epoch == 0:
            test_logger = logger.Logger(output_directory, test_result, test_csv)
        else:
            test_logger.append(test_result)

        misc.save_checkpoint({
            'args': args,
            'epoch': epoch,
            'arch': args.arch,
            'model': model,
        }, epoch, output_directory)

    train_logger.write_into_file('train')
    test_logger.write_into_file('test')

if __name__ == '__main__':
    main()