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
from loader import Food101

model_names = ['resnet50']
loss_names = ['cce', 'ccenoisy']

data_names = ['food101']

parser = argparse.ArgumentParser(description='UTKFace Training')

# fundamental arguments
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                    ' | '.join(model_names) +
                    ' (default: resnet50)')
parser.add_argument('--data', metavar='DATA', default='Food101',
                    choices=data_names,
                    help='dataset: ' +
                    ' | '.join(data_names) +
                    ' (default: food101)')

# arguments for the epochs, batchsize, loss
parser.add_argument('--epochs', default=30, type=int, metavar='N',
                    help='number of total epochs to run (default: 30)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
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
    torch.manual_seed(0)
    torch.random.manual_seed(0)

    # create results folder, if not already exists
    output_directory = misc.get_output_directory(args)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    train_csv = os.path.join(output_directory, 'train.csv')
    test_csv = os.path.join(output_directory, 'test.csv')
    best_txt = os.path.join(output_directory, 'best.txt')
    
    print('=> creating data loaders ...')
    if args.data == 'Food101':
        train_set = Food101(True, random = True)
        test_set = Food101(False, random = False)
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
            model = models.ResNet(50)
        else:
            raise RuntimeError('model not found')

        print('=> model created.')

        
    # define loss function (criterion) and optimizer
    if args.criterion == 'cce' or args.criterion == 'ccenoisy':
        criterion = nn.CrossEntropyLoss().cuda()
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
        train_result = train.train(train_loader, model, criterion, optimizer)

        if epoch == 0:
            train_logger = logger.Logger(output_directory, 0, train_result)
        else:
            train_logger.append(train_result)

        optimizer_scheduler.step()
        # evaluate on validation set
        test_result = test.validate(test_loader, model, criterion, optimizer)

        if epoch == 0:
            test_logger = logger.Logger(output_directory, 0, test_result)
        else:
            test_logger.append(test_result)

        misc.save_checkpoint({
            'args': args,
            'epoch': epoch,
            'arch': args.arch,
            'model': model,
            'best_result': best_result,
            'optimizer': optimizer,
        }, is_best, epoch, output_directory)

if __name__ == '__main__':
    main()