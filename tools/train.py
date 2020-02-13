import os
import time
from argparse import ArgumentParser
from configparser import ConfigParser
import importlib
import torch

# insert root dir path to sys.path to import easycls
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import easycls.apis as apis
import easycls.models as models
import easycls.helpers as helpers


def parse_args():
    argparser = ArgumentParser(description="EasyClassification")
    argparser.add_argument('mode',
                           metavar='MODE',
                           help='mode name(train/eval...)')
    argparser.add_argument('-c',
                           '--config',
                           type=str,
                           metavar='PATH',
                           help='configuration file path')
    argparser.add_argument('-r',
                           '--resume',
                           default='',
                           type=str,
                           metavar='PATH',
                           help='resume checkpoint file path(default: none)')
    argparser.add_argument('-p',
                           '--print-freq',
                           default=10,
                           type=int,
                           metavar='N',
                           help='print frequency (default: 10)')
    args = argparser.parse_args()
    return args


def main():
    # parse arguments
    args = parse_args()

    # parse configurations
    cfgs = helpers.parse_cfgs(args.config)

    # select mode
    if args.mode in ('train', 'eval'):
        worker(args, cfgs)
        # apis.train(args.config)
    else:
        print(f'Unsupported mode [{args.mode}]')


best_acc1 = 0


def worker(args: ArgumentParser, cfgs: ConfigParser):
    global best_acc1

    # create model
    arch = cfgs.get('model', 'arch')
    if cfgs.getboolean('model', 'pretrained'):
        print("=> using pre-trained model '{}'".format(arch))
        model = models.__dict__[arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(arch))
        model = models.__dict__[arch]()

    # use DataParallel to allocate batch data and replicate model to all available GPUs
    print("using DataParallel on CUDA")
    model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    lr = cfgs.getfloat('learning', 'learning-rate')
    criterion = torch.nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(),
                                lr,
                                momentum=cfgs.getfloat('learning', 'momentum'),
                                weight_decay=cfgs.getfloat(
                                    'learning', 'weight-decay'))
    print(f'learning-rate starts from {lr}')

    # resume as specified
    start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))

            checkpoint = torch.load(args.resume)

            start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(
                args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # speedup for batches with fixed-size input
    # cudnn.benchmark = True

    # load dataset for specific task
    taskname = cfgs.get('data', 'task')
    print(f'current task (dataset) => {taskname}')

    # run-time import dataset according to task in config
    task = importlib.import_module('easycls.datasets.' + taskname)

    # prepare dataset and dataloader
    tic = time.time()
    train_dataset = task.get_train_dataset(cfgs)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfgs.getint('learning', 'batch-size'),
        shuffle=True,
        num_workers=cfgs.getint('learning', 'dataload-workers'),
        pin_memory=True,
        sampler=None)
    toc = time.time()
    print(f'Train set loaded in {(toc-tic):.3f}s')

    tic = time.time()
    val_dataset = task.get_val_dataset(cfgs)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=cfgs.getint('learning', 'batch-size'),
        shuffle=False,
        num_workers=cfgs.getint('learning', 'dataload-workers'),
        pin_memory=True)
    toc = time.time()
    print(f'Validation set loaded in {(toc-tic):.3f}s')

    tic = time.time()
    test_dataset = task.get_test_dataset(cfgs)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=cfgs.getint('learning', 'batch-size'),
        shuffle=False,
        num_workers=cfgs.getint('learning', 'dataload-workers'),
        pin_memory=True)
    toc = time.time()
    print(f'Test set loaded in {(toc-tic):.3f}s')

    if args.mode.lower() in ('eval', 'evaluate'):
        print('Evaluate Only')
        print(f'Evaluating on Validation set:')
        apis.validate(val_loader, model, criterion, args)
        print(f'Evaluating on Test Set:')
        apis.validate(test_loader, model, criterion, args)
        return

    localtime = time.asctime(time.localtime(time.time()))
    print(f'Start training at {localtime}')
    for epoch in range(start_epoch, cfgs.getint('learning', 'epochs')):
        lr = helpers.adjust_learning_rate(optimizer, epoch, cfgs)
        print(f'Epoch: {epoch+1}\tLearning-rate: {lr}')

        # train for one epoch
        apis.train(train_loader, model, criterion, optimizer, epoch, args)

        # evaluate on validation set
        print(f'Evaluating on Validation set:')
        acc1 = apis.validate(val_loader, model, criterion, args)
        print(f'Evaluating on Test Set:')
        apis.validate(test_dataset, model, criterion, args)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        helpers.save_checkpoint(
            {
                'arch': arch,
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer': optimizer.state_dict(),
            }, is_best)


if __name__ == "__main__":
    main()