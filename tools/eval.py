import os
import time
from argparse import ArgumentParser
from configparser import ConfigParser
import importlib
import torch
import torch.backends.cudnn as cudnn

# insert root dir path to sys.path to import easycls
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import easycls.apis as apis
import easycls.models as models
import easycls.helpers as helpers


def parse_args():
    argparser = ArgumentParser(description="EasyClassification")
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

    worker(args, cfgs)


def worker(args: ArgumentParser, cfgs: ConfigParser):
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

    criterion = torch.nn.CrossEntropyLoss().cuda()

    # resume as specified
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})".format(
                args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # speedup for batches with fixed-size input
    cudnn.benchmark = cfgs.getboolean('learning', 'cudnn-benchmark')

    # load dataset for specific task
    taskname = cfgs.get('data', 'task')
    print(f'current task (dataset) => {taskname}')

    # run-time import dataset according to task in config
    task = importlib.import_module('easycls.datasets.' + taskname)

    # prepare dataset and dataloader
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

    print('Evaluate')
    print(f'Evaluating on Validation set:')
    val_acc1, val_acc5 = apis.validate(val_loader, model, criterion, args)
    print(f'[Val] Acc1: {val_acc1:.2f}%\tAcc5: {val_acc5:.2f}%')
    print(f'Evaluating on Test Set:')
    test_acc1, test_acc5 = apis.validate(test_loader, model, criterion, args)
    print(f'[Test] Acc1: {test_acc1:.2f}%\tAcc5: {test_acc5:.2f}%')
    return

if __name__ == "__main__":
    main()