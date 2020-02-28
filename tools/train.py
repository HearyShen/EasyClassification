import os
import time
import datetime
from argparse import ArgumentParser
from configparser import ConfigParser
import importlib
import torch
import torch.backends.cudnn as cudnn
import torch.nn.modules.loss as loss

# insert root dir path to sys.path to import easycls
import sys
sys.path.insert(0,
                os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import easycls.apis as apis
import easycls.models as models
import easycls.helpers as helpers
import easycls.learning as learning


def parse_args():
    argparser = ArgumentParser(description="EasyClassification")
    argparser.add_argument('-c',
                           '--config',
                           default='config.ini',
                           type=str,
                           metavar='PATH',
                           help='configuration file path')
    argparser.add_argument('-r',
                           '--resume',
                           default='',
                           type=str,
                           metavar='PATH',
                           help='resume checkpoint file path (default: none)')
    argparser.add_argument('-d',
                           '--device',
                           default='cuda',
                           choices=('cpu', 'cuda'),
                           type=str,
                           metavar='DEVICE',
                           help='computing device (default: cuda)')
    argparser.add_argument('-l',
                           '--log-freq',
                           default=10,
                           type=int,
                           metavar='N',
                           help='log frequency (default: 10)')
    args = argparser.parse_args()
    return args


def main():
    # parse arguments
    args = parse_args()

    # parse configurations
    cfgs = helpers.parse_cfgs(args.config)

    worker(args, cfgs)


best_acc1 = 0


def worker(args: ArgumentParser, cfgs: ConfigParser):
    global best_acc1

    # init logger
    taskname = cfgs.get('data', 'task')
    arch = cfgs.get('model', 'arch')
    logger = helpers.init_root_logger(filename=os.path.join(
        'logs',
        f"{taskname}_{arch}_train_{helpers.format_time(format=r'%Y%m%d-%H%M%S')}.log"
    ))
    logger.info(f'Current task (dataset): {taskname}')

    # init test
    cuda_device_count = helpers.check_cuda()

    # create model
    if cfgs.getboolean('model', 'pretrained'):
        logger.info(f"Using pre-trained model '{arch}'")
        model = models.__dict__[arch](pretrained=True)
    else:
        logger.info(f"Creating model '{arch}'")
        model = models.__dict__[arch]()

    # resume as specified
    start_epoch = 0
    if args.resume:
        try:
            checkpoint = helpers.load_checkpoint(args.resume)
        except FileNotFoundError as error:
            logger.error(
                f"Training failed, no checkpoint found at '{args.resume}'")
            return
        else:
            start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(
                f"Loaded checkpoint '{args.resume}' (epoch: {checkpoint['epoch']}, time: {helpers.readable_time(checkpoint['timestamp'])})"
            )

    # select the computing device (CPU or GPU) according to arguments and environment
    if args.device == 'cpu' or cuda_device_count == 0:
        model = model.cpu()
        args.device = 'cpu'
        logger.info("Using CPU for training.")
    elif args.device == 'cuda' and cuda_device_count >= 1:
        # use DataParallel to allocate batch data and replicate model to all available GPUs
        # model = model.cuda()      # even on single-gpu node, DataParallel is still slightly falster than non-DataParallel
        model = torch.nn.DataParallel(model).cuda()
        args.device = 'cuda'
        logger.info("Using DataParallel for GPU(s) CUDA accelerated training.")

    # create optimizer after model parameters being in the consistent location
    optimizer = learning.create_optimizer(model.parameters(), cfgs)
    if args.resume:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # create the lossfunc(loss function)
    lossfunc = learning.create_lossfunc(cfgs).to(torch.device(args.device))

    # speedup for batches with fixed-size input
    cudnn.benchmark = cfgs.getboolean('speed', 'cudnn_benchmark')

    # load dataset for specific task
    # run-time import dataset according to task in config
    task = importlib.import_module('easycls.datasets.' + taskname)
    batch_size = cfgs.getint('learning', 'batch_size')
    dataload_workers = cfgs.getint('speed', 'dataload_workers')

    # prepare dataset and dataloader
    train_dataset = task.get_train_dataset(cfgs)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=dataload_workers,
                                               pin_memory=True,
                                               sampler=None)

    val_dataset = task.get_val_dataset(cfgs)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             num_workers=dataload_workers,
                                             pin_memory=True)

    test_dataset = task.get_test_dataset(cfgs)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=dataload_workers,
                                              pin_memory=True)

    logger.info(f'Start training at {helpers.readable_time()}')
    epoch_time = helpers.AverageMeter('Tepoch', ':.3f', 's')

    # create lr_scheduler to operate lr decay
    scheduler = learning.create_lr_scheduler(optimizer,
                                             cfgs,
                                             last_epoch=start_epoch - 1)
    if args.resume:
        scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])

    total_epoch = cfgs.getint('learning', 'epochs')
    for epoch in range(start_epoch, total_epoch):
        tic = time.time()
        # lr = helpers.adjust_learning_rate(optimizer, epoch, cfgs)
        logger.info(
            f'Epoch: {epoch},\tLearning-rate: {scheduler.get_last_lr()},\tBatch-size: {batch_size}'
        )

        # train for one epoch
        train_loss = apis.train(train_loader, model, lossfunc, optimizer,
                                epoch, args)

        # evaluate on validation set
        logger.info(f'Evaluating on Validation set:')
        val_acc1, val_acc5, val_loss, val_cms = apis.validate(
            val_loader, model, lossfunc, args, cfgs)
        logger.info(
            f'[Val] Epoch: {epoch},\tAcc1: {val_acc1:.2f}%,\tAcc5: {val_acc5:.2f}%'
        )

        # evaluate on test set
        logger.info(f'Evaluating on Test Set:')
        test_acc1, test_acc5, test_loss, test_cms = apis.validate(
            test_loader, model, lossfunc, args, cfgs)
        logger.info(
            f'[Test] Epoch: {epoch},\tAcc1: {test_acc1:.2f}%,\tAcc5: {test_acc5:.2f}%'
        )

        # adjust the learning rate
        scheduler.step()

        # remember best acc@1 and save checkpoint
        if args.device == 'cuda':
            # To save a DataParallel model generically, save the model.module.state_dict().
            model_state_dict = model.module.state_dict()
        elif args.device == 'cpu':
            model_state_dict = model.state_dict()
        is_best = val_acc1 > best_acc1
        best_acc1 = max(val_acc1, best_acc1)
        helpers.save_checkpoint(
            {
                'arch': arch,
                'epoch': epoch + 1,
                'model_state_dict': model_state_dict,
                'best_acc1': best_acc1,
                'optimizer_state_dict': optimizer.state_dict(),
                'lr_scheduler_state_dict': scheduler.state_dict(),
                'timestamp': time.time()
            }, is_best, f"{taskname}_{arch}")

        # measure the elapsed time
        toc = time.time()
        epoch_time.update(toc - tic)
        eta_time, time_left = helpers.readable_eta(epoch_time.avg *
                                                   (total_epoch - epoch - 1))
        logger.info(f'{epoch_time}\tETA: {eta_time} (in {time_left})')


if __name__ == "__main__":
    main()