import os
import time
from argparse import ArgumentParser
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


def worker(args: ArgumentParser, cfgs: dict):
    # init logger
    taskname = cfgs['data'].get('task')
    arch = cfgs['model'].get('arch')
    logger = helpers.init_root_logger(filename=os.path.join(
        'logs',
        f"{taskname}_{arch}_eval_{helpers.format_time(format=r'%Y%m%d-%H%M%S')}.log"
    ))
    logger.info(f"Current task (dataset): '{taskname}'.")

    # init test
    cuda_device_count = helpers.check_cuda()

    # create model
    if cfgs['model'].get('pretrained'):
        logger.info(f"Using pre-trained model '{arch}'.")
        model = models.__dict__[arch](pretrained=True)
    else:
        logger.info(f"Creating model '{arch}'.")
        model = models.__dict__[arch]()

    # resume as specified
    if args.resume:
        try:
            checkpoint = helpers.load_checkpoint(args.resume)
        except FileNotFoundError as error:
            logger.error(
                f"Evaluation failed, no checkpoint found at '{args.resume}'.")
            return
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(
                f"Loaded checkpoint '{args.resume}' (epoch: {checkpoint['epoch']}, time: {helpers.readable_time(checkpoint['timestamp'])})."
            )

    # select the computing device (CPU or GPU) according to arguments and environment
    if args.device == 'cpu' or cuda_device_count == 0:
        model = model.cpu()
        args.device = 'cpu'
        logger.info("Using CPU for evaluating.")
    elif args.device == 'cuda' and cuda_device_count >= 1:
        # use DataParallel to allocate batch data and replicate model to all available GPUs
        # model = model.cuda()      # even on single-gpu node, DataParallel is still slightly falster than non-DataParallel
        model = torch.nn.DataParallel(model).cuda()
        args.device = 'cuda'
        logger.info(
            "Using DataParallel for GPU(s) CUDA accelerated evaluating.")

    # create the lossfunc(loss function)
    lossfunc = learning.create_lossfunc(cfgs).to(torch.device(args.device))

    # speedup for batches with fixed-size input
    cudnn.benchmark = cfgs['speed'].get('cudnn_benchmark', False)

    # load dataset for specific task
    # run-time import dataset according to task in config
    task = importlib.import_module('easycls.datasets.' + taskname)

    # prepare dataloader
    val_loader = task.load_valset(cfgs)
    test_loader = task.load_testset(cfgs)

    logger.info(f'Start Evaluating at {helpers.readable_time()}')
    logger.info(f'Evaluating on Validation set:')
    val_acc1, val_acc5, val_loss, val_cms = apis.validate(
        val_loader, model, lossfunc, args, cfgs)
    logger.info(f'[Val] Acc1: {val_acc1:.2f}%, Acc5: {val_acc5:.2f}%.')
    # logger.info(helpers.ConfusionMatrix.str_all(val_cms))
    logger.info(f'Evaluating on Test Set:')
    test_acc1, test_acc5, test_loss, test_cms = apis.validate(
        test_loader, model, lossfunc, args, cfgs)
    logger.info(f'[Test] Acc1: {test_acc1:.2f}%, Acc5: {test_acc5:.2f}%.')
    # logger.info(helpers.ConfusionMatrix.str_all(test_cms))


if __name__ == "__main__":
    main()