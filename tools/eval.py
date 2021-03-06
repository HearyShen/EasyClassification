import os
from argparse import ArgumentParser
import importlib
import random
import torch
import torch.backends.cudnn as cudnn

# insert root dir path to sys.path to import easycls
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))

import easycls
import easycls.helpers as helpers


def parse_args():
    argparser = ArgumentParser(description="EasyClassification")
    argparser.add_argument('-c',
                           '--config',
                           default='config.yml',
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


def worker(args, cfgs: dict):
    # init logger
    taskname = cfgs['data'].get('task')
    model_arch = cfgs['model'].get('arch')
    logger = helpers.init_root_logger(filename=os.path.join(
        'logs',
        f"{taskname}_{model_arch}_eval_{helpers.format_time(format=r'%Y%m%d-%H%M%S')}.log"
    ))
    logger.info(f"Current task (dataset): '{taskname}'.")

    # init test
    cuda_device_count = helpers.check_cuda()

    # set random seed
    random_seed = cfgs['basic'].get('random_seed')
    if random_seed:
        logger.info(f"Set random seed: {random_seed}")
        random.seed(random_seed)
        torch.manual_seed(random_seed)
        cudnn.deterministic = True
        logger.warning('You have chosen to seed training. '
                       'This will turn on the CUDNN deterministic setting, '
                       'which can slow down your training considerably! '
                       'You may see unexpected behavior when restarting '
                       'from checkpoints.')

    # create model
    model_arch = cfgs["model"].get("arch")
    model_kwargs = cfgs["model"].get("kwargs")
    logger.info(f"Creating model '{model_arch}' with specs: {model_kwargs}.")
    model = easycls.models.__dict__[model_arch](**model_kwargs if model_kwargs else {})
    logger.info(f"Model arch: \n{model}")

    # resume as specified
    if args.resume:
        try:
            checkpoint = helpers.load_checkpoint(args.resume)
        except FileNotFoundError as error:
            logger.error(f"Evaluation failed, no checkpoint found at '{args.resume}'.")
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
        logger.info("Using DataParallel for GPU(s) CUDA accelerated evaluating.")

    # create the lossfunc(loss function)
    loss_arch = cfgs['loss'].get('arch')
    loss_kwargs = cfgs['loss'].get('kwargs')
    logger.info(f"Creating loss '{loss_arch}' with specs: {loss_kwargs}.")
    lossfunc = easycls.modules.__dict__[loss_arch](
        **loss_kwargs if loss_kwargs else {}).to(torch.device(args.device))

    # speedup for batches with fixed-size input
    cudnn.benchmark = cfgs['speed'].get('cudnn_benchmark', False)

    # load dataset for specific task
    # run-time import dataset according to task in config
    task = importlib.import_module('easycls.datasets.' + taskname)

    # prepare dataloader
    val_loader = task.load_valset(cfgs)
    test_loader = task.load_testset(cfgs)

    logger.info(f'Start Evaluating at {helpers.readable_time()}')
    # evaluate on validation set
    logger.info(f'Evaluating on Validation set:')
    val_accs, val_loss, val_cms = easycls.apis.validate(
        val_loader, model, lossfunc, args, cfgs)
    logger.info(f"[Eval] [Val] {helpers.AverageMeter.str_all(val_accs)}, {val_loss.get_avg_str()}.")
    logger.info(f"[Eval] [Val] Confusion matrices of '{val_cms.num_classes}' classes: \n{val_cms}")
    
    # evaluate on test set
    logger.info(f'Evaluating on Test Set:')
    test_accs, test_loss, test_cms = easycls.apis.validate(
        test_loader, model, lossfunc, args, cfgs)
    logger.info(f"[Eval] [Test] {helpers.AverageMeter.str_all(test_accs)}, {test_loss.get_avg_str()}.")
    logger.info(f"[Eval] [Test] Confusion matrices of '{test_cms.num_classes}' classes: \n{test_cms}")


if __name__ == "__main__":
    main()