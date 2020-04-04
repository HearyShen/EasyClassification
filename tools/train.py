import os
import time
from argparse import ArgumentParser
import importlib
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


best_acc1 = 0


def worker(args, cfgs: dict):
    global best_acc1

    # init logger
    taskname = cfgs['data'].get('task')
    model_arch = cfgs['model'].get('arch')
    logger = helpers.init_root_logger(filename=os.path.join(
        'logs',
        f"{taskname}_{model_arch}_train_{helpers.format_time(format=r'%Y%m%d-%H%M%S')}.log"
    ))
    logger.info(f"Current task (dataset): '{taskname}'.")

    # init test
    cuda_device_count = helpers.check_cuda()

    # create model
    model_arch = cfgs["model"].get("arch")
    model_kwargs = cfgs["model"].get("kwargs")
    model = easycls.models.__dict__[model_arch](**model_kwargs if model_kwargs else {})
    logger.info(f"Creating model '{model_arch}' with specs: {model_kwargs}.")

    # resume as specified
    start_epoch = 0
    if args.resume:
        try:
            checkpoint = helpers.load_checkpoint(args.resume)
        except FileNotFoundError as error:
            logger.error(
                f"Training failed, no checkpoint found at '{args.resume}'.")
            return
        else:
            start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(
                f"Loaded checkpoint '{args.resume}' (epoch: {checkpoint['epoch']}, time: {helpers.readable_time(checkpoint['timestamp'])})."
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
    optimizer_arch = cfgs['optimizer'].get('arch')
    optimizer_kwargs = cfgs['optimizer'].get('kwargs')
    optimizer = easycls.optims.__dict__[optimizer_arch](
        model.parameters(), **optimizer_kwargs if optimizer_kwargs else {})
    logger.info(
        f"Creating optimizer '{optimizer_arch}' with specs: {optimizer_kwargs}.")
    if args.resume:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # create the lossfunc(loss function)
    loss_arch = cfgs['loss'].get('arch')
    loss_kwargs = cfgs['loss'].get('kwargs')
    lossfunc = easycls.modules.__dict__[loss_arch](
        **loss_kwargs if loss_kwargs else {}).to(torch.device(args.device))
    logger.info(f"Creating loss '{loss_arch}' with specs: {loss_kwargs}.")

    # speedup for batches with fixed-size input
    cudnn.benchmark = cfgs['speed'].get('cudnn_benchmark', False)

    # load dataset for specific task
    # run-time import dataset according to task in config
    task = importlib.import_module('easycls.datasets.' + taskname)

    # prepare dataloader
    train_loader = task.load_trainset(cfgs)
    val_loader = task.load_valset(cfgs)
    test_loader = task.load_testset(cfgs)

    logger.info(f'Start training at {helpers.readable_time()}')
    epoch_time = helpers.AverageMeter('Tepoch', ':.3f', 's')

    # create lr_scheduler to operate lr decay
    lr_scheduler_arch = cfgs['lr_scheduler'].get('arch')
    lr_scheduler_kwargs = cfgs['lr_scheduler'].get('kwargs')
    lr_scheduler = easycls.optims.lr_schedulers.__dict__[lr_scheduler_arch](
        optimizer,
        **lr_scheduler_kwargs if lr_scheduler_kwargs else {},
        last_epoch=start_epoch - 1)
    logger.info(
        f"Creating learning-rate scheduler '{lr_scheduler_arch}' with specs: {lr_scheduler_kwargs}."
    )
    if args.resume:
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])

    total_epoch = cfgs['basic'].get('epochs', 1)
    for epoch in range(start_epoch, total_epoch):
        tic = time.time()
        logger.info(
            f'Starting epoch: {epoch}/{total_epoch}, learning-rate: {lr_scheduler.get_last_lr()}.'
        )

        # train for one epoch
        train_accs, train_loss, train_cms = easycls.apis.train(train_loader, model, lossfunc, optimizer,
                                epoch, args, cfgs)
        logger.info(f"[Eval] [Train] Epoch: {epoch}, {helpers.AverageMeter.str_all(train_accs)}, {train_loss.get_avg_str()}.")
        logger.info(f"[Eval] [Train] Confusion matrices of '{train_cms.num_classes}' classes: \n{train_cms}")

        # evaluate on validation set
        logger.info(f'Evaluating on Validation set:')
        val_accs, val_loss, val_cms = easycls.apis.validate(
            val_loader, model, lossfunc, args, cfgs)
        logger.info(f"[Eval] [Val] Epoch: {epoch}, {helpers.AverageMeter.str_all(val_accs)}, {val_loss.get_avg_str()}.")
        logger.info(f"[Eval] [Val] Confusion matrices of '{val_cms.num_classes}' classes: \n{val_cms}")

        # evaluate on test set
        logger.info(f'Evaluating on Test Set:')
        test_accs, test_loss, test_cms = easycls.apis.validate(
            test_loader, model, lossfunc, args, cfgs)
        logger.info(f"[Eval] [Test] Epoch: {epoch}, {helpers.AverageMeter.str_all(test_accs)}, {test_loss.get_avg_str()}.")
        logger.info(f"[Eval] [Test] Confusion matrices of '{test_accs.num_classes}' classes: \n{test_cms}")

        # adjust the learning rate
        lr_scheduler.step()

        # remember best acc@1 and save checkpoint
        if args.device == 'cuda':
            # To save a DataParallel model generically, save the model.module.state_dict().
            model_state_dict = model.module.state_dict()
        elif args.device == 'cpu':
            model_state_dict = model.state_dict()
        val_acc1 = val_accs[0].avg
        is_best = val_acc1 > best_acc1
        best_acc1 = max(val_acc1, best_acc1)
        helpers.save_checkpoint(
            {
                'arch': model_arch,
                'epoch': epoch + 1,
                'model_state_dict': model_state_dict,
                'best_acc1': best_acc1,
                'optimizer_state_dict': optimizer.state_dict(),
                'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                'timestamp': time.time()
            }, is_best, f"{taskname}_{model_arch}")

        # measure the elapsed time
        toc = time.time()
        epoch_time.update(toc - tic)
        eta_time, time_left = helpers.readable_eta(epoch_time.avg *
                                                   (total_epoch - epoch - 1))
        logger.info(f'{epoch_time}\tETA: {eta_time} (in {time_left})')


if __name__ == "__main__":
    main()