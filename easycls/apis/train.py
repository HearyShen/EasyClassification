import time
from ..helpers import AverageMeter, ProgressMeter, accuracy, init_module_logger

logger = init_module_logger(__name__)

def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Tbatch', ':6.3f', 's')
    data_time = AverageMeter('Tdata', ':6.3f', 's')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f', '%')
    top5 = AverageMeter('Acc@5', ':6.2f', '%')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda(non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1*100, images.size(0))
        top5.update(acc5*100, images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.log_freq == 0:
            logger.info(progress.batch_str(i))