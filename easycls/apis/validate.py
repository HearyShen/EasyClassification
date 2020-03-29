import time
import torch

from ..helpers import AverageMeter, ProgressMeter, accuracy, ConfusionMatrix, init_module_logger

logger = init_module_logger(__name__)

def validate(val_loader, model, lossfunc, args, cfgs):
    """
    Validate the model's performance.

    Args:
        val_loader (DataLoader): loading data for validation.
        model (Model): PyTorch model, model to be validated.
        lossfunc (Loss): loss function.
        args (ArgumentParser): arguments from commandline inputs.
        cfgs (dict): configurations from config file.

    Returns:
        top-1 accuracy
        top-5 accuracy
        loss
        confusion matrices
    """
    batch_time = AverageMeter('Time', ':6.3f', 's')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f', '%')
    top5 = AverageMeter('Acc@5', ':6.2f', '%')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    num_classes = cfgs["model"]["kwargs"].get("num_classes")
    confusion_matrices = [ConfusionMatrix(index) for index in range(num_classes)]

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (inputs, targets) in enumerate(val_loader):
            # compute outputs
            outputs = model(inputs)
            targets = targets.to(outputs.device, non_blocking=True)
            loss = lossfunc(outputs, targets)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            batch_size = targets.size(0)
            losses.update(loss.item(), batch_size)
            top1.update(acc1*100, batch_size)
            top5.update(acc5*100, batch_size)

            # update all classes' confusion matrices
            ConfusionMatrix.update_all(confusion_matrices, outputs, targets)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.log_freq == 0:
                logger.info(progress.batch_str(i))

    return top1.avg, top5.avg, losses.avg, confusion_matrices
