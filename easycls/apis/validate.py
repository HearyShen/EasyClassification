import time
import torch

from ..helpers import AverageMeter, ProgressMeter, accuracy, ConfusionMatrix

def validate(val_loader, model, criterion, args, cfgs):
    batch_time = AverageMeter('Time', ':6.3f', 's')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f', '%')
    top5 = AverageMeter('Acc@5', ':6.2f', '%')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    num_classes = cfgs.getint("data", "num-classes")
    confusion_matrices = [ConfusionMatrix(index) for index in range(num_classes)]

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (inputs, targets) in enumerate(val_loader):
            targets = targets.cuda(non_blocking=True)

            # compute outputs
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(acc1*100, inputs.size(0))
            top5.update(acc5*100, inputs.size(0))

            # update all classes' confusion matrices
            ConfusionMatrix.update_all(confusion_matrices, outputs, targets)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        # print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
        #       .format(top1=top1, top5=top5))
        for cm in confusion_matrices:
            print(cm)

    return top1.avg, top5.avg
