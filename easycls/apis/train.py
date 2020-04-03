import time
from ..helpers import AverageMeter, ProgressMeter, accuracy, MultiConfusionMatrices, init_module_logger

logger = init_module_logger(__name__)

def train(train_loader, model, lossfunc, optimizer, epoch, args, cfgs):
    """
    Train the model for an epoch with train dataloader.

    Args:
        train_loader (DataLoader): loading data for training.
        model (Model): PyTorch model, model to be trained.
        lossfunc (Loss): loss function.
        optimizer (Optimizer): optimizer algorithm to optimize model's parameters.
        epoch (int): current epoch (starts from 0).
        args (ArgumentParser): arguments from commandline inputs.

    Returns:
        loss (float): average of current epoch's training loss.
    """
    batch_time = AverageMeter('Tbatch', ':6.3f', 's')
    data_time = AverageMeter('Tdata', ':6.3f', 's')
    losses = AverageMeter('Loss', ':.4e')
    topk_options = cfgs['basic'].get('topk_accs', [1])
    topk_accs = [AverageMeter(f'Acc@{k}', ':6.2f', '%') for k in topk_options]
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses] + topk_accs,
        prefix="Epoch: [{}]".format(epoch))

    num_classes = cfgs["model"]["kwargs"].get("num_classes")
    mcms = MultiConfusionMatrices(num_classes)

    # switch to train mode
    model.train()

    end = time.time()
    for i, (inputs, targets) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # compute outputs
        outputs = model(inputs)
        targets = targets.to(outputs.device, non_blocking=True)
        loss = lossfunc(outputs, targets)

        # measure accuracy and record loss
        accs = accuracy(outputs, targets, topk=topk_options)
        batch_size = targets.size(0)
        losses.update(loss.item(), batch_size)
        for accs_idx in range(len(topk_accs)):
            topk_accs[accs_idx].update(accs[accs_idx] * 100, batch_size)

        # update all classes' confusion matrices
        scores, predictions = outputs.max(dim=1)
        mcms.update(predictions, targets)

        # compute gradient and do optimizing step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.log_freq == 0:
            logger.info(progress.batch_str(i))
    
    return topk_accs, losses, mcms