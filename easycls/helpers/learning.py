def adjust_learning_rate(optimizer, epoch, cfgs):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = cfgs['learning'].get('learning-rate') * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    return lr
