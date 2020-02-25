from configparser import ConfigParser
import torch.optim.lr_scheduler as lr_scheduler
from ..helpers import init_module_logger

logger = init_module_logger(__name__)


def create_LambdaLR(optimizer, cfgs:ConfigParser):
    lr_lambda = lambda epoch: 0.1 ** (epoch // 30)
    return lr_scheduler.LambdaLR(optimizer, lr_lambda)

def create_MultiplicativeLR(optimizer, cfgs:ConfigParser):
    lr_lambda = lambda epoch: cfgs.getfloat("learning", "lr-multiplicative-factor")
    return lr_scheduler.MultiplicativeLR(optimizer, lr_lambda)

def create_StepLR(optimizer, cfgs:ConfigParser):
    step = cfgs.getint("learning", "lr-step", fallback=10)
    gamma = cfgs.getfloat("learning", "lr-gamma", fallback=0.1)
    return lr_scheduler.StepLR(optimizer, step, gamma)

def create_MultiStepLR(optimizer, cfgs:ConfigParser):
    milestones = [int(num) for num in cfgs.get("learning", "lr-milestones").strip().lstrip('[').rstrip(']').split()]
    gamma = cfgs.getfloat("learning", "lr-gamma", fallback=0.1)
    return lr_scheduler.MultiStepLR(optimizer, milestones, gamma)

def create_ExponentialLR(optimizer, cfgs:ConfigParser):
    gamma = cfgs.getfloat("learning", "lr-gamma")
    return lr_scheduler.ExponentialLR(optimizer, gamma)

def create_CosineAnnealingLR(optimizer, cfgs:ConfigParser):
    Tmax = cfgs.getint("learning", "lr-Tmax")
    eta_min = cfgs.getfloat("learning", "lr-min", fallback=0)
    return lr_scheduler.CosineAnnealingLR(optimizer, Tmax, eta_min)

def create_ReduceLROnPlateau(optimizer, cfgs:ConfigParser):
    # TODO: finish ReduceLROnPlateau
    return lr_scheduler.ReduceLROnPlateau(optimizer)

def create_lr_scheduler(optimizer, cfgs:ConfigParser):
    """
    Create LR scheduler according to the config.

    Avaliable learning-rate schedulers are:

    ```
    'LambdaLR', 
    'MultiplicativeLR', 
    'StepLR', 
    'MultiStepLR', 
    'ExponentialLR', 
    'CosineAnnealingLR', 
    'ReduceLROnPlateau', 
    'CyclicLR', 
    'OneCycleLR'
    ```
    """
    valid_lr_schedulers = [item for item in lr_scheduler.__dict__ if not item.startswith("_") and ('LR' in item) and callable(lr_scheduler.__dict__[item])]

    lr_decay_mode = cfgs.get("learning", "lr-decay-mode")

    if lr_decay_mode not in valid_lr_schedulers:
        logger.error(f"Valid Learning-rate schedulers are: {valid_lr_schedulers}.")
        return
    
    if lr_decay_mode == "LambdaLR":
        return create_LambdaLR(optimizer, cfgs)
    elif lr_decay_mode == "MultiplicativeLR":
        return create_MultiplicativeLR(optimizer, cfgs)
    elif lr_decay_mode == "StepLR":
        return create_StepLR(optimizer, cfgs)
    elif lr_decay_mode == "MultiStepLR":
        return create_MultiStepLR(optimizer, cfgs)
    elif lr_decay_mode == "ExponentialLR":
        return create_ExponentialLR(optimizer, cfgs)
    elif lr_decay_mode == "CosineAnnealingLR":
        return create_CosineAnnealingLR(optimizer, cfgs)
    elif lr_decay_mode == "ReduceLROnPlateau":
        pass    # TODO: finish ReduceLROnPlateau
    