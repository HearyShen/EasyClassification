"""
This module provides functions that create learning-rate 
schedulers based on configurations.
"""
from configparser import ConfigParser
import torch.optim.lr_scheduler as lr_scheduler
from ..helpers import init_module_logger

logger = init_module_logger(__name__)

CONFIG_SECTION = "lr_scheduler"


def create_LambdaLR(optimizer, cfgs: ConfigParser, last_epoch=-1):
    lr_lambda = lambda epoch: 0.1**(epoch // 30)

    return lr_scheduler.LambdaLR(optimizer,
                                 lr_lambda=lr_lambda,
                                 last_epoch=last_epoch)


def create_MultiplicativeLR(optimizer, cfgs: ConfigParser, last_epoch=-1):
    lr_lambda = lambda epoch: cfgs.getfloat(CONFIG_SECTION,
                                            "multiplicative_factor")

    return lr_scheduler.MultiplicativeLR(optimizer,
                                         lr_lambda=lr_lambda,
                                         last_epoch=last_epoch)


def create_StepLR(optimizer, cfgs: ConfigParser, last_epoch=-1):
    step_size = cfgs.getint(CONFIG_SECTION, "step_size")
    gamma = cfgs.getfloat(CONFIG_SECTION, "gamma", fallback=0.1)

    return lr_scheduler.StepLR(optimizer,
                               step_size=step_size,
                               gamma=gamma,
                               last_epoch=last_epoch)


def create_MultiStepLR(optimizer, cfgs: ConfigParser, last_epoch=-1):
    milestones = [
        int(num)
        for num in cfgs.get(CONFIG_SECTION, "milestones").strip().lstrip(
            '[').rstrip(']').split()
    ]
    gamma = cfgs.getfloat(CONFIG_SECTION, "gamma", fallback=0.1)

    return lr_scheduler.MultiStepLR(optimizer,
                                    milestones=milestones,
                                    gamma=gamma,
                                    last_epoch=last_epoch)


def create_ExponentialLR(optimizer, cfgs: ConfigParser, last_epoch=-1):
    gamma = cfgs.getfloat(CONFIG_SECTION, "gamma")

    return lr_scheduler.ExponentialLR(optimizer,
                                      gamma=gamma,
                                      last_epoch=last_epoch)


def create_CosineAnnealingLR(optimizer, cfgs: ConfigParser, last_epoch=-1):
    T_max = cfgs.getint(CONFIG_SECTION, "T_max")
    eta_min = cfgs.getfloat(CONFIG_SECTION, "eta_min", fallback=0)

    return lr_scheduler.CosineAnnealingLR(optimizer,
                                          T_max=T_max,
                                          eta_min=eta_min,
                                          last_epoch=last_epoch)


def create_ReduceLROnPlateau(optimizer, cfgs: ConfigParser):
    mode = cfgs.get(CONFIG_SECTION, "mode", fallback="min")
    factor = cfgs.getfloat(CONFIG_SECTION, "factor", fallback=0.1)
    patience = cfgs.getint(CONFIG_SECTION, "patience", fallback=10)
    verbose = cfgs.getboolean(CONFIG_SECTION, "verbose", fallback=False)
    threshold = cfgs.getfloat(CONFIG_SECTION, "threshold", fallback=1e-4)
    threshold_mode = cfgs.get(CONFIG_SECTION, "threshold_mode", fallback="rel")
    cooldown = cfgs.getint(CONFIG_SECTION, "cooldown", fallback=0)
    min_lr = cfgs.getfloat(CONFIG_SECTION, "min_lr", fallback=0)
    eps = cfgs.getfloat(CONFIG_SECTION, "eps", fallback=1e-8)

    return lr_scheduler.ReduceLROnPlateau(optimizer,
                                          mode=mode,
                                          factor=factor,
                                          patience=patience,
                                          verbose=verbose,
                                          threshold=threshold,
                                          threshold_mode=threshold_mode,
                                          cooldown=cooldown,
                                          min_lr=min_lr,
                                          eps=eps)


def create_CyclicLR(optimizer, cfgs: ConfigParser, last_epoch=-1):
    base_lr = cfgs.getfloat(CONFIG_SECTION, "base_lr")
    max_lr = cfgs.getfloat(CONFIG_SECTION, "max_lr")
    step_size_up = cfgs.getint(CONFIG_SECTION, "step_size_up", fallback=2000)
    step_size_down = cfgs.getint(CONFIG_SECTION,
                                 "step_size_down",
                                 fallback=None)
    mode = cfgs.get(CONFIG_SECTION, "mode", fallback="triangular")
    gamma = cfgs.getfloat(CONFIG_SECTION, "gamma", fallback=1.0)
    # scale_fn and scale_mode
    cycle_momentum = cfgs.getboolean(CONFIG_SECTION,
                                     "cycle_momentum",
                                     fallback=True)
    base_momentum = cfgs.getfloat(CONFIG_SECTION,
                                  "base_momentum",
                                  fallback=0.8)
    max_momentum = cfgs.getfloat(CONFIG_SECTION, "max_momentum", fallback=0.9)

    return lr_scheduler.CyclicLR(optimizer,
                                 base_lr=base_lr,
                                 max_lr=max_lr,
                                 step_size_up=step_size_up,
                                 step_size_down=step_size_down,
                                 mode=mode,
                                 gamma=gamma,
                                 cycle_momentum=cycle_momentum,
                                 base_momentum=base_momentum,
                                 max_momentum=max_momentum,
                                 last_epoch=last_epoch)


def create_OneCycleLR(optimizer, cfgs: ConfigParser, last_epoch=-1):
    max_lr = cfgs.getfloat(CONFIG_SECTION, "max_lr")
    total_steps = cfgs.getint(CONFIG_SECTION, "total_steps", fallback=None)
    epochs = cfgs.getint(CONFIG_SECTION, "epochs", fallback=None)
    steps_per_epoch = cfgs.getint(CONFIG_SECTION,
                                  "steps_per_epoch",
                                  fallback=None)
    pct_start = cfgs.getfloat(CONFIG_SECTION, "pct_start", fallback=0.3)
    anneal_strategy = cfgs.get(CONFIG_SECTION,
                               "anneal_strategy",
                               fallback="cos")
    cycle_momentum = cfgs.getboolean(CONFIG_SECTION,
                                     "cycle_momentum",
                                     fallback=True)
    base_momentum = cfgs.getfloat(CONFIG_SECTION,
                                  "base_momentum",
                                  fallback=0.85)
    max_momentum = cfgs.getfloat(CONFIG_SECTION, "max_momentum", fallback=0.95)
    div_factor = cfgs.getfloat(CONFIG_SECTION, "div_factor", fallback=25)
    final_div_factor = cfgs.getfloat(CONFIG_SECTION,
                                     "final_div_factor",
                                     fallback=1e4)

    return lr_scheduler.OneCycleLR(optimizer,
                                   max_lr=max_lr,
                                   total_steps=total_steps,
                                   epochs=epochs,
                                   steps_per_epoch=steps_per_epoch,
                                   pct_start=pct_start,
                                   anneal_strategy=anneal_strategy,
                                   cycle_momentum=cycle_momentum,
                                   base_momentum=base_momentum,
                                   max_momentum=max_momentum,
                                   div_factor=div_factor,
                                   final_div_factor=final_div_factor,
                                   last_epoch=last_epoch)


def create_CosineAnnealingWarmRestarts(optimizer,
                                       cfgs: ConfigParser,
                                       last_epoch=-1):
    T_0 = cfgs.getint(CONFIG_SECTION, "T_0")
    T_mult = cfgs.getint(CONFIG_SECTION, "T_mult", fallback=1)
    eta_min = cfgs.getfloat(CONFIG_SECTION, "eta_min", fallback=0)

    return lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                    T_0=T_0,
                                                    T_mult=T_mult,
                                                    eta_min=eta_min,
                                                    last_epoch=last_epoch)


def create_lr_scheduler(optimizer, cfgs: ConfigParser, last_epoch=-1):
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
    'OneCycleLR',
    'CosineAnnealingWarmRestarts'

    Args:
        optimizer (Optimizer): an optimizer.
        cfgs (ConfigParser): parsed configurations from configuration file.
        last_epoch (int): The index of the last batch. This parameter is used 
            when resuming a training job. Since step() should be invoked after 
            each batch instead of after each epoch, this number represents the 
            total number of batches computed, not the total number of epochs 
            computed. When last_epoch=-1, the schedule is started from the 
            beginning. Default: -1

    Returns:
        The configurated learning-rate scheduler.

    Raises:
        KeyError, if decay_mode is not supported.
    ```
    """

    lr_decay_mode = cfgs.get(CONFIG_SECTION, "decay_mode")

    if lr_decay_mode == "LambdaLR":
        return create_LambdaLR(optimizer, cfgs, last_epoch)
    elif lr_decay_mode == "MultiplicativeLR":
        return create_MultiplicativeLR(optimizer, cfgs, last_epoch)
    elif lr_decay_mode == "StepLR":
        return create_StepLR(optimizer, cfgs, last_epoch)
    elif lr_decay_mode == "MultiStepLR":
        return create_MultiStepLR(optimizer, cfgs, last_epoch)
    elif lr_decay_mode == "ExponentialLR":
        return create_ExponentialLR(optimizer, cfgs, last_epoch)
    elif lr_decay_mode == "CosineAnnealingLR":
        return create_CosineAnnealingLR(optimizer, cfgs, last_epoch)
    elif lr_decay_mode == "ReduceLROnPlateau":
        return create_ReduceLROnPlateau(optimizer, cfgs)
    elif lr_decay_mode == "CyclicLR":
        return create_CyclicLR(optimizer, cfgs, last_epoch)
    elif lr_decay_mode == "OneCycleLR":
        return create_OneCycleLR(optimizer, cfgs, last_epoch)
    elif lr_decay_mode == "CosineAnnealingWarmRestarts":
        return create_CosineAnnealingWarmRestarts(optimizer, cfgs, last_epoch)
    else:
        error_str = f"Learning-rate decay mode '{lr_decay_mode}' in configuration is not supported."
        logger.error(error_str)
        raise KeyError(error_str)
