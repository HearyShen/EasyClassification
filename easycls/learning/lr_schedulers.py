"""
This module provides functions that create learning-rate 
schedulers based on configurations.
"""
import torch.optim.lr_scheduler as lr_scheduler
from ..helpers import init_module_logger

logger = init_module_logger(__name__)

CONFIG_SECTION = "lr_scheduler"


def create_LambdaLR(optimizer, cfgs: dict, last_epoch=-1):
    lr_lambda = lambda epoch: 0.1**(epoch // 30)

    return lr_scheduler.LambdaLR(optimizer,
                                 lr_lambda=lr_lambda,
                                 last_epoch=last_epoch)


def create_MultiplicativeLR(optimizer, cfgs: dict, last_epoch=-1):
    lr_lambda = lambda epoch: cfgs[CONFIG_SECTION].get("multiplicative_factor")

    return lr_scheduler.MultiplicativeLR(optimizer,
                                         lr_lambda=lr_lambda,
                                         last_epoch=last_epoch)


def create_StepLR(optimizer, cfgs: dict, last_epoch=-1):
    step_size = cfgs[CONFIG_SECTION].get("step_size")
    gamma = cfgs[CONFIG_SECTION].get("gamma", 0.1)

    return lr_scheduler.StepLR(optimizer,
                               step_size=step_size,
                               gamma=gamma,
                               last_epoch=last_epoch)


def create_MultiStepLR(optimizer, cfgs: dict, last_epoch=-1):
    milestones = cfgs[CONFIG_SECTION].get("milestones")
    gamma = cfgs[CONFIG_SECTION].get("gamma", 0.1)

    return lr_scheduler.MultiStepLR(optimizer,
                                    milestones=milestones,
                                    gamma=gamma,
                                    last_epoch=last_epoch)


def create_ExponentialLR(optimizer, cfgs: dict, last_epoch=-1):
    gamma = cfgs[CONFIG_SECTION].get("gamma")

    return lr_scheduler.ExponentialLR(optimizer,
                                      gamma=gamma,
                                      last_epoch=last_epoch)


def create_CosineAnnealingLR(optimizer, cfgs: dict, last_epoch=-1):
    T_max = cfgs[CONFIG_SECTION].get("T_max")
    eta_min = cfgs[CONFIG_SECTION].get("eta_min", 0)

    return lr_scheduler.CosineAnnealingLR(optimizer,
                                          T_max=T_max,
                                          eta_min=eta_min,
                                          last_epoch=last_epoch)


def create_ReduceLROnPlateau(optimizer, cfgs: dict):
    mode = cfgs[CONFIG_SECTION].get("mode", "min")
    factor = cfgs[CONFIG_SECTION].get("factor", 0.1)
    patience = cfgs[CONFIG_SECTION].get("patience", 10)
    verbose = cfgs[CONFIG_SECTION].get("verbose", False)
    threshold = cfgs[CONFIG_SECTION].get("threshold", 1e-4)
    threshold_mode = cfgs[CONFIG_SECTION].get("threshold_mode", "rel")
    cooldown = cfgs[CONFIG_SECTION].get("cooldown", 0)
    min_lr = cfgs[CONFIG_SECTION].get("min_lr", 0)
    eps = cfgs[CONFIG_SECTION].get("eps", 1e-8)

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


def create_CyclicLR(optimizer, cfgs: dict, last_epoch=-1):
    base_lr = cfgs[CONFIG_SECTION].get("base_lr")
    max_lr = cfgs[CONFIG_SECTION].get("max_lr")
    step_size_up = cfgs[CONFIG_SECTION].get("step_size_up", 2000)
    step_size_down = cfgs[CONFIG_SECTION].get("step_size_down", None)
    mode = cfgs[CONFIG_SECTION].get("mode", "triangular")
    gamma = cfgs[CONFIG_SECTION].get("gamma", 1.0)
    # scale_fn and scale_mode
    cycle_momentum = cfgs[CONFIG_SECTION].get("cycle_momentum", True)
    base_momentum = cfgs[CONFIG_SECTION].get("base_momentum", 0.8)
    max_momentum = cfgs[CONFIG_SECTION].get("max_momentum", 0.9)

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


def create_OneCycleLR(optimizer, cfgs: dict, last_epoch=-1):
    max_lr = cfgs[CONFIG_SECTION].get("max_lr")
    total_steps = cfgs[CONFIG_SECTION].get("total_steps", None)
    epochs = cfgs[CONFIG_SECTION].get("epochs", None)
    steps_per_epoch = cfgs[CONFIG_SECTION].get("steps_per_epoch", None)
    pct_start = cfgs[CONFIG_SECTION].get("pct_start", 0.3)
    anneal_strategy = cfgs[CONFIG_SECTION].get("anneal_strategy", "cos")
    cycle_momentum = cfgs[CONFIG_SECTION].get("cycle_momentum", True)
    base_momentum = cfgs[CONFIG_SECTION].get("base_momentum", 0.85)
    max_momentum = cfgs[CONFIG_SECTION].get("max_momentum", 0.95)
    div_factor = cfgs[CONFIG_SECTION].get("div_factor", 25)
    final_div_factor = cfgs[CONFIG_SECTION].get("final_div_factor", 1e4)

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
                                       cfgs: dict,
                                       last_epoch=-1):
    T_0 = cfgs[CONFIG_SECTION].get("T_0")
    T_mult = cfgs[CONFIG_SECTION].get("T_mult", 1)
    eta_min = cfgs[CONFIG_SECTION].get("eta_min", 0)

    return lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                    T_0=T_0,
                                                    T_mult=T_mult,
                                                    eta_min=eta_min,
                                                    last_epoch=last_epoch)


def create_lr_scheduler(optimizer, cfgs: dict, last_epoch=-1):
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
        cfgs (dict): parsed configurations from configuration file.
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

    lr_decay_mode = cfgs[CONFIG_SECTION].get("decay_mode")
    logger.info(f"Using learning-rate decay mode '{lr_decay_mode}'.")

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
