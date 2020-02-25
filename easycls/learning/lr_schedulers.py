from configparser import ConfigParser
import torch.optim.lr_scheduler as lr_scheduler
from ..helpers import init_module_logger

logger = init_module_logger(__name__)


def create_lr_scheduler(cfgs:ConfigParser):
    """
    Create LR scheduler according to the config.
    """
    valid_lr_schedulers = [item for item in lr_scheduler.__dict__ if not item.startswith("_") and ('LR' in item) and callable(lr_scheduler.__dict__[item])]

    lr_decay_mode = cfgs.get("learning", "lr-decay-mode")

    if lr_decay_mode not in valid_lr_schedulers:
        logger.error(f"Valid Learning-rate schedulaers are: {valid_lr_schedulers}.")
        return
    
