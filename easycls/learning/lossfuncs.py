from configparser import ConfigParser
import torch.nn.modules.loss as loss
from ..helpers import init_module_logger

logger = init_module_logger(__name__)

def create_lossfunc(cfgs:ConfigParser):
    """
    Create loss function according configs.
    """
    valid_lossfuncs = [item for item in loss.__dict__ if not item.startswith("_") and ('Loss' in item) and callable(loss.__dict__[item])]

    lossfunc_name = cfgs.get("learning", "loss")

    try:
        lossfunc = loss.__dict__[lossfunc_name]()
    except KeyError as error:
        logger.error(f"Valid loss functions are: {valid_lossfuncs}.")
        return
    else:
        return lossfunc