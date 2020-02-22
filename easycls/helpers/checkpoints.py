import os
import shutil
import torch

from .logs import init_module_logger

logger = init_module_logger(__name__)

LATEST_CKPT_EXT = r'.latest.pt'
BEST_CKPT_EXT = r'.best.pt'


def save_checkpoint(obj, is_best, filename="model"):
    """
    Save latest checkpoint and best-performance checkpoint.
    """
    # save latest checkpoint
    latest_ckpt_filename = filename + LATEST_CKPT_EXT
    torch.save(obj, latest_ckpt_filename)
    logger.info(f"Latest checkpoint saved to {latest_ckpt_filename}")

    # copy best checkpoint
    if is_best:
        best_ckpt_filename = filename + BEST_CKPT_EXT
        shutil.copyfile(filename, best_ckpt_filename)
        logger.info(f"Best checkpoint saved to {best_ckpt_filename}")

def load_checkpoint(filename, map_location=None):
    """
    Load checkpoint from a file.
    """
    if os.path.isfile(filename):
        logger.info(f"Loading checkpoint '{filename}'")
        checkpoint = torch.load(filename, map_location=map_location)

        return checkpoint
    else:
        logger.error(f"No checkpoint found at '{filename}'")
        raise FileNotFoundError(f"No such file at {os.path.abspath(filename)}")