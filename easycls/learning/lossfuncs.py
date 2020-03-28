import torch.nn.modules.loss as loss
from ..helpers import init_module_logger

logger = init_module_logger(__name__)

CONFIG_SECTION = "loss"


def create_lossfunc(cfgs: dict):
    """
    Create loss function according configs.

    Available loss functions are:

    ```
    'L1Loss',
    'NLLLoss',
    'NLLLoss2d',
    'PoissonNLLLoss',
    'KLDivLoss',
    'MSELoss',
    'BCELoss',
    'BCEWithLogitsLoss',
    'HingeEmbeddingLoss',
    'MultiLabelMarginLoss',
    'SmoothL1Loss',
    'SoftMarginLoss',
    'CrossEntropyLoss',
    'MultiLabelSoftMarginLoss',
    'CosineEmbeddingLoss',
    'MarginRankingLoss',
    'MultiMarginLoss',
    'TripletMarginLoss',
    'CTCLoss'
    ```
    """
    valid_lossfuncs = [
        item for item in loss.__dict__ if not item.startswith("_") and (
            'Loss' in item) and callable(loss.__dict__[item])
    ]

    lossfunc_name = cfgs[CONFIG_SECTION].get("loss_function")

    try:
        lossfunc = loss.__dict__[lossfunc_name]()
    except KeyError as error:
        error_str = f"Loss function '{lossfunc_name}' in configuration is not supported."
        logger.error(error_str)
        raise KeyError(error_str)
    else:
        logger.info(f"Using loss function '{lossfunc_name}'.")
        return lossfunc