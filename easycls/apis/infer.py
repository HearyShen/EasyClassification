import torch

from ..helpers import init_module_logger

logger = init_module_logger(__name__)

def infer(model, inputs):
    """
    Using a model to infer outputs from inputs.

    Args:
        model (Model): a PyTorch model.
        inputs (Tensor): inputs to the model.

    Returns:
        outputs (Tensor): results inferred by the model from inputs.
    """
    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        # compute outputs
        outputs = model(inputs)

    return outputs
