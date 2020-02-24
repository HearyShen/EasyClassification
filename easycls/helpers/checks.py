import torch.cuda as cuda

from .logs import init_module_logger

logger = init_module_logger(__name__)

def check_cuda():
    """
    Check CUDA devices, returns CUDA device count.

    including:

    ```
    cuda.current_device()
    cuda.device_count()
    cuda.get_device_capability()
    cuda.get_device_name()
    cuda.get_device_properties(d)
    cuda.is_available()
    cuda.is_initialized()
    ```

    Returns:
        device_count: int, CUDA device count
    """
    is_available = cuda.is_available()
    is_initialized = cuda.is_initialized()
    device_count = cuda.device_count()

    logger.info(f"[CUDA] is_available: {is_available}, is_initialized: {is_initialized}.")
    logger.info(f"Found {device_count} CUDA device(s).")
    for device_id in range(device_count):
        logger.info(f"[cuda:{device_id}] {cuda.get_device_properties(device_id)}")

    if is_available:
        logger.info(f"Current device: [cuda:{cuda.current_device()}] ({cuda.get_device_name()}, Compute Capability: {cuda.get_device_capability()})")

    return device_count
