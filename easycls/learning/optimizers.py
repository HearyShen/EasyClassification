import torch.optim as optim
from ..helpers import init_module_logger

logger = init_module_logger(__name__)

CONFIG_SECTION = "optimizer"


def create_Adadelta(params, cfgs: dict):
    lr = cfgs[CONFIG_SECTION].get("lr", 1.0)
    rho = cfgs[CONFIG_SECTION].get("rho", 0.9)
    eps = cfgs[CONFIG_SECTION].get("eps", 1e-6)
    weight_decay = cfgs[CONFIG_SECTION].get("weight_decay", 0)

    return optim.Adadelta(params,
                          lr=lr,
                          rho=rho,
                          eps=eps,
                          weight_decay=weight_decay)


def create_Adagrad(params, cfgs: dict):
    lr = cfgs[CONFIG_SECTION].get("lr", 0.01)
    lr_decay = cfgs[CONFIG_SECTION].get("lr_decay", 0)
    weight_decay = cfgs[CONFIG_SECTION].get("weight_decay", 0)
    initial_accumulator_value = cfgs[CONFIG_SECTION].get(
        "initial_accumulator_value", 0)
    eps = cfgs[CONFIG_SECTION].get("eps", 1e-10)

    return optim.Adagrad(params,
                         lr=lr,
                         lr_decay=lr_decay,
                         weight_decay=weight_decay,
                         initial_accumulator_value=initial_accumulator_value,
                         eps=eps)


def create_Adam(params, cfgs: dict):
    lr = cfgs[CONFIG_SECTION].get("lr", 1e-3)
    betas = cfgs[CONFIG_SECTION].get("betas", (0.9,0.999))
    eps = cfgs[CONFIG_SECTION].get("eps", 1e-8)
    weight_decay = cfgs[CONFIG_SECTION].get("weight_decay", 0)
    amsgrad = cfgs[CONFIG_SECTION].get("amsgrad", False)

    return optim.Adam(params,
                      lr=lr,
                      betas=tuple(betas),
                      eps=eps,
                      weight_decay=weight_decay,
                      amsgrad=amsgrad)


def create_AdamW(params, cfgs: dict):
    lr = cfgs[CONFIG_SECTION].get("lr", 1e-3)
    betas = cfgs[CONFIG_SECTION].get("betas", (0.9,0.999))
    eps = cfgs[CONFIG_SECTION].get("eps", 1e-8)
    weight_decay = cfgs[CONFIG_SECTION].get("weight_decay", 1e-2)
    amsgrad = cfgs[CONFIG_SECTION].get("amsgrad", False)

    return optim.AdamW(params,
                       lr=lr,
                       betas=tuple(betas),
                       eps=eps,
                       weight_decay=weight_decay,
                       amsgrad=amsgrad)


def create_SparseAdam(params, cfgs: dict):
    lr = cfgs[CONFIG_SECTION].get("lr", 1e-3)
    betas = cfgs[CONFIG_SECTION].get("betas", (0.9,0.999))
    eps = cfgs[CONFIG_SECTION].get("eps", 1e-8)

    return optim.SparseAdam(params, lr=lr, betas=tuple(betas), eps=eps)


def create_Adamax(params, cfgs: dict):
    lr = cfgs[CONFIG_SECTION].get("lr", 2e-3)
    betas = cfgs[CONFIG_SECTION].get("betas", (0.9,0.999))
    eps = cfgs[CONFIG_SECTION].get("eps", 1e-8)
    weight_decay = cfgs[CONFIG_SECTION].get("weight_decay", 0)

    return optim.Adamax(params,
                        lr=lr,
                        betas=tuple(betas),
                        eps=eps,
                        weight_decay=weight_decay)


def create_ASGD(params, cfgs: dict):
    lr = cfgs[CONFIG_SECTION].get("lr", 1e-2)
    lambd = cfgs[CONFIG_SECTION].get("lambd", 1e-4)
    alpha = cfgs[CONFIG_SECTION].get("alpha", 0.75)
    t0 = cfgs[CONFIG_SECTION].get("t0", 1e6)
    weight_decay = cfgs[CONFIG_SECTION].get("weight_decay", 0)

    return optim.ASGD(params,
                      lr=lr,
                      lambd=lambd,
                      alpha=alpha,
                      t0=t0,
                      weight_decay=weight_decay)


def create_LBFGS(params, cfgs: dict):
    lr = cfgs[CONFIG_SECTION].get("lr", 1)
    max_iter = cfgs[CONFIG_SECTION].get("max_iter", 20)
    max_eval = cfgs[CONFIG_SECTION].get("max_eval", None)
    tolerance_grad = cfgs[CONFIG_SECTION].get("tolerance_grad", 1e-7)
    tolerance_change = cfgs[CONFIG_SECTION].get("tolerance_change", 1e-9)
    history_size = cfgs[CONFIG_SECTION].get("history_size", 100)
    line_search_fn = cfgs[CONFIG_SECTION].get("line_search_fn", None)

    return optim.LBFGS(params,
                       lr=lr,
                       max_iter=max_iter,
                       max_eval=max_eval,
                       tolerance_grad=tolerance_grad,
                       tolerance_change=tolerance_change,
                       history_size=history_size,
                       line_search_fn=line_search_fn)


def create_RMSprop(params, cfgs: dict):
    lr = cfgs[CONFIG_SECTION].get("lr", 1e-2)
    alpha = cfgs[CONFIG_SECTION].get("alpha", 0.99)
    eps = cfgs[CONFIG_SECTION].get("eps", 1e-8)
    weight_decay = cfgs[CONFIG_SECTION].get("weight_decay", 0)
    momentum = cfgs[CONFIG_SECTION].get("momentum", 0)
    centered = cfgs[CONFIG_SECTION].get("centered", False)

    return optim.RMSprop(params,
                         lr=lr,
                         alpha=alpha,
                         eps=eps,
                         weight_decay=weight_decay,
                         momentum=momentum,
                         centered=centered)


def create_Rprop(params, cfgs: dict):
    lr = cfgs[CONFIG_SECTION].get("lr", 1e-2)
    etas = cfgs[CONFIG_SECTION].get("etas", (0.5, 1.2))
    step_sizes = cfgs[CONFIG_SECTION].get("step_sizes", (1e-6, 50))

    return optim.Rprop(params, lr=lr, etas=tuple(etas), step_sizes=tuple(step_sizes))


def create_SGD(params, cfgs: dict):
    lr = cfgs[CONFIG_SECTION].get("lr")
    momentum = cfgs[CONFIG_SECTION].get("momentum", 0)
    dampening = cfgs[CONFIG_SECTION].get("dampening", 0)
    weight_decay = cfgs[CONFIG_SECTION].get("weight_decay", 0)
    nesterov = cfgs[CONFIG_SECTION].get("nesterov", False)

    return optim.SGD(params,
                     lr=lr,
                     momentum=momentum,
                     dampening=dampening,
                     weight_decay=weight_decay,
                     nesterov=nesterov)


def create_optimizer(params, cfgs: dict):
    """
    Create optimizer according to the config.

    Available optimizers are:

    ```
    'Adadelta',
    'Adagrad',
    'Adam',
    'AdamW',
    'SparseAdam',
    'Adamax',
    'ASGD',
    'SGD',
    'Rprop',
    'RMSprop',
    'LBFGS'
    ```

    Args:
        params (iterable): an iterable object contains parameters to be optimized.
        cfgs (dict): parsed configurations from configuration file.

    Returns:
        The configurated optimizer.

    Raises:
        KeyError, if optimizer algorithm is not supported.
    """
    algorithm = cfgs[CONFIG_SECTION].get("algorithm")
    logger.info(f"Using optimizer algorithm '{algorithm}'.")

    if algorithm == "Adadelta":
        return create_Adadelta(params, cfgs)
    elif algorithm == "Adagrad":
        return create_Adagrad(params, cfgs)
    elif algorithm == "Adam":
        return create_Adam(params, cfgs)
    elif algorithm == "AdamW":
        return create_AdamW(params, cfgs)
    elif algorithm == "SparseAdam":
        return create_SparseAdam(params, cfgs)
    elif algorithm == "Adamax":
        return create_Adamax(params, cfgs)
    elif algorithm == "ASGD":
        return create_ASGD(params, cfgs)
    elif algorithm == "LBFGS":
        return create_LBFGS(params, cfgs)
    elif algorithm == "RMSprop":
        return create_RMSprop(params, cfgs)
    elif algorithm == "Rprop":
        return create_Rprop(params, cfgs)
    elif algorithm == "SGD":
        return create_SGD(params, cfgs)
    else:
        error_str = f"Optimizer algorithm '{algorithm}' in configuration is not supported."
        logger.error(error_str)
        raise KeyError(error_str)
