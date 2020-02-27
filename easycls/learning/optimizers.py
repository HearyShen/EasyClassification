from configparser import ConfigParser
import torch.optim as optim
from ..helpers import init_module_logger

logger = init_module_logger(__name__)

CONFIG_SECTION = "optimizer"


def create_Adadelta(params, cfgs: ConfigParser):
    lr = cfgs.getfloat(CONFIG_SECTION, "lr", fallback=1.0)
    rho = cfgs.getfloat(CONFIG_SECTION, "rho", fallback=0.9)
    eps = cfgs.getfloat(CONFIG_SECTION, "eps", fallback=1e-6)
    weight_decay = cfgs.getfloat(CONFIG_SECTION, "weight_decay", fallback=0)

    return optim.Adadelta(params,
                          lr=lr,
                          rho=rho,
                          eps=eps,
                          weight_decay=weight_decay)


def create_Adagrad(params, cfgs: ConfigParser):
    lr = cfgs.getfloat(CONFIG_SECTION, "lr", fallback=0.01)
    lr_decay = cfgs.getfloat(CONFIG_SECTION, "lr_decay", fallback=0)
    weight_decay = cfgs.getfloat(CONFIG_SECTION, "weight_decay", fallback=0)
    initial_accumulator_value = cfgs.getfloat(CONFIG_SECTION,
                                              "initial_accumulator_value",
                                              fallback=0)
    eps = cfgs.getfloat(CONFIG_SECTION, "eps", fallback=1e-10)

    return optim.Adagrad(params,
                         lr=lr,
                         lr_decay=lr_decay,
                         weight_decay=weight_decay,
                         initial_accumulator_value=initial_accumulator_value,
                         eps=eps)


def create_Adam(params, cfgs: ConfigParser):
    lr = cfgs.getfloat(CONFIG_SECTION, "lr", fallback=1e-3)
    betas_list = [
        float(beta)
        for beta in cfgs.get(CONFIG_SECTION, "betas", fallback="(0.9,0.999)").
        strip().lstrip('(').rstrip(')').split(',')
    ]
    betas = (betas_list[0], betas_list[1])
    eps = cfgs.getfloat(CONFIG_SECTION, "eps", fallback=1e-8)
    weight_decay = cfgs.getfloat(CONFIG_SECTION, "weight_decay", fallback=0)
    amsgrad = cfgs.getboolean(CONFIG_SECTION, "amsgrad", fallback=False)

    return optim.Adam(params,
                      lr=lr,
                      betas=betas,
                      eps=eps,
                      weight_decay=weight_decay,
                      amsgrad=amsgrad)


def create_AdamW(params, cfgs: ConfigParser):
    lr = cfgs.getfloat(CONFIG_SECTION, "lr", fallback=1e-3)
    betas_list = [
        float(beta)
        for beta in cfgs.get(CONFIG_SECTION, "betas", fallback="(0.9,0.999)").
        strip().lstrip('(').rstrip(')').split(',')
    ]
    betas = (betas_list[0], betas_list[1])
    eps = cfgs.getfloat(CONFIG_SECTION, "eps", fallback=1e-8)
    weight_decay = cfgs.getfloat(CONFIG_SECTION, "weight_decay", fallback=1e-2)
    amsgrad = cfgs.getboolean(CONFIG_SECTION, "amsgrad", fallback=False)

    return optim.AdamW(params,
                       lr=lr,
                       betas=betas,
                       eps=eps,
                       weight_decay=weight_decay,
                       amsgrad=amsgrad)


def create_SparseAdam(params, cfgs: ConfigParser):
    lr = cfgs.getfloat(CONFIG_SECTION, "lr", fallback=1e-3)
    betas_list = [
        float(beta)
        for beta in cfgs.get(CONFIG_SECTION, "betas", fallback="(0.9,0.999)").
        strip().lstrip('(').rstrip(')').split(',')
    ]
    betas = (betas_list[0], betas_list[1])
    eps = cfgs.getfloat(CONFIG_SECTION, "eps", fallback=1e-8)

    return optim.SparseAdam(params, lr=lr, betas=betas, eps=eps)


def create_Adamax(params, cfgs: ConfigParser):
    lr = cfgs.getfloat(CONFIG_SECTION, "lr", fallback=2e-3)
    betas_list = [
        float(beta)
        for beta in cfgs.get(CONFIG_SECTION, "betas", fallback="(0.9,0.999)").
        strip().lstrip('(').rstrip(')').split(',')
    ]
    betas = (betas_list[0], betas_list[1])
    eps = cfgs.getfloat(CONFIG_SECTION, "eps", fallback=1e-8)
    weight_decay = cfgs.getfloat(CONFIG_SECTION, "weight_decay", fallback=0)

    return optim.Adamax(params,
                        lr=lr,
                        betas=betas,
                        eps=eps,
                        weight_decay=weight_decay)


def create_ASGD(params, cfgs: ConfigParser):
    lr = cfgs.getfloat(CONFIG_SECTION, "lr", fallback=1e-2)
    lambd = cfgs.getfloat(CONFIG_SECTION, "lambd", fallback=1e-4)
    alpha = cfgs.getfloat(CONFIG_SECTION, "alpha", fallback=0.75)
    t0 = cfgs.getfloat(CONFIG_SECTION, "t0", fallback=1e6)
    weight_decay = cfgs.getfloat(CONFIG_SECTION, "weight_decay", fallback=0)

    return optim.ASGD(params,
                      lr=lr,
                      lambd=lambd,
                      alpha=alpha,
                      t0=t0,
                      weight_decay=weight_decay)


def create_LBFGS(params, cfgs: ConfigParser):
    lr = cfgs.getfloat(CONFIG_SECTION, "lr", fallback=1)
    max_iter = cfgs.getint(CONFIG_SECTION, "max_iter", fallback=20)
    max_eval = cfgs.getint(CONFIG_SECTION, "max_eval", fallback=None)
    tolerance_grad = cfgs.getfloat(CONFIG_SECTION,
                                   "tolerance_grad",
                                   fallback=1e-7)
    tolerance_change = cfgs.getfloat(CONFIG_SECTION, "tolerance_change", 1e-9)
    history_size = cfgs.getint(CONFIG_SECTION, "history_size", fallback=100)
    line_search_fn = cfgs.get(CONFIG_SECTION, "line_search_fn", None)

    return optim.LBFGS(params,
                       lr=lr,
                       max_iter=max_iter,
                       max_eval=max_eval,
                       tolerance_grad=tolerance_grad,
                       tolerance_change=tolerance_change,
                       history_size=history_size,
                       line_search_fn=line_search_fn)


def create_RMSprop(params, cfgs: ConfigParser):
    lr = cfgs.getfloat(CONFIG_SECTION, "lr", fallback=1e-2)
    alpha = cfgs.getfloat(CONFIG_SECTION, "alpha", fallback=0.99)
    eps = cfgs.getfloat(CONFIG_SECTION, "eps", fallback=1e-8)
    weight_decay = cfgs.getfloat(CONFIG_SECTION, "weight_decay", fallback=0)
    momentum = cfgs.getfloat(CONFIG_SECTION, "momentum", fallback=0)
    centered = cfgs.getboolean(CONFIG_SECTION, "centered", fallback=False)

    return optim.RMSprop(params,
                         lr=lr,
                         alpha=alpha,
                         eps=eps,
                         weight_decay=weight_decay,
                         momentum=momentum,
                         centered=centered)


def create_Rprop(params, cfgs: ConfigParser):
    lr = cfgs.getfloat(CONFIG_SECTION, "lr", fallback=1e-2)
    etas_list = [
        float(eta)
        for eta in cfgs.get(CONFIG_SECTION, "etas", fallback="(0.5, 1.2)").
        strip().lstrip('(').rstrip(')').split(',')
    ]
    etas = (etas_list[0], etas_list[1])
    step_sizes_list = [
        float(step_size) for step_size in cfgs.get(
            CONFIG_SECTION, "step_sizes",
            fallback="(1e-6, 50)").strip().lstrip('(').rstrip(')').split(',')
    ]
    step_sizes = (step_sizes_list[0], step_sizes_list[1])

    return optim.Rprop(params, lr=lr, etas=etas, step_sizes=step_sizes)


def create_SGD(params, cfgs: ConfigParser):
    lr = cfgs.getfloat(CONFIG_SECTION, "lr")
    momentum = cfgs.getfloat(CONFIG_SECTION, "momentum", fallback=0)
    dampening = cfgs.getfloat(CONFIG_SECTION, "dampening", fallback=0)
    weight_decay = cfgs.getfloat(CONFIG_SECTION, "weight_decay", fallback=0)
    nesterov = cfgs.getboolean(CONFIG_SECTION, "nesterov", fallback=False)

    return optim.SGD(params,
                     lr=lr,
                     momentum=momentum,
                     dampening=dampening,
                     weight_decay=weight_decay,
                     nesterov=nesterov)


def create_optimizer(params, cfgs: ConfigParser):
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
        cfgs (ConfigParser): parsed configurations from configuration file.

    Returns:
        The configurated optimizer.

    Raises:
        KeyError, if optimizer algorithm is not supported.
    """
    algorithm = cfgs.get(CONFIG_SECTION, "algorithm")
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
