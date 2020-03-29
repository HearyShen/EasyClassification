"""
This module provides optimizers, including torch.optim and self-implemented optimizers.
"""
from torch.optim import *
del lr_scheduler

from .optimizers import *
from . import lr_schedulers