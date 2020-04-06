"""
This module provides self-implemented optimizers.
"""

def get_lrs(optimizer):
    """Return the learning-rates in optimizer's parameter groups."""
    return [pg['lr'] for pg in optimizer.param_groups]
