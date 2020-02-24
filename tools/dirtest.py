import os
# insert root dir path to sys.path to import easycls
import sys
sys.path.insert(0,
                os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import torch

import torch.optim as optim
import torch.nn.modules.loss as loss

import torchvision.models as models
import torchvision.datasets as datasets

def print_callable(package):
    print([item for item in package.__dict__ if not item.startswith("_") and callable(package.__dict__[item])])


print_callable(optim)

print_callable(loss)

print_callable(models)

print_callable(datasets)