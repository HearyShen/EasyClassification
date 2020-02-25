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

print_callable(optim.lr_scheduler)

print_callable(loss)

print_callable(models)

print_callable(datasets)



# import torch
# import torch.optim as optim
# import torchvision.models as models
# import IPython


# def get_lr(optimizer):
#     return [param_group['lr'] for param_group in optimizer.param_groups]

# model = models.resnet18().cuda()
# optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

# lambda_a = lambda epoch: 0.95 ** (epoch-1)      # f(x) = 0.95^x
# # lambda_b = lambda epoch: 0.1 ** (epoch // 30)
# # lambda_c = lambda epoch: 0.9
# scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda_a)

# # IPython.embed()

# for epoch in range(30):
#     # print(get_lr(optimizer))
#     print(scheduler.get_last_lr())
#     scheduler.step()