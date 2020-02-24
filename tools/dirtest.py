import os
# insert root dir path to sys.path to import easycls
import sys
sys.path.insert(0,
                os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import torch
import torchvision.models as models

import IPython

model = models.resnet18()

model_dp = torch.nn.DataParallel(model)

m_para = model.parameters()
m_dp_para = model_dp.parameters()

# IPython.embed()

total = 62  # len(list(m_para))
for i in range(total):
    print(
        f"{i}: {next(m_para) is next(m_dp_para)}"
    )  # all True. It means DataParallel model's parameters are just reference of original model's params.
