"""
This module provides auto-fit ResNet, fitting pretrained resnet model to target num of classes.
"""
import torch
import torchvision.models.resnet as resnet


def fresnet18(pretrained=False, progress=True, num_classes=1000, **kwargs):
    r"""AutoFit ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        num_classes (int): If pretrained is ture and num_classes is not default 1000, model's tail will be replaced after loading pretrained model.
    """
    model = resnet.resnet18(pretrained, progress)
    if pretrained and num_classes != 1000:
        model.fc = torch.nn.Linear(512 * resnet.BasicBlock.expansion,
                                   num_classes)
    return model


def fresnet34(pretrained=False, progress=True, num_classes=1000, **kwargs):
    r"""AutoFit ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        num_classes (int): If pretrained is ture and num_classes is not default 1000, model's tail will be replaced after loading pretrained model.
    """
    model = resnet.resnet34(pretrained, progress)
    if pretrained and num_classes != 1000:
        model.fc = torch.nn.Linear(512 * resnet.BasicBlock.expansion,
                                   num_classes)
    return model


def fresnet50(pretrained=False, progress=True, num_classes=1000, **kwargs):
    r"""AutoFit ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        num_classes (int): If pretrained is ture and num_classes is not default 1000, model's tail will be replaced after loading pretrained model.
    """
    model = resnet.resnet50(pretrained, progress)
    if pretrained and num_classes != 1000:
        model.fc = torch.nn.Linear(512 * resnet.Bottleneck.expansion,
                                   num_classes)
    return model


def fresnet101(pretrained=False, progress=True, num_classes=1000, **kwargs):
    r"""AutoFit ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        num_classes (int): If pretrained is ture and num_classes is not default 1000, model's tail will be replaced after loading pretrained model.
    """
    model = resnet.resnet101(pretrained, progress)
    if pretrained and num_classes != 1000:
        model.fc = torch.nn.Linear(512 * resnet.Bottleneck.expansion,
                                   num_classes)
    return model


def fresnet152(pretrained=False, progress=True, num_classes=1000, **kwargs):
    r"""AutoFit ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        num_classes (int): If pretrained is ture and num_classes is not default 1000, model's tail will be replaced after loading pretrained model.
    """
    model = resnet.resnet152(pretrained, progress)
    if pretrained and num_classes != 1000:
        model.fc = torch.nn.Linear(512 * resnet.Bottleneck.expansion,
                                   num_classes)
    return model
