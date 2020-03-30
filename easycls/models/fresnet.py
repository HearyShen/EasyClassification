"""
This module provides auto-fit ResNet, fitting pretrained resnet model to target num of classes.
"""
import torch
import torchvision.models.resnet as resnet


def fresnet18(pretrained=False, num_classes=1000, **kwargs):
    r"""AutoFit ResNet-18 model.
    Fit the pretrained model for target num_classes by replacing the last fc layer.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        num_classes (int): If pretrained is ture and num_classes is not default 1000, 
                        model's last fc layer will be replaced after loading pretrained model.
    """
    if pretrained and num_classes != 1000:
        model = resnet.resnet18(pretrained, **kwargs)
        model.fc = torch.nn.Linear(512 * resnet.BasicBlock.expansion,
                                   num_classes)
    else:
        model = resnet.resnet18(pretrained, num_classes=num_classes, **kwargs)
    return model


def fresnet34(pretrained=False, num_classes=1000, **kwargs):
    r"""AutoFit ResNet-34 model.
    Fit the pretrained model for target num_classes by replacing the last fc layer.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        num_classes (int): If pretrained is ture and num_classes is not default 1000, 
                        model's last fc layer will be replaced after loading pretrained model.
    """
    if pretrained and num_classes != 1000:
        model = resnet.resnet34(pretrained, **kwargs)
        model.fc = torch.nn.Linear(512 * resnet.BasicBlock.expansion,
                                   num_classes)
    else:
        model = resnet.resnet34(pretrained, num_classes=num_classes, **kwargs)
    return model


def fresnet50(pretrained=False, num_classes=1000, **kwargs):
    r"""AutoFit ResNet-50 model.
    Fit the pretrained model for target num_classes by replacing the last fc layer.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        num_classes (int): If pretrained is ture and num_classes is not default 1000, 
                        model's last fc layer will be replaced after loading pretrained model.
    """
    if pretrained and num_classes != 1000:
        model = resnet.resnet50(pretrained, **kwargs)
        model.fc = torch.nn.Linear(512 * resnet.Bottleneck.expansion,
                                   num_classes)
    else:
        model = resnet.resnet50(pretrained, num_classes=num_classes, **kwargs)
    return model


def fresnet101(pretrained=False, num_classes=1000, **kwargs):
    r"""AutoFit ResNet-101 model.
    Fit the pretrained model for target num_classes by replacing the last fc layer.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        num_classes (int): If pretrained is ture and num_classes is not default 1000, 
                        model's last fc layer will be replaced after loading pretrained model.
    """
    if pretrained and num_classes != 1000:
        model = resnet.resnet101(pretrained, **kwargs)
        model.fc = torch.nn.Linear(512 * resnet.Bottleneck.expansion,
                                   num_classes)
    else:
        model = resnet.resnet101(pretrained, num_classes=num_classes, **kwargs)
    return model


def fresnet152(pretrained=False, num_classes=1000, **kwargs):
    r"""AutoFit ResNet-152 model.
    Fit the pretrained model for target num_classes by replacing the last fc layer.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        num_classes (int): If pretrained is ture and num_classes is not default 1000, 
                        model's last fc layer will be replaced after loading pretrained model.
    """
    if pretrained and num_classes != 1000:
        model = resnet.resnet152(pretrained, **kwargs)
        model.fc = torch.nn.Linear(512 * resnet.Bottleneck.expansion,
                                   num_classes)
    else:
        model = resnet.resnet152(pretrained, num_classes=num_classes, **kwargs)
    return model
