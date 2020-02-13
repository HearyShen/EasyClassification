import os
from configparser import ConfigParser
from torchvision import transforms, datasets

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

def get_train_dataset(cfgs:ConfigParser):
    traindir = os.path.join(cfgs.get('data', 'path'), 'train')
    train_dataset = datasets.ImageFolder(traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    return train_dataset

def get_val_dataset(cfgs:ConfigParser):
    valdir = os.path.join(cfgs.get('data', 'path'), 'val')
    val_dataset = datasets.ImageFolder(
            valdir, 
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))
    return val_dataset
    
def get_test_dataset(cfgs:ConfigParser):
    testdir = os.path.join(cfgs.get('data', 'path'), 'test')
    test_dataset = datasets.ImageFolder(
        testdir, 
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))
    return test_dataset