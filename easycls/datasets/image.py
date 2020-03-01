import os
from configparser import ConfigParser
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from ..helpers import init_module_logger

logger = init_module_logger(__name__)

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])


def get_train_dataset(cfgs: ConfigParser):
    """
    Returns the train dataset.
    """
    traindir = os.path.join(cfgs.get('data', 'path'), 'train')
    logger.info(f"Creating train set from '{traindir}'.")
    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    return train_dataset


def get_val_dataset(cfgs: ConfigParser):
    """
    Returns the validation dataset.
    """
    valdir = os.path.join(cfgs.get('data', 'path'), 'val')
    logger.info(f"Creating validation set from '{valdir}'.")
    val_dataset = datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))
    return val_dataset


def get_test_dataset(cfgs: ConfigParser):
    """
    Returns the test dataset.
    """
    testdir = os.path.join(cfgs.get('data', 'path'), 'test')
    logger.info(f"Creating test set from '{testdir}'.")
    test_dataset = datasets.ImageFolder(
        testdir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))
    return test_dataset


def load_trainset(cfgs: ConfigParser):
    """
    Returns a DataLoader of train dataset.
    """
    dataset = get_train_dataset(cfgs)

    batch_size = cfgs.getint('learning', 'batch_size')
    dataload_workers = cfgs.getint('speed', 'dataload_workers')
    logger.info(
        f"Train set is being loaded by {dataload_workers} dataloader workers. (batch_size={batch_size})"
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,  # Shuffle samples for training for better estimating loss
        num_workers=dataload_workers,
        pin_memory=True)
    return dataloader


def load_valset(cfgs: ConfigParser):
    """
    Returns a DataLoader of validation dataset.
    """
    dataset = get_val_dataset(cfgs)

    batch_size = cfgs.getint('learning', 'batch_size')
    dataload_workers = cfgs.getint('speed', 'dataload_workers')
    logger.info(
        f"Validation set is being loaded by {dataload_workers} dataloader workers. (batch_size={batch_size})"
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # No shuffle for val set
        num_workers=dataload_workers,
        pin_memory=True)
    return dataloader


def load_testset(cfgs: ConfigParser):
    """
    Returns a DataLoader of test dataset.
    """
    dataset = get_test_dataset(cfgs)

    batch_size = cfgs.getint('learning', 'batch_size')
    dataload_workers = cfgs.getint('speed', 'dataload_workers')
    logger.info(
        f"Test set is being loaded by {dataload_workers} dataloader workers. (batch_size={batch_size})"
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # No shuffle for test set
        num_workers=dataload_workers,
        pin_memory=True)
    return dataloader
