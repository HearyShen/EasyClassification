import yaml

def parse_cfgs(path):
    """
    Parse a configuration file

    Usage:

    ```python
    cfgs = parse_cfgs('config.yml')

    task = cfgs["data"].get("task")
    pretrain = cfgs["model"].get("pretrained", False)
    num_classes = cfgs["model"].get("num-classes")
    lr = cfgs["learning"].get("learning-rate", 0.01)
    ```

    Args:
        path: configuration file path
    
    Return:
        configurations (dict)
    """
    with open(path, 'r') as f:
        cfgs = yaml.safe_load(f)

    return cfgs


# Unit Test
if __name__ == "__main__":
    cfgs = parse_cfgs('config.yml')

    task = cfgs["data"].get("task")
    pretrained = cfgs["model"].get("pretrained", False)
    num_classes = cfgs["model"].get("num_classes")
    lr = cfgs["learning"].get("learning-rate", 0.01)

    print(
        f'task={task}\tpretrained={pretrained}\tnum_classes={num_classes}\tlr={lr}'
    )
