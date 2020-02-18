from configparser import ConfigParser


def parse_cfgs(path):
    """
    Parse a configuration file

    Usage:

    ```python
    cfgs = parse_cfgs('config.ini')

    task = cfgs.get("data", "task")
    pretrain = cfgs.getboolean("model", "pretrained")
    num_classes = cfgs.getint("data", "num-classes")
    lr = cfgs.getfloat("learning", "learning-rate")
    ```

    Args:
        path: configuration file path
    
    Return:
        an instance of ConfigParser
    """
    cfgparser = ConfigParser()
    cfgparser.read(path)

    return cfgparser


# Unit Test
if __name__ == "__main__":
    cfgs = parse_cfgs('config.ini')

    task = cfgs.get("data", "task")
    pretrain = cfgs.getboolean("model", "pretrained")
    num_classes = cfgs.getint("data", "num-classes")
    lr = cfgs.getfloat("learning", "learning-rate")

    print(
        f'task={task}\tpretrain={pretrain}\tnum-classes-{num_classes}\tlr={lr}'
    )
