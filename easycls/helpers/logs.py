import logging


def init_module_logger():
    log = logging.getLogger(__name__)
    log.addHandler(logging.NullHandler())
    return log


# Example function (for testing)
def func():
    log.critical('A Critical Error')


log = init_module_logger()

if __name__ == "__main__":
    print(__name__)
    print(__package__)
    print(__doc__)