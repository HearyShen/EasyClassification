import os
import logging
from .times import format_time

DEFAULT_FORMAT = r'[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s'
DEFAULT_DATE_FORMAT = r'%Y-%m-%d %H:%M:%S'


def init_module_logger(name=None):
    """
    Initialize a logger for the a module.

    Usage:

    In module (demo.py):

    ```python
    from ..helpers import logs
    logger = logs.init_module_logger(__name__)

    def func():
        logger.debug('debug...')
        logger.info('info...')
        logger.warning('warning...')
        logger.error('error...')
        logger.critical('critical...')
    ```

    In caller (test.py):
    
    ```python
    import logging  # REMOVE this if you DON'T need logs
    import easycls.apis.demo as demo

    logging.basicConfig()   # REMOVE this if you DON'T need logs
    demo.func()
    ```

    After execution, you should see the following outputs in `stdout`:
    
    ```
    WARNING:easycls.helpers.logs:warning...
    ERROR:easycls.helpers.logs:error...
    CRITICAL:easycls.helpers.logs:critical...
    ```
    """
    logger = logging.getLogger(name)
    logger.addHandler(logging.NullHandler())
    return logger


mlogger = init_module_logger(__name__)


def init_basic_root_logger(level=logging.INFO):
    """
    Initialize the root logger with basic configuration.
    """
    logging.basicConfig()
    logger = logging.getLogger()  # get root logger as default
    logger.setLevel(level)

    return logger


def add_file_log(logger=logging.getLogger(),
                 filename=f'{format_time(format=r"%Y%m%d_%H%M%S")}.log',
                 level=logging.INFO,
                 format=DEFAULT_FORMAT,
                 dateformat=DEFAULT_DATE_FORMAT):
    """
    Add a logging.FileHandler instance to logger, output logs to file.
    """
    formatter = logging.Formatter(fmt=format, datefmt=dateformat)

    file_handler = logging.FileHandler(filename)
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    return logger


def add_stream_log(logger=logging.getLogger(),
                   level=logging.INFO,
                   format=DEFAULT_FORMAT,
                   dateformat=DEFAULT_DATE_FORMAT):
    """
    Add a logging.StreamHandler instance to logger, output logs to screen.
    """
    formatter = logging.Formatter(fmt=format, datefmt=dateformat)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)

    return logger


def init_root_logger(filename=f'{format_time(format=r"%Y%m%d_%H%M%S")}.log',
                     flog_level=logging.INFO,
                     slog_level=logging.INFO,
                     format=DEFAULT_FORMAT,
                     dateformat=DEFAULT_DATE_FORMAT):
    """
    Initialize the root logger with file-and-screen outputs configuration.

    Usage:

    ```
    import easycls.helpers as helpers

    logger = helpers.init_root_logger()
    ```
    """
    logger = logging.getLogger()  # get root logger as default
    logger.setLevel(logging.NOTSET)  # do not reject any handlers logs

    # create dirs if not existed
    dirname = os.path.dirname(filename)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    add_file_log(logger=logger,
                 filename=filename,
                 level=flog_level,
                 format=format,
                 dateformat=dateformat)
    add_stream_log(logger=logger,
                   level=slog_level,
                   format=format,
                   dateformat=dateformat)
    mlogger.info(
        f"Root logger initialized with file({filename}) and stream output")

    return logger