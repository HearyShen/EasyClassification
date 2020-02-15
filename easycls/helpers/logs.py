import logging


def init_module_logger():
    """
    Init a logger for the a module.

    Usage:

    In module (demo.py):

    ```python
    from ..helpers import logs
    logger = logs.init_module_logger()

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
    logger = logging.getLogger(__name__)
    logger.addHandler(logging.NullHandler())
    return logger
