from ..helpers import logs

logger = logs.init_module_logger(__name__)

def func():
    logger.debug('debug...')
    logger.info('info...')
    logger.warning('warning...')
    logger.error('error...')
    logger.critical('critical...')