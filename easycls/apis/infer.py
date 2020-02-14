import logging

from ..helpers import logs

log = logs.init_module_logger()

def func():
    log.debug('debug...')
    log.info('info...')
    log.warning('warning...')
    log.error('error...')
    log.critical('critical...')