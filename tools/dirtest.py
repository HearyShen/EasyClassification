import os
# insert root dir path to sys.path to import easycls
import sys
sys.path.insert(0,
                os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

# from os.path import dirname, abspath, join

# filename = abspath(__file__)
# print(f'current filepath: {filename}')

# dirpath = abspath(dirname(filename))
# print(f'current dirpath: {dirpath}')

# parent_dirpath = abspath(join(dirpath, '..'))
# print(f'parent dirpath: {parent_dirpath}')

import logging
import easycls.helpers as helpers
import easycls.apis.infer as infer

logger = logging.getLogger()  # 不加名称设置root logger
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    r'%(asctime)s - %(name)s - %(levelname)s: - %(message)s',
    datefmt=r'%Y-%m-%d %H:%M:%S')

# 使用FileHandler输出到文件
fh = logging.FileHandler(f'log_{helpers.format_time()}.log')
fh.setLevel(logging.INFO)
fh.setFormatter(formatter)

# 使用StreamHandler输出到屏幕
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(formatter)

# 添加两个Handler
logger.addHandler(ch)
logger.addHandler(fh)

# logging.basicConfig()
infer.func()
