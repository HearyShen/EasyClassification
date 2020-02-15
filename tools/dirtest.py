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
# import easycls.apis.infer as infer
from easycls.apis.infer import func

logging.basicConfig()
func()
