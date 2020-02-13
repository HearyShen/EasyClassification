from os.path import dirname, abspath, join

filename = abspath(__file__)
print(f'current filepath: {filename}')

dirpath = abspath(dirname(filename))
print(f'current dirpath: {dirpath}')

parent_dirpath = abspath(join(dirpath, '..'))
print(f'parent dirpath: {parent_dirpath}')