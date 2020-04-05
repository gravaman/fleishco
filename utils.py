from os import listdir
from os.path import join, isfile


def list_files(root_dir):
    paths = [join(root_dir, p) for p in listdir(root_dir)
             if isfile(join(root_dir, p))]
    return paths
