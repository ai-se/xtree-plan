import os
import sys
from pdb import set_trace
from glob import glob

root = os.path.join(os.getcwd().split('project')[0], 'project')
if root not in sys.path:
    sys.path.append(root)


class _Data:
    """Hold training and testing data.dat"""

    def __init__(self, dir):
        self.data = sorted(glob(os.path.abspath(os.path.join(
            dir, "*.csv"))), key=lambda x: x.split("/")[-1])


def get_all_projects():
    all = dict()
    dirs = glob(os.path.join(root, "data/*/"))
    for dir in dirs:
        all.update({dir.split('/')[-2]: _Data(dir)})
    return all


def _test():
    data = get_all_projects()


if __name__ == "__main__":
    _test()
