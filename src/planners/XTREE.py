"""
XTREE
"""

from __future__ import print_function, division

import os
import sys

# Update path
root = os.path.join(os.getcwd().split('src')[0], 'src')
if root not in sys.path:
    sys.path.append(root)

import pandas as pd
from tools import pyC45
from pdb import set_trace
from oracle.smote import SMOTE
from utils.misc_utils import flatten
from utils.experiment_utils import Changes
from utils.file_util import list2dataframe
from random import uniform, random as rand


class Patches:
    def __init__(i, train, test, trainDF, testDF, tree=None, config=False):
        i.train = train
        i.trainDF = trainDF
        i.test = test
        i.testDF = testDF
        i.config = config
        i.tree = tree
        i.change = []

    def leaves(i, node):
        """
        Returns all terminal nodes.
        """
        L = []
        if len(node.kids) > 1:
            for l in node.kids:
                L.extend(i.leaves(l))
            return L
        elif len(node.kids) == 1:
            return [node.kids]
        else:
            return [node]

    def find(i, testInst, t):
        if len(t.kids) == 0:
            return t
        for kid in t.kids:
            if i.config:
                if kid.val[0] == testInst[kid.f].values[0]:
                    return i.find(testInst, kid)
            else:
                try:
                    if kid.val[0] <= testInst[kid.f].values[0] < kid.val[1]:
                        return i.find(testInst, kid)
                    elif kid.val[1] == testInst[kid.f].values[0] == \
                            i.trainDF.describe()[kid.f]['max']:
                        return i.find(testInst, kid)
                except:
                    return i.find(testInst, kid)
        return t

    @staticmethod
    def howfar(me, other):
        # set_trace()
        common = [a[0] for a in me.branch if a[0] in [o[0] for o in other.branch]]
        return len(me.branch) - len(common)

    def patchIt(i, testInst, config=False):
        C = Changes()  # Record changes
        testInst = pd.DataFrame(testInst).transpose()
        current = i.find(testInst, i.tree)
        node = current
        while node.lvl > -1:
            node = node.up  # Move to tree root

        leaves = flatten([i.leaves(_k) for _k in node.kids])
        try:
            if i.config:
                best = sorted([l for l in leaves if l.score <= 0.9 * current.score],
                              key=lambda F: i.howfar(current, F))[0]
            else:
                best = \
                    sorted(
                        [l for l in leaves if l.score == 0 ],
                        key=lambda F: i.howfar(current, F))[0]
                # set_trace()
        except:
            return testInst.values.tolist()[0]

        def new(old, range):
            rad = abs(min(range[1] - old, old - range[1]))
            return abs(range[0]), abs(range[1])
            # return uniform(range[0], range[1])

        for ii in best.branch:
            before = testInst[ii[0]]
            if not ii in current.branch:
                then = testInst[ii[0]].values[0]
                now = ii[1] if i.config else new(testInst[ii[0]].values[0],
                                                 ii[1])
                # print(now)
                testInst[ii[0]] = str(now)
                # C.save(name=ii[0], old=then, new=now)

        testInst[testInst.columns[-1]] = 1
        # i.change.append(C.log)
        return testInst.values.tolist()[0]

    def main(i):
        newRows = []
        for n in xrange(i.testDF.shape[0]):
            if i.testDF.iloc[n][-1] > 0 or i.testDF.iloc[n][-1] == True:
                newRows.append(i.patchIt(i.testDF.iloc[n]))
            else:
                newRows.append(i.testDF.iloc[n].values.tolist())
        return pd.DataFrame(newRows, columns=i.testDF.columns)


def xtree(train_df, test_df):
    """XTREE"""

    if isinstance(train_df, list):
        train_df = list2dataframe(train_df)  # create a pandas dataframe of training data.dat
    if isinstance(test_df, list):
        test_df = list2dataframe(test_df)  # create a pandas dataframe of testing data.dat
    if isinstance(test_df, basestring):
        test_df = list2dataframe([test_df])  # create a pandas dataframe of testing data.dat

    # train_df = SMOTE(train_df, atleast=1000, atmost=1001)

    tree = pyC45.dtree(train_df)  # Create a decision tree

    patch = Patches(train=None, test=None, trainDF=train_df, testDF=test_df,
                    tree=tree)

    modified = patch.main()

    return modified


if __name__ == '__main__':
    pass
