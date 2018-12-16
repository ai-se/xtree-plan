import os
import sys

# Update path
root = os.path.join(os.getcwd().split('src')[0], 'src')
if root not in sys.path:
    sys.path.append(root)

import pandas as pd
from pdb import set_trace

from tools.decision_tree import DecisionTree
from utils.misc_utils import flatten
from utils.experiment_utils import Changes
from utils.file_util import list2dataframe
from random import uniform, random as rand

class Plan:
    def __init__(self, trainDF, testDF, tree=None, config=False):
        self.trainDF = trainDF
        self.testDF = testDF
        self.tree = tree
        self.change = []

    def leaves(self,  node):
        """
        Returns all terminal nodes.
        """
        L = []
        if len(node.kids) > 1:
            for l in node.kids:
                L.extend(self.leaves(l))
            return L
        elif len(node.kids) == 1:
            return [node.kids]
        else:
            return [node]

    def find(self,  testInst, t):
        if len(t.kids) == 0:
            return t
        for kid in t.kids:
            try:
                if kid.val[0] <= testInst[kid.f].values[0] < kid.val[1]:
                    return self.find(testInst, kid)
                elif kid.val[1] == testInst[kid.f].values[0] == \
                        self.trainDF.describe()[kid.f]['max']:
                    return self.find(testInst, kid)
            except:
                return self.find(testInst, kid)
        return t

    @staticmethod
    def howfar(me, other):
        # set_trace()
        common = [a[0] for a in me.branch if a[0] in [o[0] for o in other.branch]]
        return len(me.branch) - len(common)

    def patchIt(self,  testInst, config=False):
        C = Changes()  # Record changes
        testInst = pd.DataFrame(testInst).transpose()
        current = self.find(testInst, self.tree)
        node = current
        while node.lvl > -1:
            node = node.up  # Move to tree root

        leaves = flatten([self.leaves(_k) for _k in node.kids])
        try:
            if self.config:
                best = sorted([l for l in leaves if l.score <= 0.9 * current.score],
                              key=lambda F: self.howfar(current, F))[0]
            else:
                best = \
                    sorted(
                        [l for l in leaves if l.score == 0 ],
                        key=lambda F: self.howfar(current, F))[0]
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
                now = ii[1] if self.config else new(testInst[ii[0]].values[0],
                                                 ii[1])
                # print(now)
                testInst[ii[0]] = str(now)
                # C.save(name=ii[0], old=then, new=now)

        testInst[testInst.columns[-1]] = 1
        # self.change.append(C.log)
        return testInst.values.tolist()[0]

    def transfrom(self):
        newRows = []
        for n in range(self.testDF.shape[0]):
            if self.testDF.iloc[n][-1] > 0 or self.testDF.iloc[n][-1] == True:
                newRows.append(self.patchIt(self.testDF.iloc[n]))
            else:
                newRows.append(self.testDF.iloc[n].values.tolist())
        return pd.DataFrame(newRows, columns=self.testDF.columns)
