import sys
from collections import Counter
from pdb import set_trace

import numpy as np
import pandas as pd

from containers import Thing
from misc import explore, csv2DF
from tools.Discretize import discretize, fWeight
from sklearn.base import BaseEstimator

class DecisionTree(BaseEstimator):
    def __init__(self, opt=None):
        self.min=1,
        self.max_level=10,
        self.infoPrune=0.5,
        self.klass=-1,
        self.prune=True,
        self.debug=True,
        self.verbose=True

    def show(self, tree=None, lvl=-1):        
        if tree is None:
            tree = self.tree
        if tree.f:
            print(('|..' * lvl) + str(tree.f) + "=" + "(%0.2f, %0.2f)" % tree.val + "\t:" + "%0.2f" % (tree.score), end="")
        if tree.kids:
            print('')
            for k in tree.kids:
                self.show(k, lvl + 1)
        else:
            print("")

    def nodes(self, tree, lvl=0):
        if tree:
            yield tree, lvl
            for kid in tree.kids:
                lvl1 = lvl
                for sub, lvl1 in self.nodes(kid, lvl1 + 1):
                    yield sub, lvl1

    def leaves(self, tree):
        for node, _ in self.nodes(tree):
            # print "K>", tree.kids[0].__dict__.keys()
            if not node.kids:
                yield node

    def _tree_builder(self, tbl, rows=None, lvl=-1, asIs=10 ** 32, up=None, klass=-1, branch=[],
            f=None, val=None, opt=None):
        
        here = Thing(t=tbl, kids=[], f=f, val=val, up=up, lvl=lvl
                    , rows=rows, modes={}, branch=branch)

        features = fWeight(tbl)

        if self.prune and lvl < 0:
            features = fWeight(tbl)[:int(len(features) * self.infoPrune)]

        name = features.pop(0)
        remaining = tbl[features + [tbl.columns[self.klass]]]
        feature = tbl[name].values
        klass = tbl[tbl.columns[self.klass]].values
        N = len(klass)
        here.score = np.mean(klass)
        splits = discretize(feature, klass)
        lo, hi = min(feature), max(feature)

        def _pairs(lst):
            while len(lst) > 1:
                yield (lst.pop(0), lst[0])

        cutoffs = [t for t in _pairs(sorted(list(set(splits + [lo, hi]))))]

        if lvl > (self.max_level if self.prune else int(len(features) * self.infoPrune)):
            return here
        if asIs == 0:
            return here
        if len(features) < 1:
            return here

        def _rows():
            for span in cutoffs:
                new = []
                for f, row in zip(feature, remaining.values.tolist()):
                    if span[0] <= f < span[1]:
                        new.append(row)
                    elif f == span[1] == hi:
                        new.append(row)
                yield pd.DataFrame(new, columns=remaining.columns), span

        def _entropy(x):
            C = Counter(x)
            N = len(x)
            return sum([-C[n] / N * np.log(C[n] / N) for n in C.keys()])

        for child, span in _rows():
            n = child.shape[0]
            toBe = _entropy(child[child.columns[self.klass]])
            if self.min <= n < N:
                here.kids += [self._tree_builder(child, lvl=lvl + 1, asIs=toBe, up=here
                                    , branch=branch + [(name, span)]
                                    , f=name, val=span, opt=opt)]

        return here

    def fit(self, X, y):
        raw_data = pd.concat([X,y], axis=1)
        self.tree = self._tree_builder(raw_data)
        return self

    def predict(self, Xt):
        return self
