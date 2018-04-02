"""
PC45 - Python C4.5 Framework
````````````````````````````
A Method for generating a pruned C4.5 decision tree. Based on,
[1] Quinlan, J. Ross. "Improved use of continuous attributes in C4. 5." Journal of artificial intelligence research (1996): 77-90.
[2] Quinlan, J. Ross. C4. 5: programs for machine learning. Elsevier, 2014.

Also uses,
[3] Usama M. Fayyad, Keki B. Irani. "Multi-interval discretization of continuous valued attributes for classification learning." Thirteenth International Joint Conference on Artificial Intelligence, 1022-1027, 1993.

"""
from __future__ import division, print_function

from collections import Counter
from pdb import set_trace

import numpy as np
import pandas as pd

from containers import Thing
from misc import explore, csv2DF
from tools.Discretize import discretize, fWeight


def dtree(tbl, rows=None, lvl=-1, asIs=10 ** 32, up=None, klass=-1, branch=[],
          f=None, val=None, opt=None):
    if not opt:
        opt = Thing(
            min=1,
            maxLvL=10,
            infoPrune=0.5,
            klass=-1,
            prune=True,
            debug=True,
            verbose=True)

    here = Thing(t=tbl, kids=[], f=f, val=val, up=up, lvl=lvl
                 , rows=rows, modes={}, branch=branch)

    features = fWeight(tbl)

    if opt.prune and lvl < 0:
        features = fWeight(tbl)[:int(len(features) * opt.infoPrune)]

    name = features.pop(0)
    remaining = tbl[features + [tbl.columns[opt.klass]]]
    feature = tbl[name].values
    klass = tbl[tbl.columns[opt.klass]].values
    N = len(klass)
    here.score = np.mean(klass)
    splits = discretize(feature, klass)
    LO, HI = min(feature), max(feature)

    def pairs(lst):
        while len(lst) > 1:
            yield (lst.pop(0), lst[0])

    cutoffs = [t for t in pairs(sorted(list(set(splits + [LO, HI]))))]

    if lvl > (opt.maxLvL if opt.prune else int(len(features) * opt.infoPrune)):
        return here
    if asIs == 0:
        return here
    if len(features) < 1:
        return here

    def rows():
        for span in cutoffs:
            new = []
            for f, row in zip(feature, remaining.values.tolist()):
                if span[0] <= f < span[1]:
                    new.append(row)
                elif f == span[1] == HI:
                    new.append(row)
            yield pd.DataFrame(new, columns=remaining.columns), span

    def ent(x):
        C = Counter(x)
        N = len(x)
        return sum([-C[n] / N * np.log(C[n] / N) for n in C.keys()])

    for child, span in rows():
        n = child.shape[0]
        toBe = ent(child[child.columns[opt.klass]])
        if opt.min <= n < N:
            here.kids += [dtree(child, lvl=lvl + 1, asIs=toBe, up=here
                                , branch=branch + [(name, span)]
                                , f=name, val=span, opt=opt)]

    return here


def dtree2(tbl, rows=None, lvl=-1, asIs=10 ** 32, up=None, klass=-1, branch=[],
           f=None, val=None, opt=None):
    """
    Discrete independent variables
    """
    if not opt:
        opt = Thing(
            min=1,
            maxLvL=10,
            infoPrune=1,
            klass=-1,
            prune=True,
            debug=True,
            verbose=True)

    here = Thing(t=tbl, kids=[], f=f, val=val, up=up, lvl=lvl
                 , rows=rows, modes={}, branch=branch)

    features = fWeight(tbl)

    if opt.prune and lvl < 0:
        features = fWeight(tbl)[:int(len(features) * opt.infoPrune)]

    name = features.pop(0)
    remaining = tbl[features + [tbl.columns[opt.klass]]]
    feature = tbl[name].values
    klass = tbl[tbl.columns[opt.klass]].values
    N = len(klass)
    here.score = np.mean(klass)
    splits = discretize(feature, klass, discrete=True)
    LO, HI = min(feature), max(feature)

    def pairs(lst):
        while len(lst) > 1:
            yield (lst.pop(0), lst[0])

    cutoffs = [LO, HI]

    if lvl > (opt.maxLvL if opt.prune else int(len(features) * opt.infoPrune)):
        return here
    if asIs == 0:
        return here
    if len(features) < 1:
        return here

    def rows():
        for span in cutoffs:
            new = []
            for f, row in zip(feature, remaining.values.tolist()):
                if f == span:
                    new.append(row)
            yield pd.DataFrame(new, columns=remaining.columns), span

    def ent(x):
        C = Counter(x)
        N = len(x)
        return sum([-C[n] / N * np.log(C[n] / N) for n in C.keys()])

    for child, span in rows():
        # set_trace()
        n = child.shape[0]
        toBe = ent(child[child.columns[opt.klass]])
        if opt.min <= n < N:
            here.kids += [dtree2(child, lvl=lvl + 1, asIs=toBe, up=here
                                 , branch=branch + [(name, span)]
                                 , f=name, val=(span, span), opt=opt)]

    return here
    # # ------ Debug ------
    # set_trace()


def show(n, lvl=-1):
    import sys
    def say(x):
        sys.stdout.write(x)

    if n.f:
        say(('|..' * lvl) + str(n.f) + "=" + "(%0.2f, %0.2f)" % n.val +
            "\t:" + "%0.2f" % (n.score))
    if n.kids:
        print('')
        for k in n.kids:
            show(k, lvl + 1)
    else:
        print("")


def nodes(tree, lvl=0):
    if tree:
        yield tree, lvl
        for kid in tree.kids:
            lvl1 = lvl
            for sub, lvl1 in nodes(kid, lvl1 + 1):
                yield sub, lvl1


def leaves(tree):
    for node, _ in nodes(tree):
        # print "K>", tree.kids[0].__dict__.keys()
        if not node.kids:
            yield node


def _test():
    tbl_loc = explore(dir='../data.dat/Seigmund/', name='Apache')
    tbl = csv2DF(tbl_loc)

    # Define Tree settings
    opt = Thing(
        min=1,
        maxLvL=10,
        infoPrune=0.5,
        klass=-1,
        prune=False,
        debug=True,
        verbose=True)

    # Build a tree
    tree = dtree(tbl, opt=opt)

    # Show the tree
    if opt.verbose: show(tree)

    # ----- Debug? -----
    if opt.debug: set_trace()


if __name__ == '__main__':
    _test()
