"""
X_testREE
"""
import os
import sys
# Update path
root = os.path.join(os.getcwd().split('src')[0], 'src')
if root not in sys.path:
    sys.path.append(root)

import numpy as np
import pandas as pd
from pdb import set_trace
from collections import Counter

from tools.containers import Thing
from sklearn.base import BaseEstimator
from tools.Discretize import discretize, fWeight
from frequent_items.item_sets import ItemSetLearner

class XTREE(BaseEstimator):
    def __init__(self, opt=None):
        self.min=1
        self.klass=-1
        self.prune=False
        self.debug=True
        self.verbose=True
        self.max_levels=10
        self.infoPrune=1
        self.alpha = 0.66

    @staticmethod
    def _entropy(x):
            counts = Counter(x)
            N = len(x)
            return sum([-counts[n] / N * np.log(counts[n] / N) for n in counts.keys()])

    @staticmethod
    def pairs(lst):
        while len(lst) > 1:
            yield (lst.pop(0), lst[0])

    @staticmethod
    def best_plans(better_nodes, item_sets):
        max_intersection = float("-inf")

        def jaccard_similarity_score(set1, set2):
            intersect_length = len(set1.intersection(set2))
            set1_length = len(set1)
            set2_length = len(set2)
            return intersect_length / (set1_length + set2_length - intersect_length) 

        for node in better_nodes:
            change_set = set([bb[0] for bb in node.branch])
            for item_set in item_sets:
                jaccard_index = jaccard_similarity_score(item_set, change_set)
                if 0 < jaccard_index >= max_intersection:
                    best_path = (node, jaccard_index)
                    max_intersection = jaccard_index
        return best_path

    def pretty_print(self, tree=None, lvl=-1):        
        if tree is None:
            tree = self.tree
        if tree.f:
            print(('|...' * lvl) + str(tree.f) + "=" + "(%0.2f, %0.2f)" % tree.val + "\t:" + "%0.2f" % (tree.score), end="")
        if tree.kids:
            print('')
            for k in tree.kids:
                self.pretty_print(k, lvl + 1)
        else:
            print("")

    def _nodes(self, tree, lvl=0):
        if tree:
            yield tree, lvl
            for kid in tree.kids:
                lvl1 = lvl
                for sub, lvl1 in self._nodes(kid, lvl1 + 1):
                    yield sub, lvl1

    def _leaves(self, thresh=float("inf")):
        for node, _ in self._nodes(self.tree): 
            if not node.kids and node.score < thresh:
                yield node

    def _find(self,  test_instance, tree_node=None):
        if tree_node is None:
            tree_node = self.tree

        if len(tree_node.kids) == 0:
            return tree_node
    
        for kid in tree_node.kids:
            if kid.val[0] <= test_instance[kid.f] < kid.val[1]:
                return self._find(test_instance, kid)
            elif kid.val[1] == test_instance[kid.f] == self.tree.t.describe()[kid.f]['max']:
                return self._find(test_instance, kid)

    def _tree_builder(self, dframe, rows=None, lvl=-1, as_is=float("inf"), up=None, klass=-1, branch=[],
            f=None, val=None, opt=None):
        
        current = Thing(t=dframe, kids=[], f=f, val=val, up=up, lvl=lvl, rows=rows, modes={}, branch=branch)

        features = fWeight(dframe)

        if self.prune and lvl < 0:
            features = fWeight(dframe)[:int(len(features) * self.infoPrune)]

        name = features.pop(0)
        remaining = dframe[features + [dframe.columns[self.klass]]]
        feature = dframe[name].values
        klass = dframe[dframe.columns[self.klass]].values
        N = len(klass)
        current.score = np.mean(klass)
        splits = discretize(feature, klass)
        low = min(feature) 
        high = max(feature) 

        cutoffs = [t for t in self.pairs(sorted(list(set(splits + [low, high]))))]

        if lvl > (self.max_levels if self.prune else int(len(features) * self.infoPrune)):
            return current
        if as_is == 0:
            return current
        if len(features) < 1:
            return current

        def _rows():
            for span in cutoffs:
                new = []
                for f, row in zip(feature, remaining.values.tolist()):
                    if span[0] <= f < span[1]:
                        new.append(row)
                    elif f == span[1] == high:
                        new.append(row)
                yield pd.DataFrame(new, columns=remaining.columns), span

        for child, span in _rows():
            n = child.shape[0]
            to_be = self._entropy(child[child.columns[self.klass]])
            if self.min <= n < N:
                current.kids += [self._tree_builder(child, lvl=lvl + 1, as_is=to_be, up=current
                                    , branch=branch + [(name, span)]
                                    , f=name, val=span, opt=opt)]

        return current

    def fit(self, X_train):
        self.tree = self._tree_builder(X_train)
        return self

    def predict(self, X_test):
        new_df = pd.DataFrame(columns=X_test.columns)
        X = X_test[X_test.columns[:-1]]
        y = X_test[X_test.columns[-1]]
        
        "Itemset Learning"
        # Instantiate item set learning
        isl = ItemSetLearner()
        # Fit the data to itemset learner
        isl.fit(X, y)
        # Transform into itemsets
        item_sets = isl.transform()

        for row_num in range(len(X_test)):
            if X_test.iloc[row_num]["<bug"]: 
                old_row = X_test.iloc[row_num]
                "Find the location of the current test instance on the tree"
                pos = self._find(old_row)
                "Find all the leaf nodes on the tree that atleast alpha times smaller that current test instance"
                better_nodes = [leaf for leaf in self._leaves(thresh=self.alpha * pos.score)]
                best_path = self.best_plans(better_nodes, item_sets) 
                # ---- DEBUG -----
                set_trace()
        return self
