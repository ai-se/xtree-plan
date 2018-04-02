from collections import Counter
from pdb import set_trace
from random import choice, uniform as rand
from random import seed
import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean
from sklearn.neighbors import BallTree

import warnings
with warnings.catch_warnings():
    # Shut those god damn warnings up!
    warnings.filterwarnings("ignore")


def SMOTE(data=None, atleast=50, atmost=200, a=None, b=None, k=14):
    """
    Synthetic Minority Oversampling Technique
    """

    def knn(a, b):
        "k nearest neighbors"
        b = np.array([bb[:-1] for bb in b])
        tree = BallTree(b)
        __, indx = tree.query(a[:-1], k=6)

        return [b[i] for i in indx]
        # set_trace()
        # return sorted(b, key=lambda F: euclidean(a[:-1], F[:-1]))

    def kfn(me, my_lot, others):
        "k farthest neighbors"
        my_closest = None
        return sorted(b, key=lambda F: euclidean(a[:-1], F[:-1]))

    def extrapolate(one, two):
        new = len(one) * [None]
        new[:-1] = [a + rand(0, 1) * (b - a) for
                    a, b in zip(one[:-1], two[:-1])]
        try:
            new[-1] = int(one[-1])
        except ValueError:
            new[-1] = one[-1]
        return new

    def populate(data, atleast):
        newData = [dd.tolist() for dd in data]
        if atleast - len(newData) < 0:
            try:
                return [choice(newData) for _ in xrange(atleast)]
            except:
                set_trace()
        else:
            for _ in xrange(atleast - len(newData)):
                one = choice(data)
                neigh = knn(one, data)[1:k + 1]
                try:
                    two = choice(neigh)
                except IndexError:
                    two = one
                newData.append(extrapolate(one, two))
            return newData

    def populate2(data1, data2):
        newData = []
        for _ in xrange(atleast):
            for one in data1:
                neigh = kfn(one, data)[1:k + 1]
                try:
                    two = choice(neigh)
                except IndexError:
                    two = one
                newData.append(extrapolate(one, two))
        return [choice(newData) for _ in xrange(atleast)]

    def depopulate(data):
        return [choice(data).tolist() for _ in xrange(atmost)]

    newCells = []
    klass = lambda df: df[df.columns[-1]]
    count = Counter(klass(data))
    major, minor = count.keys()

    for u in count.keys():
        if u == minor:
            newCells.extend(populate([r for r in data.as_matrix() if r[-1] == u], atleast=atleast))
        if u == major:
            newCells.extend(populate([r for r in data.as_matrix() if r[-1] == u], atleast=atmost))
        else:
            newCells.extend([r.tolist() for r in data.as_matrix() if r[-1] == u])
    # set_trace()
    return pd.DataFrame(newCells, columns=data.columns)


def __test_smote():
    """
    A test case goes here
    :return:
    """
    pass


if __name__ == "__main__":
    __test_smote()
