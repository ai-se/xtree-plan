#! /Users/rkrsn/anaconda/bin/python
from __future__ import print_function
from __future__ import division
from os import environ
from os import getcwd
from pdb import set_trace
from weights import weights as W
from random import uniform, randint
import sys

# Update PYTHONPATH
HOME = environ['HOME']
axe = HOME + '/git/axe/axe/'  # AXE
pystat = HOME + '/git/pystat/'  # PySTAT
cwd = getcwd()  # Current Directory
sys.path.extend([axe, pystat, cwd])

from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from Prediction import *
from _imports import *
# from abcd import _Abcd
from cliffsDelta import *
from contrastset import *
# from dectree import *
from hist import *
from smote import *
import makeAmodel as mam
from methods1 import *
import numpy as np
import pandas as pd
from counts import *
# import sk

def settings(**d): return o(
  name = "WHICH 2",
  what = "WHICH2 - A Contrast Set Planner",
  author = "Rahul Krishna",
  copyleft = "(c) 2014, MIT license, http://goo.gl/3UYBp",
  seed = 1,
  f = None,
  ).update(**d)

opt = settings()

class treatments():
  def __init__(self, train, test):
    self.test, self.train = test, train
    self.new_Tab = []
    self.train_df = createTbl(self.train, isBin = True, bugThres = 1)


  def clusterer(self):
    clusters = list(set([f.cells[-1] for f in self.train_df._rows]))
    ClusterRows = {}
    for _id in list(set(clusters)):
      ClusterRows.update({_id:[f for f in self.train_df._rows \
                                if f.cells[-1] == _id]})
    return ClusterRows

  def knn(self, one, two):
    pdistVect = []
#    set_trace()
    for ind, n in enumerate(two):
      pdistVect.append([ind, euclidean(one[:-1], n[:-1])])
    indices = sorted(pdistVect, key = lambda F:F[1], reverse = True)
    return [two[n[0]] for n in indices]

  def pairs(self, lst):
    for j in lst[0:]:
      last = j
      for self in lst[0:]:
        yield last, self

  def getMeans(self, ClusterRows):
    vertices = []
    for r in ClusterRows:
      vertex = []
      dat = ClusterRows[r];
      for indx in xrange(len(dat[0].cells) - 1):
        try:
          vertex.append(float(np.mean([k.cells[indx] for k in dat])))
        except TypeError:
          set_trace()
      vertices.append(vertex)
    return vertices

  def getHyperplanes(self):
    hyperPlanes = []
    ClusterRows = self.getMeans(self.clusterer());
    while ClusterRows:
      one = ClusterRows.pop()
      try:
        two = self.knn(one, ClusterRows)[1]
      except IndexError:
        two = one
      hyperPlanes.append([one, two])
    return hyperPlanes

  def projection(self, one, two, three):
    plane = [b - a for a, b in zip(one, two)]
    norm = np.linalg.norm(plane)
    unitVect = [p / norm for p in plane]
    proj = np.dot(three, unitVect)
    return proj

#   def rankedFeatures(self, rows, t, features = None):
#     features = features if features else t.indep[:-1]
#     klass = t.klass[0].col
#     def ranked(f):
#       syms, at, n = {}, {}, len(rows)
# #       for x in f.counts.keys():
# #         syms[x] = Sym()
#       for row in rows:
#         key = row.cells[f.col]
#         val = row.cells[klass - 1]
# #         syms[key] + val
#         at[key] = at.get(key, []) + [row]
#       e = 0
#       for val in syms.values():
#         if val.n:
#           e += val.n / n * val.ent()
#       return e, f, syms, at
#     return sorted(ranked(f) for f in features)
#
#
#   def infogain(self, t, opt = The.tree):
#     for f in t.headers:
#       f.selected = False
#     lst = self.rankedFeatures(t._rows[:-1], t)
#     n = int(len(lst))
#     n = max(n, 1)
#     for _, f, _, _ in lst[:n]:
#       f.selected = True
#     return [(f, at) for e, f, syms, at in lst[:n]]

  def fWeight(self, criterion = 'Entropy'):
    lbs = W().weights(self.train_df)
    return [l / max(lbs) * 0.9 for l in lbs]

  def main(self):
    hyperPlanes = self.getHyperplanes()
    self.test_df = createTbl(self.test, isBin = True, bugThres = 1)
    opt.f = self.fWeight()
    for rows in self.test_df._rows:
      newRow = rows
      if rows.cells[-2] > 0:
        vertices = sorted(hyperPlanes, key = lambda F:self.projection(F[0][:-1],
          F[1][:-1], rows.cells[:-2]), reverse = True)[0]
        [good, bad] = sorted(vertices, key = lambda F: F[-1])
        newRow.cells[:-2] = [my + f * (better - my) for f, better,
                            my in zip(opt.f, good[:-1], rows.cells[:-2])]
        self.new_Tab.append(newRow)

    return clone(self.test_df
                 , rows = [r.cells for r in self.new_Tab]
                 , discrete = True)

def testPlanner2():
  dir = '../Data'
  one, two = explore(dir)
  newTab = treatments(one[0], two[0]).fWeight()

if __name__ == '__main__':
  testPlanner2()

