#! /Users/rkrsn/anaconda/bin/python
from __future__ import print_function
from __future__ import division
from os import environ, getcwd
from pdb import set_trace
from random import uniform, randint, shuffle
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

from collections import Counter

from Prediction import *
from _imports import *
from abcd import _Abcd
from cliffsDelta import *
# from contrastset import *
# from dectree import *
from hist import *
from smote import *
import makeAmodel as mam
from methods1 import *
import numpy as np
import pandas as pd
import sk


class deltas():

  def __init__(self, row, myTree, majority=True):
    self.row = row
    self.loc = drop(row, myTree)
    self.contrastSet = None
    self.newRow = row
    self.score = self.scorer(self.loc)

  def scorer(self, node):
    return mean([r.cells[-2] for r in node.rows])

  def createNew(self, stuff, keys, N=1):
    newElem = []
    tmpRow = self.row
    for _ in xrange(N):
      for s in stuff:
        lo, hi = s[1]
        pos = keys[s[0].name]
        tmpRow.cells[pos] = float(max(lo, min(hi, lo + rand() * abs(hi - lo))))
      newElem.append(tmpRow)
    return newElem

  def patches(self, keys, N_Patches=10):
    # Search for the best possible contrast set and apply it
    isles = []
    newRow = self.row
    for stuff in self.contrastSet:
      isles.append(self.createNew(stuff, keys, N=N_Patches))
    return isles


class store():

  def __init__(self, node, majority=False):
    self.node = node
    self.dist = 0
    self.DoC = 0
    self.majority = majority
    self.score = self.scorer(node)

  def minority(self, node):
    unique = list(set([r.cells[-1] for r in node.rows]))
    counts = len(unique) * [0]
#     set_trace()
    for n in xrange(len(unique)):
      for d in [r.cells[-1] for r in node.rows]:
        if unique[n] == d:
          counts[n] += 1
    return unique, counts

  def scorer(self, node):
    if self.majority:
      unq, counts = self.minority(node)
      id, maxel = 0, 0
      for i, el in enumerate(counts):
        if el > maxel:
          maxel = el
          id = i
      return mean([r.cells[-2] for r in node.rows if r.cells[-1] == unq[id]])
    else:
      return mean([r.cells[-2] for r in node.rows])


class xtrees():

  "Treatments"

  def __init__(self, train=None, test=None, test_DF=None,
               verbose=True, smoteit=True, bin=False):
    self.train, self.test = train, test
    self.train_DF = createTbl(train, _smote=smoteit, isBin=bin)
    if not test_DF:
      self.test_DF = createTbl(test, isBin=bin)
    else:
      self.test_DF = test_DF
    self.verbose, self.smoteit = verbose, smoteit
    self.mod, self.keys = [], self.getKey()

  def flatten(self, x):
    """
    Takes an N times nested list of list like [[a,b],[c, [d, e]],[f]]
    and returns a single list [a,b,c,d,e,f]
    """
    result = []
    for el in x:
      if hasattr(el, "__iter__") and not isinstance(el, basestring):
        result.extend(self.flatten(el))
      else:
        result.append(el)
    return result

  def leaves(self, node):
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

  def scorer(self, node):
    """
    Score an leaf node
    """
    return mean([r.cells[-2] for r in node.rows])

  def isBetter(self, me, others):
    """
    Compare [me] with a bunch of [others,...], return the best person
    """
    for notme in others:
      #       if '%.2f' % self.scorer(notme) == 0:
      if self.scorer(notme) < self.scorer(me):
        return True, notme.branch
      else:
        return False, []

  def attributes(self, nodes):
    """
    A method to handle unique branch variables that characterizes
    a bunch of nodes.
    """
    xx = []
    attr = []

    def seen(x):
      xx.append(x)
    for node in nodes:
      if not node.node.branch in xx:
        attr.append(node.node.branch)
        seen(node.node.branch)
    return attr

  def finder2(self, node, alpha=0.5, pos='far'):
    """
    finder2 is a more elegant version of finder that performs a search on
    the entire tree to find leaves which are better than a certain 'node'
    """

    euclidDist = lambda a, b: ((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2) ** 0.5
    midDist = lambda a, b: abs(sum(b) - sum(a)) / 2
    vals = []
    current = store(node)  # Store current sample
    while node.lvl > -1:
      node = node.up  # Move to tree root

    # Get all the terminal nodes
    leaves = self.flatten([self.leaves(_k) for _k in node.kids])

    for leaf in leaves:
      l = store(leaf)
      for b in leaf.branch:
        dist = []
        if b[0] in [bb[0] for bb in current.node.branch]:
          l.DoC += 1
          dist.extend([midDist(b[1], bb[1])
                       for bb in current.node.branch if b[0] == bb[0]])
      l.dist = np.sqrt(np.sum(dist))
      vals.append(l)
    vals = sorted(vals, key=lambda F: F.DoC, reverse=False)
    best = [v for v in vals if v.score < alpha * current.score]
    if not len(best) > 0:
      best = vals

    # Get a list of DoCs (DoC -> (D)epth (o)f (C)orrespondence, btw..)
    # set_trace()
    attr = {}
    bests = {}
    unq = sorted(list(set([v.DoC for v in best])))  # A list of all DoCs..
    for dd in unq:
      bests.update(
          {dd: sorted([v for v in best if v.DoC == dd], key=lambda F: F.dist)})
      attr.update({dd: self.attributes(
          sorted([v for v in best if v.DoC == dd], key=lambda F: F.score))})

    if pos == 'near':
      return attr[unq[-1]][0]
    elif pos == 'far':
      return attr[unq[0]][-1]

  def getKey(self):
    keys = {}
    for i in xrange(len(self.test_DF.headers)):
      keys.update({self.test_DF.headers[i].name[1:]: i})
    return keys

  def main(self):
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # Main
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    # Decision Tree
    t = discreteNums(self.train_DF, map(lambda x: x.cells, self.train_DF._rows))
    myTree = tdiv(t)

    # Testing data
    testCase = self.test_DF._rows
    for tC in testCase:
      newRow = tC
      node = deltas(newRow, myTree)  # A delta instance for the rows

      if newRow.cells[-2] == 0:
        node.contrastSet = []
        self.mod.append(node.newRow)
      else:
        node.contrastSet = [self.finder2(node.loc, pos='near')]

        # Now generate 1 potential patch
        patch = node.patches(self.keys, N_Patches=5)
        # Shuffle
        shuffle(patch)
        p = patch.pop()
        tmpTbl = clone(self.test_DF,
                       rows=[k.cells for k in p],
                       discrete=True)
        self.mod.append(choice(tmpTbl._rows))

      # <<<<<<<<<<< Debug >>>>>>>>>>>>>>>
        # set_trace()
    return clone(self.test_DF, rows=[k.cells for k in self.mod], discrete=True)


def _planningTest():
  # Test contrast sets
  n = 0
  Dir = 'Data/'
  one, two = explore(Dir)
  # Training data
  _ = xtrees(train=one[n],
                 test=two[n],
                 verbose=True,
                 smoteit=False).main()

  # <<<<<<<<<<< Debug >>>>>>>>>>>>>>>
  set_trace()

if __name__ == '__main__':
  _planningTest()
