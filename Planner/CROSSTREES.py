#! /Users/rkrsn/anaconda/bin/python
from __future__ import print_function
from __future__ import division

from pdb import set_trace
from os import environ, getcwd
from os import walk
from os import remove as rm
from os.path import expanduser
from pdb import set_trace
import sys

# Update PYTHONPATH
HOME = expanduser('~')
axe = HOME + '/git/axe/axe/'  # AXE
pystat = HOME + '/git/pystats/'  # PySTAT
cwd = getcwd()  # Current Directory
sys.path.extend([axe, pystat, cwd])

from pdb import set_trace
from random import uniform, randint, shuffle
import csv

from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from collections import Counter

from _imports import *
from abcd import _Abcd
from smote import *
import _imports.makeAmodel as mam
from methods1 import *
import numpy as np
import pandas as pd
import sk


def genTable(tbl, rows):
  name = str(randint(0, 1000))
  header = [h.name for h in tbl.headers[:-1]]
  with open('tmp0.csv', 'w') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerow(header)
    for el in rows:
      writer.writerow(el[:-1])
  new = createTbl(['tmp0.csv'])
  rm('tmp0.csv')
  return new



class changes():

  def __init__(self):
    self.log = {}

  def save(self, name=None, old=None, new=None):
    if not old == new:
      self.log.update({name: (old, new)})


class deltas():

  def __init__(self, row, myTree, majority=True):
    self.row = row
    self.loc = drop(row, myTree)
    self.contrastSet = None
    self.newRow = row
    self.score = self.scorer(self.loc)
    self.change = []

  def scorer(self, node):
    return np.mean([r.cells[-2] for r in node.rows])

  def createNew(self, stuff, keys, N=1):
    newElem = []
    tmpRow = self.row
    for _ in xrange(N):
      C = changes()
      for s in stuff:
        lo, hi = s[1]
        pos = keys[s[0].name]
        old = tmpRow.cells[pos]
        new = float(max(lo, min(hi, lo + rand() * abs(hi - lo))))
        C.save(name=s[0].name, old=old, new=new)
        tmpRow.cells[pos] = new
      self.change.append(C.log)
      newElem.append(tmpRow)
    return newElem

  def patches(self, keys, N_Patches=10):
    # Search for the best possible contrast set and apply it
    isles = []
    newRow = self.row
    for stuff in self.contrastSet:
      isles.append(self.createNew(stuff, keys, N=N_Patches))
    return isles, self.change


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
      return np.mean([r.cells[-2]
                      for r in node.rows if r.cells[-1] == unq[id]])
    else:
      return np.mean([r.cells[-2] for r in node.rows])


class xtrees():

  "Treatments"

  def __init__(self, train=None, test=None, test_DF=None,
               verbose=True, smoteit=True, bin=False, majority=True):
    self.train, self.test = train, test
    try:
      self.train_DF = createTbl(train, _smote=False, isBin=bin)
    except:
      set_trace()
    if not test_DF:
      self.test_DF = createTbl(test, isBin=bin)
    else:
      self.test_DF = test_DF
    self.verbose, self.smoteit = verbose, smoteit
    self.mod, self.keys = [], self.getKey()
    self.majority = majority
    t = discreteNums(
        createTbl(
            train, isBin=bin), map(
            lambda x: x.cells, self.train_DF._rows))
    self.myTree = tdiv(t)
#     showTdiv(self.myTree)
#     set_trace()

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
    return np.mean([r.cells[-2] for r in node.rows])

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
    current = store(node, majority=self.majority)  # Store current sample
    while node.lvl > -1:
      node = node.up  # Move to tree root

    # Get all the terminal nodes
    leaves = self.flatten([self.leaves(_k) for _k in node.kids])

    for leaf in leaves:
      l = store(leaf, majority=self.majority)
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
          {dd: sorted([v for v in best if v.DoC == dd], key=lambda F: F.score)})
      attr.update({dd: self.attributes(
          sorted([v for v in best if v.DoC == dd], key=lambda F: F.score))})

    if pos == 'near':
      return attr[unq[-1]][0]
    elif pos == 'far':
      return attr[unq[0]][-1]
    elif pos == 'Best':
      return self.attributes([sorted(best, key=lambda F: F.score)[0]])[0]

  def getKey(self):
    keys = {}
    for i in xrange(len(self.test_DF.headers)):
      keys.update({self.test_DF.headers[i].name[1:]: i})
    return keys

  def main(self, justDeltas=False, which='near'):
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # Main
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    Change = []
    testCase = self.test_DF._rows
    for tC in testCase:
      node = deltas(tC, self.myTree)  # A delta instance for the rows
      node.contrastSet = [self.finder2(node.loc, pos=which)]
      patch, change = node.patches(self.keys, N_Patches=1)
      Change.extend(change)
      self.mod.extend(patch[0])
    if justDeltas:
      return Change
    else:
      return genTable(self.test_DF, rows=[k.cells for k in self.mod])


def _planningTest():
  # Test contrast sets
  n = 0
  Dir = 'Data/Jureczko/'
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
