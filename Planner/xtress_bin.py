#! /Users/rkrsn/anaconda/bin/python
from __future__ import print_function
from __future__ import division
from os import environ, getcwd, path
from os import remove as rm
from pdb import set_trace
from random import uniform, randint, shuffle
import sys

# Update PYTHONPATH
HOME = path.expanduser("~")
axe = HOME + '/git/axe/axe/'  # AXE
pystat = HOME + '/git/pystat/'  # PySTAT
cwd = getcwd()  # Current Directory
sys.path.extend([axe, pystat, cwd])

from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from collections import Counter

from _imports import *
from smote import *
import _imports.makeAmodel as mam
from methods1 import *
import numpy as np
import pandas as pd
import sk
import csv


def genTable(tbl, rows):
  name = str(randint(0, 1000))
  header = [h.name for h in tbl.headers[:-1]]
  with open(name + '.csv', 'w') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerow(header)
    for el in rows:
      writer.writerow(el[:-1])
  new = createTbl([name + '.csv'])
  rm(name + '.csv')
  return new

def avoid(name='BDBC'):
  if name == 'Apache':
    return []
  elif name == 'BDBC':
    return [7, 13]
  elif name == 'BDBJ':
    return [0, 1, 2, 5, 6, 10, 13, 14, 16, 17, 18]
  elif name == 'LLVM':
    return [0]
  elif name == 'X264':
    return [0, 8, 12]
  elif name == 'SQL':
    return [0, 2, 7, 10, 23]


def alternates(name='BDBJ'):
  if name == 'Apache':
    return []
  if name == 'BDBC':
    return [range(8, 13), range(14, 18)]
  if name == 'BDBJ':
    return [[11, 12], [3, 4], [7, 8], [23, 24]]
  if name == 'LLVM':
    return []
  if name == 'X264':
    return [[9, 10, 11], [13, 14, 15]]
  if name == 'SQL':
    return [range(3, 7), [25, 27], [28, 29, 30], [32, 33], range(35, 39)]


def flatten(x):
  """
  Takes an N times nested list of list like [[a,b],[c, [d, e]],[f]]
  and returns a single list [a,b,c,d,e,f]
  """
  result = []
  for el in x:
    if hasattr(el, "__iter__") and not isinstance(el, basestring):
      result.extend(flatten(el))
    else:
      result.append(el)
  return result


class changes():

  def __init__(self):
    self.log = {}

  def save(self, name=None, old=None, new=None):
    if not old == new:
      self.log.update({name: (old, new)})


class deltas():

  def __init__(self, row, myTree, name):
    self.row = row
    self.name = name
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
    C = changes()
    for ss in stuff:
      lo, hi = ss[1]
      pos = ss[0].col
      old = tmpRow.cells[pos]
      new = int(lo)
      """
      If current pos is in alternates, only one attribute can be True at at
      a time.
      """
      if pos in flatten(alternates(self.name)):
        for alts in alternates(self.name):
          if pos in alts:
            if old == 1 and new == 0:
              tmpRow.cells[pos] = int(new)
              C.save(name=ss[0].name, old=old, new=new)
              for n in alts:
                if not n == pos:
                  o = tmpRow.cells[n]
                  tmpRow.cells[n] = 1
                  C.save(name=ss[0].name, old=o, new=1)
            elif old == 0 and new == 1:
              tmpRow.cells[pos] = int(new)
              C.save(name=ss[0].name, old=old, new=new)
              for n in alts:
                if not n == pos:
                  o = tmpRow.cells[n]
                  tmpRow.cells[n] = 0
                  C.save(name=ss[0].name, old=o, new=0)
      else:
        tmpRow.cells[pos] = int(lo)
        C.save(name=ss[0].name, old=old, new=new)
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

  def __init__(self, train=None, test=None, test_DF=None, name='Apache',
               verbose=True, smoteit=False, majority=False, bin=False):
    self.train, self.test = train, test
    self.name = name
    try:
      self.train_DF = createTbl(train, _smote=smoteit, isBin=False)
    except:
      set_trace()
    if not test_DF:
      self.test_DF = createTbl(test, isBin=False)
    else:
      self.test_DF = test_DF
    self.ignore = [self.train_DF.headers[i].name[1:]
                   for i in avoid(name=name)]
    self.verbose, self.smoteit = verbose, smoteit
    self.mod, self.keys = [], self.getKey()
    self.majority = majority
    t = discreteNums(
        createTbl(train, _smote=smoteit, isBin=bin),
        map(
            lambda x: x.cells,
            self.train_DF._rows))
    self.myTree = tdiv(t)
#     set_trace()

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
        attr.append(
            [n for n in node.node.branch if n[0].name not in self.ignore])
        seen(node.node.branch)
    return attr

  def finder2(self, node, alpha=0.01, pos='far'):
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
    leaves = flatten([self.leaves(_k) for _k in node.kids])

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
          {dd: sorted([v for v in best if v.DoC == dd], key=lambda F: F.dist)})
      attr.update({dd: self.attributes(
          sorted([v for v in best if v.DoC == dd], key=lambda F: F.score))})

    if pos == 'near':
      return attr[unq[-1]][0]
    elif pos == 'far':
      return attr[unq[0]][-1]

  def getKey(self):
    try:
      return {h.name[1:]: i for i, h in enumerate(self.test_DF.headers)}
    except:
      set_trace()

  def main(self, justDeltas=False):
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # Main
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    Change = []
    testCase = self.test_DF._rows
    for tC in testCase:
      newRow = tC
      node = deltas(
          newRow,
          self.myTree,
          name=self.name)  # A delta instance for the rows
      node.contrastSet = [self.finder2(node.loc, pos='near')]
      patch, change = node.patches(self.keys, N_Patches=1)
      Change.extend(change)
      self.mod.extend(patch[0])
    if justDeltas:
      return Change
    else:
      return genTable(
          self.test_DF, rows=[k.cells for k in self.mod])


def planningTest():
  # Test contrast sets
  n = 0
  Dir = 'Data/'
  one, two = explore(Dir)
  # Training data
  _ = treatments(train=one[n],
                 test=two[n],
                 verbose=True,
                 smoteit=False).main()

  # <<<<<<<<<<< Debug >>>>>>>>>>>>>>>
  set_trace()

if __name__ == '__main__':
  planningTest()
