#! /Users/rkrsn/anaconda/bin/python
from __future__ import print_function
from __future__ import division
from os import environ, getcwd
from pdb import set_trace
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
from abcd import _Abcd
from cliffsDelta import *
from contrastset import *
# from dectree import *
from hist import *
from smote import *
import makeAmodel as mam
from methods1 import *
import numpy as np
import pandas as pd
import sk

def lookup():
  LUT = {}
  LUT.update({'ant': (0.16, 0.33, 8, 1, 3, 8, False, True)})
  LUT.update({'camel': (0.87, 0.49, 1, 0.67, 4, 14, True, True)})
  LUT.update({'ivy': (0.89, 1, 1, 0.99, 1, 16, False, True)})
  LUT.update({'jedit': (0.68, 0.12, 8, 0.63, 2, 1, False, True)})
  LUT.update({'log4j': (0.58, 0.12, 8, 0.63, 2, 1, False, False)})
  LUT.update({'lucene': (0.62, 0.54, 1, 1, 1, 16, True, False)})
  LUT.update({'poi': (0.06, 0.38, 5, 0.95, 1, 13, True, True)})
  LUT.update({'synapse': (0.09, 0.01, 9, 0.59, 5, 5, False, False)})
  LUT.update({'velocity': (1, 1, 1, 1, 1, 9, True, False)})
  LUT.update({'xalan': (0.64, 0.36, 5, 0.88, 1, 18, False, False)})
  LUT.update({'xerces': (0.02, 0.25, 5, 0.64, 3, 20, True, False)})
  return LUT

class deltas():

  def __init__(self, row, myTree):
    self.row = row
    self.loc = drop(row, myTree)
    self.contrastSet = None
    self.newRow = row;
    self.score = self.scorer(self.loc)
  def scorer(self, node):
    return mean([r.cells[-2] for r in node.rows])
  def createNew(self, stuff, keys, N = 1):
    newElem = [];
    tmpRow = self.row
    for _ in xrange(N):
      for s in stuff:
        lo, hi = s[1]
        pos = keys[s[0].name]
        tmpRow.cells[pos] = float(max(lo, min(hi, lo + rand() * abs(hi - lo))))
      newElem.append(tmpRow)
    return newElem
  def patches(self, keys, N_Patches = 10):
    # Search for the best possible contrast set and apply it
    isles = []
    newRow = self.row
    for stuff in self.contrastSet:
      isles.append(self.createNew(stuff, keys, N = N_Patches))
    return isles


class store():
  def __init__(self, node):
    self.node = node
    self.dist = 0
    self.DoC = 0
    self.score = self.scorer(node)
  def scorer(self, node):
    return mean([r.cells[-2] for r in node.rows])


class treatments():
  "Treatments"
  def __init__(self, train = None, test = None,
               verbose = True, smoteit = False):
    self.train, self.test = train, test
    self.train_DF = createTbl(train, _smote = smoteit, isBin = True) 
    self.test_DF = createTbl(test, isBin = True)
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
    Compare [me] with a buch of [others,...], return the best person
    """
    for notme in others:
#       if '%.2f' % self.scorer(notme) == 0:
      if self.scorer(notme) < self.scorer(me):
        return True, notme.branch
      else:
        return False, []

#  def finder(self, node, oldNode = [], branch = [], Found = False):
#    """
#    A recursive searcher that looks for the nearest neighbor whose better than
#    a given node.
#    """
#    out = branch
#    if not Found:
#      if node.lvl > -1:
#        _kids = []
#        oldNode.append(node)
#        _up = node.up
##         print('Current- ', node.branch, 'Level - ', node.lvl)
#        kids = [k for k in _up.kids]
#        _kids.extend([self.leaves(_k) for _k in kids])
#        _kids = self.flatten(_kids)
##         print('Kids', _kids)
#        Found, branch = self.isBetter(node, _kids)
#        out = self.finder(_up, oldNode = oldNode, branch = branch, Found = Found)
#
#      else:
#        _kids = []
#        kids = [k for k in node.kids if not k in oldNode]
#        for k in kids:
#          for kk in k.kids: out = self.finder(kk, oldNode = oldNode,
#                                        branch = branch, Found = Found)
#
##     print(out)
#    return out
#
  def attributes(self, nodes):
    """
    A method to handle unique branch variables that charaterizes
    a bunch of nodes.
    """
    xx = []; attr = []
    def seen(x):
  		xx.append(x)
    for node in nodes:
      if not node.node.branch in xx:
        attr.append(node.node.branch)
        seen(node.node.branch)
    return attr

  def finder2(self, node, alpha = 0.5):
    """
    finder2 is a more elegant version of finder that performs a search on
    the entire tree to find leaves which are better than a certain 'node'
    """

    def range(a_tuple):
      return ((a_tuple[0])+(a_tuple[1]))/2
    vals = []
    current = store(node)
    while node.lvl > -1:
      node = node.up

    leaves = self.flatten([self.leaves(_k) for _k in node.kids])

    for leaf in leaves:
      l = store(leaf)
      for b in leaf.branch:
        dist = []
        if b[0] in [bb[0] for bb in current.node.branch]: 
          l.DoC += 1
          dist.extend([(range(b[1])-range(bb[1]))**2 for bb in current.node.branch if b[0]==bb[0]])
      l.dist = np.sqrt(np.sum(dist))
      vals.append(l)

    vals = sorted(vals, key = lambda F: F.DoC, reverse = False)
    best = [v for v in vals if v.score < alpha * current.score]
    if not len(best)>0: best = vals

    # Get a list of DoCs (DoC -> (D)epth (o)f (C)orrespondence, btw..)
    # set_trace()
    attr = {}
    bests = {}
    unq = list(set([v.DoC for v in best])) # A list of all DoCs..
    for dd in unq: 
      bests.update({dd:sorted([v for v in best if v.DoC == dd], key = lambda F: F.dist)})
      attr.update({dd:self.attributes(sorted([v for v in best if v.DoC == dd], key = lambda F: F.dist))})   
    # set_trace()
    # print(attr, unq)
    try:
      return bests, attr[unq[0]][0], attr[unq[0]][-1], attr[unq[-1]][0], attr[unq[-1]][-1]
    except IndexError:
      set_trace()

  def getKey(self):
    keys = {}
    for i in xrange(len(self.test_DF.headers)):
      keys.update({self.test_DF.headers[i].name[1:]:i})
    return keys

  def main(self):
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # Main
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    # Training data
    # if self.smoteit:
    #   # print('Smoting')
    #   self.train_DF = SMOTE(data = self.train_DF, atleast = 50, atmost = 100)

    # Decision Tree
    tmpRow = []
    t = discreteNums(self.train_DF, map(lambda x: x.cells, self.train_DF._rows))
    myTree = tdiv(t)
    if self.verbose: showTdiv(myTree)

    # Testing data
    testCase = self.test_DF._rows
    newTab = []
    weights = []
    for tC in testCase:
      newRow = tC;
      node = deltas(newRow, myTree)  # A delta instance for the rows

      if newRow.cells[-2] == 0:
        node.contrastSet = []
        self.mod.append(node.newRow)
      else:
        bests, far, farthest, near, nearest = self.finder2(node.loc)
        # set_trace()
 
        # Examine 4 possible contrast set values (nearest best, farthest best,
        # best branch in the same level, and the nearest branch in the upper 
        # level.) I call these nearest, farthest, far, and near.
        node.contrastSet = [far, farthest, near, nearest]

        # Now generate 4 patches (one for each contrast set). Each patch has
        # 10 potential solutions..
        patch = node.patches(self.keys, N_Patches = 10)
        
        found = False
        while not found and patch:
         # print(len(patch))
          p = patch.pop();
          tmpTbl = clone(self.test_DF,
                        rows = [k.cells for k in p],
                        discrete = True)
          mass = CART(createTbl(self.train, _smote = False, isBin = True)
                    , tmpTbl
                    , tunings = None
                    , smoteit = True
                    , duplicate = True)
          # print(tC.cells[-2] > np.mean(mass)) 
          found = tC.cells[-2] > np.mean(mass)
          # life -= 1; 
          # print(len(patch))
          # set_trace()
        self.mod.append(choice(tmpTbl._rows))


      # <<<<<<<<<<< Debug >>>>>>>>>>>>>>>
        # set_trace()

#       if node.score == 0:
#         node.contrastSet = []
#         self.mod.append(node.newRow)
#       else:
#         node.contrastSet = self.finder(node.loc)
#         self.mod.append(node.applyPatch(self.keys))
#
    return clone(self.test_DF, rows = [k.cells for k in self.mod], discrete = True)

def planningTest():
  # Test contrast sets
  n = 0
  dir = '../Data'
  one, two = explore(dir)
  # Training data
  newTab = treatments(train = one[n],
                      test = two[n],
                      verbose = False,
                      smoteit = False).main()

  # <<<<<<<<<<< Debug >>>>>>>>>>>>>>>
  # set_trace()

if __name__ == '__main__':
  planningTest()
