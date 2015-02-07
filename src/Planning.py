from __future__ import print_function

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
from dectree import *
from hist import *
from smote import *
import makeAmodel as mam
from methods1 import *
import numpy as np
import pandas as pd
import sk



#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# PLANNING PHASE
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
class deltas():

  def __init__(self, row, myTree):
    self.loc = drop(row, myTree)
    self.contrastSet = None
    self.newRow = row;
    self.score = self.scorer(self.loc)
  def scorer(self, node):
    return mean([r.cells[-2] for r in node.rows])
  def createNew(self, low, hi, N = 1):
    for _ in xrange(N):
      new = max(lo, min(hi, lo + rand() * abs(hi - lo)))
  def applyPatch(self, keys):
    for stuff in self.contrastSet:
      lo, hi = stuff[1]
      pos = keys[stuff[0].name]
      self.newRow.cells[pos] = createNew(lo, hi)
    return self.newRow

class treatments():
  "Treatments"
  def __init__(self, train = None, test = None,
               verbose = True, smoteit = False):
    self.train_DF, self.test_DF = createTbl(train), createTbl(test)
    self.verbose, self.smoteit = verbose, smoteit
    self.mod, self.keys = [], self.getKey()

  def leaves(self, node):
    L = []
    if len(node.kids) > 1:
      for l in node.kids:
        L.extend(self.leaves(l))
      return L
    elif len(node.kids) == 1:
      return node.kids
    else:
      return node

  def scorer(self, node):
    return mean([r.cells[-2] for r in node.rows])

  def isBetter(self, me, others):
    some1 = False
    for notme in others:
      if self.scorer(notme) < self.scorer(me):
        return False, notme.branch  # False here is for the 'notFound' variable
      else:
        return True, []

  def finder(self, node):
    notFound = True; oldNode = []
    if notFound == True and node.lvl > -1:
      # Go up one Level
        _up = node.up
      # look at the kids
        kids = [k for k in _up.kids if not k in oldNode]
        _kids = [self.leaves(_k) for _k in kids]
        print('Searching in - ', [(k[0].name, k[1]) for k in _up.branch])
        notFound, branch = self.isBetter(node, _kids)
        oldNode.append(node)
    return branch

  def getKey(self):
    keys = {}
    for i in xrange(len(self.test_DF.headers)):
      keys.update({self.test_DF.headers[i].name[1:]:i})
    return keys

  def main(self):
    #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # Main
    #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    # Training data
    if self.smoteit:
      self.train_DF = SMOTE(data = self.train_DF, atleast = 50, atmost = 100)

    # Decision Tree
    t = discreteNums(self.train_DF, map(lambda x: x.cells, self.train_DF._rows))
    myTree = tdiv(t)
    if self.verbose: showTdiv(myTree)

    # Testing data
    testCase = self.test_DF._rows
    newTab = []
    for tC in testCase:
      newRow = tC;
      node = deltas(newRow, myTree)  # A delta instance for the rows
      if node.score == 0:
        print('No Contrast Set.')
        node.contrastSet = []
        self.mod.append(node.newRow)
      else:
        print('Obtaining contrast set. .. ...')
        node.contrastSet = self.finder(node.loc)
        self.mod.append(node.applyPatch(self.keys))
      print(node.__dict__)

    return clone(self.test_DF, rows = self.mod, discrete = True)

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


if __name__ == '__main__':
  planningTest()
