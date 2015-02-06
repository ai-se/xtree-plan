from __future__ import print_function

from os import environ, getcwd
from pdb import set_trace
from random import uniform, randint
import sys

# Update PYTHONPATH
HOME = environ['HOME']
axe = HOME + '/git/axe/axe/'  # AXE
pystat = HOME + '/git/pystats/'  # PySTAT
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


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# PLANNING PHASE: 1. Decision Trees, 2. Contrast Sets
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class treatments():
  "Treatments"
  def __init__(self, train = None, test = None,
               verbose = True, smoteit = False):
    self.train, self.test = train, test
    self.verbose, self.smoteit = verbose, smoteit

  def leaves(self, node):
    L = []
    if len(node.kids):
      for l in node.kids:
        L.extend(self.leaves(node.kids))
      return L
    else:
      return node

  def isBetter(self, me, others):
    some1 = False
    for notme in others:
     if self.score(notme) == 0:
      return True, notme.branches
  
  def score(self, node):
    return mean([r.cells[-2] f in node.rows])

class deltas():
  def __init__(self, row):
    self.row  =row;
    self.contrastSet = None
  
  def main(self):
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Main
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # Training data
    train_DF = createTbl(self.train)

    if self.smoteit:
      train_DF = SMOTE(data = train_DF, atleast = 1000, atmost = 1001)

    # Testing data
    test_DF = createTbl(self.test)

    # Decision Tree
    t = discreteNums(train_DF, map(lambda x: x.cells, train_DF._rows))
    myTree = tdiv(t)
    if self.verbose: showTdiv(myTree)

    # Testing data
    testCase = test_DF._rows
    newTab = []
    for tC in testCase:
      newRow = tC;
      loc = drop(tC, myTree)
      node = deltas(loc)
      if self.score(node)==0
        node.contrastSet = []
        continue
      elif node.lvl > 0:
      # Go up one Level
        _up = node.up
      # look at the kids
        _kids = [self.leaves(_k) for _k in _up.kids]
        
        set_trace()

def planningTest():
  # Test contrast sets
  n = 2
  dir = '../Data'
  one, two = explore(dir)
  # Training data
  train_DF = createTbl(one[n])
  # Test data
  test_df = createTbl(two[n])
  newTab = treatments(train = one[n],
                      test = two[n],
                      verbose = True,
                      smoteit = True).main()


if __name__ == '__main__':
  planningTest()
