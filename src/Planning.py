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


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# PLANNING PHASE: 1. Decision Trees, 2. Contrast Sets
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class deltas():
  def __init__(self, row):
    self.row = row;
    self.contrastSet = None
  def patchUp(self):

    pass

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
      if self.score(notme) < self.score(me):
        return False, notme.branch  # False here is for the 'notFound' variable
      else:
        return True, []

  def score(self, node):
    return mean([r.cells[-2] for r in node.rows])


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


  def main(self):
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Main
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # Training data
    train_DF = createTbl(self.train)
    print('Obtaining training data..')
    if self.smoteit:
      train_DF = SMOTE(data = train_DF, atleast = 50, atmost = 100)

    # Testing data
    test_DF = createTbl(self.test)
    print('Obtaining testing data..')

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
      node = deltas(loc)  # A delta instance for the rows
      if self.score(node.row) == 0:
        print('No Contrast Set.')
        node.contrastSet = []
        continue
      else:
        print('Obtaining contrast set. .. ...')
        node.contrastSet = self.finder(node.row)
      print(node.__dict__)
      set_trace()

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
