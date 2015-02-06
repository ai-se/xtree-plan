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


def treatments(train = None, test = None, verbose = True, smoteit = False):

  def remember(node):
   key = node.f.name
   Val = node.val
   contrastSet.update({key: Val})
   # print contrastSet

  def forget(key):
   del contrastSet[key]

  def objectiveScores(lst):
   obj = ([k.cells[-2] for k in lst.rows])
   return np.mean([k for k in obj]), [k for k in obj]

  def compare(node, test):
    leaves = [n for n in test.kids] if len(test.kids) else [test]
#     set_trace()
    for k in leaves:
      return objectiveScores(k) < objectiveScores(node), [objectiveScores(k),
                                                         objectiveScores(node)]
  def getKey():
    keys = {}
    for i in xrange(len(test_DF.headers)):
      keys.update({test_DF.headers[i].name[1:]:i})
    return keys

  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  # New Methods - 02/03/2015
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  def leaves(node):
    L = []
    if node.kids:
     for l in node.kids:
       L.extend(leaves(node.kids))
       return L
    else:
      return L.extend(node)
      return L

  def score(node):
    pass

  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  # Main
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  # Training data
  train_DF = createTbl(train)
  # if smoteit: train_DF = SMOTE(data = train_DF, atleast = 50, atmost = 100)
  # Testing data
  test_DF = createTbl(test)
#   set_trace()
  # Decision Tree

  t = discreteNums(train_DF, map(lambda x: x.cells, train_DF._rows))
  myTree = tdiv(t)
  if verbose: showTdiv(myTree)

  # Testing data
  testCase = test_DF._rows

  keys = getKey();
  newTab = []
  for tC in testCase:
    newRow = tC;
    loc = drop(tC, myTree)
    newNode = loc;
    set_trace()
    if newNode.lvl > 0:
    # Go up one Level
      _up = newNode.up
    # look at the kids
      _kids = _up.kids
      _leaves = [leaves(_k) for _k in _kids]
      set_trace()
    branches = [];
    while newNode.lvl > 0:
      newNode = newNode.up;
      branches.append(newNode);
    # A dict of contrast sets
    contrastSet = {};
    # print loc.f.name, loc.lvl+1, loc.val
    for nn in branches:
      toScan = nn.kids
    #    set_trace()
      for testing in toScan:
        isBetter, obj = compare(loc, testing)
        if isBetter:
          remember(testing)
          continue  # As soon as the first better node is found, exit..

    # Pick a random value in the range suggested by the contrast set and
    # assign it to the row.
    for k in contrastSet:
      min, max = contrastSet[k]
      if isinstance(min, int) and isinstance(max, int):
        val = randint(min, max)
      else: val = uniform(min, max)
      newRow.cells[keys[k]] = val

    newTab.append(newRow.cells)

  updatedTab = clone(test_DF, rows = newTab, discrete = True)
  return updatedTab
#  saveImg(bugs(test_df), num_bins = 50, fname = 'bugsBefore', ext = '.jpg')
#  set_trace()

def planningTest():
  # Test contrast sets
  n = 1
  dir = '../Data'
  one, two = explore(dir)
  # Training data
  train_DF = createTbl(one[n])
  # Test data
  test_df = createTbl(two[n])
  newTab = treatments(train = [one[n][0]],
                      test = [one[n][1]],
                      verbose = False,
                      smoteit = False)


if __name__ == '__main__':
  planningTest()
