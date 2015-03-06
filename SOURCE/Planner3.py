#! /Users/rkrsn/anaconda/bin/python
from __future__ import print_function
from __future__ import division
from os import environ
from os import getcwd
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

def settings(**d): return o(
  name = "WHICH 2",
  what = "WHICH2 - A Contrast Set Planner",
  author = "Rahul Krishna",
  copyleft = "(c) 2014, MIT license, http://goo.gl/3UYBp",
  seed = 1,
  f = 0.9,
  ).update(**d)

opt = settings()

class treatments():
  def __init__(i, train, test):
    i.test, i.train = test, train
    i.new_Tab = []
  
  def clusterer(i):
    i.train_df = createTbl(i.train, isBin = True)
    clusters = list(set([f.cells[-1] for f in i.train_df._rows]))
    ClusterRows = {}
    for _id in list(set(clusters)):
      ClusterRows.update({_id:[f for f in i.train_df._rows if f.cells[-1] == _id]}) 
    return ClusterRows
  
  def knn(i, one, two):
    pdistVect = []
#    set_trace()
    for ind, n in enumerate(two):
      pdistVect.append([ind, euclidean(one[:-1], n[:-1])])
    indices = sorted(pdistVect, key = lambda F:F[1], reverse = True)
    return [two[n[0]] for n in indices]
  
  def pairs(i, lst):
    for j in lst[0:]:
      last = j
      for i in lst[0:]:
        yield last, i

  def getMeans(i, ClusterRows):
    vertices = []
    for r in ClusterRows:
      vertex = []
      dat = ClusterRows[r];
      for indx in xrange(len(dat[0].cells)-1):
        try:
          vertex.append(float(np.mean([k.cells[indx] for k in dat])))
        except TypeError:
          set_trace()
      vertices.append(vertex)
    return vertices

  def getHyperplanes(i):
    hyperPlanes = []
    ClusterRows = i.getMeans(i.clusterer());
    while ClusterRows:
      one = ClusterRows.pop()
      try:
        two = i.knn(one, ClusterRows)[0]
      except IndexError: 
        two = one
      hyperPlanes.append([one, two])
    return hyperPlanes

  def projection(i, one, two, three):
    plane = [b - a for a, b in zip(one, two)]
    norm = np.linalg.norm(plane)
    unitVect = [p/norm for p in plane]
    proj = np.dot(three, unitVect)
    return proj
  
  def main(i):
    hyperPlanes = i.getHyperplanes()
    i.test_df = createTbl(i.test)
    for rows in i.test_df._rows:
      newRow = rows
      if rows.cells[-2] > 0:
        vertices = sorted(hyperPlanes, key = lambda F:i.projection(F[0][:-1], 
          F[1][:-1], rows.cells[:-2]), reverse = True)[0]
        [good, bad] = sorted(vertices, key = lambda F: F[-1])
        newRow.cells[:-2] = [my + opt.f*(better - my) for better, 
                            my in zip(good[:-1], rows.cells[:-2])]
        i.new_Tab.append(newRow)
      
    return clone(i.test_df, rows = [r.cells for r in i.new_Tab], discrete = True) 

def testPlanner2():
  dir = '../Data'
  one, two = explore(dir)
  newTab = treatments(one[0],two[0]).main()

if __name__ == '__main__':
  testPlanner2()

