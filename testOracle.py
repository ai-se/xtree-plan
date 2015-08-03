#! /Users/rkrsn/miniconda/bin/python
from __future__ import print_function, division

from os import environ, getcwd
from os import walk
from pdb import set_trace
import sys

# Update PYTHONPATH
HOME = environ['HOME']
axe = HOME + '/git/axe/axe/'  # AXE
pystat = HOME + '/git/pystats/'  # PySTAT
cwd = getcwd()  # Current Directory
sys.path.extend([axe, pystat, cwd])

from Prediction import *
from _imports import dEvol
from abcd import _Abcd
from demos import cmd
from methods1 import *
from sk import rdivDemo
import numpy as np
import pandas as pd
import csv
from numpy import sum


class counter():

  def __init__(self, before, after, indx):
    self.indx = indx
    self.actual = before
    self.predicted = after
    self.TP, self.TN, self.FP, self.FN = 0, 0, 0, 0
    for a, b in zip(self.actual, self.predicted):
      if a == 1 and b == 1:
        self.TP += 1
      if a == 0 and b == 0:
        self.TN += 1
      if a == 0 and b == 1:
        self.FP += 1
      if a == 1 and b == 0:
        self.FN += 1

  def stats(self):
    Sen = self.TP / (self.TP + self.FN)
    Spec = self.TN / (self.TN + self.FP)
    Prec = self.TP / (self.TP + self.FP)
    Acc = (self.TP + self.TN) / (self.TP + self.FN + self.TN + self.FP)
    F1 = 2 * self.TP / (2 * self.TP + self.FP + self.FN)
    G = 2 * Sen * Spec / (Sen + Spec)
    G1 = Sen * Spec / (Sen + Spec)
    return Sen, 1 - Spec, Prec, Acc, F1, G


class ABCD():

  "Statistics Stuff, confusion matrix, all that jazz..."

  def __init__(self, before, after):
    self.actual = before
    self.predicted = after

  def all(self):
    uniques = set(self.actual)
    for u in list(uniques):
      yield counter(self.actual, self.predicted, indx=u)


class data():

  """
  Hold training and testing data
  """

  def __init__(self, dataName='ant', dir="./Data/Jureczko"):
    projects = [Name for _, Name, __ in walk(dir)][0]
    numData = len(projects)  # Number of data
    one, two = explore(dir)
    data = [one[i] + two[i] for i in xrange(len(one))]

    def withinClass(data):
      N = len(data)
      return [(data[:n], [data[n]]) for n in range(1, N)]

    def whereis():
      for indx, name in enumerate(projects):
        if name == dataName:
          return indx
    self.train = [dat[0] for dat in withinClass(data[whereis()])]
    self.test = [dat[1] for dat in withinClass(data[whereis()])]


class testOracle():

  def __init__(self, file='ant', tuned=True):
    self.file = file
    self.train = createTbl(data(dataName=self.file).train[-1], isBin=True)
    self.test = createTbl(data(dataName=self.file).test[-1], isBin=True)
    self.param = dEvol.tuner(rforest, data(dataName=self.file).train[-1]) if \
        tuned else None

  def main(self):
    Pd, Pf = [], []
    actual = Bugs(self.test)
    predicted = rforest(
        self.train,
        self.test,
        tunings=self.param,
        smoteit=True)
    print("Bugs, Pd, Pf")
    try:
      for k in ABCD(before=actual, after=predicted).all():
        print('%d, %0.2f, %0.2f' % (k.indx, k.stats()[0], k.stats()[1]))
    except:
      pass
    # ---------- DEBUG ----------
    #   set_trace()

if __name__ == "__main__":
  for file in ['ant', 'camel', 'ivy',
               'jedit', 'log4j', 'pbeans',
               'lucene', 'poi', 'synapse',
               'velocity', 'xalan', 'xerces']:
    print('### ' + file + ': Tuned')
    testOracle(file).main()

    print('### ' + file + ': Untuned')
    testOracle(file, tuned=False).main()
