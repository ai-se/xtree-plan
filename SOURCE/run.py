#! /Users/rkrsn/miniconda/bin/python
from __future__ import print_function, division
from os import environ, getcwd
import sys

# Update PYTHONPATH
HOME = environ['HOME']
axe = HOME + '/git/axe/axe/'  # AXE
pystat = HOME + '/git/pystats/'  # PySTAT
cwd = getcwd()  # Current Directory
sys.path.extend([axe, pystat, cwd])

from Prediction import *
from Planning import *
from _imports import *
from abcd import _Abcd
from cliffsDelta import cliffs
from contrastset import *
from dectree import *
from methods1 import *
import numpy as np
import pandas as pd
from sk import rdivDemo
from pdb import set_trace
from dEvol import tuner
from Planner2 import treatments as treatments2
from os import walk
from demos import cmd


class run():

  def __init__(self, pred=CART, _smoteit=True, _n=-1,
               _tuneit=False, dataName=None, reps=10,
               extent = 0.5, fWeight = False):
    self.pred = pred
    self.extent = extent
    self.fWeight = fWeight
    self.dataName = dataName
    self.out, self.out_pred = [], []
    self._smoteit = _smoteit
    self.train, self.test = self.categorize()
    self.reps = reps
    self._n = _n
    self.tunedParams = None if not _tuneit else tuner(
        self.pred, self.train[_n])

  def categorize(self):
    dir = '../Data'
    self.projects = [Name for _, Name, __ in walk(dir)][0]
    self.numData = len(self.projects)  # Number of data
    one, two = explore(dir)
    data = [one[i] + two[i] for i in xrange(len(one))]

    def withinClass(data):
      N = len(data)
      return [(data[:n], [data[n]]) for n in range(1, N)]

    def whereis():
      for indx, name in enumerate(self.projects):
        if name == self.dataName:
          return indx

    return [
        dat[0] for dat in withinClass(
            data[
                whereis()])], [
        dat[1] for dat in withinClass(
            data[
                whereis()])]  # Train, Test

  def go(self):

    for _ in xrange(self.reps):
      predRows = []
      train_DF = createTbl(self.train[self._n], isBin=True)
      test_df = createTbl(self.test[self._n], isBin=True)
      actual = Bugs(test_df)
      before = self.pred(train_DF, test_df,
                         tunings=self.tunedParams,
                         smoteit=True)

      for predicted, row in zip(before, test_df._rows):
        tmp = row.cells
        tmp[-2] = predicted
        if predicted > 0:
          predRows.append(tmp)

      predTest = clone(test_df, rows=predRows)

      if predRows:
        newTab = treatments2(
            train=self.train[self._n],
            test=self.test[self._n],
            test_df=predTest,
            extent = self.extent,
            far=False,
            infoPrune = 0.75,
            Prune = False).main()
      else:
        newTab = treatments2(
            train=self.train[
                self._n], test=self.test[
                self._n], far=False, extent = self.extent, 
                infoPrune = 0.75, Prune = False).main()

      after = self.pred(train_DF, newTab,
                        tunings=self.tunedParams,
                        smoteit=False)

      self.out_pred.append(_Abcd(before=actual, after=before))
      delta = cliffs(lst1=Bugs(predTest), lst2=after).delta()
      self.out.append(delta)
    self.out.insert(0, self.dataName+'_'+str(self.extent))
    self.out_pred.insert(0, self.dataName)
    print(self.out)


def _test(file):
  R = run(dataName=file, extent = 0.25).go()

if __name__ == '__main__':
  eval(cmd())
