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
from WHAT import treatments as treatments2
from os import walk
from demos import cmd


class run():

  def __init__(
          self,
          pred=CART,
          _smoteit=True,
          _n=-1,
          _tuneit=False,
          dataName=None,
          reps=1,
          extent=0.5,
          fSelect=False,
          Prune=False,
          infoPrune=0.75):
    self.pred = pred
    self.extent = extent
    self.fSelect = fSelect
    self.Prune = Prune
    self.infoPrune = infoPrune
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
            extent=self.extent,
            far=False,
            smote=True,
            resample=True,
            infoPrune=self.infoPrune,
            Prune=self.Prune).main()
      else:
        newTab = treatments2(
            train=self.train[
                self._n],
            test=self.test[
                self._n],
            far=False,
            smote=True,
            resample=True,
            extent=self.extent,
            infoPrune=self.infoPrune,
            Prune=self.Prune).main()

      after = self.pred(train_DF, newTab,
                        tunings=self.tunedParams,
                        smoteit=True)

#       set_trace()
      self.out_pred.append(_Abcd(before=actual, after=before))
      delta = cliffs(lst1=Bugs(predTest), lst2=after).delta()
      self.out.append(delta)
    if self.extent == 0:
      append = 'Base'
    else:
      if self.Prune:
        append = str(
            self.extent) + '_iP(' + str(
            int(self.infoPrune * 100)) + r'%)' if not self.fSelect else str(
            self.extent) + '_w_iP(' + str(
            int(self.infoPrune * 100)) + r'%)'
      else:
        append = str(
            self.extent) if not self.fSelect else str(
            self.extent) + '_w'

    self.out.insert(0, self.dataName + '_' + append)
    self.out_pred.insert(0, self.dataName)
    print(self.out)


def _test(file):
  """
  Baselining
  """
  R = run(
      dataName=file,
      extent=0.00,
      reps=12,
      fSelect=False,
      Prune=False,
      infoPrune=None).go()

#   """
#   Mutation without Feature Selection
#   """
#   R = run(
#       dataName=file,
#       extent=0.25,
#       reps=12,
#       fSelect=False,
#       Prune=False,
#       infoPrune=None).go()
#
#   R = run(
#       dataName=file,
#       extent=0.50,
#       reps=12,
#       fSelect=False,
#       Prune=False,
#       infoPrune=None).go()
#
#   R = run(
#       dataName=file,
#       extent=0.75,
#       reps=12,
#       fSelect=False,
#       Prune=False,
#       infoPrune=None).go()
#
#   """
#   Mutation with Feature Selection
#   """
#   R = run(
#       dataName=file,
#       extent=0.25,
#       reps=12,
#       fSelect=True,
#       Prune=False,
#       infoPrune=None).go()
#   R = run(
#       dataName=file,
#       extent=0.50,
#       reps=12,
#       fSelect=True,
#       Prune=False,
#       infoPrune=None).go()
#   R = run(
#       dataName=file,
#       extent=0.75,
#       reps=12,
#       fSelect=True,
#       Prune=False,
#       infoPrune=None).go()
#   """
#   Information Pruning with Feature Selection
#   """
#   R = run(
#       dataName=file,
#       extent=0.25,
#       reps=12,
#       fSelect=True,
#       Prune=True,
#       infoPrune=0.25).go()
#   R = run(
#       dataName=file,
#       extent=0.25,
#       reps=12,
#       fSelect=True,
#       Prune=True,
#       infoPrune=0.5).go()
#   R = run(
#       dataName=file,
#       extent=0.25,
#       reps=12,
#       fSelect=True,
#       Prune=True,
#       infoPrune=0.75).go()
#
#   R = run(
#       dataName=file,
#       extent=0.50,
#       reps=12,
#       fSelect=True,
#       Prune=True,
#       infoPrune=0.25).go()
#   R = run(
#       dataName=file,
#       extent=0.50,
#       reps=12,
#       fSelect=True,
#       Prune=True,
#       infoPrune=0.50).go()
#   R = run(
#       dataName=file,
#       extent=0.50,
#       reps=12,
#       fSelect=True,
#       Prune=True,
#       infoPrune=0.75).go()
#
#   R = run(
#       dataName=file,
#       extent=0.75,
#       reps=12,
#       fSelect=True,
#       Prune=True,
#       infoPrune=0.25).go()
#   R = run(
#       dataName=file,
#       extent=0.75,
#       reps=12,
#       fSelect=True,
#       Prune=True,
#       infoPrune=0.5).go()
#   R = run(
#       dataName=file,
#       extent=0.75,
#       reps=12,
#       fSelect=True,
#       Prune=True,
#       infoPrune=0.75).go()
#
#   """
#   Information Pruning without Feature Selection
#   """
#   R = run(
#       dataName=file,
#       extent=0.25,
#       reps=12,
#       fSelect=False,
#       Prune=True,
#       infoPrune=0.25).go()
#   R = run(
#       dataName=file,
#       extent=0.25,
#       reps=12,
#       fSelect=False,
#       Prune=True,
#       infoPrune=0.5).go()
#   R = run(
#       dataName=file,
#       extent=0.25,
#       reps=12,
#       fSelect=False,
#       Prune=True,
#       infoPrune=0.75).go()
#
#   R = run(
#       dataName=file,
#       extent=0.50,
#       reps=12,
#       fSelect=False,
#       Prune=True,
#       infoPrune=0.25).go()
#   R = run(
#       dataName=file,
#       extent=0.50,
#       reps=12,
#       fSelect=False,
#       Prune=True,
#       infoPrune=0.50).go()
#   R = run(
#       dataName=file,
#       extent=0.50,
#       reps=12,
#       fSelect=False,
#       Prune=True,
#       infoPrune=0.75).go()
#
#   R = run(
#       dataName=file,
#       extent=0.75,
#       reps=12,
#       fSelect=False,
#       Prune=True,
#       infoPrune=0.25).go()
#   R = run(
#       dataName=file,
#       extent=0.75,
#       reps=12,
#       fSelect=False,
#       Prune=True,
#       infoPrune=0.5).go()
#   R = run(
#       dataName=file,
#       extent=0.75,
#       reps=12,
#       fSelect=False,
#       Prune=True,
#       infoPrune=0.75).go()

if __name__ == '__main__':
  eval(cmd())
