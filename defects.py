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

from CROSSTREES import xtrees
from Prediction import *
from _imports import *
from abcd import _Abcd
from cliffsDelta import cliffs
from dEvol import tuner
from demos import cmd
from methods1 import *
from sk import rdivDemo
import numpy as np
import pandas as pd
import csv
from numpy import sum


def write2file(data, fname='Untitled', ext='.txt'):
  with open(fname + ext, 'w') as fwrite:
    writer = csv.writer(fwrite, delimiter=',')
    for b in data:
      writer.writerow(b)


class run():

  def __init__(
          self, pred=rforest, _smoteit=True, _n=-1
          , _tuneit=False, dataName=None, reps=1):
    self.pred = pred
    self.dataName = dataName
    self.out, self.out_pred = [self.dataName], []
    self._smoteit = _smoteit
    self.train, self.test = self.categorize()
    self.reps = reps
    self._n = _n
    self.tunedParams = None if not _tuneit \
    else tuner(self.pred, self.train[_n])
    self.headers = createTbl(self.train[self._n], isBin=False
                             , bugThres=1).headers

  def categorize(self):
    dir = './Jureczko'
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

    try:
      return [
          dat[0] for dat in withinClass(data[whereis()])], [
          dat[1] for dat in withinClass(data[whereis()])]  # Train, Test
    except:
      set_trace()

  def go(self):

    for _ in xrange(self.reps):
      predRows = []
      train_DF = createTbl(self.train[self._n], isBin=True)
      test_df = createTbl(self.test[self._n], isBin=True)
      actual = np.array(Bugs(test_df))
      before = self.pred(train_DF, test_df,
                         tunings=self.tunedParams,
                         smoteit=True)

      predRows = [row.cells for predicted
                  , row in zip(before, test_df._rows) if predicted > 0]
      predTest = clone(test_df, rows=predRows)

      newTab = xtrees(train=self.train[self._n]
                          , test_DF=predTest, bin=False).main()

      after = self.pred(train_DF, newTab,
                        tunings=self.tunedParams,
                        smoteit=True)

      self.out_pred.append(_Abcd(before=actual, after=before))
      # set_trace()
      delta = cliffs(lst2=Bugs(predTest), lst1=after).delta()
      frac = sum([0 if a < 1 else 1 for a in after]) / \
          sum([0 if b < 1 else 1 for b in before])
      self.out.append(frac)
    print(self.out)

  def delta0(self, norm):
    before, after = open('before.txt'), open('after.txt')
    for line1, line2 in zip(before, after):
      row1 = np.array([float(l) for l in line1.strip().split(',')[:-1]])
      row2 = np.array([float(l) for l in line2.strip().split(',')])
      yield ((row2 - row1) / norm).tolist()

  def deltas(self):
    predRows = []
    delta = []
    train_DF = createTbl(self.train[self._n], isBin=True, bugThres=1)
    test_df = createTbl(self.test[self._n], isBin=True, bugThres=1)
    before = self.pred(train_DF, test_df, tunings=self.tunedParams,
                       smoteit=True)
    allRows = np.array(
        map(
            lambda Rows: np.array(
                Rows.cells[
                    :-
                    1]),
            train_DF._rows +
            test_df._rows))

    def min_max():
      N = len(allRows[0])
      base = lambda X: sorted(X)[-1] - sorted(X)[0]
      return [base([r[i] for r in allRows]) for i in xrange(N)]

    for predicted, row in zip(before, test_df._rows):
      tmp = row.cells
      tmp[-2] = predicted
      if predicted > 0:
        predRows.append(tmp)

    write2file(predRows, fname='before')  # save file

    """
    Apply Learner
    """
    for _ in xrange(1):
      predTest = clone(test_df, rows=predRows)
      newTab = xtrees(train=self.train[self._n], test_DF=predTest).main()
      newRows = np.array(map(lambda Rows: Rows.cells[:-1], newTab._rows))
      write2file(newRows, fname='after')  # save file
      delta.append([d for d in self.delta0(norm=min_max())])

    return delta[0]

    # -------- DEBUG! --------
    # set_trace()


def _test(file='ant'):
  for file in ['ivy', 'jedit', 'lucene', 'poi', 'ant']:
    print('##', file)
    R = run(dataName=file, reps=10).go()


def deltaCSVwriter(type='Indv'):
  if type == 'Indv':
    for name in ['ivy', 'jedit', 'lucene', 'poi', 'ant']:
      print('##', name)
      delta = run(dataName=name, reps=4).deltas()
      y = np.median(delta, axis=0)
      yhi, ylo = np.percentile(delta, q=[75, 25], axis=0)
      dat1 = sorted([(h.name[1:], a, b, c) for h, a, b, c in zip(
          run(dataName=name).headers[:-2], y, ylo, yhi)], key=lambda F: F[1])
      dat = np.asarray([(d[0], n, d[1], d[2], d[3])
                        for d, n in zip(dat1, range(1, 21))])
      with open('/Users/rkrsn/git/GNU-Plots/rkrsn/errorbar/%s.csv' % (name), 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=' ')
        for el in dat[()]:
          writer.writerow(el)
  elif type == 'All':
    delta = []
    for name in ['ivy', 'jedit', 'lucene', 'poi', 'ant']:
      print('##', name)
      delta.extend(run(dataName=name, reps=4).deltas())
    y = np.median(delta, axis=0)
    yhi, ylo = np.percentile(delta, q=[75, 25], axis=0)
    dat1 = sorted([(h.name[1:], a, b, c) for h, a, b, c in zip(
        run(dataName=name).headers[:-2], y, ylo, yhi)], key=lambda F: F[1])
    dat = np.asarray([(d[0], n, d[1], d[2], d[3])
                      for d, n in zip(dat1, range(1, 21))])
    with open('/Users/rkrsn/git/GNU-Plots/rkrsn/errorbar/all.csv', 'w') as csvfile:
      writer = csv.writer(csvfile, delimiter=' ')
      for el in dat[()]:
        writer.writerow(el)


def rdiv():
  lst = []

  def striplines(line):
    listedline = line.strip().split(',')  # split around the = sign
    listedline[0] = listedline[0][2:-1]
    lists = [listedline[0]]
    for ll in listedline[1:-1]:
      lists.append(float(ll))
    return lists

  f = open('./jedit.dat')
  for line in f:
    lst.append(striplines(line[:-1]))

  rdivDemo(lst, isLatex=False)
  set_trace()


def deltaTest():
  for file in ['ivy', 'poi', 'jedit', 'ant', 'lucene']:
    print('##', file)
    R = run(dataName=file, reps=10).deltas()


if __name__ == '__main__':
  _test()
#deltaTest()
#rdiv()
#deltaCSVwriter(type='All')
#deltaCSVwriter(type='Indv')
#  eval(cmd())
