#! /Users/rkrsn/miniconda/bin/python
from __future__ import print_function
from __future__ import division
from os import environ
from os import getcwd
from os import walk
from pdb import set_trace
from random import uniform as rand
from random import randint as randi
from random import sample
import pandas
import sys
from sklearn.tree import DecisionTreeRegressor
# Update PYTHONPATH
HOME = environ['HOME']
axe = HOME + '/git/axe/axe/'  # AXE
pystat = HOME + '/git/pystat/'  # PySTAT
cwd = getcwd()  # Current Directory
WHAT = '../SOURCE/'
sys.path.extend([axe, pystat, cwd, WHAT])

from smote import SMOTE
from methods1 import *


class predictor():

  def __init__(
          self,
          train=None,
          test=None,
          tuning=None,
          smoteit=False,
          duplicate=False):
    self.train = train
    self.test = test
    self.tuning = tuning
    self.smoteit = smoteit
    self.duplicate = duplicate

  def CART(self):
    "  CART"
    # Apply random forest Classifier to predict the number of bugs.
    if self.smoteit:
      self.train = SMOTE(
          self.train,
          atleast=50,
          atmost=101,
          resample=self.duplicate)

    if not self.tuning:
      clf = DecisionTreeRegressor()
    else:
      clf = DecisionTreeRegressor(max_depth=int(self.tunings[0]),
                                  min_samples_split=int(self.tunings[1]),
                                  min_samples_leaf=int(self.tunings[2]),
                                  max_features=float(self.tunings[3] / 100),
                                  max_leaf_nodes=int(self.tunings[4]),
                                  criterion='entropy')
    features = self.train.columns[:-1]
    klass = self.train[self.train.columns[-1]]
    # set_trace()
    clf.fit(self.train[features].astype('float32'), klass.astype('float32'))
    preds = clf.predict(
        self.test[self.test.columns[:-1]].astype('float32')).tolist()
    return preds


def reformat(file, train_test=True, ttr=0.66, save=False):
  """
  Reformat the raw data to suit my other codes.
  **Already done, leave SAVE switched off!**
  """
  fread = open(file, 'r')
  rows = [line for line in fread]
  header = rows[0].strip().split(',')  # Get the headers
  body = [[1 if r == 'Y' else 0 if r == 'N' else r for r in row.strip().split(',')]
          for row in rows[1:]]
  if save:
    import csv
    "Format the headers by prefixing '$' and '<'"
    header = ['$' + h for h in header]
    header[-1] = header[-1][0] + '<' + header[-1][1:]
    "Write Header"
    with open(file, 'w') as fwrite:
      writer = csv.writer(fwrite, delimiter=',')
      writer.writerow(header)
      for b in body:
        writer.writerow(b)
  elif train_test:
    train = sample(body, int(ttr * len(body)))
    test = [b for b in body if not b in train]
    return header, train, test
  else:
    return header, body


def file2pandas(file):
  head, train, test = reformat(file)
  return [pandas.DataFrame(
      train, columns=head), pandas.DataFrame(
      test, columns=head)]


def explorer(dir='../CPM/'):
  files = [filenames for (
      dirpath,
      dirnames,
      filenames) in walk(dir)][0]
  return [file2pandas(dir + file) for file in files]


def main():
  dir = '../CPM/'
  files = explorer(dir)
  train, test = files[0][0], files[0][1]
  before = test[test.columns[-1]].astype('float32').tolist()
  after = predictor(train=train, test=test).CART()
  print(([(1 - abs(b - a) / b) * 100 for b, a in zip(before, after)]))


def _test():
  main()

if __name__ == '__main__':
  _test()
