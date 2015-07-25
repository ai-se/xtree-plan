#! /Users/rkrsn/miniconda/bin/python
from __future__ import division
from __future__ import print_function

from os import environ
from os import getcwd
from os import system
from os import walk
from pdb import set_trace
from random import randint as randi
from random import sample
from random import uniform as rand
from random import shuffle
from subprocess import PIPE
from subprocess import call
import sys
import csv

# Update PYTHONPATH
HOME = environ['HOME']
axe = HOME + '/git/axe/axe/'  # AXE
pystat = HOME + '/git/pystat/'  # PySTAT
cwd = getcwd()  # Current Directory
sys.path.extend([axe, pystat, cwd])


from numpy import median
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
import pandas

from Prediction import CART as cart
from Prediction import formatData
from Planner2 import *
from cliffsDelta import cliffs
from demos import cmd
from methods1 import *
from sk import rdivDemo
from sk import scottknott
from smote import SMOTE
from table import clone


def write2file(data, fname='Untitled', ext='.txt'):
  with open(fname + ext, 'w') as fwrite:
    writer = csv.writer(fwrite, delimiter=',')
    for b in data:
      writer.writerow(b)


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
      clf = DecisionTreeRegressor(random_state=1)
    else:
      clf = DecisionTreeRegressor(max_depth=int(self.tunings[0]),
                                  min_samples_split=int(self.tunings[1]),
                                  min_samples_leaf=int(self.tunings[2]),
                                  max_features=float(self.tunings[3] / 100),
                                  max_leaf_nodes=int(self.tunings[4]),
                                  criterion='entropy', random_state=1)
    features = self.train.columns[:-2]
    klass = self.train[self.train.columns[-2]]
    # set_trace()
    clf.fit(self.train[features].astype('float32'), klass.astype('float32'))
    preds = clf.predict(
        self.test[self.test.columns[:-2]].astype('float32')).tolist()
    return preds

  def rforest(self):
    "  RF"
    # Apply random forest Classifier to predict the number of bugs.
    if not self.tuning:
      clf = RandomForestRegressor(random_state=1)
    else:
      clf = RandomForestRegressor(n_estimators=int(tunings[0]),
                                  max_features=tunings[1] / 100,
                                  min_samples_leaf=int(tunings[2]),
                                  min_samples_split=int(tunings[3]),
                                  random_state=1)
    features = self.train.columns[:-2]
    klass = self.train[self.train.columns[-2]]
    # set_trace()
    clf.fit(self.train[features].astype('float32'), klass.astype('float32'))
    preds = clf.predict(
        self.test[self.test.columns[:-2]].astype('float32')).tolist()
    return preds


class fileHandler():

  def __init__(self, dir='./Seigmund/'):
    self.dir = dir

  def reformat(self, file, train_test=True, ttr=0.5, save=False):
    """
    Reformat the raw data to suit my other codes.
    **Already done, leave SAVE switched off!**
    """
    import csv
    fread = open(self.dir + file, 'r')
    rows = [line for line in fread]
    header = rows[0].strip().split(',')  # Get the headers
    body = [[1 if r == 'Y' else 0 if r == 'N' else r for r in row.strip().split(',')]
            for row in rows[1:]]
    shuffle(body)
    if save:
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
      # call(["mkdir", "./Data/" + file[:-7]], stdout=PIPE)
      with open("./SeigmundTT/" + file[:-7] + '/Train.csv', 'w+') as fwrite:
        writer = csv.writer(fwrite, delimiter=',')
        train = sample(body, int(ttr * len(body)))
        writer.writerow(header)
        for b in train:
          writer.writerow(b)

      with open("./SeigmundTT/" + file[:-7] + '/Test.csv', 'w+') as fwrite:
        writer = csv.writer(fwrite, delimiter=',')
        test = [b for b in body if not b in train]
        writer.writerow(header)
        for b in test:
          writer.writerow(b)
#       return header, train, test
    else:
      return header, body

  def file2pandas(self, file):
    fread = open(file, 'r')
    rows = [line for line in fread]
    head = rows[0].strip().split(',')  # Get the headers
    body = [[1 if r == 'Y' else 0 if r == 'N' else r for r in row.strip().split(',')]
            for row in rows[1:]]
    return pandas.DataFrame(body, columns=head)

  def explorer(self, name):
    files = [filenames for (
        dirpath,
        dirnames,
        filenames) in walk(self.dir)][0]
    for f in files:
      if f[:-7] == name:
        self.reformat(f)
    datasets = []
    projects = {}
    for (dirpath, dirnames, filenames) in walk(cwd + '/SeigmundTT/'):
      if name in dirpath:
        datasets.append([dirpath, filenames])
    return datasets

  def explorer2(self, name):
    files = [filenames for (
        dirpath,
        dirnames,
        filenames) in walk(self.dir)][0]
    for f in files:
      if name in f:
        return [self.dir + f]
#     return files, [self.file2pandas(dir + file) for file in files]

  def planner(self, train, test):
    train_df = formatData(createTbl(train, _smote=False, isBin=False))
    test_df = formatData(createTbl(test, _smote=False, isBin=False))
    actual = test_df[
        test_df.columns[-2]].astype('float32').tolist()
    before = predictor(train=train_df, test=test_df).rforest()
#           set_trace()
    newTab = treatments(
        train=train,
        test=test, smoteit=False, bin=False).main()
    newTab_df = formatData(newTab)
    after = predictor(train=train_df, test=newTab_df).rforest()
    return actual, before, after

  def delta0(self):
    before, after = open('before_cpm.txt'), open('after_cpm.txt')
    for line1, line2 in zip(before, after):
      row1 = np.array([float(l) for l in line1.strip().split(',')])
      row2 = np.array([float(l) for l in line2.strip().split(',')])
      try:
        yield np.bitwise_xor(row2, row1, dtype='int')
      except:
        set_trace()

  def deltas(self, name):
    predRows = []
    delta = []
    data = self.explorer(name)
    for d in data:
      if name == d[0].strip().split('/')[-1]:
        train = [d[0] + '/' + d[1][1]]
        test = [d[0] + '/' + d[1][0]]
        train_DF = createTbl(train, isBin=False)
        test_df = createTbl(test, isBin=False)
        self.headers = train_DF.headers
        write2file(
            map(
                lambda r: r.cells[
                    :-2],
                test_df._rows),
            fname='before_cpm')  # save file

        """
        Apply Learner
        """
        for _ in xrange(10):
          newTab = treatments(
              train=train,
              test=test,
              smoteit=False,
              bin=False).main()
          write2file(
              map(
                  lambda r: r.cells[
                      :-2],
                  newTab._rows),
              fname='after_cpm')  # save file
          delta.append([d for d in self.delta0()])

        return np.array(
            np.sum(delta[0], axis=0), dtype='float') / np.size(delta[0], axis=0)

  def flatten(self, x):
    """
    Takes an N times nested list of list like [[a,b],[c, [d, e]],[f]]
    and returns a single list [a,b,c,d,e,f]
    """
    result = []
    for el in x:
      if hasattr(el, "__iter__") and not isinstance(el, basestring):
        result.extend(self.flatten(el))
      else:
        result.append(el)
    return result

  def main(self, name='Apache', reps=20):
    effectSize = []
    Accuracy = []
    out_auc = [name]
    out_md = [name]
    out_acc = [name]
    for _ in xrange(reps):
      data = self.explorer(name)
      for d in data:
        if name == d[0].strip().split('/')[-1]:
          #           set_trace()
          train = [d[0] + '/' + d[1][1]]
          test = [d[0] + '/' + d[1][0]]
          actual, before, after = self.planner(train, test)
          cliffsdelta = cliffs(lst1=actual, lst2=after).delta()
          out_auc.append(sum(after) / sum(before))
          out_md.append(median(after) / median(before))
          out_acc.extend(
              [(1 - abs(b - a) / a) * 100 for b, a in zip(before, actual)])
    return out_acc, out_auc, out_md
    #----------- DEGUB ----------------
#     set_trace()


def deltasTester(name):
  # print('##', name)
  f = fileHandler()
  delta = [f.deltas(name) for _ in xrange(24)]
  y = np.median(delta, axis=0)
  yhi, ylo = np.percentile(delta, q=[75, 25], axis=0)
  dat1 = sorted([(h.name[1:], a, b, c) for h, a, b, c in zip(
      f.headers[:-2], y, ylo, yhi)], key=lambda F: F[1])
  dat = np.asarray([(d[0], n, d[1], d[2], d[3])
                    for d, n in zip(dat1, range(1, 21))])
  with open('/Users/rkrsn/git/GNU-Plots/rkrsn/errorbar/%s.csv' % (name), 'w') as csvfile:
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

  f = open('./dataCPM.txt')
  for line in f:
    lst.append(striplines(line[:-1]))

  rdivDemo(lst, isLatex=False)
#   set_trace()


def _test(name='Apache', doWhat='Accuracy'):
  Accuracy = []
  Gain = []
  medianDelta = []
  a, b, c = fileHandler().main(name, reps=10)
  if doWhat == 'AUC':
    print(b)
  elif doWhat == 'Median':
    print(c)

if __name__ == '__main__':

#  rdiv()
  #   _testPlot()
  #  _test('BDBC', 'Median')
  for name in ['Apache', 'BDBC', 'BDBJ', 'LLVM', 'X264', 'SQL']:
    # deltasTester(name)
    _test(name=name, doWhat='AUC')
#   eval(cmd())
