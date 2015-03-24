#! /Users/rkrsn/miniconda/bin/python
from __future__ import print_function
from __future__ import division
from os import environ
from os import getcwd
from os import walk
from os import system
from pdb import set_trace
from random import uniform as rand
from random import randint as randi
from random import sample
from subprocess import call
from subprocess import PIPE
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

from sk import rdivDemo
from smote import SMOTE
from methods1 import *
from WHAT import treatments as WHAT
from Prediction import formatData
from Prediction import CART as cart
from cliffsDelta import cliffs


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


class fileHandler():

  def __init__(self, dir='../CPM/'):
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
      call(["mkdir", "./Data/" + file[:-7]], stdout=PIPE)
      with open("./Data/" + file[:-7] + '/Train.csv', 'w+') as fwrite:
        writer = csv.writer(fwrite, delimiter=',')
        train = sample(body, int(ttr * len(body)))
        writer.writerow(header)
        for b in train:
          writer.writerow(b)

      with open("./Data/" + file[:-7] + '/Test.csv', 'w+') as fwrite:
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

  def explorer(self):
    files = [filenames for (
        dirpath,
        dirnames,
        filenames) in walk(self.dir)][0]
    for f in files:
      self.reformat(f)
    datasets = []
    projects = {}
    for (dirpath, dirnames, filenames) in walk(cwd + '/Data/'):
      datasets.append([dirpath, filenames])
    return datasets[1:]

#     return files, [self.file2pandas(dir + file) for file in files]

  def main(self):
    E = []
    out = []
    data = self.explorer()
    for d in data:
      for _ in xrange(1):
        train = createTbl([d[0] + '/' + d[1][1]], isBin=False)
        test = createTbl([d[0] + '/' + d[1][0]], isBin=False)
        train_df = self.file2pandas(d[0] + '/' + d[1][1])
        test_df = self.file2pandas(d[0] + '/' + d[1][0])
        actual = test_df[test_df.columns[-1]].astype('float32').tolist()
        before = cart(train=train, test=test, smoteit=False)
        newTab = WHAT(
            train=[d[0] + '/' + d[1][1]],
            test=[d[0] + '/' + d[1][0]],
            train_df=train,
            bin=True,
            test_df=test,
            extent=0.5,
            far=False,
            smote=False,
            resample=False,
            infoPrune=0.99,
            method='any',
            Prune=False).main()
        newTab_df = formatData(newTab)
        after = cart(train=train, test=newTab, smoteit=False)
        out.append(cliffs(lst1=before, lst2=after).delta())
      out.insert(0, d[0].strip().split('/')[-1])
      E.append(out)

    #----------- DEGUB ----------------
    set_trace()

    print(r"""\documentclass{article}
    \usepackage{colortbl}
    \usepackage{fullpage}
    \usepackage{booktabs}
    \usepackage{bigstrut}
    \usepackage[table]{xcolor}
    \usepackage{picture}
    \newcommand{\quart}[4]{\begin{picture}(100,6)
    {\color{black}\put(#3,3){\circle*{4}}\put(#1,3){\line(1,0){#2}}}\end{picture}}
    \begin{document}
    """)
    rdivDemo(E, isLatex=True)
    print(r"""
      \end{document}
      """)


def _test():
  fileHandler().main()

if __name__ == '__main__':
  _test()
