#! /Users/rkrsn/miniconda/bin/python
from __future__ import division
from __future__ import print_function

from os import environ
from os import getcwd
from os import system
from os import walk, path
from pdb import set_trace
from random import randint as randi
from random import sample
from random import uniform as rand
from random import shuffle
from subprocess import PIPE
from subprocess import call
import sys

# Update PYTHONPATH
HOME = path.expanduser('~')
axe = HOME + '/git/axe/axe/'  # AXE
pystat = HOME + '/git/pystat/'  # PySTAT
cwd = getcwd()  # Current Directory
WHAT = HOME + '/git/Transfer-Learning/old/SOURCE'
sys.path.extend([axe, pystat, cwd, WHAT])


from numpy import median
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
import pandas

from Prediction import CART as cart
from Prediction import formatData
from WHAT import treatments as WHAT
from demos import cmd
from methods1 import *
from sk import rdivDemo
from sk import scottknott
from smote import SMOTE
from table import clone


class changes():

  def __init__(self):
    self.log = {}

  def save(self, name=None, old=None, new=None):
    if old != new:
      self.log.update({name: (old, new)})


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

  def __init__(self, dir='./Data/Seigmund/'):
    self.dir = dir
    self.changes = []

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
      with open("./Data/Seigmund/tmp/" + file[:-7] + '/Train.csv', 'w+') as fwrite:
        writer = csv.writer(fwrite, delimiter=',')
        train = sample(body, int(ttr * len(body)))
        writer.writerow(header)
        for b in train:
          writer.writerow(b)

      with open("./Data/Seigmund/tmp/" + file[:-7] + '/Test.csv', 'w+') as fwrite:
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
    for (dirpath, dirnames, filenames) in walk(cwd + '/Data/Seigmund/tmp/%s/' % (name)):
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

  def planner(self, train, test, fSel, ext, _prune,
              _info, name, method='best', justDeltas=False):
    train_df = formatData(train)
    test_df = formatData(test)
    actual = test_df[
        test_df.columns[-2]].astype('float32').tolist()
    before = predictor(train=train_df, test=test_df).rforest()
#           set_trace()
    newTab = WHAT(
        name=name,
        train=None,
        test=None,
        train_df=train,
        bin=True,
        test_df=test,
        extent=ext,
        fSelect=fSel,
        far=False,
        infoPrune=_info,
        method=method,
        Prune=_prune).main(justDeltas=justDeltas)
#     newTab_df = formatData(newTab)
    after = predictor(train=train_df, test=test_df).rforest()

    return actual, before, after, newTab

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

  def kFoldCrossVal(self, data, fSel, ext, _prune, _info, method, k=5):
    acc, md, auc = [], [], []
    bef, aft = [], []
    chunks = lambda l, n: [l[i:i + n] for i in range(0, len(l), int(n))]
    from random import shuffle, sample
    rows = data._rows
    shuffle(rows)
    sqe = chunks(rows, int(len(rows) / k))
    if len(sqe) > k:
      sqe = sqe[:-2] + [sqe[-2] + sqe[-1]]
    for indx in xrange(k):
      try:
        testRows = sqe.pop(indx)
      except:
        set_trace()
      trainRows = self.flatten([s for s in sqe if not s == testRows])
      train, test = clone(data, rows=[
          i.cells for i in trainRows]), clone(data, rows=[
              i.cells for i in testRows])

      train_df = formatData(train)
      test_df = formatData(test)
      actual = test_df[
          test_df.columns[-2]].astype('float32').tolist()
      before = predictor(train=train_df, test=test_df).rforest()
      _, __, after = self.planner(
          train, test, fSel, ext, _prune, _info, method)
      bef.extend(before)
      aft.extend(after)
      md.append((median(before) - median(after)) * 100 / median(before))
      auc.append((sum(before) - sum(after)) * 100 / sum(before))
      acc.extend(
          [(1 - abs(b - a) / a) * 100 for b, a in zip(before, actual)])
      sqe.insert(k, testRows)
    return acc, auc, md, bef, aft

  def crossval(self, name, k=5, fSel=True,
               ext=0.5, _prune=False, _info=0.25, method='best'):
    before, after = [], []
    cv_acc = []
    cv_md = []
    cv_auc = []
    for _ in xrange(k):
      proj = self.explorer2(name)
      data = createTbl(proj, isBin=False)
      a, b, c, bef, aft = self.kFoldCrossVal(
          data, fSel, ext, _prune, _info, k=k, method=method)
      cv_acc.extend(a)
      cv_auc.extend(b)
      cv_md.extend(c)
      before.extend(bef)
      after.extend(aft)
    return cv_acc, cv_auc, cv_md, before, after

  def main(self, name='Apache', reps=20, fSel=True,
           ext=0.5, _prune=False, _info=0.25, justDeltas=False):
    effectSize = []
    Accuracy = []
    out_auc = []
    out_md = []
    out_acc = []
    data = self.explorer(name)
    for d in data:
      #       print("\\subsection{%s}\n \\begin{figure}\n \\centering" %
      #             (d[0].strip().split('/')[-1]))
      if name == d[0].strip().split('/')[-2]:
        #           set_trace()
        train = createTbl([d[0] + '/' + d[1][1]], isBin=False)
        test = createTbl([d[0] + '/' + d[1][0]], isBin=False)
        actual, before, after, newTab = self.planner(
            train, test, fSel, ext, _prune, _info, name=name, method='best', justDeltas=justDeltas)
        out_auc.append(sum(after) / sum(before))
        out_md.append(median(after) / median(before))
        out_acc.extend(
            [(1 - abs(b - a) / a) * 100 for b, a in zip(before, actual)])
        return newTab
  #----------- DEGUB ----------------
#     set_trace()

  def mainraw(self, name='Apache', reps=10, fSel=True,
              ext=0.5, _prune=False, _info=0.25, method='best'):
    data = self.explorer(name)
    before, after = [], []
    for _ in xrange(reps):
      for d in data:
        if name == d[0].strip().split('/')[-1]:
          train = createTbl([d[0] + '/' + d[1][1]], isBin=False)
          test = createTbl([d[0] + '/' + d[1][0]], isBin=False)
          train_df = formatData(train)
          test_df = formatData(test)
          actual = test_df[
              test_df.columns[-2]].astype('float32').tolist()
          before.append(predictor(train=train_df, test=test_df).rforest())
  #           set_trace()
          newTab = WHAT(
              train=[d[0] + '/' + d[1][1]],
              test=[d[0] + '/' + d[1][0]],
              train_df=train,
              bin=True,
              test_df=test,
              extent=ext,
              fSelect=fSel,
              far=False,
              infoPrune=_info,
              method=method,
              Prune=_prune).main()
          newTab_df = formatData(newTab)
          after.append(predictor(train=train_df, test=newTab_df).rforest())
    return before, after


def preamble1():
  print(r"""\documentclass{article}
    \usepackage{fullpage}
    \usepackage{booktabs}
    \usepackage{bigstrut}
    \usepackage[table]{xcolor}
    \usepackage{picture}
    \newcommand{\quart}[4]{\begin{picture}(100,6)
    {\color{black}\put(#3,3){\circle*{4}}\put(#1,3){\line(1,0){#2}}}\end{picture}}
    \begin{document}
    """)


def postabmle():
  print(r"""
  \end{document}
  """)


def overlayCurve(
        # w, x, y, z, base, fname=None, ext=None, textbox=False, string=None):
        x, y, base, fname=None, ext=None, textbox=False, string=None):
  from numpy import linspace, cumsum
  import matplotlib.pyplot as plt
  import matplotlib.lines as mlines
  from matplotlib.backends.backend_pdf import PdfPages
  pp = PdfPages(fname + '.pdf')

  fname = 'Untitled' if not fname else fname
  ext = '.jpg' if not ext else ext
  fig = plt.figure()
  ax = fig.add_axes([0.1, 0.1, 0.6, 0.75])
#   wlim = linspace(1, len(w[0]), len(w[0]))
  xlim = linspace(1, len(x[0]), len(x[0]))
  ylim = linspace(1, len(y[0]), len(y[0]))
  blim = linspace(1, len(base[0]), len(base[0]))
  # plt.subplot(221)
#   ax.plot(xlim, sorted(w[0]), 'r')
  ax.plot(ylim, sorted(y[0]), 'b')
  ax.plot(xlim, sorted(x[0]), 'r')
#   ax.plot(ylim, sorted(z[0]), 'b')
  ax.plot(blim, sorted(base[0]), 'k')

  # add a 'best fit' line
  ax.set_xlabel('Test Cases', size=18)
  ax.set_ylabel('Performance Scores (s)', size=18)
  plt.title(fname)
  # plt.title(r'Histogram (Median Bugs in each class)')

  # Tweak spacing to prevent clipping of ylabel
#     plt.subplots_adjust(left=0.15)
  "Legend"
  black_line = mlines.Line2D([], [], color='k', marker='*',
                             markersize=0, label='Baseline')
  blue_line = mlines.Line2D([], [], color='b', marker='*',
                            markersize=0, label='Best')
#   meg_line = mlines.Line2D([], [], color='m', marker='*',
#                            markersize=0, label=y[1])
#   green_line = mlines.Line2D([], [], color='g', marker='*',
#                              markersize=0, label=x[1])
  red_line = mlines.Line2D([], [], color='r', marker='*',
                           markersize=0, label='Worst')
  plt.legend(
      bbox_to_anchor=(
          1.05,
          1),
      loc=2,
      borderaxespad=0.,
      handles=[black_line,
               red_line,
               blue_line
               #                meg_line,
               #                green_line,
               ])
  if textbox:
    "Textbox"
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.05, 0.95, string, fontsize=14,
            verticalalignment='top', bbox=props)
  pp.savefig()
  pp.close()
#   plt.savefig('./_fig/' + fname + ext)
#   plt.close()


def test(name='Apache', doWhat='AUC', justDeltas=False):
  return fileHandler().main(name, reps=1,
                            ext=0.75,
                            _prune=True,
                            _info=0.25,
                            fSel=True, justDeltas=justDeltas)


def _doCrossVal():
  cv_acc = []
  cv_auc = []
  cv_md = []

  for name in ['Apache', 'SQL', 'BDBC', 'BDBJ', 'X264', 'LLVM']:
    a, _, __ = fileHandler().crossval(name, k=5)
    cv_acc.append(a)
#     cv_auc.append(b)
#     cv_md.append(c)
#   print(r"""\documentclass{article}
#   \usepackage{colortbl}
#   \usepackage{fullpage}
#   \usepackage[table]{xcolor}
#   \usepackage{picture}
#   \newcommand{\quart}[4]{\begin{picture}(100,6)
# {\color{black}\put(#3,3){\circle*{4}}\put(#1,3){\line(1,0){#2}}}\end{picture}}
#   \begin{document}
#   """)
#   print(r"\subsubsection*{Accuracy}")
  rdivDemo(cv_acc, isLatex=False)
#   print(r"\end{tabular}")
#   print(r"\subsubsection*{Area Under Curve}")
#   rdivDemo(cv_auc, isLatex=True)
#   print(r"\end{tabular}")
#   print(r"\subsubsection*{Median Spread}")
#   rdivDemo(cv_md, isLatex=True)
#   print(r'''\end{tabular}
#   \end{document}''')


def _testPlot(name='Apache'):
  Accuracy = []
#  fileHandler().preamble()
  figname = fileHandler().figname
  for name in ['Apache', 'SQL', 'BDBC', 'BDBJ', 'X264', 'LLVM']:
    #     print("\\subsection{%s}\n \\begin{figure}\n \\centering" % (name))

    bfr, base = fileHandler().mainraw(name, reps=1,
                                      ext=0,
                                      _prune=True,
                                      _info=0.5,
                                      fSel=False)
    bfr, worse = fileHandler().mainraw(name, reps=1,
                                       ext=0.75,
                                       _prune=True,
                                       _info=0.5,
                                       method='mean',
                                       fSel=True)
    bfr, best = fileHandler().mainraw(name, reps=1,
                                      ext=0.75,
                                      _prune=True,
                                      _info=0.5,
                                      method='best',
                                      fSel=True)
#     set_trace()
  #    print("Baseline,mean,median,any,best")
  #   for b, me, md, an, be in zip(baseline[0], best1[0], best2[0], best3[0], best4[0]):
  #   print("%0.2f,%0.2f,%0.2f,%0.2f,%0.2f" % (b, me, md, an, be))
    overlayCurve([worse[0], 'Worst'],
                 [best[0], 'Best'],
                 [base[0], 'Baseline'],
                 fname=name,
                 ext='.pdf',
                 textbox=True,
                 string=None)
#     print(
#         "\\subfloat[][]{\\includegraphics[width=0.5\\linewidth]{../_fig/%s}\\label{}}" %
#         (name + '.jpg'))
#     print(r"\end{figure}")
#   print(r"\end{document}")

if __name__ == '__main__':
  print(test(name='BDBJ', doWhat='AUC'))
