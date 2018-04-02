from __future__ import division

import warnings
from collections import Counter
from pdb import set_trace
from random import choice, uniform as rand
from time import time

import pandas as pd
from scipy.spatial.distance import euclidean
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import BallTree
from sklearn.svm import SVC, SVR

from misc import *

warnings.filterwarnings('ignore')

def SMOTE(data=None, atleast=50, atmost=100, a=None,b=None, k=5, resample=False):
  "Synthetic Minority Oversampling Technique"
  # set_trace()
  def knn(a,b):
    "k nearest neighbors"
    b=np.array([bb[:-1] for bb in b])
    tree = BallTree(b)
    __, indx = tree.query(a[:-1], k=6)

    return [b[i] for i in indx]
    # set_trace()
    # return sorted(b, key=lambda F: euclidean(a[:-1], F[:-1]))

  def kfn(me,my_lot,others):
    "k farthest neighbors"
    my_closest = None
    return sorted(b, key=lambda F: euclidean(a[:-1], F[:-1]))

  def extrapolate(one, two):
    # t=time()
    new = len(one)*[None]
    new[:-1] = [a + rand(0,1) * (b-a) for
                     a, b in zip(one[:-1], two[:-1])]
    new[-1] = int(one[-1])
    return new

  def populate(data, atleast):
    t=time()
    newData = [dd.tolist() for dd in data]
    if atleast-len(newData)<0:
      try:
        return [choice(newData) for _ in xrange(atleast)]
      except:
        set_trace()
    else:
      for _ in xrange(atleast-len(newData)):
        one = choice(data)
        neigh = knn(one, data)[1:k + 1]
        try:
          two = choice(neigh)
        except IndexError:
          two = one
        newData.append(extrapolate(one, two))
      return newData

  def populate2(data1, data2):
    newData = []
    for _ in xrange(atleast):
      for one in data1:
        neigh = kfn(one, data)[1:k + 1]
        try:
          two = choice(neigh)
        except IndexError:
          two = one
        newData.append(extrapolate(one, two))
    return [choice(newData) for _ in xrange(atleast)]

  def depopulate(data):
    # if resample:
    #   newer = []
    #   for _ in xrange(atmost):
    #     orig = choice(data.dat)
    #     newer.append(extrapolate(orig, knn(orig, data.dat)[1]))
    #   return newer
    # else:
      return [choice(data).tolist() for _ in xrange(atmost)]

  newCells = []
  # rseed(1)
  klass = lambda df: df[df.columns[-1]]
  count = Counter(klass(data))
  # set_trace()
  atleast=50# if a==None else int(a*max([count[k] for k in count.keys()]))
  atmost=100# if b==None else int(b*max([count[k] for k in count.keys()]))
  major, minor = count.keys()
  # set_trace()
  for u in count.keys():
    if u==minor:
      newCells.extend(populate([r for r in data.as_matrix() if r[-1] == u], atleast=atleast))
    if u==major:
      newCells.extend(depopulate([r for r in data.as_matrix() if r[-1] == u]))
    else:
      newCells.extend([r.tolist() for r in data.as_matrix() if r[-1] == u])
  # set_trace()
  return pd.DataFrame(newCells, columns=data.columns)

def _smote():
  "Test SMOTE"
  dir = '../data.dat/Jureczko/camel/camel-1.6.csv'
  Tbl = csv2DF([dir], as_mtx=False)
  newTbl = SMOTE(Tbl)
  print('Before SMOTE: ', Counter(Tbl[Tbl.columns[-1]]))
  print('After  SMOTE: ', Counter(newTbl[newTbl.columns[-1]]))
  # ---- ::DEBUG:: -----
  set_trace()

def rforest(train, test, tunings=None, smoteit=True, bin=True, smoteTune=True,regress=False):
  "RF "
  if tunings and smoteTune==False:
      a=b=None
  elif tunings and smoteTune==True:
    a=tunings[-2]
    b=tunings[-1]

  if not isinstance(train, pd.core.frame.DataFrame):
    train = csv2DF(train, as_mtx=False, toBin=bin)

  if not isinstance(test, pd.core.frame.DataFrame):
    test = csv2DF(test, as_mtx=False, toBin=True)

  if smoteit:
    if not tunings:
      train = SMOTE(train, resample=True)
    else:
      train = SMOTE(train, a, b, resample=True)
    # except: set_trace()

  if not tunings:
    if regress:
      clf = RandomForestRegressor(n_estimators=100, random_state=1, warm_start=True,n_jobs=-1)
    else:
      clf = RandomForestClassifier(n_estimators=100, random_state=1, warm_start=True,n_jobs=-1)
  else:
    if regress:
      clf = RandomForestRegressor(n_estimators=int(tunings[0]),
                                   max_features=tunings[1] / 100,
                                   min_samples_leaf=int(tunings[2]),
                                   min_samples_split=int(tunings[3]),
                                   warm_start=True,n_jobs=-1)
    else:
      clf = RandomForestClassifier(n_estimators=int(tunings[0]),
                                   max_features=tunings[1] / 100,
                                   min_samples_leaf=int(tunings[2]),
                                   min_samples_split=int(tunings[3]),
                                   warm_start=True,n_jobs=-1)
  features = train.columns[:-1]
  klass = train[train.columns[-1]]
  clf.fit(train[features], klass)
  actual = test[test.columns[-1]].as_matrix()
  try: preds = clf.predict(test[test.columns[:-1]])
  except: set_trace()
  return actual, preds

def SVM(train, test, tunings=None, smoteit=True, bin=True, regress=False):
  "SVM "
  if not isinstance(train, pd.core.frame.DataFrame):
    train = csv2DF(train, as_mtx=False, toBin=bin)

  if not isinstance(test, pd.core.frame.DataFrame):
    test = csv2DF(test, as_mtx=False, toBin=True)

  if smoteit:
    train = SMOTE(train, resample=True)
    # except: set_trace()
  if not tunings:
    if regress:
      clf = SVR()
    else:
      clf = SVC()
  else:
    if regress:
      clf = SVR()
    else:
      clf = SVC()

  features = train.columns[:-1]
  klass = train[train.columns[-1]]
  # set_trace()
  clf.fit(train[features], klass)
  actual = test[test.columns[-1]].as_matrix()
  try: preds = clf.predict(test[test.columns[:-1]])
  except: set_trace()
  return actual, preds



def _RF():
  dir = '../data.dat/Jureczko/'
  train, test = explore(dir)
  print('Dataset, Expt(F-Score)')
  for tr,te in zip(train, test):
    say(tr[0].split('/')[-1][:-8])
    actual, predicted = rforest(tr, te)
    abcd = ABCD(before=actual, after=predicted)
    F = np.array([k.stats()[-2] for k in abcd()])
    tC = Counter(actual)
    FreqClass=[tC[kk]/len(actual) for kk in list(set(actual))]
    ExptF = np.sum(F*FreqClass)
    say(', %0.2f\n' % (ExptF))
  # ---- ::DEBUG:: -----
  set_trace()

if __name__ == '__main__':
  _RF()
