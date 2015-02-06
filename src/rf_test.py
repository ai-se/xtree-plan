from os import walk
from pdb import set_trace
from pdb import set_trace

from abcd import _runAbcd
from dtree import *
from sklearn.ensemble import RandomForestClassifier
from where2 import *

import base
from dectree import *
from makeAmodel import makeAModel
from methods1 import *
import pandas as pd


sys.path.append(os.environ['HOME'] + '/git/axe/axe')
sys.path.insert(0, os.getcwd() + '/_imports');
import sk;  # @UnresolvedImport
# whereParams, _ = base._de()
def get_headers(data):
  # set_trace()
  data = pd.read_csv(data[0], header = 0);
  return data.columns[3:].get_values().tolist()

def createDF(data):
  makeaModel = makeAModel()
  _r = []
  for t in data:
    m = makeaModel.csv2py(t)
    _r += m._rows
  m._rows = _r
  prepare(m)
  tree = where2(m, m._rows)
  Rows = []
  for k, _ in leaves(tree):  # for k, _ in leaves(tree):
    for j in k.val:
      tmp = (j.cells)
      tmp.append('Class_' + str(id(k) % 1000))
      j.__dict__.update({'cells': tmp})
      Rows.append(j.cells)
  return pd.DataFrame(Rows, columns = get_headers(data) + ['klass'])

def haupt():
  train, test = explore('./Data')
  clf = RandomForestClassifier(n_estimators = 100, n_jobs = 2)
  """
  Cluster using FASTMAP
  """
  # Training data
  train_DF = createDF(train[1])

  # Testing data
  test_df = createDF(test[1])
  # set_trace()
  features = train_DF.columns[3:-2]
  klass = train_DF[train_DF.columns[-1]];
  clf.fit(train_DF[features], klass)
  preds = clf.predict(test_df[features]).tolist()
  # print preds

#   def isdefective(data):
  label = set(train_DF.columns[-1]);
  _id = list(set(train_DF[train_DF.columns[-1]]))
  dfct = {lbl: str(np.mean(list(train_DF[train_DF['klass'] == lbl]['$<bug'])) >= 0.3) \
             for lbl in _id}
    # print label
  predictions = [dfct[i] for i in preds]
  actuals = [str(not i == 0) for i in test_df[test_df.columns[-2]].tolist()]
  return _runAbcd(train = actuals, test = predictions, verbose = False)
  # set_trace()


def run():
 G = []
 for _ in xrange(1):
  G.append(haupt())
 G.insert(0, 'RF  ')
 # print base.main()+[G]
 sk.rdivDemo(base.main() + [G])
 set_trace()
run()
