import sys, os
sys.path.append(os.environ['HOME'] + '/git/axe/axe')
sys.path.insert(0, os.getcwd() + '/_imports');
from demos import *
import sk;  # @UnresolvedImport
from dectree import *
from settings import *
from settingsWhere  import *
from pdb import set_trace
from abcd import _Abcd
from Prediction import rforest, CART, Bugs, where2prd
from methods1 import createTbl
from random import uniform as rand, randint as randi, choice as any
tree = treeings()

# set_trace()

def say(l):
  sys.stdout.write(str(l))

def settings(**d): return o(
  name = "Differention Evolution",
  what = "DE tuner. Tune the planner parameters.",
  author = "Rahul Krishna",
  adaptation = "https://github.com/ai-se/Rahul/blob/master/DEADANT/deadant.py",
  copyleft = "(c) 2014, MIT license, http://goo.gl/3UYBp",
  seed = 1,
  np = 10,
  k = 100,
  tiny = 0.01,
  de = o(np = 5,
       iter = 5,
       epsilon = 1.01,
       N = 5,
       f = 0.5,
       cf = 0.4,
       lives = 20)
  ).update(**d)

The = settings()

class diffEvol(object):
  """
  Differential Evolution
  """

  def __init__(self, model, data):
    self.frontier = []
    self.model = model(data)
    self.xbest = []

  def new(self):
    # Creates a new random instance
    return [rand(d[0], d[1]) for d in self.model.indep()]

  def initFront(self, N):
    # Initialize frontier
    for _ in xrange(N):
      self.frontier.append(self.new())

  def extrapolate(self, xbest, l1, l2, l3, l4):
    try:
      return [max(d[0],
                min(d[1], y + The.de.f * (z + a - b - c))) for y, z, a,
                b, c, d in zip(xbest, l1, l2, l3, l4, self.model.indep())]
    except TypeError:
      set_trace()

  def one234(self, one, pop, f = lambda x:id(x)):
    def oneOther():
      x = any(pop)
      while f(x) in seen:
        x = any(pop)
      seen.append(f(x))
      return x
    seen = [ f(one) ]
    return oneOther(), oneOther(), oneOther(), oneOther()

 # def top234(self, one, pop):


  def dominates(self, one, two):
#     set_trace()
    return self.model.depen(one) > self.model.depen(two)
  
  def sortbyscore(self):
 #    front = []
 #    for f in self.frontier:
 #      sc = self.model.depen(f)
 #      f.append(sc)
 #      front.append(f)
    return sorted(self.frontier, key = lambda F: self.model.depen(F), reverse = True)
  def DE(self):
    self.initFront(The.de.N)
    lives = The.de.lives
    while lives > 0:
      better = False
      self.xbest =  self.sortbyscore()[0]
      for pos, val in enumerate(self.frontier):
        lives -= 1
        l1, l2, l3, l4 = self.one234(val, self.frontier)
        new = self.extrapolate(self.xbest, l1, l2, l3, l4)
        if  self.dominates(new, val):
          self.frontier.pop(pos)
          self.frontier.insert(pos, new)
          better = True
          lives += 1
          if self.model.depen(new) > self.model.depen(self.xbest):
            self.xbest = new
          # print(self.model.depen(new))
        elif self.dominates(val, new):
          better = False
          if self.model.depen(val) > self.model.depen(self.xbest):
            self.xbest = val
          # print(self.model.depen(new))
        else:
          self.frontier.append(new)
          if self.model.depen(new) > self.model.depen(self.xbest):
            self.xbest = new
          better = True
          lives += 1
#      print(self.model.depen(self.xbest))
    return self.xbest


class tuneRF(object):
  # Tune RF
  def __init__(self, data):
    self.data = data
    self.train = createTbl(data[:-1])
    self.test = createTbl(data[-1])
#   set_trace()
  def depen(self, rows):
    mod = rforest(self.train, self.test
                , tunings = rows  # n_est, max_feat, mss, msl
                , smoteit = True)
    g = _Abcd(before = Bugs(self.test), after = mod, show = False)[-1]
    return g

  def indep(self):
    return [(10, 1e3)  # n_estimators
          , (1, 100)  # max_features
          , (1, 10)  # min_samples_leaf
          , (2, 10)  # min_samples_split
          ]

class tuneWhere2(object):
  # Tune where
  def __init__(self, data):
    self.train = data[:-1]
    self.test = data[-1]
    self.tree = treeings()
    self.where = None
  def depen(self, row):
    # My where2pred() takes data in string format. Ex: '../Data/ant/ant-1.6.csv'
    self.where = defaults().update(minSize = row[4]
                                 , depthMin = int(row[5])
                                 , depthMax = int(row[6])
                                 , prune = row[7]>0.5)
    self.tree.infoPrune = row[1]
    self.tree.m = int(row[2])
    self.tree.n = int(row[3])
    self.tree.prune = row[8]>0.5
    actual = Bugs(createTbl([self.test], isBin = True))
    preds = where2prd(self.train, [self.test], tunings = [self.where, self.tree], thresh = row[0])
    return _Abcd(before = actual, after = preds, show = False)[-1]

  def indep(self):
    return [(0, 1)          # Threshold
          , (0, 1)          # InfoPrune
          , (1, 10)         # m          
          , (1, 10)         # n          
          , (0, 1)          # Min Size
          , (1, 6)          # Depth Min
          , (1, 20)         # Depth Max
          , (0, 1)          # Where Prune?
          , (0, 1)]         # Tree Prune?

class tuneCART(object):
  # Tune CART
  def __init__(self, data):
    self.data = data
    self.train = createTbl(data[:-1], isBin = True)
    self.test = createTbl([data[-1]], isBin = True)

  def depen(self, rows):
    mod = CART(self.train, self.test
               , tunings = rows
               , smoteit = True)
    g = _Abcd(before = Bugs(self.test), after = mod, show = False)[-1]
    return g

  def indep(self):
    return [(1, 50)  # max_depth
          , (2, 20)  # min_samples_split
          , (1, 20)  # min_samples_leaf
          , (1, 100)  # max features
          , (2, 1e3)]  # max_leaf_nodes


def _test(data):
  m = tuneRF(data)
  vals = [(m.any()) for _ in range(10)]
  vals1 = [m.score(v) for v in vals]
  print(vals, vals1)

def _de(model, data):
  "DE"
  DE = diffEvol(model, data);
#   set_trace()
  res = DE.DE()
#  print(model.depen(res))
  return res

def tuner(model, data):
  if model == rforest:
    return _de(tuneRF, data)
  elif model == CART:
    return _de(tuneCART, data)

if __name__ == '__main__':
  from timeit import time
  data = explore(dir = '../Data/')[0][0]  # Only training data to tune.
#  set_trace()
  for m in [tuneCART]:
    t = time.time()
    mdl = m(data)
#   _test(data)
    tunings = _de(m, data)
    print tunings
    print mdl.depen(tunings)
    print time.time() - t
#   print _de()
#  print main()
#  import sk; xtile = sk.xtile
#  print xtile(G)

 # main(dir = 'Data/')
