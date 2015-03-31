"""

# A Better Where

WHERE2 is a near-linear time top-down clustering alogithm.

WHERE2 updated an older where with new Python tricks.

## Standard Header Stuff

"""
from __future__ import division, print_function

from pdb import set_trace
import  sys
import types

from demos import *
from libWhere import *
from nasa93 import *
from settingsWhere  import *


sys.dont_write_bytecode = True

sys.path.insert(0, '/Users/rkrsn/git/axe/axe/')

"""

## Dimensionality Reduction with Fastmap

Project data in N dimensions down to a single dimension connecting
twp distant points. Divide that data at the median of those projects.

"""
def pairs(lst):
  for j in lst[0:]:
    last = j
    for i in lst[0:]:
      yield last, i

def somepairs(m, data):
  reps = 1; cmax = -10e32;
  for _ in xrange(reps):
    one = any(data);
    two = furthest(m, one, data)
    three = furthest(m, two, data)
    c = dist(m, two, three) + 1e-5
    if c >= cmax:
      cmax = c;
      east, west = two, three
  return west, east

def allpairs(m, data):
  cmax = -10e32
  for one in data:
    for two in [d for d in data if not d==1]: 
      c = dist(m, one, two)+1e-5;
      if c >= cmax:
        cmax = c;
        east, west = one, two
  return west, east

def fastmap(m, data):
  "Divide data into two using distance to two distant items."
  west, east = somepairs(m, data)
  """
  one = any(data)  # 1) pick anything
  west = furthest(m, one, data)  # 2) west is as far as you can go from anything
  east = furthest(m, west, data)  # 3) east is as far as you can go from west
  """
  c = dist(m, west, east) + 1e-5
  # now find everyone's distance
  lst = []
  for one in data:
    a = dist(m, one, west)
    b = dist(m, one, east)
    x = (a * a + c * c - b * b) / (2 * c)  # cosine rule
    y = max(0, a ** 2 - x ** 2) ** 0.5  # not used, here for a demo
    lst += [(x, one)]
  lst = sorted(lst)
  mid = len(lst) // 2
  wests = map(second, lst[:mid])
  easts = map(second, lst[mid:])
  return wests, west, easts, east, c

def gt(x, y): return x > y
def lt(x, y): return x < y

"""
In the above:

+ _m_ is some model that generates candidate
  solutions that we wish to niche.
+ _(west,east)_ are not _the_ most distant points
  (that would require _N*N) distance
  calculations). But they are at least very distant
  to each other.

This code needs some helper functions. _Dist_ uses
the standard Euclidean measure. Note that you tune
what it uses to define the niches (decisions or
objectives) using the _what_ parameter:

"""
def dist(m, i, j,
         what = lambda m: m.decisions):
  "Euclidean distance 0 <= d <= 1 between decisions"
  n = len(i.cells)
  deltas = 0
  for c in what(m):
    n1 = norm(m, c, i.cells[c])
    n2 = norm(m, c, j.cells[c])
    inc = (n1 - n2) ** 2
    deltas += inc
    n += abs(m.w[c])
  return deltas ** 0.5 / n ** 0.5
"""

The _Dist_ function normalizes all the raw values zero to one.

"""
def norm(m, c, val) :
  "Normalizes val in col c within model m 0..1"

  return (atom(val) - atom(m.lo[c])) / (atom(m.hi[c]) - atom(m.lo[c]) + 0.0001)
"""

Now we can define _furthest_:

"""
def furthest(m, i, all,
             init = 0,
             better = gt):
  "find which of all is furthest from 'i'"
  out, d = i, init
  for j in all:
    if i == j: continue
    tmp = dist(m, i, j)
    if better(tmp, d):
      out, d = j, tmp
  return out
"""

And of course, _closest_:

"""
def closest(m, i, all):
  return furthest(m, i, all, init = 10 ** 32, better = lt)
"""

## WHERE2 = Recursive Fastmap


WHERE2 finds everyone's else's distance from the poles
  and divide the data on the mean point of those
  distances.  This all stops if:

+  Any division has _tooFew_ solutions (say,
  less than _sqrt_ of the total number of
  solutions).
+ Something has gone horribly wrong and you are
  recursing _tooDeep_

This code is controlled by the options in [_The_ settings](settingspy).  For
example, if _The.pruning_ is true, we may ignore
some sub-tree (this process is discussed, later on).
Also, if _The.verbose_ is true, the _show_
function prints out a little tree showing the
progress (and to print indents in that tree, we use
the string _The.b4_).  For example, here's WHERE2
dividing 93 examples from NASA93.

    ---| _where |-----------------
    93
    |.. 46
    |.. |.. 23
    |.. |.. |.. 11
    |.. |.. |.. |.. 5.
    |.. |.. |.. |.. 6.
    |.. |.. |.. 12
    |.. |.. |.. |.. 6.
    |.. |.. |.. |.. 6.
    |.. |.. 23
    |.. |.. |.. 11
    |.. |.. |.. |.. 5.
    |.. |.. |.. |.. 6.
    |.. |.. |.. 12
    |.. |.. |.. |.. 6.
    |.. |.. |.. |.. 6.
    |.. 47
    |.. |.. 23
    |.. |.. |.. 11
    |.. |.. |.. |.. 5.
    |.. |.. |.. |.. 6.
    |.. |.. |.. 12
    |.. |.. |.. |.. 6.
    |.. |.. |.. |.. 6.
    |.. |.. 24
    |.. |.. |.. 12
    |.. |.. |.. |.. 6.
    |.. |.. |.. |.. 6.
    |.. |.. |.. 12
    |.. |.. |.. |.. 6.
    |.. |.. |.. |.. 6.


WHERE2 returns clusters, where each cluster contains
multiple solutions.

"""
def where2(m, data, lvl = 0, up = None, verbose = False):
  node = o(val = None, _up = up, _kids = [])
  def tooDeep(): return lvl > The.what.depthMax
  def tooFew() : return len(data) < The.what.minSize
  def show(suffix):
    if verbose:
      print(The.what.b4 * lvl, len(data),
            suffix, ' ; ', id(node) % 1000, sep = '')
  if tooDeep() or tooFew():
    show(".")
    node.val = data
  else:
    show("")
    wests, west, easts, east, c = fastmap(m, data)
    node.update(c = c, east = east, west = west)
    goLeft, goRight = maybePrune(m, lvl, west, east)
    if goLeft:
      node._kids += [where2(m, wests, lvl + 1, node)]
    if goRight:
      node._kids += [where2(m, easts, lvl + 1, node)]
  return node
"""

## An Experimental Extensions

Lately I've been experimenting with a system that
prunes as it divides the data. GALE checks for
domination between the poles and ignores data in
halves with a dominated pole. This means that for
_N_ solutions we only ever have to evaluate
_2*log(N)_ of them- which is useful if each
evaluation takes a long time.

The niches found in this way
contain non-dominated poles; i.e. they are
approximations to the Pareto frontier.
Preliminary results show that this is a useful
approach but you should treat those results with a
grain of salt.

In any case, this code supports that pruning as an
optional extra (and is enabled using the
_slots.pruning_ flag). In summary, this code says if
the scores for the poles are more different that
_slots.wriggle_ and one pole has a better score than
the other, then ignore the other pole.

"""
def maybePrune(m, lvl, west, east):
  "Usually, go left then right, unless dominated."
  goLeft, goRight = True, True  # default
  if  The.prune and lvl >= The.what.depthMin:
    sw = scores(m, west)
    se = scores(m, east)
    if abs(sw - se) > The.wriggle:  # big enough to consider
      if se > sw: goLeft = False  # no left
      if sw > se: goRight = False  # no right
  return goLeft, goRight
"""

Note that I do not allow pruning until we have
descended at least _slots.depthMin_ into the tree.

### Model-specific Stuff

WHERE2 talks to models via the the following model-specific variables:

+ _m.cols_: list of indices in a list
+ _m.names_: a list of names for each column.
+ _m.decisions_: the subset of cols relating to decisions.
+ _m.obectives_: the subset of cols relating to objectives.
+ _m.eval(m,eg)_: function for computing variables from _eg_.
+ _m.lo[c]_ : the lowest value in column _c_.
+ _m.hi[c]_ : the highest value in column _c_.
+ _m.w[c]_: the weight for each column. Usually equal to one.
  If an objective and if we are minimizing  that objective, then the weight is negative.


### Model-general stuff

Using the model-specific stuff, WHERE2 defines some
useful general functions.

"""
def some(m, x) :
  "with variable x of model m, pick one value at random"
  return m.lo[x] + by(m.hi[x] - m.lo[x])

def scores(m, it):
  "Score an individual."
  if not it.scored:
    m.eval(m, it)
    new, w = 0, 0
    for c in m.objectives:
      val = it.cells[c]
      w += abs(m.w[c])
      tmp = norm(m, c, val)
      if m.w[c] < 0:
        tmp = 1 - tmp
      new += (tmp ** 2)
    it.score = (new ** 0.5) / (w ** 0.5 + 1e-4)
    it.scored = True
  return it.score
"""

## Tree Code

Tools for manipulating the tree returned by _where2_.

### Primitive: Walk the nodes

"""
def nodes(tree, seen = None, steps = 0):
  if seen is None: seen = []
  if tree:
   if not id(tree) in seen:
     seen.append(id(tree))
     yield tree, steps
     for kid in tree._kids:
       for sub, steps1 in nodes(kid, seen, steps + 1):
         yield sub, steps1
"""

### Return nodes that are leaves

"""
def leaves(tree, seen = None, steps = 0):
  for node, steps1 in nodes(tree, seen, steps):
    if not node._kids:
      yield node, steps1
"""

### Return nodes nearest to furthest

"""
# walk sideways..
def neighbors(leaf, seen = None, steps = -1):
  """Walk the tree from 'leaf' increasingly
     distant leaves. """
  if seen is None: seen = []
  for down, steps1 in leaves(leaf, seen, steps + 1):
    yield down, steps1
  if leaf:
    for up, steps1 in neighbors(leaf._up, seen, steps + 1):
      yield up, steps1
"""

### Return nodes in Groups, Closest to Furthest


"""
def around(leaf, f = lambda x: x):
  tmp, last = [], None
  for node, dist in neighbors(leaf):
    if dist > 0:
      if dist == last:
        tmp += [f(node)]
      else:
        if tmp:
          yield last, tmp
        tmp = [f(node)]
    last = dist
  if tmp:
    yield last, tmp
"""
## Demo Code

### Code Showing the scores

"""
# @go
def _scores():
  m = nasa93()
  out = []
  for row in m._rows:
    scores(m, row)
    out += [(row.score, [row.cells[c] for c in m.objectives])]
  for s, x in sorted(out):
    print(s, x)
"""

### Code Showing the Distances

"""
# @go
def _distances(m = nasa93):
   m = m()
   seed(The.seed)
   for i in m._rows:
     j = closest(m, i, m._rows)
     k = furthest(m, i, m._rows)
     idec = [i.cells[c] for c in m.decisions]
     jdec = [j.cells[c] for c in m.decisions]
     kdec = [k.cells[c] for c in m.decisions]
     print("\n",
           gs(idec), g(scores(m, i)), "\n",
           gs(jdec), "closest ", g(dist(m, i, j)), "\n",
           gs(kdec), "furthest", g(dist(m, i, k)))
"""

### A Demo for  Where2.

"""


def prepare(m, settings = None):
  "Prepare the 'The' class"
  seed(1)
  global The
  The = settings if settings else defaults().update(verbose = True,
               minSize = len(m._rows) ** 0.5,
               prune = False,
               wriggle = 0.3)
  return The

def _where(m = nasa93):
  m = m()
  seed(1)
  told = N()
  for r in m._rows:
    s = scores(m, r)
    told += s
  global The
  The = defaults().update(verbose = True,
               minSize = len(m._rows) ** 0.5,
               prune = False,
               wriggle = 0.3 * told.sd())
  tree = where2(m, m._rows)
  n = 0
  for node, _ in leaves(tree):
   ID = id(node) % 1000

   print(node.val)

   """
    print(m,' ',end="")
    n += m
    print(id(node) % 1000, ' ',end='')
    for near,dist in neighbors(node):
     print(dist,id(near) % 1000,' ',end='')
    print("")
  print(n)

  filter = lambda z: id(z) % 1000
  for node,_ in leaves(tree):
    print(filter(node),
          [x for x in around(node,filter)])
          """



