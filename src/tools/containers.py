from numpy import log2
__author__ = 'rkrsn'

class Thing(object):
  id = -1
  def __init__(i,**fields) :
    i.override(fields)
    i.newId()
  def newId(i):
    i._id = Thing.id = Thing.id + 1
  def also(i,**d)  : return i.override(d)
  def override(i,d): i.__dict__.update(d); return i
  def __hash__(i) : return i._id

def math():
  return Thing(
      seed=1,
      inf=10 ** 32,
      ninf=-1 * 10 ** 32,
      teeny=10 ** -32,
      bootstraps=500,
      a12=Thing(
          small=[.6, .68][0],
          reverse=False),
      brink=Thing(
          hedges=[.39, 1.0][0],
          cohen=[.3, .5][0],
          conf=[.95, .99][0]))


def sample(**d):
  return Thing(
      keep=256,
      bins=5,
      tiny=0.1,
      enough=4).override(d)


class Sym(Thing):
  def __init__(i,inits=[],w=1):
    i.newId()
    i.selected=False
    i.w=w
    i.n,i.counts,i._also = 0,{},None
    for symbol in inits: i + symbol
  def __add__(i,symbol): i.inc(symbol,  1)
  def __sub__(i,symbol): i.inc(symbol, -1)
  def inc(i,x,n=1):
    i._also = None
    i.n += n
    i.counts[x] = i.counts.get(x,0) + n
  def norm(i,x): return x
  def dist(i,x,y): return 0 if x==y else 1
  def far(i,x): return '~!@#$%^&*'
  def k(i)   : return len(i.counts.keys())
  def centroid(i): return i.mode()
  def most(i): return i.also().most
  def mode(i): return i.also().mode
  def ent(i) : return i.also().e
  def also(i):
    if not i._also:
      e,most,mode = 0,0,None
      for symbol in i.counts:
        if i.counts[symbol] > most:
          most,mode = i.counts[symbol],symbol
        p = i.counts[symbol]/i.n
        if p:
          e -= p*log2(p)
        i._also = Thing(most=most,mode=mode,e=e)
        #print "also", i._also.e
    return i._also


class Num(Thing):
  "An accumulator for numbers"
  def __init__(i,init=[], opts=sample,w=1):
    i.newId()
    i.selected=False
    i.opts = opts
    i.w=w
    i.zero()
    for x in init: i + x
    for x in init: x=i.norm(i,x)
  def zero(i):
    i.lo,i.hi = 10**32,-10**32
    i.n = i.mu = i.m2 = 0
  def __lt__(i,j):
    return i.mu < j.mu
  def n(i): return i.some.n
  def sd(i) :
    if i.n < 2: return i.mu
    else:
      return (max(0,i.m2)/(i.n - 1))**0.5
  def centroid(i): return i.median()
  def median(i): return i.some.median()
  def iqr(i): return i.some.iqr()
  def breaks(i): return i.some.breaks()
  def all(i)   : return i.some.all()
  def __add__(i,x):
    if x > i.hi: i.hi = x
    if x < i.lo: i.lo = x
    i.n  += 1
    delta = x - i.mu
    i.mu += delta/(1.0*i.n)
    i.m2 += delta*(x - i.mu)
  def __sub__(i,x):
    i.some = None
    if i.n < 2: return i.zero()
    i.n  -= 1
    delta = x - i.mu
    i.mu -= delta/(1.0*i.n)
    i.m2 -= delta*(x - i.mu)
  def dist(i,x,y,normalize=True):
    if normalize:
      x,y=i.norm(x),i.norm(y)
    return (x-y)**2
  def norm(i,x):
    return (x - i.lo)/ (i.hi - i.lo + 0.00001)
  def far(i,x):
    return i.lo if x > (i.hi - i.lo)/2 else i.hi
