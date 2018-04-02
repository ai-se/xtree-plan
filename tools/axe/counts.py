from __future__ import division
import sys
from lib import *
from settings import *
sys.dont_write_bytecode = True
from numpy import log2
# XXXX what is k and why does Sym need it?

def test(f=None,cache=[]):
  if f: 
    cache += [f]
    return f
  ok = no = 0
  for t in cache: 
    print '#',t.func_name ,t.__doc__ or ''
    prefix, n, found = None, 0, t() or []
    while found:
      this, that = found.pop(0), found.pop(0)
      if this == that:
        ok, n, prefix = ok+1, n+1,'CORRECT:'
      else: 
        no, n, prefix = no+1, n+1,'WRONG  :'
      print prefix,t.func_name,'test',n
  if ok+no:
    print '\n# Final score: %s/%s = %s%% CORRECT' \
        % (ok,(ok+no),int(100*ok/(ok+no)))

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
 # def __eq__(i,j)  : return i._id == j._id
  #def __neq__(i,j) : return i._id != j._id


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

@test
def symed():
  "Counting symbols"
  s=Sym(list('first kick I took was when I hit'))
  return [ ' '   , s.mode()
         ,  7    , s.most()
         , 3.628 , round(s.ent(),3) ]

class Sample(Thing):
  "Keep a random sample of stuff seen so far."
  def __init__(i,inits=[],opts=The.sample):
    i._cache,i.n,i.opts,i._also = [],0,opts,None
    for number in inits: i + number
  def __add__(i,x):
    i.n += 1
    if len(i._cache) < i.opts.keep: # if not full
      i._also = None
      i._cache += [x]               # then add
    else: # otherwise, maybe replace an old item
      if random.random() <= i.opts.keep/i.n:
        i._also=None
        i._cache[int(random.random()*i.opts.keep)] = x
  def all(i)    : return i._cache
  def median(i) : return i.also().median
  def iqr(i)   : return i.also().iqr
  def breaks(i) : return i.also().breaks
  def also(i):
    if not i._also:
      lst  = i._cache
      n    = len(lst)
      lst  = sorted(lst)
      p= q = max(0, int(n*0.5) - 1)
      r    = int(n*(0.5 + i.opts.tiny))
      dull = lst[r] - lst[p]
      if not oddp(n) : q = p + 1
      i._also = Thing(
        median = (lst[p] + lst[q])*0.5,
        iqr    = lst[int(n*.75)] - lst[int(n*.5)],
        breaks = chops(lst, opts=i.opts,
                        sorted=True, dull=dull))
    return i._also
 
def chops(lst,sorted=False,dull=0,opts=The.sample):
  def chop(bins, before, i):
    rest = len(lst) - i
    if rest < opts.enough:
      return []
    j   = int(i + rest/bins)
    while j < len(lst) and lst[j] <= before+dull:
      j += 1
    if j >= len(lst):
      return []
    now = lst[j]
    return [now] + chop(bins - 1, now,j)
  lst = lst if sorted else sorted(lst)
  now = lst[0]
  return [now] + chop(opts.bins, now,0)

@test
def sampled():
  "Sampling up to 256 items in a distribution."
  seed()
  s0= Sample([1,1,2,2,3]*100,
             sampleings(bins=2))
  s1= Sample([1,1,1,2]*20)
  s2= Sample([rand()**2 for _ in range(1000)],
             sampleings(bins=5))
  return [ [1,2],  s0.breaks()
         , [1,2],  s1.breaks() 
         , [0, 0.09, 0.24, 0.41, 0.71],  
           gs2(s2.breaks())]

class Num(Thing):
  "An accumulator for numbers"
  def __init__(i,init=[], opts=The.sample,w=1):
    i.newId()
    i.selected=False
    i.opts = opts
    i.w=w
    i.zero()
    for x in init: i + x
    for x in init: x=i.norm(i,x)
  def zero(i):
    i.lo,i.hi = 10**32,-10**32
    i.some = Sample([],i.opts)
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
    if i.some: i.some + x
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
  def t(i,j):
    signal = abs(i.mu - j.mu)
    noise  = (i.sd()**2/i.n + j.sd()**2/j.n)**0.5
    return signal / noise
  def cohen(i,j,small=The.math.brink.cohen):
    v1 = i.sd()**2 
    v2 = j.sd()**2
    a  = (i.n - 1)*v1
    b  = (j.n - 1)*v2
    c  = i.n + j.n - 2
    s  = ((a+b)/c)**0.5
    d  = abs(i.mu - j.mu)
    return d/s < small
  def hedges(i,j,small=The.math.brink.hedges):
    "Hedges effect size test."
    num   = (i.n - 1)*i.sd()**2 + (j.n - 1)*j.sd()**2
    denom = (i.n - 1) + (j.n - 1)
    sp    = ( num / denom )**0.5
    delta = abs(i.mu - j.mu) / sp  
    c     = 1 - 3.0 / (4*(i.n + j.n - 2) - 1)
    return delta * c < small
  def bootstrap(i,j,conf = The.math.brink.conf,
                b    = The.math.bootstraps):
    return bootstrap(i.all(), j.all(),conf=conf,b=b)
  def a12(i,j,small=The.math.a12.small,
              reverse=The.math.a12.reverse):
    return a12(i.all(),j.all(),
              reverse=reverse) < small

def ttest(i,j,conf=The.math.brink.conf,
          threshold={.95:((  1, 12.70 ),( 3, 3.182),
                          (  5,  2.571),(10, 2.228),
                          ( 20,  2.086),(80, 1.99 ),
                          (320,  1.97 )),
                     .99:((  1, 63.657),( 3, 5.841),
                           (  5,  4.032),(10, 3.169),
                           ( 20,  2.845),(80, 2.64 ),
                           (320,  2.58 ))}):
  def interpolate(x, points):
    lo, hi = points[0], points[-1]
    x1, y1 = lo[0], lo[1]
    for x2,y2 in points[1:]:
      if x1 <= x <= x2:
        deltay = y2 - y1
        deltax = (x- x1)*1.0/(x2- x1)
        return y1 + deltay * deltax
      x1,y1 = x2,y2
    return hi[1]
  def ttest1(n):
    return interpolate(n,threshold[conf])
  return ttest1(i.n + j.n - 2,conf) < i.t(j)

@test
def numed():
  def push(x,n=0.2):
    return x*(1 + n*rand())
  n1=Num(x    for x in range(30))
  n2=Num(30+x for x in range(30))
  lst1 = [x   for x in range(30)]
  n3, n4, n5 = Num(lst1), Num(), Num()
  for x in lst1: n4 + x; n5 + x
  for x in lst1: n5 - x
  n6 = Num(lst1)
  n7 = Num(push(x,0) for x in lst1)
  n8 = Num(push(x,0.1) for x in lst1)
  n9 = Num(push(x,1) for x in lst1)
  return [14.5, n1.mu
         ,8.80, g2(n1.sd())
         ,14.5, n1.median()
         ,30,   n2.lo
         ,59,   n2.hi
         ,True, n3.sd() == n4.sd()
         ,0,    n5.sd()
         ,0,    n5.n
         ,True, n8.cohen(n7)
         ,False,n9.cohen(n7)
         ,True, n8.hedges(n7)
         ,False,n9.hedges(n7)
         ,True, n8.bootstrap(n7)
         ,False,n9.bootstrap(n7)
         ]

def bootstrap(y,z,
              conf = The.math.brink.conf,
              b    = The.math.bootstraps):
  """The bootstrap hypothesis test from p220 to 223 
  of Efron's book 'Introduction to the bootstrap'."""
  def someTestStatistic(one,two): 
    s1,s2 = one.sd(), two.sd()
    delta = two.mu - one.mu
    if s1+s2:
      delta =  delta/((s1/one.n + s2/two.n)**0.5)
    return delta
  def one(lst): return lst[ int(any(len(lst))) ]
  def any(n)  : return random.uniform(0,n)
  x      = y + z
  xnum,ynum,znum = Num(x), Num(y), Num(z)
  tobs   = someTestStatistic(ynum,znum)
  yhat   = [y1 - ynum.mu + xnum.mu for y1 in y]
  zhat   = [z1 - znum.mu + xnum.mu for z1 in z]
  bigger = 0.0
  for i in range(b):
    if someTestStatistic(
      Num(one(yhat) for _ in yhat),
      Num(one(zhat) for _ in zhat)) > tobs:
      bigger += 1
  return (bigger / b) <= conf

def a12gt(x,y):
  if (y - x) > 0 : return 1
  if (y - x) < 0 : return -1
  else: return 0

def a12(lst1,lst2, gt= a12gt,
        reverse= The.math.a12.reverse):
  "how often is x in lst1 more than y in lst2?"
  def loop(t,t1,t2): 
    while t1.k < t1.n and t2.k < t2.n:
      h1 = t1.l[t1.k]
      h2 = t2.l[t2.k]
      h3 = t2.l[t2.k+1] if t2.k+1 < t2.n else None 
      if gt(h1,h2) < 0:
        t1.k  += 1; t1.gt += t2.n - t2.k
      elif h1 == h2:
        if h3 and gt(h1,h3) < 0:
            t1.gt += t2.n - t2.k  - 1
        t1.k  += 1; t1.eq += 1; t2.eq += 1
      else:
        t2,t1  = t1,t2
    return t.gt*1.0, t.eq*1.0
  #--------------------------
  if reverse:
    lst1,lst2 = lst2,lst1
  lst1 = sorted(lst1, cmp=gt)
  lst2 = sorted(lst2, cmp=gt)
  n1   = len(lst1)
  n2   = len(lst2)
  t1   = Thing(l=lst1,k=0,eq=0,gt=0,n=n1)
  t2   = Thing(l=lst2,k=0,eq=0,gt=0,n=n2)
  gt,eq= loop(t1, t1, t2)
  return (gt + eq/2)/(n1*n2)

def a12slow(lst1,lst2,rev=True):              
  "how often is x in lst1 more than y in lst2?"
  more = same = 0.0
  n1,n2=len(lst1),len(lst2)
  for x in lst1:
    for y in lst2:
      if   x==y : same += 1
      elif rev     and x > y : more += 1
      elif not rev and x < y : more += 1
  return (more+ 0.5*same) / (n1*n2)

@test
def a12eged():
  imc=[0.0467727930535
       ,0.107422839506
       ,0.143231939163
       ,0.196049098581
       ,0.214018838305 #5
       ,0.295759259259 
       ,0.336425231415
       ,0.400960144928
       ,0.42 #10
       ,0.546434017595
       ,0.600305405094
       ,0.608229508197
       ,0.722651845971
       ,0.733923766816 # 1-
       ,0.780266115803 
       ,1.260375
       ,1.30157738095
       ,1.37680851064
       ,14.6394]
  twopair=[0.0982951758956
           ,0.219622928726
           ,0.238561501328
           ,0.250163795386
           ,0.254009239283 #5
           ,0.271034376595 
           ,0.311751739438
           ,0.314324693953
           ,0.477840168799
           ,0.522865519664 #10
           ,0.526105062302
           ,0.681018891615
           ,0.684250921515
           ,0.739683771336 
           ,0.77830625818 #1
           ,0.812443866931
           ,0.850571338711
           ,1.42509168327
           ,1.4263754399]
  print "a12eg>",a12slow(twopair,imc)

@test
def a12ed(small=The.math.a12.small,
          repeats=100):
  def twolists():
    lst1 = [rand() for _ in range(repeats)]
    lst2 = [rand() for _ in range(repeats)]
    c1   = a12(lst1,lst2) 
    c2   = a12slow(lst1,lst2)
    return c1==c2
  seed()
  return [ True, twolists(),
           True, twolists(),
           True, twolists(),
           True, twolists(),
           True, twolists()]

if __name__ == '__main__': eval(cmd())
