"""

# Support Code


## Standard Headers

"""
from __future__ import division,print_function
import  sys,random,math
sys.dont_write_bytecode = True
from settingsWhere import *
"""

## Simple, low-level stuff

### Maths Stuff

"""
def gt(x,y): return x > y
def lt(x,y): return x < y

def medianIQR(lst, ordered=False):
  if not ordered: 
    lst = sorted(lst)
  n = len(lst)
  q = n//4
  iqr = lst[q*3] - lst[q]
  if n % 2: 
    return lst[q*2],iqr
  else:
    p = max(0,q-1)
    return (lst[p] + lst[q]) * 0.5,iqr

def median(lst,ordered=False):
  return medianIQR(lst,ordered)[0]



"""

An accumulator for reporting on numbers.

"""
class N(): 
  "Add/delete counts of numbers."
  def __init__(i,inits=[]):
    i.zero()
    map(i.__iadd__,inits)
  def zero(i): 
    i.n = i.mu = i.m2 = 0
    i.cache= Cache()
  def sd(i)  : 
    if i.n < 2: 
      return 0
    else:       
      return (max(0,i.m2)/(i.n - 1))**0.5
  def __iadd__(i,x):
    i.cache += x
    i.n     += 1
    delta    = x - i.mu
    i.mu    += delta/(1.0*i.n)
    i.m2    += delta*(x - i.mu)
    return i
  def __isub__(i,x):
    i.cache = Cache()
    if i.n < 2: return i.zero()
    i.n  -= 1
    delta = x - i.mu
    i.mu -= delta/(1.0*i.n)
    i.m2 -= delta*(x - i.mu)  
    return i

class Cache:
  "Keep a random sample of stuff seen so far."
  def __init__(i,inits=[]):
    i.all,i.n,i._has = [],0,None
    map(i.__iadd__,inits)
  def __iadd__(i,x):
    i.n += 1
    if len(i.all) < 128: # if not full
      i._has = None
      i.all += [x]               # then add
    else: # otherwise, maybe replace an old item
      if random.random() <= The.cache.size/i.n:
        i._has=None
        i.all[int(random.random()*The.cache.size)] = x
    return i
  def has(i):
    if i._has == None:
      lst  = sorted(i.all)
      med,iqr = medianIQR(i.all,ordered=True)
      i._has = o(
        median = med,      iqr = iqr,
        lo     = i.all[0], hi  = i.all[-1])
    return i._has
"""

### Random stuff.

"""
by   = lambda x: random.uniform(0,x) 
rseed = random.seed
any  = random.choice
rand = random.random

def seed(r=None):
  global The
  if The is None: The=defaults()
  if r is None: r = The.seed
  rseed(r)

"""

### List Handling Tricks

"""
def first(lst): return lst[0]
def second(lst): return lst[1]
def third(lst): return lst[2]
"""

### Printing Stuff

Print without newline:

"""
def say(*lst): print(*lst,end="")
"""

Print a list of numbers without an excess of decimal places:

"""
def gs(lst) : return [g(x) for x in lst]
def g(x)    : 
  txt = '%g' % x
  return int(txt) if int(x) == x else float(txt)
"""

Pretty print a dictionary:

"""
def showd(d):
  def one(k,v):
    if isinstance(v,list):
      v = gs(v)
    if isinstance(v,float):
      return ":%s %g" % (k,v)
    return ":%s %s" % (k,v)
  return ' '.join([one(k,v) for k,v in
                    sorted(d.items())
                     if not "_" in k])
"""

## Decorator to run code at Start-up

"""
def go(f):
  "A decorator that runs code at load time."
  print("\n# ---|", f.__name__,"|-----------------")
  if f.__doc__: print("#", f.__doc__)
  f()
"""

## Handling command line options.

Convert command line to a function call.
e.g. if the file lib.py ends with

    if __name__ == '__main__':eval(todo())

then 

    python lib.py myfun :a 1 :b fred  

results in a call to  _myfun(a=1,b='fred')_.

"""
def todo(com="print(The._logo,'WHERE (2.0) you at?')"):
  import sys
  if len(sys.argv) < 2: return com
  def strp(x): return isinstance(x,basestring)
  def wrap(x): return "'%s'"%x if strp(x) else str(x)  
  def oneTwo(lst):
    while lst: yield lst.pop(0), lst.pop(0)
  def value(x):
    try:    return eval(x)
    except: return x
  def two(x,y): return x[1:] +"="+wrap(value(y))
  twos = [two(x,y) for x,y in oneTwo(sys.argv[2:])]
  return sys.argv[1]+'(**dict('+ ','.join(twos)+'))'
"""

## More interesting, low-level stuff

"""
def timing(f,repeats=10):
  "How long does 'f' take to run?"
  import time
  time1 = time.clock()
  for _ in range(repeats):
    f()
  return (time.clock() - time1)*1.0/repeats
"""

## Data Completion Tool

Fills in some details on a table of data. For example, 

     def nasa93():
       vl=1;l=2;n=3;h=4;vh=5;xh=6
       return data(indep= [ 
                     'Prec', 'Flex', 'Resl', 'Team', 'Pmat', 'rely', 'data', 'cplx', 'ruse',
                     'docu', 'time', 'stor', 'pvol', 'acap', 'pcap', 'pcon', 'aexp', 'plex',  
                     'ltex', 'tool', 'site', 'sced', 'kloc'],
                   less = ['effort', 'defects', 'months'],
                   _rows=[
                      [h,h,h,vh,h,h,l,h,n,n,n,n,l,n,n,n,n,n,h,n,n,l,25.9,117.6,808,15.3],
                      [h,h,h,vh,h,h,l,h,n,n,n,n,l,n,n,n,n,n,h,n,n,l,24.6,117.6,767,15.0],
                      [h,h,h,vh,h,h,l,h,n,n,n,n,l,n,n,n,n,n,h,n,n,l,7.7,31.2,240,10.1],
     ...

Adds in information on _cols_, _decisions_, _hi,lo_, etc:

    {	:cols [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 
             12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 
             22, 22, 23, 24]
 	    :decisions [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 
                  11, 12, 13, 14, 15, 16, 17, 18, 
                  19, 20, 21, 22]
 	    :eval <function <lambda> at 0x7f3f825bea28>
 	    :hi {0: 4, 1: 4, 2: 4, 3: 5, 4: 4, 5: 5, 6: 5, 
           7: 6, 8: 3, 9: 3, 10: 6, 11: 6, 12: 4, 13: 5, 
           14: 5, 15: 3, 16: 5, 17: 4, 18: 4, 19: 4, 
           20: 3, 21: 3, 22: 980, 23: 8211, 24: 50961}
 	    :lo {0: 4, 1: 4, 2: 4, 3: 5, 4: 2, 5: 2, 6: 2, 
           7: 2, 8: 3, 9: 3, 10: 3, 11: 3, 12: 2, 
           13: 3, 14: 3, 15: 3, 16: 2, 17: 1, 18: 1, 
            19: 3, 20: 3, 21: 2, 22: 0.9, 23: 8.4, 24: 28}
 	    :names ['Prec', 'Flex', 'Resl', 'Team', 'Pmat', 
              'rely', 'data', 'cplx', 'ruse', 'docu', 
              'time', 'stor', 'pvol', 'acap', 'pcap', 
              'pcon', 'aexp', 'plex', 'ltex', 'tool', 
              'site', 'sced', 'kloc', 'effort', 
              'defects', 'months']
 	    :objectives [22, 23, 24]
 	    :w {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 
          7: 1, 8: 1, 9: 1, 10: 1, 11: 1, 12: 1, 13: 1, 
          14: 1, 15: 1, 16: 1, 17: 1, 18: 1, 19: 1, 
          20: 1, 21: 1, 22: -1, 23: -1, 24: -1}
    }

Code:

"""
def data(indep=[], less=[], more=[], _rows=[]):
  nindep= len(indep)
  ndep  = len(less) + len(more)
  m= o(lo={}, hi={}, w={}, 
       eval  = lambda m,it : True,
       _rows = [o(cells=r,score=0,scored=False,
                  x0=None,y0=None) 
                for r in _rows],
       names = indep+less+more)
  m.decisions  = [x for x in range(nindep)]
  m.objectives = [nindep+ x- 1 for x in range(ndep)]
  m.cols       = m.decisions + m.objectives
  for x in m.decisions : 
    m.w[x]=  1
  for y,_ in enumerate(less) : 
    m.w[x+y]   = -1
  for z,_ in enumerate(more) : 
    m.w[x+y+z] =  1
  for x in m.cols:
    all = sorted(row.cells[x] for row in m._rows)
    m.lo[x] = all[0]
    m.hi[x] = all[-1]
  return m
"""

## Start-up Actions

"""
if __name__ == '__main__': eval(todo())
