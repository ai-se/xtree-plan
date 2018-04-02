from __future__ import division
import sys
sys.dont_write_bytecode = True

from lib    import *
from demos  import *
from counts import *
from table  import *

@demo
def _discreteTable(f="data.dat/weather2.csv"):
  for row in  discreteTable(f)._rows:
    rprint(row)

def ediv(lst, lvl=0,tiny=The.tree.min,
         dull=The.math.brink.cohen,
         num=lambda x:x[0], sym=lambda x:x[1]):
  "Divide lst of (numbers,symbols) using entropy."""
  #----------------------------------------------
  #print watch
  def divide(this,lvl): # Find best divide of 'this' lst.
    def ke(z): return z.k()*z.ent()
    lhs,rhs   = Sym(), Sym(sym(x) for x in this)
    n0,k0,e0,ke0= 1.0*rhs.n,rhs.k(),rhs.ent(),ke(rhs)
    cut, least  = None, e0
    last = num(this[0])
    for j,x  in enumerate(this): 
      rhs - sym(x); #nRhs - num(x)
      lhs + sym(x); #nLhs + num(x)
      now = num(x)
      if now != last:
        if lhs.n > tiny and rhs.n > tiny: 
          maybe= lhs.n/n0*lhs.ent()+ rhs.n/n0*rhs.ent()       
          if maybe < least : 
            gain = e0 - maybe
            delta= log2(float((3**k0-2)-(ke0- ke(rhs)-ke(lhs))))
            if gain >= (log2(n0-1) + delta)/n0: 
              cut,least = j,maybe
      last= now
    return cut,least
  #--------------------------------------------
  def recurse(this, cuts,lvl):
    cut,e = divide(this,lvl)
    if cut: 
      recurse(this[:cut], cuts, lvl+1); 
      recurse(this[cut:], cuts, lvl+1)
    else:   
      lo    = num(this[0])
      hi    = num(this[-1])
      cuts += [Thing(at=lo, 
                     e=e,_has=this,
                     range=(lo,hi))]
    return cuts
  #---| main |-----------------------------------
  return recurse(sorted(lst,key=num),[],0)
  
@demo
def _ediv():
  "Demo code to test the above."
  import random
  bell= random.gauss
  random.seed(1)
  def go(lst):
    print ""; print sorted(lst)[:10],"..."
    for d in  ediv(lst,tiny=2):
      rprint(d); nl()
  X,Y="X","Y"
  l=[(1,X),(2,X),(3,X),(4,X),(11,Y),(12,Y),(13,Y),(14,Y)]
  go(l)
  l[0] = (1,Y)
  go(l)
  go(l*2)
  go([(1,X),(2,X),(3,X),(4,X),(11,X),(12,X),(13,X),(14,X)])
  go([(64,X),(65,Y),(68,X),(69,Y),(70,X),(71,Y),
      (72,X),(72,Y),(75,X),(75,X),
      (80,Y),(81,Y),(83,Y),(85,Y)]*2)
  l=[]
  for _ in range(1000): 
    l += [(bell(20,1),  X),(bell(10,1),Y),
          (bell(30,1),'Z'),(bell(40,1),'W')] 
  go(l)
  go([(1,X)])


  
if __name__ == '__main__': eval(cmd())
