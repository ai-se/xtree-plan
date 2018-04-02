from __future__ import division, print_function
from model import rf
import numpy as np
import random
from time import time
from pdb import set_trace

class settings:
  iter=50
  N=100
  f=0.5
  cf=0
  maxIter=100
  lives=10

def flatten(x):
  """
  Takes an N times nested list of list like [[a,b],[c, [d, e]],[f]]
  and returns a single list [a,b,c,d,e,f]
  """
  result = []
  for el in x:
    if hasattr(el, "__iter__") and not isinstance(el, basestring):
      result.extend(flatten(el))
    else:
      result.append(el)
  return result

def de0(model, new=[], pop=int(1e4), iter=1000, lives=5, settings=settings):
  frontier = model.generate(pop)

  def cdom(x, y, better=['more','more']):

    def loss1(i,x,y):
      return (x - y) if better[i] == 'less' else (y - x)

    def expLoss(i,x,y,n):
      return np.exp(loss1(i,x,y) / n)

    def loss(x, y):
      n      = min(len(x), len(y)) #lengths should be equal
      losses = [expLoss(i,xi,yi,n) for i, (xi, yi) in enumerate(zip(x,y))]
      return sum(losses)/n

    "x dominates y if it losses least"
    if not isinstance(x,list):
      return x<y if better=='less' else x>y
    else:
      return loss(x,y) < loss(y,x)

  def bdom(x, y, better=['more','more']):
    # if not isinstance(x,list):
    #   print(x,y)
    #   return x<y if better=='less' else x>y
    # else:
    #   # return x[0]>0.6 and x[1]>0.6
    # print(x,y)
      return x[0]>y[0]
      #return x[0]>y[0] and x[1]>y[1]

  def extrapolate(current, l1, l2, l3):
    def extrap(i,a,x,y,z):
      try:
        return max(model.dec_lim[i][0], min(model.dec_lim[i][1], x + settings.f * (z - y))) # if random.random()>settings.cf else a
      except:
        set_trace()
    return [extrap(i, a,x,y,z) for i, a, x, y, z in zip(range(model.n_dec), current, l3, l1, l2)]

  def one234(one, pop):
    ids = [i for i,p in enumerate(pop) if not p==one]
    a = np.random.choice(ids, size=3, replace=False)
    return one, pop[a[0]], pop[a[1]], pop[a[2]]

  while lives > 0 and iter>0:
    better = False
    xbest = random.choice(frontier)
    xbestVal = model.solve(xbest)
    iter-=1
    # print(iter)
    for pos in xrange(len(frontier)):
      # print(len(frontier), pos)
      lives -= 1
      # t=time()
      now, l1, l2, l3 = one234(frontier[pos], frontier)
      # print(time()-t)
      # set_trace()
      # t=time()
      new = extrapolate(now, l1, l2, l3)
      # print(time()-t)
      # t=time()
      newVal=model.solve(new)
      # print(time()-t)
      # t=time()
      oldVal=model.solve(now)
      # print(time()-t)
      # print(iter, lives)
      # t=time()
      if bdom(newVal, oldVal):
        frontier.pop(pos)
        frontier.insert(pos, new)
        lives += 1
        if bdom(newVal, xbestVal):
          # print('Yes!')
          xbest=new
          return xbest
      elif bdom(oldVal, newVal):
        better = False
        if bdom(oldVal, xbestVal):
          # print('Yes!')
          xbest=new
          return xbest
        # print(oldVal, newVal)
      else:
        lives += 1
        if bdom(newVal, xbestVal):
          # print('Yes!')
          # lives=-10
          xbest=new
          return xbest
      # print(time()-t)

  # print([model.solve(f) for f in frontier])
  def best(aa):
    one,two = model.solve(aa)
    return 1 if one>0.3 and two>0.3 else 1 if one>0.3 or two>0.3 else 0
  best1 = [ff for ff in frontier if best(ff)>1]
  # print(model.solve(sorted(best1, key=lambda F: model.solve(F)[0])[-1]))
  if len(best1)==0:
    return sorted(frontier, key=lambda F: model.solve(F)[0])[-1]
  return sorted(best1, key=lambda F: model.solve(F)[0])[-1]
  # return xbest#sorted(frontier, key=lambda F: model.solve(F))[-1]
  # return sorted(frontier, key=lambda F: model.solve(F))[-1]

def tuner(data, smoteTune=True):
  if len(data)==1:
    return None
  else:
    return de0(model = rf(data=data, obj=-1, smoteTune=smoteTune),pop=50, iter=10)
