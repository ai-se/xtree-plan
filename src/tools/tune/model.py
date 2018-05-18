from __future__ import division, print_function

import subprocess
import sys

# Get the git root directory
root=repo_dir = subprocess.Popen(['git'
                                      ,'rev-parse'
                                      , '--show-toplevel']
                                      , stdout=subprocess.PIPE
                                    ).communicate()[0].rstrip()
sys.path.append(root)

from random import uniform
from tools.oracle import rforest
from tools.misc import *
from Utils.StatsUtils import ABCD

class rf:
  """
  Random Forest
  """
  def __init__(i, data, obj=2,n_dec=7,smoteTune=True):
    i.n_dec = n_dec
    i.train = csv2DF(data[:-1], toBin=True)
    i.test = csv2DF([data[-1]], toBin=True)
    i.n_obj = obj # 2=precision
    i.dec_lim = [(10, 1000)  # n_estimators
                , (1, 100)  # max_features
                , (1, 10)   # min_samples_leaf
                , (2, 10)   # min_samples_split
                , (2, 50)   # max_leaf_nodes
                , (1,  8)   # Minority sampling factor
                , (0,  4)]  # Majority sampling factor
    i.smoteTune=smoteTune

  def generate(i,n):
    return [[uniform(i.dec_lim[indx][0]
                     , i.dec_lim[indx][1]) for indx in xrange(i.n_dec)
             ] for _ in xrange(n)]

  def solve(i,dec):
    # t=time()
    actual, predicted = rforest(i.train, i.test, tunings=dec, smoteit=True
    ,smoteTune=i.smoteTune)
    # except: set_trace()
    # print(time()-t)
    abcd = ABCD(before=actual, after=predicted)
    qual = np.array([k.stats() for k in abcd()])
    # set_trace()
    pd=qual[1][0]
    pf=qual[1][1]
    acc=qual[1][2]
    prec=qual[1][3]
    f1=qual[1][4]
    g1=qual[1][5]
    # print(pf)

    out=1-pf if pf>0.7 else 0
    out1=1-pd if pd>0.6 else 0
    # set_trace()
    return [prec, prec]
    # return [out,out1]
    # return [pf if pf>0.6 else 0, pd if pd>0.6 else 0]
    # return [qual[0][1], qual[1][1]]

if __name__=='__main__':
  problem = DTLZ2(30,3)
  row = problem.generate(1)
  print(problem.solve(row[0]))
