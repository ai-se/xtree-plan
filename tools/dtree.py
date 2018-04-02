from __future__ import division, print_function
from PC45 import dtree
import pandas as pd
import numpy as np
from containers import *

def settings(**d):
  return Thing(min=1,
               maxLvL=10,
               infoPrune=0.33,
               klass=-1,
               prune=True).override(d)


def classify(test, tree):
  """apex=  leaf at end of biggest (most supported)
   branch that is selected by test in a tree"""
  def equals(val,span):
    if val == opt.missing or val==span:
      return True
    else:
      if isinstance(span,tuple):
        lo,hi = span
        return lo <= val < hi
      else:
        return span == val
  def apex1(cells,tree):
    found = False
    for kid in tree.kids:
      val = cells[kid.f.col]
      if equals(val,kid.val):
        for leaf in apex1(cells,kid):
          found = True
          yield leaf
    if not found:
      yield tree
  leaves= [(len(leaf.rows),leaf)
           for leaf in apex1(test,tree)]
  return second(last(sorted(leaves)))
