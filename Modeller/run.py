#! /Users/rkrsn/miniconda/bin/python
from __future__ import print_function, division
from os import environ, getcwd
import sys

# Update PYTHONPATH
HOME = environ['HOME']
axe = HOME + '/git/axe/axe/'  # AXE
pystat = HOME + '/git/pystat/'  # PySTAT
SOURCE = '../SOURCE/'
sys.path.extend([axe, pystat, SOURCE])

import csv
from Prediction import *
from _imports import *
from abcd import _Abcd
from methods1 import *
import numpy as np
import pandas as pd
from pdb import set_trace
from WHAT import treatments
from _model import xomod


class model:

  def __init__(i):
    pass

  def toCSV(i, data, dir='Data/', name=None):
    with open(dir + name, 'w') as csvfile:
      writer = csv.writer(csvfile, delimiter=',',
                          quotechar='|', quoting=csv.QUOTE_MINIMAL)
      writer.writerow(data[0])  # Header
      for cells in data[1]:  # Body
        writer.writerow(cells)
    return dir + name

  def genData(i):
    train = i.toCSV(xomod(N=100), name='Train.csv')
    test = i.toCSV(xomod(N=100), name='Test.csv')
    return createTbl(train), test

if __name__ == "__main__":
  model().genData()
