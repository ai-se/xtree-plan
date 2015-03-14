#! /Users/rkrsn/miniconda/bin/python
from __future__ import print_function, division
from os import environ, getcwd
import sys

# Update PYTHONPATH
HOME = environ['HOME']
axe = HOME + '/git/axe/axe/'  # AXE
pystat = HOME + '/git/pystats/'  # PySTAT
cwd = getcwd()  # Current Directory
sys.path.extend([axe, pystat, cwd])

from Prediction import *
from Planning import *
from _imports import *
from abcd import _Abcd
from cliffsDelta import showoff
from contrastset import *
from dectree import *
from methods1 import *
import numpy as np
import pandas as pd
from sk import rdivDemo
from pdb import set_trace
from dEvol import tuner
from Planner2 import treatments as treatments2
from os import walk


class run():

    def __init__(self, pred=CART, _smoteit=True, _n=-1,
                 _tuneit=False, dataName=None, reps=10):
        self.pred = pred
        self._smoteit = _smoteit
        self.train, self.test = self.categorize()
        self.reps = reps
        self._n = _n
        self.tunedParams = None if not _tuneit else tuner(
            self.pred, self.train[_n])

    def categorize(self):
        self.projects = [Name for _, Name, __ in walk(dir)][0]
        self.numData = len(self.projects)  # Number of data
        one, two = explore(dir)
        data = [one[i] + two[i] for i in xrange(len(one))]

        def withinClass(data):
            N = len(data)
            return [(data[:n], [data[n]]) for n in range(1, N)]

        def whereis():
            for indx, name in enumerate(self.projects):
                if name == self.dataName:
                    return indx

        return [dat[0] for dat in withinClass(data[whereis()])]\
            , [dat[1] for dat in withinClass(data[whereis()])]  # Train, Test

    def go(self):
        