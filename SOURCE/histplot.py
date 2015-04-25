#!/usr/bin/env python
# a bar plot with errorbars
from pdb import set_trace

from numpy import arange
from numpy import array
from numpy import median
from numpy import std
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import plotly.plotly as py


py.sign_in('rkrsn', '4gn17rzop8')


def histplot(dict, name='untitled', ext='.jpg'):
  pp = PdfPages(name + '.pdf')
  N = len(dict)
  bMedians = array([median(dict[l][0]) for l in dict])
  lEb = array([sorted(dict[l][0])[int(0.25 * len(dict[l][0]))] for l in dict])
  hEb = array([sorted(dict[l][0])[int(0.75 * len(dict[l][0]))] for l in dict])
  bStd = (bMedians - lEb, hEb - bMedians)
  ind = arange(N)  # the x locations for the groups
  width = 0.35       # the width of the bars

  fig, ax = plt.subplots()
  rects1 = ax.bar(
      ind,
      bMedians,
      width,
      color=[
          0.9,
          0.9,
          0.9],
      yerr=bStd,
      ecolor='k')

  aMedians = array([median(dict[l][1]) for l in dict])
  lEa = array([sorted(dict[l][1])[int(0.25 * len(dict[l][1]))] for l in dict])
  hEa = array([sorted(dict[l][1])[int(0.75 * len(dict[l][1]))] for l in dict])
  aStd = (aMedians - lEa, hEa - aMedians)
  rects2 = ax.bar(
      ind + width,
      aMedians,
      width,
      color=[
          0.7,
          0.7,
          0.7],
      yerr=aStd,
      ecolor='k')

  plt.xticks(rotation=45)
  # add some text for labels, title and axes ticks
  ax.set_ylabel('Bugs')
  ax.set_xticks(ind + width)
  ax.set_xticklabels(tuple([l[:3] for l in dict]))

  ax.legend((rects1[0], rects2[0]), ('Before', 'After'))
  pp.savefig()
  pp.close()
#   def autolabel(rects):
# attach some text labels
#     for rect in rects:
#       height = rect.get_height()
#       ax.text(rect.get_x() + rect.get_width() / 2., 1.05 * height, '%d' % int(height),
#               ha='center', va='bottom')
#
#   autolabel(rects1)
#   autolabel(rects2)

# def _test():
#   files={}
#   for i in xrange(2):
#     files.upadate({i:})
