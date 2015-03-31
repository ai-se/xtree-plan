#!/usr/bin/env python
# a bar plot with errorbars
from numpy import median
from numpy import std
from numpy import arange
import matplotlib.pyplot as plt


def histplot(dict, name='untitled', ext='.jpg'):
  N = len(dict)
  bMedians = [median(dict[l][0]) for l in dict]
  bStd = [std(dict[l][0]) for l in dict]

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

  aMedians = [median(dict[l][1]) for l in dict]
  aStd = [std(dict[l][1]) for l in dict]
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

  # add some text for labels, title and axes ticks
  ax.set_ylabel('Bugs')
  ax.set_xticks(ind + width)
  ax.set_xticklabels(tuple([l[:3] for l in dict]))

  ax.legend((rects1[0], rects2[0]), ('Before', 'After'))

#   def autolabel(rects):
# attach some text labels
#     for rect in rects:
#       height = rect.get_height()
#       ax.text(rect.get_x() + rect.get_width() / 2., 1.05 * height, '%d' % int(height),
#               ha='center', va='bottom')
#
#   autolabel(rects1)
#   autolabel(rects2)

  plt.savefig('%s%s' % (name, ext))
