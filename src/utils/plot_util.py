from __future__ import division
import os
import re
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pdb import set_trace
from collections import Counter
from scipy.stats import ttest_ind

# Update path
root = os.path.join(os.getcwd().split('src')[0], 'src')
if root not in sys.path:
    sys.path.append(root)


def plot_compare(dframe, save_path=os.path.join(root, "results"), y_lbl="", title="", postfix=None):

    #  Clear all
    plt.clf()
    #  We define a fake subplot that is in fact only the plot.
    plot = plt.figure(figsize=(3, 4)).add_subplot(111)
    #  We change the fontsize of minor ticks label
    plot.tick_params(axis='both', which='major', labelsize=12)

    #  Plot Data
    plt.plot(dframe["Overlap"], dframe["XTREEv1"],
             color='#a50f15', linewidth=1)
    plt.plot(dframe["Overlap"], dframe["Alves"],
             color='#2c7fb8', linewidth=1)
    plt.plot(dframe["Overlap"], dframe["Oliveira"],
             color='#636363', linewidth=1)
    plt.plot(dframe["Overlap"], dframe["Shatnawi"],
             color='#78c679', linewidth=1)

    #  Set title, axes labels
    plt.title(title, size=12)
    plt.ylabel(y_lbl, size=12)
    plt.xlabel("Overlap", size=12)
    plt.legend(loc="best")

    fname = os.path.join(save_path, re.sub(
        " ", "_", title).lower() + "_" + postfix + ".png")

    plt.savefig(fname, dpi=300, facecolor='w', edgecolor='w', figsize=(3, 4),
                orientation='portrait', papertype=None, format=None,
                transparent=True, bbox_inches="tight", pad_inches=0.1,
                frameon=None)


def plot_bar(dframe_inc, dframe_dec, save_path=os.path.join(root, "results"), y_lbl="", title="", postfix=""):

    #  Clear all
    plt.clf()
    #  We define a fake subplot that is in fact only the plot.
    # plot = plt.figure(figsize=(3, 4)).add_subplot(111)
    #  We change the fontsize of minor ticks label
    plt.tick_params(axis='both', which='major', labelsize=20)

    bar_width = 0.3
    group_sep = 0.1

    opacity = 0.7

    index = np.arange(len(dframe_dec['XTREE']))

    plt.bar(index, dframe_dec["XTREE"], bar_width,
            color='#9D1C29', label='XTREE (Decreased)')
    plt.bar(index + bar_width, dframe_inc["XTREE"], bar_width,
            color='#D72638', alpha=opacity,
            label='XTREE (Increased)')

#     plt.bar(index + 2 * bar_width + group_sep, dframe_dec["Alves"], bar_width,
#             color='#37002F', label='Alves (Decreased)')
#     plt.bar(index + 3 * bar_width + group_sep, dframe_inc["XTREE"], bar_width,
#             color='#53174B', alpha=opacity,
#             label='Alves (Increased)')

#     plt.bar(index + 4 * bar_width + 2 * group_sep, dframe_dec["Shatnawi"], bar_width,
#             color='#238443', label='Shatw (Decreased)')
#     plt.bar(index + 5 * bar_width + 2 * group_sep, dframe_inc["Shatnawi"], bar_width,
#             color='#238443', alpha=opacity,
#             label='Shatw (Increased)')

#     plt.bar(index + 6 * bar_width + 3 * group_sep, dframe_dec["Oliveira"], bar_width,
#             color='#E8500A', label='Olive (Decreased)')
#     plt.bar(index + 7 * bar_width + 3 * group_sep, dframe_inc["Oliveira"], bar_width,
#             color='#FF7536', alpha=opacity,
#             label='Oliveria (Increased)')

    #  Set title, axes labels
    plt.title(title, size=20)
    plt.ylabel(y_lbl, size=20)
    plt.xlabel("Overlap", size=20)
    plt.xticks(index + bar_width * 4, ('25', '50', '75', '100', ''))
    plt.legend(loc="best")

    # Filename
    fname = os.path.join(save_path, re.sub(" ", "_", title).lower() + ".png")

    # plt.show()
    plt.savefig(fname, dpi=300, facecolor='w', edgecolor='w', figsize=(3, 4),
                orientation='portrait', papertype=None, format=None,
                transparent=False, bbox_inches="tight", pad_inches=0.1,
                frameon=None)
