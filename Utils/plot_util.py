from __future__ import division
import os
import re
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pdb import set_trace
from collections import Counter
from scipy.stats import ttest_ind

# Update path
root = os.path.join(os.getcwd().split('xtree-plan')[0], 'xtree-plan')
if root not in sys.path:
    sys.path.append(root)

def plot_compare(dframe, save_path=os.path.join(root, "results"), y_lbl="", title="", postfix = None):
    
    #  Clear all
    plt.clf()
    #  We define a fake subplot that is in fact only the plot.
    plot = plt.figure(figsize=(3, 4)).add_subplot(111)
    #  We change the fontsize of minor ticks label
    plot.tick_params(axis='both', which='major', labelsize=12)

    #  Plot Data
    plt.plot(dframe["Overlap"], dframe["XTREE"],
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

    fname = os.path.join(save_path, re.sub(" ", "_", title).lower() + "_" + postfix + ".png")

    plt.savefig(fname, dpi=300, facecolor='w', edgecolor='w', figsize=(3, 4),
                        orientation='portrait', papertype=None, format=None,
                        transparent=True, bbox_inches="tight", pad_inches=0.1,
                        frameon=None)
