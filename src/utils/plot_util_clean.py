from __future__ import division
import os
import re
import sys
import seaborn as sns
from pdb import set_trace
import matplotlib.pyplot as plt

# Update path
root = os.path.join(os.getcwd().split('src')[0], 'src')
if root not in sys.path:
    sys.path.append(root)

# Set plot style
flatui = ["#FF948D", "#FFC991", "#C5C593", "#67B193"]
greyui = sns.color_palette('Greys_r', 4)
sns.palplot(flatui)
sns.set_context("paper")


def plot_violin(dframe, save_path=os.path.join(root, "results"), y_lbl="", title="", postfix=None):
    #  Clear all
    plt.clf()

    #  We define a fake subplot that is in fact only the plot.
    g = plt.figure(figsize=(6, 4)).add_subplot(111)

    #  We change the fontsize of minor ticks label
    g.tick_params(axis='both', which='major', labelsize=10)

    # Covert the Numerical axis to
    dframe['Num'] = dframe['Num'].astype('float')
    g = sns.catplot(x="Overlap", y="Num", hue="Method", palette=flatui,
                    kind='violin', inner='point', data=dframe, linewidth=0.25, legend=False)
    # g = sns.catplot(x="Overlap", y="Num", hue="Method", palette=flatui, kind='swarm', inner=None, data=dframe, linewidth=0.25, legend=False)

    sns.despine(offset=10, trim=True)

    #  Set title, axes labels
    g.set_titles(title, size=12)
    g.set_ylabels(y_lbl, size=12)
    g.set_xlabels("Overlap", size=12)
    plt.legend(loc="best", fontsize=10)

    fname = os.path.join(save_path, re.sub(
        " ", "_", title).lower() + "_" + postfix + ".png")

    g.savefig(fname, dpi=300, facecolor='w', edgecolor='w', figsize=(6, 4),
              orientation='portrait', papertype=None, format='png',
              transparent=True, bbox_inches="tight", pad_inches=0.1,
              frameon=None)


def plot_catplot(dframe, save_path=os.path.join(root, "results"), y_lbl="", title="", postfix=None, factorplot=False):
    #  Clear all
    plt.clf()

    #  We define a fake subplot that is in fact only the plot.
    plot = plt.figure(figsize=(4, 3)).add_subplot(111)
    plot.axhline(linewidth=2)
    plot.axvline(linewidth=2)

    #  We change the fontsize of minor ticks label
    plot.tick_params(axis='both', which='major', labelsize=12)

    # Covert the Numerical axis to
    dframe['Num'] = dframe['Num'].astype('float')
    sns.catplot(x="Overlap", y="Num", hue="Method", palette=flatui, data=dframe,
                linewidth=0.25, kind="bar", legend=False, errwidth=1, ci="sd")

    sns.despine(offset=10)

    #  Set title, axes labels
    plt.title(title, size=12)
    plt.ylabel(y_lbl, size=12)
    plt.xlabel("Overlap", size=12)
    plt.legend(loc="best", fontsize=10, frameon=False, shadow=False)

    fname = os.path.join(save_path, re.sub(
        " ", "_", title).lower() + postfix + ".png")

    plt.savefig(fname, dpi=300, facecolor='w', edgecolor='w', figsize=(4, 3),
                orientation='portrait', papertype=None, format='png',
                transparent=True, bbox_inches="tight", pad_inches=0.1,
                frameon=None)
