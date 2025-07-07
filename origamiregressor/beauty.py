# -------------------------------------------------- #
# Helper functions to plot the results
#
# AUTHOR: Andrea Gardin
# -------------------------------------------------- #

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def get_axes(plots: int, 
             max_col: int =2, 
             fig_frame: tuple =(3.3,3.), 
             res: int =200):
    """Define Fig and Axes objects.

    :param plots: number of plots frames in a fig.
    :type plots: int
    :param max_col: number of columns to arrange the frames, defaults to 2
    :type max_col: int, optional
    :param fig_frame: frame size, defaults to (3.3,3.)
    :type fig_frame: tuple, optional
    :param res: resolution, defaults to 200
    :type res: int, optional
    :return: fig and axes object from matplotlib.
    :rtype: _type_
    """
    # cols and rows definitions
    cols = plots if plots <= max_col else max_col
    rows = int(plots / max_col) + int(plots % max_col != 0)

    fig, axes = plt.subplots(rows,
                             cols,
                             figsize=(cols * fig_frame[0], rows * fig_frame[1]),
                             dpi=res)
    if plots > 1:
        axes = axes.flatten()
        for i in range(plots, max_col*rows):
            remove_frame(axes[i])
    elif plots == 1:
        pass
    
    return fig, axes


def remove_frame(axes) -> None:
    for side in ['bottom', 'right', 'top', 'left']:
        axes.spines[side].set_visible(False)
    axes.set_yticks([])
    axes.set_xticks([])
    axes.xaxis.set_ticks_position('none')
    axes.yaxis.set_ticks_position('none')
    pass


def plot_discrete_histogram_from_dict(data: dict, axis):
    # sort the dict
    data_dict = dict(sorted(data.items()))

    # Extract keys and values from the dictionary
    labels = list(data_dict.keys())
    dummy_labels = np.arange(len(labels))

    values = list(data_dict.values())
    cmap = sns.color_palette("Spectral_r", n_colors=len(labels))

    bars = axis.bar(dummy_labels, values, color=cmap, edgecolor='0.', zorder=2)
    for bar, value in zip(bars, values):
        axis.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), 
                  f'{value}', ha='center', va='bottom')
    axis.set_ylabel('Counts')
    axis.set_xticks(dummy_labels)
    axis.set_xticklabels(labels, rotation=0)


def get_categorical_distribution(df, categorical_vars):

    cat_distr_dict = {}

    for _,cat in enumerate(categorical_vars):

        cat_distr_dict[cat] = len(df[cat].loc[df[cat]==1])

    return cat_distr_dict


def int_to_roman(num: int) -> str:
    val = [
        1000, 900, 500, 400,
        100, 90, 50, 40,
        10, 9, 5, 4,
        1
        ]
    syb = [
        "M", "CM", "D", "CD",
        "C", "XC", "L", "XL",
        "X", "IX", "V", "IV",
        "I"
        ]
    roman_num = ''
    for i in range(len(val)):
        count = int(num / val[i])
        roman_num += syb[i] * count
        num -= val[i] * count
    return roman_num