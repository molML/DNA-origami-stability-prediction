# -------------------------------------------------- #
# Helper functions to plot the results
#
# AUTHOR: Andrea Gardin
# -------------------------------------------------- #

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import Normalize

# -------------------------------------------------- #
# Plotting functions
# -------------------------------------------------- #

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

# -------------------------------------------------- #
# Functions for the analysis of the results
# -------------------------------------------------- #


def plot_correlations_A(
    y_test_list: list, 
    y_pred_list: list, 
    y_pred_std_list: list, 
    Xlabel: str,
    Ylabel: str,
    std_model: bool,
    font_size: int =8
):

    # General plot parameters
    x_lim = (-.5,6.5)
    y_lim = x_lim
    grid = False
    colorbar_length = 0.5
    left_position = (1 - colorbar_length) / 2
    roman_numbers = [int_to_roman(r) for r in np.arange(1,len(y_test_list)+1)]

    Xc = []
    Yc = []

    # Correlation scatter plots
    fig, ax = get_axes(len(y_test_list)+1,3)
    for i,(x,y,z) in enumerate(zip(y_test_list,y_pred_list,y_pred_std_list)):
        if std_model:
            _ = ax[i].scatter(x,y, c=z, vmin=0., vmax=2.5, cmap='gnuplot', edgecolor='0.')
        else:
            _ = ax[i].scatter(x,y, c='C0', edgecolor='0.')

        ax[i].plot(x_lim, y_lim, 'k--', lw=2)
        ax[i].set_xlim(x_lim)
        ax[i].set_xticks(np.arange(0,8,2))
        ax[i].set_xlabel(Xlabel, fontsize=font_size)
        ax[i].set_ylim(y_lim)
        ax[i].set_yticks(np.arange(0,8,2))
        ax[i].set_ylabel(Ylabel, fontsize=font_size)
        if grid:
            ax[i].grid(ls='--', alpha=.5)

        Xc.append(x)
        Yc.append(y)

    if std_model:
        cbar_ax = fig.add_axes([left_position, 1.00, colorbar_length, 0.02])
        cbar = fig.colorbar(ax[0].collections[0], cax=cbar_ax, orientation='horizontal')
        cbar.ax.xaxis.set_ticks_position('top')
        cbar.ax.xaxis.set_label_position('top')
        cbar.set_label(r'Model uncertainty (standard-deviation)', fontsize=font_size)
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=1)
    for i, rom in enumerate(roman_numbers):
        ax[i].text(0.05, 0.95,
                    f'Rep. {rom}',
                    transform=ax[i].transAxes, fontsize=10,
                    verticalalignment='top', bbox=props)
        
    # Desity hexbin plot
    hb = ax[-1].hexbin(np.ravel(Xc), np.ravel(Yc), extent=(x_lim[0], x_lim[1], y_lim[0], y_lim[1]),
                    gridsize=15, cmap='Reds', edgecolor='1.')
    cb = fig.colorbar(hb, ax=ax[-1], orientation='vertical')
    cb.set_label('Counts', fontsize=font_size)

    ax[-1].plot(x_lim, y_lim, '--', lw=2, c='0.')
    ax[-1].set_xlim(x_lim)
    ax[-1].set_xticks(np.arange(0,8,2))
    ax[-1].set_ylim(y_lim)
    ax[-1].set_yticks(np.arange(0,8,2))
    ax[-1].set_ylabel(Ylabel, fontsize=font_size)
    ax[-1].set_xlabel(Xlabel, fontsize=font_size)
    if grid:
        ax[-1].grid(ls='--', alpha=.5)

    fig.tight_layout()
    
    return fig, ax


def plot_correlations_B(
    y_test_list: list, 
    y_pred_list: list,
    rmse_scores: list,
    rmse_cv_scores: list, 
    Xlabel: str,
    Ylabel: str,
    model_name: str,
    font_size: int =8
):

    # General plot parameters
    x_lim = (-.5,6.5)
    y_lim = x_lim
    roman_numbers = [int_to_roman(r) for r in np.arange(1,len(y_test_list)+1)]
    bptlabels = ['0'] + roman_numbers

    cmap = plt.cm.Blues_r
    num_plots = len(y_test_list)
    norm = Normalize(vmin=0, vmax=num_plots - 1)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([]) 

    # Correlation scatter plots
    fig, ax = get_axes(2,2)
    for i,(x,y) in enumerate(zip(y_test_list,y_pred_list)):
        color = cmap(norm(i))
        ax[0].scatter(x,y, 
                    alpha=1.,
                    edgecolor='0.', 
                    c=[color])
        
    ax[0].plot(x_lim, y_lim, 'k--', lw=2)
    ax[0].set_xlim(x_lim)
    ax[0].set_xticks(np.arange(0,8,2))
    ax[0].set_xlabel(Xlabel, fontsize=font_size)
    ax[0].set_ylim(y_lim)
    ax[0].set_yticks(np.arange(0,8,2))
    ax[0].set_ylabel(Ylabel, fontsize=font_size)
    ax[0].plot(x_lim, y_lim, '--', lw=2, c='0.')
    cbar = fig.colorbar(sm, ax=ax[0])
    cbar.set_ticklabels(roman_numbers)
    cbar.set_label('Repetition', fontsize=font_size)
    ax[0].set_title(model_name, fontsize=font_size)

    # RMSE plot
    ax[1].scatter(np.arange(1,len(bptlabels[1:])+2), np.mean(rmse_cv_scores, axis=1), label='training', facecolor='blue', edgecolor='k', s=25, zorder=4)
    ax[1].plot(np.arange(1,len(bptlabels[1:])+2), np.mean(rmse_cv_scores, axis=1), c='k', ls='--', zorder=3)
    ax[1].fill_between(
        np.arange(1,len(bptlabels[1:])+2), 
        np.mean(rmse_cv_scores, axis=1)-np.std(rmse_cv_scores, axis=1), 
        np.mean(rmse_cv_scores, axis=1)+np.std(rmse_cv_scores, axis=1),
        alpha=0.3)

    ax[1].scatter(np.arange(2,len(bptlabels[1:])+2), rmse_scores, marker='d', 
                facecolor='red', edgecolor='k',
                color='0.', s=30, label='prediction', zorder=5)
    ax[1].plot(np.arange(2,len(bptlabels[1:])+2), rmse_scores, c='k', ls='-', zorder=3)
    ax[1].legend(fontsize=font_size)
    ax[1].set_ylim(0.4, 1.1)
    ax[1].set_yticks(np.arange(0.4,1.21,0.2))
    ax[1].set_xticks(np.arange(1,len(bptlabels)+1))
    ax[1].set_xticklabels(bptlabels)
    ax[1].grid(ls='--', alpha=.5, zorder=1)
    ax[1].set_ylabel('RMSE', fontsize=font_size)
    ax[1].set_xlabel('Repetition', fontsize=font_size)
    ax[1].set_title('Scores', fontsize=font_size)

    # Add a faded colored rectangle to highlight 'rep0' (hyperparameter tuning)
    highlight_color = 'peachpuff' 
    alpha = .5  
    rect = patches.Rectangle((0.5, plt.ylim()[0]), 1, plt.ylim()[1] - plt.ylim()[0],
                            linewidth=0, edgecolor='0.', facecolor=highlight_color, 
                            alpha=alpha, zorder=1)
    ax[1].add_patch(rect)

    fig.tight_layout()
    
    return fig, ax
