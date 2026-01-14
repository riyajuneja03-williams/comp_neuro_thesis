import numpy as np
from matplotlib import pyplot as plt
import os
import pandas as pd
import seaborn as sns
import sys

def raster_plot(trains, path):
    """
    Create raster plot for 1 or more spike trains.

    Parameters
    ----------
    trains: np.array
        array(s) of spike times
    path: string
        where to save figure

    Returns
    -------
    saves raster plot

    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.eventplot(trains, color="k")
    ax.set_xlabel("Time")
    ax.set_ylabel("Train Number")
    ax.set_title("Raster Plot")
    fig_path = os.path.join(path, 'raster_plot.png')
    plt.savefig(fig_path)
    plt.close()
    
def create_heatmap(indep1, indep2, dep, fig_name):
    """
    Create heatmap.

    Parameters
    ----------
    indep1: string
        independent variable to plot
    indep2: string
        independent variable to plot
    dep: string
        dependent variable to plot
    fig_name:
        name to save figure as

    Returns
    -------
    saves heatmap

    """
    # extract data from dataframe
    frame_path = os.path.join('thesis', 'data_frame.csv')
    df = pd.read_csv(frame_path)

    # create dataframe
    df_pivoted = df.pivot_table(index=indep2, columns=indep1, values=dep, aggfunc='mean')

    # plot 
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    ax = sns.heatmap(df_pivoted, ax=ax, cmap = 'viridis', cbar_kws={'label': dep})
    plt.xlabel(str(indep1))
    plt.ylabel(str(indep2))
    fig_path = os.path.join('thesis', fig_name)
    plt.savefig(fig_path)
    plt.close()

def create_hist(var, fig_name):
    """
    Create histogram.

    Parameters
    ----------
    var: string
        variable to plot
    fig_name:
        name to save figure as

    Returns
    -------
    saves histogram

    """
    frame_path = os.path.join('thesis', 'data_frame.csv')
    df = pd.read_csv(frame_path)
    var = list(df[str(var)])
    plt.figure(figsize=(10,6))
    plt.hist(var, bins=20, color='blue', alpha=0.5, label = str(var))
    plt.xlabel(str(var))
    plt.ylabel('Count')
    plt.legend()
    fig_path = os.path.join('thesis', str(fig_name))
    plt.savefig(fig_path)
    plt.close()

def create_frcv_scatterplot(var, fig_name):
    """
    Create scatterplot.

    Parameters
    ----------
    var: string
        variable to plot
    fig_name:
        name to save figure as

    Returns
    -------
    saves scatterplot

    """
    frame_path = os.path.join('thesis', 'data_frame.csv')
    df = pd.read_csv(frame_path)
    plt.figure(figsize=(10,6))
    sns.scatterplot(data = df, x = "actual_rate", y = "cv", hue = str(var))
    plt.xlabel("firing rate")
    plt.ylabel("coefficient of variation")
    fig_path = os.path.join('thesis', fig_name)
    plt.savefig(fig_path)
    plt.close()
