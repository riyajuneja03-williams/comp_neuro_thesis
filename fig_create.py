import numpy as np
import matplotlib as mpl
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
    df[indep1] = df[indep1].round(3)
    df[indep2] = df[indep2].round(3)
    df_pivoted = df.pivot_table(index=indep2, columns=indep1, values=dep, aggfunc='mean')

    # plot 
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    ax = sns.heatmap(df_pivoted, ax=ax, cmap = 'viridis', cbar_kws={'label': dep})
    plt.xlabel(str(indep1))
    plt.ylabel(str(indep2))
    fig_path = os.path.join('thesis', fig_name)
    plt.savefig(fig_path)
    plt.close()

def create_hist(var, fig_name, log_bool):
    """
    Create histogram.

    Parameters
    ----------
    var: string
        variable to plot
    fig_name:
        name to save figure as
    log_bool:
        boolean to determine whether take log of variable

    Returns
    -------
    saves histogram

    """
    frame_path = os.path.join('thesis', 'data_frame.csv')
    df = pd.read_csv(frame_path)
    plt.figure(figsize=(10,6))
    if log_bool:
        sns.histplot(df, x=str(var), stat="probability", edgecolor="w", log_scale=True)
        plt.xlabel(f"log({var})")
        plt.ylabel('Probability')
    else:
        sns.histplot(df, x=str(var), stat="probability", edgecolor="w")
        plt.xlabel(str(var))
        plt.ylabel('Probability')
    fig_path = os.path.join('thesis', str(fig_name))
    plt.savefig(fig_path)
    plt.close()

def create_frcv_scatterplot(var, fig_name, T=None):
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

    if T is not None:
        df = df[df["T"] == T]

    plt.figure(figsize=(10,6))
    sns.scatterplot(data = df, x = "actual_rate", y = "cv", hue = str(var))

    plt.xlabel("firing rate")
    plt.ylabel("coefficient of variation")

    fig_path = os.path.join('thesis', fig_name)
    plt.savefig(fig_path)
    plt.close()

def compare_methods(param_num, train_num):
    """
    Create raster plot comparing methods.

    Parameters
    ----------
    param_num : integer
        parameter number of interest
    train_num : integer
        train number of interest

    Returns
    -------
    saves raster plot comparing train & BD methods

    """
    train = []
    ps_bursts = []
    mi_bursts = []
    logisi_bursts = []
    cma_bursts = []

    param_name = f'param_{param_num:04d}'
    train_name = f'train_{train_num:03d}'
    path_name = os.path.join('thesis', param_name, train_name)

    # get spikes and bursts from files
    spikes_path = os.path.join(path_name, 'spikes.txt')
    with open(spikes_path, 'r') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            train.append(float(line))

    ps_path = os.path.join(path_name, 'poisson_bursts.txt')
    with open(ps_path, 'r') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            bursts = [burst.strip() for burst in line.split(',') if burst.strip() != '']
            ps_bursts.append([float(burst) for burst in bursts])
    
    mi_path = os.path.join(path_name, 'mi_bursts.txt')
    with open(mi_path, 'r') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            bursts = [burst.strip() for burst in line.split(',') if burst.strip() != '']
            mi_bursts.append([float(burst) for burst in bursts])
    
    logisi_path = os.path.join(path_name, 'logisi_bursts.txt')
    with open(logisi_path, 'r') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            bursts = [burst.strip() for burst in line.split(',') if burst.strip() != '']
            logisi_bursts.append([float(burst) for burst in bursts])
    
    cma_path = os.path.join(path_name, 'cma_bursts.txt')
    with open(cma_path, 'r') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            bursts = [burst.strip() for burst in line.split(',') if burst.strip() != '']
            cma_bursts.append([float(burst) for burst in bursts])
    
    # set up parameters
    fig, ax = plt.subplots(figsize=(8, 2))
    burst_colors = ["#1f78b4", "#e7298a"]
    train_color = "#bdbdbd"
    marker = '|'
    size = 25

    # plot original train
    sns.scatterplot(
        x=train,
        y = [0] * len(train),
        marker = marker,
        s = size,
        color = train_color,
        ax=ax,
        linewidth=2,
        legend = False,
        alpha = 0.4
    )

    def plot_bursts(ax, bursts, y_row, burst_colors, marker='|', size=100):
        """
        Plot bursts 

        Parameters
        ----------
        ax : ax
            axis
        bursts : list of lists
            list of lists of bursts
        y_row : integer
            integer representing which row 
            0 = original, 1 = PS, 2 = MI, 3 = logISI, 4 = CMA
        burst_colors : list
            color optons for bursts
        marker : char    
            to plot
        size : integer
            to plot

        Returns
        -------
        plots bursts

        """       
        # for each burst
        for i, burst in enumerate(bursts):
            col = burst_colors[i % 2]

            if len(burst) == 0: 
                continue
            
            # plot burst
            sns.scatterplot(
                x=burst,
                y=[y_row] * len(burst),
                marker = marker,
                s = size,
                color = col,
                ax=ax,
                legend = False
            )

    # call plot bursts on each burst detection method
    plot_bursts(ax, ps_bursts, y_row=1, burst_colors=burst_colors, marker=marker, size=size)
    plot_bursts(ax, mi_bursts, y_row=2, burst_colors=burst_colors, marker=marker, size=size)
    plot_bursts(ax, logisi_bursts, y_row=3, burst_colors=burst_colors, marker=marker, size=size)
    plot_bursts(ax, cma_bursts, y_row=4, burst_colors=burst_colors, marker=marker, size=size)

    # label figure
    ax.set_yticks([0, 1, 2, 3, 4])
    ax.set_yticklabels(["original", "poisson surprise", "max interval", "logISI", "CMA"])
    ax.set_xlabel("Time")
    ax.set_ylim(-0.25, 4.25)
    if len(train) > 0:
        ax.set_xlim(min(train), max(train))

    fig.tight_layout()

    fig_path = os.path.join(path_name, "compare_methods_raster.png")
    fig.savefig(fig_path)
    plt.close()