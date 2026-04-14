import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
import os
import pandas as pd
import seaborn as sns
import sys
from scipy.stats import sem

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
    
def create_heatmap(indep1, indep2, dep, fig_name, T):
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
    df = df[df["T"] == T]

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

def create_hist(var, fig_name, log_bool, T, frame_path):
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
    df = pd.read_csv(frame_path)
    df = df[df["T"] == T]
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

def create_frcv_scatterplot(var, fig_name, T, frame_path):
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
    df = pd.read_csv(frame_path)

    df = df[df["T"] == T]

    plt.figure(figsize=(10,6))
    sns.scatterplot(data = df, x = "actual_rate", y = "cv", hue = str(var))

    plt.xlabel("firing rate")
    plt.ylabel("coefficient of variation")

    fig_path = os.path.join('thesis', fig_name)
    plt.savefig(fig_path)
    plt.close()

def create_lin_reg_sp(fig_name, T, frame_path):
    df = pd.read_csv(frame_path)
    df = df[df["T"] == T]

    plt.figure(figsize=(10,6))
    sns.regplot(data = df, x = "actual_rate", y = "cv", scatter=True, ci=None)
    
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
    unified_bursts = []

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
    
    unified_path = os.path.join(path_name, 'unified_bursts.txt')
    with open(unified_path, 'r') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            bursts = [burst.strip() for burst in line.split(',') if burst.strip() != '']
            unified_bursts.append([float(burst) for burst in bursts])
    
    # set up parameters
    fig, ax = plt.subplots(figsize=(7, 2), dpi=300, tight_layout=True)
    burst_colors = ["#1f78b4", "#e7298a"]
    train_color = "#bdbdbd"
    marker = '|'
    size = 100

    # plot original train
    sns.scatterplot(
        x=train,
        y = [0] * len(train),
        marker = marker,
        s = size,
        color = train_color,
        ax=ax,
        linewidth=1,
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
    plot_bursts(ax, unified_bursts, y_row=5, burst_colors=burst_colors, marker=marker, size=size)

    # label figure
    ax.set_yticks([0, 1, 2, 3, 4, 5])
    ax.set_yticklabels(["original", "poisson surprise", "max interval", "logISI", "CMA", "unified method"])
    ax.set_xlabel("Time")
    ax.set_ylim(-0.25, 5.25)
    if len(train) > 0:
        ax.set_xlim(min(train), max(train))

    fig.tight_layout()

    fig_path = os.path.join(path_name, "compare_methods_raster.png")
    fig.savefig(fig_path)
    plt.close()

def pd_compare_methods(train_num):
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
    unified_bursts = []

    train_dir = f"train_{train_num:03d}"
    path_name = os.path.join('thesis', 'pd_data', train_dir)

    # get spikes and bursts from files
    spikes_path = os.path.join(path_name, 'spikes.txt')
    with open(spikes_path, 'r') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            train.append(float(line))

    ps_path = os.path.join(path_name, 'ps_bursts.txt')
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
    
    unified_path = os.path.join(path_name, 'unified_bursts.txt')
    with open(unified_path, 'r') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            bursts = [burst.strip() for burst in line.split(',') if burst.strip() != '']
            unified_bursts.append([float(burst) for burst in bursts])
    
    # set up parameters
    fig, ax = plt.subplots(figsize=(7, 2), dpi=300, tight_layout=True)
    burst_colors = ["#1f78b4", "#e7298a"]
    train_color = "#bdbdbd"
    marker = '|'
    size = 100

    # plot original train
    sns.scatterplot(
        x=train,
        y = [0] * len(train),
        marker = marker,
        s = size,
        color = train_color,
        ax=ax,
        linewidth=1,
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
    plot_bursts(ax, unified_bursts, y_row=5, burst_colors=burst_colors, marker=marker, size=size)

    # label figure
    ax.set_yticks([0, 1, 2, 3, 4, 5])
    ax.set_yticklabels(["original", "poisson surprise", "max interval", "logISI", "CMA", "unified method"])
    ax.set_xlabel("Time")
    ax.set_ylim(-0.25, 5.25)
    if len(train) > 0:
        ax.set_xlim(min(train), max(train))

    fig.tight_layout()

    fig_path = os.path.join(path_name, "pd_compare_methods_raster.png")
    fig.savefig(fig_path)
    plt.close()

def roc_curves():
    """
    Plot 1-specificity vs sensitivity to compare method 
    """

    methods = ["PS", "MI", "LogISI", "CMA", "Unified"]
    sens = {method: [] for method in methods}
    oneminusspec = {method: [] for method in methods}
    params = {
        "T": [],
        "D": [],
        "train_rate": [],
        "predicted_burst_rate": [],
        "single_burst_rate": []
    }

    # read in values
    for param_num in range(150):
        for train_num in range(100):
            param_name = f'param_{param_num:04d}'
            train_name = f'train_{train_num:03d}'
            path1_name = os.path.join('thesis', param_name, train_name, 'analysis.txt')
            path2_name = os.path.join('thesis', param_name, train_name, 'metadata.txt')

            metadata = {}
            wanted = ["T", "D", "train_rate", "predicted_burst_rate", "single_burst_rate"]

            with open(path1_name, 'r') as file:
                for line in file:
                    if ':' not in line:
                        continue
                    key, value = line.strip().split(':', 1)
                    value = float(value)

                    method, stat = key.split(", ")
                    if stat == "sensitivity":
                        sens[method].append(value)
                    else:
                        oneminusspec[method].append(value)
            
            with open(path2_name, 'r') as file:
                for line in file:
                    if ':' not in line:
                        continue
                    key, value = line.strip().split(':', 1)
                    key = key.strip()
                    if key not in wanted:
                        continue
                    metadata[key] = float(value)

            for key in wanted:
                params[key].append(metadata[key])                    
    
    # graph 1: 1-specificity vs sensitivity, all values color coded by method
    plt.figure()

    new_methods = ["PS", "MI", "LogISI", "CMA"]

    for method in new_methods:
        plt.scatter(
            oneminusspec[method], 
            sens[method],
            label=method,
            alpha = 0.25
        )
    
    plt.xlabel("1-Specificity")
    plt.ylabel("Sensitivity")
    plt.title("Analysis of Burst Detection Method Performance")
    plt.legend()

    fig_path = os.path.join('thesis', "method_analysis.png")
    plt.savefig(fig_path)
    plt.close()

    # graph 2: 1-specificity vs sensitivity, all values for unified approach

    plt.figure()
    plt.scatter(
        oneminusspec["Unified"],
        sens["Unified"],
        label="Unified",
        alpha = 0.5
    )
    
    plt.xlabel("1-Specificity")
    plt.ylabel("Sensitivity")
    plt.title("Analysis of Burst Detection Method Performance")
    plt.legend()

    fig_path = os.path.join('thesis', "unified_analysis.png")
    plt.savefig(fig_path)
    plt.close()

    # graphs 4-6: 1-specificity vs sensitivity, all values for each method

    plt.figure()
    plt.scatter(
        oneminusspec["PS"],
        sens["PS"],
        label="PS",
        alpha = 0.5
    )
    
    plt.xlabel("1-Specificity")
    plt.ylabel("Sensitivity")
    plt.title("Analysis of Burst Detection Method Performance")
    plt.legend()

    fig_path = os.path.join('thesis', "ps_analysis.png")
    plt.savefig(fig_path)
    plt.close()

    plt.figure()
    plt.scatter(
        oneminusspec["MI"],
        sens["MI"],
        label="MI",
        alpha = 0.5
    )
    
    plt.xlabel("1-Specificity")
    plt.ylabel("Sensitivity")
    plt.title("Analysis of Burst Detection Method Performance")
    plt.legend()

    fig_path = os.path.join('thesis', "mi_analysis.png")
    plt.savefig(fig_path)
    plt.close()

    plt.figure()
    plt.scatter(
        oneminusspec["LogISI"],
        sens["LogISI"],
        label="LogISI",
        alpha = 0.5
    )
    
    plt.xlabel("1-Specificity")
    plt.ylabel("Sensitivity")
    plt.title("Analysis of Burst Detection Method Performance")
    plt.legend()

    fig_path = os.path.join('thesis', "logisi_analysis.png")
    plt.savefig(fig_path)
    plt.close()

    plt.figure()
    plt.scatter(
        oneminusspec["CMA"],
        sens["CMA"],
        label="CMA",
        alpha = 0.5
    )
    
    plt.xlabel("1-Specificity")
    plt.ylabel("Sensitivity")
    plt.title("Analysis of Burst Detection Method Performance")
    plt.legend()

    fig_path = os.path.join('thesis', "cma_analysis.png")
    plt.savefig(fig_path)
    plt.close()

    # graph 7-11: panels with 2 histograms, average shift in sensitivity and average shift in 1-specificity, for each BD method → unified and then average → unified
    # want all shifts in sensitivity to be + and 1-specificity to be -

    sens_shift = []
    spec_shift = []

    sens_shift_by_method = {method: [] for method in new_methods}
    spec_shift_by_method = {method: [] for method in new_methods}

    for method in new_methods:
        for i in range(len(sens[method])):
            sens_val = sens["Unified"][i] - sens[method][i]
            spec_val = oneminusspec["Unified"][i] - oneminusspec[method][i]
            
            sens_shift.append(sens_val)
            spec_shift.append(spec_val)

            sens_shift_by_method[method].append(sens_val)
            spec_shift_by_method[method].append(spec_val)
    
    fig, axes = plt.subplots(1, 2)

    sns.histplot(sens_shift, stat="probability", kde=True, ax=axes[0])
    axes[0].set_title(f"Shift in sensitivity\nmean={np.mean(sens_shift):.3f}, median={np.median(sens_shift):.3f}")

    sns.histplot(spec_shift, stat="probability", kde=True, ax=axes[1])
    axes[1].set_title(f"Shift in 1 - specificity\nmean={np.mean(spec_shift):.3f}, median={np.median(spec_shift):.3f}")

    plt.tight_layout()
    fig_path = os.path.join('thesis', "shift_analysis.png")
    plt.savefig(fig_path)
    plt.close()

    for method in new_methods:
        fig, axes = plt.subplots(1, 2)

        sns.histplot(sens_shift_by_method[method], stat="probability", kde=True, ax=axes[0])
        axes[0].set_title(f"{method} sensitivity shift\nmean={np.mean(sens_shift_by_method[method]):.3f}, median={np.median(sens_shift_by_method[method]):.3f}")

        sns.histplot(spec_shift_by_method[method], stat="probability", kde=True, ax=axes[1])
        axes[1].set_title(f"{method} 1-spec shift\nmean={np.mean(spec_shift_by_method[method]):.3f}, median={np.median(spec_shift_by_method[method]):.3f}")

        plt.tight_layout()

        fig_path = os.path.join('thesis', f"{method}_shift_analysis.png")
        plt.savefig(fig_path)
        plt.close(fig)

    # graphs 12-21: sensitivity/1-specificity vs each parameter
    params_to_plot = ["T", "D", "train_rate", "single_burst_rate"]

    stats = [
        ("sens", sens, "Sensitivity"),
        ("spec", oneminusspec, "1-Specificity"),
    ]

    # discrete parameters
    for param in params_to_plot:
        X = np.array(params[param])

        for stat, dict, ylabel in stats:
            plt.figure()

            for method in new_methods:
                vals = np.array(dict[method])
                x_vals = np.unique(X)
                means = [vals[X == x].mean() for x in x_vals]
                sems = [sem(vals[X == x]) for x in x_vals]

                plt.errorbar(x_vals, means, yerr=sems, fmt='o', capsize=4, label=method)

            plt.xlabel(param)
            plt.ylabel(ylabel)
            plt.title("Analysis of Burst Detection Method Performance")
            plt.legend()

            fig_path = os.path.join('thesis', f"{param}_{stat}_analysis.png")
            plt.savefig(fig_path)
            plt.close()

    # predicted burst rate
    for stat, dict, ylabel in stats:
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))

        for i, method in enumerate(new_methods):
            ax = axes.flatten()[i]
            ax.scatter(
                params["predicted_burst_rate"],
                dict[method],
                alpha=0.25
            )
            ax.set_title(method)
            ax.set_xlabel("predicted burst rate")
            ax.set_ylabel(ylabel)

        plt.tight_layout()

        fig_path = os.path.join('thesis', f"predburstrate_{stat}_analysis.png")
        plt.savefig(fig_path)
        plt.close()