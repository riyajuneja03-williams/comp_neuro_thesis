import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
import os
import pandas as pd
import seaborn as sns
import sys
from scipy.stats import sem

sns.set_context("talk")
mpl.rcParams.update({'font.size': 8})

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
    fig, ax = plt.subplots(1, 1, figsize=(10, 6), dpi=300)
    ax.eventplot(trains, color="k")
    ax.set_xlabel("Time")
    ax.set_ylabel("Train Number")
    ax.set_title("Raster Plot")
    fig_path = os.path.join(path, 'raster_plot.png')

    plt.tight_layout()
    plt.savefig(fig_path, bbox_inches='tight')
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
    fig, ax = plt.subplots(1, 1, figsize=(14, 6), dpi=300)
    ax = sns.heatmap(df_pivoted, ax=ax, cmap='viridis', cbar_kws={'label': dep.replace("_", " ")})
    plt.xlabel(str(indep1).replace("_", " "))
    plt.ylabel(str(indep2).replace("_", " "))
    fig_path = os.path.join('thesis', fig_name)
    plt.tight_layout()
    plt.savefig(fig_path, bbox_inches='tight')
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
    data = df[var]
    mean = data.mean()
    std = data.std()
    num = len(data)

    plt.figure(figsize=(10,6), dpi=300)
    if log_bool:
        sns.histplot(df, x=str(var), stat="probability", edgecolor="w", log_scale=True)
        plt.xlabel(f"log({var.replace('_', ' ')})")
        plt.ylabel('Probability')
    else:
        sns.histplot(df, x=str(var), stat="probability", edgecolor="w")
        plt.xlabel(str(var).replace("_", " "))
        plt.ylabel('Probability')
    plt.axvline(mean, linestyle='--', label=f"mean = {mean:.2f} ± {std:.2f}, n = {len(data)}")
    plt.legend()
    fig_path = os.path.join('thesis', str(fig_name))
    plt.tight_layout()
    plt.savefig(fig_path, bbox_inches='tight')
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
    T:
        time
    frame_path:
        path of data frame

    Returns
    -------
    saves scatterplot

    """
    df = pd.read_csv(frame_path)

    df = df[df["T"] == T]

    plt.figure(figsize=(10,6), dpi=300)
    sns.scatterplot(data = df, x = "actual_rate", y = "cv", hue = str(var))

    plt.xlabel("firing rate")
    plt.ylabel("coefficient of variation")

    fig_path = os.path.join('thesis', fig_name)
    plt.tight_layout()
    plt.savefig(fig_path, bbox_inches='tight')
    plt.close()

def create_lin_reg_sp(fig_name, T, frame_path):
    """
    Create linear regression scatterplot.

    Parameters
    ----------
    fig_name:
        name to save figure as
    T:
        time
    frame_path:
        path of data frame

    Returns
    -------
    saves scatterplot

    """
    df = pd.read_csv(frame_path)
    df = df[df["T"] == T]

    plt.figure(figsize=(10,6), dpi=300)
    sns.regplot(data = df, x = "actual_rate", y = "cv", scatter=True, ci=95)
    
    plt.xlabel("firing rate")
    plt.ylabel("coefficient of variation")

    fig_path = os.path.join('thesis', fig_name)
    plt.tight_layout()
    plt.savefig(fig_path, bbox_inches='tight')
    plt.close()

def compare_methods(param_num, train_num, unified):
    """
    Create raster plot comparing methods.

    Parameters
    ----------
    param_num : integer
        parameter number of interest
    train_num : integer
        train number of interest
    unified : bool
        include unified in plot?

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

    if not os.path.isdir(path_name):
        return

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
    
    if unified:
        unified_bursts = []
        unified_path = os.path.join(path_name, 'unified_bursts.txt')
        with open(unified_path, 'r') as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                bursts = [burst.strip() for burst in line.split(',') if burst.strip() != '']
                unified_bursts.append([float(burst) for burst in bursts])
    
    # set up parameters
    fig, ax = plt.subplots(figsize=(7, 2), dpi=300)
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
    if unified:
        plot_bursts(ax, unified_bursts, y_row=5, burst_colors=burst_colors, marker=marker, size=size)

    # label figure
    if unified:
        ax.set_yticks([0, 1, 2, 3, 4, 5])
        ax.set_yticklabels(["original", "poisson surprise", "max interval", "logISI", "CMA", "unified method"])
    else:
        ax.set_yticks([0, 1, 2, 3, 4])
        ax.set_yticklabels(["original", "poisson surprise", "max interval", "logISI", "CMA"])
    ax.set_xlabel("Time")
    ax.set_ylim(-0.25, 5.25)
    if len(train) > 0:
        ax.set_xlim(min(train), max(train))

    fig.tight_layout()

    fig_path = os.path.join(path_name, "compare_methods_raster.png")
    fig.savefig(fig_path)

    sens = {}
    oneminusspec = {}

    analysis_path = os.path.join(path_name, "analysis.txt")

    with open(analysis_path, "r") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue

            method_metric, value = line.split(":")
            method, metric = method_metric.rsplit("_", 1)

            method = method.strip()
            metric = metric.strip()
            value = float(value.strip())

            if metric == "sens":
                sens[method] = value
            elif metric == "1spec":
                oneminusspec[method] = value

    plt.close()
    return sens, oneminusspec

def compare_methods_sensspec(params, trains, fig_name):
    plt.figure()

    method_colors = {
        "PS": "#1f77b4",
        "MI": "#ff7f0e",
        "LogISI": "#2ca02c",
        "CMA": "#d62728",
        "Unified": "#9467bd"
    }

    labeled = set()

    for param in params:
        for train in trains:
            sens, spec = compare_methods(param, train, True)

            for method in sens:
                label = method if method not in labeled else None
                labeled.add(method)

                plt.scatter(
                    spec[method],
                    sens[method],
                    color=method_colors[method],
                    label=label
                )

    plt.xlabel("1-Specificity")
    plt.ylabel("Sensitivity")
    plt.title("Sensitivity vs. 1-Specificity")
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join("thesis", fig_name), bbox_inches='tight')
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
    fig, ax = plt.subplots(figsize=(7, 2), dpi=300)
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

            if not os.path.exists(path1_name) or not os.path.exists(path2_name):
                continue

            metadata = {}
            wanted = ["T", "D", "train_rate", "predicted_burst_rate", "single_burst_rate"]

            with open(path1_name, 'r') as file:
                for line in file:
                    if ':' not in line:
                        continue

                    key, value = line.strip().split(':', 1)
                    key = key.strip()
                    value = value.strip()

                    if "_" not in key:
                        continue

                    if value == "None":
                        continue

                    value = float(value)

                    method, stat = key.rsplit("_", 1)

                    if stat == "sens":
                        sens[method].append(value)
                    elif stat == "1spec":
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

    # graph 1: average 1-specificity vs average sensitivity, all methods
    plt.figure()

    new_methods = ["PS", "MI", "LogISI", "CMA"]

    for method in methods:
        sens_mean = np.nanmean(sens[method])
        sens_sd = np.nanstd(sens[method], ddof=1)
        spec_mean = np.nanmean(oneminusspec[method])
        spec_sd = np.nanstd(oneminusspec[method], ddof=1)

        plt.errorbar(
            spec_mean,
            sens_mean,
            xerr=spec_sd,
            yerr=sens_sd,
            fmt='o',
            label=method,
            capsize=4
        )

    plt.xlabel("1-Specificity")
    plt.ylabel("Sensitivity")
    plt.title("Analysis of Burst Detection Method Performance")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend()

    fig_path = os.path.join('thesis', "method_analysis.png")
    plt.tight_layout()
    plt.savefig(fig_path, bbox_inches='tight')
    plt.close()

    # graph 2: 1-specificity vs sensitivity, all values for each method
    for method in methods:
        sens_mean = np.nanmean(sens[method])
        sens_sd = np.nanstd(sens[method], ddof=1)
        spec_mean = np.nanmean(oneminusspec[method])
        spec_sd = np.nanstd(oneminusspec[method], ddof=1)

        plt.figure()
        plt.scatter(
            oneminusspec[method],
            sens[method],
            label=method,
            alpha=0.1
        )

        plt.errorbar(
            spec_mean,
            sens_mean,
            xerr=spec_sd,
            yerr=sens_sd,
            fmt='o',
            color='black',
            capsize=4,
            label="Mean ± SD"
        )

        plt.xlabel("1-Specificity")
        plt.ylabel("Sensitivity")
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.title(method)

        plt.text(
            0.05, 0.95,
            f"Sens = {sens_mean:.3f} ± {sens_sd:.3f}\n1-Spec = {spec_mean:.3f} ± {spec_sd:.3f}",
            transform=plt.gca().transAxes,
            va='top'
        )

        plt.legend()

        fig_path = os.path.join('thesis', f"{method.lower()}_analysis.png")
        plt.tight_layout()
        plt.savefig(fig_path, bbox_inches='tight')
        plt.close()

    # graph 7-11: panels with 2 histograms, average shift in sensitivity and average shift in 1-specificity
    sens_shift = []
    spec_shift = []

    sens_shift_by_method = {method: [] for method in new_methods}
    spec_shift_by_method = {method: [] for method in new_methods}
    sens_improve_by_method = {method: [] for method in new_methods}
    spec_improve_by_method = {method: [] for method in new_methods}
    sens_change_by_method = {method: [] for method in new_methods}
    spec_change_by_method = {method: [] for method in new_methods}

    for method in new_methods:
        for i in range(len(sens[method])):
            sens_val = sens["Unified"][i] - sens[method][i]
            spec_val = oneminusspec["Unified"][i] - oneminusspec[method][i]
            
            sens_shift.append(sens_val)
            spec_shift.append(spec_val)

            sens_shift_by_method[method].append(sens_val)
            spec_shift_by_method[method].append(spec_val)

            if sens_val > 0:
                sens_improve_by_method[method].append(sens_val)
            if sens_val != 0:
                sens_change_by_method[method].append(sens_val)
            
            if spec_val < 0:
                spec_improve_by_method[method].append(spec_val)
            if spec_val != 0:
                spec_change_by_method[method].append(spec_val)

    labels = new_methods + ["Avg"]

    sens_ratio = [len(sens_improve_by_method[m]) / len(sens_change_by_method[m]) for m in new_methods]
    spec_ratio = [len(spec_improve_by_method[m]) / len(spec_change_by_method[m]) for m in new_methods]
    sens_ratio.append(sum(len(sens_improve_by_method[m]) for m in new_methods) / sum(len(sens_change_by_method[m]) for m in new_methods))
    spec_ratio.append(sum(len(spec_improve_by_method[m]) for m in new_methods) /sum(len(spec_change_by_method[m]) for m in new_methods))

    x = np.arange(len(labels))

    plt.figure()
    plt.scatter(x, sens_ratio, label="Sensitivity")
    plt.scatter(x, spec_ratio, label="1-Specificity")

    plt.xticks(x, labels)
    plt.ylim(0, 1)
    plt.ylabel("n improved / n changed")
    plt.xlabel("Unified shift")
    plt.legend()

    plt.tight_layout()
    fig_path = os.path.join('thesis', "improved_change_analysis.png")
    plt.savefig(fig_path, bbox_inches='tight')
    plt.close()

    fig, axes = plt.subplots(1, 2)

    sns.histplot(sens_shift, stat="probability", kde=True, ax=axes[0])
    axes[0].set_title("Shift in sensitivity")
    axes[0].text(
        0.05, 0.95,
        f"mean = {np.nanmean(sens_shift):.3f} ± {np.nanstd(sens_shift):.3f}",
        transform=axes[0].transAxes,
        va='top'
    )

    sns.histplot(spec_shift, stat="probability", kde=True, ax=axes[1])
    axes[1].set_title("Shift in 1 - specificity")
    axes[1].text(
        0.05, 0.95,
        f"mean = {np.nanmean(spec_shift):.3f} ± {np.nanstd(spec_shift):.3f}",
        transform=axes[1].transAxes,
        va='top'
    )

    plt.tight_layout()
    fig_path = os.path.join('thesis', "shift_analysis.png")
    plt.savefig(fig_path, bbox_inches='tight')
    plt.close()

    for method in new_methods:
        fig, axes = plt.subplots(1, 2)

        sns.histplot(sens_shift_by_method[method], stat="probability", kde=True, ax=axes[0])
        axes[0].set_title(f"{method} sensitivity shift")
        axes[0].text(
            0.05, 0.95,
            f"mean = {np.nanmean(sens_shift_by_method[method]):.3f} ± {np.nanstd(sens_shift_by_method[method]):.3f}",
            transform=axes[0].transAxes,
            va='top'
        )

        sns.histplot(spec_shift_by_method[method], stat="probability", kde=True, ax=axes[1])
        axes[1].set_title(f"{method} 1-spec shift")
        axes[1].text(
            0.05, 0.95,
            f"mean = {np.nanmean(spec_shift_by_method[method]):.3f} ± {np.nanstd(spec_shift_by_method[method]):.3f}",
            transform=axes[1].transAxes,
            va='top'
        )

        plt.tight_layout()

        fig_path = os.path.join('thesis', f"{method}_shift_analysis.png")
        plt.savefig(fig_path, bbox_inches='tight')
        plt.close(fig)

    # graphs 12-21: sensitivity/1-specificity vs each parameter
    params_to_plot = ["T", "D", "train_rate", "single_burst_rate"]

    stats = [
        ("sens", sens, "Sensitivity"),
        ("spec", oneminusspec, "1-Specificity"),
    ]

    for param in params_to_plot:
        X = np.array(params[param])

        for stat, stat_dict, ylabel in stats:
            plt.figure()

            title_lines = []

            for method in new_methods:
                vals = np.array(stat_dict[method])
                x_vals = np.unique(X)

                means = [np.nanmean(vals[X == x]) for x in x_vals]

                plt.scatter(x_vals, means, label=method)

                overall_mean = np.nanmean(vals)
                overall_sd = np.nanstd(vals, ddof=1)
                n = np.sum(~np.isnan(vals))

                title_lines.append(
                    f"{method}: {overall_mean:.3f} ± {overall_sd:.3f} (n={n})"
                )

            plt.xlabel(param.replace("_", " "))
            plt.ylabel(ylabel)
            plt.ylim(0, 1)
            plt.title(f"{ylabel} vs {param}")

            plt.text(
                0.05, 0.95,
                "\n".join(title_lines),
                transform=plt.gca().transAxes,
                va='top'
            )

            plt.legend()

            fig_path = os.path.join('thesis', f"{param}_{stat}_analysis.png")
            plt.tight_layout()
            plt.savefig(fig_path, bbox_inches='tight')
            plt.close()

    # predicted burst rate
    for stat, stat_dict, ylabel in stats:
        fig, axes = plt.subplots(2, 2, figsize=(10, 8), dpi=300)

        for i, method in enumerate(new_methods):
            ax = axes.flatten()[i]
            vals = np.array(stat_dict[method])

            ax.scatter(
                params["predicted_burst_rate"],
                vals,
                alpha=0.25
            )

            overall_mean = np.nanmean(vals)
            overall_sd = np.nanstd(vals, ddof=1)
            n = np.sum(~np.isnan(vals))

            ax.set_title(method)
            ax.text(
                0.05, 0.95,
                f"{overall_mean:.3f} ± {overall_sd:.3f} (n={n})",
                transform=ax.transAxes,
                va='top'
            )

            ax.set_xlabel("predicted burst rate")
            ax.set_ylabel(ylabel)
            ax.set_ylim(0, 1)

        plt.tight_layout()

        fig_path = os.path.join('thesis', f"predburstrate_{stat}_analysis.png")
        plt.savefig(fig_path, bbox_inches='tight')
        plt.close()


def metric_sp(frame_path, fig_name, metric, is_pd=False, frame_path2=None):
    """
    Create scatterplot.

    Parameters
    ----------
    fig_name:
        name to save figure as
    frame_path:
        path of data frame
    metric:
        metric to plot
    is_pd: 
        is the data in 2 groups?
    frame_path2:
        if so, second dataframe

    Returns
    -------
    saves scatterplot

    """

    df = pd.read_csv(frame_path)

    methods = ["PS", "MI", "LogISI", "CMA", "Unified"]
    cols = [
        f"ps_{metric}",
        f"mi_{metric}",
        f"logisi_{metric}",
        f"cma_{metric}",
        f"unified_{metric}"
    ]

    means = [df[col].mean() for col in cols]
    stds = [df[col].std() for col in cols]

    plt.figure(figsize=(8, 6), dpi=300)

    if not is_pd:
        plt.errorbar(methods, means, yerr=stds, fmt='o', capsize=5)

        for i in range(len(methods)):
            plt.text(
                i,
                means[i],
                f"{means[i]:.2f} ± {stds[i]:.2f}",
                ha='center',
                va='bottom'
            )

    else:
        df2 = pd.read_csv(frame_path2)

        means2 = [df2[col].mean() for col in cols]
        stds2 = [df2[col].std() for col in cols]

        x = np.arange(len(methods))

        plt.errorbar(x - 0.08, means, yerr=stds, fmt='o', capsize=5, label="Healthy")
        plt.errorbar(x + 0.08, means2, yerr=stds2, fmt='o', capsize=5, label="Dopamine-depleted")

        for i in range(len(methods)):
            plt.text(
                x[i] - 0.08,
                means[i],
                f"{means[i]:.2f} ± {stds[i]:.2f}",
                ha='right',
                va='bottom',
                fontsize=8
            )

            plt.text(
                x[i] + 0.08,
                means2[i],
                f"{means2[i]:.2f} ± {stds2[i]:.2f}",
                ha='left',
                va='bottom',
                fontsize=8
            )

        plt.xticks(x, methods)
        plt.legend()

    plt.xlabel("Method")
    plt.ylabel(metric.replace("_", " "))
    plt.tight_layout()

    fig_path = os.path.join("thesis", fig_name)
    plt.savefig(fig_path, bbox_inches="tight")
    plt.close()