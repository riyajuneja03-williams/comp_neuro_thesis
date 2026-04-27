import numpy as np
from matplotlib import pyplot as plt
import os
import pandas as pd
import seaborn as sns
import sys
import maxinterval
import cma
import logisi
import poissonsurprise
import pd_data
import unified_method
import stats

(T, N) = pd_data.return_params()

for i in range(0, N):
    # get spike train
    trains = []
    train_dir = f"train_{i:03d}"
    file_name = os.path.join('thesis', 'pd_data', train_dir, 'spikes.txt')
    
    with open(file_name, 'r') as file:
        for line in file:
            trains.append(float(line))
        
    # apply BD methods
    mi_bursts = maxinterval.max_interval(trains)
    ps_bursts = [burst for burst, surprise in poissonsurprise.poisson_surprise(trains, T=T)]
    cma_bursts = cma.cma_burst_detection(trains)
    logisi_bursts = logisi.log_isi(trains)
    unified_bursts = unified_method.unified_burst_detection(trains, T)
        
    # save detected bursts
    mi_path = os.path.join('thesis', 'pd_data', train_dir, 'mi_bursts.txt')
    with open(mi_path, "w") as file:
        for burst in mi_bursts:
            burst = np.array(burst)
            np.savetxt(file, burst[None, :], fmt = "%f", newline="\n", delimiter = ",")

    ps_path = os.path.join('thesis', 'pd_data', train_dir, 'ps_bursts.txt')
    with open(ps_path, "w") as file:
        for burst in ps_bursts:
            burst = np.array(burst)
            np.savetxt(file, burst[None, :], fmt = "%f", newline="\n", delimiter = ",")

    cma_path = os.path.join('thesis', 'pd_data', train_dir, 'cma_bursts.txt')
    with open(cma_path, "w") as file:
        for burst in cma_bursts:
            burst = np.array(burst)
            np.savetxt(file, burst[None, :], fmt = "%f", newline="\n", delimiter = ",")

    logisi_path = os.path.join('thesis', 'pd_data', train_dir, 'logisi_bursts.txt')
    with open(logisi_path, "w") as file:
        for burst in logisi_bursts:
            burst = np.array(burst)
            np.savetxt(file, burst[None, :], fmt = "%f", newline="\n", delimiter = ",")
    
    unified_path = os.path.join('thesis', 'pd_data', train_dir, 'unified_bursts.txt')
    with open(unified_path, "w") as file:
        for burst in unified_bursts:
            burst = np.array(burst)
            np.savetxt(file, burst[None, :], fmt = "%f", newline="\n", delimiter = ",")
                
    # calculate burst statistics for detected bursts
    _, mi_burststats = stats.calculate_statistics(trains, mi_bursts, T, None, None, None, None)
    stats_path = os.path.join('thesis', 'pd_data', train_dir, 'mi_stats.txt')

    mi_burststats_dict = {
        "mi_num_spikes": mi_burststats['num_spikes'], 
        "mi_burst_firing_rate": mi_burststats['burst_firing_rate'], 
        "mi_avg_ISI_within_bursts": mi_burststats['avg_ISI_within_bursts'], 
        "mi_burst_rate": mi_burststats['burst_rate'], 
        "mi_%_spikes_in_burst": mi_burststats['%_spikes_in_burst'], 
        "mi_%_time_spent_bursting": mi_burststats['%_time_spent_bursting'], 
        "mi_firing_rate_non_bursting": mi_burststats['firing_rate_non_bursting'], 
        "mi_burst_firing_rate_inc": mi_burststats['burst_firing_rate_inc']
    }

    # write stats to file
    miburststats_array = np.array(list(mi_burststats_dict.items()), dtype=object)
    np.savetxt(stats_path, miburststats_array, fmt = "%s", delimiter = ":")

    # do the same for the rest of the methods
    _, ps_burststats = stats.calculate_statistics(trains, ps_bursts, T, None, None, None, None)
    stats_path = os.path.join('thesis', 'pd_data', train_dir, 'ps_stats.txt')

    ps_burststats_dict = {
        "ps_num_spikes": ps_burststats['num_spikes'],
        "ps_burst_firing_rate": ps_burststats['burst_firing_rate'], 
        "ps_avg_ISI_within_bursts": ps_burststats['avg_ISI_within_bursts'], 
        "ps_burst_rate": ps_burststats['burst_rate'], 
        "ps_%_spikes_in_burst": ps_burststats['%_spikes_in_burst'], 
        "ps_%_time_spent_bursting": ps_burststats['%_time_spent_bursting'], 
        "ps_firing_rate_non_bursting": ps_burststats['firing_rate_non_bursting'], 
        "ps_burst_firing_rate_inc": ps_burststats['burst_firing_rate_inc']
    }

    psburststats_array = np.array(list(ps_burststats_dict.items()), dtype=object)
    np.savetxt(stats_path, psburststats_array, fmt="%s", delimiter=":")

    _, cma_burststats = stats.calculate_statistics(trains, cma_bursts, T, None, None, None, None)
    stats_path = os.path.join('thesis', 'pd_data', train_dir, 'cma_stats.txt')

    cma_burststats_dict = {
        "cma_num_spikes": cma_burststats['num_spikes'], 
        "cma_burst_firing_rate": cma_burststats['burst_firing_rate'], 
        "cma_avg_ISI_within_bursts": cma_burststats['avg_ISI_within_bursts'], 
        "cma_burst_rate": cma_burststats['burst_rate'], 
        "cma_%_spikes_in_burst": cma_burststats['%_spikes_in_burst'], 
        "cma_%_time_spent_bursting": cma_burststats['%_time_spent_bursting'], 
        "cma_firing_rate_non_bursting": cma_burststats['firing_rate_non_bursting'], 
        "cma_burst_firing_rate_inc": cma_burststats['burst_firing_rate_inc']
    }

    cmaburststats_array = np.array(list(cma_burststats_dict.items()), dtype=object)
    np.savetxt(stats_path, cmaburststats_array, fmt="%s", delimiter=":")

    _, logisi_burststats = stats.calculate_statistics(trains, logisi_bursts, T, None, None, None, None)
    stats_path = os.path.join('thesis', 'pd_data', train_dir, 'logisi_stats.txt')

    logisi_burststats_dict = {
        "logisi_num_spikes": logisi_burststats['num_spikes'], 
        "logisi_burst_firing_rate": logisi_burststats['burst_firing_rate'], 
        "logisi_avg_ISI_within_bursts": logisi_burststats['avg_ISI_within_bursts'], 
        "logisi_burst_rate": logisi_burststats['burst_rate'], 
        "logisi_%_spikes_in_burst": logisi_burststats['%_spikes_in_burst'], 
        "logisi_%_time_spent_bursting": logisi_burststats['%_time_spent_bursting'], 
        "logisi_firing_rate_non_bursting": logisi_burststats['firing_rate_non_bursting'], 
        "logisi_burst_firing_rate_inc": logisi_burststats['burst_firing_rate_inc']
    }

    logisiburststats_array = np.array(list(logisi_burststats_dict.items()), dtype=object)
    np.savetxt(stats_path, logisiburststats_array, fmt="%s", delimiter=":")

    spike_stats, unified_burststats = stats.calculate_statistics(trains, unified_bursts, T, None, None, None, None)
    stats_path = os.path.join('thesis', 'pd_data', train_dir, 'unified_stats.txt')

    unified_burststats_dict = {
        "unified_num_spikes": unified_burststats['num_spikes'], 
        "unified_burst_firing_rate": unified_burststats['burst_firing_rate'], 
        "unified_avg_ISI_within_bursts": unified_burststats['avg_ISI_within_bursts'], 
        "unified_burst_rate": unified_burststats['burst_rate'], 
        "unified_%_spikes_in_burst": unified_burststats['%_spikes_in_burst'], 
        "unified_%_time_spent_bursting": unified_burststats['%_time_spent_bursting'], 
        "unified_firing_rate_non_bursting": unified_burststats['firing_rate_non_bursting'], 
        "unified_burst_firing_rate_inc": unified_burststats['burst_firing_rate_inc']
    }

    unifiedburststats_array = np.array(list(unified_burststats_dict.items()), dtype=object)
    np.savetxt(stats_path, unifiedburststats_array, fmt="%s", delimiter=":")

    spike_stats_path = os.path.join('thesis', 'pd_data', train_dir, 'spike_stats.txt')
    spike_stats_array = np.array(list(spike_stats.items()), dtype=object)
    np.savetxt(spike_stats_path, spike_stats_array, fmt = "%s", delimiter = ":")

