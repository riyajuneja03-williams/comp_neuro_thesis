import numpy as np
from matplotlib import pyplot as plt
import os
import pandas as pd
import seaborn as sns
import sys
import synspiketrain
import poissonsurprise
import stats

(D, T, N, params) = synspiketrain.return_params()

for i, param in enumerate(params):
    for j in range(0, N):

        # get spike train
        trains = []
        param_name = f'param_{i:04d}'
        train_name = f'train_{j:03d}'
        file_name = os.path.join('thesis', param_name, train_name, 'spikes.txt')
        with open(file_name, 'r') as file:
            for line in file:
                trains.append(float(line))
        
        # apply PS
        ps_bursts = [burst for burst, surprise in poissonsurprise.poisson_surprise(trains)]
        
        # save detected bursts
        ps_path = os.path.join('thesis', param_name, train_name, 'poisson_bursts.txt')
        with open(ps_path, "w") as file:
            for burst in ps_bursts:
                np.savetxt(file, burst[None, :], fmt = "%f", newline="\n", delimiter = ",")
                
        # calculate burst statistics for detected bursts
        _, ps_burststats = stats.calculate_statistics(trains, ps_bursts, param[0], param[1], param[2], param[3], param[4])
        burst_stats_file = 'poisson_stats.txt'
        stats_path = os.path.join('thesis', param_name, train_name, 'poisson_stats.txt')

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

        # write stats to file
        psburststats_array = np.array(list(ps_burststats_dict.items()), dtype=object)
        np.savetxt(stats_path, psburststats_array, fmt = "%s", delimiter = ":")
