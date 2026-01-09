# heatmaps of parameters of interest

import numpy as np
from matplotlib import pyplot as plt
import synspiketrain
import stats
import os
import pandas as pd
import seaborn as sns

# set desired time and time step
T = 1
dt = 1e-3

# refractory parameters
tau_ref = 0
tau_burst = 0

# define parameter sets
rates = np.arange(2, 12, 4)
rates_list = rates.tolist()
prob_burst = [0.2, 0.2, 0.2, 0.5, 0.5, 0.5, 0.8, 0.8, 0.8]
prob_exit = [0.2, 0.5, 0.8, 0.2, 0.5, 0.8, 0.2, 0.5, 0.8]
params = []
probs = [[prob_burst[0], prob_exit[0]], [prob_burst[1], prob_exit[1]], [prob_burst[2], prob_exit[2]], [prob_burst[3], prob_exit[3]], [prob_burst[4], prob_exit[4]], [prob_burst[5], prob_exit[5]], [prob_burst[6], prob_exit[6]], [prob_burst[7], prob_exit[7]], [prob_burst[8], prob_exit[8]]]
for rate in rates_list:
    burst_rates = [2 * rate, 5 * rate, 10 * rate]
    for burst_rate in burst_rates:
        for prob in probs:
            burst_prob = prob[0]
            exit_prob = prob[1]
            params.append([rate, burst_rate, burst_prob, exit_prob])

# make directory for each parameter
for i, param in enumerate(params):

    param_name = 'param_{}'.format(i)
    dir_name = os.path.join('thesis', param_name)
    os.mkdir(dir_name)
    all_spikes = []

    # make subdirectory for each train
    for j in range(0, 2):
        subdir_name = 'train_{}'.format(j)
        path = os.path.join(dir_name, subdir_name)
        os.makedirs(path) 

        # generate trains
        trains, bursts = synspiketrain.poisson_burst(
            rate=param[0],  # rate
            burst_rate=param[1],  # burst rate
            T=T,  # cut off time
            tau_ref=tau_ref,  # spike refractory period
            tau_burst=tau_burst,  # burst refractory period
            prob_burst=param[2],  # probability of entering a burst
            prob_end=param[3],  # probability of exiting a burst
        )
        all_spikes.append(trains)

        # write train to file
        file_name = 'spikes.txt'
        spikes_path = os.path.join(path, file_name)
        np.savetxt(spikes_path, trains, fmt = "%f", newline="\n")

        # define metadata
        parameters = {
            "rate": param[0],  
            "burst_rate": param[1], 
            "T": T,  
            "tau_ref": tau_ref, 
            "tau_burst": tau_burst,  
            "prob_burst": param[2], 
            "prob_end": param[3]
        }
        spikestats, burststats = stats.calculate_statistics(trains, bursts, rate, T, burst_rate, prob_burst, prob_exit, tau_ref, tau_burst)

        # write metadata to file
        file_name2 = 'metadata.txt'
        data_path = os.path.join(path, file_name2)

        all_metadata = parameters | spikestats | burststats
        metadata_array = np.array(list(all_metadata.items()), dtype=object)
        np.savetxt(data_path, metadata_array, fmt = "%s", delimiter = ":")
    
        # raster plots
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.eventplot(trains, color="k")
        ax.set_xlabel("Time")
        ax.set_ylabel("Train Number")
        ax.set_title("Raster Plot")
        fig_path = os.path.join(path, 'raster plot')
        plt.savefig(fig_path)
    
    # master raster plot
    print(all_spikes)
    plt.figure(figsize=(10, 6))
    plt.eventplot(all_spikes, color="k")
    plt.xlabel("time (s)")
    plt.ylabel("train number")
    plt.title("master raster plot")
    master_fig_path = os.path.join(dir_name, 'master raster plot')
    plt.savefig(master_fig_path) 

# create pandas dataframe

"""
# heatmap: firing rate & burst rate as indep vars, FR & CV as dep vars (2 panels)
# create dataframes
data1 = {
    'predicted rates': rates,
    'predicted burst rates': burst_rates,
    'actual firing rates': frs
}
df1 = pd.DataFrame(data1)
df1_pivoted = df1.pivot_table(index='predicted burst rates', columns='predicted rates', values='actual firing rates', aggfunc='mean')

data2 = {
    'predicted rates': rates,
    'predicted burst rates': burst_rates,
    'coefficients of variation': cvs
}
df2 = pd.DataFrame(data2)
df2_pivoted = df2.pivot_table(index='predicted burst rates', columns='predicted rates', values='coefficients of variation', aggfunc='mean')

# plot 2 panels
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
ax1 = sns.heatmap(df1_pivoted, ax=ax1, cmap = 'viridis', cbar_kws={'label': 'actual firing rates'})
ax2 = sns.heatmap(df2_pivoted, ax=ax2, cmap = 'viridis', cbar_kws={'label': 'coefficients of variation'})
plt.xlabel("predicted rates")
plt.ylabel("predicted burst rates")
plt.show()

"""