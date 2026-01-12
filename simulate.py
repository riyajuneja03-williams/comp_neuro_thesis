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
rates = np.arange(2, 102, 4)
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

# create dataframe
col_names = ['spikes', 'rate', 'T', 'tau_ref', 'tau_burst', 'prob_burst', 'prob_end', 'actual_rate', 'cv', 'ISI_dist', 'num_spikes', 'burst_firing_rate', 'avg_ISI_within_bursts', 'burst_rate', '%_spikes_in_burst', '%_time_spent_bursting', 'firing_rate_non_bursting', 'burst_firing_rate_inc']
df = pd.DataFrame(columns=col_names)

# make directory for each parameter
for i, param in enumerate(params):

    param_name = 'param_{}'.format(i)
    dir_name = os.path.join('thesis', param_name)
    os.mkdir(dir_name)
    all_spikes = []

    # make subdirectory for each train
    for j in range(0, 100):
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
        spikes_dict = {
            "spikes": trains
        }

        parameters = {
            "rate": param[0],  
            "burst_rate": param[1], 
            "T": T,  
            "tau_ref": tau_ref, 
            "tau_burst": tau_burst,  
            "prob_burst": param[2], 
            "prob_end": param[3]
        }
        spikestats, burststats = stats.calculate_statistics(trains, bursts, param[0], T, param[1], param[2], param[3], tau_ref, tau_burst)

        # write metadata to file
        file_name2 = 'metadata.txt'
        data_path = os.path.join(path, file_name2)

        all_metadata = parameters | spikestats | burststats

        metadata_array = np.array(list(all_metadata.items()), dtype=object)
        np.savetxt(data_path, metadata_array, fmt = "%s", delimiter = ":")

        fixed_params = {
            "rate": param[0],  
            "T": T,  
            "tau_ref": tau_ref, 
            "tau_burst": tau_burst,  
            "prob_burst": param[2], 
            "prob_end": param[3]
        }

        # save each spike train as a row in the dataframe
        frame_data = fixed_params | spikestats | burststats | spikes_dict
        df.loc[len(df)] = frame_data
    
        # raster plots
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.eventplot(trains, color="k")
        ax.set_xlabel("Time")
        ax.set_ylabel("Train Number")
        ax.set_title("Raster Plot")
        fig_path = os.path.join(path, 'raster plot.png')
        plt.savefig(fig_path)
    
    # master raster plot
    plt.figure(figsize=(10, 6))
    plt.eventplot(all_spikes, color="k")
    plt.xlabel("time (s)")
    plt.ylabel("train number")
    plt.title("master raster plot")
    master_fig_path = os.path.join(dir_name, 'master raster plot')
    plt.savefig(master_fig_path) 

# extract data from dataframe
all_rates = list(df['rate'])
all_burst_rates = list(df['burst_rate'])
all_prob_bursts = list(df['prob_burst'])
all_prob_ends = list(df['prob_end'])
all_frs = list(df['actual_rate'])
all_cvs = list(df['cv'])

# heatmap: firing rate & burst rate as indep vars, FR & CV as dep vars (2 panels)
# create dataframes
data1 = {
    'predicted rates': all_rates,
    'predicted burst rates': all_burst_rates,
    'actual firing rates': all_frs
}

df1 = pd.DataFrame(data1)
df1_pivoted = df1.pivot_table(index='predicted burst rates', columns='predicted rates', values='actual firing rates', aggfunc='mean')

data2 = {
    'predicted rates': all_rates,
    'predicted burst rates': all_burst_rates,
    'coefficients of variation': all_cvs
}
df2 = pd.DataFrame(data2)
df2_pivoted = df2.pivot_table(index='predicted burst rates', columns='predicted rates', values='coefficients of variation', aggfunc='mean')

# plot 2 panels
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
ax1 = sns.heatmap(df1_pivoted, ax=ax1, cmap = 'viridis', cbar_kws={'label': 'actual firing rates'})
ax2 = sns.heatmap(df2_pivoted, ax=ax2, cmap = 'viridis', cbar_kws={'label': 'coefficients of variation'})
plt.xlabel("predicted rates")
plt.ylabel("predicted burst rates")
fig_path = os.path.join('thesis', 'heatmap')
plt.savefig(fig_path)

# firing rate & cv histograms
plt.figure(figsize=(10,6))
plt.hist(all_frs, bins=20, color='blue', alpha=0.5, label = 'Firing rates')
plt.xlabel('Firing rate')
plt.ylabel('Count')
plt.legend()
fig_path = os.path.join('thesis', 'fr_hist.png')
plt.savefig(fig_path)

plt.figure(figsize=(10,6))
plt.hist(all_cvs, bins=20, color='blue', alpha=0.5, label = 'Coefficients of variation')
plt.xlabel('Coefficient of variation')
plt.ylabel('Count')
plt.legend()
fig_path = os.path.join('thesis', 'cv_hist.png')
plt.savefig(fig_path)

all_rates = list(df['rate'])
all_burst_rates = list(df['burst_rate'])
all_prob_bursts = list(df['prob_burst'])
all_prob_ends = list(df['prob_end'])

# fr vs cv scatterplots by different params
# rate
plt.figure(figsize=(10,6))
sns.scatterplot(data = df, x = "actual_rate", y = "cv", hue = "rate")
plt.xlabel("firing rate")
plt.ylabel("coefficient of variation")
fig_path = os.path.join('thesis', 'rate_scatterplot.png')
plt.savefig(fig_path)

# burst rate
plt.figure(figsize=(10,6))
sns.scatterplot(data = df, x = "actual_rate", y = "cv", hue = "burst_rate")
plt.xlabel("firing rate")
plt.ylabel("coefficient of variation")
fig_path = os.path.join('thesis', 'burstrate_scatterplot.png')
plt.savefig(fig_path)

# prob burst
plt.figure(figsize=(10,6))
sns.scatterplot(data = df, x = "actual_rate", y = "cv", hue = "prob_burst")
plt.xlabel("firing rate")
plt.ylabel("coefficient of variation")
fig_path = os.path.join('thesis', 'probburst_scatterplot.png')
plt.savefig(fig_path)

# prob end
plt.figure(figsize=(10,6))
sns.scatterplot(data = df, x = "actual_rate", y = "cv", hue = "prob_end")
plt.xlabel("firing rate")
plt.ylabel("coefficient of variation")
fig_path = os.path.join('thesis', 'probend_scatterplot.png')
plt.savefig(fig_path)
