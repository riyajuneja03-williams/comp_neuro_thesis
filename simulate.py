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
rates = np.arange(5, 80, 5)
rates_list = rates.tolist()
prob_burst = [0.25, 0.25, 0.75, 0.75]
prob_exit = [0.25, 0.75, 0.25, 0.75]
params = []
probs = [[prob_burst[0], prob_exit[0]], [prob_burst[1], prob_exit[1]], [prob_burst[2], prob_exit[2]], [prob_burst[3], prob_exit[3]]]
for rate in rates_list:
    burst_rates = [2 * rate, 5 * rate, 10 * rate]
    for burst_rate in burst_rates:
        for prob in probs:
            burst_prob = prob[0]
            exit_prob = prob[1]
            params.append([rate, burst_rate, burst_prob, exit_prob])

rates = []
burst_rates = []
probs_enter = []
probs_exit = []
cvs = []
frs = []

# make directory for each parameter
for i, param in enumerate(params):

    param_name = 'param_{}'.format(i)
    dir_name = os.path.join('thesis', param_name)
    os.mkdir(dir_name)

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

        # write train to file
        file_name = 'spikes.txt'
        spikes_path = os.path.join(path, file_name)
        with open(spikes_path, 'w') as file:
            file.write(str(trains))

        # FIX: np.savetxt (fmt = %f, newline="\n")

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

        # save parameters
        rates.append(param[0])
        burst_rates.append(param[1])
        probs_enter.append(param[2])
        probs_exit.append(param[3])
        cvs.append(spikestats["coefficient of variation"])
        frs.append(spikestats["actual rate (spikes/s)"])

        # write metadata to file
        file_name2 = 'metadata.txt'
        data_path = os.path.join(path, file_name2)
        with open(data_path, 'w') as file:
            file.write("parameters: ")
            file.write(str(parameters))
            file.write("\n")
            file.write("spike statistics: ")
            file.write(str(spikestats))
            file.write("\n")
            file.write("burst statistics: ")
            file.write(str(burststats))
            file.write("\n")
    
        # FIX: one big set of key : value pairs (combine 3 dictionaries)

        # raster plots
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.eventplot(trains, color="k")
        ax.set_xlabel("Time")
        ax.set_ylabel("Train Number")
        ax.set_title("Raster Plot")
        fig_path = os.path.join(path, 'raster plot')
        plt.savefig(fig_path)

# plot heatmaps: different combinations of synthetic spike train parameters on axes as indep vars, FR & CV as dep vars (2 panels)
# firing rate & burst rate
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

# firing rate & prob entering 
# create dataframes
data3 = {
    'predicted rates': rates,
    'probability of entering burst': probs_enter,
    'actual firing rates': frs
}
df3 = pd.DataFrame(data3)
df3_pivoted = df3.pivot_table(index='probability of entering burst', columns='predicted rates', values='actual firing rates', aggfunc='mean')

data4 = {
    'predicted rates': rates,
    'probability of entering burst': probs_enter,
    'coefficients of variation': cvs
}
df4 = pd.DataFrame(data4)
df4_pivoted = df4.pivot_table(index='probability of entering burst', columns='predicted rates', values='coefficients of variation', aggfunc='mean')

# plot 2 panels
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
ax1 = sns.heatmap(df3_pivoted, ax=ax1, cmap = 'viridis', cbar_kws={'label': 'actual firing rates'})
ax2 = sns.heatmap(df4_pivoted, ax=ax2, cmap = 'viridis', cbar_kws={'label': 'coefficients of variation'})
plt.xlabel("predicted rates")
plt.ylabel("probability of entering burst")
plt.show()

# firing rate & prob exiting
# create dataframes
data5 = {
    'predicted rates': rates,
    'probability of exiting burst': probs_exit,
    'actual firing rates': frs
}
df5 = pd.DataFrame(data5)
df5_pivoted = df5.pivot_table(index='probability of exiting burst', columns='predicted rates', values='actual firing rates', aggfunc='mean')

data6 = {
    'predicted rates': rates,
    'probability of exiting burst': probs_exit,
    'coefficients of variation': cvs
}
df6 = pd.DataFrame(data6)
df6_pivoted = df6.pivot_table(index='probability of exiting burst', columns='predicted rates', values='coefficients of variation', aggfunc='mean')

# plot 2 panels
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
ax1 = sns.heatmap(df5_pivoted, ax=ax1, cmap = 'viridis', cbar_kws={'label': 'actual firing rates'})
ax2 = sns.heatmap(df6_pivoted, ax=ax2, cmap = 'viridis', cbar_kws={'label': 'coefficients of variation'})
plt.xlabel("predicted rates")
plt.ylabel("probability of exiting burst")
plt.show()

# burst rate & prob entering 
# create dataframes
data7 = {
    'predicted burst rates': burst_rates,
    'probability of entering burst': probs_enter,
    'actual firing rates': frs
}
df7 = pd.DataFrame(data7)
df7_pivoted = df7.pivot_table(index='probability of entering burst', columns='predicted burst rates', values='actual firing rates', aggfunc='mean')

data8 = {
    'predicted burst rates': burst_rates,
    'probability of entering burst': probs_enter,
    'coefficients of variation': cvs
}
df8 = pd.DataFrame(data8)
df8_pivoted = df8.pivot_table(index='probability of entering burst', columns='predicted burst rates', values='coefficients of variation', aggfunc='mean')

# plot 2 panels
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
ax1 = sns.heatmap(df7_pivoted, ax=ax1, cmap = 'viridis', cbar_kws={'label': 'actual firing rates'})
ax2 = sns.heatmap(df8_pivoted, ax=ax2, cmap = 'viridis', cbar_kws={'label': 'coefficients of variation'})
plt.xlabel("predicted burst rates")
plt.ylabel("probability of entering burst")
plt.show()

# burst rate & prob exiting
# create dataframes
data9 = {
    'predicted burst rates': burst_rates,
    'probability of exiting burst': probs_exit,
    'actual firing rates': frs
}
df9 = pd.DataFrame(data9)
df9_pivoted = df9.pivot_table(index='probability of exiting burst', columns='predicted burst rates', values='actual firing rates', aggfunc='mean')

data10 = {
    'predicted burst rates': burst_rates,
    'probability of exiting burst': probs_exit,
    'coefficients of variation': cvs
}
df10 = pd.DataFrame(data10)
df10_pivoted = df10.pivot_table(index='probability of exiting burst', columns='predicted burst rates', values='coefficients of variation', aggfunc='mean')

# plot 2 panels
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
ax1 = sns.heatmap(df9_pivoted, ax=ax1, cmap = 'viridis', cbar_kws={'label': 'actual firing rates'})
ax2 = sns.heatmap(df10_pivoted, ax=ax2, cmap = 'viridis', cbar_kws={'label': 'coefficients of variation'})
plt.xlabel("predicted burst rates")
plt.ylabel("probability of exiting burst")
plt.show()

# prob entering & prob exiting
# create dataframes
data11 = {
    'probability of entering burst': probs_enter,
    'probability of exiting burst': probs_exit,
    'actual firing rates': frs
}
df11 = pd.DataFrame(data11)
df11_pivoted = df11.pivot_table(index='probability of exiting burst', columns='probability of entering burst', values='actual firing rates', aggfunc='mean')

data12 = {
    'probability of entering burst': probs_enter,
    'probability of exiting burst': probs_exit,
    'coefficients of variation': cvs
}
df12 = pd.DataFrame(data12)
df12_pivoted = df12.pivot_table(index='probability of exiting burst', columns='probability of entering burst', values='coefficients of variation', aggfunc='mean')

# plot 2 panels
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
ax1 = sns.heatmap(df11_pivoted, ax=ax1, cmap = 'viridis', cbar_kws={'label': 'actual firing rates'})
ax2 = sns.heatmap(df12_pivoted, ax=ax2, cmap = 'viridis', cbar_kws={'label': 'coefficients of variation'})
plt.xlabel("probability of entering burst")
plt.ylabel("probability of exiting burst")
plt.show()

print("hi")