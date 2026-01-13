import numpy as np
from matplotlib import pyplot as plt
import os
import pandas as pd
import seaborn as sns
import sys
import synthetic_saver
import synthetic_df
import apply_ps
import save_ps
import all_figs

df = synthetic_df.df

# raster plots per train
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
ax.eventplot(trains, color="k")
ax.set_xlabel("Time")
ax.set_ylabel("Train Number")
ax.set_title("Raster Plot")
fig_path = os.path.join(path, 'raster plot.png')
plt.savefig(fig_path)
plt.close()
    
# master raster plot per parameter set
plt.figure(figsize=(10, 6))
plt.eventplot(all_spikes, color="k")
plt.xlabel("time (s)")
plt.ylabel("train number")
plt.title("master raster plot")
master_fig_path = os.path.join(dir_name, 'master raster plot')
plt.savefig(master_fig_path) 
plt.close()

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
fig_path = os.path.join('thesis', 'cv_hist.png')
plt.savefig(fig_path)
plt.close()

# firing rate & cv histograms
# firing rate
plt.figure(figsize=(10,6))
plt.hist(all_frs, bins=20, color='blue', alpha=0.5, label = 'Firing rates')
plt.xlabel('Firing rate')
plt.ylabel('Count')
plt.legend()
fig_path = os.path.join('thesis', 'fr_hist.png')
plt.savefig(fig_path)
plt.close()

# cv
plt.figure(figsize=(10,6))
plt.hist(all_cvs, bins=20, color='blue', alpha=0.5, label = 'Coefficients of variation')
plt.xlabel('Coefficient of variation')
plt.ylabel('Count')
plt.legend()
fig_path = os.path.join('thesis', 'cv_hist.png')
plt.savefig(fig_path)
plt.close()

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
plt.close()

# burst rate
plt.figure(figsize=(10,6))
sns.scatterplot(data = df, x = "actual_rate", y = "cv", hue = "burst_rate")
plt.xlabel("firing rate")
plt.ylabel("coefficient of variation")
fig_path = os.path.join('thesis', 'burstrate_scatterplot.png')
plt.savefig(fig_path)
plt.close()

# prob burst
plt.figure(figsize=(10,6))
sns.scatterplot(data = df, x = "actual_rate", y = "cv", hue = "prob_burst")
plt.xlabel("firing rate")
plt.ylabel("coefficient of variation")
fig_path = os.path.join('thesis', 'probburst_scatterplot.png')
plt.savefig(fig_path)
plt.close()

# prob end
plt.figure(figsize=(10,6))
sns.scatterplot(data = df, x = "actual_rate", y = "cv", hue = "prob_end")
plt.xlabel("firing rate")
plt.ylabel("coefficient of variation")
fig_path = os.path.join('thesis', 'probend_scatterplot.png')
plt.savefig(fig_path)
plt.close()

# fr vs cv scatterplots by burst statistics
# synthetic bursts
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
ax1 = sns.scatterplot(data = df, x = "actual_rate", y = "cv", hue = "num_spikes")
ax2 = sns.scatterplot(data = df, x = "actual_rate", y = "cv", hue = "ps_num_spikes")
plt.xlabel("firing rate")
plt.ylabel("coefficient of variation")
fig_path = os.path.join('thesis', 'num_spikes_scatterplot.png')
plt.savefig(fig_path)
plt.close()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
ax1 = sns.scatterplot(data = df, x = "actual_rate", y = "cv", hue = "burst_firing_rate")
ax2 = sns.scatterplot(data = df, x = "actual_rate", y = "cv", hue = "ps_burst_firing_rate")
plt.xlabel("firing rate")
plt.ylabel("coefficient of variation")
fig_path = os.path.join('thesis', 'burst_firing_rate_scatterplot.png')
plt.savefig(fig_path)
plt.close()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
ax1 = sns.scatterplot(data = df, x = "actual_rate", y = "cv", hue = "avg_ISI_within_bursts")
ax2 = sns.scatterplot(data = df, x = "actual_rate", y = "cv", hue = "ps_avg_ISI_within_bursts")
plt.xlabel("firing rate")
plt.ylabel("coefficient of variation")
fig_path = os.path.join('thesis', 'avg_ISI_within_bursts_scatterplot.png')
plt.savefig(fig_path)
plt.close()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
ax1 = sns.scatterplot(data = df, x = "actual_rate", y = "cv", hue = "burst_rate")
ax2 = sns.scatterplot(data = df, x = "actual_rate", y = "cv", hue = "ps_burst_rate")
plt.xlabel("firing rate")
plt.ylabel("coefficient of variation")
fig_path = os.path.join('thesis', 'burst_rate_scatterplot.png')
plt.savefig(fig_path)
plt.close()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
ax1 = sns.scatterplot(data = df, x = "actual_rate", y = "cv", hue = "%_spikes_in_burst")
ax2 = sns.scatterplot(data = df, x = "actual_rate", y = "cv", hue = "ps_%_spikes_in_burst")
plt.xlabel("firing rate")
plt.ylabel("coefficient of variation")
fig_path = os.path.join('thesis', '%_spikes_in_burst_scatterplot.png')
plt.savefig(fig_path)
plt.close()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
ax1 = sns.scatterplot(data = df, x = "actual_rate", y = "cv", hue = "%_time_spent_bursting")
ax2 = sns.scatterplot(data = df, x = "actual_rate", y = "cv", hue = "ps_%_time_spent_bursting")
plt.xlabel("firing rate")
plt.ylabel("coefficient of variation")
fig_path = os.path.join('thesis', '%_time_spent_bursting_scatterplot.png')
plt.savefig(fig_path)
plt.close()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
ax1 = sns.scatterplot(data = df, x = "actual_rate", y = "cv", hue = "firing_rate_non_bursting")
ax2 = sns.scatterplot(data = df, x = "actual_rate", y = "cv", hue = "ps_firing_rate_non_bursting")
plt.xlabel("firing rate")
plt.ylabel("coefficient of variation")
fig_path = os.path.join('thesis', 'firing_rate_non_bursting_scatterplot.png')
plt.savefig(fig_path)
plt.close()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
ax1 = sns.scatterplot(data = df, x = "actual_rate", y = "cv", hue = "burst_firing_rate_inc")
ax2 = sns.scatterplot(data = df, x = "actual_rate", y = "cv", hue = "ps_burst_firing_rate_inc")
plt.xlabel("firing rate")
plt.ylabel("coefficient of variation")
fig_path = os.path.join('thesis', 'burst_firing_rate_inc_scatterplot.png')
plt.savefig(fig_path)
plt.close()
