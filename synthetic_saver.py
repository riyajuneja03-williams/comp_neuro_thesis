import numpy as np
from matplotlib import pyplot as plt
import os
import pandas as pd
import seaborn as sns
import synspiketrain
import stats

# set desired time and time step
T = 1
dt = 1e-3
N = 100

# refractory parameters
tau_ref = 0
tau_burst = 0

# define parameter sets
rates = np.arange(2, 102, 4)
rates_list = rates.tolist()
prob_burst = [0.2, 0.2, 0.2, 0.5, 0.5, 0.5, 0.8, 0.8, 0.8]
prob_exit = [0.2, 0.5, 0.8, 0.2, 0.5, 0.8, 0.2, 0.5, 0.8]
params = []
probs = [[prob_burst[0], prob_exit[0]], 
         [prob_burst[1], prob_exit[1]], 
         [prob_burst[2], prob_exit[2]], 
         [prob_burst[3], prob_exit[3]], 
         [prob_burst[4], prob_exit[4]], 
         [prob_burst[5], prob_exit[5]], 
         [prob_burst[6], prob_exit[6]], 
         [prob_burst[7], prob_exit[7]], 
         [prob_burst[8], prob_exit[8]]]

for rate in rates_list:
    burst_rates = [2 * rate, 5 * rate, 10 * rate]
    for burst_rate in burst_rates:
        for prob in probs:
            burst_prob = prob[0]
            exit_prob = prob[1]
            params.append([rate, burst_rate, burst_prob, exit_prob])

# make directory for each parameter
for i, param in enumerate(params):

    param_name = f'param_{i:04d}'
    dir_name = os.path.join('thesis', param_name)
    os.makedirs(dir_name, exist_ok=True)

    # make subdirectory for each train
    for j in range(0, N):
        subdir_name = f'train_{j:03d}'
        path = os.path.join(dir_name, subdir_name)
        os.makedirs(path, exist_ok=True) 

        # simulate trains
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
        spikes_name = 'spikes.txt'
        spikes_path = os.path.join(path, spikes_name)
        np.savetxt(spikes_path, trains, fmt = "%f", newline="\n")

        # define metadata 
        spikes_dict = {
            "spikes": trains,
            "bursts": bursts
        }

        parameters = {
            "rate": param[0],  
            "burst_rate_expected": param[1], 
            "T": T,  
            "tau_ref": tau_ref, 
            "tau_burst": tau_burst,  
            "prob_burst": param[2], 
            "prob_end": param[3]
        }

        spikestats, burststats = stats.calculate_statistics(trains, bursts, param[0], T, param[1], param[2], param[3], tau_ref, tau_burst)

        # write metadata to file
        data_name = 'metadata.txt'
        data_path = os.path.join(path, data_name)

        all_metadata = parameters | spikestats | burststats | spikes_dict

        metadata_array = np.array(list(all_metadata.items()), dtype=object)
        np.savetxt(data_path, metadata_array, fmt = "%s", delimiter = ":")

