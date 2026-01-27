import numpy as np
from matplotlib import pyplot as plt
import os
import pandas as pd
import seaborn as sns
import synspiketrain
import stats

(T, N, params, tau_ref, tau_burst) = synspiketrain.return_params()

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
