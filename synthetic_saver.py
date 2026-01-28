import numpy as np
from matplotlib import pyplot as plt
import os
import pandas as pd
import seaborn as sns
import synspiketrain
import stats

(D, T, N, params) = synspiketrain.return_params()

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
            T = param[0],
            D = param[1],
            train_rate=param[2],
            burst_rate=param[3],
            single_burst_rate=param[4],
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
            'T': param[0],
            'D': param[1],
            'train_rate': param[2],
            'burst_rate': param[3],
            'single_burst_rate': param[4],
        }

        spikestats, burststats = stats.calculate_statistics(trains, bursts, param[0], param[1], param[2], param[3], param[4])

        # write metadata to file
        data_name = 'metadata.txt'
        data_path = os.path.join(path, data_name)

        all_metadata = parameters | spikestats | burststats | spikes_dict

        metadata_array = np.array(list(all_metadata.items()), dtype=object)
        np.savetxt(data_path, metadata_array, fmt = "%s", delimiter = ":")
