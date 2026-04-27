import os
import sys
import numpy as np
import math
import synspiketrain

(D, T, N, params) = synspiketrain.return_params()
T_vals = [10, 30]

for i, param in enumerate(params):
    for j in range(0, N):

        train = []
        actual_bursts = []
        ps_bursts = []
        mi_bursts = []
        logisi_bursts = []
        cma_bursts = []
        unified_bursts = []

        param_name = f'param_{i:04d}'
        train_name = f'train_{j:03d}'
        path_name = os.path.join('thesis', param_name, train_name)
        if not os.path.exists(path_name):
            continue

        # get spikes and bursts from files
        actualbursts_path = os.path.join(path_name, 'bursts.txt')
        with open(actualbursts_path, 'r') as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                bursts = [burst.strip() for burst in line.split(',') if burst.strip() != '']
                actual_bursts.extend([round(float(burst), 6) for burst in bursts])
        
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
                ps_bursts.extend([round(float(burst), 6) for burst in bursts])
        
        mi_path = os.path.join(path_name, 'mi_bursts.txt')
        with open(mi_path, 'r') as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                bursts = [burst.strip() for burst in line.split(',') if burst.strip() != '']
                mi_bursts.extend([round(float(burst), 6) for burst in bursts])
        
        logisi_path = os.path.join(path_name, 'logisi_bursts.txt')
        with open(logisi_path, 'r') as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                bursts = [burst.strip() for burst in line.split(',') if burst.strip() != '']
                logisi_bursts.extend([round(float(burst), 6) for burst in bursts])
        
        cma_path = os.path.join(path_name, 'cma_bursts.txt')
        with open(cma_path, 'r') as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                bursts = [burst.strip() for burst in line.split(',') if burst.strip() != '']
                cma_bursts.extend([round(float(burst), 6) for burst in bursts])
        
        unified_path = os.path.join(path_name, 'unified_bursts.txt')
        with open(unified_path, 'r') as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                bursts = [burst.strip() for burst in line.split(',') if burst.strip() != '']
                unified_bursts.extend([round(float(burst), 6) for burst in bursts])

        analysis_dict = {}
        methods = {
            "PS": ps_bursts,
            "MI": mi_bursts,
            "LogISI": logisi_bursts,
            "CMA": cma_bursts,
            "Unified": unified_bursts
        }

        actual_set = set(round(spike, 6) for spike in actual_bursts)
        
        # iterate through each method
        for method, bursts in methods.items():
            correctly_detected = 0
            total_spikes_in_burst = len(actual_bursts)
            falsely_detected = 0
            total_spikes_not_in_burst = len(train) - len(actual_bursts)
            
            for spike in bursts:
                if round(spike, 6) in actual_set:
                    correctly_detected += 1
                else: 
                    falsely_detected += 1
            
            # calculate sensitivity and 1-specificity
            sensitivity = correctly_detected / total_spikes_in_burst if total_spikes_in_burst > 0 else np.nan
            one_minus_specificity = min(1.0, falsely_detected / total_spikes_not_in_burst) if total_spikes_not_in_burst > 0 else np.nan

            analysis_dict[f"{method}_sens"] = sensitivity
            analysis_dict[f"{method}_1spec"] = one_minus_specificity

        analysis_path = os.path.join('thesis', param_name, train_name, 'analysis.txt')
        analysis_array = np.array(list(analysis_dict.items()), dtype=object)
        np.savetxt(analysis_path, analysis_array, fmt="%s", delimiter=":")