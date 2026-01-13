import numpy as np
from matplotlib import pyplot as plt
import synspiketrain
np.random.seed(0)
import math 

def calculate_statistics(spike_train, bursts, rate, T, burst_rate, prob_burst, prob_exit, tau_ref, tau_burst):
    """
    Calculate spike train and bursting statistics.

    Parameters
    ----------
    spike_train : np.array
        List of spike times
    bursts : np.array
        List of burst times
    rate : float
        Baseline firing rate of the spike train
    burst_rate : float
        Elevated firing rate used during a burst
    T : float
        Length of time for spike train (seconds)
    tau_ref : float
        Refractory period, shortest time between spikes
    tau_burst : float
        Minimum time between consecutive bursts
    prob_burst : float
        Probability to enter a burst.
    prob_end : float
        Probability to end the burst.

    Returns
    --------
    dict
        spike statistics
    dict
        burst statistics
    """

    # calculate rate, CV, ISI
    diff = np.diff(spike_train)
    isis = []

    if diff.size == 0:
        cv = 0.0

    else:
        mean_isi = float(np.mean(diff))
        if mean_isi <= 0:
            cv = 0.0
        else: 
            cv = float(np.std(diff) / mean_isi)
            if not np.isfinite(cv):
                cv = 0.0
        
        for i in range(len(spike_train) - 1):
            isi = spike_train[i+1] - spike_train[i]
            isis.append(float(isi))
    
    actual_rate = len(spike_train) / T if T > 0 else 0.0

    spike_stats = {
        "actual_rate": actual_rate, 
        "cv": cv,
        "isi_dist": isis
    }

    # calculate burst statistics 
    num_spikes = len(spike_train)
    burst_rate = len(bursts) / T if T > 0 else 0.0
    burst_count = 0
    burst_time = 0
    spike_train_no_bursts = spike_train
    burst_isis = []
    for burst in bursts:
        burst_count += len(burst)
        burst_time += (burst[-1] - burst[0])
        for i in range(len(burst) - 1):
                burst_isi = burst[i+1] - burst[i]
                burst_isis.append(float(burst_isi))
        for b in burst:
            index_to_remove = np.where(spike_train_no_bursts == b)
            spike_train_no_bursts = np.delete(spike_train_no_bursts, index_to_remove)
    if len(burst_isis) > 0:
        avg_burst_isi = float(np.nanmean(burst_isis))
    else:
        avg_burst_isi = 0.0

    if len(spike_train) > 0:
        spikes_in_burst = burst_count / len(spike_train)
    else:
        spikes_in_burst = 0
    if burst_time > 0:
        burst_firing_rate = burst_count / burst_time
    else:
        burst_firing_rate = 0
    time_in_burst = burst_time / T if T > 0 else 0.0
    time_not_in_burst = T - burst_time
    non_burst_firing_rate = len(spike_train_no_bursts) / time_not_in_burst if time_not_in_burst > 0 else 0.0
    burst_firing_rate_increase = burst_firing_rate - non_burst_firing_rate

    burst_properties = {
        "num_spikes": num_spikes,
        "burst_firing_rate": burst_firing_rate, 
        "avg_ISI_within_bursts": avg_burst_isi,
        "burst_rate": burst_rate, 
        "%_spikes_in_burst": spikes_in_burst,
        "%_time_spent_bursting": time_in_burst,
        "firing_rate_non_bursting": non_burst_firing_rate,
        "burst_firing_rate_inc": burst_firing_rate_increase,
    }
    
    return spike_stats, burst_properties