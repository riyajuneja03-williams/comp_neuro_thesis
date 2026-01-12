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
    cv = float(np.std(diff) / np.mean(diff))
    if math.isnan(cv):
        cv = 0
    actual_rate = len(spike_train) / synspiketrain.T
    isis = []
    for i in range(len(spike_train) - 1):
        isi = spike_train[i+1] - spike_train[i]
        isis.append(float(isi))

    spike_stats = {
        "actual_rate": actual_rate, 
        "cv": cv,
        "isi_dist": isis
    }

    # calculate burst statistics 
    num_spikes = len(spike_train)
    burst_rate = len(bursts) / synspiketrain.T
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
    if np.any(np.isfinite(burst_isis)):
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
    time_in_burst = burst_time / synspiketrain.T
    time_not_in_burst = synspiketrain.T - burst_time
    non_burst_firing_rate = len(spike_train_no_bursts) / time_not_in_burst
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
    print(spike_stats)
    print(burst_properties)
    
    return spike_stats, burst_properties

calculate_statistics(synspiketrain.trains, synspiketrain.bursts, synspiketrain.rate, synspiketrain.T, synspiketrain.burst_rate, synspiketrain.prob_burst, synspiketrain.prob_exit, synspiketrain.tau_ref, synspiketrain.tau_burst)