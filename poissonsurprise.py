import numpy as np
import math 
import synspiketrain
np.random.seed(1)

def calc_surprise(burst, rate):
    """
    Calculate surprise value of a burst candidate.

    Parameters
    ----------
    burst (array): array of spike times in burst candidate
    rate (float): firing rate

    Returns
    -------
    surprise (float): surprise value of burst candidate
    """
    n = len(burst)
    interval = burst[-1] - burst[0]
    p = np.exp(-rate * interval) * np.sum(
            [
                (rate * interval) ** i / math.factorial(i)
                for i in range(len(burst), np.max([len(burst) * 4, 80]))
            ]
        ) 
    return -np.log(p)

def poisson_surprise(spike_train, min_spikes = 3, max_spikes = 10, surprise_threshold = 5):
    """
    Identify bursts in spike train using Poisson surprise detection method

    Parameters
    ----------
    spike_train (array): array of spike times in train
    min_spikes (float): minimum number of spikes in burst
    max_spikes (float): maximum number of spikes in burst
    surprise_threshold (float): threshold for surprise value to be a burst

    Returns
    -------
    bursts (array): array of arrays of spike times that make up each burst
    """

    if len(spike_train) < min_spikes:
        return []
    bursts = []

    # calculate rate & isi
    T = 1
    rate = len(spike_train) / T

    def calc_avg_isi(train):
        """
        Calculate average interspike interval of a spike train.
        """
        isis = []
        for i in range(len(train) - 1):
            isi = train[i+1] - train[i]
            isis.append(float(isi))
        avg_isi = float(np.mean(isis))
        return avg_isi

    train_avg_isi = calc_avg_isi(spike_train)
        
    spike_index = 0
    while spike_index < (len(spike_train) - min_spikes):
        # identify burst candidate
        burst_candidate = spike_train[spike_index : (spike_index + min_spikes)]
        candidate_avg_isi = calc_avg_isi(burst_candidate)
        if candidate_avg_isi <= 2 * train_avg_isi: # if meets criteria
            candidate_surprise = calc_surprise(burst_candidate, rate)
            # add forward spikes to candidate until surprise no longer increases
            add_forward = True
            while add_forward and len(burst_candidate) < max_spikes:
                new_burst_candidate = spike_train[spike_index : (spike_index + len(burst_candidate) + 1)]
                new_surprise = calc_surprise(new_burst_candidate, rate)
                if new_surprise > candidate_surprise and len(new_burst_candidate) >= min_spikes:
                    candidate_surprise = new_surprise
                    burst_candidate = new_burst_candidate
                else:
                    add_forward = False
            # remove earliest spikes from candidate until surprise no longer increases
            remove_backward = True
            while remove_backward and len(burst_candidate) > min_spikes:
                new_burst_candidate = burst_candidate[1:]
                new_surprise = calc_surprise(new_burst_candidate, rate)
                if new_surprise > candidate_surprise and len(new_burst_candidate) >= min_spikes:
                    candidate_surprise = new_surprise
                    burst_candidate = new_burst_candidate
                else:
                    remove_backward = False
            bursts.append([burst_candidate, candidate_surprise])
            # burst candidate has max surprise --> start with next non spike burst, repeat
            spike_train = np.atleast_1d(spike_train)
            spike_index = np.where(spike_train == burst_candidate[-1])[0][0] + 1
        else:
            spike_index += 1
    # store burst candidates + associated surprise values to compare to threshold
    return [[np.array(burst[0]), float(burst[1])] for burst in bursts if burst[1] >= surprise_threshold]
