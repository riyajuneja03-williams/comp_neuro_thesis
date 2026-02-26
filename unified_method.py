import numpy as np
from matplotlib import pyplot as plt
import os
import pandas as pd
import seaborn as sns
import sys
import logisi
import poissonsurprise
import cma
import maxinterval

def unified_burst_detection(spikes, T):
    """
    Detects bursts in spike train using novel unified method.

    Parameters
    ----------
    Spikes : np.array
        Array of spike times following poisson distribution

    T : integer
        Recording length.

    Returns
    --------
    bursts : list of lists
        List of list of spike times.
    
    """

    # get bursts by applying each method
    ps_bursts = [burst for burst, surprise in poissonsurprise.poisson_surprise(spikes, T=T)]
    mi_bursts = maxinterval.max_interval(spikes)
    cma_bursts = cma.cma_burst_detection(spikes)
    logisi_bursts = logisi.log_isi(spikes)

    # sort all bursts
    all_bursts = ps_bursts + mi_bursts + cma_bursts + logisi_bursts
    all_bursts = [burst.tolist() if isinstance(burst, np.ndarray) else burst for burst in all_bursts]
    all_bursts.sort()

    bursts = []

    # iterate through bursts
    while all_bursts:
        burst = all_bursts[0] # pick earliest burst

        candidate_list = []
        candidate_spikes = []
        counts = {}

        candidates_end = burst[-1]
        candidate_list.append(burst)
        candidate_spikes.extend(burst)

        # identify candidates to compare
        for other_burst in all_bursts: 
            if other_burst is burst:
                continue
            if other_burst[0] <= candidates_end and other_burst not in candidate_list:
                candidate_list.append(other_burst)
                candidate_spikes.extend(other_burst)
            
                if other_burst[-1] > candidates_end: # keep finding candidates until hit a "break"
                    candidates_end = other_burst[-1]

        # compare and save burst
        if len(candidate_list) >= 2: # majority rules if candidate is a burst
            for spike in candidate_spikes:
                if spike in counts:
                    counts[spike] += 1
                else:
                    counts[spike] = 1
            burst = sorted([spike for spike, c in counts.items() if c >= 2]) # majority rules if spike is in burst
            if len(burst) >= 3: # min spikes = 3 
                if len(burst) <= 10: # max spikes = 10
                    bursts.append(burst)
                else:
                    bursts.append(burst[:10])
                    time_end = burst[9]
                    new_all_bursts = []
                    for burst in all_bursts:
                        if burst in candidate_list:
                            keep = [spike for spike in burst if spike > time_end]
                            if keep:
                                new_all_bursts.append(keep)
                        else:
                            new_all_bursts.append(burst)
                    
                    all_bursts = new_all_bursts
                    all_bursts.sort()
                    continue

        # remove candidates from all_bursts
        all_bursts = [b for b in all_bursts if b not in candidate_list]

    return(bursts)