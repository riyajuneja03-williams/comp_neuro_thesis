import numpy as np
from scipy.stats import skew
from matplotlib import pyplot as plt
import seaborn as sns
np.random.seed(1)

def cma_burst_detection(trains, min_spikes_in_burst=3):
    """
    Detects bursts using CMA method.

    Parameters
    ----------
    trains: np.array
        array(s) of spike times
    minSpikes: integer
        minimum number of spikes

    Returns
    -------
    bursts (array): list of lists of spike times that make up each burst

    """  

    # if spike train not long enough
    if len(trains) < 3:
        return []

    # compute log histogram
    ISI_ms = np.diff(trains) * 1000
    ISI_ms = ISI_ms[np.isfinite(ISI_ms) & (ISI_ms > 0)]
    if ISI_ms.size < 2:
        return []
    logISI = np.log10(ISI_ms)
    bins = np.arange(min(logISI), max(logISI) + 0.1, 0.1)
    y, edges = np.histogram(logISI, bins=bins)
    x_mid = 0.5 * (edges[:-1] + edges[1:]) / 2

    if sum(y) == 0:
        return []

    # calculate CMA maximum
    CH = np.cumsum(y)
    CMA = [CH[i] / (i+1) for i in range(len(y))]
    max_index = np.argmax(CMA)
    CMA_max = CMA[max_index]

    # calculate skewness
    skewness = skew(ISI_ms)

    # calculate alpha1
    if skewness < 1: alpha1 = 1.0
    elif skewness < 4: alpha1 = 0.7
    elif skewness < 9: alpha1 = 0.5
    else: alpha1 = 0.3

    # calculate alpha2
    if skewness < 4: alpha2 = 0.5
    elif skewness < 9: alpha2 = 0.3
    else: alpha2 = 0.1

    # calculate ISI threshold xt1
    xt1_log = find_xt(CMA, x_mid, max_index, target=alpha1 * CMA_max)
    if xt1_log is None:
        return []
    xt1 = 10 ** xt1_log

    # detect burst cores
    burst_windows = detect_runs_of_ISIs_below(trains, maxISI_ms=xt1, minSpikes=min_spikes_in_burst)
    if len(burst_windows) == 0:
        return []

    # calculate ISI threshold xt2
    xt2_log = find_xt(CMA, x_mid, max_index, target=alpha2 * CMA_max)

    if xt2_log is not None:

        xt2 = 10 ** xt2_log 

        # find candidate burst related spikes
        br_windows = detect_runs_of_ISIs_below(trains, maxISI_ms=xt2, minSpikes=2)

        # keep only the candidates that touch other bursts
        br_kept = []
        for (related_start, related_end) in br_windows:
            keep = False
            for (burst_start, burst_end) in burst_windows:
                if related_start <= burst_end + 1 and related_end >= burst_start - 1:
                    keep = True
                    break
            if keep:
                br_kept.append((related_start, related_end))
        br_windows = br_kept

        # extend each burst to burst related spikes
        extended = []
        for (burst_start, burst_end) in burst_windows:
            start, end = burst_start, burst_end
            changed = True
            while changed:
                changed = False
                for (related_start, related_end) in br_windows:
                    if related_start <= end + 1 and related_end >= start - 1:
                        new_start = min(start, related_start)
                        new_end = max(end, related_end)
                        if new_start != start or new_end != end:
                            start, end = new_start, new_end
                            changed = True
            extended.append((start, end))
        burst_windows = extended

        # merge bursts closer to each other than threshold
        burst_windows.sort(key=lambda w: w[0])
        merged = [burst_windows[0]]
        for (start2, end2) in burst_windows[1:]:
            (start1, end1) = merged[-1]
            gap_ms = (trains[start2] - trains[end1]) * 1000
            if gap_ms < xt2:
                merged[-1] = (start1, end2)
            else:
                merged.append((start2, end2))
        burst_windows = merged

        # remerge any overlapping windows
        burst_windows.sort(key=lambda w: w[0])
        cleaned = [burst_windows[0]]
        for (start2, end2) in extended:
            if len(cleaned) == 0:
                cleaned.append((start2, end2))
            else:
                (start1, end1) = cleaned[-1]
                if start2 <= end1 + 1:
                    cleaned[-1] = (start1, max(end1, end2))
                else:
                    cleaned.append((start2, end2))

    # return list of lists of bursts
    return windows_to_bursts(trains, cleaned)


def find_xt(CMA, x_mid, max_index, target):
    """
    Identify ISI threshold.

    Parameters
    ----------
    CMA : integer
        cumulative moving average
    x_mid : np.array
        mid time points of isi bins
    max_index : integer
        index
    target : integer
        target value

    Returns
    -------
    xt1 : integer
        threshold

    """  
    if max_index >= len(CMA) - 1:
        return None

    # threshold is found at mid time point of ISI bin for which value of CMA is closest to alpha * CMAm
    best_index = None
    best_diff = np.inf
    for index in range(max_index+1, len(CMA)):
        diff = abs(CMA[index] - target)
        if diff < best_diff:
            best_diff = diff
            best_index = index

    return None if best_index is None else x_mid[best_index]


def detect_runs_of_ISIs_below(trains, maxISI_ms, minSpikes):

    """
    CH algorithm: detect runs of consecutive ISIs < maxISI_ms

    Parameters
    ----------
    trains: np.array
        array(s) of spike times
    maxISI_ms: integer
        max interspike interval in ms
    minSpikes: integer
        minimum number of spikes

    Returns
    -------
    windows : list
        burst windows

    """
    ISI_ms = np.diff(trains) * 1000
    windows = []

    i = 0
    while i < len(ISI_ms):
        if ISI_ms[i] < maxISI_ms:
            run_start = i
            while i < len(ISI_ms) and ISI_ms[i] < maxISI_ms:
                i = i + 1
            run_end = i - 1

            s = run_start
            e = run_end + 1 # map ISI indices to spike indices

            if (e - s + 1) >= minSpikes:
                windows.append((s, e))
        else:
            i = i + 1
    
    return windows

def windows_to_bursts(trains, windows):
    """
    Creates list of spikes (burst times) in window.

    Parameters
    ----------
    trains: np.array
        array(s) of spike times
    windows: list
        burst windows

    Returns
    -------
    bursts (array): list of lists of spike times that make up each burst

    """   
    bursts = []
    for (s, e) in windows:
        bursts.append(trains[s : e+1])
    return bursts