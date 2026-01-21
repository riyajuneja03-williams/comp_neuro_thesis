import numpy as np
import math 
import synspiketrain
from scipy.signal import find_peaks
from matplotlib import pyplot as plt
import seaborn as sns
import statsmodels.api as sm
np.random.seed(1)

def log_isi(trains, minSpikes=5):
    """
    Detects bursts using LogISI method.

    Parameters
    ----------
    trains: np.array
        array(s) of spike times
    minSpikes: integer
        minimum number of spikes in burst, default 5

    Returns
    -------
    bursts (array): list of lists of spike times that make up each burst

    """

    # if spike train not long enough
    if len(trains) < 2:
        return []

    # compute ISI threshold
    ISIth = compute_ISIth(trains)  

    # if algorithm fails to derive ISIth, apply CH
    if ISIth is None or ISIth > 1000:
        core_windows = detect_windows_CH(trains, maxISI_ms=100, minSpikes=minSpikes)
        return np.array(windows_to_bursts(trains, core_windows))

    # if ISIth > 100, extend bursts to include spikes at boundaries
    if ISIth > 100:
        maxISI1 = 100
        maxISI2 = ISIth
        extendFlag = True 
    else:
        maxISI1 = ISIth
        maxISI2 = None
        extendFlag = False

    # detect burst cores using CH and maxISI1
    core_windows = detect_windows_CH(trains, maxISI_ms=maxISI1, minSpikes=minSpikes)
    if len(core_windows) == 0:
        return []

    # if don't need to extend, return results 
    if extendFlag is False:
        return windows_to_bursts(trains, core_windows)

    # join two consecutive bursts if separated by single interval smaller than maxISI2
    merged = [core_windows[0]]
    for (s2, e2) in core_windows[1:]:
        (s1, e1) = merged[-1]
        gap_ms = (trains[s2] - trains[e1]) * 1000

        if gap_ms <= maxISI2:
            merged[-1] = (s1, e2)
        else:
            merged.append((s2, e2))

    # extend at burst boundaries using CH and maxISI2
    ISI_ms = np.diff(trains) * 1000  # length = length(trains) - 1
    extended = []

    for (s, e) in merged: # s = start of burst, e = end of burst
        # extend left while prior ISI < maxISI2
        while s > 0 and ISI_ms[s-1] < maxISI2:
            s = s - 1

        # extend right while ISI at e < maxISI2 
        while e < len(trains)-1 and ISI_ms[e] < maxISI2:
            e = e + 1

        extended.append((s, e))

    # remerge any overlapping windows
    extended.sort(key = lambda x: x[0])
    cleaned = []
    for (s2, e2) in extended:
        if len(cleaned) == 0:
            cleaned.append((s2, e2))
        else:
            (s1, e1) = cleaned[-1]
            if s2 <= e1 + 1:
                cleaned[-1] = (s1, max(e1, e2))
            else:
                cleaned.append((s2, e2))

    # convert windows to list-of-lists of spike times
    return np.array(windows_to_bursts(trains, cleaned))


def windows_to_bursts(trains, windows):
    """
    Creates list of spikes (burst times) in window.

    Parameters
    ----------
    trains: np.array
        array(s) of spike times
    windows: list

    Returns
    -------
    bursts (array): list of lists of spike times that make up each burst

    """   
    bursts = []
    for (s, e) in windows:
        bursts.append(trains[s : e+1])
    return bursts


def compute_ISIth(trains):
    """
    Compute ISI threshold.

    Parameters
    ----------
    trains: np.array
        array(s) of spike times

    Returns
    -------
    ISIth : integer
        threshold in milliseconds

    """

    # calculate logISI
    ISI_ms = np.diff(trains) * 1000
    logISI = np.log(ISI_ms)

    # compute histogram with bins of 0.1 in logISI units
    bins = np.arange(min(logISI), max(logISI) + 0.1, 0.1)
    counts, edges = np.histogram(logISI, bins=bins)
    g = counts / sum(counts)
    x = (edges[:-1] + edges[1:]) / 2

    # smooth g as a function of x using lowess, default is local linear regression aka degree = 1
    g_smooth = sm.nonparametric.smoothers_lowess.lowess(y=g, x=x)

    # identify principle peaks (min distance = 2 bins)
    peak_idxs = find_peaks(g_smooth, min_distance=2)
    if len(peak_idxs) == 0:
        return None

    # identify intra-burst peak: ISI < 100 ms, choose largest height
    isi_at_x_ms = 10^x
    candidates = [p for p in peak_idxs if isi_at_x_ms[p] < 100]
    if len(candidates) == 0:
        return None
    p1 = np.argmax([g_smooth[p] for p in candidates])

    # need at least one subsequent peak (p2 > p1) to define a valley
    subsequent = [p for p in peak_idxs if p > p1]
    if len(subsequent) == 0:
        return None

    # compute void parameter for each pair (p1, p2), p2 subsequent
    best_void = -np.inf
    best_min_idx = None
    void_thresh = 0.7

    for p2 in subsequent:
        left = p1
        right = p2

        m = np.argmin(g_smooth[left:right+1]) + left
        gmin = g_smooth[m]
        g1 = g_smooth[p1]
        g2 = g_smooth[p2]

        void = 1 - gmin / np.sqrt(g1 * g2)
    
        # select ISIth
        if void >= void_thresh and void > best_void:
            best_void = void
            best_min_idx = m

    if best_min_idx is None:
        return None

    return 10^(x[best_min_idx])   # ISIth in ms

def detect_windows_CH(trains, maxISI_ms, minSpikes):
    # CH algorithm: runs of consecutive ISIs < maxISI_ms, requiring at least minSpikes spikes
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