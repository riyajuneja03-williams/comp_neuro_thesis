import numpy as np
from scipy.signal import find_peaks
from matplotlib import pyplot as plt
import seaborn as sns
from statsmodels.nonparametric.smoothers_lowess import lowess
np.random.seed(1)

def log_isi(trains, minSpikes=5):
    """
    Detects bursts using LogISI method.

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
    if len(trains) < 2:
        return []

    # compute ISI threshold
    ISIth = compute_ISIth(trains)  

    # if algorithm fails to derive ISIth, apply CH
    if ISIth is None or ISIth > 1000:
        core_windows = detect_windows_CH(trains, maxISI_ms=100, minSpikes=minSpikes)
        return windows_to_bursts(trains, core_windows)

    # if ISIth > 100, extend bursts to include spikes at boundaries
    if ISIth > 100:
        maxISI1 = 100
        maxISI2 = min(ISIth) 
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
    for (start2, end2) in core_windows[1:]:
        (start1, end1) = merged[-1]
        gap_ms = (trains[start2] - trains[end1]) * 1000

        if gap_ms <= maxISI2:
            merged[-1] = (start1, end2)
        else:
            merged.append((start2, end2))

    # extend at burst boundaries using CH and maxISI2
    ISI_ms = np.diff(trains) * 1000  # length = length(trains) - 1
    extended = []

    for (start, end) in merged: # s = start of burst, e = end of burst
        # extend left while prior ISI < maxISI2
        while start > 0 and ISI_ms[start-1] < maxISI2:
            start = start - 1

        # extend right while ISI at e < maxISI2 
        while end < len(trains)-1 and ISI_ms[end] < maxISI2:
            end = end + 1

        extended.append((start, end))

    # remerge any overlapping windows
    extended.sort(key = lambda x: x[0])
    cleaned = []
    for (start2, end2) in extended:
        if len(cleaned) == 0:
            cleaned.append((start2, end2))
        else:
            (start1, end1) = cleaned[-1]
            if start2 <= end1 + 1:
                cleaned[-1] = (start1, max(end1, end2))
            else:
                cleaned.append((start2, end2))

    # convert windows to list-of-lists of spike times
    return windows_to_bursts(trains, cleaned)


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
    ISI_ms = ISI_ms[np.isfinite(ISI_ms) & (ISI_ms > 0)]
    if ISI_ms.size < 2:
        return None
    logISI = np.log10(ISI_ms)

    # compute histogram with bins of 0.1 in logISI units
    bins = np.arange(min(logISI), max(logISI) + 0.1, 0.1)
    counts, edges = np.histogram(logISI, bins=bins)
    g = counts / sum(counts)
    x = (edges[:-1] + edges[1:]) / 2

    # smooth g as a function of x using lowess, default is local linear regression aka degree = 1
    mask = np.isfinite(x) & np.isfinite(g)
    x = x[mask]
    g = g[mask]
    if x.size < 3:
        return None
    g_smooth = lowess(g, x, return_sorted = False)

    # identify principle peaks (min distance = 2 bins)
    peak_idxs, _ = find_peaks(g_smooth, distance=2)
    if len(peak_idxs) == 0:
        return None

    # identify intra-burst peak: ISI < 100 ms, choose largest height
    isi_at_x_ms = 10 ** x
    candidates = [p for p in peak_idxs if isi_at_x_ms[p] < 100]
    if len(candidates) == 0:
        return None
    p1 = candidates[np.argmax([g_smooth[p] for p in candidates])]

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

    return 10 ** (x[best_min_idx])   # ISIth in ms

def detect_windows_CH(trains, maxISI_ms, minSpikes):

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