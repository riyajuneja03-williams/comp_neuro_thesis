import numpy as np
import math 
import synspiketrain
from scipy.signal import find_peaks
from matplotlib import pyplot as plt
import seaborn as sns
import statsmodels.api as sm
np.random.seed(1)


def bins(spikes, bin_size):
    """
    Bins spikes.

    Parameters
    ----------
    spikes : np.array
        Array of spike times 
    bin_size : float
        Bin sizes

    Returns
    --------
    bins : np.array
        List of bins
    """
    return np.arange(np.min(spikes), np.max(spikes) + bin_size, bin_size) # calculate bin spike

def log_isi_hist(spikes, bin_size = 0.1, threshold=0.7, cutoff=100):
    """
    Creates histogram of spikes.

    Parameters
    ----------
    spikes : np.array
        Array of spike times 
    bin_size : float
        Bin sizes
    threshold : float
        Threshold value for burst detection
    cutoff : integer
        Cutoff value for burst detection

    Returns
    --------
    
    """
    log_spikes = np.log10(np.diff(spikes)) 
    bins = bins(log_spikes, bin_size)
    hist, _ = np.histogram(log_spikes,bins=bins)
    smoothed_hist = sm.nonparametric.lowess(hist, bins)

trains = synspiketrain.trains
bursts = synspiketrain.bursts

detected_bursts = log_isi(trains)
print(f"spike train: {trains}")
print(f"actual bursts: {bursts}")
print(f"bursts detected by LogISI: {detected_bursts}")
