# synthetic data
# generates a random spike train at some rate and plots using a built-in matplotlib graph

import numpy as np
import synspiketrain
from matplotlib import pyplot as plt

def poisson_burst(T, D, train_rate, burst_rate, single_burst_rate):
    """
    Generate Poisson spike train.

    Parameters
    ----------
    train_rate : float
        Baseline firing rate of the spike train
    burst_rate : float
        Rate of bursts
    single_burst_rate : float
        Elevated firing rate used during a burst
    T : float
        Length of time for spike train (seconds)
    D : float
        Length of time for burst (seconds)

    Returns
    --------
    np.ndarray
        array of spike times following poisson distribution
    np.ndarray
        array of arrays of spike times representing each burst
    
    Parameters
        𝜆r – train rate (hz)
        𝜆burst – burst rate (burst/s)
        𝜆b – single burst rate (Hz)
        D – burst length (s)
        T – recording length (s)
    Algorithm
        Generate train s with Poisson rate 𝜆r
        Generate burst start times with Poisson rate 𝜆burst
        For each burst start time bi:
        Generate spikes at Poisson rate 𝜆b for length [bi, bi + D]
        Merge all spikes, track bursts
    
    """

    # empty array to hold spike times
    train = []
    burst_starts = []
    bursts = []

    def poisson_helper(rate, time, rng=None):
        """
        Generate poisson spike train of length time at rate rate.
    
        Parameters
        ----------
        rate : float
            Baseline firing rate of the spike train
        time : float
            Length of time
        rng : numpy.random.generator, optional
            Defaults to none otherwise sets random seed
        
        Returns
        ----------
        list
            list of spike times following poisson distribution

        """
    
        if rate <= 0 or time <= 0:
            return np.array([])

        # empty array to hold spike times
        spikes = []
        t = 0
    
        # draw ISIs from rate distribution
        while t < time:
            t += rng.exponential(1 / rate)

            # only add spike if smaller than cutoff time
            if t < time:
                spikes.append(t)
        
        return spikes

    rng = np.random.default_rng()

    train = poisson_helper(train_rate, T, rng=rng) # generate train with poisson rate train_rate
    burst_starts = poisson_helper(burst_rate, T, rng=rng) # generate burst start times with poisson rate burst_rate

    for start in burst_starts: # for each burst start time 
        time = min(D, T-start)
        burst_rel = poisson_helper(single_burst_rate, time, rng=rng) # generate spikes at poisson rate single_burst_rate for length (start, start + D)
        burst = [t + start for t in burst_rel]
        bursts.append(burst) # track bursts
        train.extend(burst) # merge all spikes

    train = np.array(sorted(train), dtype=float)
    bursts = np.array([np.array(b, dtype=float) for b in bursts], dtype=object)

    return train, bursts

def return_params():

    # define parameters
    T = [10, 30]
    D = [0.01, 0.02, 0.05, 0.1, 0.2]
    dt = 1e-3
    N = 100
    train_rates = [5, 10, 20, 30, 50]
    rho = [2, 3, 5]
    max_burst = 150 

    # define parameter sets
    params = []

    for t in T:
        for d in D:
            for train_rate in train_rates:
                for r in rho:
                    rng = np.random.default_rng(0)
                    burst_rate = rng.uniform(0.2, 2.01) # burst rates = [0.2, 2.01]
                    single_burst_rate = min(r * train_rate, max_burst)
                    params.append([t, d, train_rate, burst_rate, single_burst_rate])

    return D, T, N, params