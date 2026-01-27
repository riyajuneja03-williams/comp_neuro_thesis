# synthetic data
# generates a random spike train at some rate and plots using a built-in matplotlib graph

import numpy as np
from matplotlib import pyplot as plt

def poisson_burst(
    rate,
    burst_rate,
    T,
    tau_ref=0.05,
    tau_burst=0.1,
    prob_burst=0.2,
    prob_end=0.5,
    rng=None,
    min_spikes=3
):
    """
    Generate Poisson spike train of length T, rate rate, and refractory period.
    There is a chance of bursting at rate burst_rate

    Parameters
    ----------
    rate : float
        Baseline firing rate of the spike train
    burst_rate : float
        Elevated firing rate used during a burst
    T : float
        Length of time for spike train (seconds)
    dt : float
        (optional) Small dt to determine binning of spike train
    tau_ref : float
        (optional) Refractory period, shortest time between spikes
    tau_burst : float
        (optional) Minimum time between consecutive bursts
    prob_burst : float
        (optional) Probability to enter a burst.
    prob_end : float
        (optional) Probability to end the burst.
    rng : numpy.random.Generator, optional
        Defaults to None, otherwise sets random seed
    min_spikes: float
        (optional) Minimum number of spikes to constitute a burst

    Returns
    --------
    np.ndarray
        array of spike times following poisson distribution
    np.ndarray
        array of arrays of spike times representing each burst
    """

    # set random seed
    rng = np.random.default_rng(rng)

    # maximum possible firing rate and guard against tau_ref = 0
    r_ceil = 1 / max(tau_ref, 1e-12)

    # enforce feasbility, rate must be < 1/tau_ref
    if rate >= 1 / max(tau_ref, 1e-12):
        raise ValueError(
            f"Requested rate {rate:.3f} Hz >= 1/tau_ref = {r_ceil:.3f} Hz. Decrease rate or tau_ref."
        )
    if burst_rate > r_ceil:
        raise ValueError(
            f"Requested burst rate {burst_rate:.3f} Hz > 1/tau_ref = {r_ceil:.3f} Hz. Decrease burst rate or tau_ref"
        )

    # empty array to hold spike times
    spikes = []
    all_bursts = []
    t = 0

    # bool for in burst
    in_burst = False

    # time of last spike in burst
    last_burst = -1

    # loop through time and draw ISIs
    while t < T:
        burst = None
        # enter burst if prob_burst met and time since last_burst is greater than tau_burst
        if rng.random() < prob_burst and (t - last_burst) >= tau_burst:
            in_burst = True
            burst = []

            # in the burst, update spike times
            while in_burst and t < T:
                # draw ISIs from burst_rate distribution and include tau_ref
                t += rng.exponential(1 / burst_rate - tau_ref) + tau_ref

                # only add spike if smaller than cutoff time
                if t < T:
                    spikes.append(t)
                    burst.append(t)

                # check if probability met to end the burst
                # record exiting burst and last spike in burst
                if rng.random() < prob_end:
                    in_burst = False
                    last_burst = t
                    if len(burst) >= min_spikes:
                        all_bursts.append(burst)
        else:
            # draw ISI from background rate if not in burst
            t += rng.exponential(1 / rate - tau_ref) + tau_ref

            # only add spike if smaller than cutoff time
            if t < T:
                spikes.append(t)
                if burst is not None:
                    burst.append(t)
    return np.array(spikes), all_bursts

def return_params():
    T = 1
    dt = 1e-3
    N = 100

    # refractory parameters
    tau_ref = 0
    tau_burst = 0

    # define parameter sets
    rates = np.arange(2, 102, 4)
    rates_list = rates.tolist()
    prob_burst = [0.2, 0.2, 0.2, 0.5, 0.5, 0.5, 0.8, 0.8, 0.8]
    prob_exit = [0.2, 0.5, 0.8, 0.2, 0.5, 0.8, 0.2, 0.5, 0.8]
    params = []
    probs = [[prob_burst[0], prob_exit[0]], 
            [prob_burst[1], prob_exit[1]], 
            [prob_burst[2], prob_exit[2]], 
            [prob_burst[3], prob_exit[3]], 
            [prob_burst[4], prob_exit[4]], 
            [prob_burst[5], prob_exit[5]], 
            [prob_burst[6], prob_exit[6]], 
            [prob_burst[7], prob_exit[7]], 
            [prob_burst[8], prob_exit[8]]]

    for rate in rates_list:
        burst_rates = [2 * rate, 5 * rate, 10 * rate]
        for burst_rate in burst_rates:
            for prob in probs:
                burst_prob = prob[0]
                exit_prob = prob[1]
                params.append([rate, burst_rate, burst_prob, exit_prob])

    return T, N, params, tau_ref, tau_burst