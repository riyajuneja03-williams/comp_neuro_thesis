import synspiketrain

# MaxInterval
def max_interval(spikes, MaxISIStart, MaxISIEnd, MinIntervalBetweenBursts, MinDurationBurst, MinNumSpikes):
    """
    Detects bursts in spike train using MaxInterval method.

    Parameters
    ----------
    MaxISIStart : float
        Maximum interspike interval at start of burst.
    MaxISIEnd : float
        Maximum interspike interval within burst.
    MinIntervalBetweenBursts : float
        Minimum interspike interval between separate bursts.
    MinDurationBurst : float
        Minimum duration of a single burst.
    MinNumSpikes : float
        Minimum number of spikes in a single burst.
    Spikes : np.array
        Array of spike times following poisson distribution

    Returns
    --------
    bursts : list
        List of time intervals corresponding to bursts in the spike train.
    """

    # define parameters (from NeuroExplorer manual)
    MaxISIStart = 0.17 #s
    MaxISIEnd = 0.3 #s
    MinIntervalBetweenBursts = 0.2 #s
    MinDurationBurst = 0.01 #s
    MinNumSpikes = 3

    bursts = []
    single_burst = []
    burst = False

    # algorithm
    for i in range(len(spikes) - 1):
        ISI = spikes[i+1] - spikes[i]
        if burst == True:
            if ISI < MaxISIEnd: # while ISIs < MaxISIEnd, spikes are included in burst
                  single_burst.append(spikes[i])
            else: # if ISI >= MaxISIEnd → burst ends
                  single_burst.append(spikes[i])
                  burst = False
                  # remove bursts with duration < MinDurationBurst or spikes < MinNumSpikes
                  if len(single_burst) >= MinNumSpikes and (single_burst[-1] - single_burst[0]) >= MinDurationBurst:
                      bursts.append(single_burst)
                      single_burst = []
        else:
            if ISI <= MaxISIStart: # scan spike train until find ISI ≤ MaxISIStart
                burst = True
                single_burst.append(spikes[i])
    return(bursts)