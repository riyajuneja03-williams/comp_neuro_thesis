import synspiketrain

# MaxInterval
def max_interval(spikes, MaxISIStart = 0.17, MaxISIEnd = 0.3, MinIntervalBetweenBursts = 0.2, MinDurationBurst = 0.01, MinNumSpikes = 3):
    """
    Detects bursts in spike train using MaxInterval method.

    Parameters
    ----------
    MaxISIStart : float
        Maximum interspike interval at start of burst.
        Default: 0.17
    MaxISIEnd : float
        Maximum interspike interval within burst.
        Default: 0.3
    MinIntervalBetweenBursts : float
        Minimum interspike interval between separate bursts.
        Default: 0.2
    MinDurationBurst : float
        Minimum duration of a single burst.
        Default: 0.01
    MinNumSpikes : float
        Minimum number of spikes in a single burst.
        Default: 3
    Spikes : np.array
        Array of spike times following poisson distribution

    Returns
    --------
    bursts : list
        List of time intervals corresponding to bursts in the spike train.
    """

    bursts = []
    single_burst = []
    burst = False

    # algorithm
    for i in range(len(spikes) - 1):
        ISI = spikes[i+1] - spikes[i]
        if burst == True: 
            if ISI < MaxISIEnd:  # while ISIs < MaxISIEnd, spikes are included in burst
                single_burst.append(spikes[i + 1])
            else:  # if ISI >= MaxISIEnd → burst ends
                burst = False
                # merge bursts less than MinIntervalBetweenBursts apart
                if bursts and (single_burst[0] - bursts[-1][-1]) < MinIntervalBetweenBursts: 
                    prev_burst = bursts.pop()
                    prev_burst.extend(single_burst)
                    bursts.append(prev_burst)
                # remove bursts with duration < MinDurationBurst or spikes < MinNumSpikes
                if (
                    len(single_burst) >= MinNumSpikes
                    and (single_burst[-1] - single_burst[0]) >= MinDurationBurst
                ):
                    bursts.append(single_burst)
                    single_burst = []
        else:
            if ISI <= MaxISIStart:  # scan spike train until find ISI ≤ MaxISIStart
                burst = True
                single_burst = [spikes[i], spikes[i + 1]]

    # if spike train ends, evaluate burst 
    if burst and (len(single_burst) >= MinNumSpikes and (single_burst[-1] - single_burst[0]) >= MinDurationBurst):
        bursts.append(single_burst) 

    return(bursts)