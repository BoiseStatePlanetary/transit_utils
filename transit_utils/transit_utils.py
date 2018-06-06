import numpy as np
from PyAstronomy.pyasl import isInTransit

def calc_phi(time, params):
    """Calculates orbital phase assuming zero eccentricity

    Args: 
        time: observational time (same units at orbital period)
        params: dict of floats/numpy arrays, including
            params["per"] - orbital period (any units)
            params["T0"] - mid-transit time (same units as period)

    Returns:
        orbital phase
    """

    T0 = params['T0']
    per = params['per']

    return ((time - T0) % per)/per

def calc_eclipse_time(params):
    """Calculates mid-eclipse time, assuming zero eccentricity

    Args:
        params: dict of floats/numpy arrays, including
            params["per"] - orbital period (any units)
            params["T0"] - mid-transit time (same units as period)

    Returns:
        mid-eclipse time
    """

    T0 = params['T0']
    per = params['per']

    return T0 + 0.5*per

def transit_duration(params, which_duration="full"):
    """Calculates transit duration

    Args:
        params: dict of floats/numpy arrays, including
            params["per"] - orbital period (any units)
            params["p"] - ratio of the planet to the star's radius
            params["b"] - impact parameter in units of stellar radius
            params["a"] - semi-major axis in units of stellar radius
        which_duration (str):
            "full" - time from first to fourth contact
            "center" - time from contact to contact between planet's center and
                stellar limb
            "short" - time from second to third contact

    Raises:
        ValueError: If which_duration is not 'full', 'center', or 'short'.

    Returns:
        transit_duration: transit duration in same units as period
    """

    period = params['per']
    rp = params['p']
    b = params['b']
    sma = params['a']

    if(which_duration == "full"):
        return period/np.pi*np.arcsin(np.sqrt((1. + rp)**2 - b**2)/sma)
    elif(which_duration == "center"):
        return period/np.pi*np.arcsin(np.sqrt(1. - b**2)/sma)
    elif(which_duration == "short"):
        return period/np.pi*np.arcsin(np.sqrt((1. - rp)**2 - b**2)/sma)
    else:
        raise ValueError("which_duration must be 'full', 'center', 'short'!")

def fit_eclipse_bottom(time, data, params, zero_eclipse_method="mean"):
    """Calculates the eclipse bottom to set the zero-point in the data

    Args:
        time: observational time (same units at orbital period)
        data: observational data
        params: dict of floats/numpy arrays, including
            params["per"] - orbital period, same units as time
            params["T0"] - mid-transit time
        zero_eclipse_method (str):
            Which method used to set zero-point -
                "mean" - Use in-eclipse average value
                "median" - Use in-eclipse median value

    Returns:
        eclipse bottom value
    """

    if(zero_eclipse_method == "mean"):
        calc_method = np.nanmean
    elif(zero_eclipse_method == "median"):
        calc_method = np.nanmedian
    else:
        raise ValueError("which_method should be mean or median!")

    # Find in-eclipse points
    period = params["per"]
    TE = calc_eclipse_time(params)
    dur = transit_duration(which_duration="short")
    ind = isInTransit(time, TE, period, 0.5*dur, boolOutput=True)

    eclipse_bottom = 0.
    if(ind.size > 0):
        eclipse_bottom = calc_method(data[ind])

    return eclipse_bottom

def supersample_time(time, supersample_factor, exp_time):
    """Creates super-sampled time array

    Args:
        time (numpy array): times
        supersample_factor (int): number of points subdividing exposure
        exp_time (float): Exposure time (in same units as `time`)

    Returns:
        Returns the super-sampled time array
    """

    if supersample_factor > 1:
        time_offsets = np.linspace(-exp_time/2., exp_time/2., 
                supersample_factor)
        time_supersample = (time_offsets +\
                time.reshape(time.size, 1)).flatten()
    else: 
        time_supersample = time

    return time_supersample
