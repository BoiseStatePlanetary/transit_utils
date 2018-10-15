from statsmodels.robust.scale import mad
from scipy.signal import medfilt
import numpy as np
from PyAstronomy.pyasl import isInTransit

__all__ = ['calc_phi', 'calc_eclipse_time', 'transit_duration',
    'fit_eclipse_bottom', 'supersample_time', 'median_boxcar_filter',
    'bindata', 'flag_outliers']

def calc_phi(time, params):
    """Calculates orbital phase assuming zero eccentricity

    Args: 
        time: observational time (same units at orbital period)
        params: dict of floats/numpy arrays, including
            params.per - orbital period (any units)
            params.T0 - mid-transit time (same units as period)

    Returns:
        orbital phase
    """

    T0 = params.T0
    per = params.per

    return ((time - T0) % per)/per

def calc_eclipse_time(params):
    """Calculates mid-eclipse time, assuming zero eccentricity

    Args:
        params: dict of floats/numpy arrays, including
            params.per - orbital period (any units)
            params.T0 - mid-transit time (same units as period)

    Returns:
        mid-eclipse time
    """

    T0 = params.T0
    per = params.per

    return T0 + 0.5*per

def transit_duration(params, which_duration="full"):
    """Calculates transit duration

    Args:
        params: dict of floats/numpy arrays, including
            params.per - orbital period (any units)
            params.p - ratio of the planet to the star's radius
            params.b - impact parameter in units of stellar radius
            params.a - semi-major axis in units of stellar radius
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

    if(isinstance(params, dict)):
        period = params['per']
        rp = params['p']
        b = params['b']
        sma = params['a']
    else:
        period = params.per
        rp = params.p
        b = params.b
        sma = params.a

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
            params.per - orbital period, same units as time
            params.T0 - mid-transit time
            params.p - ratio of the planet to the star's radius
            params.b - impact parameter in units of stellar radius
            params.a - semi-major axis in units of stellar radius
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

    eclipse_bottom = 0.
    period = params.per
    TE = calc_eclipse_time(params)

    # In some cases, the planet is never totally occulted.
    #   For those cases, fit eclipse with quadratic and return min value.
    if((1. - params.p)**2 - params.b**2. < 0.):
        # Find in-eclipse points
        dur = transit_duration(params, which_duration='full')
        ind = isInTransit(time, TE, period, 0.5*dur, boolOutput=True)

        # Fit quadratic to eclipse to estimate minimum
        coeffs = np.polyfit(time[ind], data[ind], 2)
        eclipse_bottom = np.polyval(coeffs, -coeffs[1]/2./coeffs[0])

    else:
        # Find in-eclipse points
        dur = transit_duration(params, which_duration='short')

        ind = isInTransit(time, TE, period, 0.5*dur, boolOutput=True)
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

def median_boxcar_filter(time, data, window_length=None, endpoints='reflect',
        mask_ind=None):
    """
    Creates median boxcar filter and deals with endpoints

    Parameters
    ----------
    time : numpy array
        times associated with data array
    data : numpy array 
       Data array
    window_length: int
        A scalar giving the size of the median filter window
    endpoints : str
        How to deal with endpoints. 
        Only option right now is 'reflect', which extends the data array
        on both ends by reflecting the data
    mask_ind : optional, indices
        If None, ignored
        If an array of indices, it represents all the points to be masked 
        during the boxcar filtering.

    Returns
    -------
    filter : numpy array
        The filter array
    """

    filter_array = np.copy(data)
    # Create filter array
    if(endpoints == 'reflect'):
        last_index = len(data) - 1

        # If mask_ind provided, interpolate across the masked points
        if(mask_ind is not None):
            not_mask_ind = ~mask_ind
            filter_array[mask_ind] = np.interp(time[mask_ind],
                    time[not_mask_ind], filter_array[not_mask_ind])
            
        filter_array =\
                np.concatenate((np.flip(filter_array[0:window_length], 0),
                    filter_array,
                    filter_array[last_index - window_length:last_index]))

        # Make filter
        # Check that window_length is odd
        if(window_length % 2 == 0):
            window_length += 1
        filt = medfilt(filter_array, window_length)

        filt = filt[window_length:window_length + last_index + 1]

    return filt

def bindata(time, data, binsize, 
        times_to_try=None, bin_calc='median', err_calc='mad'):
    """
    Bins data array

    Parameters
    ----------
    time : numpy array
        Array of times
    data : numpy array
        data array
    binsize : float
        Width of bins in same units at time
    times_to_try : numpy array
        times to try as the bin centers
    bin_calc : str
        Method to use to calculate datum in each bin. 
        Can be either 'mean' or 'median'
    err_calc : str
        Method to use to calculate uncertainty in each bin. 
        Can be either 'std' or 'mad'
        Default ('mad') to using 1.4826 x median absolute deviation --
        https://en.wikipedia.org/wiki/Median_absolute_deviation

    Returns
    -------
    binned_time : numpy array
        Time binned
    binned_data : numpy array
        Data binned
    binned_err : numpy array
    """

    # 2018 May 23 - There are not always points in each time bin,
    #   so we will TRY to find points but will not always find them.
    if(times_to_try is None):
        times_to_try = np.arange(np.min(time) + 0.5*binsize, 
                np.max(time) - 0.5*binsize, binsize)

    binned_time = np.array([])
    binned_data = np.array([])
    binned_err = np.array([])

    if(bin_calc == 'median'):
        bin_calc_func = np.nanmedian
    elif(bin_calc == 'mean'):
        bin_calc_func = np.nanmean

    if(err_calc == 'mad'):
        err_calc_func = lambda x : mad(x)/np.sqrt(len(x))
    elif(err_calc == 'std'):
        err_calc_func = lambda x : np.nanstd(x)/np.sqrt(len(x))

    for i in range(len(times_to_try)):
        ind = np.argwhere(np.abs(times_to_try[i] - time) <= binsize)

        if(ind.size > 0): 
            # Remove nans, too
            cur_time = time[ind[~np.isnan(data[ind])]]
            cur_data = data[ind[~np.isnan(data[ind])]]
            
            if(cur_data.size > 0):
                binned_time = np.append(binned_time, times_to_try[i])
                binned_data = np.append(binned_data, bin_calc_func(cur_data))

                # Check for bad error value
                try_error = err_calc_func(cur_data)
                if(try_error == 0.):
                    binned_err = np.append(binned_err, 1.)
                else:
                    binned_err = np.append(binned_err, err_calc_func(cur_data))

    return binned_time, binned_data, binned_err

def flag_outliers(data, outlier_group=5, num_std_desired=10.):
    """
    Flags outliers in ordered data series

    Parameters
    ----------
    data : array
        ordered data array
    outlier_group : int, optional
        number of datapoints to group together in calculating standard
        deviation; defaults to 5
    num_std_desired : float, optional
        number of standard deviations beyond which to declare an outlier

    Returns
    -------
    mask of non-outlier points
    """

    # Pad out data array to have outlier_group x N number of elements
    num_extra_elements = outlier_group - (len(data) % outlier_group)

    copy_data = np.append(data, np.zeros(num_extra_elements))

    # Using the outer product makes outlier_group copies of each standard 
    # deviation and median, allowing me to compare to each data point
    std = np.outer(mad(copy_data.reshape(-1, 5), axis=1),
            np.ones(outlier_group))
    med = np.outer(np.nanmedian(copy_data.reshape(-1, 5), axis=1),
        np.ones(outlier_group))

    # Check for NaN and zeros
    not_nan_ind = ~np.isnan(std) & (std != 0)
    num_std = np.zeros_like(med)
    num_std[not_nan_ind] =\
            abs(((copy_data.reshape(-1, 5))[not_nan_ind] -
                med[not_nan_ind])/std[not_nan_ind])
    num_std = num_std.flatten()

    ind = num_std < num_std_desired

    # Unpad index array
    ind = ind[0:len(data)]

    return ind
