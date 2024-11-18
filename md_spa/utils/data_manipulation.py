
import numpy as np
import scipy.stats

def basic_stats(data, axis=None, data_type="individuals", error_type="standard_deviation", error_descriptor="mean", confidence=0.95, population_dist_type="unknown", verbose=False):
    """
    Given a set of data, calculate the mean and standard error

    Valued of NaN are removed from the data set.

    Parameters
    ----------
    data : numpy.ndarray
        Array or matrix of data. If there is more than one dimension and an axis isn't given, then all values are used in the population.
    axis : int, Optional, default=None
        Axis over which to compute operation as defined in numpy.
    data_type : str, Optional, default="individuals"
        Define the data provided:

        - "indivduals": Provided data represent a sample of a population
        - "means": Provided data represent sample means

    error_type : str, Optional, default="standard_deviation"
        Type of error to be output, can be "standard_deviation", "variance", or "confidence" for either the population or sample mean distribution (unless ``data_type="means"``. Note that the standard error is the standard deviation of the sample mean distribution so the keyword arguements would provide this result when a sample population (e.g., ``data_type="individuals"``) is provided with: ``error_type="standard_deviation", error_descriptor="mean"``, or when sample means are provided (e.g., ``data_type="means"``) the keyword arguements would be: ``error_type="standard_deviation", error_descriptor="sample"``
    error_descriptor : str, Optional, default="mean"
        Specify whether the error value should represent the distribution of the provided data or the sample mean distribution with "sample" or "mean" respectively.
    confidence : float, Optional, default=0.95
        Confidence Interval certainty, used when ``error_type = "confidence"``. If the number of samples in the distribution is less than 30 and ``population_dist_type="unknown"``, then t-statistics are used, if  n>30 or population_dist_type="normal" then z-statistics are used.
    population_dist_type : str, Optional, default="unknown"
        If the population distribution type is unknown, then the size of the sample must be greater than or equal to 30 for z-statistics to be used. If the population is known to be normal, then this action can be overwritten with the specificiation of ``population_dist_type="normal"``.
    verbose : bool, Optional, default=False
        Display algorithmic decisions

    Returns
    -------
    mean : float
        Average value
    spread : float
        Descriptor of either population or mean distributions defined with ``error_type``.
        
    """

    if verbose:
        print("Evaluating {}, to describe the {} with the {} of a(n) {} population".format(
            data_type, error_descriptor, error_type, population_dist_type))

    data = np.array(data, float)
    if not isiterable(data):
        raise ValueError("Input data is not iterable")
 
    if np.size(data) == 0 and np.size(data) == np.isnan(data).sum():
        mean = np.nan
        spread = np.nan

        return mean, spread

    if data_type == "means" and error_descriptor == "mean":
        raise ValueError("If the provided data is a set of sample means, the error_descriptor should be 'sample', since the mean of this dataset is already the standard error.")

    # Find the sample size(s)
    if axis is None:
        lx = np.prod(np.shape(data)) - np.isnan(data).sum()
    elif isinstance(axis, tuple):
        not_axis = (x for x in range(len(np.shape(data))) if x not in axis)
        lx = np.prod([x for i,x in enumerate(np.shape(data)) if i in axis])*np.ones(not_axis)
        lx -= np.isnan(data).sum(axis)
    elif isinstance(axis, int):
        lx = np.shape(data)[axis] - np.isnan(data).sum(axis)
    else:
        raise ValueError("This entry for 'axis' is not valid: {}".format(axis))

    mean = np.nanmean(data, axis=axis)
    if error_descriptor == "mean": # If standard error is desired from sample population
        spread = np.nanstd(data, axis=axis)
        spread = spread / np.sqrt(lx)
    elif error_descriptor == "sample":
        spread = np.nanstd(data, axis=axis)
    else:
        raise ValueError("error_descriptor, {}, is not supported, should be 'mean' or 'sample'".format(error_descriptor))
   
    if population_dist_type not in ["normal", "unknown"]:
        raise ValueError("population_dist_type, {}, is not supported, should be 'normal' or 'unknown'".format(population_dist_type))
        
    if error_type == "standard_deviation":
        pass
    elif error_type == "variance":
        spread = spread**2
    elif error_type == "confidence":
        if np.all(lx >= 30) or population_dist_type == "normal":
            spread = spread * scipy.stats.norm.interval(confidence, scale=1, loc=0)[-1]
        else:
            if verbose:
                print("Using t-distribution")
            spread = spread * scipy.stats.t.interval(confidence, scale=1, loc=0, df=lx-1)[-1]
    else:
        raise ValueError("error_type, {}, is not supported".format(error_type))

    return mean, spread

def skewness(data, kwargs={}):
    """
    Given a set of data, calculate the skewness and its standard error. If skewness/SE is greater than 1.96 then the distribution is not in the 95% confidence interval of normality. By default the adjusted Fisher-Pearson standardized moment coefficient to correct for sample bias with the default kwargs.

    Valued of NaN are removed from the data set.

    Parameters
    ----------
    data : numpy.ndarray
        1D list or array of data
    kwargs : dict, Optional, default={"bias": False, "nan_policy": "omit"}
        Keyword arguments for ``scipy.stats.skew``

    Returns
    -------
    skewness : float
        By default the adjusted Fisher-Pearson skewness is calculated with ``scipy.stats.shew``
    skew_se : float
        Standard error of the skewness that is meaningful for the adjusted Fisher-Pearson skewness and is the square root of 6*n*(n-1)/((n-2)*(n+1)*(n+3)).
        
    """

    tmp_kwargs = {"bias": False, "nan_policy": "omit"}
    tmp_kwargs.update(kwargs)

    if not isiterable(data):
        raise ValueError("Input data is not iterable")
 
    if len(data) != 0 or len(data) != np.isnan(data).sum():
        data = np.array(data,np.float)
        lx = len(data) - np.isnan(data).sum()
        skewness = scipy.stats.skew(data, **tmp_kwargs)
        skew_se = np.sqrt(6.*lx*(lx-1.)/((lx-2.)*(lx+1.)*(lx+3.)))

    else:
        skewness = np.nan
        skew_se = np.nan

    return skewness, skew_se

def isiterable(array):
    """
    Check if variable is an iterable type with a length (e.g. np.array or list)

    Note that this could be test is ``isinstance(array, Iterable)``, however ``array=np.array(1.0)`` would pass that test and then fail in ``len(array)``.

    Taken from github.com/jaclark5/despasito

    Parameters
    ----------
    array
        Variable of some type, that should be iterable

    Returns
    -------
    isiterable : bool
        Will be True if indexing is possible and False is not

    """

    array_tmp = np.array(array, dtype=object)
    tmp = np.shape(array_tmp)

    if tmp:
        isiterable = True
    else:
        isiterable = False

    return isiterable

def isfloat(string):
    """
    Check if string variable is actually a float. This function allows exponential notation.

    Parameters
    ----------
    string
        String that may or may not be a float

    Returns
    -------
    isfloat : bool
        Will be True if the string can be converted into a float

    """

    try:
        float(string)
        flag = True
    except:
        flag = False

    return flag

def array2dict(array, keys):
    """
    Array to dict will group values into a dictionary

    Parameters
    ----------
    array : list
        List of lists of equal length. The first index is used as a key in the resulting dictionary
    keys : list[str]
        Strings representing each entry after the key, so ``len(array[i])`` should be one plus the length of ``keys``
   
    Returns
    -------
    dictionary : dict
        Dictionary of values
    """

    dictionary = {}
    for line in array:
       if line[0] not in dictionary:
           dictionary[line[0]] = {key: [] for key in keys}
       if len(keys) != len(line[1:]):
           raise ValueError("The number of keys must equal the number of entries len(array[i][1:])")
       for i in range(1,len(line)):
           dictionary[line[0]][keys[i-1]].append(line[i])

    return dictionary

def autocorrelation(x, mode="fft"):
    """
    Calculate the Autocorrelation function using FFT and multiple time origins.

    Parameters
    ----------
    x : numpy.ndarray
        Input function for which to calculate the autocorrelation function
    mode : str, Optional, default="fft"
        Method of calculating the autocorrelation function. FFT is two orders of magnitude faster than the ``loop`` method, and twice as fast as the numpy.correlate.
       
        - fft: Using Fourier transform to calculate the autocorrelation funct
        - numpy: Use np.correlate with multiple start times
        - loop: Hard coded loop taken from methematical definition

    Returns
    -------
    Cx : numpy.ndarray
        Autocorrelation function
    """

    lx = len(x)
    norm = (lx-np.arange(0,lx))

    if mode == "fft":
        Cx = np.fft.ifft( np.abs(np.fft.fft(x, n=2*lx))**2)[:lx].real
    elif mode == "numpy":
        Cx = np.correlate(x, x, mode='full')[lx-1:]
    elif mode == "loop":
        Cx = np.zeros(lx)
        for j in range(lx):
            Cx[:lx-j] += x[j]*x[j:]
    else:
        raise ValueError("The autocorrelation method, {}, is not supported".format(mode))

    return Cx/norm

def remove_duplicate_pairs(array):
    """
    Given an iterable list return entries with unique values for the first two indices

    Parameters
    ----------
    array : list
        Iterable strucure with a second dimension of at least two

    Returns
    -------
    new_array : list
        Iterable strucure without repeating entries

        
    """

    if not isiterable(array):
        raise ValueError("Provided array should be iterable")
    if array and (not isiterable(array[0]) or len(array[0]) < 2):
        raise ValueError("Each element in the second dimension must be iterable and at least of length two.")

    new_array = []
    for tmp_set in array:
        if tmp_set[0] not in [tmp[0] for tmp in new_array]:
            new_array.append(tmp_set)
        elif tmp_set[1] not in [tmp for tmp in [tmp2[1] for tmp2 in new_array if tmp2[0]==tmp_set[0]]]:
            new_array.append(tmp_set)

    return new_array

def block_average(xdata, ydata, block_size):
    """Block average data according to block size.
    
    If the array is not evenly divisible by the block size, the
    initial entries in the array are discarded. Note that the
    size of the blocks will be rounded up to the nearest block
    size allowed by the array spacing.

    Parameters
    ----------
    xdata : numpy.ndarray
        Independent data array
    ydata : numpy.ndarray
        Dependent data array
    block_size : float
        Block size for averaging in units of ``xdata``

    Returns
    -------
    ydata_avg : float
        Overall average of ydata
    ydata_std : float
        Standard deviaiton between averages of ydata blocks
    xdata_new : numpy.ndarray
        xdata values in the middle of the blocks being averaged
    ydata_new : numpy.ndarray
        ydata values, averaged over blocks
        
    """
    npoints = np.where(xdata-xdata[0]-block_size > 0)[0][0]
    nblocks = len(xdata) // npoints
    offset = len(xdata) % npoints
    
    xdata_new = xdata[offset] + block_size/2 + block_size * np.arange(nblocks, step=1)
    ydata_new = np.array([np.mean(ydata[-(n+1)*npoints+1:-n*npoints+1]) for n in range(nblocks)])
        
    return np.mean(ydata_new), np.std(ydata_new), xdata_new, ydata_new