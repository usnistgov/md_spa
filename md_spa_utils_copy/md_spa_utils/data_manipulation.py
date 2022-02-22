
import numpy as np

def basic_stats(data, confidence=0.95):
    """
    Given a set of data, calculate the mean and standard error

    Parameters
    ----------
    data : numpy.ndarray
        1D list or array of data
    confidence : float, Optional, default=0.95
        Confidence Interval

    Returns
    -------
    mean : float
        Average value
    std_error : float
        Standard error of the data
    interval : float
        When added and subtracted from the mean, forms the confidence interval
        
    """
    data = np.array(data,np.float)
    se = np.std(data)/np.sqrt(len(data))

    return np.mean(data), se

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

