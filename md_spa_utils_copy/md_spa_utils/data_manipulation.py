
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
