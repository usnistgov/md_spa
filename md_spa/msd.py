
#import read_lammps as f
import sys
import numpy as np
import warnings
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.ndimage.filters import gaussian_filter1d
from scipy.stats import linregress
from scipy.interpolate import InterpolatedUnivariateSpline

import md_spa_utils.os_manipulation as om

def debye_waller(time, msd, use_frac=1, show_plot=False, save_plot=False, plot_name="debye-waller.png", verbose=False):
    """
    Analyzing the ballistic region of an MSD curve yields the debye-waller factor, which relates to the cage region that the atom experiences.

    Parameters
    ----------
    time : numpy.ndarray
        Time array of the same length at MSD
    msd : numpy.ndarray
        MSD array with one dimension
    use_frac : float, Optional, default=1
        Choose what fraction of the msd to use. This will cut down on comutational time in spending time on regions with poor statistics.
    save_plot : bool, Optional, default=False
        choose to save a plot of the fit
    show_plot : bool, Optional, default=False
        choose to show a plot of the fit
    plot_name : str, Optional, default="debye-waller.png"
        If ``save_plot==True`` the msd will be saved with the debye-waller factor marked
    verbose : bool, Optional, default=False
        Will print intermediate values or not
    
    Returns
    -------
    debye-waller : float

    """

    if not om.isiterable(time):
        raise ValueError("Given distances, time, should be iterable")
    else:
        time = np.array(time)
    if not om.isiterable(msd):
        raise ValueError("Given radial distribution values, msd, should be iterable")
    else:
        msd = np.array(msd)
 
    if len(msd) != len(time):
        raise ValueError("Arrays for time and msd are not of equal length.")

    time = time[:int(len(time)*use_frac)]
    msd = msd[:int(len(time)*use_frac)]

    spline = InterpolatedUnivariateSpline(time,msd, k=4)
    dspline = spline.derivative()
    extrema = dspline.roots().tolist()
    if len(extrema) > 2:
        dw = np.nan
        warnings.warn("Found {} extrema, consider smoothing the data with `smooth_sigma` option for an approximate value, and then increase the statistics used to calculate the msd.".format(len(extrema)))
    elif len(extrema) == 2:
        dw = spline(extrema[1])
        if verbose:
            print("Found debye waller factor to be {}".format(dw))
    else:
        warnings.warn("This msd array does not contain a caged-region")
        dw = np.nan

    if save_plot or show_plot:
        plt.figure(1)
        plt.plot(time,msd,"k",label="Data", linewidth=0.5)
        if len(extrema) > 1:
            for tmp in extrema[1:]:
                plt.plot([tmp,tmp],[0,np.max(msd)], linewidth=0.5)
        plt.xlabel("time")
        plt.ylabel("MSD")
        plt.tight_layout()
        if save_plot:
            plt.savefig(plot_name,dpi=300)
        plt.figure(2)
        plt.plot(time, dspline(time),"k", linewidth=0.5)
        plt.xlabel("time")
        plt.ylabel("$d MSD / dt$")
        plt.tight_layout()
        if show_plot:
            plt.show()
        plt.close("all")

    return dw

def find_diffusivity(time, msd, min_exp=0.991, min_Npts=10, skip=1, show_plot=False, save_plot=False, plot_name="diffusivity.png", verbose=False, dim=3, use_frac=1, min_R2=0.97):
    """
    Analyzing the long-time msd, to extract the diffusivity.

    Parameters
    ----------
    time : numpy.ndarray
        Time array of the same length at MSD .
    msd : numpy.ndarray
        MSD array with one dimension.
    min_exp : float, Optional, default=0.991
        Minimum exponent value used to determine the longest acceptably linear region.
    min_Npts : int, Optional, default=10
        Minimum number of points in the "best" region outputted.
    skip : int, Optional, default=1
        Number of points to skip over in scanning different regions. A value of 1 will be most thorough, but a larger value will decrease computation time.
    min_R2 : float, Optional, default=0.97
        Minimum allowed coefficient of determination to consider proposed exponent. This prevents linearity from skipping over curves.
    save_plot : bool, Optional, default=False
        choose to save a plot of the fit
    show_plot : bool, Optional, default=False
        choose to show a plot of the fit
    plot_name : str, Optional, default="debye-waller.png"
        If ``save_plot==True`` the msd will be saved with the debye-waller factor marked
    dim : int, Optional, default=3
        Dimensions of the system, usually 3
    verbose : bool, Optional, default=False
        Will print intermediate values or not
    use_frac : float, Optional, default=1
        Choose what fraction of the msd to use. This will cut down on comutational time in spending time on regions with poor statistics.
    
    Returns
    -------
    best : np.array
        Array containing the the following results that represent a region that most closely represents a slope of unity. This region must contain at least the minimum number of points, ``min_Npts``.

        - diffusivity (slope/2/dim)
        - standard error of diffusivity
        - time interval used to evaluate this slope
        - exponent of region (should be close to unity)
        - intercept
        - number of points in calculation

    longest : np.ndarray
        The results from the longest region that fits a linear model with an exponent of at least ``min_exp``. This array includes:

        - diffusivity
        - standard error of diffusivity
        - time interval used to evaluate this slope
        - exponent of region (should be close to unity)
        - intercept
        - number of points in calculation

    """

    if not om.isiterable(time):
        raise ValueError("Given distances, time, should be iterable")
    else:
        time = np.array(time)
    if not om.isiterable(msd):
        raise ValueError("Given radial distribution values, msd, should be iterable")
    else:
        msd = np.array(msd)

    if len(msd) != len(time):
        raise ValueError("Arrays for time and msd are not of equal length.")

    msd = msd[:int(len(time)*use_frac)]
    time = time[:int(len(time)*use_frac)]

    if min_Npts > len(time):
        warning.warn("Resetting minimum number of points, {}, to be within length of provided data * use_frac, {}".format(min_Npts,len(time)))
        min_Npts = len(time)-1

    best = np.array([np.nan for x in range(7)])
    longest = np.array([np.nan for x in range(7)])
    for npts in range(min_Npts,len(time)):
        for i in range(0,len(time)-npts,skip):
            t_tmp = time[i:(i+npts)]
            msd_tmp = msd[i:(i+npts)]
            d_tmp, stder_tmp, exp_tmp, intercept, r2_tmp = diffusivity(t_tmp, msd_tmp, verbose=verbose, dim=dim)

            if r2_tmp > min_R2:
                if np.abs(exp_tmp-1.0) < np.abs(best[4]-1.0) or np.isnan(best[4]):
                    best = np.array([d_tmp, stder_tmp, t_tmp[0], t_tmp[-1], exp_tmp, intercept, npts])

                if (exp_tmp >=  min_exp and longest[-1] <= npts) or np.isnan(longest[4]):
                    if (longest[-1] < npts or np.abs(longest[4]-1.0) > np.abs(exp_tmp-1.0)) or np.isnan(longest[4]):
                        longest = np.array([d_tmp, stder_tmp, t_tmp[0], t_tmp[-1], exp_tmp, intercept, npts])

                if verbose:
                    print("Region Diffusivity: {} +- {}, from Time: {} to {}, with and exponent of {} using {} points".format(d_tmp, stder_tmp, t_tmp[0], t_tmp[-1], exp_tmp, npts))

    if save_plot or show_plot:
        plt.plot(time,msd,"k",label="Data", linewidth=0.5)
        tmp_time = np.array([time[0],time[-1]])
        tmp_best = tmp_time*best[0]*2*dim+best[5]
        plt.plot(tmp_time,tmp_best, "g", label="Best", linewidth=0.5)
        tmp_longest = tmp_time*longest[0]*2*dim+longest[5]
        plt.plot(tmp_time,tmp_longest, "b", label="Longest", linewidth=0.5)
        plt.xlabel("time")
        plt.ylabel("MSD")
        plt.tight_layout()
        if save_plot:
            plt.savefig(plot_name,dpi=300)
        if show_plot:
            plt.show()
        plt.close("all")

    if verbose:
        print("Best Region Diffusivity: {} +- {}, from Time: {} to {}, with and exponent of {} using {} points".format(*best[:5],best[-1]))
        print("Longest Region Diffusivity: {} +- {}, from Time: {} to {}, with and exponent of {} using {} points".format(*best[:5],best[-1]))

    return best, longest

def diffusivity(time, msd, verbose=False, dim=3):
    """
    Analyzing the long-time msd, to extract the diffusivity. This entire region is used and so should be linear.

    Parameters
    ----------
    time : numpy.ndarray
        Time array of the same length at MSD in picoseconds.
    msd : numpy.ndarray
        MSD array with one dimension in angstroms.
    verbose : bool, Optional, default=False
        Will print intermediate values or not
    dim : int, Optional, default=3
        Dimensions of the system, usually 3
    
    Returns
    -------
    diffusivity : float
        Diffusivity in meters squared per picosecond
    sterror : float
        Standard error of diffusivity
    exponent : float
        Exponent of fit data, should be close to unity
    intercept : float
        Intercept of time versus msd for plotting purposes
    r_squared : float
        Coefficient of determination for the linear fitted log-log plot, from which the exponent is derived.

    """

    if not om.isiterable(time):
        raise ValueError("Given distances, time, should be iterable")
    else:
        time = np.array(time)
    if not om.isiterable(msd):
        raise ValueError("Given radial distribution values, msd, should be iterable")
    else:
        msd = np.array(msd)

    if len(msd) != len(time):
        raise ValueError("Arrays for time and msd are not of equal length.")

    # Find Exponent
    t_log = np.log(time-time[0])
    msd_log = np.log(msd-msd[0])
    t_log = t_log[1:]
    msd_log = msd_log[1:]
    result = linregress(t_log,msd_log)
    exponent = result.slope
    r_squared = result.rvalue**2

    result = linregress(time,msd)
    diffusivity = result.slope/2/dim
    sterror = result.stderr/2/dim
    intercept = result.intercept

    return diffusivity, sterror, exponent, intercept, r_squared

    
