
#import read_lammps as f
import sys
import numpy as np
import warnings
import os
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy.interpolate import InterpolatedUnivariateSpline

import md_spa_utils.data_manipulation as dm
import md_spa_utils.file_manipulation as fm

def keypoints2csv(filename, fileout="msd.csv", mode="a", delimiter=",", titles=None, additional_entries=None, additional_header=None, kwargs_find_diffusivity={}, kwargs_debye_waller={}, file_header_kwargs={}):
    """
    Given the path to a csv file containing msd data, extract key values and save them to a .csv file. The file of msd data should have a first column with distance values, followed by columns with radial distribution values. These data sets will be distinguished in the resulting csv file with the column headers

    Parameters
    ----------
    filename : str
        Input filename and path to lammps msd output file
    fileout : str, Optional, default="msd.csv"
        Filename of output .csv file
    mode : str, Optional, default="a"
        Mode used in writing the csv file, either "a" or "w".
    delimiter : str, Optional, default=","
        Delimiter between data in input file
    titles : list[str], Optional, default=None
        Titles for plots if that is specified in the ``kwargs_find_diffusivity`` or ``kwargs_debye_waller``
    additional_entries : list, Optional, default=None
        This iterable structure can contain additional information about this data to be added to the beginning of the row
    additional_header : list, Optional, default=None
        If the csv file does not exist, these values will be added to the beginning of the header row. This list must be equal to the `additional_entries` list.
    kwargs_find_diffusivity : dict, Optional, default={}
        Keywords for `find_diffusivity` function
    kwargs_debye_waller : dict, Optional, default={}
        Keywords for `debye_waller` function
    file_header_kwargs : dict, Optional, default={}
        Keywords for ``md_spa_utils.os_manipulation.file_header`` function    

    Returns
    -------
    csv file
    
    """
    if not os.path.isfile(filename):
        raise ValueError("The given file could not be found: {}".format(filename))

    data = np.transpose(np.genfromtxt(filename, delimiter=delimiter))

    if titles == None:
        titles = fm.find_header(filename, **file_header_kwargs)
    if len(titles) != len(data):
        raise ValueError("The number of titles does not equal the number of columns")

    if np.all(additional_entries != None):
        flag_add_ent = True
        if not dm.isiterable(additional_entries):
            raise ValueError("The provided variable `additional_entries` must be iterable")
    else:
        flag_add_ent = False
        additional_entries = []
    if np.all(additional_header != None):
        flag_add_header = True
        if not dm.isiterable(additional_header):
            raise ValueError("The provided variable `additional_header` must be iterable")
    else:
        flag_add_header = False

    if flag_add_ent:
        if flag_add_header:
            if len(additional_entries) != len(additional_header):
                raise ValueError("The variables `additional_entries` and `additional_header` must be of equal length")
        else:
            additional_header = ["-" for x in additional_entries]
            flag_add_header = True

    t_tmp = data[0]
    tmp_data = []
    for i in range(1,len(data)):
        if "title" not in kwargs_find_diffusivity:
            kwargs_find_diffusivity["title"] = titles[i]
        if "title" not in kwargs_debye_waller:
            kwargs_debye_waller["title"] = titles[i]
        best, longest = find_diffusivity(t_tmp, data[i], **kwargs_find_diffusivity)
        dw, tau = debye_waller(t_tmp, data[i], **kwargs_debye_waller)
        tmp_data.append(list(additional_entries)+[titles[i],dw, tau]+list(best)+list(longest))

    file_headers = ["Group", "DW [Ang^2]", "tau [ps]", "Best D [Ang^2/ps]", "B D SE", "B t_bound1 [ps]", "B t_bound2 [ps]", "B Exponent", "B Intercept [Ang^2]", "B Npts", "Longest D [Ang^2/ps]", "L D SE", "L t_bound1 [ps]", "L t_bound2 [ps]", "L Exponent", "L Intercept [Ang^2]", "L Npts"]
    if not os.path.isfile(fileout) or mode=="w":
        if flag_add_header:
            file_headers = list(additional_header) + file_headers
        fm.write_csv(fileout, tmp_data, mode=mode, header=file_headers)
    else:
        fm.write_csv(fileout, tmp_data, mode=mode)


def debye_waller(time, msd, use_frac=1, show_plot=False, save_plot=False, title=None, plot_name="debye-waller.png", verbose=False):
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
    title : str, Optional, default=None
        The title used in the msd plot, note that this str is also added as a prefix to the ``plot_name``.
    show_plot : bool, Optional, default=False
        choose to show a plot of the fit
    plot_name : str, Optional, default="debye-waller.png"
        If ``save_plot==True`` the msd will be saved with the debye-waller factor marked. The ``title`` is added as a prefix to this str
    verbose : bool, Optional, default=False
        Will print intermediate values or not
    
    Returns
    -------
    debye-waller : float

    """

    if not dm.isiterable(time):
        raise ValueError("Given distances, time, should be iterable")
    else:
        time = np.array(time)
    if not dm.isiterable(msd):
        raise ValueError("Given radial distribution values, msd, should be iterable")
    else:
        msd = np.array(msd)
 
    if len(msd) != len(time):
        raise ValueError("Arrays for time and msd are not of equal length.")

    time = time[:int(len(time)*use_frac)]
    msd = msd[:int(len(time)*use_frac)]
    logtime = np.log10(time)
    logmsd = np.log10(msd)
    if np.isnan(logmsd[0]):
        logtime = logtime[1:]
        logmsd = logmsd[1:]

    spline = InterpolatedUnivariateSpline(logtime,logmsd, k=5)
    if np.all(np.isnan(logmsd)):
        raise ValueError("Spline could not be created with provided data:\n{}\n{}".format(time,msd))
    dspline = spline.derivative()

    extrema = dspline.derivative().roots().tolist()
    extrema_concavity = dspline.derivative().derivative()
    dw = np.ones(4)*np.nan
    tau = np.ones(4)*np.nan
    min_value = np.ones(4)*np.inf
    i = 0
    for min_max in extrema:
        if extrema_concavity(min_max) > 0:
            tau[i] = 10**min_max
            dw[i] = 10**spline(min_max)
            min_value[i] = dspline(min_max)
            i += 1
        if i == 3:
            break

    # Cut off minima after deepest
    ind_min = np.where(min_value==np.min(min_value))[0][0]
    for i in range(ind_min+1,4):
        min_value[i] = np.inf
        dw[i] = np.nan
        tau[i] = np.nan

    if np.all(np.isnan(dw)):
        warnings.warn("This msd array does not contain a caged-region")
    else:
        tau = np.array([x for _,x in sorted(zip(min_value,tau))])
        dw = np.array([x for _,x in sorted(zip(min_value,dw))])
        if verbose:
            print("Found debye waller factor to be {} at {}".format(dw, tau))

    if save_plot or show_plot:
        plt.figure(1)
        plt.plot(time,msd,"k",label="Data", linewidth=0.5)
        if not np.all(np.isnan(dw)):
            for tmp_tau in tau:
                plt.plot([tmp_tau,tmp_tau],[0,np.max(msd)], linewidth=0.5)
        plt.xlabel("time")
        plt.ylabel("MSD")
        if title != None:
            plt.title(title)
        plt.tight_layout()
        if save_plot:
            if title != None:
                tmp = os.path.split(plot_name)
                plot_name = os.path.join(tmp[0],title.replace(" ", "")+"_"+tmp[1])
            plt.savefig(plot_name,dpi=300)
        plt.figure(2)
        yarray = dspline(logtime)
        plt.plot(logtime, yarray,"k", linewidth=0.5)
        if not np.all(np.isnan(dw)):
            for tmp_tau in tau:
                plt.plot(np.log10([tmp_tau,tmp_tau]),[0,np.max(yarray)], linewidth=0.5)
        plt.xlabel("log(t)")
        plt.ylabel("$d log(MSD) / d log(t)$")
        plt.tight_layout()
        if save_plot:
            tmp = os.path.split(plot_name)
            tmp_plot_name = os.path.join(tmp[0],"dlog_"+tmp[1])
            plt.savefig(tmp_plot_name,dpi=300)
        if show_plot:
            plt.show()
        plt.close("all")

    return dw, tau

def find_diffusivity(time, msd, min_exp=0.991, min_Npts=10, skip=1, show_plot=False, title=None, save_plot=False, plot_name="diffusivity.png", verbose=False, dim=3, use_frac=1, min_R2=0.97):
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
    title : str, Optional, default=None
        The title used in the msd plot, note that this str is also added as a prefix to the ``plotname``.
    show_plot : bool, Optional, default=False
        choose to show a plot of the fit
    plot_name : str, Optional, default="debye-waller.png"
        If ``save_plot==True`` the msd will be saved with the debye-waller factor marked, The ``title`` is added as a prefix to this str
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

    if not dm.isiterable(time):
        raise ValueError("Given distances, time, should be iterable")
    else:
        time = np.array(time)
    if not dm.isiterable(msd):
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
                    print("Region Diffusivity: {} +- {}, from Time: {} to {}, with and exponent of {} using {} points, Exp Rsquared: {}".format(d_tmp, stder_tmp, t_tmp[0], t_tmp[-1], exp_tmp, npts, r2_tmp))

    if save_plot or show_plot:
        plt.plot(time,msd,"k",label="Data", linewidth=0.5)
        tmp_time = np.array([time[0],time[-1]])
        tmp_best = tmp_time*best[0]*2*dim+best[5]
        plt.plot(tmp_time,tmp_best, "g", label="Best", linewidth=0.5)
        tmp_longest = tmp_time*longest[0]*2*dim+longest[5]
        plt.plot(tmp_time,tmp_longest, "b", label="Longest", linewidth=0.5)
        plt.xlabel("time")
        plt.ylabel("MSD")
        if title != None:
            plt.title(title)
        plt.tight_layout()
        if save_plot:
            if title != None:
                tmp = os.path.split(plot_name)
                plot_name = os.path.join(tmp[0],title.replace(" ", "")+"_"+tmp[1])
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

    if not dm.isiterable(time):
        raise ValueError("Given distances, time, should be iterable")
    else:
        time = np.array(time)
    if not dm.isiterable(msd):
        raise ValueError("Given radial distribution values, msd, should be iterable")
    else:
        msd = np.array(msd)

    if len(msd) != len(time):
        raise ValueError("Arrays for time and msd are not of equal length.")

    # Find Exponent
    t_log = np.log(time[1:]-time[0])
    msd_log = np.log(msd[1:]-msd[0])
    result = linregress(t_log,msd_log)
    exponent = result.slope
    r_squared = result.rvalue**2

    result = linregress(time,msd)
    diffusivity = result.slope/2/dim
    sterror = result.stderr/2/dim
    intercept = result.intercept

    return diffusivity, sterror, exponent, intercept, r_squared

    
