
import sys
import copy
import numpy as np
import warnings
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import signal
from scipy import integrate
from scipy.ndimage.filters import gaussian_filter1d
from scipy.stats import linregress
from scipy.interpolate import InterpolatedUnivariateSpline

import md_spa_utils.data_manipulation as dm
import md_spa_utils.file_manipulation as fm
from . import custom_fit as cfit

def pressure2viscosity_csv(time, p_xy, filename="viscosity_running_integral.csv", viscosity_kwargs={}, csv_kwargs={}):
    """
    Calculate the viscosity of an equilibrium molecular dynamics simulation from the pressure tensor.

    Edward J. Maginn1*, Richard A. Messerly2*, Daniel J. Carlson3, Daniel R. Roe4, J. Richard Elliott5 Living J. Comp. Mol. Sci. 2019, 1(1), 6324
    DOI: 10.33011/livecoms.1.1.6324

    Parameters
    ----------
    time : numpy.ndarray
        Array of time values corresponding to pressure tensor data
    p_xy : numpy.ndarray
        Pressure tensor data at each time step. The first dimension can be of either length 3 or 6, representing the three off diagonals (xy, xz, yz) or the entire pressure tensor (xx, yy, zz, xy, xz, yz). The second dimension is of the same length as ``time``.
    filename : str, Optional, default="viscosity_running_integral.csv"
        Filename for csv file
    viscosity_kwargs : dict, Optional, default={}
        Keyword arguements for ``running_viscosity_integral``
    csv_kwargs : dict, Optional, default={}
        Keywords for ``md_spa_utils.file_manipulation.write_csv``
        
    Returns
    -------
    eta : numpy.ndarray
        Viscosity coefficient at each time frame

    """

    if not dm.isiterable(time) or dm.isiterable(time[0]):
        raise ValueError("The array `time` must be of shape (,N)")

    if not dm.isiterable(p_xy) or not dm.isiterable(p_xy[0]) or dm.isiterable(p_xy[0][0]):
        raise ValueError("The array `p_xy` must be of shape (6,N)")
    elif len(p_xy) != 6:
        raise ValueError("The array `p_xy` must be of shape (6,N)")

    header = ["time", "cumulative integral", "integral SE"]

    eta, stnderr = running_viscosity_integral(time, p_xy, **viscosity_kwargs)

    csv_kwargs["header"] = header
    fm.write_csv(filename, np.transpose(np.array([time,eta,stnderr])), **csv_kwargs)

def running_viscosity_integral(time, p_xy, error_type="standard error", scale_coefficient=1, weighting=[1, 1, 1, 4/3, 4/3, 4/3]):
    """
    Calculate the viscosity of an equilibrium molecular dynamics simulation from the pressure tensor.

    Edward J. Maginn1*, Richard A. Messerly2*, Daniel J. Carlson3, Daniel R. Roe4, J. Richard Elliott5 Living J. Comp. Mol. Sci. 2019, 1(1), 6324
    DOI: 10.33011/livecoms.1.1.6324

    Parameters
    ----------
    time : numpy.ndarray
        Array of time values corresponding to pressure tensor data
    p_xy : numpy.ndarray
        Pressure tensor data at each time step. The first dimension can be of either length 3 or 6, representing the three off diagonals (xy, xz, yz) or the entire pressure tensor (xx, yy, zz, xy, xz, yz). The second dimension is of the same length as ``time``.
    scale_coefficient : float, Optional, default=1.0
        Prefactor to scale the viscosity coefficient. The default results in a value of .. math::`\eta k_{B}T/V`
    weighting : list, Optional, default=[1, 1, 1, 4/3, 4/3, 4/3]
        Weighting factors used in averaging pressure tensor autocorrelation functions.
    error_type : str, Optional, default="standard error"
        Type of error to be saved, either "standard error" or "standard deviation"

    Returns
    -------
    cumulative_integral : numpy.ndarray
        Viscosity coefficient at each time frame
        
    """
    if not dm.isiterable(time) or dm.isiterable(time[0]):
        raise ValueError("The array `time` must be of shape (,N)")

    if not dm.isiterable(p_xy) or not dm.isiterable(p_xy[0]) or dm.isiterable(p_xy[0][0]):
        raise ValueError("The array `p_xy` must be of shape (6,N)")
    elif len(p_xy) != 6:
        raise ValueError("The array `p_xy` must be of shape (6,N)")

    if len(p_xy) == 3:
        weighting = weighting[:3] 

    lx = len(time)

    # Same result whether the integral is taken before or after averaging the tensor components
    acf_set = np.zeros_like(p_xy)
    integral_set = np.zeros_like(p_xy)
    for i, p_tmp in enumerate(p_xy):
        tmp_acf = np.correlate(p_tmp, p_tmp, mode='full')
        acf_set[i,:] = tmp_acf[len(p_tmp)-1:]
        integral_set[i,:] = np.array([integrate.simps(acf_set[i,:j], time[:j]) for j in range(1,lx+1)])
    eta = np.mean(integral_set*np.expand_dims(weighting, axis=1), axis=0)/np.sum(weighting)
    if error_type == "standard error":
        stnderror = np.std(integral_set*np.expand_dims(weighting, axis=1), axis=0)/np.sum(weighting)/np.sqrt(len(integral_set))
    elif error_type == "standard deviation":
        stnderror = np.std(integral_set*np.expand_dims(weighting, axis=1), axis=0)/np.sum(weighting)
    else:
         raise ValueError("The `error_type`, {}, is not supported".format(error_type))

    #for eta_tmp in integral_set:
    #    plt.plot(time, eta_tmp, linewidth=0.5)
    #plt.plot(time, eta, "k")
    #plt.fill_between(time, eta-stnderror, eta+stnderror, alpha=0.25, color="black")
    #plt.plot([time[0], time[-1]], [0,0], "k", linewidth=0.5)
    #plt.show()
        
    return scale_coefficient*eta, scale_coefficient*stnderror

def keypoints2csv(filename, fileout="viscosity.csv", mode="a", delimiter=",", title=None, additional_entries=None, additional_header=None, kwargs_find_viscosity={}, file_header_kwargs={}):
    """
    Given the path to a csv file containing msd data, extract key values and save them to a .csv file. The file of msd data should have a first column with distance values, followed by columns with radial distribution values. These data sets will be distinguished in the resulting csv file with the column headers

    Parameters
    ----------
    filename : str
        Input filename and path to lammps msd output file
    fileout : str, Optional, default="viscosity.csv"
        Filename of output .csv file
    mode : str, Optional, default="a"
        Mode used in writing the csv file, either "a" or "w".
    delimiter : str, Optional, default=","
        Delimiter between data in input file
    title : list[str], Optional, default=None
        Titles for plots if that is specified in the ``kwargs_find_viscosity``
    additional_entries : list, Optional, default=None
        This iterable structure can contain additional information about this data to be added to the beginning of the row
    additional_header : list, Optional, default=None
        If the csv file does not exist, these values will be added to the beginning of the header row. This list must be equal to the `additional_entries` list.
    kwargs_find_viscosity : dict, Optional, default={}
        Keywords for `find_viscosity` function
    file_header_kwargs : dict, Optional, default={}
        Keywords for ``md_spa_utils.os_manipulation.file_header`` function    

    Returns
    -------
    csv file
    
    """
    if not os.path.isfile(filename):
        raise ValueError("The given file could not be found: {}".format(filename))

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

    data = np.transpose(np.genfromtxt(filename, delimiter=delimiter))
    tmp_kwargs = copy.deepcopy(kwargs_find_viscosity)
    if "title" not in tmp_kwargs and title != None:
        tmp_kwargs["title"] = title
    viscosity, parameters, stnderror = find_viscosity(data[0], data[1], data[2], **tmp_kwargs)
    tmp_list = [val for pair in zip([viscosity]+list(parameters), list(stnderror)) for val in pair]
    output = [list(additional_entries)+[title]+list(tmp_list)]

    file_headers = ["Group", "eta [Ang^2]", "eta SE", "1 [ps]", "tau1 SE", "tau2 [ps]", "tau2 SE", "A [???]", "A SE", "alpha", "alpha SE"]
    if not os.path.isfile(fileout) or mode=="w":
        if flag_add_header:
            file_headers = list(additional_header) + file_headers
        fm.write_csv(fileout, output, mode=mode, header=file_headers)
    else:
        fm.write_csv(fileout, output, mode=mode)

def find_viscosity(time, cumulative_integral, integral_error, show_plot=False, title=None, save_plot=False, plot_name="viscosity.png", verbose=False, fit_kwargs={}):
    """
    Extract the viscosity from the 

    Parameters
    ----------
    time : numpy.ndarray
        Array of time values corresponding to pressure tensor data
    cumulative_integral : numpy.ndarray
        Viscosity coefficient at each time frame
    integral_error : numpy.ndarray
        Viscosity coefficient error for each time frame. The inverse of these values are used to weight the fitting process. 
    show_plot : bool, Optional, default=False
        choose to show a plot of the fit
    save_plot : bool, Optional, default=False
        choose to save a plot of the fit
    title : str, Optional, default=None
        The title used in the msd plot, note that this str is also added as a prefix to the ``plotname``.
    plot_name : str, Optional, default="debye-waller.png"
        If ``save_plot==True`` the msd will be saved with the debye-waller factor marked, The ``title`` is added as a prefix to this str
    verbose : bool, Optional, default=False
        Will print intermediate values or not
    fit_kwargs : dict, Optional, default={}
        Keyword arguements for exponential functions in ``custom_fit``

    Returns
    -------
    viscosity : float
        Estimation of the viscosity from the fit of a double cumlulative exponential distribution to the running integral of 


    """

    tmp_kwargs = copy.deepcopy(fit_kwargs)
    if "weighting" not in tmp_kwargs:
        tmp_kwargs["weighting"] = 1/np.array(integral_error)
    if "verbose" not in tmp_kwargs:
        tmp_kwargs["verbose"] = verbose

    parameters, uncertainties = cfit.double_cumulative_exponential(time, cumulative_integral, **tmp_kwargs)
    # Assuming parameters = [A, alpha, 1, tau2]
    viscosity =  parameters[0]*parameters[1]*parameters[2] + parameters[0]*(1-parameters[1])*parameters[3]
    # Propagation of error
    visc_error = np.sqrt(uncertainties[0]**2*(parameters[1]*parameters[2] + (1-parameters[1])*parameters[3])**2 \
                         + uncertainties[1]**2*(parameters[0]*(parameters[2] - parameters[3]))**2 \
                         + uncertainties[2]**2*(parameters[0]*parameters[1])**2 \
                         + uncertainties[3]**2*(parameters[0]*(1 - parameters[1]))**2 \
                 )

    def double_cumulative_exponential(x):
        return x[0]*x[1]*x[2]*(1-np.exp(-(time/x[2]))) \
            + x[0]*(1-x[1])*x[2]*(1-np.exp(-(time/x[2])))

    if save_plot or show_plot:
        plt.fill_between(time, cumulative_integral-integral_error, cumulative_integral+integral_error, edgecolor=None, alpha=0.15, facecolor="black")
        plt.plot(time,double_cumulative_exponential(parameters),"r--",label="Fit", linewidth=0.5)
        plt.plot(time,cumulative_integral,"k",label="Data", linewidth=0.5)
        plt.plot([time[0], time[-1]],[viscosity,viscosity], linewidth=0.5)
        plt.legend(loc="lower right")
        plt.xlabel("time")
        plt.ylabel("$\eta$")
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

    return viscosity, parameters, np.insert(uncertainties, 0, visc_error)

