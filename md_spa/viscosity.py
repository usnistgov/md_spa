
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

def pressure2viscosity_csv(time, p_xy, filename="viscosity_running_integral.csv", viscosity_kwargs={}, csv_kwargs={}, method="Green-Kubo"):
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
        Keyword arguements for ``running_acf_integral``
    csv_kwargs : dict, Optional, default={}
        Keywords for ``md_spa_utils.file_manipulation.write_csv``
    method : str, Optional, default="Einstein"
        Method of extracting viscosity from equilibrium MD calculation
        
    Returns
    -------
    eta : numpy.ndarray
        Viscosity coefficient at each time frame. [Units of pressure]^2*[units of time]

    """

    if not dm.isiterable(time) or dm.isiterable(time[0]):
        raise ValueError("The array `time` must be of shape (,N)")

    if not dm.isiterable(p_xy) or not dm.isiterable(p_xy[0]) or dm.isiterable(p_xy[0][0]):
        raise ValueError("The array `p_xy` must be of shape (6,N) or (3,N)")
    elif len(p_xy) not in [3,6]:
        raise ValueError("The array `p_xy` must be of shape (6,N) or (3,N)")

    if method == "Green-Kubo":
        eta, stnderr = running_acf_integral(time, p_xy, **viscosity_kwargs)
    elif method == "Einstein":
        eta, stnderr = running_einstein(time, p_xy, **viscosity_kwargs)
    else:
        raise ValueError("Define one of two supported methods: Einstein or Green-Kubo")

    csv_kwargs["header"] = ["time", "cumulative integral", "integral SE"]
    csv_kwargs["header_comment"] = "# Calculating Viscosity with {} method.\n#".format(method) 

    fm.write_csv(filename, np.transpose(np.array([time,eta,stnderr])), **csv_kwargs)

def running_einstein(time, p_xy, error_type="standard error", scale_coefficient=1, skip=1, show_plot=False, title=None, save_plot=False, plot_name="einstein_viscosity_components.png"):
    """
    Calculate the viscosity of an equilibrium molecular dynamics simulation from the pressure tensor. This function outputs the running integral of the pressure "displacement", averaged over multiple time origins (set by ``skip``) and tensor components. The latter of which the error is computed from.

    Edward J. Maginn1*, Richard A. Messerly2*, Daniel J. Carlson3, Daniel R. Roe4, J. Richard Elliott5 Living J. Comp. Mol. Sci. 2019, 1(1), 6324
    DOI: 10.33011/livecoms.1.1.6324

    Parameters
    ----------
    time : numpy.ndarray
        Array of time values corresponding to pressure tensor data
    p_xy : numpy.ndarray
        Pressure tensor data at each time step. The first dimension can be of either length 3 or 6, representing the three off diagonals (xy, xz, yz) or the entire pressure tensor (xx, yy, zz, xy, xz, yz). The second dimension is of the same length as ``time``.
    scale_coefficient : float, Optional, default=1.0
        Prefactor to scale the viscosity coefficient. The default results in a value of .. math::`2\eta k_{B}T/V`
    error_type : str, Optional, default="standard error"
        Type of error to be saved, either "standard error" or "standard deviation"
    skip : int, Optional, default=1
        Number of frame to skip to obtain an independent trajectory
    show_plot : bool, Optional, default=False
        choose to show a plot of the fit
    save_plot : bool, Optional, default=False
        choose to save a plot of the fit
    title : str, Optional, default=None
        The title used in the cumulative_integral plot, note that this str is also added as a prefix to the ``plotname``.
    plot_name : str, Optional, default="einstein_viscosity_components.png"
        If ``save_plot==True`` the cumulative_integral will be saved with the debye-waller factor marked, The ``title`` is added as a prefix to this str

    Returns
    -------
    cumulative_integral : numpy.ndarray
        Viscosity coefficient at each time frame units of pressure**2 * units of time.
        
    """
    if not dm.isiterable(time) or dm.isiterable(time[0]):
        raise ValueError("The array `time` must be of shape (,N)")

    if not dm.isiterable(p_xy) or not dm.isiterable(p_xy[0]) or dm.isiterable(p_xy[0][0]):
        raise ValueError("The array `p_xy` must be of shape (6,N)")
    elif len(p_xy) != 6:
        raise ValueError("The array `p_xy` must be of shape (6,N)")

    _, npts = np.shape(p_xy)
    lx = 5
    p_new = np.zeros((5,npts))
    p_new[0] = (p_xy[0]-p_new[1])/2
    p_new[1] = (p_xy[1]-p_new[2])/2
    p_new[2:] = p_xy[3:]

    # Same result whether the integral is taken before or after averaging the tensor components
    acf_set = np.zeros_like(p_xy)
    integral_set = np.zeros_like(p_xy)
    nblocks = int(npts/skip)
    for i, p_tmp in enumerate(p_xy):
        count = np.zeros(npts)
        tmp_array = np.zeros(npts)
        for j in range(nblocks):
            tmp_in = p_tmp[j*skip:]
            tmp_npts = len(tmp_in)
            tmp_array[:tmp_npts] += integrate.cumtrapz(tmp_in, time[j*skip:], initial=0)**2
            count[:tmp_npts] += np.ones(tmp_npts)
        integral_set[i,:] = tmp_array/count
    eta = np.mean(integral_set, axis=0)
    stnderror = np.sqrt(np.sum(np.square(integral_set-eta), axis=0)/(lx-1))
    if error_type == "standard error":
        stnderror = stnderror/np.sqrt(lx) # pxy, pxz, pyz, (pxx-pyy)/2, (pyy-pzz)/2
    elif error_type == "standard deviation":
        pass
    else:
         raise ValueError("The `error_type`, {}, is not supported".format(error_type))

    if save_plot or show_plot:
        p_labels = {0: "($\eta_{xx} - \eta_{yy}$)/2", 1: "($\eta_{yy} - \eta_{zz}$)/2", 2: "$\eta_{xy}$", 3: "$\eta_{xz}$", 4: "$\eta_{yz}$"}
        for i,eta_tmp in enumerate(integral_set):
            if lx ==5 and i < 2:
                color = "c"
            else:
                color = "b"
            if lx <5:
                label = p_labels[i+2]
            else:
                label = p_labels[i]
            plt.plot(time, scale_coefficient*eta_tmp, linewidth=0.5, color=color, label=label)
        plt.plot(time, scale_coefficient*eta, "k", label="$\bar{\eta}$")
        plt.fill_between(time, scale_coefficient*(eta-stnderror), scale_coefficient*(eta+stnderror), alpha=0.25, color="black")
        plt.plot([time[0], time[-1]], [0,0], "k", linewidth=0.5)
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

    return scale_coefficient*eta, scale_coefficient*stnderror

def running_acf_integral(time, p_xy, error_type="standard error", scale_coefficient=1, skip=1, show_plot=False, title=None, save_plot=False, plot_name="green-kubo_viscosity_components.png"):
    """
    Calculate the viscosity of an equilibrium molecular dynamics simulation from the independent pressure components: pxy, pxz, pyz, (pxx-pyy)/2, and (pyy-pzz)/2.

    Edward J. Maginn1*, Richard A. Messerly2*, Daniel J. Carlson3, Daniel R. Roe4, J. Richard Elliott5 Living J. Comp. Mol. Sci. 2019, 1(1), 6324
    DOI: 10.33011/livecoms.1.1.6324
    D. Alfe and M. J. Gillan, Phys. Rev. Lett., 1998, 81, 5161. DOI: 10.1103/PhysRevLett.81.5161


    Parameters
    ----------
    time : numpy.ndarray
        Array of time values corresponding to pressure tensor data
    p_xy : numpy.ndarray
        Pressure tensor data at each time step. The first dimension can be of either length 3 or 6, representing the three off diagonals (xy, xz, yz) or the entire pressure tensor (xx, yy, zz, xy, xz, yz). The second dimension is of the same length as ``time``.
    scale_coefficient : float, Optional, default=1.0
        Prefactor to scale the viscosity coefficient. The default results in a value of .. math::`\eta k_{B}T/V`
    error_type : str, Optional, default="standard error"
        Type of error to be saved, either "standard error" or "standard deviation"
    skip : int, Optional, default=1
        Number of frame to skip to obtain an independent trajectory
    show_plot : bool, Optional, default=False
        choose to show a plot of the fit
    save_plot : bool, Optional, default=False
        choose to save a plot of the fit
    title : str, Optional, default=None
        The title used in the cumulative_integral plot, note that this str is also added as a prefix to the ``plotname``.
    plot_name : str, Optional, default="green-kubo_viscosity_components.png"
        If ``save_plot==True`` the cumulative_integral will be saved with the debye-waller factor marked, The ``title`` is added as a prefix to this str

    Returns
    -------
    cumulative_integral : numpy.ndarray
        Viscosity coefficient at each time frame. Units of pressure**2 * units of time.
        
    """
    if not dm.isiterable(time) or dm.isiterable(time[0]):
        raise ValueError("The array `time` must be of shape (,N)")

    if not dm.isiterable(p_xy) or not dm.isiterable(p_xy[0]) or dm.isiterable(p_xy[0][0]):
        raise ValueError("The array `p_xy` must be of shape (6,N)")
    elif len(p_xy) != 6:
        raise ValueError("The array `p_xy` must be of shape (6,N)")

    _, npts = np.shape(p_xy)
    lx = 5
    p_new = np.zeros((5,npts))
    p_new[0] = (p_xy[0]-p_new[1])/2
    p_new[1] = (p_xy[1]-p_new[2])/2
    p_new[2:] = p_xy[3:]

    # Same result whether the integral is taken before or after averaging the tensor components
    acf_set = np.array([dm.autocorrelation(x) for x in p_new])
    integral_set = np.array([integrate.cumtrapz(x, time, initial=0) for x in acf_set])
    eta = np.mean(integral_set, axis=0)
    stnderror = np.sqrt(np.sum(np.square(integral_set-eta), axis=0)/(lx-1))
    if error_type == "standard error":
        stnderror = stnderror/np.sqrt(lx)
    elif error_type == "standard deviation":
        pass
    else:
         raise ValueError("The `error_type`, {}, is not supported".format(error_type))

    if save_plot or show_plot:
        p_labels = {0: "($\eta_{xx} - \eta_{yy}$)/2", 1: "($\eta_{yy} - \eta_{zz}$)/2", 2: "$\eta_{xy}$", 3: "$\eta_{xz}$", 4: "$\eta_{yz}$"}
        for i,eta_tmp in enumerate(integral_set):
            if lx ==5 and i < 2:
                color = "c"
            else:
                color = "b"
            if lx <5:
                label = p_labels[i+2]
            else:
                label = p_labels[i]
            plt.plot(time, scale_coefficient*eta_tmp, linewidth=0.5, color=color, label=label)
        plt.plot(time, scale_coefficient*eta, "k", label=r"$\bar{\eta}$")
        plt.fill_between(time, scale_coefficient*(eta-stnderror), scale_coefficient*(eta+stnderror), alpha=0.25, color="black")
        plt.plot([time[0], time[-1]], [0,0], "k", linewidth=0.5)
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
        
    return scale_coefficient*eta, scale_coefficient*stnderror

def keypoints2csv(filename, fileout="viscosity.csv", mode="a", delimiter=",", title=None, additional_entries=None, additional_header=None, kwargs_find_viscosity={}, file_header_kwargs={}, method=None):
    """
    Given the path to a csv file containing cumulative_integral data, extract key values and save them to a .csv file. The file of cumulative_integral data should have a first column with distance values, followed by columns with radial distribution values. These data sets will be distinguished in the resulting csv file with the column headers

    Parameters
    ----------
    filename : str
        Input filename and path to lammps cumulative_integral output file
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
        Keywords for ``find_green_kubo_viscosity`` or ``find_einstein_viscosity`` functions depending on ``method``
    file_header_kwargs : dict, Optional, default={}
        Keywords for ``md_spa_utils.os_manipulation.file_header`` function    
    method : str, Optional, default=None
        Can be 'Einstein' or 'Green-Kubo', specifies the type of data to be analyzed. If ``pressure2viscosity_csv`` was used to generate the data, the method is extracted from the header.

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

    if method == None:
        with open(filename, "r") as f:
            line = f.readline()
        if "Green-Kubo" in line:
            method = "Green-Kubo"
        elif "Einstein" in line:
            method = "Einstein"
        else:
            raise ValueError("Viscosity data ``method`` must be specified as either 'Einstein' or 'Green-Kubo'")

    data = np.transpose(np.genfromtxt(filename, delimiter=delimiter))
    tmp_kwargs = copy.deepcopy(kwargs_find_viscosity)
    if "title" not in tmp_kwargs and title != None:
        tmp_kwargs["title"] = title

    if method == "Green-Kubo":
        viscosity, parameters, stnderror, tcut = find_green_kubo_viscosity(data[0], data[1], data[2], **tmp_kwargs)
        tmp_list = [val for pair in zip([viscosity]+list(parameters), list(stnderror)) for val in pair]
        output = [list(additional_entries)+[title, tcut]+list(tmp_list)]
        file_headers = ["Group", "tcut", "eta", "eta SE", "A", "A SE", "alpha", "alpha SE", "tau1", "tau1 SE", "tau2", "tau2 SE"]
    else:
        best, longest = find_einstein_viscosity(data[0], data[1], **tmp_kwargs)
        output = [list(additional_entries)+[title]+list(best)+list(longest)]
        file_headers = ["Group", "Best Eta", "B Eta SE", "B t_bound1", "B t_bound2", "B Exponent", "B Intercept", "B Npts", "Longest Eta", "L Eta SE", "L t_bound1", "L t_bound2", "L Exponent", "L Intercept", "L Npts"]

    if not os.path.isfile(fileout) or mode=="w":
        if flag_add_header:
            file_headers = list(additional_header) + file_headers
        fm.write_csv(fileout, output, mode=mode, header=file_headers)
    else:
        fm.write_csv(fileout, output, mode=mode)

def find_green_kubo_viscosity(time, cumulative_integral, integral_error, fit_limits=(None,None), weighting_method="b-exponent", b_exponent=None, tcut_fraction=0.4, show_plot=False, title=None, save_plot=False, plot_name="green-kubo_viscosity.png", verbose=False, fit_kwargs={}):
    """
    Extract the viscosity from the running integral of the autocorrelation function. 

    Parameters
    ----------
    time : numpy.ndarray
        Array of time values corresponding to pressure tensor data
    cumulative_integral : numpy.ndarray
        Viscosity coefficient at each time frame
    integral_error : numpy.ndarray
        Viscosity coefficient error for each time frame. The inverse of these values are used to weight the fitting process. See ``weighting_method`` to specify how this is handled.
    fit_limits : tuple, Optional, default=(None,None)
        Choose the indices to frame the area from which to estimate the viscosity so that, ``time[fit_limits[0]:fit_limits[1]]``. This window tends to be less than 100ps. This will cut down on comutational time in spending time on regions with poor statistics.
    weighting_method : str, Optional, default="b-exponent"
        The error of the cumulative integral increases with time as does the autocorrelation function from which it is computed. Method options include:

        - "b-exponent": Calculates weighing from the function ``time**(-b)`` where b is defined in ``b_exponent``
        - "inverse": Takes the inverse of the provided standard deviation

    b_exponent : float, Optional, default=None
        Exponent used if ``weighting_method == "b-exponent"``. This value can be calculated in ``md_spa.viscosity.fit_standard_deviation_exponent``
    tcut_fraction : float, Optional, default=0.4
        Choose a maximum cut-off in time at ``np.where(integral_error > tcut_fraction*np.mean(cumulative_integral))[0][0]`` where these arrays have been bound with ``fit_limits``. If this feature is not desired, set ``tcut_fraction=None``
    show_plot : bool, Optional, default=False
        choose to show a plot of the fit
    save_plot : bool, Optional, default=False
        choose to save a plot of the fit
    title : str, Optional, default=None
        The title used in the cumulative_integral plot, note that this str is also added as a prefix to the ``plotname``.
    plot_name : str, Optional, default="green-kubo_viscosity.png"
        If ``save_plot==True`` the cumulative_integral will be saved with the debye-waller factor marked, The ``title`` is added as a prefix to this str
    verbose : bool, Optional, default=False
        Will print intermediate values or not
    fit_kwargs : dict, Optional, default={}
        Keyword arguements for exponential functions in ``custom_fit``

    Returns
    -------
    viscosity : float
        Estimation of the viscosity from the fit of a double cumlulative exponential distribution to the running integral of 


    """
    if not dm.isiterable(time):
        raise ValueError("Given distances, time, should be iterable")
    else:
        time = np.array(time)
    if not dm.isiterable(cumulative_integral):
        raise ValueError("Given cumulative_integral, should be iterable")
    else:
        cumulative_integral = np.array(cumulative_integral)

    if len(fit_limits) != 2:
        raise ValueError("`fit_limits` should be of length 2.")

    if fit_limits[1] != None:
        try:
            cumulative_integral = cumulative_integral[:fit_limits[1]]
            time = time[:fit_limits[1]]
            integral_error = np.array(integral_error)[:fit_limits[1]]
        except:
            pass
    if fit_limits[0] != None:
        try:
            cumulative_integral = cumulative_integral[fit_limits[0]:]
            time = time[fit_limits[0]:]
            integral_error = np.array(integral_error)[fit_limits[0]:]
        except:
            pass

    if tcut_fraction != None:
        if isinstance(tcut_fraction, (np.floating, float)):
            try:
                tcut = np.where(integral_error > tcut_fraction*np.mean(cumulative_integral))[0][0]
            except:
                tcut = None
            cumulative_integral = cumulative_integral[:tcut]
            time = time[:tcut]
            integral_error = np.array(integral_error)[:tcut]
        else:
            raise ValueError("`tcut_fraction` must be a float or None.")
            
    npts = len(time)

    tmp_kwargs = copy.deepcopy(fit_kwargs)
    if "weighting" not in tmp_kwargs:
        if weighting_method == "inverse":
            tmp_kwargs["weighting"] = np.array([1/x if x > np.finfo(float).eps else 1/np.finfo(float).eps for x in integral_error])
        elif weighting_method == "b-exponent":
            if not isinstance(b_exponent, (np.floating, float)):
                raise ValueError("To weight provided data with function time^(-b), b must be a float. Calculate this value with `md_spa.viscosity.fit_standard_deviation_exponent`")
            tmp_kwargs["weighting"] = np.array([x**(-b_exponent) if x > np.finfo(float).eps else np.finfo(float).eps**(-b_exponent) for x in integral_error])
        else:
            raise ValueError("Weighting method, {}, is not supported".format(weighting_method))

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
            + x[0]*(1-x[1])*x[3]*(1-np.exp(-(time/x[3])))

    if save_plot or show_plot:
        plt.fill_between(time, cumulative_integral-integral_error, cumulative_integral+integral_error, edgecolor=None, alpha=0.15, facecolor="black")
        plt.plot(time,double_cumulative_exponential(parameters),"r--",label="Fit", linewidth=0.5)
        plt.plot(time,cumulative_integral,"k",label="Data", linewidth=0.5)
        plt.plot([time[0], time[-1]],[viscosity,viscosity], linewidth=0.5)
        plt.legend(loc="lower right")
        plt.xlabel("time")
        plt.ylabel("$\eta$")
        plt.ylim((0,np.max(cumulative_integral+integral_error)))
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

    return viscosity, parameters, np.insert(uncertainties, 0, visc_error), tcut

def fit_standard_deviation_exponent(time, standard_deviation, fit_kwargs={}, show_plot=False, title=None, save_plot=False, plot_name="power-law_fit.png"):
    """
    It has been suggested that in using the Green-Kubo method, a double cumulative exponential function should be fit to the data with a weighting function based on the standard deviation of the data. Here the standard deviation vs. time is fit to the power-law functional form: A*time**b. 

    DOI: 10.1021/acs.jctc.5b00351

    Parameters
    ----------
    time : numpy.ndarray
        Array of time values corresponding to standard deviation of autocorrelation running integral
    standard_deviation : numpy.ndarray
        Viscosity coefficient error for each time frame.
    fit_kwargs : dict, Optional, default={}
        Keyword arguements for power law function in ``custom_fit``
    show_plot : bool, Optional, default=False
        choose to show a plot of the fit
    save_plot : bool, Optional, default=False
        choose to save a plot of the fit
    title : str, Optional, default=None
        The title used in the cumulative_integral plot, note that this str is also added as a prefix to the ``plot_name``.
    plot_name : str, Optional, default="power-law_fit.png"
        If ``save_plot==True`` the cumulative_integral will be saved with the debye-waller factor marked, The ``title`` is added as a prefix to this str

    Returns
    -------
    b-exponent : float
        Exponent in functional form: A*time**b
    prefactor : float
        Prefactor, A, in functional form A*time**b
    uncertainties : list
        List of uncertainties in provided fit

    """

    parameters, uncertainties = cfit.power_law(time, standard_deviation, **fit_kwargs)

    def power_law(x):
        return x[0]*time**x[1]

    if save_plot or show_plot:
        plt.plot(time,power_law(parameters),"r--",label="Fit", linewidth=0.5)
        plt.plot(time,standard_deviation,"k",label="Data", linewidth=0.5)
        plt.plot([time[0], time[-1]],[0,0], linewidth=0.5)
        plt.legend(loc="upper left")
        plt.xlabel("time")
        plt.ylabel("$\sigma$")
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

    return parameters[0], parameters[1], uncertainties

def find_einstein_viscosity(time, cumulative_integral,  min_exp=0.991, min_Npts=10, skip=1, show_plot=False, title=None, save_plot=False, plot_name="einstein_viscosity.png", verbose=False, fit_limits=(None,None), min_R2=0.97):
    """
    Extract the viscosity from the Einstein relation: ..math:`2\eta = d (cumulative_integral)/dt` where the ``cumulative_integral`` is the square of the average running integral of the pressure tensor components vs time.

    Parameters
    ----------
    time : numpy.ndarray
        Array of time values corresponding to pressure tensor data
    cumulative_integral : numpy.ndarray
        Viscosity coefficient at each time frame
    min_exp : float, Optional, default=0.991
        Minimum exponent value used to determine the longest acceptably linear region.
    min_Npts : int, Optional, default=10
        Minimum number of points in the "best" region outputted.
    skip : int, Optional, default=1
        Number of points to skip over in scanning different regions. A value of 1 will be most thorough, but a larger value will decrease computation time.
    min_R2 : float, Optional, default=0.97
        Minimum allowed coefficient of determination to consider proposed exponent. This prevents linearity from skipping over curves.
    fit_limits : tuple, Optional, default=(None,None)
        Choose the indices to frame the area from which to estimate the viscosity. This window tends to be less than 100ps. This will cut down on comutational time in spending time on regions with poor statistics.
    show_plot : bool, Optional, default=False
        choose to show a plot of the fit
    save_plot : bool, Optional, default=False
        choose to save a plot of the fit
    title : str, Optional, default=None
        The title used in the cumulative_integral plot, note that this str is also added as a prefix to the ``plotname``.
    plot_name : str, Optional, default="einstein_viscosity.png"
        If ``save_plot==True`` the cumulative_integral will be saved with the debye-waller factor marked, The ``title`` is added as a prefix to this str
    verbose : bool, Optional, default=False
        Will print intermediate values or not
    fit_kwargs : dict, Optional, default={}
        Keyword arguements for exponential functions in ``custom_fit``

    Returns
    -------
    Returns
    -------
    best : np.array
        Array containing the the following results that represent a region that most closely represents a slope of unity. This region must contain at least the minimum number of points, ``min_Npts``.

        - viscosity (slope/2)
        - standard error of viscosity
        - time interval used to evaluate this slope
        - exponent of region (should be close to unity)
        - intercept
        - number of points in calculation

    longest : np.ndarray
        The results from the longest region that fits a linear model with an exponent of at least ``min_exp``. This array includes:

        - viscosity
        - standard error of viscosity
        - time interval used to evaluate this slope
        - exponent of region (should be close to unity)
        - intercept
        - number of points in calculation

    """

    if not dm.isiterable(time):
        raise ValueError("Given distances, time, should be iterable")
    else:
        time = np.array(time)
    if not dm.isiterable(cumulative_integral):
        raise ValueError("Given cumulative_integral, should be iterable")
    else:
        cumulative_integral = np.array(cumulative_integral)

    cumulative_integral /= 2.0

    if len(cumulative_integral) != len(time):
        raise ValueError("Arrays for time and cumulative_integral are not of equal length.")

    if len(fit_limits) != 2:
        raise ValueError("`fit_limits` should be of length 2.")

    if fit_limits[1] != None:
        try:
            cumulative_integral = cumulative_integral[:fit_limits[1]]
            time = time[:fit_limits[1]]
        except:
            pass
    if fit_limits[0] != None:
        try:
            cumulative_integral = cumulative_integral[fit_limits[0]:]
            time = time[fit_limits[0]:]
        except:
            pass

    npts = len(time)

    if min_Npts > len(time):
        warning.warn("Resetting minimum number of points, {}, to be within length of provided data cut with `fit_limits`, {}".format(min_Npts,len(time)))
        min_Npts = len(time)-1

    best = np.array([np.nan for x in range(7)])
    longest = np.array([np.nan for x in range(7)])
    for npts in range(min_Npts,len(time)):
        for i in range(0,len(time)-npts,skip):
            t_tmp = time[i:(i+npts)]
            cumulative_integral_tmp = cumulative_integral[i:(i+npts)]
            d_tmp, stder_tmp, exp_tmp, intercept, r2_tmp = slope_viscosity(t_tmp, cumulative_integral_tmp, verbose=verbose)

            if r2_tmp > min_R2:
                if np.abs(exp_tmp-1.0) < np.abs(best[4]-1.0) or np.isnan(best[4]):
                    best = np.array([d_tmp, stder_tmp, t_tmp[0], t_tmp[-1], exp_tmp, intercept, npts])

                if (exp_tmp >=  min_exp and longest[-1] <= npts) or np.isnan(longest[4]):
                    if (longest[-1] < npts or np.abs(longest[4]-1.0) > np.abs(exp_tmp-1.0)) or np.isnan(longest[4]):
                        longest = np.array([d_tmp, stder_tmp, t_tmp[0], t_tmp[-1], exp_tmp, intercept, npts])

                if verbose:
                    print("Region Viscosity: {} +- {}, from Time: {} to {}, with and exponent of {} using {} points, Exp Rsquared: {}".format(d_tmp, stder_tmp, t_tmp[0], t_tmp[-1], exp_tmp, npts, r2_tmp))

    if save_plot or show_plot:
        plt.plot(time,cumulative_integral,"k",label="Data", linewidth=0.5)
        tmp_time = np.array([time[0],time[-1]])
        tmp_best = tmp_time*best[0]+best[5]
        plt.plot(tmp_time,tmp_best, "g", label="Best", linewidth=0.5)
        tmp_longest = tmp_time*longest[0]+longest[5]
        plt.plot(tmp_time,tmp_longest, "b", label="Longest", linewidth=0.5)
        plt.xlabel("time")
        plt.ylabel("Mean Squared Pressure Displacement / 2")
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
        print("Best Region Viscosity: {} +- {}, from Time: {} to {}, with and exponent of {} using {} points".format(*best[:5],best[-1]))
        print("Longest Region Viscosity: {} +- {}, from Time: {} to {}, with and exponent of {} using {} points".format(*best[:5],best[-1]))

    return best, longest


def slope_viscosity(time, cumulative_integral, verbose=False):
    """
    Analyzing the long-time cumulative_integral, to extract the viscosity. This entire region is used and so should be linear.

    Parameters
    ----------
    time : numpy.ndarray
        Time array of the same length at MSD in picoseconds.
    cumulative_integral : numpy.ndarray
        Cumulative integral representing the mean squared pressure displacement.
    verbose : bool, Optional, default=False
        Will print intermediate values or not
    
    Returns
    -------
    viscosity : float
        Viscosity in meters squared per picosecond
    sterror : float
        Standard error of viscosity
    exponent : float
        Exponent of fit data, should be close to unity
    intercept : float
        Intercept of time versus cumulative_integral for plotting purposes
    r_squared : float
        Coefficient of determination for the linear fitted log-log plot, from which the exponent is derived.

    """
    if not dm.isiterable(time):
        raise ValueError("Given distances, time, should be iterable")
    else:
        time = np.array(time)
    if not dm.isiterable(cumulative_integral):
        raise ValueError("Given radial distribution values, cumulative_integral, should be iterable")
    else:
        cumulative_integral = np.array(cumulative_integral)

    if len(cumulative_integral) != len(time):
        raise ValueError("Arrays for time and cumulative_integral are not of equal length.")

    # Find Exponent
    t_log = np.log(time[1:]-time[0])
    cumulative_integral_log = np.log(cumulative_integral[1:]-cumulative_integral[0])
    result = linregress(t_log,cumulative_integral_log)
    exponent = result.slope
    r_squared = result.rvalue**2

    result = linregress(time,cumulative_integral)
    viscosity = result.slope
    sterror = result.stderr
    intercept = result.intercept

    return viscosity, sterror, exponent, intercept, r_squared

