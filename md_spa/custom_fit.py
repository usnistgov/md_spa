import os
import warnings
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as spo

import lmfit
from lmfit import minimize, Parameters

def matrix_least_squares(xdata, ydata, sample_array={}, method="nnls", method_kwargs={}, function="exponential_decay", verbose=False):
    """
    Use non-negative least squares to determine a set of additive ``functions`` (e.g., exponential decay) to reproduce a data set. 
    Unlike curve-fitting where the number of additive functions is predetermined and fit, the data is fit as sets of delta functions to reveal a smaller set of a yet unknown number of parameters. The result is highly accurate, but dependant on the parameters used to generate the ``sample_array`` and how fine the mesh is.

    The target function can only have two parameters. The first are the prefactors of the linearly combined functions which must sum to unity. The second lies within a more complicated functional form and so is captured in a matrix of trial values (e.g., np.exp(np.outer(-xdata, 1/tau_array))).

    Parameters
    ----------
    xdata : numpy.ndarray
        Independent data set
    ydata : numpy.ndarray
        Dependent data set
    sample_array : dict, Optional, default={mode: "log", lower: "-4", "upper": 4, npts: 1e+5}
        Details of producing an array of arbitrary length for possible values of the parameter within the functional form. This array may be generated on a linear or log scale by specifying the ``mode``. If the ``mode`` is "log" the default values are {lower: "-4", "upper": 4, Npts: 1e+5}, if the mode is "linear", the default parameters are {lower: 1e-4, "upper": 1e+4, Npts: 1e+4}. These arrays are produced with np.logspace and np.linspace, respectively.

        - mode (str): Choose between an array generated on a "linear" or "log" scale
        - lower (float): Lower bound of array
        - upper (float): Upper bound of array
        - npts (int): Number of points in array
        - **kwargs: Additional keyword arguments for np.logspace or np.linspace

    method : str, Optional, default="nnls"
        Default method of solving, can be "nnls" or "lsq_linear" corresponding to the functions in ``scipy.optimize``.
    method_kwargs : dict, Optional, default={}
        Additional keyword arguments used in chosen ``scipy.optimize`` method. For ``method=="lsq_linear"``, ``method_kwargs={"bounds": (0, np.inf), tol=1e-16, method="bvls"}``
    function : str, Optional, default="exponential_decay"
        Function type of which a linear combination reproduced the dependent data. Options include: "exponential_decay". 
    verbose : bool, Optional, default=False
        Output fitting information. (Also see method_kwargs)

    Returns
    -------
    prefactors : numpy.ndarray
        Prefactors for the summed quatities where all values are positive and sum to unity. Array of the same length as ``tau``.
    tau : numpy.ndarray
        Characteristic variable scales. Array of the same length as ``prefactors``
        
    """ 

    if len(xdata) != len(ydata):
        raise ValueError("Length of dependent and independent variables must be equivalent.")

    lt = len(ydata)
    mode = sample_array.pop("mode", "log")
    lt2 = int(sample_array.pop("npts", 1e+4))
    if mode == "log":
        lower = sample_array.pop("lower", -4)
        upper = sample_array.pop("upper", 4)
        tau_array = np.logspace(lower,upper,lt2, **sample_array)
    elif mode == "linear":
        lower = sample_array.pop("lower", 1e-4)
        upper = sample_array.pop("upper", 1e+4)
        tau_array = np.linspace(lower,upper,lt2, **sample_array)
    else:
        raise ValueError("The sample array mode, {}, is not recognized. Please use 'linear' or 'log'".format(mode))

    if function == "exponential_decay":
        matrix_1 = np.exp(np.outer(-xdata, 1/tau_array))
    else:
        raise ValueError("The function type, {}, is not yet supported, consider submitted adding it and submitting a pull request!".format(function))

    matrix_1 = np.concatenate((matrix_1, [np.ones(lt2)]), axis=0)

    matrix_2 = np.zeros((lt+1,2*lt))
    for i in range(1,lt):
        matrix_2[i,2*i-1] = 1
        matrix_2[i,2*i] = -1

    matrix = np.concatenate((matrix_1, matrix_2), axis=1)
    rhs = np.concatenate((ydata,[1]))

    if method == "nnls":
        solution, residual = spo.nnls(matrix, rhs, **method_kwargs)
    elif  method == "lsq_linear":
        kwargs = {"bounds": (0, np.inf), "tol": 1e-16, "method": "bvls"}
        kwargs.update(method_kwargs)
        result = spo.lsq_linear(matrix, rhs, **kwargs)
        solution = result.x
        residual = result.cost
        if verbose:
            print(result.cost, result.status, result.message, result.success)
    else:
        raise ValueError("The method, {}, is not recognized. Use 'nnls' or 'lsq_linear'".format(method))

    amplitudes = solution[:lt2]
    final_ind = np.where(amplitudes>np.finfo(float).eps)[0]
    final_tau = tau_array[final_ind]
    final_prefactors = amplitudes[final_ind]

    final_tau = [x for _, x in sorted(zip(final_prefactors, final_tau), reverse=True)]
    final_prefactors = sorted(final_prefactors, reverse=True)

    if mode=="log" and np.abs(final_tau[0]-10**upper) <= np.finfo(float).eps:
        raise ValueError("Dominant parameter exceeds upper limit. Increase sample_array['upper']")
    elif mode=="linear" and np.abs(final_tau[0]-upper) <= np.finfo(float).eps:
        raise ValueError("Dominant parameter exceeds upper limit. Increase sample_array['upper']")

    if mode=="log" and np.abs(final_tau[0]-10**lower) <= np.finfo(float).eps:
        raise ValueError("Dominant parameter exceeds lower limit. Decrease sample_array['lower']")
    elif mode=="linear" and np.abs(final_tau[0]-lower) <= np.finfo(float).eps:
        raise ValueError("Dominant parameter exceeds upper limit. Decrease sample_array['lower']")

    if verbose:
       print("residual", residual)
       print("prefactors: ", final_prefactors, np.sum(final_prefactors))
       print("residence times: ", final_tau, np.sum(final_tau*final_prefactors))

    return final_prefactors, final_tau

def exponential(xdata, ydata, delimiter=",", minimizer="leastsq", verbose=False, save_plot=False, show_plot=False, plot_name="exponential_fit.png", kwargs_minimizer={}, kwargs_matrix_lsq={}, ydata_min=0.1, tau_max=1e+8):
    """
    Provided data fit is fit to one, two, and three exponentials, where the sum of the prefactors equals unity. 

    Values of zero and NaN are ignored in the fit.

    Parameters
    ----------
    xdata : numpy.ndarray
        independent data set
    ydata : numpy.ndarray
        dependent data set
    delimiter : str, Optional, default=","
        Delimiter between columns used in ``numpy.genfromtxt()``
    minimizer : str, Optional, default="leastsq"
        Fitting method supported by ``lmfit.minimize``
    verbose : bool, Optional, default=False
        Output fitting statistics
    save_plots : bool, Optional, default=False
        If not None, plots comparing the exponential fits will be saved to this filename 
    plot_name : str, Optional, default=None
        Plot filename and path
    show_plot : bool, Optional, default=False
        If true, the fits will be shown
    kwargs_minimizer : dict, Optional, default={}
        Keyword arguments for ``lmfit.minimizer()``
    kwargs_matrix_lsq : dict, Optional, default={}
        Kwayword arguments for ``matrix_least_squares``
    ydata_min : float, Optional, default=0.1
        Minimum value of ydata allowed before beginning fitting process. If ydata[-1] is greater than this value, an error is thrown.
    tau_max : float, Optional, default=1e+5
        Maximum allowed value of characteristic time

    Returns
    -------
    parameters : numpy.ndarray
        Array containing parameters: ['1: t1', '2: a1', '2: t1', '2: a2', '2: t2', '3: a1', '3: t1', '3: a2', '3: t2', '3: a3', '3: t3']
    stnd_errors : numpy.ndarray
        Array containing parameter standard errors: ['1: t1', '2: a1', '2: t1', '2: a2', '2: t2', '3: a1', '3: t1', '3: a2', '3: t2', '3: a3', '3: t3']
        
    """

    xarray = xdata[ydata>0]
    yarray = ydata[ydata>0]

    if np.all(np.isnan(ydata[1:])):
        raise ValueError("y-axis data is NaN")

    if yarray[-1] > ydata_min:
        warnings.warn("Exponential decays to {}, above threshold {}. Run for a longer period of time or increase the keyword value of ydata_min.".format(yarray[-1],ydata_min))
        flag_long = True
        params, _ = spo.curve_fit(lambda x, t: -x/t, xarray, np.log(yarray), bounds=(0, 1e+4))
        prefactors = np.array([0.9,0.1])
        tau = [params[0]*x for x in [1, 0.01]]
    else:
        prefactors, tau = matrix_least_squares(xdata, ydata, **kwargs_matrix_lsq)
        flag_long = False

        if len(tau) == 0:
            raise ValueError("No parameter sets where derived from ``matrix_least_squares``. Consider adjusting kwargs_matrix_lsq={'sample_array': {'lower', 'upper', 'npts'}}")
        elif len(tau) < 2:
            params, _ = spo.curve_fit(lambda x, t: -x/t, xarray, np.log(yarray), bounds=(0, 1e+4))
            tau[0] = params[0]
            prefactors = np.array([0.7,0.29, 0.01])
            tau = [tau[0]*x for x in [1, 10, 100]]
        elif len(tau) < 3:
            if prefactors[1] > 1e-2:
                prefactors = [prefactors[0], prefactors[1]*0.9, prefactors[1]*0.1]
                tau = [tau[0], tau[1], tau[1]*10]
            else:
                params, _ = spo.curve_fit(lambda x, t: -x/t, xarray, np.log(yarray), bounds=(0, 1e+4))
                tau[0] = params[0]
                prefactors = np.array([0.7,0.29, 0.01])
                tau = [tau[0]*x for x in [1, 10, 100]]
        elif len(tau) > 3:
            prefactors = prefactors[:3]
            tau = tau[:3]

    def exponential_res_1(x):
        #return np.exp(-xarray/x["t1"]) - yarray
        return -xarray/x["t1"] - np.log(yarray)
    
    def exponential_res_2(x):
        return x["a1"]*np.exp(-xarray/x["t1"]) + x["a2"]*np.exp(-xarray/x["t2"]) - yarray
    
    def exponential_res_3(x):
        return x["a1"]*np.exp(-xarray/x["t1"]) + x["a2"]*np.exp(-xarray/x["t2"]) + x["a3"]*np.exp(-xarray/x["t3"]) - yarray
    ####
    def dexponential_res_1(x):
        #return np.array([xarray/x["t1"]**2*np.exp(-xarray/x["t1"])])
        return np.array([xarray/x["t1"]**2])

    def dexponential_res_2(x):
        tmp_exp1 = np.exp(-xarray/x["t1"])
        tmp_exp2 = np.exp(-xarray/x["t2"])
        dt1 = x["a1"]*xarray/x["t1"]**2*tmp_exp1
        da1 = tmp_exp1 - tmp_exp2
        dt2 = (1-x["a1"])*xarray/x["t2"]**2*tmp_exp2
        return np.transpose([da1, dt1, dt2])

    def dexponential_res_3(x):
        tmp_exp1 = np.exp(-xarray/x["t1"])
        tmp_exp2 = np.exp(-xarray/x["t2"])
        tmp_exp3 = np.exp(-xarray/x["t3"])
        dt1 = x["a1"]*xarray/x["t1"]**2*tmp_exp1
        da1 = tmp_exp1 - tmp_exp3
        dt2 = x["a2"]*xarray/x["t2"]**2*tmp_exp2
        da2 = tmp_exp2 - tmp_exp3
        dt3 = (1-x["a1"]-x["a2"])*xarray/x["t3"]**2*tmp_exp3
        return np.transpose([da1, dt1, da2, dt2, dt3])

    ####    

    output = np.zeros(11)
    uncertainties = np.zeros(11)
    
    exp1 = Parameters()
    exp1.add("t1", min=0, max=tau_max, value=tau[0])
    if minimizer in ["leastsq"]:
        kwargs_minimizer["Dfun"] = dexponential_res_1
    Result1 = lmfit.minimize(exponential_res_1, exp1, method=minimizer, **kwargs_minimizer)
    output[0] = Result1.params["t1"].value
    uncertainties[0] = Result1.params["t1"].stderr

    exp2 = Parameters()
    exp2.add("a1", min=0, max=1, value=prefactors[0])
    exp2.add("t1", min=0, max=tau_max, value=tau[0])
    exp2.add("a2", min=0, max=1, value=1-prefactors[0], expr="1 - a1")
    exp2.add("t2", min=0, max=tau_max, value=tau[1])
    if minimizer in ["leastsq"]:
        kwargs_minimizer["Dfun"] = dexponential_res_2
    Result2 = lmfit.minimize(exponential_res_2, exp2, method=minimizer, **kwargs_minimizer)
    for i,(param, value) in enumerate(Result2.params.items()):
        output[i+1] = value.value
        uncertainties[i+1] = value.stderr

    if flag_long:
        output[5:] = np.ones(6)*np.nan
        uncertainties[5:] = np.ones(6)*np.nan
    else:
        exp3 = Parameters()
        exp3.add("a1", min=0, max=1, value=prefactors[0])
        exp3.add("t1", min=0, max=tau_max, value=tau[0])
        exp3.add("a2", min=0, max=1, value=prefactors[1])
        exp3.add("t2", min=0, max=tau_max, value=tau[1])
        exp3.add("a3", min=0, max=1, value=prefactors[2], expr="1 - a1 - a2")
        exp3.add("t3", min=0, max=tau_max, value=tau[2])
        if minimizer in ["leastsq"]:
            kwargs_minimizer["Dfun"] = dexponential_res_3
        Result3 = lmfit.minimize(exponential_res_3, exp3, method=minimizer, **kwargs_minimizer)
        for i,(param, value) in enumerate(Result3.params.items()):
            output[i+5] = value.value
            uncertainties[i+5] = value.stderr

       # if output[5] < np.finfo(float).eps or output[7] < np.finfo(float).eps or output[9] < np.finfo(float).eps: # NoteHere
       #     print("\n", plot_name)
       #     params, _ = spo.curve_fit(lambda x, t: -x/t, xarray, np.log(yarray), bounds=(0, 1e+4))
       #     print(params[0])
       #     print(matrix_least_squares(xdata, ydata, **kwargs_matrix_lsq))
       #     print(prefactors, tau)
       #     print(output[5:])
       #     sys.exit()

    if verbose:
        if minimizer == "leastsq":
            print("1 Exp. Termination: {}".format(Result1.lmdif_message))
            print("2 Exp. Termination: {}".format(Result2.lmdif_message))
            if not flag_long:
                print("3 Exp. Termination: {}".format(Result3.lmdif_message))
        else:
            print("1 Exp. Termination: {}".format(Result1.message))
            print("2 Exp. Termination: {}".format(Result2.message))
            if not flag_long:
                print("3 Exp. Termination: {}".format(Result3.message))
        lmfit.printfuncs.report_fit(Result1.params)
        lmfit.printfuncs.report_fit(Result2.params, min_correl=0.5)
        print("Sum: {}".format(Result2.params["a1"]+Result2.params["a2"]))
        if not flag_long:
            lmfit.printfuncs.report_fit(Result3.params, min_correl=0.5)
            print("Sum: {}".format(Result3.params["a1"]+Result3.params["a2"]+Result3.params["a3"]))

    if save_plot or show_plot:
        yfit1 = np.exp(-xarray/Result1.params["t1"])
        plt.plot(xarray,yarray,".",label="Data")
        plt.plot(xarray,yfit1,label="1 Exp.",linewidth=1)
        yfit2 = exponential_res_2(Result2.params) + yarray
        plt.plot(xarray,yfit2,label="2 Exp.",linewidth=1)
        if not flag_long:
            yfit3 = exponential_res_3(Result3.params) + yarray
            plt.plot(xarray,yfit3,label="3 Exp.",linewidth=1)
        plt.ylabel("Probability")
        plt.xlabel("Time")
        plt.tight_layout()
        plt.legend(loc="best")
        if save_plot:
            plt.savefig(plot_name,dpi=300)
    
        plt.figure(2)
        plt.plot(xarray,yarray,".",label="Data")
        plt.plot(xarray,yfit1,label="1 Exp.",linewidth=1)
        plt.plot(xarray,yfit2,label="2 Exp.",linewidth=1)
        if not flag_long:
            plt.plot(xarray,yfit3,label="3 Exp.",linewidth=1)
        plt.ylabel("log Probability")
        plt.xlabel("Time")
        plt.tight_layout()
        plt.yscale('log')
        plt.legend(loc="best")
        if save_plot:
            tmp_save_plot = list(os.path.split(plot_name))
            tmp_save_plot[1] = "log_"+tmp_save_plot[1]
            plt.savefig(os.path.join(*tmp_save_plot),dpi=300)

        if show_plot:
            plt.show()
        plt.close("all")

    return output, uncertainties

def one_exponential(xdata, ydata, minimizer="nelder", verbose=False):
    """
    Provided data fit to:

    Values of zero and NaN are ignored in the fit.

    Parameters
    ----------
    xdata : numpy.ndarray
        independent data set
    ydata : numpy.ndarray
        dependent data set
    minimizer : str, Optional, default="nelder"
        Fitting method supported by ``lmfit.minimize``
    verbose : bool, Optional, default=False
        Output fitting statistics

    Returns
    -------
    parameters : numpy.ndarray
        Array containing parameters: ['t1']
    stnd_errors : numpy.ndarray
        Array containing parameter standard errors: [t1']
        
    """

    xarray = xdata[ydata>0]
    yarray = ydata[ydata>0]

    if np.all(np.isnan(ydata[1:])):
        raise ValueError("y-axis data is NaN")


    def exponential_res_1(x):
        return np.exp(-xarray/x["t1"]) - yarray

    def exponential_fit_1(x, array):
        return np.exp(-xarray/x["t1"])

    exp1 = Parameters()
    exp1.add("t1", min=0, max=1e+4, value=0.1)
    Result1 = lmfit.minimize(exponential_res_1, exp1, method=minimizer)

    # Format output
    output = Result1.params["t1"].value
    uncertainties = Result1.params["t1"].stderr

    if verbose:
        lmfit.printfuncs.report_fit(Result1.params)

    return np.array([output]), np.array([uncertainties])

def two_exponentials(xdata, ydata, minimizer="nelder", verbose=False):
    """
    Provided data fit to:

    Values of zero and NaN are ignored in the fit.

    Parameters
    ----------
    xdata : numpy.ndarray
        independent data set
    ydata : numpy.ndarray
        dependent data set
    minimizer : str, Optional, default="nelder"
        Fitting method supported by ``lmfit.minimize``
    verbose : bool, Optional, default=False
        Output fitting statistics

    Returns
    -------
    parameters : numpy.ndarray
        Array containing parameters: ['a1', 't1', 'a2', 't2']
    stnd_errors : numpy.ndarray
        Array containing parameter standard errors: ['a1', 't1', 'a2', 't2']
        
    """

    xarray = xdata[ydata>0]
    yarray = ydata[ydata>0]

    if np.all(np.isnan(ydata[1:])):
        raise ValueError("y-axis data is NaN")

    def exponential_res_2(x):
        return x["a1"]*np.exp(-xarray/x["t1"]) + x["a2"]*np.exp(-xarray/x["t2"]) - yarray

    def exponential_fit_2(x, array):
        return x["a1"]*np.exp(-xarray/x["t1"]) + x["a2"]*np.exp(-xarray/x["t2"])

    exp2 = Parameters()
    exp2.add("a1", min=0, max=1, value=0.8)
    exp2.add("t1", min=0, max=1e+4, value=0.1)
    exp2.add("a2", min=0, max=1, value=0.2, expr="1 - a1")
    exp2.add("t2", min=0, max=1e+4, value=0.09)
    Result2 = lmfit.minimize(exponential_res_2, exp2, method=minimizer)

    # Format output
    output = np.zeros(4)
    uncertainties = np.zeros(4)
    for i,(param, value) in enumerate(Result2.params.items()):
        output[i] = value.value
        uncertainties[i] = value.stderr

    if verbose:
        lmfit.printfuncs.report_fit(Result2.params, min_correl=0.5)
        print("Sum: {}".format(Result2.params["a1"]+Result2.params["a2"]))

    return output, uncertainties

def three_exponentials(xdata, ydata, minimizer="nelder", verbose=False):
    """
    Provided data fit to:

    Values of zero and NaN are ignored in the fit.

    Parameters
    ----------
    xdata : numpy.ndarray
        independent data set
    ydata : numpy.ndarray
        dependent data set
    minimizer : str, Optional, default="nelder"
        Fitting method supported by ``lmfit.minimize``
    verbose : bool, Optional, default=False
        Output fitting statistics

    Returns
    -------
    parameters : numpy.ndarray
        Array containing parameters: ['a1', 't1', 'a2', 't2', 'a3', 't3']
    stnd_errors : numpy.ndarray
        Array containing parameter standard errors: ['a1', 't1', 'a2', 't2', 'a3', 't3']
        
    """

    xarray = xdata[ydata>0]
    yarray = ydata[ydata>0]

    if np.all(np.isnan(ydata[1:])):
        raise ValueError("y-axis data is NaN")

    def exponential_res_3(x):
        return x["a1"]*np.exp(-xarray/x["t1"]) + x["a2"]*np.exp(-xarray/x["t2"]) + x["a3"]*np.exp(-xarray/x["t3"]) - yarray

    def exponential_fit_3(x, array):
        return x["a1"]*np.exp(-xarray/x["t1"]) + x["a2"]*np.exp(-xarray/x["t2"]) + x["a3"]*np.exp(-xarray/x["t3"])

    exp3 = Parameters()
    exp3.add("a1", min=0, max=1, value=0.8)
    exp3.add("t1", min=0, max=1e+4, value=0.1)
    exp3.add("a2", min=0, max=1, value=0.19)
    exp3.add("t2", min=0, max=1e+4, value=0.09)
    exp3.add("a3", min=0, max=1, value=0.01, expr="1 - a1 - a2")
    exp3.add("t3", min=0, max=1e+4, value=0.02)
    Result3 = lmfit.minimize(exponential_res_3, exp3, method=minimizer)

    # Format output
    output = np.zeros(6)
    uncertainties = np.zeros(6)
    for i,(param, value) in enumerate(Result3.params.items()):
        output[i] = value.value
        uncertainties[i] = value.stderr

    if verbose:
        lmfit.printfuncs.report_fit(Result3.params, min_correl=0.5)
        print("Sum: {}".format(Result3.params["a1"]+Result3.params["a2"]+Result3.params["a3"]))

    return output, uncertainties

def gaussian(xdata, ydata, fit_kws={}, set_params={}, verbose=False):
    """
    Fit Gaussian function to data with ``lmfit.GaussianModel``

    Parameters
    ----------
    xdata : numpy.ndarray
        independent data set
    ydata : dependent data set
    fit_kws : dict, Optional, default={}
        Keywords used in ``lmfit.minimize``
    set_parameters : dict, Optional, default={}
        Dictionary where keys are one of the Gaussian parameters: "center", "amplitude", "height", "sigma", "fwhm" and the value is a dictionary containing parameter settings according to ``lmfit.Model.set_param_hint()``
    verbose : bool, Optional, default=False
        If true, final parameters will be printed

    Returns
    -------
    parameters : numpy.ndarray
        Array containing parameters: ["amplitude", "center", "sigma", "fwhm", "height"]
    stnd_errors : numpy.ndarray
        Array containing uncertainties for parameters
    
    """

    xarray = xdata[ydata>0]
    yarray = ydata[ydata>0]

    if np.all(np.isnan(ydata[1:])):
        raise ValueError("y-axis data is NaN")

    model = lmfit.models.GaussianModel(nan_policy="omit")
    for param, values in set_params.items():
        if param not in ["center", "amplitude", "height", "sigma", "fwhm"]:
            raise ValueError("Given parameter, {}, is not supported".format(param))
        else:
            model.set_param_hint(param, **values)

    pars = model.guess(yarray, x=xarray)
    result = model.fit(yarray, pars, x=xarray, fit_kws=fit_kws)

    if verbose:
        lmfit.printfuncs.report_fit(result.params)

    output = np.zeros(5)
    uncertainties = np.zeros(5)
    for i,(param, value) in enumerate(result.params.items()):
        output[i] = value.value
        uncertainties[i] = value.stderr

    return output, uncertainties

def double_cumulative_exponential(xdata, ydata, minimizer="nelder", verbose=False, weighting=None, minimizer_kwargs={}):
    """
    Provided data fit to:
    ..math:`\eta(t)=\eta_{\inf}(1-exp(-(t/\tau_{2})^{\beta_{S}}))` 

    Values of zero and NaN are ignored in the fit.

    Parameters
    ----------
    xdata : numpy.ndarray
        independent data set
    ydata : numpy.ndarray
        dependent data set
    minimizer : str, Optional, default="nelder"
        Fitting method supported by ``lmfit.minimize``
    verbose : bool, Optional, default=False
        Output fitting statistics

    Returns
    -------
    parameters : numpy.ndarray
        Array containing parameters: ["A", "alpha", "tau1", "tau2"]
    stnd_errors : numpy.ndarray
        Array containing parameter standard errors: ["A", "alpha", "tau1", "tau2"]
        
    """

    xarray = xdata[ydata>0]
    yarray = ydata[ydata>0]

    if np.all(np.isnan(ydata[1:])):
        raise ValueError("y-axis data is NaN")

    if np.all(weighting != None):
        weighting = weighting[ydata>0]
        if minimizer == "emcee":
            minimizer_kwargs["is_weighted"] = True
    elif minimizer == "emcee":
        minimizer_kwargs["is_weighted"] = False

    def exponential_res_1(x):
        error = x["A"]*x["alpha"]*x["tau1"]*(1-np.exp(-(xarray/x["tau1"]))) \
            + x["A"]*(1-x["alpha"])*x["tau2"]*(1-np.exp(-(xarray/x["tau2"]))) - yarray
        if np.all(weighting != None):
            if len(weighting) != len(error):
                raise ValueError("Length of `weighting` array must be of equal length to input data arrays")
            error = error*np.array(weighting)
        return error

    exp1 = Parameters()
    exp1.add("A", min=0, max=1e+4, value=np.nanmax(yarray))
    exp1.add("alpha", min=0, max=1, value=0.1)
    exp1.add("tau1", min=0, max=1e+4, value=1.0)
    exp1.add("tau2", min=0, max=1e+4, value=0.1)
    result = lmfit.minimize(exponential_res_1, exp1, method=minimizer, **minimizer_kwargs)

    # Format output
    output = np.zeros(4)
    uncertainties = np.zeros(4)
    for i,(param, value) in enumerate(result.params.items()):
        output[i] = value.value
        uncertainties[i] = value.stderr

    if verbose:
        lmfit.printfuncs.report_fit(result.params)

    return output, uncertainties

def power_law(xdata, ydata, minimizer="nelder", verbose=False, weighting=None, minimizer_kwargs={}):
    """
    Provided data fit to: ..math:`A*x^{b}` after linearizing with a log transform.

    Values of zero and NaN are ignored in the fit.

    Parameters
    ----------
    xdata : numpy.ndarray
        independent data set
    ydata : numpy.ndarray
        dependent data set
    minimizer : str, Optional, default="nelder"
        Fitting method supported by ``lmfit.minimize``
    verbose : bool, Optional, default=False
        Output fitting statistics

    Returns
    -------
    parameters : numpy.ndarray
        Array containing parameters: ["A", "alpha", "tau1", "tau2"]
    stnd_errors : numpy.ndarray
        Array containing parameter standard errors: ["A", "alpha", "tau1", "tau2"]
        
    """
    
    xarray = xdata[ydata>0]
    yarray = ydata[ydata>0]

    if np.all(np.isnan(ydata[1:])):
        raise ValueError("y-axis data is NaN")

    if np.all(weighting != None):
        weighting = weighting[ydata>0]
        if minimizer == "emcee":
            minimizer_kwargs["is_weighted"] = True
    elif minimizer == "emcee":
        minimizer_kwargs["is_weighted"] = False

    def power_law(x):
        return x["A"]*xarray**x["b"] - yarray
        if np.all(weighting != None):
            if len(weighting) != len(error):
                raise ValueError("Length of `weighting` array must be of equal length to input data arrays")
            error = error*np.array(weighting)
        return error

    exp1 = Parameters()
    exp1.add("A", min=0, max=1e+4, value=1.0)
    exp1.add("b", min=0, max=100, value=2)
    result = lmfit.minimize(power_law, exp1, method=minimizer, **minimizer_kwargs)

    # Format output
    output = np.zeros(2)
    uncertainties = np.zeros(2)
    for i,(param, value) in enumerate(result.params.items()):
        output[i] = value.value
        uncertainties[i] = value.stderr

    if verbose:
        lmfit.printfuncs.report_fit(result.params)

    return output, uncertainties

