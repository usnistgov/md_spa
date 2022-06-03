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

def exponential(xdata, ydata, minimizer="leastsq", verbose=False, save_plot=False, show_plot=False, plot_name="exponential_fit.png", kwargs_minimizer={}, kwargs_matrix_lsq={}, ydata_min=0.1, tau_max=1e+8):
    """

    CONSIDER: Use residence_time.characteristic_time instead
    Provided data fit is fit to one, two, and three exponentials, where the sum of the prefactors equals unity. 

    Values of zero and NaN are ignored in the fit.

    Parameters
    ----------
    xdata : numpy.ndarray
        independent data set
    ydata : numpy.ndarray
        dependent data set
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

    warnings.warn("Consider using residence_time.characteristic_time instead.")

    xarray = xdata[ydata>0]
    yarray = ydata[ydata>0]

    if np.all(np.isnan(ydata[1:])):
        raise ValueError("y-axis data is NaN")

    if yarray[-1] > ydata_min:
        warnings.warn("Exponential decays to {}, above threshold {}. Maximum tau value to evaluate the residence time, or increase the keyword value of ydata_min.".format(yarray[-1],ydata_min))
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

def exponential_decay(xdata, ydata, minimizer="leastsq", kwargs_minimizer={}, kwargs_parameters={}, verbose=False):
    """
    Provided data fit to:

    Values of zero and NaN are ignored in the fit.

    Parameters
    ----------
    xdata : numpy.ndarray
        independent data set
    ydata : numpy.ndarray
        dependent data set
    minimizer : str, Optional, default="leastsq"
        Fitting method supported by ``lmfit.minimize``
    kwargs_minimizer : dict, Optional, default={}
        Keyword arguments for ``lmfit.minimizer()``
    kwargs_parameters : dict, Optional
        Dictionary containing the following variables and their default keyword arguments in the form ``kwargs_parameters = {"var": {"kwarg1": var1...}}`` where ``kwargs1...`` are those from lmfit.Parameters.add() and ``var`` is one of the following parameter names.

        - ``"a1" = {"value": 1.0, "vary": False}``
        - ``"t1" = {"value": 0.1, "min": 0, "max":1e+4}``

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

    param_kwargs = {
                    "a1": {"value": 1.0, "vary": False},
                    "t1": {"value": 0.1, "min": 0, "max":1e+4},
                   }
    for key, value in kwargs_parameters.items():
        if key in param_kwargs:
            param_kwargs[key].update(value)
        else:
            raise ValueError("The parameter, {}, was given to custom_fit.exponential_decay, which requires parameters: 'a1' and 't1'".format(key))

    switch = [True for x in range(len(param_kwargs))]
    for i, (key, value) in enumerate(param_kwargs.items()):
        if "vary" in value and value["vary"] == False:
            switch[i] = False
        if "expr" in value:
            switch[i] = False

    exp1 = Parameters()
    exp1.add("a1", **param_kwargs["a1"])
    exp1.add("t1", **param_kwargs["t1"])
    if minimizer in ["leastsq"]:
        kwargs_minimizer["Dfun"] = _d_exponential_decay
    Result1 = lmfit.minimize(_res_exponential_decay, exp1, method=minimizer, args=(xarray, yarray), kws={"switch": switch}, **kwargs_minimizer)

    # Format output
    output = np.zeros(2)
    uncertainties = np.zeros(2)
    for i,(param, value) in enumerate(Result1.params.items()):
        output[i] = value.value
        uncertainties[i] = value.stderr

    if verbose:
        if minimizer == "leastsq":
            print("1 Exp. Termination: {}".format(Result1.lmdif_message))
        else:
            print("1 Exp. Termination: {}".format(Result1.message))
        lmfit.printfuncs.report_fit(Result1.params)

    return output, uncertainties

def _res_exponential_decay(params, xarray, yarray, switch=None):
    return np.log(params["a1"])-xarray/params["t1"] - np.log(yarray)

def _d_exponential_decay(params, xarray, yarray, switch=None):
    #return np.array([xarray/params["t1"]**2*np.exp(-xarray/params["t1"])])

    tmp_output = []
    tmp_output.append([1/params["a1"] for x in range(len(xarray))]) # a1
    tmp_output.append(xarray/params["t1"]**2) # t1

    output = []
    if np.all(switch != None):
        for i, tf in enumerate(switch):
            if tf:
                output.append(tmp_output[i])
    else:
        output = tmp_output

    return np.transpose(np.array(output))

def two_exponential_decays(xdata, ydata, minimizer="leastsq", kwargs_minimizer={}, kwargs_parameters={}, verbose=False):
    """
    Provided data fit to:

    Values of zero and NaN are ignored in the fit.

    Parameters
    ----------
    xdata : numpy.ndarray
        independent data set
    ydata : numpy.ndarray
        dependent data set
    minimizer : str, Optional, default="leastsq"
        Fitting method supported by ``lmfit.minimize``
    kwargs_minimizer : dict, Optional, default={}
        Keyword arguments for ``lmfit.minimizer()``
    kwargs_parameters : dict, Optional
        Dictionary containing the following variables and their default keyword arguments in the form ``kwargs_parameters = {"var": {"kwarg1": var1...}}`` where ``kwargs1...`` are those from lmfit.Parameters.add() and ``var`` is one of the following parameter names.
        Although ``kwargs_parameters["a2"]["expr"]`` can be overwritten to be None, no other expressions can be specified for vaiables if the method ``leastsq`` is used, as the Jacobian does not support this.

        - ``"a1" = {"value": 0.8, "min": 0, "max":1}``
        - ``"t1" = {"value": 0.1, "min": 0, "max":1e+4}``
        - ``"a2" = {"value": 0.2, "min": 0, "max":1, "expr":"1 - a1"}``
        - ``"t2" = {"value": 0.05, "min": 0, "max":1e+4}``

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

    param_kwargs = {
                    "a1": {"value": 0.8, "min": 0, "max":1},
                    "t1": {"value": 0.1, "min": 0, "max":1e+4},
                    "a2": {"value": 0.2, "min": 0, "max":1, "expr":"1 - a1"},
                    "t2": {"value": 0.05, "min": 0, "max":1e+4},
                   }
    for key, value in kwargs_parameters.items():
        if key in param_kwargs:
            param_kwargs[key].update(value)
        else:
            raise ValueError("The parameter, {}, was given to custom_fit.exponential_decay, which requires parameters: 'a1', 't1', 'a2', and 't2'".format(key))

    switch = [True for x in range(len(param_kwargs))]
    for i, (key, value) in enumerate(param_kwargs.items()):
        if "vary" in value and value["vary"] == False:
            switch[i] = False
        if "expr" in value:
            switch[i] = False

    exp1 = Parameters()
    exp1.add("a1", **param_kwargs["a1"])
    exp1.add("t1", **param_kwargs["t1"])
    exp1.add("a2", **param_kwargs["a2"])
    exp1.add("t2", **param_kwargs["t2"])
    if minimizer in ["leastsq"]:
        kwargs_minimizer["Dfun"] = _d_two_exponential_decays
    Result2 = lmfit.minimize(_res_two_exponential_decays, exp1, method=minimizer, args=(xarray, yarray), kws={"switch": switch}, **kwargs_minimizer)

    # Format output
    output = np.zeros(4)
    uncertainties = np.zeros(4)
    for i,(param, value) in enumerate(Result2.params.items()):
        output[i] = value.value
        uncertainties[i] = value.stderr

    if verbose:
        if minimizer == "leastsq":
            print("2 Exp. Termination: {}".format(Result2.lmdif_message))
        else:
            print("2 Exp. Termination: {}".format(Result2.message))
        lmfit.printfuncs.report_fit(Result2.params, min_correl=0.5)
        print("Sum: {}".format(Result2.params["a1"]+Result2.params["a2"]))

    return output, uncertainties

def _res_two_exponential_decays(params, xarray, yarray, switch=None):
    return params["a1"]*np.exp(-xarray/params["t1"]) + params["a2"]*np.exp(-xarray/params["t2"]) - yarray

def _d_two_exponential_decays(params, xarray, yarray, switch=None):

    tmp_output = []
    tmp_exp1 = np.exp(-xarray/params["t1"])
    tmp_exp2 = np.exp(-xarray/params["t2"])
    if not switch[2]: 
        tmp_output.append(tmp_exp1 - tmp_exp2) #a1
    else:
        tmp_output.append(tmp_exp1) #a1
    tmp_output.append(params["a1"]*xarray/params["t1"]**2*tmp_exp1) # t1
    tmp_output.append(tmp_exp2) #a2
    if not switch[2]:
        tmp_output.append((1-params["a1"])*xarray/params["t2"]**2*tmp_exp2) # t2
    else:
        tmp_output.append(params["a2"]*xarray/params["t2"]**2*tmp_exp2) # t2

    output = []
    if np.all(switch != None):
        for i, tf in enumerate(switch):
            if tf:
                output.append(tmp_output[i])
    else:
        output = tmp_output

    return np.transpose(np.array(output))

def three_exponential_decays(xdata, ydata, minimizer="leastsq", kwargs_minimizer={}, kwargs_parameters={}, verbose=False):
    """
    Provided data fit to:

    Values of zero and NaN are ignored in the fit.

    Parameters
    ----------
    xdata : numpy.ndarray
        independent data set
    ydata : numpy.ndarray
        dependent data set
    minimizer : str, Optional, default="leastsq"
        Fitting method supported by ``lmfit.minimize``
    kwargs_minimizer : dict, Optional, default={}
        Keyword arguments for ``lmfit.minimizer()``
    kwargs_parameters : dict, Optional
        Dictionary containing the following variables and their default keyword arguments in the form ``kwargs_parameters = {"var": {"kwarg1": var1...}}`` where ``kwargs1...`` are those from lmfit.Parameters.add() and ``var`` is one of the following parameter names.
        Although ``kwargs_parameters["a2"]["expr"]`` can be overwritten to be None, no other expressions can be specified for vaiables if the method ``leastsq`` is used, as the Jacobian does not support this.

        - ``"a1" = {"value": 0.8, "min": 0, "max":1}``
        - ``"t1" = {"value": 0.1, "min": 0, "max":1e+4}``
        - ``"a2" = {"value": 0.19, "min": 0, "max":1}``
        - ``"t2" = {"value": 0.09, "min": 0, "max":1e+4}``
        - ``"a3" = {"value": 0.01, "min": 0, "max":1, "expr":"1 - a1 - a2"}``
        - ``"t3" = {"value": 0.02, "min": 0, "max":1e+4}``

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

    param_kwargs = {
                    "a1": {"value": 0.8, "min": 0, "max":1},
                    "t1": {"value": 0.1, "min": 0, "max":1e+4},
                    "a2": {"value": 0.19, "min": 0, "max":1},
                    "t2": {"value": 0.09, "min": 0, "max":1e+4},
                    "a3": {"value": 0.01, "min": 0, "max": 1, "expr":"1 - a1 - a2"},
                    "t3": {"value": 0.02, "min": 0, "max":1e+4},
                   }
    for key, value in kwargs_parameters.items():
        if key in param_kwargs:
            param_kwargs[key].update(value)
        else:
            raise ValueError("The parameter, {}, was given to custom_fit.exponential_decay, which requires parameters: 'a1', 't1', 'a2', 't2', 'a3' and 't3'".format(key))

    switch = [True for x in range(len(param_kwargs))]
    for i, (key, value) in enumerate(param_kwargs.items()):
        if "vary" in value and value["vary"] == False:
            switch[i] = False
        if "expr" in value:
            switch[i] = False

    exp1 = Parameters()
    exp1.add("a1", **param_kwargs["a1"])
    exp1.add("t1", **param_kwargs["t1"])
    exp1.add("a2", **param_kwargs["a2"])
    exp1.add("t2", **param_kwargs["t2"])
    exp1.add("a3", **param_kwargs["a3"])
    exp1.add("t3", **param_kwargs["t3"])
    if minimizer in ["leastsq"]:
        kwargs_minimizer["Dfun"] = _d_three_exponential_decays
    Result3 = lmfit.minimize(_res_three_exponential_decays, exp1, method=minimizer, args=(xarray, yarray), kws={"switch": switch}, **kwargs_minimizer)

    # Format output
    output = np.zeros(6)
    uncertainties = np.zeros(6)
    for i,(param, value) in enumerate(Result3.params.items()):
        output[i] = value.value
        uncertainties[i] = value.stderr

    if verbose:
        if minimizer == "leastsq":
            print("3 Exp. Termination: {}".format(Result3.lmdif_message))
        else:
            print("3 Exp. Termination: {}".format(Result3.message))
        lmfit.printfuncs.report_fit(Result3.params, min_correl=0.5)
        print("Sum: {}".format(Result3.params["a1"]+Result3.params["a2"]+Result3.params["a3"]))

    return output, uncertainties

def _res_three_exponential_decays(params, xarray, yarray, switch=None):
    tmp1 = params["a1"]*np.exp(-xarray/params["t1"])
    tmp2 = params["a2"]*np.exp(-xarray/params["t2"])
    tmp3 = params["a3"]*np.exp(-xarray/params["t3"])
    return tmp1 + tmp2 + tmp3 - yarray

def _d_three_exponential_decays(params, xarray, yarray, switch=None):
    tmp_output = []
    tmp_exp1 = np.exp(-xarray/params["t1"])
    tmp_exp2 = np.exp(-xarray/params["t2"])
    tmp_exp3 = np.exp(-xarray/params["t3"])
    if not switch[4]: 
        tmp_output.append(tmp_exp1 - tmp_exp3) #a1
    else:
        tmp_output.append(tmp_exp1) #a1
    tmp_output.append(params["a1"]*xarray/params["t1"]**2*tmp_exp1) # t1
    if not switch[4]:
        tmp_output.append(tmp_exp2 - tmp_exp3) #a2
    else:
        tmp_output.append(tmp_exp2) #a2
    tmp_output.append(params["a2"]*xarray/params["t2"]**2*tmp_exp2) # t2
    tmp_output.append(tmp_exp3) #a3

    if not switch[4]:
        tmp_output.append((1-params["a1"])-params["a2"]*xarray/params["t3"]**2*tmp_exp3) # t3
    else:
        tmp_output.append(params["a3"]*xarray/params["t3"]**2*tmp_exp3) # t3

    output = []
    if np.all(switch != None):
        for i, tf in enumerate(switch):
            if tf:
                output.append(tmp_output[i])
    else:
        output = tmp_output

    return np.transpose(np.array(output))

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

def n_gaussians(xarray, yarray, num, minimizer="leastsq", kwargs_minimizer={}, kwargs_parameters={}, verbose=False):
    """
    Fit data to a flexible number of gaussian functions. Parameters are ``"a{}".format(n_gaussians)``, ``"b{}".format(n_gaussians)``, and ``"c{}".format(n_gaussians)``.

    Values of zero and NaN are ignored in the fit.

    Parameters
    ----------
    xdata : numpy.ndarray
        independent data set
    ydata : numpy.ndarray
        dependent data set
    num : int
        Number of Gaussian functions in model
    minimizer : str, Optional, default="leastsq"
        Fitting method supported by ``lmfit.minimize``
    kwargs_minimizer : dict, Optional, default={}
        Keyword arguments for ``lmfit.minimizer()``
    kwargs_parameters : dict, Optional
        Dictionary containing the following variables and their default keyword arguments in the form ``kwargs_parameters = {"var": {"kwarg1": var1...}}`` where ``kwargs1...`` are those from lmfit.Parameters.add() and ``var`` is one of the following parameter names.

        - ``"a{}".format(n_gaussian) = {"value": 1.0, "min": 0, "max":1e+4}``
        - ``"b{}".format(n_gaussian) = {"value": 1.0}``
        - ``"c{}".format(n_gaussian) = {"value": 0.1, "min": 0, "max":1e+4}``

    verbose : bool, Optional, default=False
        Output fitting statistics

    Returns
    -------
    parameters : numpy.ndarray
        Array containing parameters: ['t1']
    stnd_errors : numpy.ndarray
        Array containing parameter standard errors: [t1']
        
    """

    if np.all(np.isnan(yarray[1:])):
        raise ValueError("y-axis data is NaN")

    if not isinstance(kwargs_parameters, dict):
        raise ValueError("kwargs_parameters must be a dictionary")

    gaussian = Parameters()
    param_kwargs = {}
    for n in range(1,num+1):
        param_kwargs["a{}".format(n)] = {"value": 1.0, "min": 0, "max":1e+4}
        if "a{}".format(n) in kwargs_parameters:
            param_kwargs["a{}".format(n)].update(kwargs_parameters["a{}".format(n)])
            del kwargs_parameters["a{}".format(n)]
        gaussian.add("a{}".format(n), **param_kwargs["a{}".format(n)])

        param_kwargs["b{}".format(n)] = {"value": 1.0,}
        if "b{}".format(n) in kwargs_parameters:
            param_kwargs["b{}".format(n)].update(kwargs_parameters["b{}".format(n)])
            del kwargs_parameters["b{}".format(n)]
        gaussian.add("b{}".format(n), **param_kwargs["b{}".format(n)])

        param_kwargs["c{}".format(n)] = {"value": 1.0, "min": np.finfo(float).eps, "max":1e+4}
        if "c{}".format(n) in kwargs_parameters:
            param_kwargs["c{}".format(n)].update(kwargs_parameters["c{}".format(n)])
            del kwargs_parameters["c{}".format(n)]
        gaussian.add("c{}".format(n), **param_kwargs["c{}".format(n)])

    if len(kwargs_parameters) > 0:
        raise ValueError("The following parameters were given to custom_fit.num but are not used: {}".format(", ".join(list(kwargs_parameters.keys()))))

    if minimizer in ["leastsq"]:
        kwargs_minimizer["Dfun"] = _d_n_gaussians
    elif minimizer in ["trust-exact"]:
        kwargs_minimizer["jac"] = _d_n_gaussians
    Result1 = lmfit.minimize(_res_n_gaussians, gaussian, method=minimizer, args=(xarray, yarray, num), **kwargs_minimizer)

    # Format output
    output = np.zeros(3*num)
    uncertainties = np.zeros(3*num)
    for i,(param, value) in enumerate(Result1.params.items()):
        output[i] = value.value
        uncertainties[i] = value.stderr

    if verbose:
        if minimizer == "leastsq":
            print("N-Gaussians. Termination: {}".format(Result1.lmdif_message))
        else:
            print("N-Gaussians. Termination: {}".format(Result1.message))
        lmfit.printfuncs.report_fit(Result1.params)

    return output, uncertainties

def _res_n_gaussians(params, xarray, yarray, num):

    out = np.zeros(len(xarray))
    for n in range(1,num+1):
        out += params["a{}".format(n)]*np.exp(-( xarray - params["b{}".format(n)] )**2/(2*params["c{}".format(n)]**2))
        tmp = params["a{}".format(n)]*np.exp(-( xarray - params["b{}".format(n)] )**2/(2*params["c{}".format(n)]**2))

    return out - yarray

def _d_n_gaussians(params, xarray, yarray, num):

    out = []
    for n in range(1,num+1):
        tmp = ( xarray - params["b{}".format(n)] )/params["c{}".format(n)]
        out.append((np.exp(-np.square(tmp)/2))[np.newaxis,:]) # derivative of a_n 
        out.append((params["a{}".format(n)]*tmp*np.exp(-np.square(tmp)/2))[np.newaxis,:]) # derivative of b_n
        out.append((np.square(tmp)/2*np.exp(-np.square(tmp)/2))[np.newaxis,:])

    output = np.concatenate(out, axis=0)

    return np.transpose(np.array(output))

def stretched_cumulative_exponential(xarray, yarray, minimizer="leastsq", weighting=None, kwargs_minimizer={}, kwargs_parameters={}, verbose=False):
    """
    Fit data to a cumulative stretch exponential: ``f(x)=A*(1-np.exp(-(x/lc)**beta)) + C``
    This function is fit with the expression ``f(x)=A*(1-np.exp(-(x)**beta/lc_beta)) + C`` and lc is derived from the resulting exponential

    Values of zero and NaN are ignored in the fit.

    Parameters
    ----------
    xdata : numpy.ndarray
        independent data set
    ydata : numpy.ndarray
        dependent data set
    minimizer : str, Optional, default="nelder"
        Fitting method supported by ``lmfit.minimize``
    weighting : numpy.ndarray, Optional, default=None
        Of the same length as the provided data, contains the weights for each data point.
    kwargs_minimizer : dict, Optional, default={}
        Keyword arguments for ``lmfit.minimizer()``
    kwargs_parameters : dict, Optional
        Dictionary containing the following variables and their default keyword arguments in the form ``kwargs_parameters = {"var": {"kwarg1": var1...}}`` where ``kwargs1...`` are those from lmfit.Parameters.add() and ``var`` is one of the following parameter names.

        - ``"A": {"value": np.nanmax(yarray), "min": 0.0, "max": 1e+4}``
        - ``"lc_beta": {"value": np.max(xarray), "min": np.finfo(float).eps, "max":1e+4}``
        - ``"beta": {"value": 0.5, "min": 0.0, "max":1.0}``
        - ``"C": {"value": 0.0, "min": -1e+4, "max": 1e+4}``

    verbose : bool, Optional, default=False
        Output fitting statistics

    Returns
    -------
    parameters : numpy.ndarray
        Array containing parameters: ['A', 'lc', 'beta', 'C']
    stnd_errors : numpy.ndarray
        Array containing parameter standard errors: ['A', 'lc', 'beta', 'C']
        
    """

    if np.all(np.isnan(yarray[1:])):
        raise ValueError("y-axis data is NaN")

    if not isinstance(kwargs_parameters, dict):
        raise ValueError("kwargs_parameters must be a dictionary")

    if np.all(weighting != None):
        if minimizer == "emcee":
            kwargs_minimizer["is_weighted"] = True
    elif minimizer == "emcee":
        kwargs_minimizer["is_weighted"] = False
    kwargs_minimizer.update({"nan_policy": "omit"})

    tmp_tau = xarray[np.where(np.abs(yarray-np.nanmax(yarray))>0.80*np.nanmax(yarray))[0]]
    if len(tmp_tau) > 0:
        tmp_tau = tmp_tau[0]
    else:
        tmp_tau = xarray[int(len(xarray)/2)]
    param_kwargs = {
                    "A": {"value": np.nanmax(yarray), "min": 0, "max": 1e+4},
                    "lc_beta": {"value": tmp_tau, "min": np.finfo(float).eps, "max":1e+4},
                    "beta": {"value": 0.5, "min": 0.0, "max":1.0},
                    "C": {"value": 0.0, "min": -1e+4, "max": 1e+4},
                   }

    for key, value in kwargs_parameters.items():
        if key in param_kwargs:
            param_kwargs[key].update(value)
        else:
            raise ValueError("Restrictions for the parameter, {}, were given to custom_fit.stretched_cumulative_exponential although this model does not use this parameter".format(key))

    switch = [True for x in range(len(param_kwargs))]
    for i, (key, value) in enumerate(param_kwargs.items()):
        if "vary" in value and value["vary"] == False:
            switch[i] = False
        if "expr" in value:
            switch[i] = False
    
    stretched_exp = Parameters()
    for key, value in param_kwargs.items():
        stretched_exp.add(key, **value)

    if minimizer in ["leastsq"]:
        kwargs_minimizer["Dfun"] = _d_stretched_cumulative_exponential
    elif minimizer in ["trust-exact"]:
        kwargs_minimizer["jac"] = _d_stretched_cumulative_exponential
    Result1 = lmfit.minimize(_res_stretched_cumulative_exponential, stretched_exp, method=minimizer, args=(xarray, yarray), kws={"weighting": weighting, "switch": switch}, **kwargs_minimizer)

    # Format output
    output = np.zeros(4)
    uncertainties = np.zeros(4)
    for i,(param, value) in enumerate(Result1.params.items()):
        output[i] = value.value
        uncertainties[i] = value.stderr
    output[1] = output[1]**(1/output[2])
    tmp_output = 1/output[2]
    tmp_uncert = uncertainties[2]/output[2]**2
    uncertainties[1] = np.sqrt(output[1]**(2*output[2])*( (uncertainties[1]*tmp_output/output[1])**2 * (np.log(output[1])*tmp_uncert)**2 ))

    if verbose:
        if minimizer == "leastsq":
            print("N-Gaussians. Termination: {}".format(Result1.lmdif_message))
        else:
            print("N-Gaussians. Termination: {}".format(Result1.message))
        lmfit.printfuncs.report_fit(Result1.params)

    return output, uncertainties

def _res_stretched_cumulative_exponential(params, xarray, yarray, weighting=None, switch=None):

    out = params["A"]*(1-np.exp(-(xarray)**params["beta"]/params["lc_beta"])) + params["C"]

#    print(params["A"].value, params["lc_beta"].value, params["beta"].value, params["C"].value)
    if np.all(weighting != None):
        if len(weighting) != len(out):
            raise ValueError("Length of `weighting` array must be of equal length to input data arrays")
        out = out*np.array(weighting)

    return out - yarray

def _d_stretched_cumulative_exponential(params, xarray, yarray, weighting=None, switch=None):

    out = np.zeros((len(xarray),4))
    tmp_powerlaw = (xarray)**params["beta"]/params["lc_beta"]
    tmp_exp = np.exp(-tmp_powerlaw)

    out[:,0] = (1-tmp_exp)
    out[:,1] = params["A"]*tmp_exp*tmp_powerlaw/params["lc_beta"]
    out[:,2] = params["A"]*tmp_exp*tmp_powerlaw*np.log(xarray)
    out[:,3] = np.ones(len(xarray))

    output = []
    if np.all(switch != None):
        for i, tf in enumerate(switch):
            if tf:
                output.append(out[:,i])
        out = np.transpose(np.array(output))


    return out


def double_cumulative_exponential(xdata, ydata, minimizer="nelder", verbose=False, weighting=None, kwargs_minimizer={}, kwargs_parameters={}, include_C=False):
    """
    Provided data fit to:
    ..math:`y= A_{1}*\{alpha}*(1-exp(-(x/\tau_{1}))) + A_{2}*(1-\{alpha})*(1-exp(-(x/\tau_{2}))) +C` 

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
    weighting : numpy.ndarray, Optional, default=None
        Of the same length as the provided data, contains the weights for each data point.
    kwargs_minimizer : dict, Optional, default={}
        Keyword arguments for ``lmfit.minimizer()``
    kwargs_parameters : dict, Optional
        Dictionary containing the following variables and their default keyword arguments in the form ``kwargs_parameters = {"var": {"kwarg1": var1...}}`` where ``kwargs1...`` are those from lmfit.Parameters.add() and ``var`` is one of the following parameter names.

        - ``"A": {"value": max(yarray), "min": 0, "max":1e+4}``
        - ``"alpha": {"value": 0.1, "min": 0, "max":1.0}``
        - ``"tau1": {"value": xarray[yarray==max(yarray)][0], "min": 0, "max":1e+4}``
        - ``"tau2": {"value": xarray[yarray==max(yarray)][0]/2, "min": 0, "max":1e+4}``
        - ``"C": {"value": 0.0, "min": 0, "max":np.max(yarray)}``

    include_C : bool, default=False
        Whether to include vertical offset

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
            kwargs_minimizer["is_weighted"] = True
    elif minimizer == "emcee":
        kwargs_minimizer["is_weighted"] = False

    tmp_tau = xarray[np.where(np.abs(yarray-np.nanmax(yarray))>0.80*np.nanmax(yarray))[0]]
    if len(tmp_tau) > 0:
        tmp_tau = tmp_tau[0]
    else:
        tmp_tau = xarray[int(len(xarray)/2)]
    param_kwargs = {
                    "A": {"min": 0.0, "max": np.nanmax(yarray)*100, "value": np.nanmax(yarray)},
                    "alpha": {"min":0, "max":1.0, "value":0.1},
                    "tau1": {"min":0, "max":np.max(xarray)*10, "value": tmp_tau},
                    "tau2": {"min":0, "max":np.max(xarray)*10, "value": tmp_tau/2},
                   }

    if include_C:
        param_kwargs["C"] = {"min":-1e+4, "max":np.nanmax(yarray), "value": -1.0}
    exp1 = Parameters()
    for param, kwargs in param_kwargs.items():
        if param in kwargs_parameters:
            kwargs.update(kwargs_parameters[param])
            del kwargs_parameters[param]
        exp1.add(param, **kwargs)
    if len(kwargs_parameters) != 0:
        raise ValueError("The parameter(s), {}, was/were given to custom_fit.double_cumulative_exponential, which requires parameters: 'A', 'alpha', 'tau1', 'tau2'".format(list(kwargs_parameters.keys())))

    result = lmfit.minimize(_res_double_cumulative_exponential, exp1, method=minimizer, args=(xarray, yarray), kws={"weighting": weighting}, **kwargs_minimizer)

    # Format output
    lx = len(result.params)
    output = np.zeros(lx)
    uncertainties = np.zeros(lx)
    for i,(param, value) in enumerate(result.params.items()):
        output[i] = value.value
        uncertainties[i] = value.stderr

    if verbose:
        lmfit.printfuncs.report_fit(result.params)

    return output, uncertainties

def _res_double_cumulative_exponential(x, xarray, yarray, weighting=None):
    error = x["A"]*x["alpha"]*(1-np.exp(-(xarray/x["tau1"]))) \
        + x["A"]*(1-x["alpha"])*(1-np.exp(-(xarray/x["tau2"]))) - yarray
    if len(x) > 4:
        error += x["C"]
    if np.all(weighting != None):
        if len(weighting) != len(error):
            raise ValueError("Length of `weighting` array must be of equal length to input data arrays")
        error = error*np.array(weighting)
    return error

def double_viscosity_cumulative_exponential(xdata, ydata, minimizer="nelder", verbose=False, weighting=None, kwargs_minimizer={}, kwargs_parameters={}):
    """
    Provided data fit to:
    ..math:`y= A_{1}*\{alpha}*\tau_{1}*(1-exp(-(x/\tau_{1}))) + A_{2}*(1-\{alpha})*\tau_{2}*(1-exp(-(x/\tau_{2})))` 

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
    weighting : numpy.ndarray, Optional, default=None
        Of the same length as the provided data, contains the weights for each data point.
    kwargs_minimizer : dict, Optional, default={}
        Keyword arguments for ``lmfit.minimizer()``
    kwargs_parameters : dict, Optional
        Dictionary containing the following variables and their default keyword arguments in the form ``kwargs_parameters = {"var": {"kwarg1": var1...}}`` where ``kwargs1...`` are those from lmfit.Parameters.add() and ``var`` is one of the following parameter names.

        - ``"A": {"value": max(yarray), "min": 0, "max":1e+4}``
        - ``"alpha": {"value": 0.1, "min": 0, "max":1.0}``
        - ``"tau1": {"value": xarray[yarray==max(yarray)][0], "min": 0, "max":1e+4}``
        - ``"tau2": {"value": xarray[yarray==max(yarray)][0]/2, "min": 0, "max":1e+4}``

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
            kwargs_minimizer["is_weighted"] = True
    elif minimizer == "emcee":
        kwargs_minimizer["is_weighted"] = False

    tmp_tau = xarray[np.where(np.abs(yarray-np.nanmax(yarray))>0.80*np.nanmax(yarray))[0]]
    if len(tmp_tau) > 0:
        tmp_tau = tmp_tau[0]
    else:
        tmp_tau = xarray[int(len(xarray)/2)]
    param_kwargs = {
                    "A": {"min": 0.0, "max": np.nanmax(yarray)*100, "value": np.nanmax(yarray)},
                    "alpha": {"min":0, "max":1.0, "value":0.1},
                    "tau1": {"min":0, "max":np.max(xarray)*10, "value": tmp_tau},
                    "tau2": {"min":0, "max":np.max(xarray)*10, "value": tmp_tau/2},
                   }

    exp1 = Parameters()
    for param, kwargs in param_kwargs.items():
        if param in kwargs_parameters:
            kwargs.update(kwargs_parameters[param])
            del kwargs_parameters[param]
        exp1.add(param, **kwargs)
    if len(kwargs_parameters) != 0:
        raise ValueError("The parameter(s), {}, was/were given to custom_fit.double_cumulative_exponential, which requires parameters: 'A', 'alpha', 'tau1', 'tau2'".format(list(kwargs_parameters.keys())))

    result = lmfit.minimize(_res_double_cumulative_exponential, exp1, method=minimizer, args=(xarray, yarray), kws={"weighting": weighting}, **kwargs_minimizer)

    # Format output
    lx = len(result.params)
    output = np.zeros(lx)
    uncertainties = np.zeros(lx)
    for i,(param, value) in enumerate(result.params.items()):
        output[i] = value.value
        uncertainties[i] = value.stderr

    if verbose:
        lmfit.printfuncs.report_fit(result.params)

    return output, uncertainties

def _res_double_viscosity_cumulative_exponential(x, xarray, yarray, weighting=None):
    error = x["A"]*x["alpha"]*x["tau1"]*(1-np.exp(-(xarray/x["tau1"]))) \
        + x["A"]*(1-x["alpha"])*x["tau2"]*(1-np.exp(-(xarray/x["tau2"]))) - yarray
    if np.all(weighting != None):
        if len(weighting) != len(error):
            raise ValueError("Length of `weighting` array must be of equal length to input data arrays")
        error = error*np.array(weighting)
    return error

def power_law(xdata, ydata, minimizer="nelder", verbose=False, weighting=None, kwargs_minimizer={}):
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
            kwargs_minimizer["is_weighted"] = True
    elif minimizer == "emcee":
        kwargs_minimizer["is_weighted"] = False

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
    result = lmfit.minimize(power_law, exp1, method=minimizer, **kwargs_minimizer)

    # Format output
    output = np.zeros(2)
    uncertainties = np.zeros(2)
    for i,(param, value) in enumerate(result.params.items()):
        output[i] = value.value
        uncertainties[i] = value.stderr

    if verbose:
        lmfit.printfuncs.report_fit(result.params)

    return output, uncertainties

