import os
import copy
import warnings
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as spo
import scipy.special as sps

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


def jones_dole(xdata, ydata, minimizer="leastsq", kwargs_minimizer={}, kwargs_parameters={}, verbose=False, weighting=None):
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

        - ``"A" = {"value": 1.0, "min": np.finfo(float).eps, "max":1e+3}``
        - ``"B" = {"value": 0.0, "min": -1e+3, "max":1e+3}``

    verbose : bool, Optional, default=False
        Output fitting statistics
    weighting : numpy.ndarray, Optional, default=None
        Of the same length as the provided data, contains the weights for each data point.

    Returns
    -------
    parameters : numpy.ndarray
        Array containing parameters: ['A', 'B']
    stnd_errors : numpy.ndarray
        Array containing parameter standard errors: ['A', 'B']
        
    """
    kwargs_min = copy.deepcopy(kwargs_minimizer)

    if np.all(weighting != None):
        if minimizer == "emcee":
            kwargs_min["is_weighted"] = True
        weighting = weighting[ydata>0]
        if np.all(np.isnan(weighting[1:])):
            weighting = None
    elif minimizer == "emcee":
        kwargs_min["is_weighted"] = False
    kwargs_min.update({"nan_policy": "omit"})

    xdata = np.array(xdata)
    ydata = np.array(ydata)
    xarray = xdata[ydata>0]
    yarray = ydata[ydata>0]
    if np.all(np.isnan(ydata[1:])):
        raise ValueError("y-axis data is NaN")

    param_kwargs = {
                    "A": {"value": 1.0, "min": np.finfo(float).eps, "max":1e+3},
                    "B": {"value": 0.0, "min": -1e+3, "max":1e+3},
                   }
    for key, value in kwargs_parameters.items():
        if key in param_kwargs:
            param_kwargs[key].update(value)
        else:
            raise ValueError("The parameter, {}, was given to custom_fit.jones_dole, which requires parameters: 'A' and 'B'".format(key))

    switch = [True for x in range(len(param_kwargs))]
    for i, (key, value) in enumerate(param_kwargs.items()):
        if "vary" in value and value["vary"] == False:
            switch[i] = False
        if "expr" in value:
            switch[i] = False

    Params = Parameters()
    Params.add("A", **param_kwargs["A"])
    Params.add("B", **param_kwargs["B"])
    if minimizer in ["leastsq"]:
        kwargs_min["Dfun"] = _d_jones_dole
    Result1 = lmfit.minimize(_res_jones_dole, Params, method=minimizer, args=(xarray, yarray), kws={"switch": switch, "weighting": weighting}, **kwargs_min)

    # Format output
    output = np.zeros(len(param_kwargs))
    uncertainties = np.zeros(len(param_kwargs))
    for i,(param, value) in enumerate(Result1.params.items()):
        output[i] = value.value
        uncertainties[i] = value.stderr

    if verbose:
        if minimizer == "leastsq":
            print("Termination: {}".format(Result1.lmdif_message))
        elif minimizer == "emcee":
            print("Termination:  Success? {}, Aborted? {}, Chi^2: {}".format(Result1.success, Result1.aborted, Result1.chisqr))
        else:
            print("Termination: {}".format(Result1.message))
        lmfit.printfuncs.report_fit(Result1.params)

    return output, uncertainties

def _res_jones_dole(params, xarray, yarray, switch=None, weighting=None):
    ynew = (yarray - 1.0) / np.sqrt(xarray)
    out = np.sqrt(xarray) * params["B"] + params["A"] - ynew

    if np.all(weighting != None):
       if len(weighting) != len(out):
           raise ValueError("Length of `weighting` array must be of equal length to input data arrays")
       out = out*np.array(weighting * np.sqrt(xarray))

    return out

def _d_jones_dole(params, xarray, yarray, switch=None,  weighting=None):

    tmp_output = []
    tmp_output.append(np.ones(len(xarray))) # A
    tmp_output.append(np.sqrt(xarray)) # B

    output = []
    if np.all(switch != None):
        for i, tf in enumerate(switch):
            if tf:
                output.append(tmp_output[i])
    else:
        output = tmp_output

    return np.transpose(np.array(output))


def exponential_decay(xdata, ydata, minimizer="leastsq", weighting=None, kwargs_minimizer={}, kwargs_parameters={}, verbose=False, log_transform=False):
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
    weighting : numpy.ndarray, Optional, default=None
        Of the same length as the provided data, contains the weights for each data point.
    kwargs_minimizer : dict, Optional, default={}
        Keyword arguments for ``lmfit.minimizer()``
    kwargs_parameters : dict, Optional
        Dictionary containing the following variables and their default keyword arguments in the form ``kwargs_parameters = {"var": {"kwarg1": var1...}}`` where ``kwargs1...`` are those from lmfit.Parameters.add() and ``var`` is one of the following parameter names.

        - ``"a1" = {"value": 1.0, "min": np.finfo(float).eps, "max":1e+3}``
        - ``"t1" = {"value": 0.1, "min": np.finfo(float).eps, "max":1e+3}``

    verbose : bool, Optional, default=False
        Output fitting statistics
    log_transform : bool, Optional, default=False
        Choose whether to transform the data with the log of both sides

    Returns
    -------
    parameters : numpy.ndarray
        Array containing parameters: ["a1", 't1']
    stnd_errors : numpy.ndarray
        Array containing parameter standard errors: ["a1", t1']
        
    """

    kwargs_min = copy.deepcopy(kwargs_minimizer)

    if np.all(weighting != None):
        if minimizer == "emcee":
            kwargs_min["is_weighted"] = True
        weighting = weighting[ydata>0]
        if np.all(np.isnan(weighting[1:])):
            weighting = None
    elif minimizer == "emcee":
        kwargs_min["is_weighted"] = False
    kwargs_min.update({"nan_policy": "omit"})

    xarray = xdata[ydata>0]
    yarray = ydata[ydata>0]
    if np.all(np.isnan(ydata[1:])):
        raise ValueError("y-axis data is NaN")

    param_kwargs = {
                    "a1": {"value": 1.0, "min": np.finfo(float).eps, "max":1e+3},
                    "t1": {"value": 0.1, "min": np.finfo(float).eps, "max":1e+3},
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
        kwargs_min["Dfun"] = _d_exponential_decay
    Result1 = lmfit.minimize(_res_exponential_decay, exp1, method=minimizer, args=(xarray, yarray), kws={"switch": switch, "weighting": weighting, "log_transform": log_transform}, **kwargs_min)

    # Format output
    output = np.zeros(len(params_kwargs))
    uncertainties = np.zeros(len(param_kwargs))
    for i,(param, value) in enumerate(Result1.params.items()):
        output[i] = value.value
        uncertainties[i] = value.stderr

    if verbose:
        if minimizer == "leastsq":
            print("Termination: {}".format(Result1.lmdif_message))
        elif minimizer == "emcee":
            print("Termination:  Success? {}, Aborted? {}, Chi^2: {}".format(Result1.success, Result1.aborted, Result1.chisqr))
        else:
            print("Termination: {}".format(Result1.message))
        lmfit.printfuncs.report_fit(Result1.params)

    return output, uncertainties

def _res_exponential_decay(params, xarray, yarray, switch=None, weighting=None, log_transform=False):
    if log_transform:
        out = np.log(params["a1"])-xarray/params["t1"] - np.log(yarray)
    else:
        out = params["a1"]*np.exp(-xarray/params["t1"]) - yarray
    if np.all(weighting != None):
        if len(weighting) != len(out):
            raise ValueError("Length of `weighting` array must be of equal length to input data arrays")
        out = out*np.array(weighting)
    return out

def _d_exponential_decay(params, xarray, yarray, switch=None, weighting=None, log_transform=False):
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


def q_dependent_exponential_decay(xdata, ymatrix, q_array,  minimizer="leastsq", weighting=None, kwargs_minimizer={}, kwargs_parameters={}, verbose=False, log_transform=False, ymax=None, ymin=None):
    """
    Provided data fit to:

    Values of zero and NaN are ignored in the fit.

    Parameters
    ----------
    xdata : numpy.ndarray
        independent data set
    ymatrix : numpy.ndarray
        dependent data set where each row is a function to be fit corresponding to each q_value. Shape is ``(len(q_array), len(xdata))``
   q_array : numpy.ndarray
        q-values associated with first dimension of ``ymatrix``
    minimizer : str, Optional, default="leastsq"
        Fitting method supported by ``lmfit.minimize``
    weighting : numpy.ndarray, Optional, default=None
        Of the same length as the provided data, contains the weights for each data point.
    kwargs_minimizer : dict, Optional, default={}
        Keyword arguments for ``lmfit.minimizer()``
    kwargs_parameters : dict, Optional
        Dictionary containing the following variables and their default keyword arguments in the form ``kwargs_parameters = {"var": {"kwarg1": var1...}}`` where ``kwargs1...`` are those from lmfit.Parameters.add() and ``var`` is one of the following parameter names.

        - ``"A" = {"value": 1.0, "min": np.finfo(float).eps, "max":1e+3, "vary": False}`` Prefactor, one created for every q-value unless {"equal": True}
        - ``"D" = {"value": 0.1, "min": np.finfo(float).eps, "max":1e+3}``

    verbose : bool, Optional, default=False
        Output fitting statistics
    log_transform : bool, Optional, default=False
        Choose whether to transform the data with the log of both sides
    ymax : float, Optional, default=None
        Maximum value of the exponential decay to capture the long time relaxation only
    ymin : float, Optional, default=None
        Minimum value of the exponential decay to capture the short time relaxation only

    Returns
    -------
    parameters : numpy.ndarray
        Array containing parameters: ["A", 'D']
    stnd_errors : numpy.ndarray
        Array containing parameter standard errors: ["A", D']
        
    """

    xdata = np.array(xdata)
    ymatrix = np.array(ymatrix)
    q_array = np.array(q_array)

    ymatrix = ymatrix[np.argsort(q_array)]
    q_array = q_array[np.argsort(q_array)]

    kwargs_min = copy.deepcopy(kwargs_minimizer)

    if np.all(weighting != None):
        if minimizer == "emcee":
            kwargs_min["is_weighted"] = True
        if np.all(np.isnan(weighting[:,1:])):
            weighting = None
    elif minimizer == "emcee":
        kwargs_min["is_weighted"] = False
    kwargs_min.update({"nan_policy": "omit"})

    if np.all(np.isnan(ymatrix[:,1:])):
        raise ValueError("y-axis data is NaN")

    param_kwargs = {
                    "D": {"value": 0.1, "min": np.finfo(float).eps, "max":1e+3},
                   }
    if "A" in kwargs_parameters:
        if "equal" in kwargs_parameters["A"] and kwargs_parameters["A"]["equal"]:
            param_kwargs["A"] = {"value": 0.5, "min": np.finfo(float).eps, "max":1}
            del kwargs_parameters["A"]["equal"]
            param_kwargs["A"].update(kwargs_parameters["A"])
        else:
            lq = len(q_array)
            for i in range(lq):
                tmp_dict = {"value": 0.5, "min": np.finfo(float).eps, "max":1}
                if i > 0:
                    param_kwargs["A{}minusA{}".format(i,i+1)] =  {"value": 0.01, "min": np.finfo(float).eps, "max":1}
                    param_kwargs["A{}".format(i+1)] = tmp_dict
                    param_kwargs["A{}".format(i+1)]["expr"] = "A{} - A{}minusA{}".format(i, i, i+1)
                else:
                    param_kwargs["A{}".format(i+1)] = tmp_dict
                if "equal" in kwargs_parameters["A"]:
                    del kwargs_parameters["A"]["equal"]
                param_kwargs["A{}".format(i+1)].update(kwargs_parameters["A"])

        del kwargs_parameters["A"]
    else:
        for i in range(len(q_array)):
            param_kwargs["A{}".format(i+1)] = {"value": 0.5, "min": np.finfo(float).eps, "max":1}

    for key, value in kwargs_parameters.items():
        if key in param_kwargs:
            param_kwargs[key].update(value)
        else:
            raise ValueError("The parameter, {}, was given to custom_fit.exponential_decay, which requires parameters: 'A' and 'D'".format(key))

    switch = [True for x in range(len(param_kwargs))]
    for i, (key, value) in enumerate(param_kwargs.items()):
        if "vary" in value and value["vary"] == False:
            switch[i] = False
        if "expr" in value:
            switch[i] = False

    exp1 = Parameters()
    for key, value in param_kwargs.items():
        exp1.add(key, **value)

    Result1 = lmfit.minimize(_res_q_dependent_exponential_decay, exp1, method=minimizer, args=(xdata, ymatrix, q_array), 
        kws={"switch": switch, "weighting": weighting, "log_transform": log_transform, "ymax": ymax, "ymin": ymin}, **kwargs_min)

    # Format output
    lx = len([x for x in Result1.params.keys() if "minus" not in x])
    output = np.zeros(lx)
    uncertainties = np.zeros(lx)
    ii = 0
    for i,(param, value) in enumerate(Result1.params.items()):
        if "minus" not in param:
            output[ii] = value.value
            uncertainties[ii] = value.stderr
            ii += 1

    if verbose:
        if minimizer == "leastsq":
            print("Termination: {}".format(Result1.lmdif_message))
        elif minimizer == "emcee":
            print("Termination:  Success? {}, Aborted? {}, Chi^2: {}".format(Result1.success, Result1.aborted, Result1.chisqr))
        else:
            print("Termination: {}".format(Result1.message))
        lmfit.printfuncs.report_fit(Result1.params)

    return output, uncertainties

def _res_q_dependent_exponential_decay(params, xdata, ymatrix, q_array, switch=None, weighting=None, log_transform=False, ymax=None, ymin=None):

    if ymax is not None:
        ymatrix[np.where(ymatrix > ymax)] = np.nan

    if ymin is not None:
        ymatrix[np.where(ymatrix < ymin)] = np.nan

    if "A" in params:
        A_array = params["A"]
    else:
        A_array = np.array([params["A{}".format(i+1)] for i in range(len(q_array))])[:, None]

    if log_transform:
        out = np.log(A_array)-xdata[None, :] * (params["D"]*q_array[:,None]) - np.log(ymatrix)
    else:
        out = A_array * np.exp(-xdata[None,:] * (params["D"]*q_array[:,None])) - ymatrix
    if np.all(weighting != None):
        if len(weighting) != len(out):
            raise ValueError("Length of `weighting` array must be of equal length to input data arrays")
        out = out*np.array(weighting)

    return np.nansum(out, axis=0)


def q_dependent_stretched_and_exponential_decay(xdata, ymatrix, q_array,  minimizer="leastsq", weighting=None, kwargs_minimizer={}, kwargs_parameters={}, verbose=False, ymax=None, ymin=None):
    """
    Provided data fit to:

    Values of zero and NaN are ignored in the fit.

    Parameters
    ----------
    xdata : numpy.ndarray
        independent data set
    ymatrix : numpy.ndarray
        dependent data set where each row is a function to be fit corresponding to each q_value. Shape is ``(len(q_array), len(xdata))``
   q_array : numpy.ndarray
        q-values associated with first dimension of ``ymatrix``
    minimizer : str, Optional, default="leastsq"
        Fitting method supported by ``lmfit.minimize``
    weighting : numpy.ndarray, Optional, default=None
        Of the same length as the provided data, contains the weights for each data point.
    kwargs_minimizer : dict, Optional, default={}
        Keyword arguments for ``lmfit.minimizer()``
    kwargs_parameters : dict, Optional
        Dictionary containing the following variables and their default keyword arguments in the form ``kwargs_parameters = {"var": {"kwarg1": var1...}}`` where ``kwargs1...`` are those from lmfit.Parameters.add() and ``var`` is one of the following parameter names.

        - ``"D" = {"value": 0.1, "min": np.finfo(float).eps, "max":1e+3}``
        - ``"taubeta" = {"value": 1.0, "min": np.finfo(float).eps, "max": 1e+4}``
        - ``"beta" = {"value": 1.0, "min": np.finfo(float).eps, "max":2}``
        - ``"A" = {"value": 1.0, "min": np.finfo(float).eps, "max":1.0}`` Prefactor, one created for every q-value unless {"equal": True}
        - ``"A*" = {"value": 1.0}`` May provide individual coefficients, value only. 

    verbose : bool, Optional, default=False
        Output fitting statistics
    ymax : float, Optional, default=None
        Maximum value of the exponential decay to capture the long time relaxation only
    ymin : float, Optional, default=None
        Minimum value of the exponential decay to capture the short time relaxation only

    Returns
    -------
    parameters : numpy.ndarray
        Array containing parameters: ["A", 'D']
    stnd_errors : numpy.ndarray
        Array containing parameter standard errors: ["A", D']
        
    """

    xdata = np.array(xdata)
    ymatrix = np.array(ymatrix)
    q_array = np.array(q_array)

    kwargs_min = copy.deepcopy(kwargs_minimizer)

    if np.all(weighting != None):
        if minimizer == "emcee":
            kwargs_min["is_weighted"] = True
        if np.all(np.isnan(weighting[:,1:])):
            weighting = None
    elif minimizer == "emcee":
        kwargs_min["is_weighted"] = False
    kwargs_min.update({"nan_policy": "omit"})

    if np.all(np.isnan(ymatrix[:,1:])):
        raise ValueError("y-axis data is NaN")

    param_kwargs = {
                    "D": {"value": 0.1, "min": np.finfo(float).eps, "max":1e+3},
                    "taubeta": {"value": 1, "min": np.finfo(float).eps, "max": 1e+4},
                    "beta": {"value": 1.0, "min": np.finfo(float).eps, "max": 2},
                   }
    if "A" in kwargs_parameters:
        if "equal" in kwargs_parameters["A"] and kwargs_parameters["A"]["equal"]:
            param_kwargs["A"] = {"value": 0.5, "min": np.finfo(float).eps, "max":1}
            del kwargs_parameters["A"]["equal"]
            param_kwargs["A"].update(kwargs_parameters["A"])
        else:
            lq = len(q_array)
            for i in range(lq):
                tmp_dict = {"value": 0.9, "min": np.finfo(float).eps, "max": 1}
                if "A{}".format(i+1) in kwargs_parameters:
                    tmp_dict.update(kwargs_parameters["A{}".format(i+1)])
                if i > 0:
                    value = param_kwargs["A{}".format(i)]["value"] - tmp_dict["value"]
                    param_kwargs["A{}minusA{}".format(i,i+1)] =  {"value": value, "min": np.finfo(float).eps, "max": 0.3}
                    param_kwargs["A{}".format(i+1)] = tmp_dict
                    param_kwargs["A{}".format(i+1)]["expr"] = "A{} - A{}minusA{}".format(i, i, i+1)
                else:
                    param_kwargs["A{}".format(i+1)] = tmp_dict
                if "equal" in kwargs_parameters["A"]:
                    del kwargs_parameters["A"]["equal"]
                param_kwargs["A{}".format(i+1)].update(kwargs_parameters["A"])
        del kwargs_parameters["A"]
    else:
        for i in range(len(q_array)):
            param_kwargs["A{}".format(i+1)] = {"value": 0.5, "min": np.finfo(float).eps, "max":1}

    for key, value in kwargs_parameters.items():
        if key in param_kwargs:
            param_kwargs[key].update(value)
        else:
            raise ValueError("The parameter, {}, was given to custom_fit.exponential_decay, which requires parameters: 'A' and 'D'".format(key))

    switch = [True for x in range(len(param_kwargs))]
    for i, (key, value) in enumerate(param_kwargs.items()):
        if "vary" in value and value["vary"] == False:
            switch[i] = False
        if "expr" in value:
            switch[i] = False

    exp1 = Parameters()
    for key, value in param_kwargs.items():
        exp1.add(key, **value)

    Result1 = lmfit.minimize(_res_q_dependent_stretched_and_exponential_decay, exp1, method=minimizer, args=(xdata, ymatrix, q_array),
        kws={"switch": switch, "weighting": weighting, "ymax": ymax, "ymin": ymin}, **kwargs_min)

    # Format output
    lx = len([x for x in Result1.params.keys() if "minus" not in x])
    output = np.zeros(lx)
    uncertainties = np.zeros(lx)
    ii = 0
    for i,(param, value) in enumerate(Result1.params.items()):
        if "minus" not in param:
            output[ii] = value.value
            uncertainties[ii] = value.stderr
            ii += 1

    if verbose:
        if minimizer == "leastsq":
            print("Termination: {}".format(Result1.lmdif_message))
        elif minimizer == "emcee":
            print("Termination:  Success? {}, Aborted? {}, Chi^2: {}".format(Result1.success, Result1.aborted, Result1.chisqr))
        else:
            print("Termination: {}".format(Result1.message))
        lmfit.printfuncs.report_fit(Result1.params)

    return output, uncertainties

def _res_q_dependent_stretched_and_exponential_decay(params, xdata, ymatrix, q_array, switch=None, weighting=None, ymax=None, ymin=None):

    if ymax is not None:
        ymatrix[np.where(ymatrix > ymax)] = np.nan

    if ymin is not None:
        ymatrix[np.where(ymatrix < ymin)] = np.nan

    if "A" in params:
        A_array = params["A"]
    else:
        A_array = np.array([params["A{}".format(i+1)] for i in range(len(q_array))])[:, None]

    out = A_array * np.exp(-xdata[None,:] * (params["D"]*q_array[:,None])) + (1 - A_array) * np.exp(-xdata[None,:]**params["beta"] / params["taubeta"]) - ymatrix

    if np.all(weighting != None):
        if len(weighting) != len(out):
            raise ValueError("Length of `weighting` array must be of equal length to input data arrays")
        out = out*np.array(weighting)

    return np.nansum(out, axis=0)


def q_dependent_hydrodynamic_exponential_decay(xdata, ymatrix, q_array,  minimizer="leastsq", weighting=None, kwargs_minimizer={}, kwargs_parameters={}, verbose=False, ymax=None, ymin=None):
    """
    Provided data fit to:

    Values of zero and NaN are ignored in the fit.

    The parameter ``C`` should be q-dependent, but for now we assume it's constant

    Parameters
    ----------
    xdata : numpy.ndarray
        independent data set
    ymatrix : numpy.ndarray
        dependent data set where each row is a function to be fit corresponding to each q_value. Shape is ``(len(q_array), len(xdata))``
   q_array : numpy.ndarray
        q-values associated with first dimension of ``ymatrix``
    minimizer : str, Optional, default="leastsq"
        Fitting method supported by ``lmfit.minimize``
    weighting : numpy.ndarray, Optional, default=None
        Of the same length as the provided data, contains the weights for each data point.
    kwargs_minimizer : dict, Optional, default={}
        Keyword arguments for ``lmfit.minimizer()``
    kwargs_parameters : dict, Optional
        Dictionary containing the following variables and their default keyword arguments in the form ``kwargs_parameters = {"var": {"kwarg1": var1...}}`` where ``kwargs1...`` are those from lmfit.Parameters.add() and ``var`` is one of the following parameter names.

        - ``"C" = {"value": 0.5, "min": np.finfo(float).eps, "max":1}`` Prefactor, one created for every q-value unless {"equal": True}
        - ``"G" = {"value": 1.0, "min": np.finfo(float).eps, "max":1e+3}`` The Acoustic Attenuation
        - ``"c" = {"value": 1.0, "min": np.finfo(float).eps, "max":1e+3}``: The sound velocity
        - ``"tau" = {"value": 1.0, "min": np.finfo(float).eps, "max":1e+3}`` Relaxation of Q0-mode

    verbose : bool, Optional, default=False
        Output fitting statistics
    ymax : float, Optional, default=None
        Maximum value of the exponential decay to capture the long time relaxation only
    ymin : float, Optional, default=None
        Minimum value of the exponential decay to capture the short time relaxation only

    Returns
    -------
    parameters : numpy.ndarray
        Array containing parameters: ["A", 'D']
    stnd_errors : numpy.ndarray
        Array containing parameter standard errors: ["A", D']
        
    """

    xdata = np.array(xdata)
    ymatrix = np.array(ymatrix)
    ymatrix /= ymatrix[:,0][:,None]
    q_array = np.array(q_array)

    kwargs_min = copy.deepcopy(kwargs_minimizer)

    if np.all(weighting != None):
        if minimizer == "emcee":
            kwargs_min["is_weighted"] = True
        if np.all(np.isnan(weighting[:,1:])):
            weighting = None
    elif minimizer == "emcee":
        kwargs_min["is_weighted"] = False
    kwargs_min.update({"nan_policy": "omit"})

    if np.all(np.isnan(ymatrix[:,1:])):
        raise ValueError("y-axis data is NaN")

    param_kwargs = {
        "G": {"value": 1.0, "min": np.finfo(float).eps, "max":1e+3},
        "c": {"value": 1.0, "min": np.finfo(float).eps, "max":1e+3},
        "tau": {"value": 1.0, "min": np.finfo(float).eps, "max":1e+3},
    }

    if "C" in kwargs_parameters:
        if "equal" in kwargs_parameters["C"] and kwargs_parameters["C"]["equal"]:
            param_kwargs["C"] = {"value": 0.5, "min": np.finfo(float).eps, "max":1}
            del kwargs_parameters["C"]["equal"]
            param_kwargs["C"].update(kwargs_parameters["C"])
        else:
            for i in range(len(q_array)):
                param_kwargs["C{}".format(i+1)] = {"value": 0.5, "min": np.finfo(float).eps, "max":1}
                if "equal" in kwargs_parameters["C"]:
                    del kwargs_parameters["C"]["equal"]
                param_kwargs["C{}".format(i+1)].update(kwargs_parameters["C"])
        del kwargs_parameters["C"]
    else:
        for i in range(len(q_array)):
            param_kwargs["C{}".format(i+1)] = {"value": 0.5, "min": np.finfo(float).eps, "max":1}

    for key, value in kwargs_parameters.items():
        if key in param_kwargs:
            param_kwargs[key].update(value)
        else:
            raise ValueError("The parameter, {}, was given to custom_fit.exponential_decay, which requires parameters: 'A' and 'D'".format(key))

    switch = [True for x in range(len(param_kwargs))]
    for i, (key, value) in enumerate(param_kwargs.items()):
        if "vary" in value and value["vary"] == False:
            switch[i] = False
        if "expr" in value:
            switch[i] = False

    exp1 = Parameters()
    for key, value in param_kwargs.items():
        exp1.add(key, **value)

    Result1 = lmfit.minimize(_res_q_dependent_hydrodynamic_exponential_decay, exp1, method=minimizer, args=(xdata, ymatrix, q_array),
        kws={"switch": switch, "weighting": weighting, "ymax": ymax, "ymin": ymin}, **kwargs_min)

    # Format output
    output = np.zeros(len(Result1.params))
    uncertainties = np.zeros(len(Result1.params))
    for i,(param, value) in enumerate(Result1.params.items()):
        output[i] = value.value
        uncertainties[i] = value.stderr

    if verbose:
        if minimizer == "leastsq":
            print("Termination: {}".format(Result1.lmdif_message))
        elif minimizer == "emcee":
            print("Termination:  Success? {}, Aborted? {}, Chi^2: {}".format(Result1.success, Result1.aborted, Result1.chisqr))
        else:
            print("Termination: {}".format(Result1.message))
        lmfit.printfuncs.report_fit(Result1.params)

    return output, uncertainties

def _res_q_dependent_hydrodynamic_exponential_decay(params, xdata, ymatrix, q_array, switch=None, weighting=None, ymax=None, ymin=None):

    if ymax is not None:
        ymatrix[np.where(ymatrix > ymax)] = np.nan

    if ymin is not None:
        ymatrix[np.where(ymatrix < ymin)] = np.nan

    if "C" in params:
        C_array = params["C"]
    else:
        C_array = np.array([params["C{}".format(i+1)] for i in range(len(q_array))])[:, None]

    out_v = np.exp(-xdata[None,:] * params["G"] * q_array[:,None]**2) * (
        np.cos(params["c"] * q_array[:,None] * xdata[None,:]) +
        q_array[:,None] * xdata[None,:] / params["c"] * 
        np.sin(params["c"] * q_array[:,None] * xdata[None,:])
    ) 
    out_l = np.exp(-xdata[None,:] / params["tau"])

    out = (1-C_array) * out_v + C_array * out_l - ymatrix

    if np.all(weighting != None):
        if len(weighting) != len(out):
            raise ValueError("Length of `weighting` array must be of equal length to input data arrays")
        out = out*np.array(weighting)

    return np.nansum(out, axis=0)


def two_exponential_decays(xdata, ydata, minimizer="leastsq", weighting=None, kwargs_minimizer={}, kwargs_parameters={}, tau_logscale=False, verbose=False):
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
    weighting : numpy.ndarray, Optional, default=None
        Of the same length as the provided data, contains the weights for each data point.
    kwargs_minimizer : dict, Optional, default={}
        Keyword arguments for ``lmfit.minimizer()``
    kwargs_parameters : dict, Optional
        Dictionary containing the following variables and their default keyword arguments in the form ``kwargs_parameters = {"var": {"kwarg1": var1...}}`` where ``kwargs1...`` are those from lmfit.Parameters.add() and ``var`` is one of the following parameter names.
        Although ``kwargs_parameters["a2"]["expr"]`` can be overwritten to be None, no other expressions can be specified for vaiables if the method ``leastsq`` is used, as the Jacobian does not support this.

        - ``"a1" = {"value": 0.8, "min": 0, "max":1}``
        - ``"t1" = {"value": 0.1, "min": np.finfo(float).eps, "max":1e+3}``
        - ``"a2" = {"value": 0.2, "min": 0, "max":1, "expr":"1 - a1"}``
        - ``"t2" = {"value": 0.05, "min": np.finfo(float).eps, "max":1e+3}``

    tau_logscale : bool, Optional, default=False
        Have minimization algorithm fit the residence times with a log transform to search orders of magnitude
    verbose : bool, Optional, default=False
        Output fitting statistics

    Returns
    -------
    parameters : numpy.ndarray
        Array containing parameters: ['a1', 't1', 'a2', 't2']
    stnd_errors : numpy.ndarray
        Array containing parameter standard errors: ['a1', 't1', 'a2', 't2']
        
    """

    kwargs_min = copy.deepcopy(kwargs_minimizer)

    if np.all(weighting != None):
        if minimizer == "emcee":
            kwargs_min["is_weighted"] = True
        weighting = weighting[ydata>0]
        if np.all(np.isnan(weighting[1:])):
            weighting = None
    elif minimizer == "emcee":
        kwargs_min["is_weighted"] = False
    kwargs_min.update({"nan_policy": "omit"})

    xarray = xdata[ydata>0]
    yarray = ydata[ydata>0]
    if np.all(np.isnan(ydata[1:])):
        raise ValueError("y-axis data is NaN")

    param_kwargs = {
                    "a1": {"value": 0.8, "min": 0, "max":1},
                    "t1": {"value": 0.1, "min": np.finfo(float).eps, "max":1e+3},
                    "a2": {"value": 0.2, "min": 0, "max":1, "expr":"1 - a1"},
                    "t2": {"value": 0.05, "min": np.finfo(float).eps, "max":1e+3},
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
    if tau_logscale:
        param_kwargs["logt1"] = {}
        for key, value in param_kwargs["t1"].items():
            if key in ["min", "max", "value","brute_step"]:
                param_kwargs["logt1"][key] = np.log(value)
            elif key == "vary":
                param_kwargs["logt1"][key] = value
        exp1.add("logt1", **param_kwargs["logt1"])
    else:
        exp1.add("t1", **param_kwargs["t1"])
    exp1.add("a2", **param_kwargs["a2"])
    if tau_logscale:
        param_kwargs["logt2"] = {}
        for key, value in param_kwargs["t2"].items():
            if key in ["min", "max", "value","brute_step"]:
                param_kwargs["logt2"][key] = np.log(value)
            elif key == "vary":
                param_kwargs["logt2"][key] = value
        exp1.add("logt2", **param_kwargs["logt2"])
    else:
        exp1.add("t2", **param_kwargs["t2"])

    if minimizer in ["leastsq"]:
        kwargs_min["Dfun"] = _d_two_exponential_decays
    Result2 = lmfit.minimize(_res_two_exponential_decays, exp1, method=minimizer, args=(xarray, yarray), kws={"switch": switch, "weighting": weighting, "tau_logscale": tau_logscale}, **kwargs_min)

    # Format output
    output = np.zeros(len(params_kwargs))
    uncertainties = np.zeros(len(param_kwargs))
    for i,(param, value) in enumerate({key: value for key, value in Result2.params.items() if key[0] != "_"}.items()):
        if "log" in param:
            output[i] = np.exp(value.value)
            try:
                uncertainties[i] = value.value*value.stderr # Propagation of uncertainty           
            except:
                uncertainties[i] = value.stderr
        else:
            output[i] = value.value
            uncertainties[i] = value.stderr

    if verbose:
        if minimizer == "leastsq":
            print("Termination: {}".format(Result2.lmdif_message))
        elif minimizer == "emcee":
            print("Termination:  Success? {}, Aborted? {}, Chi^2: {}".format(Result2.success, Result2.aborted, Result2.chisqr))
        else:
            print("Termination: {}".format(Result2.message))
        lmfit.printfuncs.report_fit(Result2.params, min_correl=0.5)
        print("Sum: {}".format(Result2.params["a1"]+Result2.params["a2"]))

    return output, uncertainties

def _res_two_exponential_decays(params0, xarray, yarray, switch=None, weighting=None, tau_logscale=False):
    params = copy.deepcopy(params0)
    if tau_logscale:
        params.add("t1",value=np.exp(params["logt1"].value))
        params.add("t2",value=np.exp(params["logt2"].value))

    out =  params["a1"]*np.exp(-xarray/params["t1"]) + params["a2"]*np.exp(-xarray/params["t2"]) - yarray

    if np.all(weighting != None):
        if len(weighting) != len(out):
            raise ValueError("Length of `weighting` array must be of equal length to input data arrays")
        out = out*np.array(weighting)

    return out

def _d_two_exponential_decays(params0, xarray, yarray, switch=None, weighting=None, tau_logscale=False):
    params = copy.deepcopy(params0)
    if "t1" not in params:
        if not isinstance(params, lmfit.parameter.Parameters):
            params["t1"] = np.exp(params["logt1"].value)
        else:
            params.add("t1",value=np.exp(params["logt1"].value))
    if "t2" not in params:
        if not isinstance(params, lmfit.parameter.Parameters):
            params["t2"] = np.exp(params["logt2"].value)
        else:
            params.add("t2",value=np.exp(params["logt2"].value))
    tmp_output = []
    tmp_exp1 = np.exp(-xarray/params["t1"])
    tmp_exp2 = np.exp(-xarray/params["t2"])
    if not switch[2]: 
        tmp_output.append(tmp_exp1 - tmp_exp2) #a1
    else:
        tmp_output.append(tmp_exp1) #a1
    if "logt1" in params:
        tmp_output.append(params["a1"]*xarray*np.exp(-np.exp(-params["logt1"])*xarray-params["logt1"])) # logt1
    else:
        tmp_output.append(params["a1"]*xarray/params["t1"]**2*tmp_exp1) # t1
    tmp_output.append(tmp_exp2) #a2
    if not switch[2]:
        if "logt2" in params:
            tmp_output.append((1-params["a1"])*xarray*np.exp(-np.exp(-params["logt2"])*xarray-params["logt2"])) # logt1
        else:
            tmp_output.append((1-params["a1"])*xarray/params["t2"]**2*tmp_exp2) # t2
    else:
        if "logt2" in params:
            tmp_output.append(params["a2"]*xarray*np.exp(-np.exp(-params["logt2"])*xarray-params["logt2"])) # logt1
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


def three_exponential_decays(xdata, ydata, minimizer="leastsq", kwargs_minimizer={}, kwargs_parameters={}, tau_logscale=False, verbose=False, weighting=None):
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
        - ``"t1" = {"value": 0.1, "min": np.finfo(float).eps, "max":1e+3}``
        - ``"a2" = {"value": 0.19, "min": 0, "max":1}``
        - ``"t2" = {"value": 0.09, "min": np.finfo(float).eps, "max":1e+3}``
        - ``"a3" = {"value": 0.01, "min": 0, "max":1, "expr":"1 - a1 - a2"}``
        - ``"t3" = {"value": 0.02, "min": np.finfo(float).eps, "max":1e+3}``

    tau_logscale : bool, Optional, default=False
        Have minimization algorithm fit the residence times with a log transform to search orders of magnitude
    weighting : numpy.ndarray, Optional, default=None
        Of the same length as the provided data, contains the weights for each data point.
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
    kwargs_min = copy.deepcopy(kwargs_minimizer)

    if np.all(weighting != None):
        if minimizer == "emcee":
            kwargs_min["is_weighted"] = True
        weighting = weighting[ydata>0]
        if np.all(np.isnan(weighting[1:])):
            weighting = None
    elif minimizer == "emcee":
        kwargs_min["is_weighted"] = False
    kwargs_min.update({"nan_policy": "omit"})

    if np.all(np.isnan(ydata[1:])):
        raise ValueError("y-axis data is NaN")

    param_kwargs = {
                    "a1": {"value": 0.8, "min": 0, "max":1},
                    "t1": {"value": 0.1, "min": np.finfo(float).eps, "max":1e+3},
                    "a2": {"value": 0.19, "min": 0, "max":1},
                    "t2": {"value": 0.09, "min": np.finfo(float).eps, "max":1e+3},
                    "a3": {"value": 0.01, "min": 0, "max": 1, "expr":"1 - a1 - a2"},
                    "t3": {"value": 0.02, "min": np.finfo(float).eps, "max":1e+3},
                   }
    for key, value in kwargs_parameters.items():
        if key in param_kwargs:
            param_kwargs[key].update(value)
        else:
            raise ValueError("The parameter, {}, was given to custom_fit.exponential_decay, which requires parameters: 'a1', 't1', 'a2', 't2', 'a3' and 't3'".format(key))

    switch = [0 for x in range(len(param_kwargs))]
    for i, (key, value) in enumerate(param_kwargs.items()):
        if "vary" in value and value["vary"] == False:
            switch[i] = 1
        if "expr" in value:
            switch[i] = 2

    exp1 = Parameters()
    exp1.add("a1", **param_kwargs["a1"])
    if tau_logscale:
        param_kwargs["logt1"] = {}
        for key, value in param_kwargs["t1"].items():
            if key in ["min", "max", "value","brute_step"]:
                param_kwargs["logt1"][key] = np.log(value)
            elif key == "vary":
                param_kwargs["logt1"][key] = value
        exp1.add("logt1", **param_kwargs["logt1"])
    else:
        exp1.add("t1", **param_kwargs["t1"])
    exp1.add("a2", **param_kwargs["a2"])
    if tau_logscale:
        param_kwargs["logt2"] = {}
        for key, value in param_kwargs["t2"].items():
            if key in ["min", "max", "value","brute_step"]:
                param_kwargs["logt2"][key] = np.log(value)
            elif key == "vary":
                param_kwargs["logt2"][key] = value
        exp1.add("logt2", **param_kwargs["logt2"])
    else:
        exp1.add("t2", **param_kwargs["t2"])
    exp1.add("a3", **param_kwargs["a3"])
    if tau_logscale:
        param_kwargs["logt3"] = {}
        for key, value in param_kwargs["t3"].items():
            if key in ["min", "max", "value","brute_step"]:
                param_kwargs["logt3"][key] = np.log(value)
            elif key == "vary":
                param_kwargs["logt3"][key] = value
        exp1.add("logt3", **param_kwargs["logt3"])
    else:
        exp1.add("t3", **param_kwargs["t3"])

    if minimizer in ["leastsq"]:
        kwargs_min["Dfun"] = _d_three_exponential_decays
    Result3 = lmfit.minimize(_res_three_exponential_decays, exp1, method=minimizer, args=(xarray, yarray), kws={"switch": switch, "weighting": weighting}, **kwargs_min)

    # Format output
    output = np.zeros(len(params_kwargs))
    uncertainties = np.zeros(len(param_kwargs))
    for i,(param, value) in enumerate(Result3.params.items()):
        if "log" in param:
            output[i] = np.exp(value.value)
            try:
                uncertainties[i] = value.value*value.stderr # Propagation of uncertainty           
            except:
                uncertainties[i] = value.stderr
        else:
            output[i] = value.value
            uncertainties[i] = value.stderr

    if verbose:
        if minimizer == "leastsq":
            print("Termination: {}".format(Result3.lmdif_message))
        elif minimizer == "emcee":
            print("Termination:  Success? {}, Aborted? {}, Chi^2: {}".format(Result3.success, Result3.aborted, Result3.chisqr))
        else:
            print("Termination: {}".format(Result3.message))
        lmfit.printfuncs.report_fit(Result3.params, min_correl=0.5)
        print("Sum: {}".format(Result3.params["a1"]+Result3.params["a2"]+Result3.params["a3"]))

    return output, uncertainties

def _res_three_exponential_decays(params0, xarray, yarray, switch=None, weighting=None):
    params = copy.deepcopy(params0)
    if "t1" not in params:
        if not isinstance(params, lmfit.parameter.Parameters):
            params["t1"] = np.exp(params["logt1"].value)
        else:
            params.add("t1",value=np.exp(params["logt1"].value))
    if "t2" not in params:
        if not isinstance(params, lmfit.parameter.Parameters):
            params["t2"] = np.exp(params["logt2"].value)
        else:
            params.add("t2",value=np.exp(params["logt2"].value))
    if "t3" not in params:
        if not isinstance(params, lmfit.parameter.Parameters):
            params["t3"] = np.exp(params["logt3"].value)
        else:
            params.add("t3",value=np.exp(params["logt3"].value))
    tmp1 = params["a1"]*np.exp(-xarray/params["t1"])
    tmp2 = params["a2"]*np.exp(-xarray/params["t2"])
    tmp3 = params["a3"]*np.exp(-xarray/params["t3"])

    out = tmp1 + tmp2 + tmp3 - yarray
    if np.all(weighting != None):
        if len(weighting) != len(out):
            raise ValueError("Length of `weighting` array must be of equal length to input data arrays")
        out = out*np.array(weighting)
    return out

def _d_three_exponential_decays(params0, xarray, yarray, switch=None, weighting=None):
    params = copy.deepcopy(params0)
    if "t1" not in params:
        if not isinstance(params, lmfit.parameter.Parameters):
            params["t1"] = np.exp(params["logt1"].value)
        else:
            params.add("t1",value=np.exp(params["logt1"].value))
    if "t2" not in params:
        if not isinstance(params, lmfit.parameter.Parameters):
            params["t2"] = np.exp(params["logt2"].value)
        else:
            params.add("t2",value=np.exp(params["logt2"].value))
    if "t3" not in params:
        if not isinstance(params, lmfit.parameter.Parameters):
            params["t3"] = np.exp(params["logt3"].value)
        else:
            params.add("t3",value=np.exp(params["logt3"].value))
    tmp_output = []
    tmp_exp1 = np.exp(-xarray/params["t1"])
    tmp_exp2 = np.exp(-xarray/params["t2"])
    tmp_exp3 = np.exp(-xarray/params["t3"])
    if switch[4] == 2: 
        tmp_output.append(tmp_exp1 - tmp_exp3) #a1
    else:
        tmp_output.append(tmp_exp1) #a1
    if "logt1" in params:
        tmp_output.append(params["a1"]*xarray*np.exp(-np.exp(-params["logt1"])*xarray-params["logt1"])) # logt1
    else:
        tmp_output.append(params["a1"]*xarray/params["t1"]**2*tmp_exp1) # t1
    if switch[4] == 2:
        tmp_output.append(tmp_exp2 - tmp_exp3) #a2
    else:
        tmp_output.append(tmp_exp2) #a2
    if "logt2" in params:
        tmp_output.append(params["a2"]*xarray*np.exp(-np.exp(-params["logt2"])*xarray-params["logt2"])) # logt2
    else:
        tmp_output.append(params["a2"]*xarray/params["t2"]**2*tmp_exp2) # t2
    tmp_output.append(tmp_exp3) #a3

    if switch[4] == 2:
        if "logt3" in params:
            tmp_output.append((1-params["a1"]-params["a2"])*xarray*np.exp(-np.exp(-params["logt3"])*xarray-params["logt3"])) # logt3
        else:
            tmp_output.append((1-params["a1"]-params["a2"])*xarray/params["t3"]**2*tmp_exp3) # t3
    else:
        if "logt3" in params:
            tmp_output.append(params["a3"]*xarray*np.exp(-np.exp(-params["logt3"])*xarray-params["logt3"])) # logt3
        else:
            tmp_output.append(params["a3"]*xarray/params["t3"]**2*tmp_exp3) # t3

    output = []
    if np.all(switch != None):
        for i, tf in enumerate(switch):
            if tf == 0:
                output.append(tmp_output[i])
    else:
        output = tmp_output

    return np.transpose(np.array(output))


def scattering_3_exponential_decays(xdata, ydata, minimizer="leastsq", kwargs_minimizer={}, kwargs_parameters={}, tau_logscale=False, verbose=False):
    r"""
    Provided data fit to: 
    ..math:`y= (1 - C)exp(-(x/\tau_{1})) + C(1+A)exp(-(x/\tau_{2})) + C \cdot Aexp(-(x/\tau_{3}))` 

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

        - ``"C" = {"value": 0.8, "min": 0, "max":1}``
        - ``"A" = {"value": 0.8, "min": 0, "max":1}``
        - ``"t1" = {"value": 0.02, "min": np.finfo(float).eps, "max":1e+4}``
        - ``"t2" = {"value": 0.09, "min": np.finfo(float).eps, "max":1e+4}``
        - ``"t3" = {"value": 1.0, "min": np.finfo(float).eps, "max":1e+4``

    tau_logscale : bool, Optional, default=False
        Have minimization algorithm fit the residence times with a log transform to search orders of magnitude
    verbose : bool, Optional, default=False
        Output fitting statistics

    Returns
    -------
    parameters : numpy.ndarray
        Array containing parameters: ['C', 'A', 't1', 't2', 't3']
    stnd_errors : numpy.ndarray
        Array containing parameter standard errors: ['C', 'A', 't1', 't2', 't3']
        
    """

    xarray = xdata[ydata>0]
    yarray = ydata[ydata>0]
    kwargs_min = copy.deepcopy(kwargs_minimizer)

    if np.all(np.isnan(ydata[1:])):
        raise ValueError("y-axis data is NaN")

    param_kwargs = {
                    "C": {"value": 0.8, "min": 0, "max":1},
                    "A": {"value": 0.8, "min": 0, "max":1},
                    "t1": {"value": 0.02, "min": np.finfo(float).eps, "max":1e+4},
                    "t2": {"value": 0.09, "min": np.finfo(float).eps, "max":1e+4},
                    "t3": {"value": 1.0, "min": np.finfo(float).eps, "max":1e+4},
                   }
    for key, value in kwargs_parameters.items():
        if key in param_kwargs:
            param_kwargs[key].update(value)
        else:
            raise ValueError("The parameter, {}, was given to custom_fit.exponential_decay, which requires parameters: '{}'".format(key, "', '".join(list(param_kwargs.keys()))))

    switch = [True for x in range(len(param_kwargs))]
    for i, (key, value) in enumerate(param_kwargs.items()):
        if "vary" in value and value["vary"] == False:
            switch[i] = False
        if "expr" in value:
            switch[i] = False

    exp1 = Parameters()
    exp1.add("C", **param_kwargs["C"])
    exp1.add("A", **param_kwargs["A"])
    log_params = ["t1", "t2", "t3"]
    if tau_logscale:
        log_params = ["t1", "t2", "t3"]
        for param in log_params:
            new_param = "log{}".format(param)
            param_kwargs[new_param] = {}
            for key, value in param_kwargs[param].items():
                if key in ["min", "max", "value","brute_step"]:
                    param_kwargs[new_param][key] = np.log(value)
                elif key == "vary":
                    param_kwargs[new_param][key] = value
            exp1.add(new_param, **param_kwargs[new_param])
    else:
        for param in log_params:
            exp1.add(param, **param_kwargs[param])

    if minimizer in ["leastsq"]:
        kwargs_min["Dfun"] = _d_scattering_3_exponential_decays
    Result3 = lmfit.minimize(_res_scattering_3_exponential_decays, exp1, method=minimizer, args=(xarray, yarray), kws={"switch": switch}, **kwargs_min)

    # Format output
    output = np.zeros(len(params_kwargs))
    uncertainties = np.zeros(len(param_kwargs))
    for i,(param, value) in enumerate(Result3.params.items()):
        if "log" in param:
            output[i] = np.exp(value.value)
            try:
                uncertainties[i] = value.value*value.stderr # Propagation of uncertainty           
            except:
                uncertainties[i] = value.stderr
        else:
            output[i] = value.value
            uncertainties[i] = value.stderr

    if verbose:
        if minimizer == "leastsq":
            print("Termination: {}".format(Result3.lmdif_message))
        elif minimizer == "emcee":
            print("Termination:  Success? {}, Aborted? {}, Chi^2: {}".format(Result3.success, Result3.aborted, Result3.chisqr))
        else:
            print("Termination: {}".format(Result3.message))
        lmfit.printfuncs.report_fit(Result3.params, min_correl=0.5)

    return output, uncertainties

def _res_scattering_3_exponential_decays(params0, xarray, yarray, switch=None):

    params = copy.deepcopy(params0)
    if "t1" not in params:
        if not isinstance(params, lmfit.parameter.Parameters):
            params["t1"] = np.exp(params["logt1"].value)
            params["t2"] = np.exp(params["logt2"].value)
            params["t3"] = np.exp(params["logt3"].value)
        else:
            params.add("t1",value=np.exp(params["logt1"].value))
            params.add("t2",value=np.exp(params["logt2"].value))
            params.add("t3",value=np.exp(params["logt3"].value))

    tmp1 = (1.0-params["C"])*np.exp(-xarray/params["t1"])
    tmp2 = params["C"]*(1.0-params["A"])*np.exp(-xarray/params["t2"])
    tmp3 = params["C"]*params["A"]*np.exp(-xarray/params["t3"])
#    return np.log(tmp1 + tmp2 + tmp3) - np.log(yarray)

    return tmp1 + tmp2 + tmp3 - yarray

def _d_scattering_3_exponential_decays(params0, xarray, yarray, switch=None):
    params = copy.deepcopy(params0)
    if "t1" not in params:
        if not isinstance(params, lmfit.parameter.Parameters):
            params["t1"] = np.exp(params["logt1"].value)
            params["t2"] = np.exp(params["logt2"].value)
            params["t3"] = np.exp(params["logt3"].value)
        else:
            params.add("t1",value=np.exp(params["logt1"].value))
            params.add("t2",value=np.exp(params["logt2"].value))
            params.add("t3",value=np.exp(params["logt3"].value))

    tmp_output = []
    tmp_exp1 = np.exp(-xarray/params["t1"])
    tmp_exp2 = np.exp(-xarray/params["t2"])
    tmp_exp3 = np.exp(-xarray/params["t3"])

    tmp_output.append(-tmp_exp1 + (1.0-params["A"])*tmp_exp2 + params["A"]*tmp_exp3) # C
    tmp_output.append(-params["C"]*tmp_exp2 + params["C"]*tmp_exp3) # A
    if "logt1" in params:
        tmp_output.append((1.0-params["C"])*xarray*np.exp(-xarray/np.exp(params["logt1"])-params["logt1"])) # logt1
        tmp_output.append(params["C"]*(1.0-params["A"])*xarray*np.exp(-xarray/np.exp(params["logt2"])-params["logt2"])) # logt2
        tmp_output.append(params["A"]*params["C"]*xarray*np.exp(-xarray/np.exp(params["logt3"])-params["logt3"])) # logt3
    else:
        tmp_output.append(-(1.0-params["C"])*xarray/params["t1"]**2*tmp_exp1) # t1
        tmp_output.append(-params["C"]*(1.0-params["A"])*xarray/params["t2"]**2*tmp_exp2) # t2
        tmp_output.append(-params["A"]*params["C"]*xarray/params["t3"]**2*tmp_exp3) # t3

    output = []
    if np.all(switch != None):
        for i, tf in enumerate(switch):
            if tf:
                output.append(tmp_output[i])
    else:
        output = tmp_output

    return np.transpose(np.array(output))


def stretched_exponential_decay(xdata, ydata, minimizer="leastsq", kwargs_minimizer={}, kwargs_parameters={}, verbose=False, weighting=None):
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

        - ``"taubeta" = {"value": 1.0, "min": np.finfo(float).eps, "max":1e+4}`` : Calculated as a constant, the time constant can be extract when beta is known.
        - ``"beta" = {"value": 3/2, "min": np.finfo(float).eps, "max":5}``

    weighting : numpy.ndarray, Optional, default=None
        Of the same length as the provided data, contains the weights for each data point.
    verbose : bool, Optional, default=False
        Output fitting statistics

    Returns
    -------
    parameters : numpy.ndarray
        Array containing parameters: ['tau']
    stnd_errors : numpy.ndarray
        Array containing parameter standard errors: [beta']
        
    """

    kwargs_min = copy.deepcopy(kwargs_minimizer)

    if np.all(weighting != None):
        if minimizer == "emcee":
            kwargs_min["is_weighted"] = True
        weighting = weighting[ydata>0]
        if np.all(np.isnan(weighting[1:])):
            weighting = None
    elif minimizer == "emcee":
        kwargs_min["is_weighted"] = False
    kwargs_min.update({"nan_policy": "omit"})

    xarray = xdata[ydata>0]
    yarray = ydata[ydata>0]
    if np.all(np.isnan(ydata[1:])):
        raise ValueError("y-axis data is NaN")

    param_kwargs = {
                    "taubeta": {"value": 1.0, "min": np.finfo(float).eps, "max":1e+4},
                    "beta": {"value": 0.1, "min": np.finfo(float).eps, "max":5},
                   }
    for key, value in kwargs_parameters.items():
        if key in param_kwargs:
            param_kwargs[key].update(value)
        else:
            raise ValueError("The parameter, {}, was given to custom_fit.stretched_exponential_decay, which requires parameters: 'tau' and 'beta'".format(key))

    exp = Parameters()
    switch = [True for x in range(len(param_kwargs))]
    for i, (key, value) in enumerate(param_kwargs.items()):
        exp.add(key, **param_kwargs[key])
        if "vary" in value and value["vary"] == False:
            switch[i] = False
        if "expr" in value:
            switch[i] = False

    if minimizer in ["leastsq"]:
        kwargs_min["Dfun"] = _d_stretched_exponential_decay
    Result1 = lmfit.minimize(_res_stretched_exponential_decay, exp, method=minimizer, args=(xarray, yarray), kws={"switch": switch, "weighting": weighting}, **kwargs_min)

    # Format output
    output = np.zeros(len(param_kwargs))
    uncertainties = np.zeros(len(param_kwargs))
    for i,(param, value) in enumerate(Result1.params.items()):
        output[i] = value.value
        uncertainties[i] = value.stderr

    if verbose:
        if minimizer == "leastsq":
            print("Termination: {}".format(Result1.lmdif_message))
        elif minimizer == "emcee":
            print("Termination:  Success? {}, Aborted? {}, Chi^2: {}".format(Result1.success, Result1.aborted, Result1.chisqr))
        else:
            print("Termination: {}".format(Result1.message))
        lmfit.printfuncs.report_fit(Result1.params)

    return output, uncertainties

def _res_stretched_exponential_decay(params, xarray, yarray, switch=None, weighting=None,):
    out =  np.exp(-xarray**params["beta"]/params["taubeta"]) - yarray
    if np.all(weighting != None):
        if len(weighting) != len(out):
            raise ValueError("Length of `weighting` array must be of equal length to input data arrays")
        out = out*np.array(weighting)
    return out


def _d_stretched_exponential_decay(params, xarray, yarray, switch=None, weighting=None,):

    tmp_output = []
    ratio = xarray**params["beta"]/params["taubeta"]
    exp = np.exp(-ratio)
    tmp_output.append( ratio / params["taubeta"] * exp )# taubeta
    tmp_output.append( -np.log(xarray) * ratio * exp ) # beta

    output = []
    if np.all(switch != None):
        for i, tf in enumerate(switch):
            if tf:
                output.append(tmp_output[i])
    else:
        output = tmp_output

    if len(tmp_output) == 2 and np.isnan(tmp_output[1][0]):
        raise ValueError("Dfun in scipy.optimize.leastsq cannot handle NaN values, please exclude t=0 from the fit or don't analytically calculate the Jacobian.")

    return np.transpose(np.array(output))


def two_stretched_exponential_decays(xdata, ydata, minimizer="leastsq", kwargs_minimizer={}, kwargs_parameters={}, verbose=False, weighting=None):
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
    weighting : numpy.ndarray, Optional, default=None
        Of the same length as the provided data, contains the weights for each data point.

        - ``"A" = {"value": 0.8, "min": 0, "max":1}``
        - ``"tau1beta1" = {"value": 0.5, "min": np.finfo(float).eps, "max":1e+4}``
        - ``"beta1" = {"value": 1/2, "min": np.finfo(float).eps, "max":5}``
        - ``"tau2beta2" = {"value": 0.5, "min": np.finfo(float).eps, "max":1e+4}``
        - ``"beta2" = {"value": 3/2, "min": np.finfo(float).eps, "max":5}``

    verbose : bool, Optional, default=False
        Output fitting statistics

    Returns
    -------
    parameters : numpy.ndarray
        Array containing parameters: ['A', 'tau1', 'beta1', 'tau2', 'beta2']
    stnd_errors : numpy.ndarray
        Array containing parameter standard errors: ['A', 'tau1', 'beta1', 'tau2', 'beta2']
        
    """

    kwargs_min = copy.deepcopy(kwargs_minimizer)

    if np.all(weighting != None):
        if minimizer == "emcee":
            kwargs_min["is_weighted"] = True
        weighting = weighting[ydata>0]
        if np.all(np.isnan(weighting[1:])):
            weighting = None
    elif minimizer == "emcee":
        kwargs_min["is_weighted"] = False
    kwargs_min.update({"nan_policy": "omit"})

    xarray = xdata[ydata>0]
    yarray = ydata[ydata>0]
    if np.all(np.isnan(ydata[1:])):
        raise ValueError("y-axis data is NaN")

    param_kwargs = {
        "A": {"value": 0.8, "min": np.finfo(float).eps, "max":1},
        "tau1beta1": {"value": 0.5, "min": np.finfo(float).eps, "max":1e+4},
        "beta1": {"value": 1/2, "min": np.finfo(float).eps, "max":5},
        "tau2beta2": {"value": 0.5, "min": np.finfo(float).eps, "max":1e+4},
        "beta2": {"value": 3/2, "min": np.finfo(float).eps, "max":5},
    }
    for key, value in kwargs_parameters.items():
        if key in param_kwargs:
            param_kwargs[key].update(value)
        else:
            raise ValueError("The parameter, {}, was given to custom_fit.exponential_decay, which requires parameters: {}".format(key, ", ".join(param_kwargs.keys())))

    exp = Parameters()
    switch = [True for x in range(len(param_kwargs))]
    for i, (key, value) in enumerate(param_kwargs.items()):
        exp.add(key, **value)
        if "vary" in value and value["vary"] == False:
            switch[i] = False
        if "expr" in value:
            switch[i] = False

    if minimizer in ["leastsq"]:
        kwargs_min["Dfun"] = _d_two_stretched_exponential_decays
    Result2 = lmfit.minimize(
        _res_two_stretched_exponential_decays, 
        exp, 
        method=minimizer, 
        args=(xarray, yarray), 
        kws={"switch": switch, 
        "weighting": weighting}, 
        **kwargs_min
    )

    # Format output
    output = np.zeros(len(param_kwargs))
    uncertainties = np.zeros(len(param_kwargs))
    for i,(param, value) in enumerate(Result2.params.items()):
        output[i] = value.value
        uncertainties[i] = value.stderr

    if verbose:
        #for key, value in Result2.__dict__.items():
        #    print("\n", key, value)
        if minimizer == "leastsq":
            print("Termination: {}".format(Result2.lmdif_message))
        elif minimizer == "emcee":
            print("Termination:  Success? {}, Aborted? {}, Chi^2: {}".format(Result2.success, Result2.aborted, Result2.chisqr))
        else:
            print("Termination: {}".format(Result2.message))
        lmfit.printfuncs.report_fit(Result2.params, min_correl=0.5)

    return output, uncertainties

def _res_two_stretched_exponential_decays(params, xarray, yarray, switch=None, weighting=None,):
   
    out =  params["A"]*np.exp(-xarray**params["beta1"] / params["tau1beta1"]) + (1-params["A"])*np.exp(-xarray**params["beta2"] / params["tau2beta2"]) - yarray
    if np.all(weighting != None):
        if len(weighting) != len(out):
            raise ValueError("Length of `weighting` array must be of equal length to input data arrays")
        out = out*np.array(weighting)

    return out

def _d_two_stretched_exponential_decays(params, xarray, yarray, switch=None, weighting=None,):

    tmp_output = []
    ratio1 = xarray**params["beta1"] / params["tau1beta1"]
    ratio1 = xarray**params["beta2"] / params["tau2beta2"]
    exp1 = np.exp(-ratio1)
    exp2 = np.exp(-ratio2)
    tmp_output.append(exp1-exp2)
    tmp_output.append( params["A"] * ratio1 / params["tau1beta1"] * exp1 ) # tau1
    tmp_output.append( -params["A"]*np.log(xarray) * ratio1 * exp1 ) # beta1 
    tmp_output.append( -params["A"] * ratio2 / params["tau2beta2"] * exp2 ) # tau2
    tmp_output.append( params["A"]*np.log(xarray) * ratio2 * exp2 ) # beta2 

    output = []
    if np.all(switch != None):
        for i, tf in enumerate(switch):
            if tf:
                output.append(tmp_output[i])
                if i == 2 and np.isnan(tmp_output[i][0]):
                    raise ValueError("Dfun in scipy.optimize.leastsq cannot handle NaN values, please exclude t=0 from the fit or don't analytically calculate the Jacobian.")
    else:
        output = tmp_output
        if np.isnan(tmp_output[2][0]):
            raise ValueError("Dfun in scipy.optimize.leastsq cannot handle NaN values, please exclude t=0 from the fit or don't analytically calculate the Jacobian.")

    return np.transpose(np.array(output))


def reg_n_stretched_exponential_decays(xdata, ydata, minimizer="leastsq", kwargs_minimizer={}, kwargs_parameters={}, verbose=False, weighting=None):
    """
    Provided data fit to a regular and stretched exponential

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
    weighting : numpy.ndarray, Optional, default=None
        Of the same length as the provided data, contains the weights for each data point.

        - ``"A" = {"value": 0.8, "min": 0, "max":1}``
        - ``"tau1beta1" = {"value": 0.5, "min": np.finfo(float).eps, "max":1e+4}``
        - ``"beta1" = {"value": 1/2, "min": np.finfo(float).eps, "max":5}``
        - ``"tau2" = {"value": 0.5, "min": np.finfo(float).eps, "max":1e+2}``

    verbose : bool, Optional, default=False
        Output fitting statistics

    Returns
    -------
    parameters : numpy.ndarray
        Array containing parameters: ['A', 'tau1', 'beta1', 'tau2', 'beta2']
    stnd_errors : numpy.ndarray
        Array containing parameter standard errors: ['A', 'tau1', 'beta1', 'tau2', 'beta2']
        
    """

    kwargs_min = copy.deepcopy(kwargs_minimizer)

    if np.all(weighting != None):
        if minimizer == "emcee":
            kwargs_min["is_weighted"] = True
        weighting = weighting[ydata>0]
        if np.all(np.isnan(weighting[1:])):
            weighting = None
    elif minimizer == "emcee":
        kwargs_min["is_weighted"] = False
    kwargs_min.update({"nan_policy": "omit"})

    xarray = xdata[ydata>0]
    yarray = ydata[ydata>0]
    if np.all(np.isnan(ydata[1:])):
        raise ValueError("y-axis data is NaN")

    param_kwargs = {
        "A": {"value": 0.8, "min": 0, "max":1},
        "tau1beta1": {"value": 0.5, "min": np.finfo(float).eps, "max":1e+4},
        "beta1": {"value": 1/2, "min": np.finfo(float).eps, "max":5},
        "tau2": {"value": 0.5, "min": np.finfo(float).eps, "max":1e+2},
    }
    for key, value in kwargs_parameters.items():
        if key in param_kwargs:
            param_kwargs[key].update(value)
        else:
            raise ValueError("The parameter, {}, was given to custom_fit.exponential_decay, which requires parameters: 'A', 'tau1', 'beta1', 'tau2' and 'beta2'".format(key))

    exp = Parameters()
    switch = [True for x in range(len(param_kwargs))]
    for i, (key, value) in enumerate(param_kwargs.items()):
        exp.add(key, **value)
        if "vary" in value and value["vary"] == False:
            switch[i] = False
        if "expr" in value:
            switch[i] = False

    if minimizer in ["leastsq"]:
        kwargs_min["Dfun"] = _d_reg_n_stretched_exponential_decays
    Result2 = lmfit.minimize(
        _res_reg_n_stretched_exponential_decays,
        exp,
        method=minimizer,
        args=(xarray, yarray),
        kws={"switch": switch,
        "weighting": weighting},
        **kwargs_min
    )

    # Format output
    output = np.zeros(len(param_kwargs))
    uncertainties = np.zeros(len(param_kwargs))
    for i,(param, value) in enumerate(Result2.params.items()):
        output[i] = value.value
        uncertainties[i] = value.stderr

    if verbose:
        if minimizer == "leastsq":
            print("Termination: {}".format(Result2.lmdif_message))
        elif minimizer == "emcee":
            print("Termination:  Success? {}, Aborted? {}, Chi^2: {}".format(Result2.success, Result2.aborted, Result2.chisqr))
        else:
            print("Termination: {}".format(Result2.message))
        lmfit.printfuncs.report_fit(Result2.params, min_correl=0.5)

    return output, uncertainties

def _res_reg_n_stretched_exponential_decays(params, xarray, yarray, switch=None, weighting=None,):

    out =  params["A"]*np.exp(-xarray**params["beta1"]/params["tau1beta1"]) + (1-params["A"])*np.exp(-(xarray/params["tau2"])) - yarray
    if np.all(weighting != None):
        if len(weighting) != len(out):
            raise ValueError("Length of `weighting` array must be of equal length to input data arrays")
        out = out*np.array(weighting)

    return out

def _d_reg_n_stretched_exponential_decays(params, xarray, yarray, switch=None, weighting=None,):

    tmp_output = []
    ratio1 = xarray**params["beta1"] / params["tau1beta1"]
    exp1 = np.exp(-ratio1)
    exp2 = np.exp(-xarray/params["tau2"])
    tmp_output.append(exp1-exp2)
    tmp_output.append( params["A"] * ratio1 / params["tau1beta1"] * exp1 ) # tau1beta1
    tmp_output.append( -params["A"]*np.log(xarray) * ratio1 * exp1 ) # beta1 
    tmp_output.append( params["A"]*xarray/params["tau2"]**2 * exp2 ) # tau2

    output = []
    if np.all(switch != None):
        for i, tf in enumerate(switch):
            if tf:
                output.append(tmp_output[i])
                if i == 2 and np.isnan(tmp_output[i][0]):
                    raise ValueError("Dfun in scipy.optimize.leastsq cannot handle NaN values, please exclude t=0 from the fit or don't analytically calculate the Jacobian.")
    else:
        output = tmp_output
        if np.isnan(tmp_output[2][0]):
            raise ValueError("Dfun in scipy.optimize.leastsq cannot handle NaN values, please exclude t=0 from the fit or don't analytically calculate the Jacobian.")

    return np.transpose(np.array(output))


def reg_n_two_stretched_exponential_decays(xdata, ydata, minimizer="leastsq", kwargs_minimizer={}, kwargs_parameters={}, verbose=False, weighting=None):
    """
    Provided data fit to a regular and two stretched exponential decays

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
    weighting : numpy.ndarray, Optional, default=None
        Of the same length as the provided data, contains the weights for each data point.

        - ``"A1" = {"value": 0.8, "min": 0, "max":1}``
        - ``"tau1beta1" = {"value": 0.5, "min": np.finfo(float).eps, "max":1e+4}``
        - ``"beta1" = {"value": 1/2, "min": np.finfo(float).eps, "max":5}``
        - ``"A2" = {"value": 0.8, "min": 0, "max":1}``
        - ``"tau2beta2" = {"value": 0.5, "min": np.finfo(float).eps, "max":1e+4}``
        - ``"beta2" = {"value": 2, "min": np.finfo(float).eps, "max":5}``
        - ``"tau3" = {"value": 0.5, "min": np.finfo(float).eps, "max":1e+2}``

    verbose : bool, Optional, default=False
        Output fitting statistics

    Returns
    -------
    parameters : numpy.ndarray
        Array containing parameters: ['A', 'tau1beta1', 'beta1', 'tau2beta2', 'beta2', 'tau3']
    stnd_errors : numpy.ndarray
        Array containing parameter standard errors: ['A', 'tau1beta1', 'beta1', 'tau2beta2', 'beta2', 'tau3']
        
    """

    kwargs_min = copy.deepcopy(kwargs_minimizer)

    if np.all(weighting != None):
        if minimizer == "emcee":
            kwargs_min["is_weighted"] = True
        weighting = weighting[ydata>0]
        if np.all(np.isnan(weighting[1:])):
            weighting = None
    elif minimizer == "emcee":
        kwargs_min["is_weighted"] = False
    kwargs_min.update({"nan_policy": "omit"})

    xarray = xdata[ydata>0]
    yarray = ydata[ydata>0]
    if np.all(np.isnan(ydata[1:])):
        raise ValueError("y-axis data is NaN")

    param_kwargs = {
        "A1": {"value": 0.8, "min": 0, "max":1},
        "tau1beta1": {"value": 0.5, "min": np.finfo(float).eps, "max":1e+4},
        "beta1": {"value": 1/2, "min": np.finfo(float).eps, "max":5},
        "A2": {"value": 0.8, "min": 0, "max":1},
        "tau2beta2": {"value": 0.5, "min": np.finfo(float).eps, "max":1e+4},
        "beta2": {"value": 2, "min": np.finfo(float).eps, "max":5},
        "tau3": {"value": 0.5, "min": np.finfo(float).eps, "max":1e+2},
    }
    for key, value in kwargs_parameters.items():
        if key in param_kwargs:
            param_kwargs[key].update(value)
        else:
            raise ValueError("The parameter, {}, was given to custom_fit.exponential_decay, which requires parameters: {}".format(key, ", ".join(list(param_kwargs.keys()))))

    exp = Parameters()
    switch = [True for x in range(len(param_kwargs))]
    for i, (key, value) in enumerate(param_kwargs.items()):
        exp.add(key, **value)
        if "vary" in value and value["vary"] == False:
            switch[i] = False
        if "expr" in value:
            switch[i] = False

    if minimizer in ["leastsq"]:
        kwargs_min["Dfun"] = _d_reg_n_two_stretched_exponential_decays
    Result2 = lmfit.minimize(
        _res_reg_n_two_stretched_exponential_decays,
        exp,
        method=minimizer,
        args=(xarray, yarray),
        kws={"switch": switch,
        "weighting": weighting},
        **kwargs_min
    )

    # Format output
    lp = len(param_kwargs)
    output = np.zeros(lp)
    uncertainties = np.zeros(lp)
    for i,(param, value) in enumerate(Result2.params.items()):
        output[i] = value.value
        uncertainties[i] = value.stderr

    if verbose:
        if minimizer == "leastsq":
            print("Termination: {}".format(Result2.lmdif_message))
        elif minimizer == "emcee":
            print("Termination:  Success? {}, Aborted? {}, Chi^2: {}".format(Result2.success, Result2.aborted, Result2.chisqr))
        else:
            print("Termination: {}".format(Result2.message))
        lmfit.printfuncs.report_fit(Result2.params, min_correl=0.5)

    return output, uncertainties

def _res_reg_n_two_stretched_exponential_decays(params, xarray, yarray, switch=None, weighting=None,):

    out =  (params["A1"]*np.exp(-xarray**params["beta1"]/params["tau1beta1"]) 
            + params["A2"]*np.exp(-xarray**params["beta2"]/params["tau2beta2"])
            + (1-params["A1"]-params["A2"])*np.exp(-(xarray/params["tau3"]))
           ) - yarray
    if np.all(weighting != None):
        if len(weighting) != len(out):
            raise ValueError("Length of `weighting` array must be of equal length to input data arrays")
        out = out*np.array(weighting)

    return out

def _d_reg_n_two_stretched_exponential_decays(params, xarray, yarray, switch=None, weighting=None,):

    tmp_output = []
    ratio1 = xarray**params["beta1"] / params["tau1beta1"]
    ratio2 = xarray**params["beta2"] / params["tau2beta2"]
    exp1 = np.exp(-ratio1)
    exp2 = np.exp(-ratio2)
    exp3 = np.exp(-xarray/params["tau3"])

    tmp_output.append(exp1-exp3) # A1
    tmp_output.append( params["A1"] * ratio1 / params["tau1beta1"] * exp1 ) # tau1beta1
    tmp_output.append( -params["A1"]*np.log(xarray) * ratio1 * exp1 ) # beta1 
    tmp_output.append(exp2-exp3) # A2
    tmp_output.append( params["A2"] * ratio2 / params["tau2beta2"] * exp2 ) # tau2beta2
    tmp_output.append( -params["A2"]*np.log(xarray) * ratio2 * exp2 ) # beta2 
    tmp_output.append( (1-params["A1"]-params["A2"])*xarray/params["tau3"]**2 * exp3 ) # tau2

    output = []
    if np.all(switch != None):
        for i, tf in enumerate(switch):
            if tf:
                output.append(tmp_output[i])
                if i == 2 and np.isnan(tmp_output[i][0]):
                    raise ValueError("Dfun in scipy.optimize.leastsq cannot handle NaN values, please exclude t=0 from the fit or don't analytically calculate the Jacobian.")
    else:
        output = tmp_output
        if np.isnan(tmp_output[2][0]):
            raise ValueError("Dfun in scipy.optimize.leastsq cannot handle NaN values, please exclude t=0 from the fit or don't analytically calculate the Jacobian.")

    return np.transpose(np.array(output))


def three_stretched_exponential_decays(xdata, ydata, minimizer="leastsq", kwargs_minimizer={}, kwargs_parameters={}, verbose=False, weighting=None):
    """
    Provided data fit to three stretched exponential decays

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
    weighting : numpy.ndarray, Optional, default=None
        Of the same length as the provided data, contains the weights for each data point.

        - ``"A1" = {"value": 0.8, "min": 0, "max":1}``
        - ``"tau1beta1" = {"value": 0.5, "min": np.finfo(float).eps, "max":1e+4}``
        - ``"beta1" = {"value": 1/2, "min": np.finfo(float).eps, "max":5}``
        - ``"A2" = {"value": 0.8, "min": 0, "max":1}``
        - ``"tau2beta2" = {"value": 0.5, "min": np.finfo(float).eps, "max":1e+4}``
        - ``"beta2" = {"value": 2, "min": np.finfo(float).eps, "max":5}``
        - ``"tau3beta3" = {"value": 0.5, "min": np.finfo(float).eps, "max":1e+4}``
        - ``"beta3" = {"value": 2, "min": np.finfo(float).eps, "max":5}``

    verbose : bool, Optional, default=False
        Output fitting statistics

    Returns
    -------
    parameters : numpy.ndarray
        Array containing parameters: ['A', 'tau1beta1', 'beta1', 'tau2beta2', 'beta2', 'tau2beta3', 'beta3',]
    stnd_errors : numpy.ndarray
        Array containing parameter standard errors: ['A', 'tau1beta1', 'beta1', 'tau2beta2', 'beta2', 'tau3beta3', 'beta3',]
        
    """

    kwargs_min = copy.deepcopy(kwargs_minimizer)

    if np.all(weighting != None):
        if minimizer == "emcee":
            kwargs_min["is_weighted"] = True
        weighting = weighting[ydata>0]
        if np.all(np.isnan(weighting[1:])):
            weighting = None
    elif minimizer == "emcee":
        kwargs_min["is_weighted"] = False
    kwargs_min.update({"nan_policy": "omit"})

    xarray = xdata[ydata>0]
    yarray = ydata[ydata>0]
    if np.all(np.isnan(ydata[1:])):
        raise ValueError("y-axis data is NaN")

    param_kwargs = {
        "A1": {"value": 0.8, "min": 0, "max":1},
        "tau1beta1": {"value": 0.5, "min": np.finfo(float).eps, "max":1e+4},
        "beta1": {"value": 1/2, "min": np.finfo(float).eps, "max":5},
        "A2": {"value": 0.8, "min": 0, "max":1},
        "tau2beta2": {"value": 0.5, "min": np.finfo(float).eps, "max":1e+4},
        "beta2": {"value": 2, "min": np.finfo(float).eps, "max":5},
        "tau3beta3": {"value": 0.5, "min": np.finfo(float).eps, "max":1e+4},
        "beta3": {"value": 2, "min": np.finfo(float).eps, "max":5},
    }
    for key, value in kwargs_parameters.items():
        if key in param_kwargs:
            param_kwargs[key].update(value)
        else:
            raise ValueError("The parameter, {}, was given to custom_fit.exponential_decay, which requires parameters: {}".format(key, ", ".join(list(param_kwargs.keys()))))

    exp = Parameters()
    switch = [True for x in range(len(param_kwargs))]
    for i, (key, value) in enumerate(param_kwargs.items()):
        exp.add(key, **value)
        if "vary" in value and value["vary"] == False:
            switch[i] = False
        if "expr" in value:
            switch[i] = False

    if minimizer in ["leastsq"]:
        kwargs_min["Dfun"] = _d_three_stretched_exponential_decays
    Result2 = lmfit.minimize(
        _res_three_stretched_exponential_decays,
        exp,
        method=minimizer,
        args=(xarray, yarray),
        kws={"switch": switch,
        "weighting": weighting},
        **kwargs_min
    )

    # Format output
    print(list(Result2.params.keys()))
    lp = len(param_kwargs)
    output = np.zeros(lp)
    uncertainties = np.zeros(lp)
    for i,(param, value) in enumerate(Result2.params.items()):
        output[i] = value.value
        uncertainties[i] = value.stderr

    if verbose:
        if minimizer == "leastsq":
            print("Termination: {}".format(Result2.lmdif_message))
        elif minimizer == "emcee":
            print("Termination:  Success? {}, Aborted? {}, Chi^2: {}".format(Result2.success, Result2.aborted, Result2.chisqr))
        else:
            print("Termination: {}".format(Result2.message))
        lmfit.printfuncs.report_fit(Result2.params, min_correl=0.5)

    return output, uncertainties

def _res_three_stretched_exponential_decays(params, xarray, yarray, switch=None, weighting=None,):

    out =  (params["A1"]*np.exp(-xarray**params["beta1"]/params["tau1beta1"])
            + params["A2"]*np.exp(-xarray**params["beta2"]/params["tau2beta2"])
            + (1-params["A1"]-params["A2"])*np.exp(-xarray**params["beta3"]/params["tau3beta3"])
           ) - yarray
    if np.all(weighting != None):
        if len(weighting) != len(out):
            raise ValueError("Length of `weighting` array must be of equal length to input data arrays")
        out = out*np.array(weighting)

    return out

def _d_three_stretched_exponential_decays(params, xarray, yarray, switch=None, weighting=None,):

    tmp_output = []
    ratio1 = xarray**params["beta1"] / params["tau1beta1"]
    ratio2 = xarray**params["beta2"] / params["tau2beta2"]
    ratio3 = xarray**params["beta3"] / params["tau3beta3"]
    exp1 = np.exp(-ratio1)
    exp2 = np.exp(-ratio2)
    exp3 = np.exp(-ratio3)

    tmp_output.append(exp1-exp3) # A1
    tmp_output.append( params["A1"] * ratio1 / params["tau1beta1"] * exp1 ) # tau1beta1
    tmp_output.append( -params["A1"]*np.log(xarray) * ratio1 * exp1 ) # beta1 
    tmp_output.append(exp2-exp3) # A2
    tmp_output.append( params["A2"] * ratio2 / params["tau2beta2"] * exp2 ) # tau2beta2
    tmp_output.append( -params["A2"]*np.log(xarray) * ratio2 * exp2 ) # beta2 
    tmp_output.append( (1-params["A1"]-params["A2"]) * ratio3 / params["tau3beta3"] * exp3 ) # tau3beta3
    tmp_output.append( -params["A3"]*np.log(xarray) * ratio3 * exp3 ) # beta3 

    output = []
    if np.all(switch != None):
        for i, tf in enumerate(switch):
            if tf:
                output.append(tmp_output[i])
                if i == 2 and np.isnan(tmp_output[i][0]):
                    raise ValueError("Dfun in scipy.optimize.leastsq cannot handle NaN values, please exclude t=0 from the fit or don't analytically calculate the Jacobian.")
    else:
        output = tmp_output
        if np.isnan(tmp_output[2][0]):
            raise ValueError("Dfun in scipy.optimize.leastsq cannot handle NaN values, please exclude t=0 from the fit or don't analytically calculate the Jacobian.")

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

    kwargs_min = copy.deepcopy(kwargs_minimizer)

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
        kwargs_min["Dfun"] = _d_n_gaussians
    elif minimizer in ["trust-exact"]:
        kwargs_min["jac"] = _d_n_gaussians
    Result1 = lmfit.minimize(_res_n_gaussians, gaussian, method=minimizer, args=(xarray, yarray, num), **kwargs_min)

    # Format output
    output = np.zeros(3*num)
    uncertainties = np.zeros(3*num)
    for i,(param, value) in enumerate(Result1.params.items()):
        output[i] = value.value
        uncertainties[i] = value.stderr

    if verbose:
        if minimizer == "leastsq":
            print("Termination: {}".format(Result1.lmdif_message))
        elif minimizer == "emcee":
            print("Termination:  Success? {}, Aborted? {}, Chi^2: {}".format(Result1.success, Result1.aborted, Result1.chisqr))
        else:
            print("Termination: {}".format(Result1.message))
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


def cumulative_exponential(xdata, ydata, minimizer="leastsq", weighting=None, kwargs_minimizer={}, kwargs_parameters={}, verbose=False):
    """
    Fit data to a cumulative exponential: ``f(x)=A*(1-np.exp(-x/lc)) + C``

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
        - ``"lc": {"value": np.max(xarray), "min": np.finfo(float).eps, "max":1e+4}``
        - ``"C": {"value": 0.0, "min": -1e+4, "max": 1e+4}``

    verbose : bool, Optional, default=False
        Output fitting statistics

    Returns
    -------
    parameters : numpy.ndarray
        Array containing parameters: ['A', 'lc', 'C']
    stnd_errors : numpy.ndarray
        Array containing parameter standard errors: ['A', 'lc', 'C']
        
    """

    kwargs_min = copy.deepcopy(kwargs_minimizer)

    if np.all(weighting != None):
        if minimizer == "emcee":
            kwargs_min["is_weighted"] = True
        weighting = weighting[ydata>0]
        if np.all(np.isnan(weighting[1:])):
            weighting = None
    elif minimizer == "emcee":
        kwargs_min["is_weighted"] = False
    kwargs_min.update({"nan_policy": "omit"})

    xarray = xdata[ydata>0]
    yarray = ydata[ydata>0]
    if np.all(np.isnan(ydata[1:])):
        raise ValueError("y-axis data is NaN")

    if not isinstance(kwargs_parameters, dict):
        raise ValueError("kwargs_parameters must be a dictionary")

    tmp_tau = xarray[np.where(np.abs(yarray-np.nanmax(yarray))>0.80*np.nanmax(yarray))[0]]
    if len(tmp_tau) > 0:
        tmp_tau = tmp_tau[0]
    else:
        tmp_tau = xarray[int(len(xarray)/2)]
    param_kwargs = {
                    "A": {"value": np.nanmax(yarray), "min": 0, "max": 1e+4},
                    "lc": {"value": tmp_tau, "min": np.finfo(float).eps, "max":1e+4},
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

    cum_exp = Parameters()
    for key, value in param_kwargs.items():
        cum_exp.add(key, **value)

    if minimizer in ["leastsq"]:
        kwargs_min["Dfun"] = _d_cumulative_exponential
    elif minimizer in ["trust-exact"]:
        kwargs_min["jac"] = _d_cumulative_exponential
    Result1 = lmfit.minimize(_res_cumulative_exponential, cum_exp, method=minimizer, args=(xarray, yarray), kws={"weighting": weighting, "switch": switch}, **kwargs_min)

    # Format output
    output = np.zeros(len(param_kwargs))
    uncertainties = np.zeros(len(param_kwargs))
    for i,(param, value) in enumerate(Result1.params.items()):
        output[i] = value.value
        uncertainties[i] = value.stderr
    output[1] = output[1]**(1/output[2])
    tmp_output = 1/output[2]
    tmp_uncert = uncertainties[2]/output[2]**2
    uncertainties[1] = np.sqrt(output[1]**(2*output[2])*( (uncertainties[1]*tmp_output/output[1])**2 * (np.log(output[1])*tmp_uncert)**2 ))

    if verbose:
        if minimizer == "leastsq":
            print("Termination: {}".format(Result1.lmdif_message))
        elif minimizer == "emcee":
            print("Termination:  Success? {}, Aborted? {}, Chi^2: {}".format(Result1.success, Result1.aborted, Result1.chisqr))
        else:
            print("Termination: {}".format(Result1.message))
        lmfit.printfuncs.report_fit(Result1.params)

    return output, uncertainties

def _res_cumulative_exponential(params, xarray, yarray, weighting=None, switch=None):

    out = params["A"]*(1-np.exp(-xarray/params["lc"])) + params["C"] - yarray

#    print(params["A"].value, params["lc_beta"].value, params["beta"].value, params["C"].value)
    if np.all(weighting != None):
        if len(weighting) != len(out):
            raise ValueError("Length of `weighting` array must be of equal length to input data arrays")
        out = out*np.array(weighting)

    return out

def _d_cumulative_exponential(params, xarray, yarray, weighting=None, switch=None):

    out = np.zeros((len(xarray),4))
    tmp_xscale = xarray/params["lc"]
    tmp_exp = np.exp(-tmp_xscale)

    out[:,0] = (1-tmp_exp)
    out[:,1] = -params["A"]*xarray*tmp_exp/params["lc"]**2
    out[:,2] = np.ones(len(xarray))

    output = []
    if np.all(switch != None):
        for i, tf in enumerate(switch):
            if tf:
                output.append(out[:,i])
        out = np.transpose(np.array(output))


    return out


def stretched_cumulative_exponential(xdata, ydata, minimizer="leastsq", weighting=None, kwargs_minimizer={}, kwargs_parameters={}, verbose=False):
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

    kwargs_min = copy.deepcopy(kwargs_minimizer)
    
    if np.all(weighting != None):
        if minimizer == "emcee":
            kwargs_min["is_weighted"] = True
        weighting = weighting[ydata>0]
        if np.all(np.isnan(weighting[1:])):
            weighting = None
    elif minimizer == "emcee":
        kwargs_min["is_weighted"] = False
    kwargs_min.update({"nan_policy": "omit"})

    xarray = xdata[ydata>0]
    yarray = ydata[ydata>0]
    if np.all(np.isnan(ydata[1:])):
        raise ValueError("y-axis data is NaN")

    if not isinstance(kwargs_parameters, dict):
        raise ValueError("kwargs_parameters must be a dictionary")

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
        kwargs_min["Dfun"] = _d_stretched_cumulative_exponential
    elif minimizer in ["trust-exact"]:
        kwargs_min["jac"] = _d_stretched_cumulative_exponential
    Result1 = lmfit.minimize(_res_stretched_cumulative_exponential, stretched_exp, method=minimizer, args=(xarray, yarray), kws={"weighting": weighting, "switch": switch}, **kwargs_min)

    # Format output
    output = np.zeros(len(param_kwargs))
    uncertainties = np.zeros(len(param_kwargs))
    for i,(param, value) in enumerate(Result1.params.items()):
        output[i] = value.value
        uncertainties[i] = value.stderr
    output[1] = output[1]**(1/output[2])
    tmp_output = 1/output[2]
    tmp_uncert = uncertainties[2]/output[2]**2
    uncertainties[1] = np.sqrt(output[1]**(2*output[2])*( (uncertainties[1]*tmp_output/output[1])**2 * (np.log(output[1])*tmp_uncert)**2 ))

    if verbose:
        if minimizer == "leastsq":
            print("Termination: {}".format(Result1.lmdif_message))
        elif minimizer == "emcee":
            print("Termination:  Success? {}, Aborted? {}, Chi^2: {}".format(Result1.success, Result1.aborted, Result1.chisqr))
        else:
            print("Termination: {}".format(Result1.message))
        lmfit.printfuncs.report_fit(Result1.params)

    return output, uncertainties

def _res_stretched_cumulative_exponential(params, xarray, yarray, weighting=None, switch=None):

    out = params["A"]*(1-np.exp(-(xarray)**params["beta"]/params["lc_beta"])) + params["C"] - yarray

#    print(params["A"].value, params["lc_beta"].value, params["beta"].value, params["C"].value)
    if np.all(weighting != None):
        if len(weighting) != len(out):
            raise ValueError("Length of `weighting` array must be of equal length to input data arrays")
        out = out*np.array(weighting)

    return out

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


def double_cumulative_exponential(xdata, ydata, minimizer="leastsq", verbose=False, weighting=None, kwargs_minimizer={}, kwargs_parameters={}, include_C=False):
    r"""
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
        - ``"tau1": {"value": xarray[yarray==max(yarray)][0], "min": np.finfo(float).eps, "max":1e+4}``
        - ``"tau2": {"value": xarray[yarray==max(yarray)][0]/2, "min": np.finfo(float).eps, "max":1e+4}``
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

    kwargs_min = copy.deepcopy(kwargs_minimizer)

    if np.all(np.isnan(ydata[1:])):
        raise ValueError("y-axis data is NaN")

    if np.all(weighting != None):
        weighting = weighting[ydata>0]
        if minimizer == "emcee":
            kwargs_min["is_weighted"] = True
        weighting = weighting[ydata>0]
    elif minimizer == "emcee":
        kwargs_min["is_weighted"] = False

    xarray = xdata[ydata>0]
    yarray = ydata[ydata>0]

    tmp_tau = xarray[np.where(np.abs(yarray-np.nanmax(yarray))>0.80*np.nanmax(yarray))[0]]
    if len(tmp_tau) > 0:
        tmp_tau = tmp_tau[0]
    else:
        tmp_tau = xarray[int(len(xarray)/2)]
    param_kwargs = {
                    "A": {"min": 0.0, "max": np.nanmax(yarray)*100, "value": np.nanmax(yarray)},
                    "alpha": {"min":0, "max":1.0, "value":0.1},
                    "tau1": {"min":np.finfo(float).eps, "max":np.max(xarray)*10, "value": tmp_tau},
                    "tau2": {"min":np.finfo(float).eps, "max":np.max(xarray)*10, "value": tmp_tau/2},
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

    result = lmfit.minimize(_res_double_cumulative_exponential, exp1, method=minimizer, args=(xarray, yarray), kws={"weighting": weighting}, **kwargs_min)

    # Format output
    lx = len(result.params)
    output = np.zeros(lx)
    uncertainties = np.zeros(lx)
    for i,(param, value) in enumerate(result.params.items()):
        output[i] = value.value
        uncertainties[i] = value.stderr

    if verbose:
        if minimizer == "leastsq":
            print("Termination: {}".format(result.lmdif_message))
        elif minimizer == "emcee":
            print("Termination:  Success? {}, Aborted? {}, Chi^2: {}".format(result.success, result.aborted, result.chisqr))
        else:
            print("Termination: {}".format(Result1.message))
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


def double_viscosity_cumulative_exponential(xdata, ydata, minimizer="leastsq", verbose=False, weighting=None, kwargs_minimizer={}, kwargs_parameters={}):
    r"""
    Provided data fit to:
    ..math:`y= A*\{alpha}*\tau_{1}*(1-exp(-(x/\tau_{1}))) + A*(1-\{alpha})*\tau_{2}*(1-exp(-(x/\tau_{2})))` 

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
        - ``"tau1": {"value": xarray[yarray==max(yarray)][0], "min": np.finfo(float).eps, "max":1e+4}``
        - ``"tau2": {"value": xarray[yarray==max(yarray)][0]/2, "min": np.finfo(float).eps, "max":1e+4}``

    Returns
    -------
    parameters : numpy.ndarray
        Array containing parameters: ["A", "alpha", "tau1", "tau2"]
    stnd_errors : numpy.ndarray
        Array containing parameter standard errors: ["A", "alpha", "tau1", "tau2"]
        
    """

    kwargs_min = copy.deepcopy(kwargs_minimizer)
    
    if np.all(weighting != None):
        if minimizer == "emcee":
            kwargs_min["is_weighted"] = True
        weighting = weighting[ydata>0]
        if np.all(np.isnan(weighting[1:])):
            weighting = None
    elif minimizer == "emcee":
        kwargs_min["is_weighted"] = False
    kwargs_min.update({"nan_policy": "omit"})

    xarray = xdata[ydata>0]
    yarray = ydata[ydata>0]
    if np.all(np.isnan(ydata[1:])):
        raise ValueError("y-axis data is NaN")

    tmp_tau = xarray[np.where(np.abs(yarray-np.nanmax(yarray))>0.80*np.nanmax(yarray))[0]]
    if len(tmp_tau) > 0:
        tmp_tau = tmp_tau[0]
    else:
        tmp_tau = xarray[int(len(xarray)/2)]
    param_kwargs = {
                    "A": {"min": 0.0, "max": np.nanmax(yarray)*100, "value": np.nanmax(yarray)},
                    "alpha": {"min":0, "max":1.0, "value":0.1},
                    "tau1": {"min":np.finfo(float).eps, "max":np.max(xarray)*10, "value": tmp_tau},
                    "tau2": {"min":np.finfo(float).eps, "max":np.max(xarray)*10, "value": tmp_tau/2},
                   }

    exp1 = Parameters()
    for param, kwargs in param_kwargs.items():
        if param in kwargs_parameters:
            kwargs.update(kwargs_parameters[param])
            del kwargs_parameters[param]
        exp1.add(param, **kwargs)
    if len(kwargs_parameters) != 0:
        raise ValueError("The parameter(s), {}, was/were given to custom_fit.double_cumulative_exponential, which requires parameters: 'A', 'alpha', 'tau1', 'tau2'".format(list(kwargs_parameters.keys())))

    if minimizer in ["leastsq"]:
        kwargs_min["Dfun"] = _d_double_viscosity_cumulative_exponential
    elif minimizer in ["trust-exact"]:
        kwargs_min["jac"] = _d_double_viscosity_cumulative_exponential
    result = lmfit.minimize(_res_double_viscosity_cumulative_exponential, exp1, method=minimizer, args=(xarray, yarray), kws={"weighting": weighting}, **kwargs_min)

    # Format output
    lx = len(result.params)
    output = np.zeros(lx)
    uncertainties = np.zeros(lx)
    for i,(param, value) in enumerate(result.params.items()):
        output[i] = value.value
        uncertainties[i] = value.stderr

    if verbose:
        if minimizer == "leastsq":
            print("Termination: {}".format(result.lmdif_message))
        elif minimizer == "emcee":
            print("Termination:  Success? {}, Aborted? {}, Chi^2: {}".format(result.success, result.aborted, result.chisqr))
        else:
            print("Termination: {}".format(result.message))
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

def _d_double_viscosity_cumulative_exponential(params, xarray, yarray, weighting=None, switch=None):

    out = np.zeros((len(xarray),4))
    tmp_exp1 = np.exp(-xarray/params["tau1"])
    tmp_exp2 = np.exp(-xarray/params["tau2"])

    out[:,0] = params["tau1"]*params["alpha"]*(1-tmp_exp1) + params["tau2"]*(1-params["alpha"])*(1-tmp_exp2) # dA
    out[:,1] = params["A"]*(params["tau1"]*(1-tmp_exp1) - params["tau2"]*(1-tmp_exp2)) #dalpha
    out[:,2] = params["A"]*params["alpha"]*(1-((params["tau1"]+xarray)/params["tau1"])*tmp_exp1) # dtau1
    out[:,3] = params["A"]*(1-params["alpha"])*(1-((params["tau2"]+xarray)/params["tau2"])*tmp_exp2) # dtau2

    output = []
    if np.all(switch != None):
        for i, tf in enumerate(switch):
            if tf:
                output.append(out[:,i])
        out = np.transpose(np.array(output))


    return out



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
        Array containing parameters: ["A", "b"]
    stnd_errors : numpy.ndarray
        Array containing parameter standard errors: ["A", "b"]
        
    """

    kwargs_min = copy.deepcopy(kwargs_minimizer)
    
    if np.all(weighting != None):
        if minimizer == "emcee":
            kwargs_min["is_weighted"] = True
        weighting = weighting[ydata>0]
        if np.all(np.isnan(weighting[1:])):
            weighting = None
    elif minimizer == "emcee":
        kwargs_min["is_weighted"] = False
    kwargs_min.update({"nan_policy": "omit"})

    xarray = xdata[ydata>0]
    yarray = ydata[ydata>0]
    if np.all(np.isnan(ydata[1:])):
        raise ValueError("y-axis data is NaN")
    
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
    result = lmfit.minimize(power_law, exp1, method=minimizer, **kwargs_min)

    # Format output
    output = np.zeros(2)
    uncertainties = np.zeros(2)
    for i,(param, value) in enumerate(result.params.items()):
        output[i] = value.value
        uncertainties[i] = value.stderr

    if verbose:
        if minimizer == "leastsq":
            print("Termination: {}".format(result.lmdif_message))
        elif minimizer == "emcee":
            print("Termination:  Success? {}, Aborted? {}, Chi^2: {}".format(result.success, result.aborted, result.chisqr))
        else:
            print("Termination: {}".format(result.message))
        lmfit.printfuncs.report_fit(result.params)

    return output, uncertainties


def gamma_distribution(xdata, ydata, minimizer="leastsq", weighting=None, kwargs_minimizer={}, kwargs_parameters={}, verbose=False,):
    """
    Provided data fit to: ..math:`A*x^{b}` after linearizing with a log transform.

    Values of zero and NaN are ignored in the fit.

    Parameters
    ----------
    xdata : numpy.ndarray
        independent data set
    ydata : numpy.ndarray
        dependent data set
    minimizer : str, Optional, default="leastsq"
        Fitting method supported by ``lmfit.minimize``
    weighting : numpy.ndarray, Optional, default=None
        Of the same length as the provided data, contains the weights for each data point.
    kwargs_minimizer : dict, Optional, default={}
        Keyword arguments for ``lmfit.minimizer()``
    kwargs_parameters : dict, Optional
        Dictionary containing the following variables and their default keyword arguments in the form ``kwargs_parameters = {"var": {"kwarg1": var1...}}`` where ``kwargs1...`` are those from lmfit.Parameters.add() and ``var`` is one of the following parameter names.

        - ``"alpha" = {"value": 0.1, "min": 0, "max":100}``
        - ``"beta" = {"value": 1.0, "min": 0, "max":100}``

    verbose : bool, Optional, default=False
        Output fitting statistics

    Returns
    -------
    parameters : numpy.ndarray
        Array containing parameters: ["alpha", "beta"]
    stnd_errors : numpy.ndarray
        Array containing parameter standard errors: ["alpha", "beta"]
        
    """

    kwargs_min = copy.deepcopy(kwargs_minimizer)
    
    if np.all(weighting != None):
        if minimizer == "emcee":
            kwargs_min["is_weighted"] = True
        weighting = weighting[ydata>0]
        if np.all(np.isnan(weighting[1:])):
            weighting = None
    elif minimizer == "emcee":
        kwargs_min["is_weighted"] = False
    kwargs_min.update({"nan_policy": "omit"})

    xarray = xdata[ydata>0]
    yarray = ydata[ydata>0]
    if np.all(np.isnan(ydata[1:])):
        raise ValueError("y-axis data is NaN")

    param_kwargs = {
                    "alpha": {"value": 0.1, "min": np.finfo(float).eps, "max":1e+2},
                    "beta": {"value": 1.0, "min": np.finfo(float).eps, "max":1e+2},
                   }
    for key, value in kwargs_parameters.items():
        if key in param_kwargs:
            param_kwargs[key].update(value)
        else:
            raise ValueError("The parameter, {}, was given to custom_fit.gamma_distribution, which requires parameters: 'alpha' and 'beta'".format(key))

    switch = [True for x in range(len(param_kwargs))]
    for i, (key, value) in enumerate(param_kwargs.items()):
        if "vary" in value and value["vary"] == False:
            switch[i] = False
        if "expr" in value:
            switch[i] = False

    exp1 = Parameters()
    exp1.add("alpha", **param_kwargs["alpha"])
    exp1.add("beta", **param_kwargs["beta"])
    if minimizer in ["leastsq"]:
        kwargs_min["Dfun"] = _d_gamma_distribution
    Result1 = lmfit.minimize(_res_gamma_distribution, exp1, method=minimizer, args=(xarray, yarray), kws={"switch": switch, "weighting": weighting,}, **kwargs_min)

    # Format output
    output = np.zeros(len(param_kwargs))
    uncertainties = np.zeros(len(param_kwargs))
    for i,(param, value) in enumerate(Result1.params.items()):
        output[i] = value.value
        uncertainties[i] = value.stderr

    if verbose:
        if minimizer == "leastsq":
            print("Termination: {}".format(Result1.lmdif_message))
        elif minimizer == "emcee":
            print("Termination:  Success? {}, Aborted? {}, Chi^2: {}".format(Result1.success, Result1.aborted, Result1.chisqr))
        else:
            print("Termination: {}".format(Result1.message))
        lmfit.printfuncs.report_fit(Result1.params)

    return output, uncertainties, 1 - Result1.residual.var() / np.var(yarray)

def _res_gamma_distribution(params, xarray, yarray, switch=None, weighting=None,):
    term1 = params["beta"]**params["alpha"] / sps.gamma(params["alpha"])
    term2 = xarray**(params["alpha"] - 1.0)
    term3 = np.exp(-params["beta"]*xarray)
    out = term1 * term2 * term3 - yarray
    if np.all(weighting != None):
        if len(weighting) != len(out):
            raise ValueError("Length of `weighting` array must be of equal length to input data arrays")
        out = out*np.array(weighting)
    return out

def _d_gamma_distribution(params, xarray, yarray, switch=None, weighting=None,):
    term1 = params["beta"]**params["alpha"] / sps.gamma(params["alpha"])
    term2 = xarray**(params["alpha"] - 1.0)
    term3 = np.exp(-params["beta"]*xarray)

    tmp_output = []
    tmp_output.append(term1 * term2 * term3 * (-sps.digamma(params["alpha"]) + np.log(params["beta"]) + np.log(xarray))) # alpha
    tmp_output.append(term1/params["beta"] * term2 * term3 * (params["alpha"] - params["beta"] * xarray)) # beta

    output = []
    if np.all(switch != None):
        for i, tf in enumerate(switch):
            if tf:
                output.append(tmp_output[i])
    else:
        output = tmp_output

    return np.transpose(np.array(output))

