import numpy as np
import matplotlib.pyplot as plt
import os
import lmfit
from lmfit import minimize, Parameters

def exponential(xdata, ydata, delimiter=",", minimizer="nelder", verbose=False, save_plot=False, show_plot=False, plot_name="exponential_fit.png"):
#def exponential(filename, delimiter=",", minimizer="nelder", verbose=False, save_plots=None, show_plot=False):
    """
    Data within a file with two columns is fit to one, two, and three exponentials, where the sum of the prefactors equals unity. 

    Values of zero and NaN are ignored in the fit.

    Parameters
    ----------
    filename : str
        Path and filename of data set to be fit
    delimiter : str, Optional, default=","
        Delimiter between columns used in ``numpy.genfromtxt()``
    minimizer : str, Optional, default="nelder"
        Fitting method supported by ``lmfit.minimize``
    verbose : bool, Optional, default=False
        Output fitting statistics
    save_plots : bool, Optional, default=False
        If not None, plots comparing the exponential fits will be saved to this filename 
    plot_name : str, Optional, default=None
        Plot filename and path
    show_plot : bool, Optional, default=False
        If true, the fits will be shown

    Returns
    -------
    parameters : numpy.ndarray
        Array containing parameters: ['1: t1', '2: a1', '2: t1', '2: a2', '2: t2', '3: a1', '3: t1', '3: a2', '3: t2', '3: a3', '3: t3']
    stnd_errors : numpy.ndarray
        Array containing parameter standard errors: ['1: t1', '2: a1', '2: t1', '2: a2', '2: t2', '3: a1', '3: t1', '3: a2', '3: t2', '3: a3', '3: t3']
        
    """

#    data = np.transpose(np.genfromtxt(filename,delimiter=","))
#    xarray = data[0][data[1]>0]
#    yarray = data[1][data[1]>0]
    xarray = xdata[ydata>0]
    yarray = ydata[ydata>0]

    def exponential_res_1(x):
        return np.exp(-xarray/x["t1"]) - yarray
    
    def exponential_res_2(x):
        return x["a1"]*np.exp(-xarray/x["t1"]) + x["a2"]*np.exp(-xarray/x["t2"]) - yarray
    
    def exponential_res_3(x):
        return x["a1"]*np.exp(-xarray/x["t1"]) + x["a2"]*np.exp(-xarray/x["t2"]) + x["a3"]*np.exp(-xarray/x["t3"]) - yarray
    
    def exponential_fit_1(x, array):
        return np.exp(-xarray/x["t1"])
    
    def exponential_fit_2(x, array):
        return x["a1"]*np.exp(-xarray/x["t1"]) + x["a2"]*np.exp(-xarray/x["t2"])
    
    def exponential_fit_3(x, array):
        return x["a1"]*np.exp(-xarray/x["t1"]) + x["a2"]*np.exp(-xarray/x["t2"]) + x["a3"]*np.exp(-xarray/x["t3"])
    
    exp1 = Parameters()
    exp1.add("t1", min=0, max=1e+4, value=0.1)
    Result1 = lmfit.minimize(exponential_res_1, exp1, method=minimizer)
    
    exp2 = Parameters()
    exp2.add("a1", min=0, max=1, value=0.8)
    exp2.add("t1", min=0, max=1e+4, value=0.1)
    exp2.add("a2", min=0, max=1, value=0.2, expr="1 - a1")
    exp2.add("t2", min=0, max=1e+4, value=0.09)
    Result2 = lmfit.minimize(exponential_res_2, exp2, method=minimizer)
    
    exp3 = Parameters()
    exp3.add("a1", min=0, max=1, value=0.8)
    exp3.add("t1", min=0, max=1e+4, value=0.1)
    exp3.add("a2", min=0, max=1, value=0.19)
    exp3.add("t2", min=0, max=1e+4, value=0.09)
    exp3.add("a3", min=0, max=1, value=0.01, expr="1 - a1 - a2")
    exp3.add("t3", min=0, max=1e+4, value=0.02)
    Result3 = lmfit.minimize(exponential_res_3, exp3, method=minimizer)

    # Format output
    output = np.zeros(11)
    uncertainties = np.zeros(11)
    output[0] = Result1.params["t1"].value
    uncertainties[0] = Result1.params["t1"].stderr
    for i,(param, value) in enumerate(Result2.params.items()):
        output[i+1] = value.value
        uncertainties[i+1] = value.stderr
    for i,(param, value) in enumerate(Result3.params.items()):
        output[i+5] = value.value
        uncertainties[i+5] = value.stderr

    if verbose:
        lmfit.printfuncs.report_fit(Result1.params)
        lmfit.printfuncs.report_fit(Result2.params, min_correl=0.5)
        print("Sum: {}".format(Result2.params["a1"]+Result2.params["a2"]))
        lmfit.printfuncs.report_fit(Result3.params, min_correl=0.5)
        print("Sum: {}".format(Result3.params["a1"]+Result3.params["a2"]+Result3.params["a3"]))

    if save_plot or show_plot:
        yfit1 = exponential_res_1(Result1.params) + yarray
        yfit2 = exponential_res_2(Result2.params) + yarray
        yfit3 = exponential_res_3(Result3.params) + yarray 
        plt.plot(xarray,yarray,".",label="Data")
        plt.plot(xarray,yfit1,label="1 Gauss",linewidth=1)
        plt.plot(xarray,yfit2,label="2 Gauss",linewidth=1)
        plt.plot(xarray,yfit3,label="3 Gauss",linewidth=1)
        plt.ylabel("Probability")
        plt.xlabel("Time")
        plt.tight_layout()
        plt.legend(loc="best")
        if save_plot:
            plt.savefig(plot_name,dpi=300)
    
        plt.figure(2)
        plt.plot(xarray,yarray,".",label="Data")
        plt.plot(xarray,yfit1,label="1 Gauss",linewidth=1)
        plt.plot(xarray,yfit2,label="2 Gauss",linewidth=1)
        plt.plot(xarray,yfit3,label="3 Gauss",linewidth=1)
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
    Data within a file with two columns is fit to an exponential function. 

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
    Data within a file with two columns is fit to two exponential functions where the prefactors sum to unity. 

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
    Data within a file with two columns is fit to two exponential functions where the prefactors sum to unity. 

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

def double_cumulative_exponential(xdata, ydata, minimizer="nelder", verbose=False, weighting=None):
    """
    Data within a file with two columns is fit to the function:
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
    weighting = weighting[ydata>0]

    def exponential_res_1(x):
        error = x["A"]*x["alpha"]*x["tau1"]*(1-np.exp(-(xarray/x["tau1"]))) \
            + x["A"]*(1-x["alpha"])*x["tau2"]*(1-np.exp(-(xarray/x["tau2"]))) - yarray
        if np.all(weighting != None):
            if len(weighting) != len(error):
                raise ValueError("Length of `weighting` array must be of equal length to input data arrays")
            error = error*np.array(weighting)
        return error

    exp1 = Parameters()
    exp1.add("A", min=0, max=1e+4, value=1.0)
    exp1.add("alpha", min=0, max=1, value=0.1)
    exp1.add("tau1", min=0, max=1e+4, value=1.0)
    exp1.add("tau2", min=0, max=1e+4, value=0.1)
    result = lmfit.minimize(exponential_res_1, exp1, method=minimizer)

    # Format output
    output = np.zeros(4)
    uncertainties = np.zeros(4)
    for i,(param, value) in enumerate(result.params.items()):
        output[i] = value.value
        uncertainties[i] = value.stderr

    if verbose:
        lmfit.printfuncs.report_fit(result.params)

    return output, uncertainties
    


