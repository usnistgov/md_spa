import numpy as np
import matplotlib.pyplot as plt
import os
import lmfit
from lmfit import minimize, Parameters

def exponential(filename, delimiter=",", minimizer="nelder", verbose=False, save_plots=None, show_plots=False):
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
    save_plots : str, Optional, default=None
        If not None, plots comparing the exponential fits will be saved to this filename 
    show_plots : bool, Optional, default=False
        If true, the fits will be shown

    Returns
    -------
    parameters : numpy.ndarray
        Array containing parameters: ['1: t1', '2: a1', '2: t1', '2: a2', '2: t2', '3: a1', '3: t1', '3: a2', '3: t2', '3: a3', '3: t3']
    stnd_errors : numpy.ndarray
        Array containing parameter standard errors: ['1: t1', '2: a1', '2: t1', '2: a2', '2: t2', '3: a1', '3: t1', '3: a2', '3: t2', '3: a3', '3: t3']
        
    """

    data = np.transpose(np.genfromtxt(filename,delimiter=","))
    xarray = data[0][data[1]>0]
    yarray = data[1][data[1]>0]
    
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

    if save_plots != None or show_plots:
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
        if save_plots != None:
            plt.savefig(save_plots,dpi=300)
    
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
        if save_plots != None:
            tmp_save_plot = list(os.path.split(save_plots))
            tmp_save_plot[1] = "log_"+tmp_save_plot[1]
            plt.savefig(os.path.join(*tmp_save_plot),dpi=300)

        if show_plots:
            plt.show()
        plt.close("all")

    return output, uncertainties

