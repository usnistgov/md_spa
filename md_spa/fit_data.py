
import numpy as np
import matplotlib.pyplot as plt

import misc_modules.custom_plotting as cplot
import md_spa.custom_fit as cfit

def extract_gaussians(xdata, ydata, n_gaussians=None, normalize=False, kwargs_peaks={}, kwargs_fit={}, show_plot=False, save_plot=False, plot_name="n_gaussians.png"):
    """
    Fit parameters to a set of gaussian curves using ``custom_fit.n_gaussians``. If the number to fit to is not provided, then the number of gaussians is set to the number of maxima in the data set.

    Parameters
    ----------
    xdata : numpy.ndarray
        Independent dataset
    ydata : numpy.ndarray
        Dependent dataset
    n_gaussians : int, Optional, default=None
        The number of gaussian functions to fit to. If None the number of peaks is extracted and used as an initial guess.
    normalize : bool, Optional, default=False
        If True, the prefactors are scaled by the integral of the set of Gaussian functions from negative infinity to infinity. 
    kwargs_peaks : dict, Optional, default={}
        Keyword arguments used in ``custom_plotting.pull_extrema``
    kwargs_fit : dict, Optional, default={}
        Keyword arguments for ``custom_fit.n_gaussians``
    show_plot : bool, Optional, default=False
        Show comparison plot of each rdf being analyzed. Note that is option will interrupt the process until the plot is closed.
    save_fit : bool, Optional, default=False
        Save comparison plot of each rdf being analyzed. With the name ``plot_name``
    plot_name : str, Optional, default="n_gaussians.png"
        If `save_fit` is true, the generated plot is saved

    Returns
    -------
    parameters : numpy.ndarray
        An array of the dimensions (n_gaussians, 3) containing the parameters to define the gaussian functions that describe the input data.
    """

    if n_gaussians == None:
        _, maxima, _ = cplot.pull_extrema(xdata, ydata, **kwargs_peaks)
        n_gaussians = len(maxima)
        min_height = np.min(maxima.T[1])
        max_height = np.max(maxima.T[1])
        for i,max_pt in enumerate(maxima):
            if "kwargs_parameters" not in kwargs_fit:
                kwargs_fit["kwargs_parameters"] = {}
            kwargs_fit["kwargs_parameters"]["b{}".format(i+1)] = {"value": max_pt[0], "max": np.max(xdata), "min": np.min(xdata), "vary": False}
            kwargs_fit["kwargs_parameters"]["a{}".format(i+1)] = {"value": max_pt[1], "max": max_height, "min": min_height}
            kwargs_fit["kwargs_parameters"]["c{}".format(i+1)] = {"value": max_pt[0]/10}
    else:
        maxima = None

    if n_gaussians == None:
        # Fit parameters to set peak locations
        parameters, uncertainty = cfit.n_gaussians(xdata, ydata, len(maxima), **kwargs_fit)
        parameters = np.reshape(parameters, (len(maxima), 3))
        # Fit parameters with all free peak locations
        for n, (a,b,c) in enumerate(parameters):
            del kwargs_fit["kwargs_parameters"]["a{}".format(i+1)]["vary"]
            kwargs_fit["kwargs_parameters"]["b{}".format(i+1)]["value"] = b
            kwargs_fit["kwargs_parameters"]["c{}".format(i+1)]["value"] = c

    n_gaussians = len(maxima)    
    parameters, uncertainty = cfit.n_gaussians(xdata, ydata, n_gaussians, **kwargs_fit)
    parameters = np.reshape(parameters, (n_gaussians, 3))
    uncertainties = np.reshape(uncertainty, (n_gaussians, 3))

    if normalize:
        integral = 0
        for n in range(n_gaussians):
            integral += parameters[n][0]*parameters[n][2]*np.sqrt(2*np.pi)
        tmp = np.transpose(parameters)
        tmp[0] /= integral
        parameters = np.transpose(tmp)
        tmp = np.transpose(uncertainties)
        tmp[0] /= integral
        uncertainties = np.transpose(tmp)
        ydata /= integral

    if show_plot or save_plot:
        plt.plot(xdata,ydata,"k",label="Data")
        xarray2 = np.linspace(xdata[0],xdata[-1],int(1e+4))
        params = {}
        for i, (a,b,c) in enumerate(parameters):
            params["a{}".format(i+1)] = a
            params["b{}".format(i+1)] = b
            params["c{}".format(i+1)] = c
        yarray2 = cfit._res_n_gaussians(params, xarray2, np.zeros(len(xarray2)), n_gaussians)
        plt.plot(xarray2,yarray2,"r",linewidth=0.5,label="Fit")
        if np.all(maxima != None):
            for i in range(len(maxima)):
                plt.plot([maxima[i][0],maxima[i][0]],[min(ydata),max(ydata)],"c",linewidth=0.5)
        plt.xlim(xdata[0],xdata[-1])
        plt.tight_layout()
        if save_plot:
            plt.savefig(plot_name.replace(" ","_"), dpi=300)
        if show_plot:
            plt.show()
        plt.close()

    return parameters, uncertainties