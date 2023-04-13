
import numpy as np
import matplotlib.pyplot as plt
import os
import copy
import warnings

import md_spa_utils.file_manipulation as fm
import md_spa_utils.data_manipulation as dm

from . import custom_fit as cfit

def characteristic_time(xdata, ydata, minimizer="leastsq", verbose=False, save_plot=False, show_plot=False, plot_name="exponential_fit.png", kwargs_minimizer={}, ydata_min=0.1, tau_logscale=False):
    """
    Extract the characteristic times fit and weighted with one, two, and three exponential functions for a total of three fits. Because the prefactors sum to unity, the total characteristic time is equal to a sum weighted by the prefactors. See ``custom_fit.exponential`` for more details.

    Parameters
    ----------
    xdata : numpy.ndarray
        Independent data set ranging from 0 to some quantity
    ydata : numpy.ndarray
        Dependent data set, starting at unity and decaying exponentially 
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
        Keyword arguments for ``lmfit.minimize()``
    ydata_min : float, Optional, default=0.1
        Minimum value of ydata allowed before beginning fitting process. If ydata[-1] is greater than this value, an error is thrown.
    tau_logscale : bool, Optional, default=False
        Have minimization algorithm fit the residence times with a log transform to search orders of magnitude

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
        warnings.warn("Exponential decays to {}, above threshold {}. Maximum tau value to evaluate the residence time, or increase the keyword value of ydata_min.".format(yarray[-1],ydata_min))
        flag_long = True
    else:
        flag_long = False

    output = np.zeros(12)
    uncertainties = np.zeros(12)
    # One Exp
    tmp_output1, tmp_error1 = cfit.exponential_decay(xarray, yarray,
                                                     verbose=verbose,
                                                     minimizer=minimizer,
                                                     kwargs_minimizer=kwargs_minimizer,
                                                     kwargs_parameters={"a1": {"vary": False}},
                                                    )
    output[:2] = tmp_output1
    uncertainties[:2] = tmp_error1
    # Two Exp
    if minimizer != "differential_evolution":
        tmp_output2a, _ = cfit.two_exponential_decays(xarray, yarray,
                                                      verbose=verbose,
                                                      minimizer=minimizer,
                                                      kwargs_minimizer=kwargs_minimizer,
                                                      tau_logscale=tau_logscale,
                                                      kwargs_parameters={
                                                                         "t1": {"value": tmp_output1[1], "vary": False},
                                                                         "a1": {"vary": False}
                                                                        }
                                                     )
        tmp_output2, tmp_error2 = cfit.two_exponential_decays(xarray, yarray, 
                                                         verbose=verbose,
                                                         minimizer=minimizer,
                                                         kwargs_minimizer=kwargs_minimizer,
                                                         tau_logscale=tau_logscale,
                                                         kwargs_parameters={
                                                                            "a1": {"value": tmp_output2a[0]},
                                                                            "t1": {"value": tmp_output2a[1]},
                                                                            "t2": {"value": tmp_output2a[3]},
                                                                           }
                                                        )
    else:
        tmp_output2, tmp_error2 = cfit.two_exponential_decays(xarray, yarray,
                                                         verbose=verbose,
                                                         minimizer=minimizer,
                                                         kwargs_minimizer=kwargs_minimizer,
                                                         tau_logscale=tau_logscale,
                                                        )
    output[2:6] = tmp_output2
    uncertainties[2:6] = tmp_error2

    # Three Exp
    if flag_long:
        output[6:] = np.ones(6)*np.nan
        uncertainties[6:] = np.ones(6)*np.nan
    else:
        if minimizer != "differential_evolution":
            tmp_output3a, _ = cfit.three_exponential_decays(xarray, yarray,
                                                          verbose=verbose,
                                                          minimizer=minimizer,
                                                          kwargs_minimizer=kwargs_minimizer,
                                                          tau_logscale=tau_logscale,
                                                          kwargs_parameters={
                                                                             "t1": {"value": 10*tmp_output2[1], "vary": False},
                                                                             "t2": {"value": tmp_output2[3], "vary": False},
                                                                             "t3": {"value": np.exp((np.log(10*tmp_output2[1])+np.log(tmp_output2[3]))/2)},
                                                                            }  
                                                         )  
            tmp_output3, tmp_error3 = cfit.three_exponential_decays(xarray, yarray,
                                                             verbose=verbose,
                                                             minimizer=minimizer,
                                                             kwargs_minimizer=kwargs_minimizer,
                                                             tau_logscale=tau_logscale,
                                                             kwargs_parameters={
                                                                                "a1": {"value": tmp_output3a[0]},
                                                                                "t1": {"value": tmp_output3a[1]},
                                                                                "a2": {"value": tmp_output3a[2]},
                                                                                "t2": {"value": tmp_output3a[3]},
                                                                                "t3": {"value": tmp_output3a[5]},
                                                                               }
                                                            )
        else:
            tmp_output3, tmp_error3 = cfit.three_exponential_decays(xarray, yarray,
                                                             verbose=verbose,
                                                             minimizer=minimizer,
                                                             kwargs_minimizer=kwargs_minimizer,
                                                             tau_logscale=tau_logscale,
                                                            )

        output[6:] = tmp_output3
        uncertainties[6:] = tmp_error3

    if save_plot or show_plot:
        fig, ax = plt.subplots(1,2, figsize=(6,3))
        yfit1 = tmp_output1[0]*np.exp(-xarray/tmp_output1[1])
        ax[0].plot(xarray,yarray,".",label="Data")
        ax[0].plot(xarray,yfit1,label="1 Exp.",linewidth=1)
        params = {key: value for key, value in zip(["a1","t1","a2","t2"],tmp_output2)}
        yfit2 = cfit._res_two_exponential_decays(params, xarray, 0.0)
        ax[0].plot(xarray,yfit2,label="2 Exp.",linewidth=1)
        if not flag_long:
            params = {key: value for key, value in zip(["a1","t1","a2","t2", "a3", "t3"],tmp_output3)}
            yfit3 = cfit._res_three_exponential_decays(params, xarray, 0.0)
            ax[0].plot(xarray,yfit3,label="3 Exp.",linewidth=1)
        ax[0].set_ylabel("Probability")
        ax[0].set_xlabel("Time")

        ax[1].plot(xarray,yarray,".",label="Data")
        ax[1].plot(xarray,yfit1,label="1 Exp.",linewidth=1)
        ax[1].plot(xarray,yfit2,label="2 Exp.",linewidth=1)
        if not flag_long:
            ax[1].plot(xarray,yfit3,label="3 Exp.",linewidth=1)
        ax[1].set_ylabel("log Probability")
        ax[1].set_xlabel("Time")
        ax[1].set_yscale('log')
        ax[1].legend(loc="best")
        fig.tight_layout()
        if save_plot:
            fig.savefig(plot_name,dpi=300)

        if show_plot:
            plt.show()
        plt.close("all")

    return output, uncertainties

def keypoints2csv(filename, fileout="res_time.csv", mode="a", delimiter=",", titles=None, additional_entries=None, additional_header=None, kwargs_fit={}, file_header_kwargs={}, verbose=False):
    """
    Given the path to a csv file containing residence time data, extract key values and save them to a .csv file. The file of residence time decay data should have a first column with distance values, followed by columns with radial distribution values. These data sets will be distinguished in the resulting csv file with the column headers

    Parameters
    ----------
    filename : str
        Input filename and path to file with two column format
    fileout : str, Optional, default="res_time.csv"
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
    kwargs_fit : dict, Optional, default={}
        Keywords for `characteristic_time` function
    file_header_kwargs : dict, Optional, default={}
        Keywords for ``md_spa_utils.os_manipulation.file_header`` function    
    verbose : bool, Optional, default=False
        Output fitting statistics

    Returns
    -------
    csv file with additional columns and "Types", "1 Exp: tau", "2 Exp: tau", "3 Exp: tau", "1 Exp: t1", "2 Exp a1", "2 Exp t1", "2 Exp a2", "2 Exp t2", "3 Exp a1", "3 Exp t1", "3 Exp a2", "3 Exp t2", "3 Exp a3", "3 Exp t3"
    
    """
    if not os.path.isfile(filename):
        raise ValueError("The given file could not be found: {}".format(filename))

    if titles == None:
        titles = fm.find_header(filename, **file_header_kwargs)
    data = np.transpose(np.genfromtxt(filename, delimiter=delimiter))
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
        if verbose:
            print("Calculating {} of {}, {}".format(i, len(data)-1, titles[i]))

        kwargs_fit_tmp = copy.deepcopy(kwargs_fit)
        if "plot_name" in kwargs_fit:
            tmp = kwargs_fit["plot_name"].split(".")
            tmp[0] += "_"+str(titles[i])
            kwargs_fit_tmp["plot_name"] = ".".join(tmp)

        if len(np.where(np.isnan(data[i][1:7]))[0]) == 6: # Least-squares fit will not function with number of points less than number of parameters for 3 exponentials
            output = np.nan*np.ones(12)
        elif len(np.where(data[i][1:7] < np.finfo(np.float).eps)[0]) != 0: # Least-squares fit will not function with number of points less than number of parameters for 3 exponentials
            output = np.zeros(12)
        else:
            tmp_in = np.array([0. if np.isnan(x) else x for x in data[i]])
            output, _ = characteristic_time(t_tmp, tmp_in, **kwargs_fit_tmp)
        tmp = [titles[i], output[1], output[2]*output[3]+output[4]*output[5], output[6]*output[7]+output[8]*output[9]+output[10]*output[11]]
        tmp_data.append(list(additional_entries)+tmp+list(output))

    file_headers = ["Types", "1 Exp: tau", "2 Exp: tau", "3 Exp: tau", "1 Exp: a1", "1 Exp: t1", "2 Exp a1", "2 Exp t1", "2 Exp a2", "2 Exp t2", "3 Exp a1", "3 Exp t1", "3 Exp a2", "3 Exp t2", "3 Exp a3", "3 Exp t3"]
    if not os.path.isfile(fileout) or mode=="w":
        if flag_add_header:
            file_headers = list(additional_header) + file_headers
        fm.write_csv(fileout, tmp_data, mode=mode, header=file_headers)
    else:
        fm.write_csv(fileout, tmp_data, mode=mode)




