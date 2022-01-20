
import numpy as np
import matplotlib.pyplot as plt
import os

import md_spa_utils.file_manipulation as fm
import md_spa_utils.data_manipulation as dm

from . import custom_fit as cfit

def characteristic_time(filename, delimiter=",", save_plots=None, show_plots=False, verbose=False, kwargs_fit={}):
    """
    Extract the characteristic times fit and weighted with one, two, and three exponential functions for a total of three fits. Because the prefactors sum to unity, the total characteristic time is equal to a sum weighted by the prefactors. See ``custom_fit.exponential`` for more details.

    Parameters
    ----------
    filename : str
        Path and filename of data set to be fit
    delimiter : str, Optional, default=","
        Delimiter between columns used in ``numpy.genfromtxt()``
    save_plots : str, Optional, default=None
        If not None, plots comparing the exponential fits will be saved to this filename 
    show_plots : bool, Optional, default=False
        If true, the fits will be shown
    verbose : bool, Optional, default=False
        Output fitting statistics
    kwargs_fit : dict, Optional, default={}
        Keyword arguements for exponential functions in ``custom_fit``

    Returns
    -------
    """

    data = np.transpose(np.genfromtxt(filename,delimiter=","))
    output1, error1 = cfit.exponential(data[0],data[1], verbose=verbose, **kwargs_fit)
    output1, error1 = cfit.two_exponentials(data[0],data[1], verbose=verbose, **kwargs_fit)
    output1, error1 = cfit.three_exponentials(data[0],data[1], verbose=verbose, **kwargs_fit)

    #if save_plots != None or show_plots:
    #    yfit1 = exponential_res_1(Result1.params) + yarray
    #    yfit2 = exponential_res_2(Result2.params) + yarray
    #    yfit3 = exponential_res_3(Result3.params) + yarray
    #    plt.plot(xarray,yarray,".",label="Data")
    #    plt.plot(xarray,yfit1,label="1 Gauss",linewidth=1)
    #    plt.plot(xarray,yfit2,label="2 Gauss",linewidth=1)
    #    plt.plot(xarray,yfit3,label="3 Gauss",linewidth=1)
    #    plt.ylabel("Probability")
    #    plt.xlabel("Time")
    #    plt.tight_layout()
    #    plt.legend(loc="best")
    #    if save_plots != None:
    #        plt.savefig(save_plots,dpi=300)

    #    plt.figure(2)
    #    plt.plot(xarray,yarray,".",label="Data")
    #    plt.plot(xarray,yfit1,label="1 Gauss",linewidth=1)
    #    plt.plot(xarray,yfit2,label="2 Gauss",linewidth=1)
    #    plt.plot(xarray,yfit3,label="3 Gauss",linewidth=1)
    #    plt.ylabel("log Probability")
    #    plt.xlabel("Time")
    #    plt.tight_layout()
    #    plt.yscale('log')
    #    plt.legend(loc="best")
    #    if save_plots != None:
    #        tmp_save_plot = list(os.path.split(save_plots))
    #        tmp_save_plot[1] = "log_"+tmp_save_plot[1]
    #        plt.savefig(os.path.join(*tmp_save_plot),dpi=300)

    #    if show_plots:
    #        plt.show()
    #    plt.close("all")

    return output, uncertainties

def keypoints2csv(filename, fileout="res_time.csv", mode="a", delimiter=",", titles=None, additional_entries=None, additional_header=None, kwargs_fit={}, file_header_kwargs={}):
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
        if "plot_name" in kwargs_fit:
            tmp = kwargs_fit["plot_name"].split(".")
            tmp[0] += str(titles[i])
            kwargs_fit["plot_name"] = ".".join(tmp)
        output, _ = cfit.exponential(t_tmp, data[i], **kwargs_fit)
        tmp = [titles[i], output[0], output[1]*output[2]+output[3]*output[4], output[5]*output[6]+output[7]*output[8]+output[9]*output[10]]
        tmp_data.append(list(additional_entries)+tmp+list(output))

    file_headers = ["Types", "1 Exp: tau", "2 Exp: tau", "3 Exp: tau", "1 Exp: t1", "2 Exp a1", "2 Exp t1", "2 Exp a2", "2 Exp t2", "3 Exp a1", "3 Exp t1", "3 Exp a2", "3 Exp t2", "3 Exp a3", "3 Exp t3"]
    if not os.path.isfile(fileout) or mode=="w":
        if flag_add_header:
            file_headers = list(additional_header) + file_headers
        fm.write_csv(fileout, tmp_data, mode=mode, header=file_headers)
    else:
        fm.write_csv(fileout, tmp_data, mode=mode)




