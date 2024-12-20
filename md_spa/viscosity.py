""" Calculate viscosity and infinite shear modulus from pressure tensor data.

    Recommend loading with:
    ``import md_spa.viscosity as visc``

"""

import copy
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy import integrate
from scipy.stats import linregress

from md_spa.utils import data_manipulation as dm
from md_spa.utils import file_manipulation as fm
from md_spa import custom_fit as cfit

styles = ["-", "--", ":"]

def pressure2viscosity_csv(time, p_xy, filename="viscosity_values.csv", calc_kwargs={}, csv_kwargs={}, method="Green-Kubo"):
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
    filename : str, default="viscosity_running_integral.csv"
        Filename for csv file
    calc_kwargs : dict, default={}
        Keyword arguements for :func:`md_spa.viscosity.running_acf_integral` or :func:`md_spa.viscosity.running_einstein`
    csv_kwargs : dict, default={}
        Keywords for :func:`md_spa.file_manipulation.write_csv`
    method : str, default="Green-Kubo"
        Method of extracting viscosity from equilibrium MD calculation, kwargs adjusted with ``calc_kwargs``

        - "Einstein": See :func:`md_spa.viscosity.running_einstein`
        - "Green-Kubo": See :func:`md_spa.viscosity.running_acf_integral`
        
    Returns
    -------
    Saved file with the viscosity coefficient at each time frame. See ``method`` for units, default according to :math:`\eta k_{B}T/V` with units of ``pressure**2 * time``.

    """

    if not dm.isiterable(time) or dm.isiterable(time[0]):
        raise ValueError("The array `time` must be of shape (,N)")

    if not dm.isiterable(p_xy) or not dm.isiterable(p_xy[0]) or dm.isiterable(p_xy[0][0]):
        raise ValueError("The array `p_xy` must be of shape (6,N) or (3,N)")
    elif len(p_xy) not in [3,6]:
        raise ValueError("The array `p_xy` must be of shape (6,N) or (3,N)")

    tmp_kwargs = {}
    if "scale_coefficient" in calc_kwargs:
        tmp_kwargs["scale_coefficient"] = calc_kwargs["scale_coefficient"]
    if "error_type" in calc_kwargs:
        tmp_kwargs["error_type"] = calc_kwargs["error_type"]
    G_inf = high_freq_shear_modulus(p_xy, **tmp_kwargs)

    if method == "Green-Kubo":
        eta, stnderr = running_acf_integral(time, p_xy, **calc_kwargs)
    elif method == "Einstein":
        eta, stnderr = running_einstein(time, p_xy, **calc_kwargs)
    else:
        raise ValueError("Define one of two supported methods: Einstein or Green-Kubo")

    tmp_kwargs = {}
    tmp_kwargs["header"] = ["time", "cumulative integral", "integral SE"]
    tmp_kwargs["header_comment"] = "# Calculating Viscosity with {} method. G_inf={}+-{}\n#".format(method,*G_inf) 
    tmp_kwargs.update(csv_kwargs)
    fm.write_csv(filename, np.transpose(np.array([time,eta,stnderr])), **tmp_kwargs)


def pressure2shear_modulus_csv(time, p_xy, filename="shear_modulus_values.csv", calc_kwargs={}, csv_kwargs={}):
    """
    Calculate the shear autocorrelation function of an equilibrium molecular dynamics simulation from the pressure tensor.

    DOI: 10.1063/5.0098265

    Parameters
    ----------
    time : numpy.ndarray
        Array of time values corresponding to pressure tensor data
    p_xy : numpy.ndarray
        Pressure tensor data at each time step. The first dimension can be of either length 3 or 6, representing the three off diagonals (xy, xz, yz) or the entire pressure tensor (xx, yy, zz, xy, xz, yz). The second dimension is of the same length as ``time``.
    filename : str, default="shear_modulus_values.csv"
        Filename for csv file
    calc_kwargs : dict, default={}
        Keyword arguements for :func:`md_spa.viscosity.dynamic_shear_modulus`
    csv_kwargs : dict, default={}
        Keywords for :func:`md_spa.file_manipulation.write_csv`
        
    Returns
    -------
    Saved file with the shear autocorrelation function according to :math:`G k_{B}T/V` with units of ``pressure**2 * time``.

    """

    if not dm.isiterable(time) or dm.isiterable(time[0]):
        raise ValueError("The array `time` must be of shape (,N)")

    if not dm.isiterable(p_xy) or not dm.isiterable(p_xy[0]) or dm.isiterable(p_xy[0][0]):
        raise ValueError("The array `p_xy` must be of shape (6,N) or (3,N)")
    elif len(p_xy) not in [3,6]:
        raise ValueError("The array `p_xy` must be of shape (6,N) or (3,N)")

    tmp_kwargs = {}
    if "scale_coefficient" in calc_kwargs:
        tmp_kwargs["scale_coefficient"] = calc_kwargs["scale_coefficient"]
    if "error_type" in calc_kwargs:
        tmp_kwargs["error_type"] = calc_kwargs["error_type"]

    Gshear, stnderr = dynamic_shear_modulus(time, p_xy, **calc_kwargs)

    tmp_kwargs = {}
    tmp_kwargs["header"] = ["time", "G [pressure^2 time]", "G SE"]
    tmp_kwargs.update(csv_kwargs)
    fm.write_csv(filename, np.transpose(np.array([time, Gshear, stnderr])), **tmp_kwargs)


def high_freq_shear_modulus(p_xy, error_type="standard error", scale_coefficient=1,):
    """
    Calculate the high frequency shear modulus

    Here the quantity is evalulated from the time average of each off diagonal stress tensor, squared

    Parameters
    ----------
    p_xy : numpy.ndarray
        Pressure tensor data at each time step. The first dimension can be of either length 6, representing the entire pressure tensor (xx, yy, zz, xy, xz, yz). The second dimension is of the same length as ``time``.
    scale_coefficient : float, default=1.0
        Prefactor to scale the viscosity coefficient. The default results in a value of :math:`G_{\infty} k_{B}T/V`
    error_type : str, default="standard error"
        Type of error to be saved, either "standard error" or "standard deviation"

    Returns
    -------
    G_inf : float
        High frequency shear modulus. If ``scale_coefficient`` is unchanged, units are :math:`G_{\infty} k_{B}T/V` with units of ``pressure**2 * time``. 
    G_inf_st : float
        Standard error or standard deviation of the high frequency shear modulus

    """

    if not dm.isiterable(p_xy) or not dm.isiterable(p_xy[0]) or dm.isiterable(p_xy[0][0]):
        raise ValueError("The array `p_xy` must be of shape (6,N)")
    elif len(p_xy) != 6:
        raise ValueError("The array `p_xy` must be of shape (6,N)")

    tmp = np.mean(np.square(p_xy[3:]), axis=1)
    G_inf = np.mean(tmp)
    stnderror = np.std(tmp)
    if error_type == "standard error":
        stnderror = stnderror/np.sqrt(len(tmp)) # pxy, pxz, pyz
    elif error_type == "standard deviation":
        pass
    else:
         raise ValueError("The `error_type`, {}, is not supported".format(error_type))

    return scale_coefficient*G_inf, scale_coefficient*stnderror

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
    scale_coefficient : float, default=1.0
        Prefactor to scale the viscosity coefficient. The default results in a value of :math:`2\eta k_{B}T/V`
    error_type : str, default="standard error"
        Type of error to be saved, either "standard error" or "standard deviation"
    skip : int, default=1
        Number of frame to skip to obtain an independent trajectory
    show_plot : bool, default=False
        choose to show a plot of the fit
    save_plot : bool, default=False
        choose to save a plot of the fit
    title : str, default=None
        The title used in the cumulative_integral plot, note that this str is also added as a prefix to the ``plot_name``.
    plot_name : str, default="einstein_viscosity_components.png"
        If ``save_plot==True`` the cumulative_integral will be saved with the debye-waller factor marked, The ``title`` is added as a prefix to this str

    Returns
    -------
    cumulative_integral : numpy.ndarray
        Viscosity coefficient at each time frame according to :math:`2\eta k_{B}T/V` with units of ``pressure**2 * time``. 
        
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
    p_new[0] = (p_xy[0]-p_xy[1])/2
    p_new[1] = (p_xy[1]-p_xy[2])/2
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
            color = "c" if (lx ==5 and i < 2) else "b"
            if lx <5:
                label = p_labels[i+2]
            else:
                label = p_labels[i]
            plt.plot(time, scale_coefficient*eta_tmp, linewidth=0.5, color=color, label=label, linestyle=style)
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

def running_acf_integral(time, p_xy, error_type="standard error", scale_coefficient=1, skip=1, show_plot=False, title=None, save_plot=False, plot_name="green-kubo_viscosity_components.png", dims="xyz"):
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
    scale_coefficient : float, default=1.0
        Prefactor to scale the viscosity coefficient. The default results in a value of :math:`\eta k_{B}T/V`
    error_type : str, default="standard error"
        Type of error to be saved, either "standard error" or "standard deviation"
    skip : int, default=1
        Number of frame to skip to obtain an independent trajectory
    show_plot : bool, default=False
        choose to show a plot of the fit
    save_plot : bool, default=False
        choose to save a plot of the fit
    title : str, default=None
        The title used in the cumulative_integral plot, note that this str is also added as a prefix to the ``plot_name``.
    plot_name : str, default="green-kubo_viscosity_components.png"
        If ``save_plot==True`` the cumulative_integral will be saved with the debye-waller factor marked, The ``title`` is added as a prefix to this str

    Returns
    -------
    cumulative_integral : numpy.ndarray
        Viscosity coefficient at each time frame. Units of ``pressure**2 * time``.
        
    """
    if not dm.isiterable(time) or dm.isiterable(time[0]):
        raise ValueError("The array `time` must be of shape (,N)")

    if not dm.isiterable(p_xy) or not dm.isiterable(p_xy[0]) or dm.isiterable(p_xy[0][0]):
        raise ValueError("The array `p_xy` must be of shape (6,N)")
    elif len(p_xy) != 6:
        raise ValueError("The array `p_xy` must be of shape (6,N)")

    p_new, lx, labels = _dim_integrals(p_xy, dims)

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
        for i,eta_tmp in enumerate(integral_set):
            color = "c" if i < lx-3 else "b"
            style = styles[i] if i < lx-3 else styles[i-lx+3]
            plt.plot(time, scale_coefficient*eta_tmp, linewidth=0.5, color=color, label=labels[i], linestyle=style)
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


def _dim_integrals(p_xy, dims):
    """Given target dimensions, provide indices
    """

    _, npts = np.shape(p_xy)
    if dims == "xyz":
        labels = ["($\eta_{xx} - \eta_{yy}$)/2", "($\eta_{yy} - \eta_{zz}$)/2", "$\eta_{xy}$", "$\eta_{xz}$", "$\eta_{yz}$"]
        lx = 5
        p_new = np.zeros((lx,npts))
        p_new[0] = (p_xy[0]-p_xy[1])/2
        p_new[1] = (p_xy[1]-p_xy[2])/2
        p_new[2:] = p_xy[3:]
    elif dims == "xy":
        labels = ["($\eta_{xx} - \eta_{yy}$)/2", "$\eta_{xy}$", "$\eta_{xz}$", "$\eta_{yz}$"]
        lx = 4
        p_new = np.zeros((lx,npts))
        p_new[0] = (p_xy[0]-p_xy[1])/2
        p_new[1:] = p_xy[3:]
    elif dims == "xz":
        labels = ["($\eta_{xx} - \eta_{zz}$)/2", "$\eta_{xy}$", "$\eta_{xz}$", "$\eta_{yz}$"]
        lx = 4
        p_new = np.zeros((lx,npts))
        p_new[1] = (p_xy[0]-p_xy[2])/2
        p_new[1:] = p_xy[3:]
    elif dims == "yz":
        labels = ["($\eta_{yy} - \eta_{zz}$)/2", "$\eta_{xy}$", "$\eta_{xz}$", "$\eta_{yz}$"]
        lx = 4
        p_new = np.zeros((lx,npts))
        p_new[1] = (p_xy[1]-p_xy[2])/2
        p_new[1:] = p_xy[3:]
    else:
        raise ValueError(f"Dimensions, {dims}, are not recognized. Choose 'xyz', 'xy', 'xz', or 'xz'.")

    return p_new, lx, labels

def dynamic_shear_modulus(time, p_xy, error_type="standard error", scale_coefficient=1, skip=1, show_plot=False, title=None, save_plot=False, plot_name="green-kubo_viscosity_components.png"):
    """
    Calculate the time dependent dynamic shear modulus of an equilibrium molecular dynamics simulation from the independent pressure components: pxy, pxz, pyz, (pxx-pyy)/2, and (pyy-pzz)/2.

    DOI: 10.1063/5.0098265

    Parameters
    ----------
    time : numpy.ndarray
        Array of time values corresponding to pressure tensor data
    p_xy : numpy.ndarray
        Pressure tensor data at each time step. The first dimension can be of either length 3 or 6, representing the three off diagonals (xy, xz, yz) or the entire pressure tensor (xx, yy, zz, xy, xz, yz). The second dimension is of the same length as ``time``.
    scale_coefficient : float, default=1.0
        Prefactor to scale the viscosity coefficient. The default results in a value of :math:`G k_{B}T/V`
    error_type : str, default="standard error"
        Type of error to be saved, either "standard error" or "standard deviation"
    skip : int, default=1
        Number of frame to skip to obtain an independent trajectory
    show_plot : bool, default=False
        choose to show a plot of the fit
    save_plot : bool, default=False
        choose to save a plot of the fit
    title : str, default=None
        The title used in the cumulative_integral plot, note that this str is also added as a prefix to the ``plot_name``.
    plot_name : str, default="green-kubo_viscosity_components.png"
        If ``save_plot==True`` the cumulative_integral will be saved with the debye-waller factor marked, The ``title`` is added as a prefix to this str

    Returns
    -------
    cumulative_integral : numpy.ndarray
        Time dependent shear modulus at each time frame. Units of ``pressure**2 * time``.
        
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
    p_new[0] = (p_xy[0]-p_xy[1])/2
    p_new[1] = (p_xy[1]-p_xy[2])/2
    p_new[2:] = p_xy[3:]

    # Same result whether the integral is taken before or after averaging the tensor components
    acf_set = np.array([dm.autocorrelation(x) for x in p_new])
    Gshear = np.mean(acf_set, axis=0)
    stnderror = np.sqrt(np.sum(np.square(acf_set-Gshear), axis=0)/(lx-1))
    if error_type == "standard error":
        stnderror = stnderror/np.sqrt(lx)
    elif error_type == "standard deviation":
        pass
    else:
         raise ValueError("The `error_type`, {}, is not supported".format(error_type))

    if save_plot or show_plot:
        p_labels = {0: "($G_{xx} - G_{yy}$)/2", 1: "($G_{yy} - G_{zz}$)/2", 2: "$G_{xy}$", 3: "$G_{xz}$", 4: "$G_{yz}$"}
        for i,G_tmp in enumerate(acf_set):
            if lx ==5 and i < 2:
                color = "c"
            else:
                color = "b"
            if lx <5:
                label = p_labels[i+2]
            else:
                label = p_labels[i]
            plt.plot(time, scale_coefficient*G_tmp, linewidth=0.5, color=color, label=label)
        plt.plot(time, scale_coefficient*Gshear, "k", label=r"G")
        plt.fill_between(time, scale_coefficient*(Gshear-stnderror), scale_coefficient*(Gshear+stnderror), alpha=0.25, color="black")
        plt.plot([time[0], time[-1]], [0,0], "k", linewidth=0.5)
        plt.legend(loc="lower right")
        plt.ylim((0,scale_coefficient*np.max(Gshear+stnderror)))
        plt.xlim((0,time[np.where(Gshear/Gshear[0] < 0.01)[0][0]]))
        plt.xlabel("time")
        plt.ylabel("G")
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

    return scale_coefficient*Gshear, scale_coefficient*stnderror


def keypoints2csv(filename, fileout="viscosity.csv", mode="a", delimiter=",", title=None, additional_entries=None, additional_header=None, kwargs_find_viscosity={}, file_header_kwargs={}, method=None):
    """
    Given the path to a csv file containing viscosity coefficient vs. time data, extract key values and save them to a .csv file. The file of cumulative_integral data should have a first column with distance values, followed by columns with radial distribution values. These data sets will be distinguished in the resulting csv file with the column headers

    Parameters
    ----------
    filename : str
        Input filename and path to lammps cumulative_integral output file
    fileout : str, default="viscosity.csv"
        Filename of output .csv file
    mode : str, default="a"
        Mode used in writing the csv file, either "a" or "w".
    delimiter : str, default=","
        Delimiter between data in input file
    title : list[str], default=None
        Titles for plots if that is specified in the ``kwargs_find_viscosity``
    additional_entries : list, default=None
        This iterable structure can contain additional information about this data to be added to the beginning of the row
    additional_header : list, default=None
        If the csv file does not exist, these values will be added to the beginning of the header row. This list must be equal to the `additional_entries` list.
    kwargs_find_viscosity : dict, default={}
        Keywords for :func:`md_spa.viscosity.find_green_kubo_viscosity` or :func:`md_spa.viscosity.find_einstein_viscosity` functions depending on ``method``
    file_header_kwargs : dict, default={}
        Keywords for :func:`md_spa.os_manipulation.file_header` function    
    method : str, default=None
        Can be 'Einstein' or 'Green-Kubo', specifies the type of data to be analyzed. If :func:`md_spa.viscosity.pressure2viscosity_csv` was used to generate the data, the method is extracted from the header.

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

    with open(filename, "r") as f:
        line = f.readline()

    if method == None:
        if "Green-Kubo" in line:
            method = "Green-Kubo"
        elif "Einstein" in line:
            method = "Einstein"
        else:
            raise ValueError("Viscosity data ``method`` must be specified as either 'Einstein' or 'Green-Kubo'")
    tmp = line.strip().split("=")[1].split("+-")
    if len(tmp) == 2:
        G_inf = [float(x) for x in tmp]   
    else:
        G_inf = [None, None]

    data = np.transpose(np.genfromtxt(filename, delimiter=delimiter))
    tmp_kwargs = copy.deepcopy(kwargs_find_viscosity)
    if "title" not in tmp_kwargs and title != None:
        tmp_kwargs["title"] = title

    if method == "Green-Kubo":
        viscosity, parameters, stnderror, tcut, redchi = find_green_kubo_viscosity(data[0], data[1], data[2], **tmp_kwargs)
        tmp_list = [val for pair in zip([viscosity]+list(parameters), list(stnderror)) for val in pair]
        output = [list(additional_entries)+[title, *G_inf, tcut]+list(tmp_list)]
        file_headers = ["Group", "G_inf", "G_inf StD", "tcut", "eta", "eta SE", "A", "A SE", "alpha", "alpha SE", "tau1", "tau1 SE", "tau2", "tau2 SE"]
    else:
        best, longest = find_einstein_viscosity(data[0], data[1], **tmp_kwargs)
        output = [list(additional_entries)+[title, *G_inf]+list(best)+list(longest)]
        file_headers = ["Group", "G_inf", "G_inf StD" "Best Eta", "B Eta SE", "B t_bound1", "B t_bound2", "B Exponent", "B Intercept", "B Npts", "Longest Eta", "L Eta SE", "L t_bound1", "L t_bound2", "L Exponent", "L Intercept", "L Npts"]

    if not os.path.isfile(fileout) or mode=="w":
        if flag_add_header:
            file_headers = list(additional_header) + file_headers
        fm.write_csv(fileout, output, mode=mode, header=file_headers)
    else:
        fm.write_csv(fileout, output, mode=mode)

def shear_modulus2csv(filename, fileout="shear_modulus.csv", mode="a", delimiter=",", title=None, additional_entries=None, additional_header=None, fit_exp_kwargs={}, fit_stretched_exp_kwargs={}, fit_two_stretched_exp_kwargs={}, fit_three_stretched_exp_kwargs={}, fit_limits=(None,None), show_plot=False, save_plot=False, plot_name="shear_modulus.png", verbose=False,):
    """
    Given the path to a csv file containing viscosity coefficient vs. time data, extract key values and save them to a .csv file. The file of cumulative_integral data should have a first column with distance values, followed by columns with radial distribution values. These data sets will be distinguished in the resulting csv file with the column headers

    Parameters
    ----------
    filename : str
        Input filename and path to lammps cumulative_integral output file
    fileout : str, default="viscosity.csv"
        Filename of output .csv file
    mode : str, default="a"
        Mode used in writing the csv file, either "a" or "w".
    delimiter : str, default=","
        Delimiter between data in input file
    additional_entries : list, default=None
        This iterable structure can contain additional information about this data to be added to the beginning of the row
    additional_header : list, default=None
        If the csv file does not exist, these values will be added to the beginning of the header row. This list must be equal to the `additional_entries` list.
    fit_exp_kwargs : dict, default={}
        Keywords for :func:`md_spa.custom_fit.exponential_decay`
    fit_stretched_exp_kwargs : dict, default={}
        Keywords for :func:`md_spa.custom_fit.stretched_exponential_decay`
    fit_two_stretched_exp_kwargs : dict, default={}
        Keywords for :func:`md_spa.custom_fit.two_stretched_exponential_decays`
    fit_three_stretched_exp_kwargs : dict, default={}
        Keywords for :func:`md_spa.custom_fit.three_stretched_exponential_decays`
    fit_limits : tuple, default=(None,None)
        Choose the time values to frame the area from which to estimate the viscosity so that, ``fit_limits[0] < time < fit_limits[1]``.
    show_plot : bool, default=False
        choose to show a plot of the fit
    save_plot : bool, default=False
        choose to save a plot of the fit
    title : str, default=None
        The title used in the Gshear plot, note that this str is also added as a prefix to the ``plot_name``.
    plot_name : str, default="shear_modulus.png"
        If ``save_plot==True`` the Gshear will be saved with the viscosity value marked in blue, The ``title`` is added as a prefix to this str
    verbose : bool, default=False
        Will print intermediate values or not

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
    tmp_kwargs = copy.deepcopy(fit_exp_kwargs)
    if "title" not in tmp_kwargs and title != None:
        tmp_kwargs["title"] = title
    tmp2_kwargs = copy.deepcopy(fit_stretched_exp_kwargs)
    if "title" not in tmp_kwargs and title != None:
        tmp2_kwargs["title"] = title

    if fit_limits[1] != None:
        try:
            data = data[:, data[0] < fit_limits[1]]
        except:
            pass
    if fit_limits[0] != None:
        try:
            data = data[:, data[0] > fit_limits[0]]
        except:
            pass

    Gshear0 = data[1,0] if not np.isnan(data[1,0]) else data[1,1]
    if np.isnan(Gshear0):
        raise ValueError("Why are your first two values NaN???")

    tmp_kwargs = copy.deepcopy(fit_exp_kwargs)
    tmp2_kwargs = copy.deepcopy(fit_stretched_exp_kwargs)
    tmp3_kwargs = {"kwargs_parameters":{
        "A": {"max": 0.4},
        "beta1": {"min": 1.5, "max": 2, "value": 2},
        "beta2": {"min": 0.9, "max": 1.5, "value": 1},
    }}
    tmp3_kwargs.update(copy.deepcopy(fit_two_stretched_exp_kwargs))
    tmp4_kwargs = {"kwargs_parameters":{
        "A1": {"max": 0.4},
        "beta1": {"min": 1.5, "max": 2, "value": 2},
        "A2": {"max": 0.6},
        "beta1": {"min": 1, "max": 1.9, "value": 1.6},
        "beta3": {"min": 0.9, "max": 1.5, "value": 1},
    }}
    tmp4_kwargs.update(copy.deepcopy(fit_three_stretched_exp_kwargs))
    if "weighting" not in tmp_kwargs:
        tmp_kwargs["weighting"] = np.array([1/x if x > np.finfo(float).eps else 1/np.finfo(float).eps for x in data[2]/Gshear0])
        tmp2_kwargs["weighting"] = np.array([1/x if x > np.finfo(float).eps else 1/np.finfo(float).eps for x in data[2]/Gshear0])
        tmp3_kwargs["weighting"] = np.array([1/x if x > np.finfo(float).eps else 1/np.finfo(float).eps for x in data[2]/Gshear0])
        tmp4_kwargs["weighting"] = np.array([1/x if x > np.finfo(float).eps else 1/np.finfo(float).eps for x in data[2]/Gshear0])

    if "verbose" not in tmp_kwargs:
        tmp_kwargs["verbose"] = verbose
        tmp2_kwargs["verbose"] = verbose
        tmp3_kwargs["verbose"] = verbose
        tmp4_kwargs["verbose"] = verbose

    # Maxwell Relaxation Exp Decay
    parameters, uncertainties, redchi = cfit.exponential_decay(data[0], data[1]/Gshear0, **tmp_kwargs)
    tmp_list = [Gshear0] + [val for pair in zip(parameters, uncertainties) for val in pair] + [redchi]

    # Stretched Exp Decay
    parameters2, uncertainties2, redchi = cfit.stretched_exponential_decay(data[0], data[1]/Gshear0, **tmp2_kwargs)
    tmp_list += [val for pair in zip(parameters2, uncertainties2) for val in pair] + [redchi]

    # Two Stretched
    parameters3, uncertainties3, redchi = cfit.two_stretched_exponential_decays(data[0], data[1]/Gshear0, **tmp3_kwargs)
    tmp_list += [val for pair in zip(parameters3, uncertainties3) for val in pair] + [redchi]

    # Three Stretched
    parameters4, uncertainties4, redchi = cfit.three_stretched_exponential_decays(data[0], data[1]/Gshear0, **tmp4_kwargs)
    tmp_list += [val for pair in zip(parameters4, uncertainties4) for val in pair] + [redchi]

    # Output
    output = [list(additional_entries)+[title]+list(tmp_list)]
    file_headers = ["Group", "G_inf", "exp tau", "exp tau SE", "exp red chi^2", "str exp taubeta", "str exp taubeta SE", "str exp beta tau", "str exp beta SE", "str exp red chi^2", "2 str exp A", "2 str exp A SE", "2 str exp tau1beta1", "2 str exp tau1beta1 SE", "2 str exp beta1", "2 str exp beta1 SE", "2 str exp tau2beta2", "2 str exp tau2beta2 SE", "2 str exp beta2", "2 str exp beta2 SE", "2 str exp red chi^2", "3 str exp A1", "3 str exp A1 SE", "3 str exp tau1beta1", "3 str exp tau1beta1 SE", "3 str exp beta1", "3 str exp beta1 SE", "3 str exp A2", "3 str exp A2 SE", "3 str exp tau2beta2", "3 str exp tau2beta2 SE", "3 str exp beta2", "3 str exp beta2 SE", "3 str exp tau3beta3", "3 str exp tau3beta3 SE", "3 str exp beta3", "3 str exp beta3 SE", "3 str exp red chi^2", "4 str exp A1", "4 str exp A1 SE", "4 str exp tau1beta1", "4 str exp tau1beta1 SE", "4 str exp beta1", "4 str exp beta1 SE", "4 str exp A2", "4 str exp A2 SE", "4 str exp tau2beta2", "4 str exp tau2beta2 SE", "4 str exp beta2", "4 str exp beta2 SE", "4 str exp A3", "4 str exp A3 SE", "4 str exp tau3beta3", "4 str exp tau3beta3 SE", "4 str exp beta3", "4 str exp beta3 SE", "4 str exp tau4beta4", "4 str exp tau4beta4 SE", "4 str exp beta4", "4 str exp beta4 SE", "4 str exp red chi^2",]

    if not os.path.isfile(fileout) or mode=="w":
        if flag_add_header:
            file_headers = list(additional_header) + file_headers
        fm.write_csv(fileout, output, mode=mode, header=file_headers)
    else:
        fm.write_csv(fileout, output, mode=mode)

    if save_plot or show_plot:
        linewidth = 0.75
        fig, axs = plt.subplots(1,2, figsize=(6,4))
        for i in range(2):
            axs[i].fill_between(data[0], data[1]-data[2], data[1]+data[2], edgecolor=None, alpha=0.15, facecolor="black")
            tmp = cfit._res_exponential_decay({"a1": parameters[0], "t1": parameters[1]}, data[0], np.zeros(len(data[0])))
            axs[i].plot( data[0], data[1], "k", label="Data", marker=".", linestyle="None", ms=4)
            axs[i].plot(
                data[0], tmp,
                "r--", label="Exp Fit", linewidth=linewidth
            )
            axs[i].plot(
                data[0],
                Gshear0*cfit._res_stretched_exponential_decay({"taubeta": parameters2[0], "beta": parameters2[1]}, data[0], np.zeros(len(data[0]))),
                "g-.", label="Stretched Exp", linewidth=linewidth
            )
            axs[i].plot(
                data[0],
                Gshear0*cfit._res_two_stretched_exponential_decays({"A": parameters3[0], "tau1beta1": parameters3[1], "beta1": parameters3[2], "tau2beta2": parameters3[3], "beta2": parameters3[4],}, data[0], np.zeros(len(data[0]))),
                color="b", linestyle=":", label="2 Stretched Exp", linewidth=linewidth
            )
            axs[i].plot(
                data[0],
                Gshear0*cfit._res_three_stretched_exponential_decays({"A1": parameters4[0], "tau1beta1": parameters4[1], "beta1": parameters4[2], "A2": parameters4[3], "tau2beta2": parameters4[4], "beta2": parameters4[5], "tau3beta3": parameters4[6], "beta3": parameters4[7],}, data[0], np.zeros(len(data[0]))),
                color="c", linestyle=":", label="3 Stretched Exp", linewidth=linewidth
            )
            if i == 0:
                axs[i].legend(loc="upper right")
                axs[i].set_ylim((0, Gshear0))
            else:
                axs[i].set_yscale('log')
                axs[i].set_xscale('log')
                axs[i].set_ylim((1e-6, Gshear0))
            axs[i].set_xlabel("time")
            axs[i].set_ylabel("$G$")
            axs[i].set_xlim((0, 6*parameters[1]))
        if title != None:
            axs[i].set_title(title)
        plt.tight_layout()
        if save_plot:
            if title != None:
                tmp = os.path.split(plot_name)
                plot_name = os.path.join(tmp[0],title.replace(" ", "")+"_"+tmp[1])
            plt.savefig(plot_name,dpi=300)
        if show_plot:
            plt.show()
        plt.close("all")


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
    fit_limits : tuple, default=(None,None)
        Choose the time values to frame the area from which to estimate the viscosity so that, ``fit_limits[0] < time < fit_limits[1]``.
    weighting_method : str, default="b-exponent"
        The error of the cumulative integral increases with time as does the autocorrelation function from which it is computed. Method options include:

        - "b-exponent": Calculates weighing from the function ``time**(-b)`` where b is defined in ``b_exponent``
        - "inverse": Takes the inverse of the provided standard deviation

    b_exponent : float, default=None
        Exponent used if ``weighting_method == "b-exponent"``. This value can be calculated in :func:`md_spa.viscosity.fit_standard_deviation_exponent`
    tcut_fraction : float, default=0.4
        Choose a maximum cut-off in time at ``np.where(integral_error > tcut_fraction*np.mean(cumulative_integral))[0][0]`` where these arrays have been bound with ``fit_limits``. If this feature is not desired, set ``tcut_fraction=None``
    show_plot : bool, default=False
        choose to show a plot of the fit
    save_plot : bool, default=False
        choose to save a plot of the fit
    title : str, default=None
        The title used in the cumulative_integral plot, note that this str is also added as a prefix to the ``plot_name``.
    plot_name : str, default="green-kubo_viscosity.png"
        If ``save_plot==True`` the cumulative_integral will be saved with the viscosity value marked in blue, The ``title`` is added as a prefix to this str
    verbose : bool, default=False
        Will print intermediate values or not
    fit_kwargs : dict, default={}
        Keyword arguements for exponential functions in :func:`md_spa.custom_fit.double_viscosity_cumulative_exponential`

    Returns
    -------
    viscosity : float
        Estimation of the viscosity from the fit of a double cumlulative exponential distribution to the running integral of 
    parameters : numpy.ndarray
        Parameters from :func:`md_spa.custom_fit.double_viscosity_cumulative_exponential` fit
    uncertainties : numpy.ndarray
        Standard error for viscosity and parameters (the former propagated from the latter).
    tcut : float
        Value of time used as an upper bound in fitting process
    redchi : float
        Reduced Chi^2 from `lmfit.MinimizerResult <https://lmfit.github.io/lmfit-py/fitting.html#lmfit.minimizer.MinimizerResult>`_

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
            cumulative_integral = cumulative_integral[time < fit_limits[1]]
            integral_error = np.array(integral_error)[time < fit_limits[1]]
            time = time[fit_limits[1] > time]
        except:
            pass
    if fit_limits[0] != None:
        try:
            cumulative_integral = cumulative_integral[fit_limits[0] < time]
            integral_error = np.array(integral_error)[fit_limits[0] < time]
            time = time[fit_limits[0] < time]
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
    else:
        tcut = None

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

    parameters, uncertainties, redchi = cfit.double_viscosity_cumulative_exponential(time, cumulative_integral, **tmp_kwargs)

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

    return viscosity, parameters, np.insert(uncertainties, 0, visc_error), tcut, redchi

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
    fit_kwargs : dict, default={}
        Keyword arguements in :func:`md_spa.custom_fit.power_law`
    show_plot : bool, default=False
        choose to show a plot of the fit
    save_plot : bool, default=False
        choose to save a plot of the fit
    title : str, default=None
        The title used in the cumulative_integral plot, note that this str is also added as a prefix to the ``plot_name``.
    plot_name : str, default="power-law_fit.png"
        If ``save_plot==True`` the cumulative_integral will be saved with the debye-waller factor marked, The ``title`` is added as a prefix to this str

    Returns
    -------
    prefactor : float
        Prefactor, A, in functional form A*time**b
    b-exponent : float
        Exponent in functional form: A*time**b
    uncertainties : list
        List of uncertainties in provided fit

    """

    parameters, uncertainties, redchi = cfit.power_law(time, standard_deviation, **fit_kwargs)

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
    Extract the viscosity from the Einstein relation: :math:`2\eta = d (cumulative\_integral)/dt` where the ``cumulative_integral`` is the square of the average running integral of the pressure tensor components vs time.

    Parameters
    ----------
    time : numpy.ndarray
        Array of time values corresponding to pressure tensor data
    cumulative_integral : numpy.ndarray
        Viscosity coefficient at each time frame
    min_exp : float, default=0.991
        Minimum exponent value used to determine the longest acceptably linear region.
    min_Npts : int, default=10
        Minimum number of points in the "best" region outputted.
    skip : int, default=1
        Number of points to skip over in scanning different regions. A value of 1 will be most thorough, but a larger value will decrease computation time.
    min_R2 : float, default=0.97
        Minimum allowed coefficient of determination to consider proposed exponent. This prevents linearity from skipping over curves.
    fit_limits : tuple, default=(None,None)
        Choose the indices to frame the area from which to estimate the viscosity. This window tends to be less than 100 ps. This will cut down on comutational time in spending time on regions with poor statistics.
    show_plot : bool, default=False
        choose to show a plot of the fit
    save_plot : bool, default=False
        choose to save a plot of the fit
    title : str, default=None
        The title used in the cumulative_integral plot, note that this str is also added as a prefix to the ``plot_name``.
    plot_name : str, default="einstein_viscosity.png"
        If ``save_plot==True`` the cumulative_integral will be saved with the debye-waller factor marked, The ``title`` is added as a prefix to this str
    verbose : bool, default=False
        Will print intermediate values or not
    fit_kwargs : dict, default={}
        Keyword arguements for exponential functions in :mod:`md_spa.custom_fit`

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
    for npts in range(min_Npts,len(time),skip):
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
    verbose : bool, default=False
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

