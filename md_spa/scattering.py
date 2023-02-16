
import numpy as np
import matplotlib.pyplot as plt

from md_spa_utils import file_manipulation as fm
from md_spa_utils import data_manipulation as dm

from .cython_modules import _scattering_functions as scat
from . import misc_functions as mf
from . import read_lammps as rl
from . import custom_fit as cfit


def static_structure_factor(traj, dims, elements=None, qmax=10, qmin=None, kwargs_linspace=None, flag="python"):
    """ Calculate the isotropic static structure factor

    Parameters
    ----------
    traj : numpy.ndarray
        Trajectory of atoms, where spacing in time doesn't matter. Dimensions are 
        (Nframes, Natoms, Ndims)
    dims : numpy.ndarray
        Maximum coordinate values in the box representing the box lengths. This
        requires that the lowest box values be located at the origin
    elements : list, Optional, default=None
        List of atom elements symbols from which to pull the atomic form factor. 
        Note that an isotope may be chosen by preceeding the elemental symbol with the 
        number of neutrons. If None, f_values of one are used.
    qmax : float, Optional, default=10
        The maximum qvalue in the array in inverse angstroms
    qmin : float, Optional, default=None
        If no value is given, half of the largest distance is taken.
    kwargs_linspace : dict, Optional, default={"num":1000}
        Keyword arguments for np.linspace function to produce q array to calculate the 
        structure factor 

    Returns
    -------
    structure_factor : numpy.ndarray
        Array of the same length at the q array containing the isotropic static structure
        factor.
    q_array : numpy.ndarray
        q values generated from input options.

    """

    if qmin is None:
        qmin = 1/np.max(dims)
    if kwargs_linspace is None:
        kwargs_linspace = {"num": 1000}
    q_array = np.linspace(qmin,qmax,**kwargs_linspace, dtype=float)

    ## Refactor to use internal data file
    #if elements is None:
    #    f_values = np.array([1 for x in range(traj[0])], dtype=float)
    #else:
    #    filename = "/Users/jac16/bin/md_spa/dat/scattering_lengths_cross_sections.dat"
    #    atom_scattering = fm.csv2dict(filename)
    #    key_f = "Coh b"
    #    f_values = np.array([atom_scattering[x][key_f] for x in elements], dtype=float)
    f_values = np.array([1 for x in range(traj[0])], dtype=float)

    traj = traj[:2] # NoteHere

    if flag == "cython":
        sq = scat.static_structure_factor(traj, f_values, q_array, dims)
    else:
        nq = len(q_array)
        (nframes, natoms, ndims) = np.shape(traj)
        sq = np.zeros(nq)
        for i in range(natoms):
            fi = f_values[i]
            displacements = np.sqrt(np.sum(np.square(mf.check_wrap(traj-traj[:,i,:][:,None,:], dims)),axis=-1))
            # displacements is an array Nframes, Natoms
            qr = np.einsum("ij,k->ijk", displacements, q_array)
            sq += np.nansum(fi*f_values[None,:,None]*np.sin(qr)/qr, axis=(0,1))
        cumf2 = np.sum(np.square(f_values))
        sq = sq/cumf2/nframes

    return sq, q_array


def isotropic_coherent_scattering(traj, elements=None, q_value=2.25, flag="python", group_ids=None):
    """ Calculate the isotropic coherent intermediate scattering function

    Parameters
    ----------
    traj : numpy.ndarray
        Trajectory of atoms, usually log spaced by some base (e.g., 2) although this 
        choice doesn't affect the inner workings of this function. Dimensions are 
        (Nframes, Natoms, Ndims)
    elements : list, Optional, default=None
        List of atom elements symbols from which to pull the atomic form factor. 
        Note that an isotope may be chosen by preceeding the elemental symbol with the 
        number of neutrons. If ``None``, f_values of one are used.
    q_value : float, Optional, default=2.25
        The qvalue at which to calculate the isotropic static structure factor. The default
        value of 2.25 inverse angstroms comes from DOI:10.1063/1.4941946
    group_ids : list[lists], Optional, default=None
        Optional list of group ids will produce the total isf, in addition to isf of other groups
        destinguished by the given ids.
    flag : str, Optional, default='python'
        Choose 'python' implementation or accelerated 'cython' option.

    Returns
    -------
    structure_factor : numpy.ndarray
        Array the isotropic self (coherent) intermediate scattering function.

    """

    ## Refactor to use internal data file
    #if elements is None:
    #    f_values = np.array([1 for x in range(traj[0])], dtype=float)
    #else:
    #    filename = "/Users/jac16/bin/md_spa/dat/scattering_lengths_cross_sections.dat"
    #    atom_scattering = fm.csv2dict(filename)
    #    key_f = "Coh b"
    #    f_values = np.array([atom_scattering[x][key_f] for x in elements], dtype=float)
    f_values = np.array([1 for x in range(traj[0])], dtype=float)

    if flag == "cython":
        if group_ids is not None:
            raise ValueError("The use of the group_ids keyword is not yet supported with cython.")
        isf = scat.isotropic_coherent_scattering(traj, f_values, q_value)
    else:
        (nframes, natoms, ndims) = np.shape(traj)
        isf = np.zeros(nframes)
        qr = np.sqrt(np.sum(np.square(traj-traj[0,:,:][None,:,:]),axis=-1))*q_value
        isf = np.nanmean(np.square(f_values)[None,:] * np.sin( qr ) / ( qr ), axis=1)
        #cumf2 = np.sum(np.square(f_values))
        cumf2 = 1
        isf /= cumf
        if group_ids is not None:
            isf = [isf]
            for tmp_ids in group_ids:
                tmp_isf = np.nanmean(np.square(f_values[tmp_ids])[None,:] * np.sin( qr[:,tmp_ids] ) / ( qr[:,tmp_ids] ), axis=1)
                #cumf2 = np.sum(np.square(f_values[tmp_ids]))
                cumf2 = 1
                tmp_isf /= cumf2
                isf.append(tmp_isf)

    return np.array(isf)

def self_van_hove(traj, r_max=7.0, dr=0.1, flag="python", group_ids=None):
    """ Calculate the self van Hove equation.

    Parameters
    ----------
    traj : numpy.ndarray
        Trajectory of atoms, usually log spaced by some base (e.g., 2) although this 
        choice doesn't affect the inner workings of this function. Dimensions are 
        (Nframes, Natoms, Ndims)
    
    group_ids : list[lists], Optional, default=None
        Optional list of group ids will produce the total isf, in addition to isf of other groups
        destinguished by the given ids.
    flag : str, Optional, default='python'
        Not yet enabled! Choose 'python' implementation or accelerated 'cython' option.

    Returns
    -------
    structure_factor : numpy.ndarray
        Array the isotropic self (coherent) intermediate scattering function.

    """
    if flag == "cython":
        ValueError("Cython module is not yet available.")
    else:
        (nframes, natoms, ndims) = np.shape(traj)
        r_array = np.arange(r_max+dr, dr)
        nr = len(r_array)
        displacements = np.sqrt(np.sum(np.square(traj-traj[0,:,:][None,:,:]),axis=-1))
        gs = np.zeros((nr, nframes+1))
        gs[:,0] = r_array[:-1]+dr/2
        for i, disp in enumerate(displacements):
            gs[:, i+1] = np.histogram(disp, bins=r_array)[0]/natoms

        if group_ids is not None:
            gs = [gs]
            for tmp_ids in group_ids:
                tmp_gs = np.zeros((nr, nframes+1))
                tmp_gs[:,0] = r_array[:-1]+dr/2
                for i, disp in enumerate(displacements):
                    tmp_gs[:, i+1] = np.histogram(disp[tmp_ids], bins=r_array)[0]/len(tmp_ids)
                gs.append(tmp_gs)

    return np.array(gs)

def characteristic_time(xdata, ydata, minimizer="leastsq", verbose=False, save_plot=False, show_plot=False, plot_name="stretched_exponential_fit.png", kwargs_minimizer={}, ydata_min=0.1, n_exponentials=1, beta=None):
    """
    Extract the characteristic time fit to a stretched exponential or two stretched exponential

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
    n_exponentials : int, Optional, default=1
        Can be 1 or 2, if the latter, the default exponent for the first exponential is the phenomenological form of 3/2 (DOI: 10.1007/978-3-030-60443-1_5) unless ``beta`` is defined.
    beta : float, Optional, default=None
        Exponent of the stretched exponential decay. If ``n_exponential == 2 and beta == None`` then ``beta = 3/2`` is used. Set beta as an iterable of length two to set both exponents when appropriate. Set ``beta == False`` to remove the set value of 3/2.

    Returns
    -------
    parameters : numpy.ndarray
        Array containing parameters: ['tau', 'beta'] or ['A', 'tau1', 'beta1', 'tau2', 'beta2']
    stnd_errors : numpy.ndarray
        Array containing parameter standard errors from lmfit: ['tau'] or ['A', 'tau1', 'beta1', 'tau2', 'beta2']

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

    if n_exponentials == 1:
        if beta is not None:
            if isinstance(beta, (int,float)) and not isinstance(beta, bool):
                kwargs_parameters={"beta": {"value": beta, "vary": False}}
            else:
                raise ValueError("Beta must be an int or float")
        else:
            kwargs_parameters = {}
        output, uncertainties = cfit.stretched_exponential_decay(xarray, yarray,
                                                         verbose=verbose,
                                                         minimizer=minimizer,
                                                         kwargs_minimizer=kwargs_minimizer,
                                                         kwargs_parameters=kwargs_parameters,
                                                        )
    elif n_exponentials == 2:
        if beta is not None:
            if isinstance(beta, (int,float)) and not isinstance(beta, bool):
                kwargs_parameters={"beta1": {"value": beta, "vary": False}}
            elif dm.isiterable(beta) and len(beta) <= 2:
                if len(beta) == 1:
                    kwargs_parameters={"beta1": {"value": beta[0], "vary": False}}
                else:
                    kwargs_parameters={
                        "beta1": {"value": beta[0], "vary": False},
                        "beta2": {"value": beta[1], "vary": False},
                    }
            elif isinstance(beta, bool) and not beta:
                kwargs_parameters = {}
            else:
                raise ValueError("Beta must be an int, float, or iterable of length <= 2")
        else:
            kwargs_parameters={"beta1": {"value": 3/2, "vary": False}}
        # Make function to guess starting parameters based on the derivative of spline from data?
        output, uncertainties = cfit.two_stretched_exponential_decays(xarray, yarray,
                                                      verbose=verbose,
                                                      minimizer=minimizer,
                                                      kwargs_minimizer=kwargs_minimizer,
                                                      kwargs_parameters=kwargs_parameters,
                                                     )
        if output[1] > output [3]:
            output[0] = 1 - output[0]
            tmp_output = output[1:3].copy()
            output[1:3] = output[3:]
            output[3:] = tmp_output
            tmp_uncertainties = uncertainties[1:3].copy()
            uncertainties[1:3] = uncertainties[3:]
            uncertainties[3:] = tmp_uncertainties
    else:
        raise ValueError("Only 1 or 2 stretched exponentials are supported")


    if save_plot or show_plot:
        fig, ax = plt.subplots(1,2, figsize=(6,3))
        if n_exponentials == 1:
            yfit = np.exp(-(xarray/output[0])**output[1])
            ax[0].plot(xarray,yarray,".",label="Data")
            ax[0].plot(xarray,yfit, label="Fit",linewidth=1)
            ax[1].plot(xarray,yarray,".",label="Data")
            ax[1].plot(xarray,yfit, label="Fit",linewidth=1)
        else:
            params = {key: value for key, value in zip(['A', 'tau1', 'beta1', 'tau2', 'beta2'], output)}
            yfit = cfit._res_two_stretched_exponential_decays(params, xarray, 0.0)
            ax[0].plot(xarray,yarray,".",label="Data")
            ax[0].plot(xarray,yfit, label="Fit",linewidth=1)
            ax[1].plot(xarray,yarray,".",label="Data")
            ax[1].plot(xarray,yfit, label="Fit",linewidth=1)
        ax[0].set_ylabel("$F_S(q,t)$")
        ax[0].set_xlabel("Time")
        ax[0].set_xscale('log')
        ax[1].set_ylabel("$F_S(q,t)$")
        ax[1].set_xlabel("Time")
        ax[1].set_xscale('log')
        ax[1].set_yscale('log')
        ax[1].legend(loc="best")
        plt.tight_layout()
        if save_plot:
            plt.savefig(plot_name,dpi=300)

        if show_plot:
            plt.show()
        plt.close("all")

    return output, uncertainties

