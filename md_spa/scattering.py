
import numpy as np
import matplotlib.pyplot as plt
import warnings

from md_spa_utils import file_manipulation as fm
from md_spa_utils import data_manipulation as dm

from .cython_modules import _scattering_functions as scat
from . import misc_functions as mf
from . import read_lammps as rl
from . import custom_fit as cfit


def scattering_length(elements, natoms, scattering_type):
    """ Output the cross Section Weighting for each particle

    Parameters
    ----------
    elements : list[str]
        List of atom elements symbols from which to pull the atomic form factor. 
        Note that an isotope may be chosen by preceeding the elemental symbol with the 
        number of neutrons. If None, f_values of one are used.
    natoms : int
        Number of atoms in the system. This should be the same length as ``elements``
    scattering_type : str
        May be coherent or incoherent
    
    Returns
    -------
    f_values : numpy.ndarray
        Array of the same length as elements containing the scattering cross sectional area
    """

    elements = None # Use until file is operational

    if elements is None:
        f_values = np.array([1 for x in range(natoms)], dtype=float)
    else:
        filename = "/Users/jac16/bin/md_spa/dat/scattering_lengths_cross_sections.dat"
        atom_scattering = fm.csv2dict(filename)
        if scattering_type == "coherent":
            key_f = "Coh b"
        else:
            key_f = "Inc b"
        f_values = np.array([atom_scattering[x][key_f] for x in elements], dtype=float)

    return f_values

def finite_size_correction(q_value, NR, natoms, dims, R=None):
    """ Calulate the correction for finite size effects

    This factor was presented for the "scaler" or "r-space presentation" of the
    structure factor and intermediate scattering function.

    See 10.1103/PhysRevE.53.2382, 10.1103/physreve.53.2390, and 10.1103/PhysRevE.64.051201
    for more details.

    Parameters
    ----------
    q_value : float/numpy.ndarray
        The qvalue between 2*np.pi/(L/2) and 2*np.pi/sigma
    NR : float/numpy.ndarray
        Output from :func:`isotropic_weighted_coherent_distances` where q=0
    natoms : int
        Number of atoms in the system to calculate the number density
    dims : numpy.ndarray
        Maximum coordinate values in the box representing the box lengths of length equal to the number of 
        dimensions. This requires that the lowest box values be located at the origin
    R : float, Optional, default=None
        Spherical cutoff value. This should rarely be set unless a q_value less than 2*np.pi/(L/2) is used.        

    Returns
    -------
    correction : float/numpy.ndarray
        Correction for finite size effects in the shape/size of q_value or NR

    """

    if R is None:
        R = np.min(dims)/2

    density = natoms / np.prod(dims)
    qR = q_value * R
    tmp = 4/3*np.pi*density*R**3
    S0 = (NR - tmp) / (1 - tmp/natoms) # Eq 23  DOI: 10.1103/PhysRevE.53.2382, Eq 7 10.1103/physreve.53.2390
    u_qR = 3/(R*q_value)**3 * (np.sin(qR) - qR*np.cos(qR)) # Eq 4d  DOI: 10.1103/PhysRevE.53.2382

    return tmp * u_qR * S0 / natoms # Eq 22 DOI: 10.1103/PhysRevE.53.2382

def n_particles_radius_R(q_value, natoms, dims, R=None):
    """ Calulate the average number of particles in the sphere of size R

    This factor was presented for the "scaler" or "r-space presentation" of the
    structure factor and intermediate scattering function.

    See 10.1103/PhysRevE.53.2382, 10.1103/physreve.53.2390, and 10.1103/PhysRevE.64.051201
    for more details.

    Parameters
    ----------
    q_value : float/numpy.ndarray
        The qvalue between 2*np.pi/(L/2) and 2*np.pi/sigma
    natoms : int
        Number of atoms in the system to calculate the number density
    dims : numpy.ndarray
        Maximum coordinate values in the box representing the box lengths of length equal to the number of 
        dimensions. This requires that the lowest box values be located at the origin
    R : float, Optional, default=None
        Spherical cutoff value. This should rarely be set unless a q_value less than 2*np.pi/(L/2) is used.        

    Returns
    -------
    correction : float/numpy.ndarray
        Average number of particles in region of radius R

    """

    if R is None:
        R = np.min(dims)/2

    density = natoms / np.prod(dims)
    qR = q_value * R
    u_qR = 3/qR**3 * (np.sin(qR) - qR*np.cos(qR))

    return 4/3*np.pi*density*R**3 * u_qR # Eq 4c DOI: 10.1103/PhysRevE.53.2382

def isotropic_weighted_incoherent_distances( displacements, f_values, q_values, dims, R=None):
    """ Distribution of distances weighted by isotropic expression with q-values

    See 10.1103/PhysRevE.53.2382, 10.1103/physreve.53.2390, and 10.1103/PhysRevE.64.051201
    for more details. See 10.1016/B978-012370535-8/50006-9 for discussion on the scattering lengths.

    Parameters
    ----------
    displacements : numpy.ndarray
        Atomic displacements. 2D array, the dimensions will be (nframes, natoms) where the 2nd
        dimension represents the displacement of an atom in time within distance ``R``.
    f_values : numpy.ndarray or list
        Array of the same length as elements containing the scattering cross sectional area or a list 
    q_value : float
        The qvalue between 2*np.pi/(L/2) and 2*np.pi/sigma
    dims : numpy.ndarray
        Maximum coordinate values in the box representing the box lengths of length equal to the number of 
        dimensions. This requires that the lowest box values be located at the origin
    R : float, Optional, default=None
        Spherical cutoff value. This should rarely be set unless a q_value less than 2*np.pi/(L/2) is used. 

    Returns
    -------
    X_q_t : numpy.ndarray
        This array may be used in the calulcation of structure factors or intermediate scattering
        functions. Can be (nframes,) or (nframes, nq)
        
    """

    if R is None:
        R = np.min(dims)/2

    if dm.isiterable(q_values):
        nq = len(q_values)
        if np == 1:
            q_values = q_values[0]
            nq = None
    else:
        nq = None

    disp_shape = np.shape(displacements)
    if len(disp_shape) != 2:
        ValueError("Provided displacement matrix must have the dimensions (nframes, natoms)")

    if nq:
        if disp_shape[1] != len(f_values):
            raise ValueError("Element and second dim of displacement values must be equal to natoms")
        qr = displacements[:, :, None] * q_values[None, None, :]
    else:
        qr = displacements * q_values

    weighting = np.sin(qr)/qr
    weighting[qr <= np.finfo(float).eps] = 1 # lim of sin(x)/x as x-> 0 equals 1

    avgf2 = np.nanmean(f_values**2)
    if nq is not None:
        weighting *= (f_values**2 / avgf2)[None, :, None] # (nframes, natoms, nq)
        weighted_average = np.nanmean(weighting, axis=-2) # (nframes, nq)
    else:
        weighting *= (f_values**2 / avgf2)[None, :] # (nframes, natoms)
        weighted_average = np.nanmean(weighting, axis=-1) # (nframes)

    return weighted_average

def isotropic_weighted_coherent_distances( displacements, f_reference, f_values, q_values, dims, R=None, include_self=True):
    """ Distribution of distances weighted by isotropic expression with q-values

    See 10.1103/PhysRevE.53.2382, 10.1103/physreve.53.2390, and 10.1103/PhysRevE.64.051201
    for more details. See 10.1016/B978-012370535-8/50006-9 for discussion on the scattering lengths.

    Note that self displacements are identified from displacements less than machine precision in the first
    frame. Thus, displacements must start in the first frame.

    Parameters
    ----------
    displacements : numpy.ndarray
        Atomic displacements. 2D array, the dimensions will be (nframes, natoms) where the 2nd
        dimension represents the centers of spheres and the third dimension is the displacement between that
        center and each atom within distance ``R``. If 2D, then the dimensions are for (nframes, natoms).
    f_reference : numpy.ndarray or list
        Scattering cross sectional area of reference atom
    f_values : numpy.ndarray or list
        Array of the same length as elements containing the scattering cross sectional area or a list 
    q_values : float/numpy.ndarray
        The qvalue between 2*np.pi/(L/2) and 2*np.pi/sigma
    dims : numpy.ndarray
        Maximum coordinate values in the box representing the box lengths of length equal to the number of 
        dimensions. This requires that the lowest box values be located at the origin
    R : float, Optional, default=None
        Spherical cutoff value. This should rarely be set unless a q_value less than 2*np.pi/(L/2) is used. 
    include_self : bool, Optional, default=True
        If False, the self displacements are discarded

    Returns
    -------
    X_q_t * mean(f_ref) : numpy.ndarray
        This array may be used in the calulcation of structure factors or intermediate scattering
        functions. Can be (nframes,) or (nframes, nq)
        
    """

    if R is None:
        R = np.min(dims)/2

    if dm.isiterable(q_values):
        nq = len(q_values)
        if np == 1:
            q_values = q_values[0]
            nq = None
    else:
        nq = None

    disp_shape = np.shape(displacements)
    if len(disp_shape) != 2:
        ValueError("Provided displacement matrix must have the dimensions (nframes, natoms)")

    if nq:
        if disp_shape[1] != len(f_values):
            raise ValueError("Element and second dim of displacement values must be equal to natoms")
        qr = displacements[:, :, None] * q_values[None, None, :]
    else:
        qr = displacements * q_values

    weighting = np.sin(qr)/qr
    weighting[qr <= np.finfo(float).eps] = 1 # lim of sin(x)/x as x-> 0 equals 1
    if not include_self:
        ind_self = np.where(displacements[0] < np.finfo(float).eps)[0]
        weighting[:,ind_self] = np.nan

    avgf2 = np.nanmean(f_values[:, None] * f_values[None, :])
    if nq is not None:
        weighting *= f_values[None, :, None]*f_reference / avgf2 # (nframes, natoms, nq)
        weighted_average = np.nansum(weighting, axis=-2) # (nframes, nq)
    else:
        weighting *= f_values[None, :]*f_reference / avgf2 # (nframes, natoms)
        weighted_average = np.nansum(weighting, axis=-1) # (nframes)

    return weighted_average

def static_structure_factor(traj, dims, elements=None, sigma=1, kwargs_linspace={}, flag="python"):
    """ Calculate the isotropic static structure factor

    DOI: 10.1063/5.0074588, 10.1103/physreve.53.2390

    Parameters
    ----------
    traj : numpy.ndarray
        Trajectory of atoms, where spacing in time doesn't matter. Dimensions are 
        (Nframes, Natoms, Ndims)
    dims : numpy.ndarray
        Maximum coordinate values in the box representing the box lengths of length equal to the number of 
        dimensions. This requires that the lowest box values be located at the origin
    elements : list, Optional, default=None
        List of atom elements symbols from which to pull the atomic form factor. 
        Note that an isotope may be chosen by preceeding the elemental symbol with the 
        number of neutrons. If None, f_values of one are used.
    sigma : float, Optional, default=1
        The particle size in the system used to define q_max = 2*np.pi/sigma
    kwargs_linspace : dict, Optional, default={"num":200}
        Keyword arguments for np.linspace function to produce q array to calculate the 
        structure factor 
    flag : str, Optional, default='python'
        Choose 'python' implementation or accelerated 'cython' option.

    Returns
    -------
    structure_factor : numpy.ndarray
        Array of the same length at the q array containing the isotropic static structure
        factor.
    q_array : numpy.ndarray
        q values generated from input options.

    """

    qmin = 2*np.pi/(np.max(dims)/2)
    qmax = 2*np.pi/(sigma/2)
    tmp_kwargs = {"num": 200}
    tmp_kwargs.update(kwargs_linspace)
    q_array = np.linspace(qmin,qmax,**tmp_kwargs, dtype=float)
    R = np.min(dims)/2

    nq = len(q_array)
    (nframes, natoms, ndims) = np.shape(traj)
    f_values = scattering_length(elements, natoms, "coherent")
    if not np.all(f_values == 1.0):
        raise ValueError("Corrections haven't been propagated to corrections")

    if flag == "cython":
        sq, sq_self = scat.static_structure_factor(traj, f_values, q_array, dims)
    else:
        s_n = np.zeros((nframes, nq))
        s_n_self = np.zeros((nframes, nq))
        NR = np.zeros(nframes)
        for j in range(natoms): # Must reduce dimensions to fit in memory
            disp = np.sqrt(np.sum(np.square(mf.check_wrap(traj-traj[:,j,:][:, None, :], dims)), axis=-1)) # nframe, natoms
            disp[disp > R] = np.nan
            s_n += isotropic_weighted_coherent_distances( disp, f_values[j], f_values, q_array, dims, R=R)
            NR += isotropic_weighted_coherent_distances( disp, 1.0, np.ones(natoms), 0.0, dims, R=R)
            s_n_self += isotropic_weighted_incoherent_distances( disp[:, j][:, None], f_values[j][None], q_array, dims, R=R)
        sq = np.nanmean(s_n, axis=0)/natoms/np.nanmean(f_values)**2 - n_particles_radius_R(q_array, natoms, dims, R=R)
        NR = np.nanmean(NR)/natoms/np.nanmean(f_values)
        sq += finite_size_correction(q_array, NR, natoms, dims, R=R)

        sq_self = np.ones(nq)/np.nanmean(f_values**2) - n_particles_radius_R(q_array, 1, dims, R=R)
        sq_self += finite_size_correction(q_array, 1, 1, dims, R=R)
      
    return sq, sq_self, q_array


def collective_intermediate_scattering(traj, dims, elements=None, q_value=2.25, flag="python", group_ids=None, include_self=True):
    """ Calculate the isotropic collective intermediate scattering function

    Taken from DOI: 10.1103/PhysRevE.64.051201

    Note that if the calculation for the whole system or specific atom types is desired without further
    breakdown into subcategories (e.g., hydration shells) consider a faster alternative `LiquidLab 
    <https://z-laboratory.github.io/LiquidLib/>`  

    Note that self displacements are identified from displacements less than machine precision in the first
    frame. Thus, the trajectory must start from t=0.

    Parameters
    ----------
    traj : numpy.ndarray
        Trajectory of atoms, usually log spaced by some base (e.g., 2) although this 
        choice doesn't affect the inner workings of this function. Dimensions are 
        (Nframes, Natoms, Ndims)
    dims : numpy.ndarray
        Maximum coordinate values in the box representing the box lengths of length equal to the number of 
        dimensions. This requires that the lowest box values be located at the origin
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
    include_self : bool, Optional, default=True
        If False, the self displacements are discarded leaving only the distinct signal

    Returns
    -------
    structure_factor : numpy.ndarray
        Array the isotropic self (coherent) intermediate scattering function.

    """

    R = np.min(dims)/2

    if q_value < 2*np.pi/R:
        ValueError("Given the box size, this q-value requires a correction. See doi: 10.1103/PhysRevE.64.051201 for more details")
    (nframes, natoms, ndims) = np.shape(traj)
    f_values = scattering_length(elements, natoms, "coherent")
    if not np.all(f_values == 1.0):
        raise ValueError("Corrections haven't been propagated to corrections")

    if flag == "cython":
        if group_ids is not None:
            raise ValueError("The use of the group_ids keyword is not yet supported with cython.")
        isf = scat.collective_intermediate_scattering(traj, f_values, q_value, dims, include_self)
    else:
        isf = np.zeros(nframes)
        NR = np.zeros(nframes)
        for j in range(natoms):
            disp = np.sqrt(np.sum(np.square(mf.check_wrap(traj[:,j,:][:, None, :]-traj[0,:,:][None, :, :], dims)),axis=-1))
            disp[disp > R] = np.nan
            isf += isotropic_weighted_coherent_distances( disp, f_values[j], f_values, q_value, dims, R=R, include_self=include_self)
            NR += isotropic_weighted_coherent_distances( disp, 1.0, np.ones(natoms), 0.0, dims, R=R, include_self=include_self)

        isf = isf/natoms/np.nanmean(f_values) - n_particles_radius_R(q_value, natoms, dims, R=R)
        NR = np.nanmean(NR)/natoms/np.nanmean(f_values[:,None]*f_values[None,:])
        isf += finite_size_correction(q_value, NR, natoms, dims, R=R)

        if group_ids is not None:
            group_pairs = [(i,y) for i,x in enumerate(group_ids) for y in range(len(group_ids))]
            group_combinations = [(x,y) for i,x in enumerate(group_ids) for y in group_ids]
            tmp_isf = isf.copy()
            isf = np.zeros((len(group_combinations)+1, len(isf)))
            isf[0] = tmp_isf
            for i, (ids1, ids2) in enumerate(group_combinations):
                if ids1 and ids2:
                    tmp_N = len(ids1)
                    tmp_isf = np.zeros(nframes)
                    NR = np.zeros(nframes)
                    for j in ids1:
                        disp = np.sqrt(np.sum(np.square(mf.check_wrap(traj[:,ids2,:]-traj[0,j,:][None, None, :], dims)),axis=-1))
                        disp[disp > R] = np.nan
                        tmp_isf += isotropic_weighted_coherent_distances( disp, f_values[j], f_values[ids2], q_value, dims, R=R, include_self=include_self)
                        NR += isotropic_weighted_coherent_distances( disp, 1.0, np.ones(len(ids2)), 0.0, dims, R=R, include_self=include_self)
                    tmp_isf = tmp_isf/tmp_N/np.mean(f_values[ids1])  - n_particles_radius_R(q_value, tmp_N, dims, R=R)
                    NR = np.nanmean(NR)/tmp_N//np.nanmean(f_values)
                    isf[i+1] = tmp_isf + finite_size_correction(q_value, NR, tmp_N, dims, R=R)
                 #   print(group_pairs[i], "\n", isf[i+1])
                else:
                    isf[i+1] = np.nan*np.ones(nframes)

    return np.array(isf)

def self_intermediate_scattering(traj, dims, elements=None, q_value=2.25, flag="python", group_ids=None, wrapped=True):
    """ Calculate the isotropic self intermediate scattering function

    Taken from DOI: 10.1103/PhysRevE.64.051201

    Note that if the calculation for the whole system or specific atom types is desired without further
    breakdown into subcategories (e.g., hydration shells) consider a faster alternative `LiquidLab 
    <https://z-laboratory.github.io/LiquidLib/>`  

    Parameters
    ----------
    traj : numpy.ndarray
        Trajectory of atoms, usually log spaced by some base (e.g., 2) although this 
        choice doesn't affect the inner workings of this function. Dimensions are 
        (Nframes, Natoms, Ndims)
    dims : numpy.ndarray
        Maximum coordinate values in the box representing the box lengths of length equal to the number of 
        dimensions. This requires that the lowest box values be located at the origin
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
    wrapped : bool, Optional, default=True
        Assume the coordinates are wrapped in determining displacement. Timeframes where a particle has traversed
        more than half a box length are set to nan.

    Returns
    -------
    structure_factor : numpy.ndarray
        Array the isotropic self (coherent) intermediate scattering function.

    """

    R = np.min(dims)/2

    if q_value < 2*np.pi/R:
        ValueError("Given the box size, this q-value requires a correction. See doi: 10.1103/PhysRevE.64.051201 for more details")
    (nframes, natoms, ndims) = np.shape(traj)
    f_values = scattering_length(elements, natoms, "incoherent")

    if flag == "cython":
        isf = scat.self_intermediate_scattering(traj, f_values, q_value, dims)
        if group_ids is not None:
            tmp_isf = isf.copy()
            isf = np.zeros((len(group_ids)+1, len(isf)))
            isf[0] = tmp_isf
            for i,tmp_ids in enumerate(group_ids):
                if tmp_ids:
                    tmp_isf = scat.self_intermediate_scattering( traj[:, tmp_ids, :], f_values[tmp_ids], q_value, dims)
                    isf[i+1] = tmp_isf
                else:
                    isf[i+1] = np.nan*np.ones(nframes)
    else:
        isf = np.zeros(nframes)
        if wrapped:
            disp = np.sqrt(np.sum(np.square(mf.check_wrap(traj-traj[0,:,:][None,:,:], dims)),axis=-1))
            check_max = np.array([len(np.where(x > R)[0]) for x in disp])
            if np.any(check_max > 0):
                warnings.warn("Self displacements beyond half the box size in frames: {}".format(np.where(check_max > 0)[0]))
        else:
            check_max = np.zeros(nframes)
        isf = isotropic_weighted_incoherent_distances( disp, f_values, q_value, dims, R=R)
        isf[check_max > 0] = np.nan

        if group_ids is not None:
            tmp_isf = isf.copy()
            isf = np.zeros((len(group_ids)+1, len(isf)))
            isf[0] = tmp_isf
            for i,tmp_ids in enumerate(group_ids):
                if tmp_ids:
                    tmp_N = len(tmp_ids)
                    if wrapped:
                        check_max = np.array([len(np.where(x > R)[0]) for x in disp[:, tmp_ids]])
                    else:
                        check_max = np.zeros(nframes)
                    tmp_isf = isotropic_weighted_incoherent_distances( disp[:, tmp_ids], f_values[tmp_ids], q_value, dims, R=R)
                    tmp_isf[check_max > 0] = np.nan
                    isf[i+1] = tmp_isf
                else:
                    isf[i+1] = np.nan*np.ones(nframes)

    return isf

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
    Gs : numpy.ndarray, list
        Self van Hove for matrix, if no ``group_ids``, the output is a matrix of the distance values by the 
        number of log spaced frames. If ``group_ids`` is not ``None``, then the output is a list of matrices
        for each group.

    """
    if flag == "cython":
        ValueError("Cython module is not yet available.")
    else:
        (nframes, natoms, ndims) = np.shape(traj)
        r_array = np.arange(0, r_max+dr, dr)
        nr = len(r_array)-1
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

def characteristic_time(xdata, ydata, minimizer="leastsq", verbose=False, save_plot=False, show_plot=False, plot_name="stretched_exponential_fit.png", kwargs_minimizer={}, ydata_min=0.1, n_exponentials=1, beta=None, weighting=None, collective=False):
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
    weighting : numpy.ndarray, Optional, default=None
        Of the same length as the provided data, contains the weights for each data point.
    collective : bool, Optional, default=False
        If True, the data will be normalized to the first value and plot axis labeled accordingly

    Returns
    -------
    parameters : numpy.ndarray
        Array containing parameters: ['tau', 'beta'] or ['A', 'tau1', 'beta1', 'tau2', 'beta2']
    stnd_errors : numpy.ndarray
        Array containing parameter standard errors from lmfit: ['tau'] or ['A', 'tau1', 'beta1', 'tau2', 'beta2']

    """

    if weighting is not None:
        weighting = weighting[ydata>0]
    xarray = xdata[ydata>0]
    yarray = ydata[ydata>0]
    if collective:
        if not np.isnan(yarray[0]):
            yarray /= yarray[0]
        elif not np.isnan(yarray[1]):
            yarray /= yarray[1]
        else:
            raise ValueError("Why are the first two values NaN?!")

    if np.size(yarray) == 0:
        if n_exponentials == 1:
            return np.nan*np.ones(2), np.nan*np.ones(2)
        elif n_exponentials == 2:
            return np.nan*np.ones(5), np.nan*np.ones(5)

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
            elif isinstance(beta, bool) and not beta:
                kwargs_parameters = {}
            else:
                raise ValueError("Beta must be an int or float")
        else:
            kwargs_parameters = {}
        output, uncertainties = cfit.stretched_exponential_decay(xarray, yarray,
                                                         verbose=verbose,
                                                         minimizer=minimizer,
                                                         kwargs_minimizer=kwargs_minimizer,
                                                         kwargs_parameters=kwargs_parameters,
                                                         weighting=weighting,
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
                                                      weighting=weighting,
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
        if not collective:
            ax[0].set_ylabel("$F_S(q,t)$")
            ax[1].set_ylabel("$F_S(q,t)$")
        else:
            ax[0].set_ylabel("$F_C(q,t)$")
            ax[1].set_ylabel("$F_C(q,t)$")
        ax[0].set_xlabel("Time")
        ax[0].set_xscale('log')
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

