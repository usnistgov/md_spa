
import numpy as np

from cython_modules import _scattering_functions as scat

from md_spa import misc_functions as mf
from md_spa import read_lammps as rl
from md_spa_utils import file_manipulation as fm

def static_structure_factor(traj, dims, elements, qmax=10, qmin=None, kwargs_linspace=None, flag="python"):
    """ Calculate the isotropic static structure factor

    Parameters
    ----------
    traj : numpy.ndarray
        Trajectory of atoms, usually log spaced by some base (e.g., 2) although this 
        choice doesn't affect the inner workings of this function. Dimensions are 
        (Nframes, Natoms, Ndims)
    dims : numpy.ndarray
        Maximum coordinate values in the box representing the box lengths. This
        requires that the lowest box values be located at the origin
    elements : list
        List of atom elements symbols from which to pull the atomic form factor. 
        Note that an isotope may be chosen by preceeding the elemental symbol with the 
        number of neutrons.
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
    #filename = "/Users/jac16/bin/md_spa/dat/scattering_lengths_cross_sections.dat"
    #atom_scattering = fm.csv2dict(filename)
    #key_f = "Coh b"
    #f_values = np.array([atom_scattering[x][key_f] for x in elements], dtype=float)
    f_values = np.array([1 for x in elements], dtype=float)

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


def isotropic_coherent_scattering(traj, elements, q_value=2.25, flag="python", group_ids=None):
    """ Calculate the isotropic static structure factor

    Parameters
    ----------
    traj : numpy.ndarray
        Trajectory of atoms, usually log spaced by some base (e.g., 2) although this 
        choice doesn't affect the inner workings of this function. Dimensions are 
        (Nframes, Natoms, Ndims)
    elements : list
        List of atom elements symbols from which to pull the atomic form factor. 
        Note that an isotope may be chosen by preceeding the elemental symbol with the 
        number of neutrons.
    q_value : float, Optional, default=2.25
        The qvalue at which to calculate the isotropic static structure factor. The default
        value of 2.25 inverse angstroms comes from DOI:10.1063/1.4941946
    group_ids : list[lists], Optional, default=None
        Optional list of group ids will produce the total isf, in addition to isf of other groups
        destinguished by the given ids.

    Returns
    -------
    structure_factor : numpy.ndarray
        Array the isotropic self (coherent) intermediate scattering function.

    """

    ## Refactor to use internal data file
    #filename = "/Users/jac16/bin/md_spa/dat/scattering_lengths_cross_sections.dat"
    #atom_scattering = fm.csv2dict(filename)
    #key_f = "Coh b"
    #f_values = np.array([atom_scattering[x][key_f] for x in elements], dtype=float)
    f_values = np.array([1 for x in elements], dtype=float)

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
        isf /= cumf2
        if group_ids is not None:
            isf = [isf]
            for tmp_ids in group_ids:
                print(len(isf),len(isf[-1]),tmp_ids)
                tmp_isf = np.nanmean(np.square(f_values)[None,tmp_ids] * np.sin( qr[:,tmp_ids] ) / ( qr[:,tmp_ids] ), axis=1)
                #cumf2 = np.sum(np.square(f_values[tmp_ids]))
                cumf2 = 1
                tmp_isf /= cumf2
                isf.append(tmp_isf)
                print(len(isf),len(isf[-1]))

    return np.array(isf)
