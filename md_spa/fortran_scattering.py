
import numpy as np
import matplotlib.pyplot as plt
import warnings

from md_spa_utils import file_manipulation as fm
from md_spa_utils import data_manipulation as dm

from .fortran_modules import intermediate_scattering 
from .fortran_modules import static_structure_factor
from . import misc_functions as mf
from . import read_lammps as rl
from . import custom_fit as cfit

natoms = 10054
#natoms = 300000
dummy_args = (np.zeros(natoms), np.zeros(natoms), np.zeros(natoms))

def total_static_structure_factor(traj, dims):
    """ Calculate the isotropic static structure factor

    Take from Alexandros Chremos

    Parameters
    ----------
    traj : numpy.ndarray
        Trajectory of atoms, where spacing in time doesn't matter. Dimensions are 
        (Nframes, Natoms, Ndims)

    """

    static_structure_factor.structure_factor(0, dims[0], natoms, *dummy_args)
    for frame in traj:
        static_structure_factor.structure_factor(1, dims[0], natoms, frame[:,0], frame[:,1], frame[:,2])
    static_structure_factor.structure_factor(2, dims[0], natoms, *dummy_args)


def total_intermediate_scattering(traj, dims):
    """ Calculate the isotropic collective intermediate scattering function

    Take from Alexandros Chremos

    Parameters
    ----------
    traj : numpy.ndarray
        Trajectory of atoms, usually log spaced by some base (e.g., 2) although this 
        choice doesn't affect the inner workings of this function. Dimensions are 
        (Nframes, Natoms, Ndims)

    """

    intermediate_scattering.intermediate_scattering(*dummy_args, natoms, dims[0], 0)
    for frame in traj:
        intermediate_scattering.intermediate_scattering(frame[:,0], frame[:,1], frame[:,2], natoms, dims[0], 1)
    intermediate_scattering.intermediate_scattering(*dummy_args, natoms, dims[0], 2)

