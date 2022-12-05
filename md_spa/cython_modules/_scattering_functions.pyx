# distutils: language=gcc

import cython
import numpy as np
cimport numpy as np
from libc.math cimport sin, sqrt

def static_structure_factor(traj, f_values, q_array, dims):

        if traj.shape[1] != f_values.shape[0]:
            raise ValueError("The number of atoms in `traj` does not equal the number of atoms in `f_values`")
        if traj.shape[2] != dims.shape[0]:
            raise ValueError("The number of dimensions in `traj` does not equal the number of entries in `dims`")

        sq0 = np.zeros(len(q_array))
        sq = _structure_factor(traj, f_values, q_array, dims, sq0)

        return sq

def isotropic_coherent_scattering(traj, f_values, q_value):

        if traj.shape[1] != f_values.shape[0]:
            raise ValueError("The number of atoms in `traj` does not equal the number of atoms in `f_values`")

        isf0 = np.zeros(traj.shape[0])
        isf = _isotropic_coherent_scattering(traj, f_values, q_value, isf0)

        return isf

@cython.boundscheck(False)
@cython.cdivision(True)
cdef _structure_factor( double[:,:,:] traj, 
                             double[:] f_values, 
                             double[:] q_array,
                             double[:] dims,
                             double[:] sq,
                           ):

    cdef int nframes = traj.shape[0]
    cdef int natoms = traj.shape[1]
    cdef int ndims = traj.shape[2] 
    cdef int nq = q_array.shape[0]

    cdef double cumf2
    cdef double tmp
    cdef double disp
    cdef int image
    cdef int i, j, k, l, n

    cumf2 = 0.0
    for j in range(natoms):
        cumf2 = cumf2 + f_values[j]**2

    for i in range(nframes):
        for j in range(natoms):
            for k in range(natoms):
                if j != k:
                    disp = 0.0
                    for n in range(ndims):
                        tmp = traj[i,j,n] - traj[i,k,n] 
                        image = <int> ((dims[n]/2 - tmp) // dims[n])
                        disp = disp + (tmp + image * dims[n])**2
                    disp = sqrt(disp)
                    for l in range(nq):
                        sq[l] = sq[l] + f_values[j] * f_values[k] *  sin( q_array[l] * disp ) / ( q_array[l] * disp )/cumf2/nframes

    return sq


@cython.boundscheck(False)
@cython.cdivision(True)
cdef _isotropic_coherent_scattering(double[:,:,:] traj,
                                    double[:] f_values,
                                    double q_value,
                                    double[:] isf,
                                  ):

    cdef int nframes = traj.shape[0]
    cdef int natoms = traj.shape[1]
    cdef int ndims = traj.shape[2]

    cdef float cumf2
    cdef float tmp
    cdef float disp
    cdef int image
    cdef int i, j, n

    #cumf2 = 0
    #for j in range(natoms):
    #    cumf2 = cumf2 + f_values[j]**2
    cumf2 = 1.0

    for i in range(1,nframes):
        for j in range(natoms):
            disp = 0.0
            for n in range(ndims):
                disp = disp + ( traj[i,j,n] - traj[0,j,n] )**2
            disp = sqrt(disp)
            isf[i] = isf[i] + f_values[j]**2 *  sin( q_value * disp ) / ( q_value * disp ) / ( cumf2 * natoms )

    return isf

