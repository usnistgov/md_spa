# distutils: language=gcc

import cython
from cython.parallel import prange
import numpy as np
cimport numpy as np
from libc.math cimport sin, cos, sqrt, M_PI, isnan
from libc.stdio cimport printf

def static_structure_factor(traj, f_values, q_array, dims):

        if traj.shape[1] != f_values.shape[0]:
            raise ValueError("The number of atoms in `traj` does not equal the number of atoms in `f_values`")
        if traj.shape[2] != dims.shape[0]:
            raise ValueError("The number of dimensions in `traj` does not equal the number of entries in `dims`")

        sq0 = np.zeros(len(q_array))
        sqself = np.zeros(len(q_array))
        nframes, natoms, ndims = np.shape(traj)
        NR_values = np.zeros(natoms)
        nq = len(q_array)

        cdef double[:,:,:] traj_view = traj
        cdef double[:] f_values_view = f_values
        cdef double[:] NR_values_view = NR_values
        cdef double[:] q_array_view = q_array
        cdef double[:] dims_view = dims
        cdef double[:] sq0_view = sq0
        cdef double[:] sqself_view = sqself
        cdef int nframes_view = nframes
        cdef int natoms_view = natoms
        cdef int ndims_view = ndims
        cdef int nq_view = nq

        sq = _structure_factor(
            traj_view, f_values_view, NR_values_view, q_array_view, dims_view, sq0_view, nframes_view, natoms_view, ndims_view, nq_view
        )
        sq_self = _self_structure_factor(
            traj_view, f_values_view, q_array_view, dims_view, sqself_view, nframes_view, natoms_view, ndims_view, nq_view
        )

        return np.asarray(sq), np.asarray(sq_self)

def self_intermediate_scattering(traj, f_values, q_value, dims):

        if traj.shape[1] != f_values.shape[0]:
            raise ValueError("The number of atoms in `traj` does not equal the number of atoms in `f_values`")

        isf0 = np.zeros(traj.shape[0])
        count = np.zeros(traj.shape[0], dtype=np.int32)
        (nframes, natoms, ndims) = np.shape(traj)

        cdef double[:,:,:] traj_view = traj
        cdef double[:] f_values_view = f_values
        cdef double q_value_view = q_value
        cdef double[:] dims_view = dims
        cdef double[:] isf0_view = isf0
        cdef int[:] count_view = count
        cdef int nframes_view = nframes
        cdef int natoms_view = natoms
        cdef int ndims_view = ndims

        isf = _self_intermediate_scattering(
            traj_view, f_values_view, q_value_view, dims_view, isf0_view, count_view, nframes_view, natoms_view, ndims_view
        )

        return np.asarray(isf)

def collective_intermediate_scattering(traj, f_values, q_value, dims, include_self):

        if traj.shape[1] != f_values.shape[0]:
            raise ValueError("The number of atoms in `traj` does not equal the number of atoms in `f_values`")
        if traj.shape[2] != dims.shape[0]:
            raise ValueError("The number of dimensions in `traj` does not equal the number of entries in `dims`")

        isf0 = np.zeros(traj.shape[0])
        nframes, natoms, ndims = np.shape(traj)
        NR_values = np.zeros(natoms)

        cdef double[:,:,:] traj_view = traj
        cdef double[:] f_values_view = f_values
        cdef double[:] NR_values_view = NR_values
        cdef double q_value_view = q_value
        cdef double[:] dims_view = dims
        cdef double[:] isf0_view = isf0
        cdef int nframes_view = nframes
        cdef int natoms_view = natoms
        cdef int ndims_view = ndims
        cdef bint flag_view = include_self 

        isf = _collective_intermediate_scattering(
            traj_view, f_values_view, NR_values_view, q_value_view, dims_view, isf0_view, nframes_view, natoms_view, ndims_view, flag_view
        )

        return np.asarray(isf)

@cython.boundscheck(False)
@cython.cdivision(True)
cdef double[:] _structure_factor(
    double[:,:,:] traj, 
    double[:] f_values, 
    double[:] NR,
    double[:] q_array,
    double[:] dims,
    double[:] sq,
    int nframes,
    int natoms,
    int ndims,
    int nq,
) nogil:

    cdef double avgf2
    cdef double tmp
    cdef double tmp2
    cdef double disp
    cdef double R
    cdef double NRavg
    cdef int image
    cdef int i, j, k, l, n

    R = dims[0] / 2

    avgf2 = 0.0
    for i in range(natoms): 
        for j in range(natoms):
            avgf2 = avgf2 + f_values[i] * f_values[j]
    avgf2 = avgf2 / natoms**2

    for i in prange(nframes, nogil=True):
        for j in range(natoms):
            for k in range(natoms):
                disp = 0.0
                for n in range(ndims):
                    tmp = traj[i,j,n] - traj[i,k,n] 
                    image = <int> ((dims[n]/2 - tmp) // dims[n])
                    disp = disp + (tmp + image * dims[n])**2
                disp = sqrt(disp)
                if disp < R:
                    for l in range(nq):
                        tmp = f_values[j] * f_values[k] *  sin( q_array[l] * disp ) / ( q_array[l] * disp )
                        if isnan(tmp):
                            sq[l] = sq[l] + 1
                        else:
                            sq[l] = sq[l] + tmp
                    NR[j] = NR[j] + 1

    NRavg = 0.0
    for i in range(natoms):
        NRavg = NRavg + NR[i]
    NRavg = NRavg / nframes / natoms / avgf2
    for l in prange(nq, nogil=True):
        tmp = _n_particles_radius_R(q_array[l], natoms, dims, R)
        tmp2 = _finite_size_correction(q_array[l], NRavg, natoms, dims, R)
        sq[l] = sq[l] / avgf2 / nframes / natoms - tmp + tmp2

    return sq


@cython.boundscheck(False)
@cython.cdivision(True)
cdef double[:] _self_structure_factor(
    double[:,:,:] traj,
    double[:] f_values,
    double[:] q_array,
    double[:] dims,
    double[:] sq,
    int nframes,
    int natoms,
    int ndims,
    int nq,
) nogil:

    cdef double avgf2
    cdef double tmp
    cdef double tmp2
    cdef double disp
    cdef double R
    cdef int image
    cdef int i, j, k, l, n

    R = dims[0] / 2

    avgf2 = 0.0
    for i in range(natoms):
        avgf2 = avgf2 + f_values[i] * f_values[i]
    avgf2 = avgf2 / natoms

    for l in prange(nq, nogil=True):
        tmp = _n_particles_radius_R(q_array[l], 1, dims, R)
        tmp2 = _finite_size_correction(q_array[l], 1, 1, dims, R)
        sq[l] = 1 / avgf2 - tmp + tmp2 

    return sq


@cython.boundscheck(False)
@cython.cdivision(True)
cdef double _n_particles_radius_R(
    double q_value,
    int natoms,
    double[:] dims,
    double R,
) nogil:

    cdef double density
    cdef double qR
    cdef double u_qR
    cdef double correction

    density = natoms / (dims[0] * dims[1] * dims[2])
    qR = q_value * R
    u_qR = 3/(qR)**3 * ( sin(qR) - qR * cos(qR))
    correction = 4.0 / 3.0 * M_PI * density * R**3 * u_qR

    return correction


@cython.boundscheck(False)
@cython.cdivision(True)
cdef double _finite_size_correction(
    double q_value,
    double NR,
    int natoms,
    double[:] dims,
    double R,
) nogil:

    cdef double density
    cdef double qR
    cdef double u_qR
    cdef double tmp
    cdef double S0
    cdef double correction

    density = natoms / (dims[0] * dims[1] * dims[2])
    qR = q_value * R
    u_qR = 3/(qR)**3 * ( sin(qR) - qR * cos(qR) )
    tmp = 4.0 / 3.0 * M_PI * density * R**3

    S0 = (NR - tmp) / (1 - tmp/natoms)
    correction = tmp * u_qR * S0 / natoms

    return correction


@cython.boundscheck(False)
@cython.cdivision(True)
cdef double[:] _self_intermediate_scattering(
    double[:,:,:] traj,
    double[:] f_values,
    double q_value,
    double[:] dims,
    double[:] isf,
    int[:] count,
    int nframes,
    int natoms,
    int ndims,
) nogil:

    cdef float avgf2
    cdef float tmp
    cdef float disp
    cdef double R
    cdef int image
    cdef int i, j, n

    avgf2 = 0.0
    for j in range(natoms):
        avgf2 = avgf2 + f_values[j]**2
    avgf2 = avgf2 / natoms 

    R = dims[0] / 2

    for i in prange(nframes, nogil=True):
        for j in range(natoms):
            disp = 0.0
            for n in range(ndims):
                disp = disp + ( traj[i,j,n] - traj[0,j,n] )**2
            disp = sqrt(disp)
            if disp < R:
                tmp = f_values[j]**2 *  sin( q_value * disp ) / ( q_value * disp )
                if isnan(tmp):
                    isf[i] = isf[i] + 1
                else:
                    isf[i] = isf[i] + tmp
            else:
                count[i] = count[i] + 1
        isf[i] = isf[i] / ( avgf2 * ( natoms - count[i] ) )
        if count[i] > 0:
            printf("Frame %i has %i atoms that traveled more than half the box length\n", i, count[i])

    return isf

@cython.boundscheck(False)
@cython.cdivision(True)
cdef double[:] _collective_intermediate_scattering( 
    double[:,:,:] traj,
    double[:] f_values,
    double[:] NR,
    double q_value,
    double[:] dims,
    double[:] isf,
    int nframes,
    int natoms,
    int ndims,
    bint include_self,
) nogil:

    cdef double avgf2
    cdef double tmp
    cdef double tmp2
    cdef double disp
    cdef double NRavg
    cdef double R
    cdef int image
    cdef int i, j, k, l, n

    R = dims[0] / 2

    avgf2 = 0.0
    for i in range(natoms):
        for j in range(natoms):
            avgf2 = avgf2 + f_values[i] * f_values[j]
    avgf2 = avgf2 / natoms**2

    for i in prange(nframes, nogil=True):
        for j in range(natoms):
            for k in range(natoms):
                if (j != k  or include_self):
                    disp = 0.0
                    for n in range(ndims):
                        tmp = traj[i,j,n] - traj[0,k,n]
                        image = <int> ((dims[n]/2 - tmp) // dims[n])
                        disp = disp + (tmp + image * dims[n])**2
                    disp = sqrt(disp)
                    if disp < R:
                        NR[j] = NR[j] + 1.0
                        tmp = f_values[j] * f_values[k] *  sin( q_value * disp ) / ( q_value * disp )
                        if isnan(tmp):
                            isf[i] = isf[i] + 1
                        else:
                            isf[i] = isf[i] + tmp

        isf[i] = isf[i] / ( avgf2 * natoms )

    NRavg = 0.0
    for i in range(natoms):
        NRavg = NRavg + NR[i]
    NRavg = NRavg / nframes / natoms / avgf2
    tmp = _n_particles_radius_R(q_value, natoms, dims, R)
    tmp2 = _finite_size_correction(q_value, NRavg, natoms, dims, R)
    for i in range(nframes):
        isf[i] = isf[i] - tmp + tmp2

    return isf


