# distutils: language=gcc

import cython
from cython.parallel import prange
import numpy as np
cimport numpy as np
from libc.math cimport sin, cos, sqrt, M_PI, isnan
from libc.stdio cimport printf

def analyze_clustering(cluster_traj, ncut):

        nframes, natoms = np.shape(cluster_traj)
        nclusters = int(np.nanmax(cluster_traj))
        nparticles = np.zeros(nframes, dtype=np.int32)
        maxclust_array = np.zeros(nframes, dtype=np.int32)
        indices = np.zeros(nframes, dtype=np.int32)
        cluster_output = np.zeros(( nframes, nclusters+1), dtype=np.int32)

        cdef double[:,:] cluster_view = cluster_traj
        cdef int[:,:] output_view = cluster_output
        cdef int[:] nparticles_view = nparticles
        cdef int[:] maxclust_array_view = maxclust_array
        cdef int[:] indices_view = indices
        cdef int nframes_view = nframes
        cdef int natoms_view = natoms
        cdef int nclusters_view = nclusters
        cdef double ncut_view = ncut
        cdef int max_clusters

        cluster_array = _compile_clusters(
            cluster_view, output_view, nframes_view, natoms_view
        )
        nparticles = _calc_nparticles(
            cluster_array, nparticles_view, ncut_view, nframes_view, nclusters_view
        )
        max_nclusters = _calc_max_nclusters(
            cluster_array, maxclust_array_view, ncut_view, nframes_view, nclusters_view
        )
        cluster_final0 = np.zeros((nframes,max_nclusters), dtype=np.int32)
        cdef int[:,:] cluster_final_view = cluster_final0
        cluster_final = _calc_clusters(
            cluster_array, cluster_final_view, indices_view, ncut_view, nframes_view, nclusters_view, max_nclusters
        )

        return np.asarray(cluster_final), np.asarray(nparticles)

@cython.boundscheck(False)
@cython.cdivision(True)
cdef int[:,:] _compile_clusters(
    double[:,:] clust_traj, 
    int[:,:] clust_output,
    int nframes,
    int natoms,
) nogil:

    cdef int i, j

    for i in prange(nframes, nogil=True):
        for j in range(natoms):
            if not isnan(clust_traj[i][j]) and clust_traj[i][j] > 0:
                clust_output[i][<int> clust_traj[i][j]] += 1

    return clust_output

@cython.boundscheck(False)
@cython.cdivision(True)
cdef int[:] _calc_nparticles(
    int[:,:] clust_array,
    int[:] nparticles,
    double ncut,
    int nframes,
    int nclusters,
) nogil:

    cdef int i, j

    for i in prange(nframes, nogil=True):
        for j in range(nclusters+1):
            if clust_array[i][j] < (ncut + 1) and clust_array[i][j] > 0:
                nparticles[i] += clust_array[i][j]

    return nparticles

@cython.boundscheck(False)
@cython.cdivision(True)
cdef int _calc_max_nclusters(
    int[:,:] clust_array,
    int[:] max_array,
    double ncut,
    int nframes,
    int nclusters,
) nogil:

    cdef int i, j, tmp_max, maxnclusters

    for i in prange(nframes, nogil=True):
        for j in range(nclusters+1):
            if clust_array[i][j] > ncut:
                max_array[i] += 1

    maxnclusters = 0
    for i in range(nframes):
        if max_array[i] > maxnclusters:
            maxnclusters = max_array[i]

    return maxnclusters

@cython.boundscheck(False)
@cython.cdivision(True)
cdef int[:,:] _calc_clusters(
    int[:,:] clust_array,
    int[:,:] cluster_final,
    int[:] indices,
    double ncut,
    int nframes,
    int nclusters,
    int maxclusters,
) nogil:

    cdef int i, j, jj

    for i in prange(nframes, nogil=True):
        for j in range(nclusters+1):
            if clust_array[i][j] > ncut:
                cluster_final[i][indices[i]] = clust_array[i][j]
                indices[i] += 1

    return cluster_final


