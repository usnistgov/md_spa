# distutils: language=gcc

import cython
from cython.parallel import prange
import numpy as np
cimport numpy as np
from libc.math cimport sin, cos, sqrt, M_PI, isnan
from libc.stdio cimport printf

def analyze_clustering(cluster_traj, ncut):

        nframes, natoms = np.shape(cluster_traj)
        nclusters = int(np.nanmax(cluster_array))
        nparticles = np.zeros(nframes)
        cluster_output = np.nan*np.ones(( nframes, nclusters))
        
        cdef int[:,:] cluster_view = cluster_traj
        cdef int[:,:] output_view = cluster_output
        cdef int nframes_view = nframes
        cdef int natoms_view = natoms

        cluster_array = np.asarray(_calc_clusters(
            cluster_view, output_view, nframes_view, natoms_view
        ))

        return cluster_array

@cython.boundscheck(False)
@cython.cdivision(True)
cdef int[:,:] _calc_clusters(
    int[:,:] clust_traj, 
    int[:,:] clust_output,
    int nframes,
    int natoms,
) nogil:

    cdef int i, j

    for i in prange(nframes, nogil=True):
        for j in range(natoms):
            clust_output[i][clust_traj[i][j]] += 1

    return clust_output


