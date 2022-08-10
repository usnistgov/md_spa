
import numpy as np

def vector_alignment( target_pt, target_vec, ref_pts, ref_vecs):
    """
    Calculate cos(theta) of effective dipole moment

    The effective dipole moment is calculated as a weighted sum of dipole moments scaled by the cube of the distance between vectors. 

    Parameters
    ----------
    target_pt : numpy.ndarray
        The position vector of the object being evaluated
    target_vec : numpy.ndarray
        The direction vector (such as a dipole) of the target object
    ref_pts : numpy.ndarray
        The position vectors of the reference objects expected to have influence on the target object
    ref_vecs : numpy.ndarray
        The direction vectors (such as a dipole) of the reference objects expected to have influence on the target object

    Returns
    -------
    metric : float
        The dot product of the unit vectors for the target vector and effective dipole moment of reference vectors. A value of unity indicated the target is aligned with the effective reference, and a value of -1 indicates that it's antiparallel.
    ref_vec : nump.ndarray
        Vector of the same length as the provided vectors that is the unit vector indicating the direction of the effective dipole moment.

    """
    r3 = np.sum(np.square(ref_pt-target_pt),axis=1)
    ref_vec = np.sqrt(np.sum(ref_vec/r3[:,np.newaxis], axis=0))**3
    ref_vec /= np.sqrt(np.sum(np.square(ref_vec)))
    target_vec /= np.sqrt(np.sum(np.square(target_vec)))
    
    metric = np.dot(target_vec,ref_vec)
    
    return metric, ref_vec
