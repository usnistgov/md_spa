
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

def angle(vertex, pos1, pos2):
    """
    Calculate the angle between two vectors

    Parameters
    ----------
    vertex : numpy.ndarray
        Position of the angle vertex
    pos1 : numpy.ndarray
        Position of one point defining the angle
    pos2 : numpy.ndarray
        Position of another point defining the angle

    Returns
    -------
    angle : float
        Angle in degrees    

    """

    vec1 = pos1 - vertex
    vec2 = pos2 - vertex

    vec1 /= np.sqrt(np.sum(np.square(vec1)))
    vec2 /= np.sqrt(np.sum(np.square(vec2)))

    angle = np.arccos(np.dot(vec1,vec2))*180/np.pi
    if angle > 180.:
        angle -= 180.

    return angle

def check_wrap(pos_diff, dimensions, dist_type="distance"):
    """
    Check that the difference in positions is possible in the box dimensions, and wrap accordingly.

    Parameters
    ----------
    pos_diff : numpy.ndarray
        Matrix of atoms by coordinates, represents the difference between atom set and some reference point.
    dimensions : numpy.ndarray
        Array of box length in three dimensions, assuming origin at (0,0,0)
    dist_type : str, Optional, default="distance"
        Type of coordinate data. By default it is assumed to be an array to calculate the distance between atoms and some reference. Thus, the "coordinates" should be transformed if the value is greater than half the box length. 
    
        - distance : A matrix of displacement vectors representing the distance between atoms and some reference 
        - wrap : A matrix of coordinates to wrap into the box

    Returns
    -------
    pos_diff_new : numpy.ndarray
        Matrix of atoms by coordinates, represents the difference between atom set and some reference poin
t, wrapped if approproate.

    """
    if dist_type == "distance":
        images = ( dimensions/2 - pos_diff ) // dimensions
    elif dist_type == "wrap":
        images = ( dimensions - pos_diff ) // dimensions
    pos_diff_new = pos_diff + images * dimensions

    return pos_diff_new


