""" Calculate the volume of overlappying spheres

    Recommend loading with:
    ``import md_spa.monte_carlo_volume as mcv``

"""

import numpy as np

import md_spa_utils.data_manipulation as dm

def random_points(npts, ranges):
    """
    Random points for Monte Carlo use.

    Using `numpy.random.uniform <https://numpy.org/doc/stable/reference/random/generated/numpy.random.uniform.html>`_, ``npts`` vectors are generated of length ``len(ranges)``.
    The iterable, ranges contains iterables of length 2 with the minimum and maximum of that point.

    An example of this might be ``npts=1e+3`` with a 3x4x2 box, ``ranges=[[0,3],[0,4],[0,2]]``

    Parameters
    ----------
    npts : int
        Number of vector "points" to produce
    ranges : list[list]
        Iterable of min and max values for each dimension of the vectors

    Returns
    -------
    output : numpy.ndarray
        Returns a matrix of random points that is ``npts`` long and ``len(ranges)`` deep. 

    """

    np.random.seed()
    output = np.zeros((int(npts),len(ranges)))
    for i,(xmin, xmax) in enumerate(ranges):
        output[:,i] = np.random.uniform( xmin, xmax, int(npts))
    return output

def overlapping_spheres(rcut, ref_pts, npts=1e+4, repeats=3):
    """
    Monte Carlo code to determine the volume of several overlapping spheres

    Parameters
    ----------
    rcut : float/numpy.ndarray
        Radius of the circles, or an array of different radii for each circle.
    ref_pts : numpy.ndarray
        Array of reference points for centers of circles    
    npts : int, default=1e+4
        Number of vector "points" to produce. 
    repeats : int, default=3
        Number of times this MC calculation is performed to obtain the standard deviation

    Returns
    -------
    region_volume : float
        Volume of region consisting of overlapping spheres in cubic units of the scale used in the reference coordinates 
    volume_std : float
        The standard deviation based on the number of times this calculation was repeated

    """
    npts = int(npts)
    repeats = int(repeats)

    if not dm.isiterable(rcut):
        rcut = rcut*np.ones(len(ref_pts))
    elif len(ref_pts) != len(rcut):
        raise ValueError("Length of rcut (circle radii) and number of reference points (circle centers) must be equal")

    nrefs = len(ref_pts)
    ref_pts = np.array(ref_pts)
    ref_pts -= np.mean(ref_pts, axis=0)
    ranges = [[np.min(x)-rcut[i], np.max(x)+rcut[i]] for i,x in enumerate(ref_pts.T)]

    volume = np.prod(np.array([xmax-xmin for (xmin,xmax) in ranges]))
    if npts == None:
        npts = int(volume)*10

    test_points = np.reshape(random_points(repeats*npts, ranges), (repeats, npts, 3))
    distances = np.sum(np.square(test_points[:,:,None,:] - ref_pts[None, None, :, :]), axis=-1)
    in_vol_count = np.sum(distances <= np.square(rcut)[None, None, :], axis=-1) # Sum over ref points
    in_vol = in_vol_count > 0 # Bool, in volume or out?
    vol_repeats = np.sum(in_vol, axis=-1)/npts*volume
    final_stats = dm.basic_stats(vol_repeats, error_descriptor="sample")

    return final_stats[0], final_stats[1]


