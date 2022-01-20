import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.image as img
from scipy.interpolate import CubicSpline
from scipy.interpolate import UnivariateSpline

import md_spa_utils.data_manipulation as dm

from . import read_lammps as f

def consolidate(target_dir, boxes, column_names, file_in="coord.lammpstrj", file_out="coord_histogram.txt", bins=None):
    """
    This function will take equivalent lammps dump files that contain coordination data and write out a consolidated histogram.

    Parameters
    ----------
    target_dir : str
        This string should be common to all equivalent boxes, except for one variable entry to be added with the str format function. (e.g. "path/to/box{}/run1")
    boxes : list
        List of entries that individually complete the path to a lammps data.
    column_names : list
        List of column names used the lammps dump file to indicate coordination data
    file_in : str, Optional, default='coord.lammpstrj'
        Name of lammps dump file with coordination information
    file_out : str, Optional, default='coord_histogram.txt'
        Filename for consolidated data file
    bins : list, Optional, default = (0,20)
        Define the range of bins in the histogram

    Returns
    -------
    mean_coord : list
        Mean coordination number for each of the ``columns_names``
    stderr_coord : list
        Standard error in coordination number for each of the ``columns_names``

    """
    
    Nsets = len(column_names)

    # Check inputs
    if not dm.isiterable(boxes):
        raise ValueError("The input `boxes` should be iterable")

    try:
        tmp_file = target_dir.format(boxes[0])
    except:
        raise ValueError("The given input `target_dir` should have a placeholder {} with which to format the path with entries in `boxes`")

    if np.all(bins == None):
        bins = [20]

    if not dm.isiterable(bins) or len(bins) > 2:
        raise ValueError("The range given for the number of bins must be a list of length one or two.")
    else:
        if len(bins) == 2:
            bins[1] += 2
            Nbins = bins[1] - bins[0]
        else:
            bins[0] += 2
            Nbins = bins[0]

    # Check Files
    remove = []
    for b in boxes:
        tmp_path = os.path.join(target_dir.format(b),file_in)
        if not os.path.isfile(tmp_path):
            remove.append(b)
    boxes = [b for b in boxes if b not in remove]

    # Consolidate data
    coord_matrix = f.read_files([os.path.join(target_dir.format(i),file_in) for i in boxes],f.read_lammps_dump,col_name=column_names, dtype=int)
    coord_sets = np.transpose(np.reshape(coord_matrix, (-1,Nsets)))

    if np.size(coord_sets)==0:
        raise ValueError("Data was not extracted")

    Npoints = np.zeros(Nsets)
    coord_hist = np.zeros((Nsets+1,Nbins-1))
    for i, tmp_set in enumerate(coord_sets):
        Npoints[i] = len(tmp_set)
        tmp,binedg = np.histogram(tmp_set,bins=range(*bins))
        coord_hist[i+1] = tmp/np.sum(tmp)
    coord_hist[0] = binedg[:-1]
    coord_hist = np.transpose(np.array(coord_hist))

    # Write output files
    with open(file_out,'w') as ff:
        ff.write("# Coordination for Boxes: {}, with {} data points in each respective histogram\n".format(boxes,Npoints))
        ff.write("# Count, {}\n".format(", ".join(column_names)))
        for line in coord_hist:
            ff.write(", ".join(str(x) for x in line)+"\n")

    # Write out coordination statistics
    Nboxes = np.shape(coord_matrix)[0]
    coord_sets = np.reshape(coord_matrix, (Nboxes,-1,Nsets))
    
    mean_coord = []
    stderr_coord = []
    for i in range(Nsets):
        mean, stderr = dm.basic_stats([np.mean(coord_sets[x,:,i]) for x in range(Nboxes)])        
        mean_coord.append(mean)
        stderr_coord.append(stderr)

    return mean_coord, stderr_coord

