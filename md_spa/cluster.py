import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.image as img
from scipy.interpolate import CubicSpline
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt

import md_spa_utils.data_manipulation as dm
import md_spa_utils.file_manipulation as fm

from . import read_lammps as rl

def consolidate(target_dir, boxes, column_name, file_in="coord.lammpstrj", file_out="cluster_histogram.txt", max_frames=None, kwargs_analysis={}, kwargs_read_file={}):
    """
    This function will take equivalent lammps dump files that contain coordination data and write out a consolidated histogram.

    Parameters
    ----------
    target_dir : str
        This string should be common to all equivalent boxes, except for one variable entry to be added with the str format function. (e.g. "path/to/box{}/run1")
    boxes : list
        List of entries that individually complete the path to a lammps data.
    column_name : string
       Lammps dump file column name used the lammps dump file to indicate coordination data
    file_in : str, Optional, default='coord.lammpstrj'
        Name of lammps dump file with coordination information
    file_out : str, Optional, default='coord_histogram.txt'
        Filename for consolidated data file
    kwargs_read_file : dict, Optional, default={"col_name": column_name, "dtype": int}
        Keyword arguments for :func:`md_spa.read_lammps.read_lammps_dump`
    kwargs_analysis : dict, Optional, default={}
        Keyword arguments for :func:`md_spa.cluster.analyze_clustering`    

    """
    
    if not dm.isiterable(boxes):
        raise ValueError("The input `boxes` should be iterable")

    try:
        tmp_file = target_dir.format(boxes[0])
    except:
        raise ValueError("The given input `target_dir` should have a placeholder {} with which to format the path with entries in `boxes`")

    # Check Files
    remove = []
    for b in boxes:
        tmp_path = os.path.join(target_dir.format(b),file_in)
        if not os.path.isfile(tmp_path):
            remove.append(b)
    boxes = [b for b in boxes if b not in remove]

    # Consolidate data
    tmp_kwargs = {"col_name": column_name, "dtype": int}
    tmp_kwargs.update(kwargs_read_file)
    cluster_arrays = []
    avg_nparticles = []
    maxparticles = 0
    for i, box in enumerate(boxes):
        cluster_array = rl.read_lammps_dump( os.path.join(target_dir.format(box),file_in), **tmp_kwargs)[0]
        clust_sizes, nparticles = analyze_clustering(cluster_array, **kwargs_analysis)
        cluster_arrays.append(clust_sizes)
        avg_nparticles.append(np.nanmean(nparticles))
        if np.nanmax(clust_sizes) > maxparticles:
            maxparticles = np.nanmax(clust_sizes)
        

    bins = np.arange(maxparticles+1)
    output = [bins[:-1]+(bins[1:]-bins[:-1]/2)] + [np.histogram(x.flatten(), bins=bins)[0] for x in cluster_arrays]
    header = ",".join(["#Avg number of nonpercolated beads: {}\n#Particles/Cluster".format(avg_nparticles)]+["Box {}".format(box) for box in boxes])
    fm.write_csv(output.T, delimiter=",", header=[header])

def analyze_clustering(cluster_array, ncut=1, show_plot=False):
    """
    Determine the number of non-clustered particles and number of particles in larger clusters.

    This data is taken from a trajectory of cluster assignments to determine the clustering statistics.
    The non-clustering paricles are defined as those not in a percolated network, which are defined by a maximum population size.

    Parameters
    ----------
    cluster_array : numpy.ndarray
        Two dimensional array for each frame containing the cluster numbers assigned to the atoms.
    ncut : int, Optional, default=1
        Number of particles in a cluster that is below the percolation limit, such as 6 beads in a hydration shell.
    show_plot : bool, Optional, default=False
        If true, the outputs are plotted. This is useful for debugging purposes.

    Returns
    -------
    clusters : numpy.ndarray
        Two dimensional array where for each frame the number of particles in each cluster above the percolation threshold is listed
    nparticles : numpy.ndarray
        Array of the number of particles below the percolation threshhold 

    """

    if len(np.shape(cluster_array)) != 2:
        raise ValueError("Input array should be 2D")

    nclusters = int(np.nanmax(cluster_array))
    nparticles = np.zeros(len(cluster_array))
    cluster_output = np.nan*np.ones(( len(cluster_array), nclusters))
    for i,tmp in enumerate(cluster_array):
        clusters = np.sort(np.array([np.sum([x==y for x in tmp]) for y in range(1,nclusters+1)]))
        nparticles[i] = len(np.where(np.logical_and(clusters < ncut+1, clusters > 0))[0])
        clusters = clusters[np.where(clusters > ncut)]
        cluster_output[i, :len(clusters)] = clusters
        if show_plot:
            plt.plot(clusters, label=str(i))

    if show_plot:
        plt.xlabel("Cluster Number")
        plt.ylabel("Number of Particles")
        plt.figure(2)
        plt.plot(nparticles)
        plt.xlabel("Frame")
        plt.ylabel("Number of Single Particles")
        plt.legend(loc="best")
        plt.figure(3)
        plt.hist(clusters.flatten(), bins=np.max(clusters)+1)
        plt.xlabel("Number of Single Particles")
        plt.show()

    return clusters, nparticles

def write_vmd(filename, column_name, frame=0, file_xyz="cluster_{}.xyz", file_vmd="cluster_{}.vmd", sigma=1, kwargs_read_file={}, cmap="tab20", npixels=600):
    """
    This function will take equivalent lammps dump files that contain coordination data and write out a consolidated histogram.

    Parameters
    ----------
    filename : str
        Name of lammps dump file with coordination information
    column_name : string
        Lammps dump file column name used the lammps dump file to indicate coordination data
    frame : int, Optional, default=0
        Frame number to render
    file_out : str, Optional, default="cluster_{}.vmd".format(frame)
        Name of VMD file
    sigma : float, default=1
        Bead diameter
    kwargs_read_file : dict, Optional, default={"col_name": column_name, "dtype": int}
        Keyword arguments for :func:`md_spa.read_lammps.read_lammps_dump`
    cmap : str, Optional, default="tab20"
        Name of matplotlib colormap
    npixels : int, Optional, default=600
        Number of pixels for height and width

    """
    # Consolidate data
    tmp_kwargs = {"col_name": [ "x", "y", "z", column_name], "dtype": float}
    tmp_kwargs.update(kwargs_read_file)

    cluster_array = rl.read_lammps_dump( filename, **tmp_kwargs)[0]
    if len(cluster_array) < frame:
        raise ValueError("Number of imported frames, {}, is less than request frame number, {}".format(len(cluster_array), frame))
    else:
        cluster_array = cluster_array[frame]

    output = [["X"]+list(x[:3]) for x in cluster_array]
    header = ["{}\nCluster {}".format(len(cluster_array), frame)]
    fm.write_csv(file_xyz.format(frame), output, delimiter=" ", header=header, header_comment="")

    cluster_ids = list(set(list(cluster_array[:,-1])))
    clusters = [[i for i,x in enumerate(cluster_array) if x[-1] == y] for y in cluster_ids]

    Cmap = plt.get_cmap(name=cmap)
    colors = Cmap(np.linspace(0, 1, len(cluster_ids)))
    for i, color in enumerate(colors):
        plt.plot([0, 10],[i, i], color=color)
    plt.show()

    lx = np.nanmax(cluster_array[:,:3])
    with open(file_vmd.format(frame), "w") as f:
        f.write("# Output clustering visualizaiton created by MD_SPA\n")
        f.write("\n# Background\n")
        f.write("display projection Orthographic\n")
        f.write("display depthcue off\n")
        f.write("axes location Off\n")
        f.write("color Display Background white\n")
        f.write("display resize {} {}\n".format(npixels, npixels))
        f.write("\n# Read file\n")
        f.write("display resetview\n")
        f.write("mol addrep 0\n")
        f.write("display resetview\n")
        f.write("mol new {"+file_xyz.format(frame)+"} type {xyz} first 0 last -1 step 1 waitfor 1\n")
        f.write('pbc set "{'+"{} {} {}".format(lx,lx,lx)+'}"\n')
        f.write("pbc box -color blue\n")
        f.write("color change rgb 0 0 0 0\n")
        for i, clust in enumerate(clusters):
            f.write("\n# Cluster {}\n".format(i))
            f.write("mol addrep 0\n")
            f.write("mol modselect {} 0 index {}\n".format(i, " ".join([str(x) for x in clust])))
            f.write("mol material {} 0 BrushedMetal\n".format(i))
            f.write("mol modstyle {} 0 VDW 1.000000 12.000000\n".format(i))
            f.write("color change rgb {} {} {} {}\n".format(i+9, *colors[i][:3]))
            f.write("mol modcolor {} 0 ColorID {}\n".format(i, i+9))
        f.write("\n# Render Figure\n")
        f.write("render POV3 {} povray +W%w +H%h -I%s -O%s.png +D +X +A +UA +FN -res {} {} +FT\n".format(file_xyz.split(".")[0].format(frame), npixels, npixels))
#        f.write("exit\n")


     


