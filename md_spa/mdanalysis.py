
import numpy as np
import sys
import os
import warnings

import MDAnalysis as mda
from MDAnalysis.analysis import polymer as mda_polymer
from MDAnalysis.analysis import rdf as mda_rdf
from MDAnalysis.transformations import unwrap
from MDAnalysis.analysis.hydrogenbonds.hbond_analysis import HydrogenBondAnalysis as HBA
from MDAnalysis.analysis.waterdynamics import SurvivalProbability as SP
import MDAnalysis.analysis.msd as mda_msd
from MDAnalysis import transformations as trans

from . import misc_functions as mf
import md_spa_utils.data_manipulation as dm

def center_universe_around_group(universe, select, verbose=False, reference="initial", com_kwargs={"unwrap": True}):
    """
    Recenter the universe around a specified group. This can be done from the initial center of mass of the group, or have each frame centered if only an ensemble average of some property in each frame in desired.

    Parameters
    ----------
    universe : obj
        ``MDAnalysis.Universe`` object instance
    select : str
        This string should align with ``MDAnalysis.universe.select_atoms(select)`` rules
    verbose : bool, Optional, default=False
        Set whether calculation will be run with comments
    reference : str, Optional, default="intitial"
        Options for how to transform the coordinates:
        
        - "initial": Move the center of mass for the group in the first frame to the center of the box and use that same transformation in all other frames. This will allow further dynamic analysis without interference of boundary conditions.
        - "relative": Move the center of mass for a group to the center of the box for each respective frame. All dynamic movement will be distorted and lost, but the group will not migrate to the boundary for static property calculations.

    com_kwargs : dict, Optional, default={"unwrap": True}
        Keyword arguements for mdanalysis.core.topologyattrs.center_of_mass()

    Returns
    -------
    universe : obj 
        ``MDAnalysis.Universe`` object instance

    """

    universe.trajectory[0]
    group_target = universe.select_atoms(select)
    group_remaining = universe.select_atoms("all") - group_target
    dimensions = universe.trajectory[0].dimensions[:3]
    
    if verbose:
        print("Polymer Center of Mass Before", group_target.center_of_mass(**com_kwargs))

    if reference == "initial":
        box_center = dimensions / 2
        com = group_target.center_of_mass(**com_kwargs)
        transforms = [
            trans.unwrap(group_target),
            trans.translate(box_center-com),
            trans.wrap(group_remaining)
        ]
        universe.trajectory.add_transformations(*transforms)
    elif reference == "relative":
        transforms = [
            trans.unwrap(group_target),
            trans.center_in_box(group_target, center="mass"),
            trans.wrap(group_remaining)
        ]
        universe.trajectory.add_transformations(*transforms)

    else:
        raise ValueError("Transformation reference, {}, is not supported.".format(reference))

    if verbose:
        print("Group Center of Mass After", group_target.center_of_mass())

    universe.trajectory[0]

    return universe

def check_universe(universe):
    """
    Function that checks whether a valid universe was provided, and if not, a universe is generated with the provided information.

    Parameters
    ----------
    universe : obj/tuple
        Can either be:
        
        - MDAnalysis universe
        - A tuple of length 2 containing the format type keyword from :func:`md_spa.mdanalysis.generate_universe` and appropriate arguments in a tuple 
        - A tuple of length 3 containing the format type keyword from :func:`md_spa.mdanalysis.generate_universe`, appropriate arguments in a tuple, and dictionary of keyword arguments

    Returns
    -------
    universe : obj
        ``MDAnalysis.Universe`` object instance
    """

    if isinstance(universe, mda.core.universe.Universe):
        u = universe
    elif dm.isiterable(universe):
        length = len(universe)
        if length < 2:
            raise ValueError("To generate an mda universe, the format type keyword from md_spa.mdanalysis.generate_universe and appropriate arguments in a tuple")
        elif length > 3:
            raise ValueError("To generate an mda universe, the format type keyword from md_spa.mdanalysis.generate_universe, appropriate arguments in a tuple, and dictionary of keyword arguments.")
        elif length == 2:
            u = generate_universe(*universe)
        elif length == 3:
            u = generate_universe(universe[0], universe[1], **universe[2])
    else:
        raise ValueError("Input, {}, of type, {}, cannot be used to produce an MDAnalysis universe".format(universe, type(universe)))

    return u

def generate_universe(package, args, kwargs={}):
    """
    Generate MDAnalysis universe

    e.g., ``universe = spa_mda.generate_universe("LAMMPS",(data_file, dump_files), {"dt":dt})``, where data_file is a filename and path, and dump_files is a list of filenames and paths.

    Parameters
    ----------
    package : str
        String to indicate the universe ``format`` to interpret. The supported formats are:
            
        - LAMMPS: Must include the path to a data file with the atom_type full (unless otherwise specified.), and optionally include a list of paths to sequential dump files.           

    args : tuple
        Arguments for ``MDAnalysis.Universe``
    kwargs : dict, Optional, default={"format": package_identifier, "continuous": True}
        Keyword arguments for ``MDAnalysis.Universe``. Note that the ``format`` will be added internally.
                 
    Returns
    -------
    universe : obj
        ``MDAnalysis.Universe`` object instance

    """

    supported_packages = ["LAMMPS"]
    if isinstance(args[1],list) and len(args[1]) > 1:
        tmp_kwargs = {"continuous": True}
    else:
        tmp_kwargs = {}
    tmp_kwargs.update(kwargs)

    if package == "LAMMPS":
        if "format" not in tmp_kwargs:
            tmp_kwargs.update({"format": "LAMMPSDUMP"})
        universe = mda.Universe(*args, **tmp_kwargs)
    else:
        raise ValueError("The package, {}, is not supported, choose one of the following: {}".format(package,", ".join(supported_packages)))

    return universe
        
def calc_partial_rdf(u, groups, rmin=0.1, rmax=12.0, nbins=1000, verbose=False, exclusion_block=None, exclusion_mode="block", exclude=True, run_kwargs={}):
    """
    Calculate the partial rdf of a polymer given a LAMMPS data file, dump file(s), and a list of the atomIDs along the backbone (starting at 1).

    Parameters
    ----------
    u : obj/tuple
        Can either be:
        
        - MDAnalysis universe
        - A tuple of length 2 containing the format type keyword from :func:`md_spa.mdanalysis.generate_universe` and appropriate arguments in a tuple 
        - A tuple of length 3 containing the format type keyword from :func:`md_spa.mdanalysis.generate_universe`, appropriate arguments in a tuple, and dictionary of keyword arguments

    groups : list[tuples]
        List of tuples containing pairs of strings that individually identify a MDAnalysis AtomGroup with universe.select_atoms(groups[i])
    rmin : float, Optional, default=0.1
        Minimum rdf cutoff
    rmax : float, Optional, default=12.0
        Maximum rdf cutoff
    nbins : 100, Optional, default=1000
        Number of bins in the rdf
    verbose : bool, Optional, default=False
        Set whether calculation will be run with comments
    exclusion_block : tuple, Optional, default=None
        Allows the masking of pairs from within the same molecule.  For example, if there are 7 of each atom in each molecule, the exclusion mask `(7, 7)` can be used. The default removes self interaction.
    exclusion_mode : str, Optional, default="block"
        Set to "block" for traditional mdanalysis use, and use "relative" to remove ``exclusion_block[0]`` neighbors around reference atom.
    exclude : str, Optional, default=None
        Set the type of exclusion: ["include bound", "exclude bound", "explicit"]
    run_kwargs : dict, Optional, default={}
        Other keyword arguments from ``MDAnalysis.analysis.rdf.InterRDF.run()``

    Returns
    -------
    rdf_output : numpy.ndarray
        Array of distances and rdf values. rdf_output[0] is the distances and the remaining rows represent the given groups pairs.

    """

    u = check_universe(u)

    if not dm.isiterable(groups):
        raise ValueError("The entry `groups` must be an iterable structure with selection strings.")

    if dm.isiterable(exclusion_block) and not dm.isiterable(exclusion_block[0]):
        exclusion_block = [exclusion_block]

    if exclusion_block == None:
        exclusion_block = [None for x in range(len(groups))]
    elif len(exclusion_block) != len(groups):
        raise ValueError("Number of exclusion_blocks does not equal the number of group pairs.")

    group_dict = {}
    rdf_output = np.zeros((len(groups)+1,nbins))
    flag_bins = False
    for i, (group1, group2) in enumerate(groups):
        if group1 not in group_dict:
            group_dict[group1] = u.select_atoms(group1)
        if group2 not in group_dict:
            group_dict[group2] = u.select_atoms(group2)
        Rdf = mda_rdf.InterRDF(group_dict[group1],group_dict[group2],nbins=nbins,range=(rmin,rmax),verbose=verbose, exclusion_block=exclusion_block[i],exclusion_mode=exclusion_mode, exclude=exclude)
        Rdf.run(verbose=verbose, **run_kwargs)
        if not flag_bins:
            flag_bins = True
            rdf_output[0] = Rdf.results.bins
        rdf_output[i+1] = Rdf.results.rdf

    return rdf_output

def calc_msds(u, groups, dt=1, verbose=False, fft=False, run_kwargs={}):
    """
    Calculate the partial rdf of a polymer given a LAMMPS data file, dump file(s), and a list of the atomIDs along the backbone (starting at 1).

    Parameters
    ----------
    u : obj/tuple
        Can either be:
        
        - MDAnalysis universe
        - A tuple of length 2 containing the format type keyword from :func:`md_spa.mdanalysis.generate_universe` and appropriate arguments in a tuple 
        - A tuple of length 3 containing the format type keyword from :func:`md_spa.mdanalysis.generate_universe`, appropriate arguments in a tuple, and dictionary of keyword arguments

    group : list
        List of strings that identify a MDAnalysis AtomGroup with universe.select_atoms(group[i])
    dt : float, Optional, default=1
        Define dt to convert frame numbers to time in picoseconds
    verbose : bool, Optional, default=False
        Set whether calculation will be run with comments
    fft : bool, Optional, default=False
        Choose whether to use the fft method, in that case the non gaussian parameter will not be computed.
    run_kwargs : dict, Optional, default={}
        Other keyword arguments from ``MDAnalysis.analysis.msd.MSD.run()``

    Returns
    -------
    msd_output : numpy.ndarray
        Array of times and MSD values. msd_output[0] is the time values and the remaining rows represent the given group MSDs.
    second_order_nongaussian_parameter : numpy.ndarray
        Array of times and second order nongaussian parameter of the MSD. A first peak for an atomistic system may indicate bond vibration, while a second or solitary peak will represent the characteristic time for the cross over from ballistic to diffusive regimes. The MSD at this time equals six times the mean localization characterization length squared. DOI: 10.1103/PhysRevE.65.041804 

    """

    u = check_universe(u)

    if not dm.isiterable(groups):
        raise ValueError("The entry `groups` must be an iterable structure with selection strings.")

    nbins = len(u.trajectory)
    msd_output = np.zeros((len(groups)+1,nbins))
    if not fft:
        nongaussian_parameter = np.zeros((len(groups)+1,nbins))
    flag_bins = False
    for i, group in enumerate(groups):
        MSD = mda_msd.EinsteinMSD(u, select=group, msd_type='xyz', fft=fft)
        MSD.run(verbose=verbose, **run_kwargs)
        if verbose:
            print("Generated MSD for group {}".format(group))

        if not flag_bins:
            flag_bins = True
            msd_output[0,:MSD.n_frames] = np.arange(MSD.n_frames)*dt # ps
            if not fft:
                nongaussian_parameter[0,:MSD.n_frames] = np.arange(MSD.n_frames)*dt # ps
        msd_output[i+1,:MSD.n_frames] = MSD.results.timeseries # angstroms^2
        if not fft:
            nongaussian_parameter[i+1,:MSD.n_frames] = MSD.results.nongaussian_parameter

    if not fft:
        return msd_output, nongaussian_parameter
    else:
        return msd_output

def calc_persistence_length(u, backbone_indices, save_plot=True, figure_name="plot_lp_fit.png", verbose=True):
    """
    Calculate the persistence length of a polymer given a LAMMPS data file, dump file(s), and a list of the atomIDs along the backbone (starting at 1).

    Parameters
    ----------
    u : obj/tuple
        Can either be:
        
        - MDAnalysis universe
        - A tuple of length 2 containing the format type keyword from :func:`md_spa.mdanalysis.generate_universe` and appropriate arguments in a tuple 
        - A tuple of length 3 containing the format type keyword from :func:`md_spa.mdanalysis.generate_universe`, appropriate arguments in a tuple, and dictionary of keyword arguments

    backbone_indices : list
        List of atomID for atoms along backbone, starting at 1
    save_plot : bool, Optional, default=True
        Optionally save plot of data and fit
    figure_name : str, Optional, default="plot_lp_fit.png"
        Name of the plot to be saved
    verbose : bool, Optional, default=True
        If True, the persistence length will be printed to the screen

    Returns
    -------
    lp : float
        The persistence length of the provided files

    """

    u = check_universe(u)
    
    tmp_indices = [x-1 for x in backbone_indices]
    #tmp_indices = backbone_indices
    backbone = mda.core.groups.AtomGroup(tmp_indices,u)
    sorted_bb = mda_polymer.sort_backbone(backbone) # Should be unnecessary if backbone_indices are in order
    
    plen = mda_polymer.PersistenceLength([sorted_bb])
    plen.run(verbose=verbose)
    
    if verbose:
        print('The persistence length is {}'.format(plen.results.lp))
    
    if save_plot:
        ax = plen.plot()
        import matplotlib.pyplot as plt
        plt.tight_layout()
        plt.savefig(figure_name,dpi=400)
        ax.set_yscale('log')
        ax.set_ylabel("log C(x)")
        new_filename = figure_name.split("/")
        new_filename[-1] = "log_" + new_filename[-1]
        plt.savefig("/".join(new_filename),dpi=400)

    return plen.results.lp

def calc_gyration(universe, select="all", pbc=False):
    """
    Calculate the radius of gyration and anisotropy of a polymer given a LAMMPS data file, dump file(s).

    Parameters
    ----------
    universe : obj/tuple
        Can either be:
        
        - MDAnalysis universe
        - A tuple of length 2 containing the format type keyword from :func:`md_spa.mdanalysis.generate_universe` and appropriate arguments in a tuple 
        - A tuple of length 3 containing the format type keyword from :func:`md_spa.mdanalysis.generate_universe`, appropriate arguments in a tuple, and dictionary of keyword arguments

    select : str, Optional, default="all"
        This string should align with MDAnalysis universe.select_atoms(select) rules
    pbc : bool, Optional, default=True
        If ``True``, move all atoms within the primary unit cell before calculation.

    Returns
    -------
    rg : numpy.ndarray
        The radius of gyration is calculated for the provided dump file
    kappa : numpy.ndarray
        The anisotropy is calculated from the gyration tensor, where zero indicates a sphere and one indicates an infinitely long rod.
    anisotropy : numpy.ndarray
        Another form of anisotropy where the square of the largest eigenvalue in the gyration tensor is scales by the square of the smallest. A spherical cluster would then have a value of unity and an infinitely long rod would approach infinity.

    """

    universe = check_universe(universe)
    group = universe.select_atoms(select)  # a selection (a AtomGroup)
    com = [[],[]]
    center_universe_around_group(universe, select, reference="relative")
    if not np.any(group._ix):
        raise ValueError("No atoms met the selection criteria")

    lx = len(universe.trajectory)
    rgyr = np.zeros(lx)
    kappa = np.zeros(lx)
    anisotropy = np.zeros(lx)
    for i,ts in enumerate(universe.trajectory):  # iterate through all frames
        rgyr[i] = group.radius_of_gyration(pbc=pbc)  # method of a AtomGroup; updates with each frame
        kappa[i], anisotropy[i] = group.anisotropy(pbc=pbc)  # method of a AtomGroup; updates with each frame
    
    return rgyr, kappa, anisotropy

def calc_end2end(u, indices):
    """
    Calculate the end to end distance of a polymer given a MDAnalysis universe and the atomIDs of the first and last particle in the backbone.

    Parameters
    ----------
    u : obj/tuple
        Can either be:
        
        - MDAnalysis universe
        - A tuple of length 2 containing the format type keyword from :func:`md_spa.mdanalysis.generate_universe` and appropriate arguments in a tuple 
        - A tuple of length 3 containing the format type keyword from :func:`md_spa.mdanalysis.generate_universe`, appropriate arguments in a tuple, and dictionary of keyword arguments

    indices : list
        Pair of atomIDs to obtain distance between (starting at 1)

    Returns
    -------
    r_end2end : numpy.ndarray
        Distance between two atomIDs in the given trajectory

    """

    try:
        if len(indices) != 2:
            raise ValueError("A pair of indices are required. {} given.".format(indices))
    except:
        raise ValueError("A pair of indices are required. {} given.".format(indices))
        

    u = check_universe(u)
    u = u.copy()
    center_universe_around_group(u, "index {} or index {}".format(*indices), reference="relative")
    ag = u.atoms
    tmp_indices = [x-1 for x in indices]

    r_end2end = np.zeros(len(u.trajectory))
    for i,ts in enumerate(u.trajectory):  # iterate through all frames
        r = u.atoms.positions[indices[0]] - u.atoms.positions[indices[1]]  # end-to-end vector from atom positions
        r_end2end[i] = np.linalg.norm(r)   # end-to-end distance

    return r_end2end

def hydrogen_bonding(u, indices, dt, tau_max=200, verbose=False, show_plot=False, intermittency=0, path="", filename="lifetime.csv", acceptor_kwargs={}, donor_kwargs={}, hydrogen_kwargs={}, d_h_cutoff=1.2, d_a_cutoff=3.5, d_h_a_angle_cutoff=125.0, kwargs_run={}):
    """
    Calculation the hydrogen bonding statistics for the given type interactions.

    Parameters
    ----------
    u : obj/tuple
        Can either be:
        
        - MDAnalysis universe
        - A tuple of length 2 containing the format type keyword from :func:`md_spa.mdanalysis.generate_universe` and appropriate arguments in a tuple 
        - A tuple of length 3 containing the format type keyword from :func:`md_spa.mdanalysis.generate_universe`, appropriate arguments in a tuple, and dictionary of keyword arguments

    indices : list
        A list of lists containing the type for the hydrogen bond donor, hydrogen, and acceptor. If an atom type is unknown or more than one is desired, put None instead.
    dt : float
        Define the timestep as used in ``mdanalysis.Universe``, or the number of ps between frames.
    tau_max : int, Optional, default=200
        Number of timesteps to calculate the decay to, this value times dt is the maximum time. Cannot be greater than the number of frames,
    verbose : bool, Optional, default=False
        Print intermediate updates on progress
    show_plot : bool, Optional, default=False
        In a debugging mode, this option allows the time autocorrelation function to be viewed
    intermittency : int, Optional, default=0
        The intermittency specifies the number of times a hydrogen bond can be made and break while still being considered in the correlation function.
    path : str, Optional, default=""
        Path to save files
    filename : str, Optional, default="lifetime.csv"
        Prefix on filename to further differentiate
    acceptor_kwargs : dict, Optional, default={}
        Keyword arguments for ``MDAnalysis.analysis.hydrogenbonds.hbond_analysis.HydrogenBondAnalysis.guess_acceptors()``
    donor_kwargs : dict, Optional, default={}
        Keyword arguments for ``MDAnalysis.analysis.hydrogenbonds.hbond_analysis.HydrogenBondAnalysis.guess_donors()``
    hydrogen_kwargs : dict, Optional, default={}
        Keyword arguments for ``MDAnalysis.analysis.hydrogenbonds.hbond_analysis.HydrogenBondAnalysis.guess_hydrogens()``
    d_h_cutoff : float/numpy.ndarray, Optional, default=1.2
        Cutoff distance between hydrogen bond donor and hydrogen. If an array, it must be of the same length as ``indices``.
    d_a_cutoff : float/numpy.ndarray, Optional, default=3.5
        Cutoff distance between hydrogen bond donor and acceptor. If an array, it must be of the same length as ``indices``.
    d_h_a_angle_cutoff : float/numpy.ndarray, Optional, default=125.0
        Cutoff angle for hydrogen bonding. Assumed to be 125.0 to eliminate the constraint imposed in MDAnalysis, since MD isn't constrained by orbitals. If an array, it must be of the same length as ``indices``. This value was chosen from a lower limit determined in our MD simulations.
    kwargs_run : dict, Optional, default={}
        Keyword arguments for ``MDAnalysis.analysis.base.AnalysisBase.run()``

    Returns
    -------
    Writes out .csv file: "{file_prefix}res_time_#typedonor_#typehydrogen_#typeacceptor.csv", where the type number is zero if a type is defined as None.

    """

    if dm.isiterable(indices):
        if not dm.isiterable(indices[0]):
            indices = [indices]
    else:
        raise ValueError("Input, indices, but be a list of iterables of length three")

    if dm.isiterable(d_h_cutoff):
        if len(d_h_cutoff) != len(indices):
            raise ValueError("If multiple values for `d_h_cutoff` are given, the array must be of the same length as `indices`")

    if dm.isiterable(d_a_cutoff):
        if len(d_a_cutoff) != len(indices):
            raise ValueError("If multiple values for `d_a_cutoff` are given, the array must be of the same length as `indices`")

    if dm.isiterable(d_h_a_angle_cutoff):
        if len(d_h_a_angle_cutoff) != len(indices):
            raise ValueError("If multiple values for `d_h_a_angle_cutoff` are given, the array must be of the same length as `indices`")

    if verbose:
        for ind in indices:
            print("Lifetime stastistics of donor type {}, hydrogen type {}, and acceptor type {}".format(*ind))

    u = check_universe(u)
    if verbose:
        print("Imported trajectory")

    Hbonds = HBA(universe=u)
    acceptor_list = [x for x in Hbonds.guess_acceptors(**acceptor_kwargs).split() if x not in ["type", "or"]]
    hydrogen_list = [x for x in Hbonds.guess_hydrogens(**hydrogen_kwargs).split() if x not in ["type", "or"]]
    donor_list_per_hydrogen = {}
    for hyd_type in hydrogen_list:
        Hbonds = HBA(universe=u, hydrogens_sel="type {}".format(hyd_type))
        donor_list_per_hydrogen[hyd_type] = [x for x in Hbonds.guess_donors(**donor_kwargs).split() if x not in ["type", "or"]]

    if tau_max > len(u.trajectory):
        tau_max = len(u.trajectory)
        warnings.warn("tau_max is longer than given trajectory, resetting to {}".format(len(u.trajectory)))

    if "stop" in kwargs_run and kwargs_run["stop"] < tau_max:
        tau_max = tau_stop
        warnings.warn("tau_max is longer than hbond.run(stop=stop), resetting to {}".format(tau_stop))

    new_indices = []
    d_h_cutoff_array, d_a_cutoff_array, d_h_a_angle_cutoff_array = [],[],[] 
    for i in range(len(indices)):
        if len(indices[i]) != 3:
            raise ValueError("Index set: {}, must be of length three containing the atoms types or None for [donor, hydrogen, acceptor]")

        if indices[i][1] == None:
            hydrogens = hydrogen_list
        else:
            if dm.isiterable(indices[i][1]):
                hydrogens = [str(x) for x in indices[i][1]]
            else:
                hydrogens = [str(indices[i][1])]

        if indices[i][2] == None:
            acceptors = acceptor_list
        else:
            if dm.isiterable(indices[i][2]):
                acceptors = [str(x) for x in indices[i][2]]
            else:
                acceptors = [indices[i][2]]

        for h in hydrogens:
            for a in acceptors:
                donor_list = donor_list_per_hydrogen[h]
                if indices[i][0] != None and str(indices[i][0]) in donor_list:
                    donor_list = [str(indices[i][0])]
                else:
                    raise ValueError("The given donor, {}, does not bond to hydrogen, {}. Choose one of the following donors: {}".format(indices[i][0]), h, donor_list)

                for d in donor_list:
                        new_indices.append([d,h,a])
                        if dm.isiterable(d_h_cutoff):
                            d_h_cutoff_array.append(d_h_cutoff[i])
                        else:
                            d_h_cutoff_array.append(d_h_cutoff)
                        if dm.isiterable(d_a_cutoff):
                            d_a_cutoff_array.append(d_a_cutoff[i])
                        else:
                            d_a_cutoff_array.append(d_a_cutoff)
                        if dm.isiterable(d_h_a_angle_cutoff):
                            d_h_a_angle_cutoff_array.append(d_h_a_angle_cutoff[i])
                        else:
                            d_h_a_angle_cutoff_array.append(d_h_a_angle_cutoff)

    output = []
    titles = []
    for i,ind in enumerate(new_indices):
        if verbose:
            print("Analyzing Donor Type {}, Hydrogen Type {}, Acceptor Type {}".format(*ind))
        if ind[0] != None:
            donor = "type {}".format(ind[0])
        else:
            donor = None
        if ind[1] != None:
            hydrogen = "type {}".format(ind[1])
        else:
            hydrogen = None
        if ind[2] != None:
            acceptor = "type {}".format(ind[2])
        else:
            acceptor_list = HBA(universe=u)
            acceptor = None
        Hbonds = HBA(universe=u, donors_sel=donor, hydrogens_sel=hydrogen, acceptors_sel=acceptor,
                     d_h_cutoff=d_h_cutoff_array[i],
                     d_a_cutoff=d_a_cutoff_array[i],
                     d_h_a_angle_cutoff=d_h_a_angle_cutoff_array[i],
                    )
        if verbose:
            print("    Primary initialization complete")
        Hbonds.run(**kwargs_run)
        if verbose:
            print("    Primary analysis complete")

        # Lifetime analysis
        tau, timeseries = Hbonds.lifetime(tau_max=tau_max,  intermittency=intermittency)
        time = tau*dt
        if not output:
            output.append(time)

        output.append(timeseries)
        titles.append("{}-{}-{}".format(*ind))

        tmp_path, tmp_filename = os.path.split(filename)
        tmp_filename, ext = tmp_filename.split(".")
        tmp_filename += "_{}-{}-{}.".format(*ind) + ext
        if not tmp_path:
            tmp_path = path
        with open(os.path.join(tmp_path,tmp_filename), "w") as f:
            f.write("# time, {}-{}-{}\n".format(*ind))
            for tmp in np.transpose(np.array([time, timeseries])):
                f.write("{}\n".format(", ".join([str(x) for x in tmp])))
            

        if verbose:
            print("    Finished lifetime analysis")
        
        if show_plot:
            plt.plot(time,timeseries,".-")
            plt.xlabel("Time")
            plt.ylabel("Probability")
            plt.title("Donor Type {}, Hydrogen Type {}, Acceptor Type {}".format(*ind))
            plt.show()

    with open(os.path.join(path,filename),"w") as f:
        f.write("# time, {}\n".format(", ".join(titles)))
        for tmp in np.transpose(np.array(output)):
            f.write("{}\n".format(", ".join([str(x) for x in tmp])))

def survival_probability(u, indices, dt, zones=[(0, 3)], select="isosurface", stop_frame=None, tau_max=200, intermittency=0, verbose=False, show_plot=False, path="", filename="survival.csv"):
    """
    Calculate the survival probability for radially dependent zones from a specified bead type (or overwrite with other selection criteria) showing distance dependent changes in mobility.

    This method requires a characteristic time, generally on the order of a picosecond. This time is equal to the beta relaxation time and is generally temperature independent, and so can be determined from the minimum of the logarithmic derivative of the MSD, and low temperature may be necessary.

    Parameters
    ----------
    u : obj/tuple
        Can either be:
        
        - MDAnalysis universe
        - A tuple of length 2 containing the format type keyword from :func:`md_spa.mdanalysis.generate_universe` and appropriate arguments in a tuple 
        - A tuple of length 3 containing the format type keyword from :func:`md_spa.mdanalysis.generate_universe`, appropriate arguments in a tuple, and dictionary of keyword arguments

    indices : list
        A list of lists containing the type for the (reference_atom_type, target_atom_type)
    dt : float
        Define the timestep as used in ``mdanalysis.Universe``, or the number of ps between frames.
    zones : list[tuple], Optional, default=[(0, 3.0)*len(indices)]
        List of tuples containing zones boundary information to evaluate the survival probability for each interaction pair in ``indices``. Must be the same length as ``indices``, and each sublist the same length as the number of format placeholders minus two (e.g. min max values for ``select="isolayer"`` and a max value for ``select="around"``.
    select : str, Optional, default="isosurface"
        Can be "isosurface", "around", or a string to overwrite one of the previous two selection criteria where the number of placeholders must be ``2+len(zones[0])``.

        - "isolayer": ``type {select_target} and isolayer {zones[i][0]} {zones[i][1]} type {select_reference}``. This option will make "one" region, even if parts of it are fragmented (in this case the "around" is probably what you want). Consider this option where the backbone of a polymer is the reference and a uniform "shell" is desired.
        - "around": ``type {select_target} and around {zones[i][0]} type {select_reference}``. Useful for obtaining the residence time of ions around specific groups (https://nms.kcl.ac.uk/lorenz.lab/wp/?p=1045).

    stop_frame : int, Optional, default=None
        Frame at which to stop calculation. This function can take a long time, so the entire trajectory may not be desired.
    tau_max : int, Optional, default=200
        Number of timesteps to calculate the decay to, this value times dt is the maximum time. See mdanalysis ``waterdynamics.SurvivalProbability``
    intermittency : int, Optional, default=0
        The intermittency specifies the number of times a hydrogen bond can be made and break while still being considered in the correlation function.
    verbose : bool, Optional, default=False
        If true, progress will be printed
    show_plot : bool, Optional, default=False
        In a debugging mode, this option allows the time autocorrelation function to be viewed
    path : str, Optional, default=""
        Path to save files
    filename : str, Optional, default="survival.csv"
        Prefix on filename to further differentiate

    Returns
    -------
    Writes out .csv file

    """

    if dm.isiterable(indices):
        if not dm.isiterable(indices[0]):
            indices = [indices]
    else:
        raise ValueError("Input, indices, but be a list of iterables of length three")

    if verbose:
        for ind in indices:
            print("Lifetime stastistics of reference type {}, and target type {}".format(*ind))

    u = check_universe(u)
    if verbose:
        print("Imported trajectory")

    if tau_max > len(u.trajectory):
        tau_max = len(u.trajectory)
        warnings.warn("tau_max is longer than given trajectory, resetting to {}".format(len(u.trajectory)))

    if not dm.isiterable(zones) or not np.all([dm.isiterable(x) for x in zones]):
        raise ValueError("The input, zones, but be an list (or other iterable) of lists (or other iterable)")

    if len(zones) == 1:
        zones = [zones[0] for x in range(len(indices))]
    else:
        if len(zones) != len(indices):
            raise ValueError("Length of zones and indices must be equivalent.")

    if select == "isolayer":
        select = "type {} and isolayer {} {} type {}"
        if not all([len(x)==2 for x in zones]):
            raise ValueError("All sublists in `zones` must be of length two")
    elif select == "around":
        select = "type {} and around {} type {}"
        if not all([len(x)==1 for x in zones]):
            raise ValueError("All sublists in `zones` must be of length one")
    else:
        try:
            tmp = select.format(indices[0][1], *zones[i], indices[0][0])
        except:
            raise ValueError("Provided select string must take 2+len(zones[0]) values to define (1) the target atom type, (2) zone boundaries, and (3) the reference atom type")

    output = []
    titles = []
    for i,ind in enumerate(indices):
        if verbose:
            print("Analyzing Survival Time of Type {} within {} Units of Type {}".format(ind[0], zones[i], ind[1]))

        sp = SP(u, select.format(ind[1], *zones[i], ind[0]), verbose=verbose)
        sp.run(stop=stop_frame, tau_max=tau_max, intermittency=intermittency, verbose=verbose)

        tau = np.array(sp.tau_timeseries)
        timeseries = np.array(sp.sp_timeseries)
        time = tau*dt
        if not output:
            output.append(time)
        output.append(timeseries)
        titles.append("{}-{}".format(*ind))

        if verbose:
            print("    Finished residence time analysis")

        if show_plot:
            plt.plot(time,timeseries,".-")
            plt.xlabel("Time")
            plt.ylabel("Probability")
            plt.title("Reference Type {}, Interacting Type {}".format(*ind))
            plt.show()

    with open(os.path.join(path,filename),"w") as f:
        f.write("# time, {}\n".format(", ".join(titles)))
        for tmp in np.transpose(np.array(output)):
            f.write("{}\n".format(", ".join([str(x) for x in tmp])))

def survival_probability_by_zones(u, dt, type_reference=None, type_target=None, zones=[(2, 5)], select=None, stop_frame=None, tau_max=100, intermittency=0, verbose=False, show_plot=False, path="", filename="survival.csv"):
    """
    Calculate the survival probability for radially dependent zones from a specified bead type (or overwrite with other selection criteria) showing distance dependent changes in mobility.

    This method requires a characteristic time, generally on the order of a picosecond. This time is equal to the beta relaxation time and is generally temperature independent, and so can be determined from the minimum of the logarithmic derivative of the MSD, and low temperature may be necessary.

    Parameters
    ----------
    u : obj/tuple
        Can either be:
        
        - MDAnalysis universe
        - A tuple of length 2 containing the format type keyword from :func:`md_spa.mdanalysis.generate_universe` and appropriate arguments in a tuple 
        - A tuple of length 3 containing the format type keyword from :func:`md_spa.mdanalysis.generate_universe`, appropriate arguments in a tuple, and dictionary of keyword arguments

    dt : float
        Define the timestep as used in ``mdanalysis.Universe``, or the number of ps between frames.
    type_reference : int, Optional, default=None
        This atom type must be provided unless ``select`` is provided. This atom type represents the core for the radial analysis. Note that with how this selection criteria is written, that if two atoms of this type are close together, a sort of iso-line is formed around the two atoms so that the DW parameter is not skewed by target atoms that are far from one reference type and close to another.
    type_target : int, Optional, default=None
        This atom type must be provided unless ``select`` is provided. The DW parameter of this atom type is the output of this function. 
    zones : list[tuple], Optional, default=[(2.0, 5.0)]
        List of tuples containing minimum and maximum zones to evaluate the survival probability
    select : str, Optional, default=None
        A string to overwrite the default selection criteria: ``type {type_target} and isolayer {zones[i][0]} {zones[i][1]} type {type_reference}``
    stop_frame : int, Optional, default=None
        Frame at which to stop calculation. This function can take a long time, so the entire trajectory may not be desired.
    tau_max : int, Optional, default=100
        Number of timesteps to calculate the decay to, this value times dt is the maximum time. See mdanalysis ``waterdynamics.SurvivalProbability``
    intermittency : int, Optional, default=0
        The intermittency specifies the number of times a hydrogen bond can be made and break while still being considered in the correlation function.
    verbose : bool, Optional, default=False
        If true, progress will be printed
    show_plot : bool, Optional, default=False
        In a debugging mode, this option allows the time autocorrelation function to be viewed
    path : str, Optional, default=""
        Path to save files
    filename : str, Optional, default="survival.csv"
        Prefix on filename to further differentiate

    Returns
    -------
    Writes out .csv file

    """

    u = check_universe(u)
    if verbose:
        print("Imported trajectory")

    if tau_max > len(u.trajectory):
        tau_max = len(u.trajectory)
        warnings.warn("tau_max is longer than given trajectory, resetting to {}".format(len(u.trajectory)))

    if select == None:
        if type_reference==None or type_target==None:
            raise ValueError("Both type_reference and type_target must be defined, respectively given: {} and {}".format(type_reference,type_target))
        select = "type {} and isolayer {} {} type {}".format(type_target, {}, {}, type_reference)
    else:
        try:
            tmp = select.format(0, 0)
        except:
            raise ValueError("Provided select string must take two values to define the zones")

    output = []
    titles = []
    for r_start, r_end in zones:
        sp = SP(u, select.format(r_start, r_end), verbose=verbose)
        sp.run(stop=stop_frame, tau_max=tau_max, intermittency=intermittency, verbose=verbose)

        tau = np.array(sp.tau_timeseries)
        timeseries = np.array(sp.sp_timeseries)
        time = tau*dt
        if not output:
            output.append(time)
        output.append(timeseries)
        titles.append("{:.3f}-{:.3f}".format(r_start, r_end))

        if verbose:
            print("    Finished survival analysis")

        if show_plot:
            plt.plot(time,timeseries,".-")
            plt.xlabel("Time")
            plt.ylabel("Probability")
            plt.title("Type {} around type {} from {} to {}".format(type_target,type_reference,r_start,r_end))
            plt.show()

    with open(os.path.join(path,filename),"w") as f:
        f.write("# time, {}\n".format(", ".join(titles)))
        for tmp in np.transpose(np.array(output)):
            f.write("{}\n".format(", ".join([str(x) for x in tmp])))


def debye_waller_by_zones(universe, frames_per_tau, select_reference=None, select_target=None, select=None, stop_frame=None, dr=1.0, r_start=2.0, nzones=5, verbose=False, write_increment=100, select_recenter=None, flag_com=False, include_center=True):
    """
    Calculate the per atom Debye-Waller (DW) parameter for radially dependent zones from a specified bead type (or overwrite with other selection criteria) showing distance dependent changes in mobility.

    This method requires a characteristic time, generally on the order of a picosecond. This time is equal to the beta relaxation time and is generally temperature independent, and so can be determined from the minimum of the logarithmic derivative of the MSD, and low temperature may be necessary.

    Note that this method should not be used for distances expected to be close to half the box length.

    Parameters
    ----------
    u : obj/tuple
        Can either be:
        
        - MDAnalysis universe
        - A tuple of length 2 containing the format type keyword from :func:`md_spa.mdanalysis.generate_universe` and appropriate arguments in a tuple 
        - A tuple of length 3 containing the format type keyword from :func:`md_spa.mdanalysis.generate_universe`, appropriate arguments in a tuple, and dictionary of keyword arguments

    frames_per_tau : int
        Number of frames in the characteristic time, tau.
    select_reference : int, Optional, default=None
        This atom type must be provided unless ``select`` is provided. This atom type represents the core for the radial analysis. Note that with how this selection criteria is written, that if two atoms of this type are close together, a sort of iso-line is formed around the two atoms so that the DW parameter is not skewed by target atoms that are far from one reference type and close to another.
    select_target : int, Optional, default=None
        This atom type must be provided unless ``select`` is provided. The DW parameter of this atom type is the output of this function. 
    select : str, Optional, default=None
        A string to overwrite the default selection criteria: ``({select_target}) and around {r_start} {r_start+i_zone*dr} ({select_reference})``
    stop_frame : int, Optional, default=None
        Frame at which to stop calculation. This function can take a long time, so the entire trajectory may not be desired.
    r_start : float, Optional, default=2.0
        Inner most radius to define the core sphere, this is Zone 0
    dr : float, Optional, default=1.0
        Thickness of the radial zones, for LAMMPS this is in angstroms
    nzones : int, Optional, default=5.0
        Number of zones, counting the central Zone 0
    verbose : bool, Optional, default=False
        If true, progress will be printed
    write_increment : int, Optional, default=100
        If ``verbose`` write out progress every, this many frames.
    select_recenter : str, Optional, default=None
        If not None, the trajectory will be centered around the select_atom() group in a relative sense. The center of mass will be extracted first to correct the difference in positions.
    flag_com : bool, Optional, default=False
        If true, the trajectory is centered around the group specified with ``select_recenter`` for every frame, and in calculating the displacement, a correction between the frame centers of mass is applied. With this option, the effect of periodic boundary conditions is ensureed to be removed for calculations sufficiently far from the boundary. 
    include_center : bool, Optional, default=True
        If True, the first shell will be a sphere from zero to ``r_start`` and the second shell will be ``r_start`` to ``r_start+dr``. If ``include_center==False``, the first shell is ``r_start+dr``

    Returns
    -------
    debye_waller_total : numpy.ndarray
        The mean squared displacement of the atoms that initially start in their respective zones after ``frames_per_tau``
    debye_waller_total_std : numpy.ndarray
        Standard deviation of ``debye_waller_total``. Note that the number of data points would make the standard trivial.
    debye_waller_retained : numpy.ndarray
        The mean squared displacement of the atoms that stay in the start and end in their respective zones after ``frames_per_tau``. 
    debye_waller_retained_std : numpy.ndarray
        Standard deviation of ``debye_waller_retained``. Note that the number of data points would make the standard trivial.
    survival_fraction : numpy.ndarray
        Fraction of atoms retained in their respective zones after ``frames_per_tau``    
    survival_fraction_std : numpy.ndarray
        Standard deviation of ``survival_fraction``, Note that the number of data points would make the standard trivial.
        
    """

    if select == None:
        if select_reference==None or select_target==None:
            raise ValueError("Both select_reference and select_target must be defined, respectively given: {} and {}".format(select_reference,select_target))
        select = "({}) and isolayer {} {} ({})".format(select_target, {}, {}, select_reference)
    else:
        try:
            tmp = select.format(0)
        except:
            raise ValueError("Provided select string must take one value to define the zones")

    universe = check_universe(universe)
    group = universe.select_atoms(select_recenter)
    lx = len(universe.trajectory)
    com = np.zeros((lx,3))
    if select_recenter != None:
        if flag_com:
            for i, ts in enumerate(universe.trajectory):
                com[i,:] = group.center_of_mass()
            universe.trajectory[0] 
            center_universe_around_group(universe, select_recenter, reference="relative")
        else:
            center_universe_around_group(universe, select_recenter, reference="initial")
        print("Recentered trajectory")
    
    if verbose:
        print("Imported trajectory")
    dimensions = universe.trajectory[0].dimensions[:3]

    frames_per_tau = int(frames_per_tau)
    if stop_frame == None:
        stop_frame = lx-frames_per_tau
    elif lx-frames_per_tau < stop_frame:
        stop_frame = lx-frames_per_tau
        if verbose:
            print("`stop_frame` has been reset to align with the trajectory length") 

    msd_total = [[] for x in range(nzones)]
    msd_retained = [[] for x in range(nzones)]
    fraction_retained = [[] for x in range(nzones)]

    for i in range(int(stop_frame)):
        if verbose and i%write_increment == 0:
            print("Calculating Frame {} out of {}".format(i,stop_frame))
        positions_start_finish = [[] for x in range(nzones)]

        # Calculate initial positions of atoms that start in respective zones
        universe.trajectory[i]
        if include_center:
            Zones = [universe.select_atoms(select.format(0, r_start))]
            zone_min = 1
        else:
            Zones = []
            zone_min = 0
        for z in range(zone_min,nzones):
            Zones.append(universe.select_atoms(select.format(r_start+dr*(z-1), r_start+dr*z)))

        for z in range(nzones):
            positions_start_finish[z].append(Zones[z].positions)

        # Calculate final positions of atoms that started in each respective zone,
        # and find atoms that are currently in respective zones (i.e. Zone2)
        universe.trajectory[i+frames_per_tau]
        if include_center:
            Zones2 = [universe.select_atoms(select.format(0, r_start))]
        else:
            Zones2 = []
        for z in range(zone_min,nzones):
            Zones2.append(universe.select_atoms(select.format(r_start+dr*(z-1), r_start+dr*z)))

        for z in range(nzones):
            if flag_com:
                positions_start_finish[z].append(Zones[z].positions + (+com[i+frames_per_tau] - com[i]))
            else:
                positions_start_finish[z].append(Zones[z].positions)
            tmp = mf.check_wrap(positions_start_finish[z][1]-positions_start_finish[z][0], dimensions)
            msd_total[z].extend(list(np.sum(np.square(tmp),axis=1)))

            tmp_ind1 = [x.ix for x in Zones[z]]
            RetainedFraction = Zones[z] - Zones[z].difference(Zones2[z])
            for atom in RetainedFraction:
                ind = tmp_ind1.index(atom.ix)
                tmp = mf.check_wrap(positions_start_finish[z][1][ind]-positions_start_finish[z][0][ind], dimensions)
                msd_retained[z].append(np.sum(np.square(tmp)))

            try: 
                fraction_retained[z].append(len(RetainedFraction)/len(Zones[z]))
            except:
                fraction_retained[z].append(0)

    debye_waller_total = np.zeros(nzones)
    debye_waller_total_std = np.zeros(nzones)
    debye_waller_retained = np.zeros(nzones)
    debye_waller_retained_std = np.zeros(nzones)
    survival_fraction = np.zeros(nzones)
    survival_fraction_std = np.zeros(nzones)
    for i in range(nzones):
        debye_waller_total[i] = np.nanmean(msd_total[i])
        debye_waller_total_std[i] = np.nanstd(msd_total[i])
        debye_waller_retained[i] = np.nanmean(msd_retained[i])
        debye_waller_retained_std[i] = np.nanstd(msd_retained[i])
        survival_fraction[i] = np.nanmean(fraction_retained[i])
        survival_fraction_std[i] = np.nanstd(fraction_retained[i])

    zone_boundaries = np.array(range(nzones))*dr+r_start

    return zone_boundaries, debye_waller_total, debye_waller_total_std, debye_waller_retained, debye_waller_retained_std, survival_fraction, survival_fraction_std


def debye_waller_by_selection(universe, frames_per_tau, select_list, stop_frame=None, verbose=False, write_increment=100):
    """
    Calculate the per atom Debye-Waller (DW) parameter for radially dependent zones from a specified bead type (or overwrite with other selection criteria) showing distance dependent changes in mobility.

    This method requires a characteristic time, generally on the order of a picosecond. This time is equal to the beta relaxation time and is generally temperature independent, and so can be determined from the minimum of the logarithmic derivative of the MSD, and low temperature may be necessary.

    Note that this method should not be used for distances expected to be close to half the box length.

    Parameters
    ----------
    u : obj/tuple
        Can either be:
        
        - MDAnalysis universe
        - A tuple of length 2 containing the format type keyword from :func:`md_spa.mdanalysis.generate_universe` and appropriate arguments in a tuple 
        - A tuple of length 3 containing the format type keyword from :func:`md_spa.mdanalysis.generate_universe`, appropriate arguments in a tuple, and dictionary of keyword arguments

    frames_per_tau : int
        Number of frames in the characteristic time, tau.
    select_list : list
        List of atom selection strings according to the MDAnalysis selection string format. 
    stop_frame : int, Optional, default=None
        Frame at which to stop calculation. This function can take a long time, so the entire trajectory may not be desired.
    verbose : bool, Optional, default=False
        If true, progress will be printed
    write_increment : int, Optional, default=100
        If ``verbose`` write out progress every, this many frames.

    Returns
    -------
    debye_waller_total : numpy.ndarray
        The mean squared displacement of the atoms that initially start in their respective zones after ``frames_per_tau``
    debye_waller_total_std : numpy.ndarray
        Standard deviation of ``debye_waller_total``. Note that the number of data points would make the standard trivial.
        
    """

    universe = check_universe(universe)
    lx = len(universe.trajectory)

    if not dm.isiterable(select_list):
        select_list = [select_list]
    ngroups = len(select_list)
    groups = []
    for select in select_list:
        try:
            groups.append(universe.select_atoms(select))
        except:
            raise ValueError("The provided selection string, {}, resulted in an error for the provided universe.".format(select))

    if verbose:
        print("Imported trajectory")
    dimensions = universe.trajectory[0].dimensions[:3]

    frames_per_tau = int(frames_per_tau)
    if stop_frame == None:
        stop_frame = lx-frames_per_tau
    elif lx-frames_per_tau < stop_frame:
        stop_frame = lx-frames_per_tau
        if verbose:
            print("`stop_frame` has been reset to align with the trajectory length")

    msd_total = [[] for x in range(ngroups)]
    for i in range(int(stop_frame)):
        if verbose and i%write_increment == 0:
            print("Calculating Frame {} out of {}".format(i,stop_frame))
        positions_start_finish = [[] for x in range(ngroups)]

        # Calculate initial positions of atoms
        universe.trajectory[i]
        for j, group in enumerate(groups):
            groups[j] = universe.select_atoms(select_list[j])
            positions_start_finish[j].append(group.positions)

        # Calculate final positions of atoms
        universe.trajectory[i+frames_per_tau]
        for j, group in enumerate(groups):
            tmp = mf.check_wrap(group.positions-positions_start_finish[j][0], dimensions)
            msd_total[j].extend(list(np.sum(np.square(tmp),axis=1)))

    debye_waller_total = np.zeros(ngroups)
    debye_waller_total_std = np.zeros(ngroups)
    for i in range(ngroups):
        debye_waller_total[i] = np.nanmean(msd_total[i])
        debye_waller_total_std[i] = np.nanstd(msd_total[i])

    return debye_waller_total, debye_waller_total_std


def tetrahedral_order_parameter_by_zone(universe, select_target, select_reference, select_neighbor, step=None, select=None, dr=1.0, r_start=2.0, nzones=5, stop_frame=None, skip_frame=1, verbose=False, write_increment=100, bins=100, kwargs_metric={}, include_center=True):
    """
    Calculate the per atom tetrahedral order parameter for radially dependent zones from a specified bead type (or overwrite with other selection criteria) showing distance dependent changes in structure.

    Notice that if more than four atoms are found, a warning is issued and the closest four are used. In the case that there are less than four atoms within the cutoff, the metric is still computed for n==3, but reported in a separate matrix.

    Note that the reference select should encompass atom types, and not regions for this function to be meaningful.

    Parameters
    ----------
    u : obj/tuple
        Can either be:
        
        - MDAnalysis universe
        - A tuple of length 2 containing the format type keyword from :func:`md_spa.mdanalysis.generate_universe` and appropriate arguments in a tuple 
        - A tuple of length 3 containing the format type keyword from :func:`md_spa.mdanalysis.generate_universe`, appropriate arguments in a tuple, and dictionary of keyword arguments

    select_target : list
        Atom type selection string according to the MDAnalysis selection string format. 
    select_reference : int, Optional, default=None
        This atom type must be provided unless ``select`` is provided. This atom type represents the core for the radial analysis. Note that with how this selection criteria is written, that if two atoms of this type are close together, a sort of iso-line is formed around the two atoms so that the DW parameter is not skewed by target atoms that are far from one reference type and close to another.
    select_neighbor : str
        Selection string restricting the atom types to be included. For example, ``reference_select="type 2 and type 4"``, although one atom type is probably recommended.
    step : int, Optional, default=None
        Optionally evaulate for a single step, this will overwrite ``r_cut, stop_frame, skip_frame``
    select : str, Optional, default=None
        A string to overwrite the default selection criteria: ``({select_target}) and around {r_start} {r_start+i_zone*dr} ({select_reference})``
    r_start : float, Optional, default=2.0
        Inner most radius to define the core sphere, this is Zone 0
    dr : float, Optional, default=1.0
        Thickness of the radial zones, for LAMMPS this is in angstroms
    nzones : int, Optional, default=5.0
        Number of zones, counting the central Zone 0
    stop_frame : int, Optional, default=None
        Frame at which to stop calculation. This function can take a long time, so the entire trajectory may not be desired.
    skip_frame : int, Optional, default=1
        Interval of frames to skip. Advised if given universe frames are not independent.
    verbose : bool, Optional, default=False
        If true, progress will be printed
    write_increment : int, Optional, default=100
        If ``verbose`` write out progress every, this many frames.
    bins = int, Optional, default=100
        Number of binds used in histogram. Can be any valid input for ``numpy.histogram(x, bins=bins)``
    kwargs_metric : dict, Optional, default={}
        Keyword arguments for execution of `tetrahedral_order_parameter`
    include_center : bool, Optional, default=True
        If True, the first shell will be a sphere from zero to ``r_start`` and the second shell will be ``r_start`` to ``r_start+dr``. If ``include_center==False``, the first shell is ``r_start+dr``

    Returns
    -------
    q_hist : numpy.ndarray
        The tetrahedral order parameter histogram for a given selection criteria, matrix where the number of rows equals the number of selection strings plus one, by the number of bins.
    q_hist_interface : numpy.ndarray
        The tetrahedral order parameter histogram for a given selection criteria, matrix where the number of rows equals the number of selection strings plus one, by the number of bins. Computed from values that do not have up to four adjacent atoms. This is most likely at an interface.
        
    """

    if select == None:
        if select_reference==None or select_target==None:
            raise ValueError("Both select_reference and select_target must be defined, respectively given: {} and {}".format(select_reference,select_target))
        select = "({}) and isolayer {} {} ({})".format(select_target, {}, {}, select_reference)
    else:
        try:
            tmp = select.format(0)
        except:
            raise ValueError("Provided select string must take one value to define the zones")

    universe = check_universe(universe)
    lx = len(universe.trajectory)

    if verbose:
        print("Imported trajectory")

    if step == None:
        if stop_frame == None or lx < stop_frame:
            stop_frame = lx
            if verbose:
                print("`stop_frame` has been reset to align with the trajectory length, {}".format(lx))
        timesteps = range(0,int(stop_frame),int(skip_frame))
    else:
        timesteps = [step]

    q_dict = [{} for x in range(nzones)]
    for i in timesteps:
        if verbose and i%write_increment == 0:
            print("Calculating Frame {} out of {}".format(i,stop_frame))

        universe.trajectory[i]
        if include_center:
            groups = [universe.select_atoms(select.format(0, r_start))]
            zone_min = 1
        else:
            groups = []
            zone_min = 0
        for z in range(zone_min,nzones):
            groups.append(universe.select_atoms(select.format(r_start+dr*(z-1), r_start+dr*z)))

        for j, group in enumerate(groups):
            q_tmp = tetrahedral_order_parameter(universe, group, select_neighbor, **kwargs_metric)
            for key, values in q_tmp.items():
                if key not in q_dict[j]:
                    q_dict[j][key] = [] 
                q_dict[j][key].extend(values)

    q_out = {}
    for i in range(nzones):
        for q, values in q_dict[i].items():
            hist, bin_edges = np.histogram(values, bins=bins, range=(0.,1.))
            if i == 0:
                bin_centers = (bin_edges[1:]+bin_edges[:-1])/2
                q_hist = np.zeros((len(bin_centers),len(q_dict[i])+1))
                q_hist[:,0] = bin_centers
            q_hist[:,i+1] = hist

        if i == 0:
            key = f"0.00-{r_start:.2f}"
        else:
            key = "{.2f}-{.2f}".format(r_start+dr*(i-1), r_start+dr*i)
        q_out[key] = {
                      "q_hist": q_hist,
                      "q_frac": np.sum(q_hist[:,1:], axis=1)/np.sum(q_hist[:,1:]),
                      "q_coord": list(q_dict[i].keys())
                     }

    return q_out

def tetrahedral_order_parameter(universe, group, select_neighbor, r_cut=3.4, angle_cut=30):
    """
    Calculate the per atom tetrahedral order parameter for radially dependent zones from a specified bead type (or overwrite with other selection criteria) showing distance dependent changes in structure.

    Notice that if more than four atoms are found, a warning is issued and the closest four are used to calculate the metric. However, the parameter is still categorized by the number of neighbors meet the geometric criteria, based off of

    Note that the reference select should encompass atom types, and not regions for this function to be meaningful.

    Parameters
    ----------
    u : obj/tuple
        Can either be:
        
        - MDAnalysis universe
        - A tuple of length 2 containing the format type keyword from :func:`md_spa.mdanalysis.generate_universe` and appropriate arguments in a tuple 
        - A tuple of length 3 containing the format type keyword from :func:`md_spa.mdanalysis.generate_universe`, appropriate arguments in a tuple, and dictionary of keyword arguments

    group : list
        Atom group to be analyzed 
    select_neighbor : str
        Selection string restricting the atom types to be included. For example, ``reference_select="type 2 and type 4"``, although one atom type is probably recommended.
    r_cut : float, Optional, default=3.4
        Cutoff used to determine ``reference_atoms`` used in calculation.
    angle_cut :float, Optional, default=30 
        Cutoff for angle between interacting hydrogen, its donor, and the acceptor oxygen in the tetrahedral. A value of None will remove this contraint.

    Returns
    -------
    q_dict : dictionary
         Dictionary of tetrahedral order parameters categorized by their coordination number
        
    """

    universe = check_universe(universe)
    lx = len(universe.trajectory)
    dimensions = universe.trajectory[0].dimensions[:3]

    q_dict = {}
    for k, atm in enumerate(group):
        atm_pos = universe.select_atoms("index {}".format(atm.ix))[0].position
        atoms = universe.select_atoms("({}) ".format(select_neighbor)+"and around {} index {}".format(r_cut, atm.ix))

        remove = []
        if atm.bonded_atoms:
            for atm2 in atoms:
                if len(atm.bonded_atoms) != 2 or len(atm2.bonded_atoms) != 2:
                    raise ValueError("This function has been written for use with water. Feel free to contribute changes to generalize this function.")
                pos = np.zeros((4,3,3))
                pos[:2,0,:] = atm.bonded_atoms.positions
                pos[:2,1,:] = atm2.position
                pos[:2,2,:] = atm.position
                pos[2:,0,:] = atm2.bonded_atoms.positions
                pos[2:,1,:] = atm.position
                pos[2:,2,:] = atm2.position
                angles = np.array([mf.angle(x[2],x[0],x[1]) for x in pos])
                dist = np.array([[mf.check_wrap([x[0]-x[2]], dimensions)[0], mf.check_wrap([x[1]-x[2]], dimensions)[0]] for x in pos])
                dist = np.array([np.sqrt(np.sum(np.square(x), axis=1)) for x in dist])
                if angle_cut is None:
                    if np.all([ dist[x][0] > dist[x][1] for x in range(len(angles))]):
                        remove.append(atm2)
                else:
                    if np.all([ dist[x][0] > dist[x][1] or angles[x] >= angle_cut for x in range(len(angles))]):
                        remove.append(atm2)
            atoms -= mda.AtomGroup([x.ix for x in remove],universe)

        positions = mf.check_wrap(atoms.positions - atm_pos, dimensions)
        if np.any(np.sqrt(np.sum(np.square(positions), axis=1)) > r_cut):
            raise ValueError("Neighbor atoms are not within cutoff")

        l_atoms = len(atoms)
        if len(atoms) > 4:
            warnings.warn("tetrahedral order parameter: cutoff produced more than four adjacent atoms.")
            dist = np.sum(positions**2, axis=1)
            ind = sorted([x for _, x in sorted(zip(dist,list(range(len(atoms)))))][:4])
            positions = positions[ind]
        lx = len(positions)

        positions /= np.sqrt(np.sum(np.square(positions),axis=1))[:,np.newaxis]
        tmp_cos = [(np.dot(positions[x],positions[y])+1.0/3.0)**2 for x in range(lx) for y in range(x+1,lx)]
        tmp = 1.0 - 3.0/8.0*np.sum([(np.dot(positions[x],positions[y])+1.0/3.0)**2 for x in range(lx-1) for y in range(x+1,lx)])

        if l_atoms not in q_dict:
            q_dict[l_atoms] = []
        q_dict[l_atoms].append(tmp)

    return q_dict

def create_ndx(groups=[], names=None, separate_by=None, filename="index.ndx", kwargs_writer={}):
    """ Write a Gromacs style ndx file from a list of atom groups.

    This function can be used to prepare a calculation in the scattering function package, `dynasor <https://dynasor.materialsmodeling.org/index.html>`_

    Parameters
    ----------
    u : obj/tuple
        Can either be:
        
        - MDAnalysis universe
        - A tuple of length 2 containing the format type keyword from :func:`md_spa.mdanalysis.generate_universe` and appropriate arguments in a tuple 
        - A tuple of length 3 containing the format type keyword from :func:`md_spa.mdanalysis.generate_universe`, appropriate arguments in a tuple, and dictionary of keyword arguments

    groups : list, Optional, default=[]
        List of ``AtomGroups`` to be included in the index file.
    names : list, Optional, default=None
        List of equal length to ``groups`` to separate different "molecules" (i.e., groups) to call in gromacs
    separate_by : str, Optional, default=None
        Subdivide groups with this mdanalysis `selection string <https://docs.mdanalysis.org/stable/documentation_pages/selections.html>`_. Could be any value used in AtomGroup.groupby method.
    kwargs_writer : dict, Optional, default={}
        Keyword arguments used in the `Gromacs writer <https://docs.mdanalysis.org/documentation_pages/selections/gromacs.html>`_ 

    Returns
    -------
    Saved index file in gromacs style.

    """
    if not groups:
        raise ValueError("No groups were provided for writing an index file.")
    if len(groups) > 1 and names is None:
        warnings.warn("More than one AtomGroup is provided without differentiating `names`.")
    elif names is not None and len(groups) != len(names):
        raise ValueError("The number of `AtomGroups` must equal the number of `names`")

    with mda.selections.gromacs.SelectionWriter(filename, **kwargs_writer) as ndx:
        for i, group in enumerate(groups):
            if separate_by is None:
                if names is not None:
                    ndx.write(group, name=names[i])
                else:
                    ndx.write(group)
            else:
                subgroups = group.groupby(separate_by)
                for sel, subgroup in subgroups.items():
                    name = "{}_{}".format(separate_by[:-1], sel)
                    if names is not None:
                        ndx.write(subgroup, name=names[i]+"_"+name)
                    else:
                        ndx.write(subgroup, name=name)

def add_elements2types(universe, conversion_dict):
    """ Add elements to universe based on atom type

    This function is useful to write out lammps files in the xyz format.

    Parameters
    ----------
    u : obj/tuple
        Can either be:
        
        - MDAnalysis universe
        - A tuple of length 2 containing the format type keyword from :func:`md_spa.mdanalysis.generate_universe` and appropriate arguments in a tuple 
        - A tuple of length 3 containing the format type keyword from :func:`md_spa.mdanalysis.generate_universe`, appropriate arguments in a tuple, and dictionary of keyword arguments

    conversion_dict : dict
        Dictionary defining the element for each type, e.g., ``{"1": "H", "2": "O"}``
    
    Returns
    -------
    universe : obj
        universe with elements 

    """

    universe = check_universe(universe)

    group = universe.select_atoms("all")
    group_by_type = group.groupby("types")

    elements = ["X"]*universe.atoms.n_atoms
    for atm_type, subgroup in group_by_type.items():
        if atm_type in conversion_dict:
            for i in subgroup.ix:
                elements[i] = conversion_dict[atm_type]
        else:
            warnings.warn("Atom type, {}, is not found in the provided `conversion_dict`, elements are defined as 'X',".format(atm_type))

    universe.add_TopologyAttr('elements', elements)
    return universe



