
import numpy as np
import sys
import os

import MDAnalysis as mda
from MDAnalysis.analysis import polymer
from MDAnalysis.analysis import rdf as mda_rdf
from MDAnalysis.transformations import unwrap
from MDAnalysis.analysis.hydrogenbonds.hbond_analysis import HydrogenBondAnalysis as HBA
from MDAnalysis.analysis.waterdynamics import SurvivalProbability as SP

import md_spa_utils.os_manipulation as om

def calc_partial_rdf(data_file, dump_file, groups, rmin=0.0, rmax=12.0, nbins=1000, verbose=False, exclusion_block=None, run_kwargs={}, universe_kwargs={}):
    """
    Calculate the partial rdf of a polymer given a LAMMPS data file, dump file(s), and a list of the atomIDs along the backbone (starting at 1).

    Parameters
    ----------
    data_file : str
        LAMMPS data file with "full" formatting
    dump_file : list
        The name of a dump file or list of dump files in "atom" formatting containing trajectories pertaining to `data_file`.
    groups : list[tuples]
        List of tuples containing pairs of strings that individually identify a MDAnalysis AtomGroup with universe.select_atoms(groups[i])
    rmin : float, Optional, default=0.0
        Minimum rdf cutoff
    rmax : float, Optional, default=12.0
        Maximum rdf cutoff
    nbins : 100, Optional, default=1000
        Number of bins in the rdf
    verbose : bool, Optional, default=False
        Set whether calculation will be run with comments
    exclusion_block : tuple, Optional, default=None
        A tuple representing the tile to exclude from the distance array.
    run_kwargs : dict, Optional, default={}
        Other keyword arguments from MDAnalysis.analysis.rdf.InterRDF.run()
    universe_kwargs :dict, Optional, default={}
        Other keyword arguments from MDAnalysis.Universe.universe

    Returns
    -------
    rdf_output : numpy.ndarray
        Array of distances and rdf values. rdf_output[0] is the distances and the remaining rows represent the given groups pairs.

    """

    # Extract System information and trajectory
    u = mda.Universe(data_file, dump_file, format="LAMMPSDUMP",**universe_kwargs)

    if not om.isiterable(groups):
        raise ValueError("The entry `groups` must be an iterable structure with selection strings.")

    group_dict = {}
    rdf_output = np.zeros((len(groups)+1,nbins))
    flag_bins = False
    for i, (group1, group2) in enumerate(groups):
        if group1 not in group_dict:
            group_dict[group1] = u.select_atoms(group1)
        if group2 not in group_dict:
            group_dict[group2] = u.select_atoms(group2)
        Rdf = mda_rdf.InterRDF(group_dict[group1],group_dict[group2],nbins=nbins,range=(rmin,rmax),verbose=verbose)
        Rdf.run(verbose=verbose, **run_kwargs)
        if not flag_bins:
            flag_bins = True
            rdf_output[0] = Rdf.results.bins
        rdf_output[i+1] = Rdf.results.rdf

    return rdf_output

def calc_persistence_length(data_file, dump_file, backbone_indices, save_plot=True, figure_name="plot_lp_fit.png", verbose=True):
    """
    Calculate the persistence length of a polymer given a LAMMPS data file, dump file(s), and a list of the atomIDs along the backbone (starting at 1).

    Parameters
    ----------
    data_file : str
        LAMMPS data file with "full" formatting
    dump_file : list
        The name of a dump file or list of dump files in "atom" formatting containing trajectories pertaining to `data_file`.
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

    u = mda.Universe(data_file, dump_file, format="LAMMPSDUMP")
    
    tmp_indices = [x-1 for x in backbone_indices]
    #tmp_indices = backbone_indices
    backbone = mda.core.groups.AtomGroup(tmp_indices,u)
    sorted_bb = polymer.sort_backbone(backbone) # Should be unnecessary if backbone_indices are in order
    
    plen = polymer.PersistenceLength([sorted_bb])
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

def calc_gyration(data_file, dump_file, select="all"):
    """
    Calculate the radius of gyration and anisotropy of a polymer given a LAMMPS data file, dump file(s).

    Parameters
    ----------
    data_file : str
        LAMMPS data file with "full" formatting
    dump_file : list
        The name of a dump file or list of dump files in "atom" formatting containing trajectories pertaining to `data_file`.
    select : str, Optional, default="all"
        This string should align with MDAnalysis universe.select_atoms(select) rules

    Returns
    -------
    rg : numpy.ndarray
        The radius of gyration is calculated for the provided dump file
    kappa : numpy.ndarray
        The anisotropy is calculated from the gyration tensor, where zero indicates a sphere and one indicates an infinitely long rod.

    """

    u = mda.Universe(data_file, dump_file, format="LAMMPSDUMP")
    ag = u.atoms
    u.trajectory.add_transformations(unwrap(ag))
    group = u.select_atoms(select)  # a selection (a AtomGroup)
    if not np.any(group._ix):
        raise ValueError("No atoms met the selection criteria")

    rgyr = np.zeros(len(u.trajectory))
    kappa = np.zeros(len(u.trajectory))
    anisotropy = np.zeros(len(u.trajectory))
    for i,ts in enumerate(u.trajectory):  # iterate through all frames
        rgyr[i] = group.radius_of_gyration()  # method of a AtomGroup; updates with each frame
        kappa[i], anisotropy[i] = group.anisotropy()  # method of a AtomGroup; updates with each frame
    
    return rgyr, kappa, anisotropy

def calc_end2end(data_file, dump_file, indices):
    """
    Calculate the persistence length of a polymer given a LAMMPS data file, dump file(s), and a list of the atomIDs along the backbone (starting at 1).

    Parameters
    ----------
    data_file : str
        LAMMPS data file with "full" formatting
    dump_file : list
        The name of a dump file or list of dump files in "atom" formatting containing trajectories pertaining to `data_file`.
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
        

    u = mda.Universe(data_file, dump_file, format="LAMMPSDUMP")
    ag = u.atoms
    u.trajectory.add_transformations(unwrap(ag))
    tmp_indices = [x-1 for x in indices]

    r_end2end = np.zeros(len(u.trajectory))
    for i,ts in enumerate(u.trajectory):  # iterate through all frames
        r = u.atoms.positions[indices[0]] - u.atoms.positions[indices[1]]  # end-to-end vector from atom positions
        r_end2end[i] = np.linalg.norm(r)   # end-to-end distance


    return r_end2end

def hydrogen_bonding(data_file, dump_file, indices, dt, tau_max=100, verbose=False, show_plot=False, intermittency=0, path="", file_prefix=""):
    """
    Calculation the hydrogen bonding statistics for the given type interactions.

    Parameters
    ----------
    data_file : str
        LAMMPS data file with "full" formatting
    dump_file : list
        The name of a dump file or list of dump files in "atom" formatting containing trajectories pertaining to `data_file`.
    indices : list
        A list of lists containing the type for the hydrogen bond donor, hydrogen, and acceptor. If an atom type is unknown or more than one is desired, put None instead.
    dt : float
        Define the timestep as used in ``mdanalysis.Universe``, or the number of ps between frames.
    tau_max : int, Optional, default=100
        Number of timesteps to calculate the decay to, this value times dt is the maximum time.
    verbose : bool, Optional, default=False
        Print intermediate updates on progress
    show_plot : bool, Optional, default=False
        In a debugging mode, this option allows the time autocorrelation function to be viewed
    intermittency : int, Optional, default=0
        The intermittency specifies the number of times a hydrogen bond can be made and break while still being considered in the correlation function.
    path : str, Optional, default=""
        Path to save files
    file_prefix : str, Optional, default=""
        Prefix on filename to further differentiate

    Returns
    -------
    Writes out .csv file: "{file_prefix}res_time_#typedonor_#typehydrogen_#typeacceptor.csv", where the type number is zero if a type is defined as None.

    """

    if verbose:
        for ind in indices:
            print("Lifetime stastistics of donor type {}, hydrogen type {}, and acceptor type {}".format(*ind))

    if om.isiterable(indices):
        if not om.isiterable(indices[0]):
            indices = [indices]
    else:
        raise ValueError("Input, indices, but be a list of iterables of length three")

    u = mda.Universe(data_file, dump_file, format="LAMMPSDUMP", dt=dt)
    if verbose:
        print("Imported trajectory")

    Hbonds = HBA(universe=u)
    acceptor_list = [x for x in Hbonds.guess_acceptors().split() if x not in ["type", "or"]]
    hydrogen_list = [x for x in Hbonds.guess_hydrogens().split() if x not in ["type", "or"]]
    donor_list_per_hydrogen = {}
    for hyd_type in hydrogen_list:
        Hbonds = HBA(universe=u, hydrogens_sel="type {}".format(hyd_type))
        donor_list_per_hydrogen[hyd_type] = [x for x in Hbonds.guess_donors().split() if x not in ["type", "or"]]

    new_indices = []
    for i in range(len(indices)):
        if len(indices[i]) != 3:
            raise ValueError("Index set: {}, must be of length three containing the atoms types or None for [donor, hydrogen, acceptor]")

        if indices[i][1] == None:
            hydrogens = hydrogen_list
        else:
            if om.isiterable(indices[i][1]):
                hydrogens = [str(x) for x in indices[i][1]]
            else:
                hydrogens = [str(indices[i][1])]

        if indices[i][2] == None:
            acceptors = acceptor_list
        else:
            if om.isiterable(indices[i][2]):
                acceptors = [str(x) for x in indices[i][2]]
            else:
                acceptors = [indices[i][2]]

        for h in hydrogens:
            for d in donor_list_per_hydrogen[h]:
                for a in acceptors:
                    new_indices.append([d,h,a])

    for ind in new_indices:
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
        Hbonds = HBA(universe=u, donors_sel=donor, hydrogens_sel=hydrogen, acceptors_sel=acceptor)
        Hbonds.run()
        if verbose:
            print("    Primary analysis complete")

        # Lifetime analysis
        tau, timeseries = Hbonds.lifetime(tau_max=tau_max,  intermittency=intermittency)
        time = tau*dt
        ind = [x if x != None else 0 for x in ind]
        with open(os.path.join(path,"{}res_time_{}_{}_{}.csv".format(file_prefix,*ind)),"w") as f:
            f.write("# time, probability\n")
            for i in range(len(time)):
                f.write("{}, {}\n".format(time[i],timeseries[i]))

        if verbose:
            print("    Finished lifetime analysis")
        
        if show_plot:
            plt.plot(time,timeseries,".-")
            plt.xlabel("Time")
            plt.ylabel("Probability")
            plt.title("Donor Type {}, Hydrogen Type {}, Acceptor Type {}".format(*ind))
            plt.show()

def survival_probability(data_file, dump_file, dt, type_reference=None, type_target=None, r_start=2.0, r_end=5, select=None, stop_frame=None, tau_max=100, intermittency=0, verbose=False, show_plot=False, path="", file_prefix=""):
    """
    Calculate the per atom Debye-Waller (DW) parameter for radially dependent zones from a specified bead type (or overwrite with other selection criteria) showing distance dependent changes in mobility.

    This method requires a characteristic time, generally on the order of a picosecond. This time is equal to the beta relaxation time and is generally temperature independent, and so can be determined from the minimum of the logarithmic derivative of the MSD, and low temperature may be necessary.

    Parameters
    ----------
    data_file : str
        LAMMPS data file with "full" formatting
    dump_file : list
        The name of a dump file or list of dump files in "atom" formatting containing trajectories pertaining to `data_file`.
    dt : float
        Define the timestep as used in ``mdanalysis.Universe``, or the number of ps between frames.
    type_reference : int, Optional, default=None
        This atom type must be provided unless ``select`` is provided. This atom type represents the core for the radial analysis. Note that with how this selection criteria is written, that if two atoms of this type are close together, a sort of iso-line is formed around the two atoms so that the DW parameter is not skewed by target atoms that are far from one reference type and close to another.
    type_target : int, Optional, default=None
        This atom type must be provided unless ``select`` is provided. The DW parameter of this atom type is the output of this function. 
    r_start : float, Optional, default=2.0
        Inner most radius to define the sphere
    r_end : float, Optional, default=5
        Outer most radius to define the sphere
    select : str, Optional, default=None
        A string to overwrite the default selection criteria: ``type {type_target} and around {r_start+i_zone*dr} type {type_reference}``
    stop_frame : int, Optional, default=None
        Frame at which to stop calculation. This function can take a long time, so the entire trajectory may not be desired.
    tau_max : int, Optional, default=100
        Number of timesteps to calculate the decay to, this value times dt is the maximum time.
    intermittency : int, Optional, default=0
        The intermittency specifies the number of times a hydrogen bond can be made and break while still being considered in the correlation function.
    verbose : bool, Optional, default=False
        If true, progress will be printed
    show_plot : bool, Optional, default=False
        In a debugging mode, this option allows the time autocorrelation function to be viewed
    path : str, Optional, default=""
        Path to save files
    file_prefix : str, Optional, default=""
        Prefix on filename to further differentiate

    Returns
    -------
    Writes out .csv file: "{file_prefix}res_time_#typedonor_#typehydrogen_#typeacceptor.csv", where the type number is zero if a type is defined as None.

    """

    if select == None:
        if type_reference==None or type_target==None:
            raise ValueError("Both type_reference and type_target must be defined, respectively given: {} and {}".format(type_reference,type_target))
#        select = "type {} and around {} type {} and not around {} type {}".format(type_target, r_start, type_reference, r_end, type_reference)
        select = "type {} and sphlayer {} {} type {}".format(type_target, r_start, r_end, type_reference)
#        select = "type {} and around {} type {}".format(type_target, r_start, type_reference)
#        select = "type {} and around {} type {}".format(type_target, r_end, type_reference)
    else:
        try:
            tmp = select.format(0)
        except:
            raise ValueError("Provided select string must take one value to define the zones")

    u = mda.Universe(data_file, dump_file, format="LAMMPSDUMP", dt=dt)
    if verbose:
        print("Imported trajectory")

    sp = SP(u, select, verbose=verbose)
    sp.run(stop=stop_frame, tau_max=tau_max, intermittency=intermittency, verbose=verbose)

    tau = np.array(sp.tau_timeseries)
    timeseries = np.array(sp.sp_timeseries)
    time = tau*dt

    with open(os.path.join(path,"{}survival_{}_around_{}_{}_to_{}.csv".format(file_prefix,type_target,type_reference,int(r_start),int(r_end))),"w") as f:
        f.write("# For type {} around type {} from {} to {} \n# time, probability\n".format(type_target,type_reference,r_start,r_end))
        for i in range(len(time)):
            f.write("{}, {}\n".format(time[i],timeseries[i]))

    if verbose:
        print("    Finished survival analysis")

    if show_plot:
        plt.plot(time,timeseries,".-")
        plt.xlabel("Time")
        plt.ylabel("Probability")
        plt.title("Type {} around type {} from {} to {}".format(type_target,type_reference,r_start,r_end))
        plt.show()



def debye_waller_by_zones(data_file, dump_file, frames_per_tau, type_reference=None, type_target=None, select=None, stop_frame=None, dr=1.0, r_start=2.0, nzones=5, verbose=False):
    """
    Calculate the per atom Debye-Waller (DW) parameter for radially dependent zones from a specified bead type (or overwrite with other selection criteria) showing distance dependent changes in mobility.

    This method requires a characteristic time, generally on the order of a picosecond. This time is equal to the beta relaxation time and is generally temperature independent, and so can be determined from the minimum of the logarithmic derivative of the MSD, and low temperature may be necessary.

    Parameters
    ----------
    data_file : str
        LAMMPS data file with "full" formatting
    dump_file : list
        The name of a dump file or list of dump files in "atom" formatting containing trajectories pertaining to `data_file`.
    frames_per_tau : int
        Number of frames in the characteristic time, tau.
    type_reference : int, Optional, default=None
        This atom type must be provided unless ``select`` is provided. This atom type represents the core for the radial analysis. Note that with how this selection criteria is written, that if two atoms of this type are close together, a sort of iso-line is formed around the two atoms so that the DW parameter is not skewed by target atoms that are far from one reference type and close to another.
    type_target : int, Optional, default=None
        This atom type must be provided unless ``select`` is provided. The DW parameter of this atom type is the output of this function. 
    select : str, Optional, default=None
        A string to overwrite the default selection criteria: ``type {type_target} and around {r_start+i_zone*dr} type {type_reference}``
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
        if type_reference==None or type_target==None:
            raise ValueError("Both type_reference and type_target must be defined, respectively given: {} and {}".format(type_reference,type_target))
        select = "type {} and isolayer {} {} type {}".format(type_target, {}, {}, type_reference)
    else:
        try:
            tmp = select.format(0)
        except:
            raise ValueError("Provided select string must take one value to define the zones")

    u = mda.Universe(data_file, dump_file, format="LAMMPSDUMP")
    if verbose:
        print("Imported trajectory")

    if stop_frame == None:
        stop_frame = len(u.trajectory)-frames_per_tau
    elif len(u.trajectory)-frames_per_tau < stop_frame:
        stop_frame = len(u.trajectory)-frames_per_tau
        if verbose:
            print("`stop_frame` has been reset to align with the trajectory length") 

    msd_total = [[] for x in range(nzones)]
    msd_retained = [[] for x in range(nzones)]
    fraction_retained = [[] for x in range(nzones)]
    for i in range(stop_frame):
        if verbose:
            print("Calculating Frame {} out of {}".format(i,stop_frame))
        positions_start_finish = [[] for x in range(nzones)]

        # Calculate initial positions of atoms that start in respective zones
        u.trajectory[i]
        Zones = [u.select_atoms(select.format(0, r_start))]
        for z in range(1,nzones):
            Zones.append(u.select_atoms(select.format(r_start+dr*(z-1), r_start+dr*z)))

        for z in range(nzones):
            positions_start_finish[z].append(Zones[z].positions)

        # Calculate final positions of atoms that started in each respective zone,
        # and find atoms that are currently in respective zones (i.e. Zone2)
        u.trajectory[i+frames_per_tau]
        Zones2 = [u.select_atoms(select.format(0, r_start))]
        for z in range(1,nzones):
            Zones2.append(u.select_atoms(select.format(r_start+dr*(z-1), r_start+dr*z)))

        for z in range(nzones):
            positions_start_finish[z].append(Zones[z].positions)
            msd_total[z].extend(list(np.sum(np.square(positions_start_finish[z][1]-positions_start_finish[z][0]),axis=1)))

            tmp_ind1 = [x.ix for x in Zones[z]]
            RetainedFraction = Zones[z] - Zones[z].difference(Zones2[z])
            for atom in RetainedFraction:
                ind = tmp_ind1.index(atom.ix)
                msd_retained[z].append(np.sum(np.square(
                    positions_start_finish[z][1][ind]-positions_start_finish[z][0][ind]
                )))

            fraction_retained[z].append(len(RetainedFraction)/len(Zones[z]))

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

    return debye_waller_total, debye_waller_total_std, debye_waller_retained, debye_waller_retained_std, survival_fraction, survival_fraction_std


