
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

import md_spa_utils.data_manipulation as dm

def center_universe_around_group(universe, select, verbose=False, reference="initial"):
    """

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

    Returns
    -------
    universe : obj 
        ``MDAnalysis.Universe`` object instance

    """

    group_target = universe.select_atoms(select)
    group_remaining = universe.select_atoms("all") - group_target
    
    if verbose:
        print("Polymer Center of Mass Before", group_target.center_of_mass())

    if reference == "initial":
        transforms = [
            trans.unwrap(group_target),
            trans.center_in_box(group_target),
            trans.wrap(group_remaining)
        ]
        universe.trajectory.add_transformations(*transforms)
    elif reference == "relative":
        box_center = universe.trajectory[0].dimensions[:3] / 2
        for ts in universe.trajectory:
            universe.atoms.translate(box_center - group_target.center_of_mass(pbc=True))
            group_remaining.wrap()
    else:
        raise ValueError("Transformation reference, {}, is not supported.".format(reference))

    if verbose:
        print("Group Center of Mass After", group_target.center_of_mass())

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

    Parameters
    ----------
    package : str
        String to indicate the universe ``format`` to interpret. The supported formats are:
            
        - LAMMPS: Must include the path to a data file with the atom_type full (unless otherwise specified.), and optionally include a list of paths to sequential dump files.           

    args : tuple
        Arguments for ``MDAnalysis.Universe``
    kwargs : dict, Optional, default={"format": package_identifier}
        Keyword arguments for ``MDAnalysis.Universe``. Note that the ``format`` will be added internally.
                 
    Returns
    -------
    universe : obj
        ``MDAnalysis.Universe`` object instance

    """

    supported_packages = ["LAMMPS"]

    if package == "LAMMPS":
        kwargs["format"] = "LAMMPSDUMP"
        u = mda.Universe(*args, **kwargs)
    else:
        raise ValueError("The package, {}, is not supported, choose one of the following: {}".format(package,", ".join(supported_packages)))

    return u
        
def calc_partial_rdf(u, groups, rmin=0.1, rmax=12.0, nbins=1000, verbose=False, exclusion_block=(1,1), exclusion_mode="block", exclude=True, run_kwargs={}):
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
    exclusion_block : tuple, Optional, default=(1,1)
        Allows the masking of pairs from within the same molecule.  For example, if there are 7 of each atom in each molecule, the exclusion mask `(7, 7)` can be used. The default removes self interaction.
    exclusion_mode : str, Optional, default="block"
        Set to "block" for traditional mdanalysis use, and use "relative" to remove ``exclusion_block[0]`` neighbors around reference atom.
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

def calc_msds(u, groups, dt=1, verbose=False, run_kwargs={}):
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
    nongaussian_parameter = np.zeros((len(groups)+1,nbins))
    flag_bins = False
    for i, group in enumerate(groups):
        MSD = mda_msd.EinsteinMSD(u, select=group, msd_type='xyz', fft=True)
        MSD.run(verbose=verbose, **run_kwargs)
        if verbose:
            print("Generated MSD for group {}".format(group))

        if not flag_bins:
            flag_bins = True
            msd_output[0] = np.arange(MSD.n_frames)*dt # ps
            nongaussian_parameter[0] = np.arange(MSD.n_frames)*dt # ps
        msd_output[i+1] = MSD.results.timeseries # angstroms^2
        nongaussian_parameter[i+1] = MSD.results.second_order_nongaussian_parameter

    return msd_output, nongaussian_parameter

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

def calc_gyration(u, select="all"):
    """
    Calculate the radius of gyration and anisotropy of a polymer given a LAMMPS data file, dump file(s).

    Parameters
    ----------
    u : obj/tuple
        Can either be:
        
        - MDAnalysis universe
        - A tuple of length 2 containing the format type keyword from :func:`md_spa.mdanalysis.generate_universe` and appropriate arguments in a tuple 
        - A tuple of length 3 containing the format type keyword from :func:`md_spa.mdanalysis.generate_universe`, appropriate arguments in a tuple, and dictionary of keyword arguments

    select : str, Optional, default="all"
        This string should align with MDAnalysis universe.select_atoms(select) rules

    Returns
    -------
    rg : numpy.ndarray
        The radius of gyration is calculated for the provided dump file
    kappa : numpy.ndarray
        The anisotropy is calculated from the gyration tensor, where zero indicates a sphere and one indicates an infinitely long rod.

    """

    u = check_universe(u)
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

def calc_end2end(u, indices):
    """
    Calculate the persistence length of a polymer given a LAMMPS data file, dump file(s), and a list of the atomIDs along the backbone (starting at 1).

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
    ag = u.atoms
    u.trajectory.add_transformations(unwrap(ag))
    tmp_indices = [x-1 for x in indices]

    r_end2end = np.zeros(len(u.trajectory))
    for i,ts in enumerate(u.trajectory):  # iterate through all frames
        r = u.atoms.positions[indices[0]] - u.atoms.positions[indices[1]]  # end-to-end vector from atom positions
        r_end2end[i] = np.linalg.norm(r)   # end-to-end distance


    return r_end2end

def hydrogen_bonding(u, indices, dt, tau_max=100, verbose=False, show_plot=False, intermittency=0, path="", filename="lifetime.csv", acceptor_kwargs={}, donor_kwargs={}, hydrogen_kwargs={}):
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
    filename : str, Optional, default="lifetime.csv"
        Prefix on filename to further differentiate
    acceptor_kwargs : dict, Optional, default={}
        Keyword arguments for ``MDAnalysis.analysis.hydrogenbonds.hbond_analysis.HydrogenBondAnalysis.guess_acceptors()``
    donor_kwargs : dict, Optional, default={}
        Keyword arguments for ``MDAnalysis.analysis.hydrogenbonds.hbond_analysis.HydrogenBondAnalysis.guess_donors()``
    hydrogen_kwargs : dict, Optional, default={}
        Keyword arguments for ``MDAnalysis.analysis.hydrogenbonds.hbond_analysis.HydrogenBondAnalysis.guess_hydrogens()``

    Returns
    -------
    Writes out .csv file: "{file_prefix}res_time_#typedonor_#typehydrogen_#typeacceptor.csv", where the type number is zero if a type is defined as None.

    """

    if dm.isiterable(indices):
        if not dm.isiterable(indices[0]):
            indices = [indices]
    else:
        raise ValueError("Input, indices, but be a list of iterables of length three")

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

    new_indices = []
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
            for d in donor_list_per_hydrogen[h]:
                for a in acceptors:
                    new_indices.append([d,h,a])

    output = []
    titles = []
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
        if not output:
            output.append(time)
        output.append(timeseries)
        titles.append("{}-{}-{}".format(*ind))

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

def survival_probability(u, dt, type_reference=None, type_target=None, zones=[(2, 5)], select=None, stop_frame=None, tau_max=100, intermittency=0, verbose=False, show_plot=False, path="", filename="survival.csv"):
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
        Number of timesteps to calculate the decay to, this value times dt is the maximum time.
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


def debye_waller_by_zones(u, frames_per_tau, select_reference=None, select_target=None, select=None, stop_frame=None, dr=1.0, r_start=2.0, nzones=5, verbose=False, write_increment=100):
    """
    Calculate the per atom Debye-Waller (DW) parameter for radially dependent zones from a specified bead type (or overwrite with other selection criteria) showing distance dependent changes in mobility.

    This method requires a characteristic time, generally on the order of a picosecond. This time is equal to the beta relaxation time and is generally temperature independent, and so can be determined from the minimum of the logarithmic derivative of the MSD, and low temperature may be necessary.

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

    u = check_universe(u)
    if verbose:
        print("Imported trajectory")
    dimensions = u.trajectory[0].dimensions[:3]

    frames_per_tau = int(frames_per_tau)
    if stop_frame == None:
        stop_frame = len(u.trajectory)-frames_per_tau
    elif len(u.trajectory)-frames_per_tau < stop_frame:
        stop_frame = len(u.trajectory)-frames_per_tau
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
            tmp = positions_start_finish[z][1]-positions_start_finish[z][0]
            for ii in range(len(tmp)):
                tmp_check = np.abs(tmp[ii]) > dimensions/2
                if np.any(tmp_check):
                    ind = np.where(tmp_check)[0]
                    for jj in ind:
                        if tmp[ii][jj] > 0:
                            tmp[ii][jj] -= dimensions[jj]
                        else:
                            tmp[ii][jj] += dimensions[jj]
            msd_total[z].extend(list(np.sum(np.square(tmp),axis=1)))

            tmp_ind1 = [x.ix for x in Zones[z]]
            RetainedFraction = Zones[z] - Zones[z].difference(Zones2[z])
            for atom in RetainedFraction:
                ind = tmp_ind1.index(atom.ix)
                msd_retained[z].append(np.sum(np.square(
                    positions_start_finish[z][1][ind]-positions_start_finish[z][0][ind]
                )))

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


