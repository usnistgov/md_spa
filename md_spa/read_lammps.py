

import numpy as np
import fileinput
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image as img

from md_spa_utils import data_manipulation as dm

def read_lammps_ave_time(filename, dtype=float):
    """Read a LAMMPS ave/time file. Written By Lauren Abbott"""
    
    all_data = []
    with open(filename, 'r') as f:
        while(True):
            line = f.readline()
            if len(line) == 0:
                break
            elif line == '\n':
                continue  
            parts = line.strip().split()
            
            if parts[0] == '#':
                continue
            
            else:
                num = int(parts[1])
                data = [0]*num
                for i in range(num):
                    data[i] = f.readline().strip().split()
                all_data.append(data)
                break # Added JAC
    
    return np.array(all_data, dtype=dtype)

def read_lammps_ave_time_flat(filename, dtype=float):
    """Read a LAMMPS ave/time file. Written By Lauren Abbott"""
    
    data = []
    with open(filename, 'r') as f:
        while(True):
            line = f.readline()
            if len(line) == 0:
                break
            elif line == '\n':
                continue  
            parts = line.strip().split()
            
            if parts[0] == '#':
                continue
            
            else:
                num = int(parts[1])
                for i in range(num):
                    data.append(np.array(f.readline().strip().split(), dtype=dtype))
    
    return np.array(data)

def redefine_lammps_dump_timesteps( filename, timestep0, filename_out=None, overwrite=False, replace=False):
    """
    Renumber timestamps in lammps dump file. This is most appropriate when one accidentally restarts the numbering when continuing a simulation.

    Parameters
    ----------
    filename : str
        Filename and path to lammps dump file
    timestep0 : int
        The initial timestep the dump file should be set to. The later timesteps will be renumbered according to the frequency detected in the file.
    filename_out : str, Optional, default=None
        Filename and path for the output file. If None, the new dump file uses the given name and path with the prefix "renumbered\_".
    overwrite : bool, Optional, default=False
        If True, and filename_out already exists, the file will be overwritten
    replace : bool, Optional, default=False
        If True, the original file, filename, will be overwritten

    Returns
    -------
    lammps dump file

    """

    if not isinstance(timestep0,int):
        raise ValueError("Given timestep should be an integer")

    if not os.path.isfile(filename):
        raise ValueError("The file, {}, does not exist.".format(filename))

    tmp = os.path.split(filename)
    if filename_out == None:
        filename_out = os.path.join(tmp[0],"renumbered_"+tmp[1])
        if os.path.isfile(filename_out):
            if not overwrite:
                raise ValueError("The file, {}, already exists".format(filename_out))

    Nfreq = None
    timestep = None
    with open(filename, "r") as fin:
        with open(filename_out,"w") as fout:
            flag_timestep = False
            for line in fin:
                if "TIMESTEP" in line:
                    flag_timestep = True
                    fout.write(line)
                elif flag_timestep:
                    flag_timestep = False
                    if timestep == None:
                        timestep_old = int(line.strip())
                        timestep = timestep0
                    else:
                        if Nfreq == None:
                            Nfreq = int(line.strip())-timestep_old
                        timestep += Nfreq
                    fout.write("{}\n".format(timestep))
                else:
                    fout.write(line)

    if replace:
        os.system("mv {} {}".format(filename_out, filename))

def read_lammps_dump(filename, col_name='', atom_indices=None, max_frames=None, unwrap=False, dtype=float):
    """Read a LAMMPS dump file.

    Note:

    - Atoms are automatically sorted by index according to the appropriate dimension
    - Because the box lengths are returned, the coordinates are translated so that the minimum is at the origin.
    - One may write a lammps dump file with each frame containing a different number of atoms or a different set of atoms. This function will record values of NaN when one of the specified atom indices are not present.

    Parameters
    ----------
    filename : str
        Path and filename to lammps dump file
    col_name : list/str, Optional, default=""
        A list of column names in the header or a specific column name to pull from the dump file
    atom_indices : list, Optional, default=None
        A list of atom indices to pull from the trajectory where the indices start from 0 to N-1 (this function offsets the lammps default of numbering from 1 to N)
    max_frames : int, Optional, default=None
        Stop reading the file when this many timesteps have been read
    unwrap : bool, Optional, default=False
        If the columns ['x', 'y', 'z'] are requested and the columns ['ix', 'iy', 'iz'] are found, the coordinates will be unwrapped. 

    Returns
    -------
    dump_output : numpy.ndarray
        3D array of (Nframes, Natoms, Ncolumns), unless one column name is provided, in which case the output is (Nframes, Natoms)
    box_lengths : numpy.ndarray
        An array of the box lengths in the x, y, and z direction. The box cannot change in size.

    """
    if atom_indices is not None:
        atom_indices = list(atom_indices)
    
    arr_all = []
    with open(filename, 'r') as f:
        while True:
            line = f.readline()
            if len(line) == 0:
                break
            elif line == '\n':
                continue
            line_array = line.strip().split()
            
            flag = 'all'
            if isinstance(col_name, list) and len(col_name) > 0:
                flag = 'matrix'
            elif len(col_name) > 0:
                flag = 'single'
                if unwrap:
                    raise ValueError("Cannot unwrap single column")
            
            if line_array[1] == "TIMESTEP": # Not Used
                timestep = int(f.readline().strip())
            if len(arr_all) >= max_frames:
                break
            
            elif line_array[1] == "NUMBER":
                natoms = int(f.readline().strip())
                if atom_indices is not None:
                    if len(atom_indices) > natoms or any([(x >= natoms or x < 0) for x in atom_indices]):
                        raise ValueError("Provided indices are incompatible with provided dump file")
                    else:
                        num_matrix = len(atom_indices)
                else:
                    num_matrix = natoms
                
            elif line_array[1] == "BOX":
                xlo, xhi = map(float, f.readline().strip().split())
                ylo, yhi = map(float, f.readline().strip().split())
                zlo, zhi = map(float, f.readline().strip().split())
                tmp_box_dims = np.array([xhi-xlo, yhi-ylo, zhi-zlo], dtype=float)
                try:
                    if not np.all(tmp_box_dims == box_dims):
                        raise ValueError("The box size changes from {} to {}".format(box_dims, tmp_box_dims))
                except:
                    box_dims = tmp_box_dims
                
            elif line_array[1] == "ATOMS":
                fields = line_array[2:]
                if flag != 'single':
                    try:
                        image_cols = [fields.index(s) for s in ["ix", "iy", "iz"]]
                    except:
                        image_cols = None
                    if flag == "matrix":
                        if "xu" in col_name:
                            coord_cols = [col_name.index(s) for s in ["xu", "yu", "zu"]]
                        elif "x" in col_name:
                            coord_cols = [col_name.index(s) for s in ["x", "y", "z"]]
                        else:
                            coord_cols = None
                    else:
                        if "xu" in fields:
                            coord_cols = [fields.index(s)-1 for s in ["xu", "yu", "zu"]]
                        elif "x" in fields:
                            coord_cols = [fields.index(s)-1 for s in ["x", "y", "z"]]
                        else:
                            coord_cols = None

                if flag == 'matrix':
                    arr = np.nan*np.ones((num_matrix, len(col_name)), dtype=dtype)
                    col = [fields.index(s) for s in col_name]
                elif flag == 'single':
                    arr = np.nan*np.ones(num_matrix, dtype=dtype)
                    col = fields.index(col_name)
                elif flag == 'all':
                    arr = np.nan*np.ones((num_matrix, len(fields)-1), dtype=dtype)
                if unwrap:
                    if image_cols is None:
                        raise ValueError("The given trajectory cannot be unwrapped without image flags")
                    elif not all([s in col_name for s in ["x", "y", "z"]]):
                        raise ValueError("The given trajectory cannot be unwrapped without x, y, z columns")
                    
                for i in range(natoms):
                    dat = [x if not dm.isfloat(x) else float(x) for x in f.readline().strip().split()]
                    ind = int(dat[0])-1
                    if atom_indices is not None:
                        if ind in atom_indices:
                            ind = atom_indices.index(ind)
                        else:
                            continue

                    if flag == 'matrix':
                        arr[ind] = [dat[c] for c in col]
                    elif flag == 'single':
                        arr[ind] = dat[col]
                    else:
                        arr[ind] = dat[1:]

                    if unwrap:
                        arr[ind][coord_cols] +=  np.array([dat[c] for c in image_cols]) * box_dims
                    if coord_cols is not None:
                        arr[ind][coord_cols] -= np.array([xlo, ylo, zlo])
                
                arr_all.append(arr)
    
    return np.array(arr_all, dtype=dtype), box_dims

def read_lammps_data(filename, section="atoms"):
    """Read specific section from LAMMPS data file. Written By Lauren Abbott"""
    
    with open(filename, 'r') as f:
        header = f.readline().strip()
        counts = {}
        
        while True:
            line = f.readline()
            if len(line) == 0:
                break
            elif line == '\n':
                continue
            parts = line.strip().split()
            
            if len(parts) > 1 and parts[1] == 'atoms':
                natoms = int(parts[0])
                counts['atoms'] = natoms
            elif len(parts) > 1 and parts[1] == 'bonds':
                num_bonds = int(parts[0])
                counts['bonds'] = num_bonds
            elif len(parts) > 1 and parts[1] == 'angles':
                num_angles = int(parts[0])
                counts['angles'] = num_angles
            elif len(parts) > 1 and parts[1] == 'dihedrals':
                num_diheds = int(parts[0])
                counts['dihedral'] = num_diheds
            
            elif len(parts) > 1 and parts[1] == 'atom':
                num_atom_types = int(parts[0])
                counts['atom types'] = num_atom_types
            elif len(parts) > 1 and parts[1] == 'bond':
                num_bond_types = int(parts[0])
                counts['bond types'] = num_bond_types
            elif len(parts) > 1 and parts[1] == 'angle':
                num_angle_types = int(parts[0])
                counts['angle types'] = num_angle_types
            elif len(parts) > 1 and parts[1] == 'dihedral':
                num_dihed_types = int(parts[0])
                counts['dihedral types'] = num_dihed_types
            
            elif len(parts) > 2 and parts[2] == 'xlo':
                xlo = float(parts[0])
                xhi = float(parts[1])
                xdims = np.array([xlo, xhi, xhi-xlo])
            elif len(parts) > 2 and parts[2] == 'ylo':
                ylo = float(parts[0])
                yhi = float(parts[1])
                ydims = np.array([ylo, yhi, yhi-ylo])
            elif len(parts) > 2 and parts[2] == 'zlo':
                zlo = float(parts[0])
                zhi = float(parts[1])
                zdims = np.array([zlo, zhi, zhi-zlo])
                if section == "counts" or section == "box":
                    break
                    
            elif parts[0] == 'Masses':
                masses = np.zeros(num_atom_types + 1, dtype=float)
                f.readline()
                for i in range(num_atom_types):
                    bits = f.readline().strip().split()
                    masses[int(bits[0])] = bits[1]
                if section == "masses":
                    break
            
            elif parts[0] == 'Atoms':
                atoms = np.zeros((natoms + 1, 9), dtype=float)
                f.readline()
                for i in range(natoms):
                    bits = f.readline().strip().split()
                    atoms[int(bits[0])] = bits[1:]
                if section == "atoms":
                    break
            
            elif parts[0] == 'Velocities':
                velocities = np.zeros((natoms + 1, 3), dtype=float)
                f.readline()
                for i in range(natoms):
                    bits = f.readline().strip().split()
                    velocities[int(bits[0])] = bits[1:]
                if section == "velocities":
                    break
                    
            elif parts[0] == 'Bonds':
                bonds = np.zeros((num_bonds + 1, 3), dtype=int)
                f.readline()
                for i in range(num_bonds):
                    bits = f.readline().strip().split()
                    bonds[int(bits[0])] = bits[1:]
                if section == "bonds":
                    break
                    
            elif parts[0] == 'Angles':
                angles = np.zeros((num_angles + 1, 4), dtype=int)
                f.readline()
                for i in range(num_angles):
                    bits = f.readline().strip().split()
                    angles[int(bits[0])] = bits[1:]
                if section == "angles":
                    break
                    
            elif parts[0] == 'Dihedrals':
                diheds = np.zeros((num_diheds + 1, 5), dtype=int)
                f.readline()
                for i in range(num_diheds):
                    bits = f.readline().strip().split()
                    diheds[int(bits[0])] = bits[1:]
                if section == "dihedrals":
                    break
                    
    if section == "counts":
        return counts
    elif section == "box":
        return np.array([xdims, ydims, zdims], dtype=float)
    elif section == "masses":
        return masses
    elif section == "atoms":
        return atoms
    elif section == "velocities":
        return velocities
    elif section == "bonds":
        return bonds
    elif section == "angles":
        return angles
    elif section == "dihedrals":
        return diheds
    else:
        return 0

def read_files(files, func, **kwargs):
    """Read multiple files using the given function. Written By Lauren Abbott"""
    
    data = [0]*len(files)
    for i, f in enumerate(files):
        data[i] = func(f, **kwargs)
        
    return np.array(data)

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    """Make a new colormap by truncating an existing color map. Written By Lauren Abbott"""
    
    new_cmap = mpl.colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

def pairs_indicies(group1, group2, shift=0):
    """Returns lists of pairs for all pairs with one from group1 and one from group2. Written By Lauren Abbott"""
    
    indicies = []
    for a1 in group1:
        for a2 in group2:
            if types_str.index(a1) <= types_str.index(a2):
                pair = (a1,a2)
            else:
                pair = (a2,a1)
            index = types_pairs.index(pair) + shift
            if index not in indicies:
                indicies.append(index)
                
    return indicies

def diagonalize_matrix(mat, tol=10e-10, flat=False, sort=True):
    """Diagonalize a square matrix. Written By Lauren Abbott"""
    
    if len(mat.shape) == 1:
        flat = True
    if flat:
        mat = mat.reshape(int(np.sqrt(mat.shape[0])), -1)
    
    if mat.shape[0] != mat.shape[1]:
        print("Warning: non-square matrix cannot be diagonalized")
        return mat
    
    matp = np.linalg.eig(mat)[1] # Extract eigen vectors as columns in matrix P
    if mat.shape != matp.shape:
        print("Warning: matrix is not diagonizable")
        return mat
    
    matpi = np.linalg.inv(matp) # Multiplicative inverse of matrix P
    matd = np.dot(matpi, np.dot(mat, matp)) # Calculate eigenvalues 
    matd[np.abs(matd) < tol] = 0
    diag = matd.diagonal()
    if sort:
        diag = np.sort(diag)
    return diag

def group_by_clust(pairs):
    """Group [type,clustID] pairs by clustID and return sorted list of types per clustID Written By Lauren Abbott"""
    
    num = np.max(pairs[:,1])
    return [np.sort(pairs[np.any(pairs == [0,i], axis=1),0]) for i in range(1,num+1)]
