
import ast
import csv
import os
import warnings
from sys import platform
import numpy as np


def remove_lines(filename, start=None, end=None):
    """
    Remove lines in a file inclusively. If a starting place isn't given, then the lines up to that point are removed. Similarly, if the ending point isn't given, then the tail of the file is deleted. However, if neither start or end is given, nothing happens.

    The first line is designated with start=1.

    Parameters
    ----------
    filename : str
        Name of the file to alter
    start : int, Optional, default=None
        Starting point from which to start deleting lines 
    end : int, Optional, default=None
        Ending point to stop deleting lines after

    """

    if not os.path.isfile(filename):
        raise ImportError("Given filename cannot be found.")

    if start == None and end == None:
        raise ValueError("At least a starting or ending line must be provided.")

    if start == None:
        start = 0
    if not isinstance(start,int):
        raise ValueError("Given line number for 'start' must be an integer")

    if not isinstance(end,int):
        raise ValueError("Given line number for 'end' must be an integer")

    with open(filename, "r") as f:
        lines = f.readlines()
        Nlines = len(lines)
        if end == None or end == -1:
            end = Nlines
        new_lines = [lines[i] for i in range(Nlines) if (i+1 < start or i+1 > end)]

    with open(filename,"w") as f:
        for line in new_lines:
            f.write(line)

def find_csv_entries(filename, matching_entries=None, indices=None):
    """
    This function will find a specific line in a csv file, and return the requested indices. The lines are specified by the `matching_entries` variable, that uses an iterable structure to narrow the number of rows down to those with the initial columns matching this list. The remaining matrix is then returned according to the `indices`.

    Parameters
    ----------
    filename : str
        The filename and path to the target csv file.
    matching_entries : list, Optional, default=None
        This list indicates the criteria for narrowing the selection of rows. The first columns of each considered row mus
t match these entries.
    indices : float/list, Optional, default=None
        The index of a column or a list of indices of the columns to extract from those rows that meet specification. A value of None returns all columns. WARNING! The indexing for this variable is np.shape(data)[1]-len(matching_entries), so the column after the columns that meet the matching criteria is specified with indices=0.

    Returns
    -------
    output : list/float
         The resulting structure is returned. If more than one row meet the specified criteria this is a list or list of lists. If more than one index with specified this is a list or float.

    """

    if not os.path.isfile(filename):
        raise ValueError("The file, {}, could not be found.".format(filename))

    with open(filename, "r") as f:
        contents = csv.reader(f)
        data = list(map(list, contents))
    data = [[ast.literal_eval(x.strip()) if x.strip().replace('.','',1).isdigit() else x.strip() for x in y] for y in data]

    for j,match in enumerate(matching_entries):
        row_indices = []
        for i,row in enumerate(data):
            if row[j] == match:
                row_indices.append(i)
        if not row_indices:
            raise ValueError("Rows that meet your critera, {}, could not be found".format(matching_entries[:j+1]))
        data = [data[x] for x in range(len(data)) if x in row_indices]

    Nbuffer = len(matching_entries)
    if isiterable(indices):
        output = [[y[Nbuffer+x] for x in indices] for y in data] 
    else:
        if indices != None:
            tmp_slice = indices + Nbuffer
            output = [y[tmp_slice] for y in data]
        else:
            output = [y[Nbuffer:] for y in data]

    if len(output) == 1:
        output = output[0]

    return output


def find_header(filename, delimiter=",", comments="#"):
    """
    This function finds the column headers from a file and outputs a list. This function assumes the column headers are within the last commented line at the top of the file. 

    Parameters
    ----------
    filename : str
        Filename and path to target file.
    delimiter : str, Optional, default=","
        Character separating strings
    comments : str, Optional, default="#"
        Character at the start of a commented line

    Returns
    -------
    col_headers : list
        List of columns headers

    """

    if not os.path.isfile(filename):
        raise ValueError("The given file could not be found: {}".format(filename))

    with open(filename, "r") as f:
        while(True):
            line = f.readline()

            if line == '\n':
                continue 
            linearray = line.split(delimiter)
            if comments in linearray[0]:
                col_headers = linearray
            else:
                break
    col_headers = [x.strip().strip("# ") for x in col_headers]
    if col_headers[0] == "":
        col_headers = col_headers[1:]

    return col_headers

def isiterable(array):
    """
    Check if variable is an iterable type with a length (e.g. np.array or list)

    Note that this could be test is ``isinstance(array, Iterable)``, however ``array=np.array(1.0)`` would pass that test and then fail in ``len(array)``.

    Taken from github.com/jaclark5/despasito

    Parameters
    ----------
    array
        Variable of some type, that should be iterable

    Returns
    -------
    isiterable : bool
        Will be True if indexing is possible and False is not

    """

    array_tmp = np.array(array, dtype=object)
    tmp = np.shape(array_tmp)

    if tmp:
        isiterable = True
    else:
        isiterable = False

    return isiterable

def check_create_dir(path, create=False, verbose=True):
    """
    Check whether the directory exists, if it doesn't, optionally create the path.

    Parameters
    ----------
    path : str
        Path to the target directory
    create : bool, Optional, default=False
        If true, `os.makedirs` will be used to create the target path
    verbose : bool, Optional, default=True
        If False, removes warnings when I directory is being creates

    """

    if not os.path.isdir(path):
        if create:
            if verbose:
                warnings.warn("Directory does not exist and is being created: {}".format(path))
        else:
            raise ValueError("Directory does not exist: {}".format(path))

        os.makedirs(path)

def sed_file(target_dir, template_file, replaced_values, replacement_values):
    """
    This function will copy a template file to a target directory and replace the specified keywords with the specified replacement.

    Parameters
    ----------
    target_dir : str
        The directory in which to place the file
    template_file : str
        The path and filename for the template file
    replaced_values : list
        This iterable structure contains the values to replace
    replacement_values : list
        This iterable structure of the same length as `replaced_values` contains the values to be replaced.

    """

    cwd = os.getcwd()

    if not os.path.isdir(target_dir):
        raise ValueError("The directory: {}\n Cannot be located from the directory: {}".format(target_dir, cwd))

    if not os.path.isfile(template_file):
        raise ValueError("Cannot find file: {}\n from directory: {}".format(template_file, cwd))

    os.system("cp {} {}".format(template_file,target_dir))
    newfile = os.path.join(target_dir,os.path.split(template_file)[1])
    if not os.path.isfile(newfile):
        raise ValueError("Template file: {}\n was not copied to new directory: {}".format(template_file,target_dir))

    if "darwin" in platform:
        flag_mac = True
    elif "win" in platform:
        raise ValueError("This function has not been designed to support Windows")
    else:
        flag_mac = False

    if len(replaced_values) != len(replacement_values):
        raise ValueError("The list of values to replace and the list of replacements must be the same length")

    for before, after in zip(replaced_values, replacement_values):
        if flag_mac:
            os.system('sed -i "" "s/{}/{}/g" {}'.format(before, after, newfile))
        else:
            os.system('sed -i "s/{}/{}/g" {}'.format(before, after, newfile))

if __name__ == "__main__":

    L = len(sys.argv)
    if L == 8:
        target_dir = sys.argv[1]
        template_file = [sys.argv[2], sys.argv[5]]
        replaced_values = [sys.argv[3].strip("[] ").split(","),sys.argv[6].strip("[] ").split(",")]
        replacement_values = [sys.argv[4].strip("[] ").split(","),sys.argv[7].strip("[] ").split(",")]
    else:
        print(sys.argv)
        raise ValueError("Input should be 'python run_lammps.py [target_dir] [lammps_input] [lammps replaced values] [lammps replacement values] [submission file] [submission replaced values] [submission replacement values]'")

    om.check_create_dir(target_dir, create=True)
    replace_submit_lammps(target_dir, template_file, replaced_values, replacement_values)




