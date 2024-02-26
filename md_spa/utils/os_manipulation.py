
import os
import warnings
from sys import platform

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

