

import os
import csv
import numpy as np
import ast
import warnings

from . import data_manipulation as dm

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

    with open(filename, "r", encoding='utf-8-sig') as f:
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

def find_csv_entries(filename, matching_entries=None, indices=None, convert_float=True, verbose=False):
    """
    This function will find a specific line in a csv file, and return the requested indices. The lines are specified by the `matching_entries` variable, that uses an iterable structure to narrow the number of rows down to those with the initial columns matching this list. The remaining matrix is then returned according to the `indices`.

    Parameters
    ----------
    filename : str
        The filename and path to the target csv file.
    matching_entries : list, Optional, default=None
        This list indicates the criteria for narrowing the selection of rows. The first columns of each considered row must match these entries. If ``None`` the column is skipped, and the value added to the resulting row.
    indices : float/list, Optional, default=None
        The index of a column or a list of indices of the columns to extract from those rows that meet specification. A value of None returns all columns. The column numbers begin at 0 and increase according to the number of columns in the file. 
    convert_float : bool, Optional, default=True
        Convert all applicable entries into floats
    verbose : bool, default=False
        If True, warnings will be provided.

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

    row_indices = [i for i,row in enumerate(data) if np.all([(row[j] == x or x is None) for j,x in enumerate(matching_entries)])]
    if not row_indices:
        if verbose:
            warnings.warn("Matching entries : {} cannot be found in {}".format(matching_entries, filename))
        return []

    lx = len(matching_entries)
    tmp_indices = [i for i,x in enumerate(matching_entries) if x is None]
    if dm.isiterable(indices):
        pass
    elif indices is not None:
        indices = list(range(indices+lx, len(data[0])))
    else:
        indices = tmp_indices + list(range(lx, len(data[0])))

    output = [[data[x][y] for y in indices] for x in row_indices]
    if convert_float:
        for i in range(len(output)):
            output[i] = [float(x) if dm.isfloat(x) else x for x in output[i]]

    if len(output) == 1:
        output = output[0]

    return output

def average_csv_files(filenames, file_out, headers=None, delimiter=",", calc_error=False, error_kwargs={}):
    """
    Average multiple data files of the same type and size across eachother.

    Parameters
    ----------
    filenames : str
        Iterable array of file names
    file_out : str
        Output combined file
    headers : list[str], Optional, default=None
        If the header for the new file is not given, the header (first line) of the first provided file is used. Notice that headers for the error do not need to be provided.
    delimiter : str, Optional, default=","
        Data separating string used in ``numpy.genfromtxt``
    calc_error : bool, Optional, default=False
        If True, the standard error is calculated and interleaved into data.
    error_kwargs : dict, Optional, default={"axis": 0}
        Keyword arguments for :func:`data_manipulation.basic_stats`. The default takes the standard error for the corresponding elements across files. 
        

    Returns
    -------
    New file written to ``file_out``
    """

    if headers is not None and not isinstance(type(headers),str) and not dm.isiterable(headers):
        raise ValueError("The input `headers` should be iterable")

    if not dm.isiterable(filenames):
        raise ValueError("A list of filenames should have been provided.")

    data_in = np.array([np.transpose(np.genfromtxt(filename, delimiter=delimiter)) for filename in filenames])
    if len(np.shape(data_in)) != 3:
        raise ValueError("Data in given files are not equivalent in size: {}".format(", ".join([str(np.shape(x)) for x in data_in])))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        error_kwargs.update({"axis": 0})
        data, data_se = dm.basic_stats(data_in, **error_kwargs)
        data = np.transpose(data)
        data_se = np.transpose(data_se)
    
    if headers is None:
        with open(filenames[0],"r") as f:
            headers = f.readline().rstrip()
    elif dm.isiterable(headers):
        headers = "# {}".format([str(x) for x in headers])

    with open(file_out, "w") as f:
        if calc_error:
            tmp_header = headers.split(",")
            headers = [xx for x in tmp_header for xx in (x, x+" Error")]
            f.write(", ".join(headers)+"\n")
        else:
            f.write(headers+"\n")

        for i in range(len(data)):
            if calc_error:
                f.write(", ".join([str(y) for y in [xx for x in zip(data[i],data_se[i]) for xx in x]])+"\n")    
            else:
                f.write(", ".join([str(x) for x in data[i]])+"\n")
    

def write_csv(filename, array, mode="w", header=None, header_comment="#", delimiter=", "):
    """
    Write or append csv file.

    Parameters
    ----------
    filename : str
        Filename and path to csv file
    array : list
        This iterable object should be oriented so that axis=0 represents rows
    mode : str, Optional, default="w"
        String to identify the mode with which to ``open(filename, mode)``
    header : list, Optional, default=None
        List of the same length as the second dimension
    delimiter : str, Optional, default=", "
        Delimiter between header and line entries
    header_comment : str, Optional, default="#"
        Symbol to comment out header for importing later (e.g. numpy.genfromtxt). Note that an additional header line could be placed before the headers if a `\\n` was added.

    Returns
    -------
    Write csv file

    """

    if not dm.isiterable(array) or not dm.isiterable(array[0]):
        raise ValueError("Input `array` must be an iterable type containing iterable elements.")

    mode = "w" if not os.path.isfile(filename) else mode
    with open(filename,mode) as f:
        if header is not None and "w" in mode:
            f.write(header_comment+delimiter.join([str(x) for x in header])+"\n")
        for line in array:
            f.write(delimiter.join([str(x) for x in line])+"\n")

def csv2dict(filename, comment="#", label_col=None, tiers=1, skip_cols=0, header_row=0):
    """ Create dict from csv file

    The last commented header line is assumed to be the categories and the first column the keys.
    Thus the following csv file:

    # This is a file
    # type, prop1, prop2, prop3
    A, 1, 2, 3
    B, 4, 5, 6

    Will result in the dictionary: ``dictionary = {"A": {"prop1": 1, "prop2": 2, "prop3": 3}, 
        "B": {"prop1": 4, "prop2": 5, "prop3": 6}}`` for ``tiers == 1``

    If ``tiers == 2`` then the second column entries are also used as keys such as the following:
 
    # This is a file
    # type, subtype, prop1, prop2, prop3
    A, a, 1, 2, 3
    A, b, 4, 5, 6
    B, b, 7, 8, 9

    Will result in the dictionary: ``dictionary = {"A": {"a": {"prop1": 1, "prop2": 2, "prop3": 3}, 
        "b": {"prop1": 4, "prop2": 5, "prop3": 6}}, "B": {"b": {"prop1": 7, "prop2": 8, "prop3": 9}}}``

    Numbers are converted accordingly

    Parameters
    ----------
    filename : str
        File name to the csv file
    comment : str, Optional, default="#"
        String at the beginning of a line to denote a comment or header
    tiers : int, Optional, default=1
        Number of tiers in dictionary before the remaining entries are in a single dictionary with column header values as keys.
    skip_cols : int, Optional, default=0
        Number of columns to skip 
    label_col : int, Optional, default=None
        Specify the label column, regardless of the `skip_cols` value. This allows one to choose the first column as the label, but ignore a series of labels after the fact.
    header_row : int, Optional, default=0
        Define the row used to pull the header and generate keys

    Returns
    -------
     output : dict
         Dictionary of csv entries

    """

    if not os.path.isfile(filename):
        raise ValueError("The file, {}, could not be found.".format(filename))

    with open(filename, "r") as f:
        contents = csv.reader(f)
        data = list(map(list, contents))
    data = [[ast.literal_eval(x.strip()) if x.strip().replace('.','',1).isdigit() else x.strip() for x in y] for y in data]

    # find header, discard comments
    header = None
    output = {}
    for i,line in enumerate(data):
        if i == header_row:
            header = line[skip_cols:][tiers:] # Column headers for tiers are irrelevant
            header = [x.replace(comment, "") for x in header]
        else:
            if label_col is not None and label_col < skip_cols:
                label = line[label_col]
            else:
                label = skip_cols
            line = line[skip_cols:]
            if header is None:
                raise ValueError("Commented column headers were not found")
            if len(header) != len(line[tiers:]):
                raise ValueError("The number of column headers ({}) does not equal the number of columns ({}) in line {}".format(len(header),len(line[tiers:]),i))

            if label_col is None:
                tier_keys = line[:tiers]
            else:
                tier_keys = [label] + line[:tiers-1]
            tmp_dict = output
            for j, key in enumerate(tier_keys):
                if j+1 == tiers:
                    if key in tmp_dict:
                        warnings.warn("Overwriting entries for tiers: dict[{}]. Consider adding more Tiers.".format("][".join([str(x) for x in tier_keys])))
                    tmp_dict[key] = {header[x]: line[x+tiers] for x in range(len(header))}
                else:
                    if key not in tmp_dict:
                        tmp_dict[key] = {}
                    tmp_dict = tmp_dict[key]

    return output
    
