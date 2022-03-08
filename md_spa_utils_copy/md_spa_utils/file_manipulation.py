

import os
import csv
import numpy as np
import ast

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

def find_csv_entries(filename, matching_entries=None, indices=None, convert_float=True):
    """
    This function will find a specific line in a csv file, and return the requested indices. The lines are specified by the `matching_entries` variable, that uses an iterable structure to narrow the number of rows down to those with the initial columns matching this list. The remaining matrix is then returned according to the `indices`.

    Parameters
    ----------
    filename : str
        The filename and path to the target csv file.
    matching_entries : list, Optional, default=None
        This list indicates the criteria for narrowing the selection of rows. The first columns of each considered row must match these entries.
    indices : float/list, Optional, default=None
        The index of a column or a list of indices of the columns to extract from those rows that meet specification. A value of None returns all columns. WARNING! The indexing for this variable is ``np.shape(data)[1]-len(matching_entries)``, so the column after the columns that meet the matching criteria is specified with indices=0.
    convert_float : bool, Optional, default=True
        Convert all applicable entries into floats

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
    if dm.isiterable(indices):
        output = [[y[Nbuffer+x] for x in indices] for y in data]
    else:
        if indices != None:
            tmp_slice = indices + Nbuffer
            output = [y[tmp_slice] for y in data]
        else:
            output = [y[Nbuffer:] for y in data]

    if len(output) == 1:
        output = output[0]

    if convert_float:
        for i in range(len(output)):
            output[i] = [float(x) for x in output[i] if dm.isfloat(x)]

    return output

def average_csv_files(filenames, file_out, headers=None, delimiter=",", calc_standard_error=False):
    """
    Average multiple data files of the same type and size across eachother.

    Parameters
    ----------
    filenames : str
        Iterable array of file names
    file_out : str
        Output combined file
    headers : list[str], Optional, default=None
        If the header for the new file is not given, the header of the first provided file is used.
    delimiter : str, Optional, default=","
        Data separating string used in ``numpy.genfromtxt``
    calc_standard_error : bool, Optional, default=False
        If True, the standard error is calculated and interleaved into data.

    Returns
    -------
    New file written to ``file_out``
    """

    if headers != None and not isinstance(type(headers),str) and not dm.isiterable(headers):
        raise ValueError("The input `headers` should be iterable")

    if not dm.isiterable(filenames):
        raise ValueError("A list of filenames should have been provided.")

    data_in = np.array([np.transpose(np.genfromtxt(filename, delimiter=delimiter)) for filename in filenames])
    if len(np.shape(data_in)) != 3:
        raise ValueError("Data in given files are not equivalent in size: {}".format(", ".join([str(np.shape(x)) for x in data_in])))

    data = np.mean(data_in, axis=0)
    if calc_standard_error:
        data_se = np.std(data_in, axis=0)/np.sqrt(len(data_in))
    
    if headers == None:
        with open(filenames[0],"r") as f:
            headers = f.readline().rstrip()
    elif dm.isiterable(headers):
        headers = "# {}".format([str(x) for x in headers])

    data = np.transpose(data)
    if calc_standard_error:
        data_se = np.transpose(data_se)

    with open(file_out, "w") as f:
        if calc_standard_error:
            tmp_header = headers.split(",")
            headers = [xx for x in tmp_header for xx in (x, x+" SE")]
            f.write(", ".join(headers)+"\n")
        else:
            f.write(headers+"\n")

        for i in range(len(data)):
            if calc_standard_error:
                f.write(", ".join([str(y) for y in [xx for x in zip(data[i],data_se[i]) for xx in x]])+"\n")    
            else:
                f.write(", ".join([str(x) for x in data[i]])+"\n")
    

def write_csv(filename, array, mode="a", header=None, header_comment="#", delimiter=", "):
    """
    Write or append csv file.

    Parameters
    ----------
    filename : str
        Filename and path to csv file
    array : list
        This iterable object should be oriented so that axis=0 represents rows
    mode : str, Optional, default="a"
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

    flag = not os.path.isfile(filename)

    with open(filename,mode) as f:
        if header != None and flag:
            f.write(header_comment+delimiter.join([str(x) for x in header])+"\n")
        for line in array:
            f.write(delimiter.join([str(x) for x in line])+"\n")
