""" Process `Zeno <https://zeno.nist.gov>`_ output file to a dictionary
    Recommend loading with:
    ``import md_spa.zeno as zeno``

"""

import os
import warnings
import numpy as np

from md_spa.utils import data_manipulation as dm

def extract_csv(filename, extract_array=None):
    """
    Pull data from zeno calculation using keywords defined both by default and in ``extract_array``. Only the string before the first comma is needed, as the lines for "values", "std_dev", and "units" will be pulled.

    https://zeno.nist.gov

    Parameters
    ----------
    filename : str
        Filename and path for zeno .csv output file
    extract_array : list[str], default=None
        List of lines headers, where the value, units, and std_dev will be pulled. By default values for the following are pulled: "hydrodynamic_radius","gyration_eigenvalues[0]","gyration_eigenvalues[1]","gyration_eigenvalues[2]","intrinsic_viscosity".

    Returns
    -------
    output : dict
        A dictionary where the keys are values extracted with ``extract_array``, each representing a dictionary with the following values: ``{"value": values, "std_dev": std_devs, "units": units}``.

    """

    if not os.path.isfile(filename):
        raise ImportError("Filename, {}, does not exist with current directory, {}.".format(filename, os.getcwd()))

    extract = ["hydrodynamic_radius","gyration_eigenvalues[0]","gyration_eigenvalues[1]","gyration_eigenvalues[2]","intrinsic_viscosity"]
    if not np.all(extract_array == None):
        extract += extract_array

    weird_ind = []
    output = {}
    with open(filename, "r+") as f:
        for line in f:
            line_array = line.strip().split(",")
            property_key = line_array[0]
            if property_key in extract:
                output[property_key] = {}
                output[property_key]["units"] = line_array[2]
                line_array = next(f).strip().split(",")
                tmp = []
                output[property_key]["value"] = [float(x) if dm.isfloat(x) else np.nan for x in line_array[2:]]
                line_array = next(f).strip().split(",")
                output[property_key]["std_dev"] = [float(x) if dm.isfloat(x) else np.nan for x in line_array[2:]]
                extract.remove(property_key)
                if np.any(np.isnan(output[property_key]["value"])):
                    inds = np.where(np.isnan(output[property_key]["value"]))[0]
                    for ind in inds:
                        if ind not in weird_ind:
                            weird_ind.append(ind)

    if weird_ind:
        warnings.warn("Zeno frames: {}, output non-float values in {}. Setting frame values to NaN".format(weird_ind,filename))
        for ind in weird_ind:
            for key in output:
                output[key]["value"][ind] = np.nan
                output[key]["std_dev"][ind] = np.nan

    if "mass" in output and np.all(output["mass"]["value"]==1.0):
        warnings.warn("Provided mass equals unity for all cases. In post-processing be sure to scale: intrinsic_viscosity_mass_units, Sedimentation coefficient")

    if extract:
        raise ValueError("The keywords: {} are not represent an output of zeno output {}".format(", ".join(extract),filename))

    return output
