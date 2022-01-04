import sys
import numpy as np


def periodic2rb(filename, fileout="rb_constants.csv"):
    """
    This function imports a csv file of periodic torsion potential constants and converts them to Ryckaert-Bellemans potential constants. The functional forms are of the form reported in OpenMM.

    Parameters
    ----------
    filename : str
        Filename of csv file containing parameters
    fileout : str, Optional, default="rb_constants.csv"
        Name of output RB constants.

    Returns
    -------
    Saves file
    
    """

    data = np.genfromtxt(filename, dtype=np.float, delimiter=",", missing_values='', filling_values=0.0).T
    Nconst, Nsets = np.shape(data)

    if Nconst not in [3,4]:
        raise ValueError("The periodic torsion potential should have four coefficients.")
    elif Nconst == 3:
        data = np.vstack([data,np.zeros(Nsets)])

    rb_constants = np.zeros((6,Nsets))

    rb_constants[0] = data[0]+data[2]
    rb_constants[1] = 3*data[2]-data[0]
    rb_constants[2] = 8*data[3]-2*data[1]
    rb_constants[3] = -4*data[2]
    rb_constants[4] = -8*data[3]
    rb_constants[5] = 0.0

    rb_constants = rb_constants.T

    with open(fileout, "w") as f:
        f.write("# These Ryckaert-Belleman constants were converted from the periodic torsion potential of OpenMM\n")
        for consts in rb_constants:
            f.write("{}, {}, {}, {}, {}, {}\n".format(*consts.tolist()))

