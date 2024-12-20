""" Calculate and pull meaningful features from the radial distribution function.

    Recommend loading with:
    ``import md_spa.rdf as rdf``

"""

import numpy as np

import copy
import os
import warnings
import matplotlib.pyplot as plt
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.ndimage.filters import gaussian_filter1d
import scipy.integrate as spi
import scipy.stats as sps

from md_spa.utils import file_manipulation as fm
from md_spa.utils import data_manipulation as dm
from md_spa import custom_fit as cfit

def consolidate_arrays(rdf_array, file_out="rdf_out.txt", pairs=None):
    """
    This function will take equivalent rdf data output from several independent lammps "boxes" and average it together.

    Parameters
    ----------
    rdf_array : numpy.ndarray
        The first dimension of this array disappears as the equivalent systems are averaged together. The remaining 2D array consists of the first row is the distance and subsequent rows are the rdf values that correspond to the input `pairs` (if provided). 
    file_out : str, default='rdf_out.txt'
        Filename for consolidated data file, the output file will be equipped with two prefixes, for the rdf\_ and integrated coord\_ data.
    pairs : list, default=None
        Optional list of iterable structure of length 2 each containing pairs of values that represent the rdf

    Returns
    -------
    text file

    """
    # Check inputs
    if not dm.isiterable(pairs):
        raise ValueError("The input `pairs` should be iterable")

    if not dm.isiterable(rdf_array):
        rdf_array = np.array([rdf_array])

    shape = np.shape(rdf_array)
    if len(shape) != 3:
        raise ValueError("Provided matrix should have 3 dimensions. The first distinguishes rdf data from equivalent but independent systems, the second contains first the distance and second the pair distributions (the same length in all systems), and third is the length of the data array.")

    rdf_data = np.mean(rdf_array, axis=0)

    # Prepare to write output files
    Npairs = len(rdf_data)-1

    if pairs == None:
        tmp_header = ", ".join(["Pair {}".format(i+1) for i in range(Npairs)])
    else:
        if len(pairs) != Npairs:
            warnings.warn("The number of given pairs does not match the output, using standard header.")
            tmp_header = ", ".join(["Pair {}".format(i+1) for i in range(Npairs)])
        elif not all([len(x)==2 for x in pairs]):
            warnings.warn("The list of pairs should be iterable structures of length two. Using standard header.")
            tmp_header = ", ".join(["Pair {}".format(i+1) for i in range(Npairs)])
        else:
            tmp_header = ", ".join(["{}-{}".format(x,y) for x,y in pairs])

    # Write output files
    rdf_data = np.array(rdf_data).T
    with open(file_out,'w') as ff:
        ff.write("# RDF from consolidated array (probably from MDAnalysis)\n")
        ff.write("# r [Distance Units], {}\n".format(tmp_header))
        for line in rdf_data:
            ff.write(", ".join(str(x) for x in line)+"\n")

def consolidate_lammps(target_dir, boxes, file_in="rdf.txt", file_out="out.csv", pairs=None):
    """
    This function will take equivalent rdf data output from several independent lammps "boxes" and average it together.

    Parameters
    ----------
    target_dir : str
        This string should be common to all equivalent boxes, except for one variable entry to be added with the str format function. (e.g. "path/to/box{}/run1")
    boxes : list
        List of entries that individually complete the path to a lammps rdf output file.
    file_in : str, default='rdf.txt'
        Name of rdf data output from lammps. If None, the filename is included in ``target_dir``
    file_out : str, default='out.csv'
        Filename for consolidated data file, the output file will be equipped with two prefixes, for the rdf\_ and integrated coord\_ data.
    pairs : list, default=None
        Optional list of iterable structure of length 2 each containing pairs of values that represent the rdf

    Returns
    -------
    Comma separated files representing the average rdf, average coordination, rdf boxes, and coordination from boxes.

    """

    # Check inputs
    if not dm.isiterable(boxes):
        raise ValueError("The input `boxes` should be iterable")

    try:
        tmp_file = target_dir.format(boxes[0])
    except:
        raise ValueError("The given input `target_dir` should have a placeholder {} with which to format the path with entries in `boxes`")

    # Consolidate data
    tmp_boxes = []
    for box in boxes:
        filename = os.path.join(target_dir.format(box),file_in) if file_in is not None else target_dir.format(box)
        if os.path.isfile(filename):
            tmp_boxes.append(np.transpose(np.genfromtxt(filename, delimiter=" ", skip_header=4)))
            if np.size(tmp_boxes[-1]) == 0:
                warnings.warn("File empty: {}".format(filename))
                tmp_boxes = tmp_boxes[:-1]

    if len(set([np.shape(x) for x in tmp_boxes])) > 1:
        raise ValueError("Provided files are not the same size: {}".format(filename))

    if not tmp_boxes:
        warnings.warn("Files from: {}, could not be found.".format(filename))
        return
    data_boxes = np.array(tmp_boxes)
    lx = len(data_boxes[0])

    r_data = data_boxes[0][1][None, :]
    rdf_boxes = data_boxes[:, list(range(2,lx,2)), :]
    coord_boxes = data_boxes[:, list(range(3,lx,2)), :]
    rdf, rdf_error = dm.basic_stats( rdf_boxes, axis=0, error_descriptor="sample")
    coord, coord_error = dm.basic_stats( coord_boxes, axis=0, error_descriptor="sample")

    rdf_data = np.transpose(np.concatenate((r_data, np.array([xx for x in zip(rdf, rdf_error) for xx in x])), axis=0))
    coord_data = np.transpose(np.concatenate((r_data, np.array([xx for x in zip(coord, coord_error) for xx in x])), axis=0))

    # Prepare to write output files
    Npairs = int(lx/2 - 1)
    tmp = os.path.split(file_out)
    fout = os.path.join(tmp[0],"{}_"+tmp[1])

    tmp_header = ["r [Distance Units]"] + [x for i in range(Npairs) for x in ["Pair {}".format(i+1), "Pair {} StD".format(i+1)]]
    if len(pairs) != Npairs:
        warnings.warn("The number of given pairs does not match the output, using standard header.")
    elif not all([len(x)==2 for x in pairs]):
        warnings.warn("The list of pairs should be iterable structures of length two. Using standard header.")
    else:
        tmp_header = ["r [Distance Units]"] + [z for x,y in pairs for z in ["{}-{}".format(x,y), "{}-{} StD".format(x,y)]]

    fm.write_csv( fout.format("rdf"), rdf_data, header=tmp_header)
    fm.write_csv( fout.format("coord"), coord_data, header=tmp_header)

def extract_keypoints(r, gr, tol=1e-3, show_fit=False, smooth_sigma=None, error_length=8, save_fit=False, plot_name="rdf.png",title="Pair-RDF", extrema_cutoff=0.01):
    """
    From rdf, extract key points of interest.

    Parameters
    ----------
    r : np.array
        Distance between bead centers
    gr : np.array
        Radial distribution function
    tol : float, default=1e-3
        Tolerance used in determining r_0.
    show_fit : bool, default=False
        Show comparison plot of each rdf being analyzed. Note that is option will interrupt the process until the plot is closed.
    smooth_sigma : float, default=None
        If the data should be smoothed, provide a value of sigma used in `scipy.ndimage.gaussian_filter1d <https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_filter1d.html>`_
    error_length : int, default=8
        The number of extrema found to trigger an error. This indicates the data is noisy and should be smoothed. 
    save_fit : bool, default=False
        Save comparison plot of each rdf being analyzed. With the name ``plot_name``
    plot_name : str, default="rdf.png"
        If `save_fit` is true, the generated plot is saved. The ``title`` is added as a prefix to this str
    title : str, default="Pair-RDF"
        The title used in the pair distribution function plot, note that this str is also added as a prefix to the ``plot_name``.
    extrema_cutoff : float, default=0.01
        All peaks with an absolute value of gr that is less than this value are ignored.

    Returns
    -------
    r_peaks : np.array
        Distance for first through third peak 
    gr_peaks : np.array
        gr values for first through third peaks
    r_mins : np.array
        Distance for first through third minimum
    r_0 : float
        Distance where gr first becomes non-zero, this value is taken directly from the provided data
    r_clust : float
        Distance where gr=1 after the first peak. This metric has been used as a measure of clustering.

    """

    if not dm.isiterable(r):
        raise ValueError("Given distances, r, should be iterable")
    else:
        r = np.array(r)
    if not dm.isiterable(gr):
        raise ValueError("Given radial distribution values, gr, should be iterable")
    else:
        gr = np.array(gr)

    # Find beginning of g(r) where becomes nonzero
    ind_0 = None
    for i in range(len(gr)):
        if gr[i] > tol:
            ind_0 = i-1
            break

    r_0 = r[ind_0]
    rfit = r[ind_0:]
    grfit = gr[ind_0:]

    #######
    #if smooth_sigma == None: # Trying to automate choosing a smoothing value
    #    Npts_avg = 20
    #    stderror = 1e-5
    #    sigma_min = 1.1
    #    window_len = (np.std(grfit[-Npts_avg:])/stderror)**2
    #    smooth_sigma = (window_len-1)/6.0
    #    print(smooth_sigma, window_len, np.std(grfit[-Npts_avg:]))
    #    if smooth_sigma < sigma_min:
    #        smooth_sigma = sigma_min 
    #print(smooth_sigma)
    #grfit = gaussian_filter1d(grfit, sigma=smooth_sigma)
    #######
    if smooth_sigma != None:
        grfit = gaussian_filter1d(grfit, sigma=smooth_sigma)
    ######

    spline = InterpolatedUnivariateSpline(rfit,grfit-1.0)
    roots = spline.roots().tolist()
    spline = InterpolatedUnivariateSpline(rfit,grfit, k=4)
    extrema = spline.derivative().roots().tolist()
    extrema = [x for i,x in enumerate(extrema) if (i<4 or np.abs(spline(x)-1.0)>extrema_cutoff)]
    extrema = [x for x in extrema if np.abs(spline(x))>extrema_cutoff]

    if len(extrema) > error_length:
        if show_fit:
            plt.plot(r,gr,"k",label="Data")
            plt.plot([r[0],r[-1]],[1,1],"k",linewidth=0.5)
            plt.plot([r[0],r[-1]],[0,0],"k",linewidth=0.5)
            r2 = np.linspace(rfit[0],rfit[-1],int(1e+4))
            gr2 = spline(r2)
            for tmp in extrema:
                plt.plot([tmp,tmp],[0,np.max(gr)],"c",linewidth=0.5)
            plt.plot(r2,gr2,"r",label="Spline",linewidth=0.5)
            plt.show()
        warnings.warn("Found {} extrema, consider smoothing the data with `smooth_sigma` option.".format(len(extrema)))

    tmp_spline = InterpolatedUnivariateSpline(rfit,grfit, k=5).derivative().derivative()
    concavity = tmp_spline(extrema)

    r_peaks = []
    gr_peaks = []
    r_mins = []
    for i,rtmp in enumerate(extrema):
        if concavity[i] > 0:
            r_mins.append(rtmp)
        else:
            r_peaks.append(rtmp)
            gr_peaks.append(spline(rtmp))

        if i == 8 or len(r_mins) == 3:
            break

    if len(r_peaks) < 3:
        for i in range(3-len(r_peaks)):
            r_peaks.append(np.nan)
            gr_peaks.append(np.nan)
    elif len(r_peaks) > 3:
        r_peaks = r_peaks[:3]
        gr_peaks = gr_peaks[:3]
    if len(r_mins) != 3:
        for i in range(3-len(r_mins)):
            r_mins.append(np.nan)

    if len(roots) < 2:
        if (len(roots) == 1 and r_peaks[0] != np.nan) and roots[0] > r_peaks[0]:
                r_clust = roots[0]
        else:
            if title != "Pair-RDF":
                tmp = title
            elif plot_name != "rdf.png":
                tmp = plot_name.split(".")[0]
            else:
                tmp = ""
            warnings.warn("{} RDF has not decayed to unity. Consider regenerating data with a larger cutoff.".format(tmp))
            r_clust = np.nan
    else:
        r_clust = roots[1]

    if show_fit or save_fit:
        plt.plot(r,gr,"k",label="Data")
        plt.plot([r[0],r[-1]],[1,1],"k",linewidth=0.5)
        plt.plot([r[0],r[-1]],[0,0],"k",linewidth=0.5)
        r2 = np.linspace(rfit[0],rfit[-1],int(1e+4))
        gr2 = spline(r2)
        plt.plot(r2,gr2,"r",linewidth=0.5,label="Spline")
        for i in range(len(r_peaks)):
            plt.plot([r_peaks[i],r_peaks[i]],[min(gr),max(gr)],"c",linewidth=0.5)
        for i in range(len(r_mins)):
            plt.plot([r_mins[i],r_mins[i]],[min(gr),max(gr)],"b",linewidth=0.5)
        plt.plot([r_0,r_0],[min(gr),max(gr)],"m",linewidth=0.5)
        plt.plot([r_clust,r_clust],[min(gr),max(gr)],"m",linewidth=0.5)
        plt.ylabel("g(r)")
        plt.xlabel("r [Distance Units]")
        plt.title(title)
        plt.xlim(rfit[0],rfit[-1])
        plt.tight_layout()
        if save_fit:
            tmp = os.path.split(plot_name)
            filename = os.path.join(tmp[0],title.replace(" ", "")+"_"+tmp[1])
            plt.savefig(filename, dpi=300)
        if show_fit:
            plt.show()
        plt.close()

    if len(extrema) > error_length:
        return np.ones(3)*np.nan, np.ones(3)*np.nan, np.ones(3)*np.nan, np.nan, np.nan
    else:
        return np.array(r_peaks), np.array(gr_peaks), np.array(r_mins), r_0, r_clust


def keypoints2csv(filename, fileout="rdf.csv", mode="a", titles=None, additional_entries=None, additional_header=None, extract_keypoints_kwargs={}, file_header_kwargs={}, column_list=None, kwargs_genfromtxt={}):
    """
    Given the path to a csv file containing rdf data, extract key values and save them to a .csv file. The file of rdf data should have a first column with distance values, followed by columns with radial distribution values. These data sets will be distinguished in the resulting csv file with the column headers

    Parameters
    ----------
    filename : str
        Input filename and path to lammps rdf output file
    fileout : str, default="rdf.csv"
        Filename of output .csv file
    mode : str, default="a"
        Mode used in writing the csv file, either "a" or "w".
    titles : list[str], default=None
        Titles for plots if that is specified in the ``extract_keypoints_kwargs``
    additional_entries : list, default=None
        This iterable structure can contain additional information about this data to be added to the beginning of the row
    additional_header : list, default=None
        If the csv file does not exist, these values will be added to the beginning of the header row. This list must be equal to the `additional_entries` list.
    extract_keywords_kwargs : dict, default={}
        Keywords for `extract_keypoints` function
    file_header_kwargs : dict, default={}
        Keywords for :func:`md_spa.utils.os_manipulation.file_header` function    
    column_list : list, default=None
        If not None, this list specifies the column indices (starting from 0) to analyze. Note, the first column must be the distance.
    kwargs_genfromtxt : dict, default={}
        Dictionary of keyword arguments for `numpy.genfromtxt() <https://numpy.org/doc/stable/reference/generated/numpy.genfromtxt.html>`_ to import the file, ``filename``.

    Returns
    -------
    csv file
    
    """

    if not os.path.isfile(filename):
        raise ValueError("The given file could not be found: {}".format(filename))

    if titles == None:
        titles = fm.find_header(filename, **file_header_kwargs)
        if column_list is not None:
            titles = titles[column_list]

    data = np.transpose(np.genfromtxt(filename, **kwargs_genfromtxt))
    if np.size(data) == 0:
        warnings.warn("File empty: {}".format(filename))
        return

    if column_list is not None:
        try:
            data = data[column_list]
        except:
            raise ValueError("Provided column_indices do not align with the number of file columns, {}".format(len(data)))
    if len(titles) != len(data):
        raise ValueError("The number of titles does not equal the number of columns")

    if np.all(additional_entries != None):
        flag_add_ent = True
        if not dm.isiterable(additional_entries):
            raise ValueError("The provided variable `additional_entries` must be iterable")
    else:
        flag_add_ent = False
        additional_entries = []
    if np.all(additional_header != None):
        flag_add_header = True
        if not dm.isiterable(additional_header):
            raise ValueError("The provided variable `additional_header` must be iterable")
    else:
        flag_add_header = False
 
    if flag_add_ent:
        if flag_add_header:
            if len(additional_entries) != len(additional_header):
                raise ValueError("The variables `additional_entries` and `additional_header` must be of equal length")
        else:
            additional_header = ["-" for x in additional_entries]
            flag_add_header = True

    r = data[0]
    tmp_data = []
    for i in range(1,len(data)):
        tmp_extract_keypoints_kwargs = copy.deepcopy(extract_keypoints_kwargs)
        if "title" not in tmp_extract_keypoints_kwargs:
            tmp_extract_keypoints_kwargs["title"] = titles[i]
        tmp = extract_keypoints(r,data[i], **tmp_extract_keypoints_kwargs)
        if not np.all(np.isnan(np.array(tmp[0]))):
            tmp_data.append(list(additional_entries)
                +[titles[i]]+list(tmp[0])+list(tmp[1])+list(tmp[2])+list(tmp[3:]))

    file_headers = ["Pairs", "r_peak1", "r_peak2", "r_peak3", "gr_peak1", "gr_peak2", "gr_peak3", "r_min1", "r_min2", "r_min3", "r_0", "r_clust"]
    if len(tmp_data) == 0:
        return
    if not os.path.isfile(fileout) or mode=="w":
        if flag_add_header:
            file_headers = list(additional_header) + file_headers
        fm.write_csv(fileout, tmp_data, mode=mode, header=file_headers)
    else:
        fm.write_csv(fileout, tmp_data, mode=mode)


def extract_debye_waller(r, gr, tol=1e-3, show_fit=False, smooth_sigma=None, error_length=8, save_fig=False, plot_name="rdf_debye-waller.png",title="Pair-RDF", extrema_cutoff=0.01, verbose=False):
    """ Extract Debye-Waller FACTOR (not parameter like from MSD)

    Based on the work found at `DOI: 10.1021/jp064661f <https://doi.org/10.1021/jp064661f>`_

    Parameters
    ----------
    r : np.array
        Distance between bead centers
    gr : np.array
        Radial distribution function
    tol : float, default=1e-3
        Tolerance used in determining where the rdf function first becomes nonzero.
    show_fit : bool, default=False
        Show comparison plot of each rdf being analyzed. Note that is option will interrupt the process until the plot is closed.
    smooth_sigma : float, default=None
        If the data should be smoothed, provide a value of sigma used in `scipy.ndimage.gaussian_filter1d <https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_filter1d.html>`_
    error_length : int, default=8
        The number of extrema found to trigger an error. This indicates the data is noisy and should be smoothed. 
    save_fit : bool, default=False
        Save comparison plot of each rdf being analyzed. With the name ``plot_name``
    plot_name : str, default="rdf_debye-waller.png"
        If `save_fit` is true, the generated plot is saved. The ``title`` is added as a prefix to this str
    title : str, default="Pair-RDF"
        The title used in the pair distribution function plot, note that this str is also added as a prefix to the ``plot_name``.
    extrema_cutoff : float, default=0.01
        All peaks with an absolute value of gr that is less than this value are ignored.
    verbose : bool, default=False
        Set whether calculation will be run with comments

    Returns
    -------
    parameters : numpy.ndarray
        Array containing parameters: ["amplitude", "center", "sigma", "fwhm", "height"] where sigma is the debye-waller factor and the amplitude is equal to N_0/rho_0, (coordination number) / (number density)
    stnd_errors : numpy.ndarray
        Array containing uncertainties for parameters
    redchi : float
        Reduced Chi^2 from `lmfit.MinimizerResult <https://lmfit.github.io/lmfit-py/fitting.html#lmfit.minimizer.MinimizerResult>`_

    """

    flag = None

    if not dm.isiterable(r):
        raise ValueError("Given distances, r, should be iterable")
    else:
        r = np.array(r)
    if not dm.isiterable(gr):
        raise ValueError("Given radial distribution values, gr, should be iterable")
    else:
        gr = np.array(gr)

    # Find beginning of g(r) where becomes nonzero
    ind_0 = None
    for i in range(len(gr)):
        if gr[i] > tol:
            ind_0 = i-1
            break

    rfit = r[ind_0:]
    grfit = gr[ind_0:]

    if len(rfit) < 10:
        raise ValueError("RDF has too few points to analyze with length {}".format(len(rfit)))

    # Smooth data if specified
    if smooth_sigma != None:
        grfit = gaussian_filter1d(grfit, sigma=smooth_sigma)

    # Fit spline and identify extrema
    spline = InterpolatedUnivariateSpline(rfit,grfit-1.0)
    roots = spline.roots().tolist()
    spline = InterpolatedUnivariateSpline(rfit,grfit, k=4)
    extrema = spline.derivative().roots().tolist()
    extrema = [x for i,x in enumerate(extrema) if (i<4 or np.abs(spline(x)-1.0)>extrema_cutoff)]
    extrema = [x for x in extrema if np.abs(spline(x))>extrema_cutoff]

    if len(extrema) > error_length:
        flag = "extrema"

    tmp_spline = InterpolatedUnivariateSpline(rfit,grfit, k=5).derivative().derivative()
    inflection = tmp_spline.roots().tolist()
    concavity = tmp_spline(extrema)
    inflection = [x for x in inflection if x < extrema[-1]]

    r_peak = None
    r_min = None
    for i,rtmp in enumerate(extrema):
        if concavity[i] > 0 and r_min == None:
            r_min = rtmp
        elif concavity[i] < 0 and r_peak == None:
            r_peak = rtmp

        if r_min != None and r_peak != None:
            break

    r_inflection = [None, None]
    for rtmp in inflection:
        if rtmp < r_peak:
            if r_inflection[0] == None:
                r_inflection[0] = rtmp
            else:
                flag = "first inflection"

        if rtmp > r_peak and rtmp < r_min:
            if r_inflection[1] == None:
                r_inflection[1] = rtmp
            else:
                flag = "second inflection"

        if not None in r_inflection:
            break

    # Throw errors if any with extrema or inflection points
    if flag != None:
        if show_fit:
            plt.plot(r,gr,"k",label="Data")
            plt.plot([r[0],r[-1]],[1,1],"k",linewidth=0.5)
            plt.plot([r[0],r[-1]],[0,0],"k",linewidth=0.5)
            r2 = np.linspace(rfit[0],rfit[-1],int(1e+4))
            gr2 = spline(r2)
            for tmp in extrema:
                plt.plot([tmp,tmp],[0,np.max(gr)],"c",linewidth=0.5)
            for tmp in inflection:
                plt.plot([tmp,tmp],[0,np.max(gr)],"m",linewidth=0.5)
            plt.plot(r2,gr2,"r",label="Spline",linewidth=0.5)
            plt.show()

        if flag == "extrema":
            raise ValueError("Found {} extrema, consider smoothing the data with `smooth_sigma` option.".format(len(extrema)))
        elif flag == "first inflection":
            raise ValueError("Found more than one inflection point before first peak.")
        elif flag == "second inflection":
            raise ValueError("Found more than one inflection point between first peak ({}) and first minimum ({}): {}".format(r_peak, r_min, inflection))

    if None in r_inflection:
        raise ValueError("Two inflection points are needed for gaussian fitting.")

    # Use inflection points to bound actual data
    ind_r_lower = np.where((r-r_inflection[0]) > 0)[0][0]
    ind_r_upper = np.where((r-r_inflection[1]) > 0)[0][0]

    r_array = r[ind_r_lower:ind_r_upper]
    y_array = 4*np.pi*r_array**2*gr[ind_r_lower:ind_r_upper]

    set_params = {
                  #"center": {"vary": False, "value": r_peak},
                  "height": {"vary": False, "value": np.max(y_array), "expr": None},
                  #"fwhm": {},
                  "amplitude": {"expr": "height*sigma/0.3989423"}, # Equal to N_0/rho_0, (coordination number) / (number density)
                  #"sigma": {}, # debye-waller factor
                 }

    output, uncertainty, redchi = cfit.gaussian( r_array, y_array, set_params=set_params, verbose=verbose)
    # "amplitude", "center", "sigma", "fwhm", "height"
    output[-1] = output[-1]/(4*np.pi*r_array[np.where(y_array==np.max(y_array))[0][0]]**2)

    if show_fit or save_fit:
        # Figure 1
        plt.figure(1)
        plt.plot(r,gr,"k",label="Data", linewidth=0.5)
        plt.plot([r[0],r[-1]],[1,1],"k",linewidth=0.5)
        plt.plot([r[0],r[-1]],[0,0],"k",linewidth=0.5)

        r2 = np.linspace(rfit[0],rfit[-1],int(1e+4))
        gr2 = spline(r2)
        plt.plot(r2,gr2,"r",linewidth=0.5,label="Spline")

        gr2 = output[0]/output[2]/np.sqrt(2*np.pi)*np.exp(-(r2-output[1])**2/(2*output[2]**2))/(4*np.pi*r2**2)
        plt.plot(r2,gr2,"--g",linewidth=0.5,label="Gaussian")
        plt.plot([r_peak,r_peak],[min(gr),max(gr)],"c",linewidth=0.5)
        for i in range(len(r_inflection)):
            plt.plot([r_inflection[i],r_inflection[i]],[min(gr),max(gr)],"m",linewidth=0.5)
        plt.ylabel("g(r)")
        plt.xlabel("r [Distance Units]")
        plt.title(title)
        plt.xlim(rfit[0],r_min)
        plt.legend(loc="best")
        plt.tight_layout()
        if save_fit:
            tmp = os.path.split(plot_name)
            filename = os.path.join(tmp[0],"rdf_"+title.replace(" ", "")+"_"+tmp[1])
            plt.savefig(filename, dpi=300)

        # Figure 2
        plt.figure(2)
        plt.plot(r_array,y_array,"k",label="Data", linewidth=0.5)
        plt.plot([r[0],r[-1]],[1,1],"k",linewidth=0.5)
        plt.plot([r[0],r[-1]],[0,0],"k",linewidth=0.5)

        r2 = np.linspace(*r_inflection,int(1e+3))
        gr2 = output[0]/output[2]/np.sqrt(2*np.pi)*np.exp(-(r2-output[1])**2/(2*output[2]**2))
        plt.plot(r2,gr2,"--g",linewidth=0.5,label="Gaussian")
        plt.ylabel("4$\pi$$r^2$g(r)")
        plt.xlabel("r [Distance Units]")
        plt.title(title)
        plt.xlim(rfit[0],r_min)
        plt.legend(loc="best")
        plt.tight_layout()
        if save_fit:
            tmp = os.path.split(plot_name)
            filename = os.path.join(tmp[0],"fit_"+title.replace(" ", "")+"_"+tmp[1])
            plt.savefig(filename, dpi=300)
        if show_fit:
            plt.show()
        plt.close("all")

    return output, uncertainty, redchi

def debye_waller2csv(filename, fileout="debye-waller.csv", mode="a", delimiter=",", titles=None, additional_entries=None, additional_header=None, extract_debye_waller_kwargs={}, file_header_kwargs={}):
    """
    Given the path to a csv file containing rdf data, extract key values and save them to a .csv file. The file of rdf data should have a first column with distance values, followed by columns with radial distribution values. These data sets will be distinguished in the resulting csv file with the column headers

    Parameters
    ----------
    filename : str
        Input filename and path to lammps rdf output file
    fileout : str, default="rdf.csv"
        Filename of output .csv file
    mode : str, default="a"
        Mode used in writing the csv file, either "a" or "w".
    delimiter : str, default=","
        Delimiter between data in input file
    titles : list[str], default=None
        Titles for plots if that is specified in the ``extract_debye_waller_kwargs``
    additional_entries : list, default=None
        This iterable structure can contain additional information about this data to be added to the beginning of the row
    additional_header : list, default=None
        If the csv file does not exist, these values will be added to the beginning of the header row. This list must be equal to the `additional_entries` list.
    extract_debye_waller_kwargs : dict, default={}
        Keywords for :func:`md_spa.rdf.extract_debye_waller`
    file_header_kwargs : dict, default={}
        Keywords for :func:`md_spa.utils.os_manipulation.file_header` function    

    Returns
    -------
    csv file
    
    """
    if not os.path.isfile(filename):
        raise ValueError("The given file could not be found: {}".format(filename))

    if titles == None:
        titles = fm.find_header(filename, **file_header_kwargs)
    data = np.transpose(np.genfromtxt(filename, delimiter=delimiter))
    if len(titles) != len(data):
        raise ValueError("The number of titles does not equal the number of columns")

    if np.all(additional_entries != None):
        flag_add_ent = True
        if not dm.isiterable(additional_entries):
            raise ValueError("The provided variable `additional_entries` must be iterable")
    else:
        flag_add_ent = False
    if np.all(additional_header != None):
        flag_add_header = True
        if not dm.isiterable(additional_header):
            raise ValueError("The provided variable `additional_header` must be iterable")
    else:
        flag_add_header = False

    if flag_add_ent:
        if flag_add_header:
            if len(additional_entries) != len(additional_header):
                raise ValueError("The variables `additional_entries` and `additional_header` must be of equal length")
        else:
            additional_header = ["-" for x in additional_entries]
            flag_add_header = True

    r = data[0]
    tmp_data = []
    for i in range(1,len(data)):
        if "title" not in extract_debye_waller_kwargs:
            extract_debye_waller_kwargs["title"] = titles[i]
        tmp = extract_debye_waller(r,data[i], **extract_debye_waller_kwargs)
        tmp_data.append(list(additional_entries)
            +[titles[i]]+list(tmp[0])+list(tmp[1]))


    file_headers = ["Pairs", "amplitude", "center", "sigma", "fwhm", "height", "amplitude SE", "center SE", "sigma SE", "fwhm SE", "height SE"]
    if not os.path.isfile(fileout) or mode=="w":
        if flag_add_header:
            file_headers = list(additional_header) + file_headers
        fm.write_csv(fileout, tmp_data, mode=mode, header=file_headers)
    else:
        fm.write_csv(fileout, tmp_data, mode=mode)


def spherical_distribution_function(xdata, ydata, smear_radius=2.3):
    """Produce a smeared Radial Distribution Function
    
    Assume that xdata points are evenly distributed. The contents of this function were published in
    Clark and Douglas, Hydration Contribution to the Solvation Free Energy of Water-Soluble Polymers

    Parameters
    ----------
    xdata : numpy.ndarray
        Distance values corresponding to the radial distribution function
    ydata : numpy.ndarray
        Values from the radial distribution function
    smear_radius : float, optional
        Radius over which to smear the radial distribution function, by default 2.3
        
    Returns
    -------
    xdata_new : numpy.ndarray
        Distance values corresponding to the spherical distribution function that have
        been truncated from the original due to the spherical smearing size.
    ydata_new : numpy.ndarray
        Values from the spherical distribution function    
    """
    
    xdata, ydata = np.array(xdata), np.array(ydata)
    dr = xdata[1] - xdata[0]
    if np.isnan(dr):
        raise ValueError("Initial values of distance, xdata, array should not be NaN.")
    npoints = int(smear_radius // dr)
    smear = npoints * dr

    new_r = xdata[np.where(xdata < xdata[-1] - smear_radius)]
    
    new_rdf = np.zeros_like(new_r)
    for i, r in enumerate(new_r):
        if npoints < 2:
            new_rdf[i] = ydata[np.where(xdata == r)[0]] 
            warnings.warn("Smear radius is less then steps in distance. Increase the smear radius.")
        else:
            ind1 = max(int(i - npoints), 0)
            ind2 = int(i + npoints + 1)
            if ind1 == 0:
                ind3 = abs(i - npoints)
                xdata_subset = np.concatenate(( -np.flip(xdata[1:ind3]), xdata[ind1:ind2])) - r
                ydata_subset = np.concatenate((np.flip(ydata[1:ind3]), ydata[ind1:ind2]))
            else:
                xdata_subset, ydata_subset = xdata[ind1:ind2] - r, ydata[ind1:ind2]

            p_Acircle = np.pi * (r + xdata_subset) / r * (smear**2 - xdata_subset**2) # Area of slice
            norm = 4 / 3 * np.pi * smear**3
            new_rdf[i] = spi.simpson(
                ydata_subset * p_Acircle,
                x=xdata_subset
            ) / norm
    
    return new_r, new_rdf


def kbintegral(xdata, ydata, volume, closed_system_correction="Lockwood", geometry="spherical", height=None):
    """Compute the Kirkwood Buff Integral values with r dependence
    
    This function corrects for the closed systems used in NVT and 
    NPT MD simulations.

    Parameters
    ----------
    xdata : numpy.ndarray
        Distance values that correspond to the radial distribution function
    ydata : numpy.ndarray
        Values in the radial distribution function
    volume : float
        Volume of the simulation box
    closed_system_correction : str, default="Kruger"
        Correction used for closed system. May be:
        
        - "Kruger" from from Kruger et al. J. Phys. Chem. Lett. 2013, 4, 235-238 (dx.doi.org/10.1021/jz301992u, 10.1080/00268976.2018.1434908)
        - "Lockwood" from Lockwood and Rossky, J. Phys. Chem. B 1999, 103 1982, meant for an NVT system that is
          equilibrated for solvent conditions and a solute is added
        - ``None``
        
    geometry : str, default="spherical"
        Geometry of the system, most commonly a ``"spherical"``, but can also be ``"cylindrical"``,
        in which case, ``height`` must be defined.
    height : float, default=None
        Height of the box representing the axial direction of a cylinder.

    Returns
    -------
    numpy.ndarray
        Kirkwood Buff Integral values dependent on cutoff value
    """
    
    xdata, ydata = np.array(xdata), np.array(ydata)
    
    kbi = np.zeros_like(xdata)
    for i, _ in enumerate(xdata):
        if i == 0:
            kbi[i] = 0
        else:
            if geometry == "spherical":
                geomfactor = 4 * np.pi * xdata[:i]**2
            elif geometry == "cylindrical":
                geomfactor = height * 2 * np.pi * xdata[:i] 
            else:
                raise ValueError("Geometry option, {}, is not recognized. Choose 'spherical' or 'cylindrical'".format(geometry))
                
            if closed_system_correction == "Kruger":
                x = xdata[:i]/xdata[i]
                if geometry == "spherical":
                    factor = (1.0 - 1.5 * x + 0.5 * x**3)
                elif geometry == "cylindrical":
                    raise ValueError("The Krugar estimation method is not defined for the cylindrical distribution function.")
                else:
                    raise ValueError("Geometry option, {}, is not recognized. Choose 'spherical' or 'cylindrical'".format(geometry))
                correction = 1
            elif closed_system_correction == "Lockwood":
                factor = 1
                correction = 1 - 1 / volume * spi.simpson(
                    ydata[:i] * geomfactor,
                    x=xdata[:i]
                )  
            elif closed_system_correction is None:
                correction, factor = 1, 1
            else:
                raise ValueError(
                    f"Closed system correction, {closed_system_correction} not recognized."
                    " Choose either 'Kruger', 'Lockwood', or None"
                )
                
            kbi0 = -spi.simpson(
                (1 - ydata[:i]) * geomfactor * factor,
                x=xdata[:i]
            )
            kbi[i] = kbi0 / correction
 
    return kbi          


def extrapolate_kbi(rinv, kbi_array, rmin, rmax=None, save_fig=False, plot_name="kbi.png", show_fig=False):
    """Extrapolate the Kirkwood Buff Integral (KBI) to zero

    This method is used for the calculation of KBI's presented by Kruger where the formulation results
    in an expression with a linear dependence on 1 / L, where L is the cutoff in r for G(L).

    Parameters
    ----------
    rinv : numpy.ndarray
        Inverse distance used in calculating the KBI
    kbi_array : numpy.ndarray
        Kirkwood Buff Integral from r is zero to some value of r
    rmin : float
        Distance used for the upper bound (in inverse distance) to calculate the KBI
    rmax : float, default=None
        Distance used for the lower bound (in inverse distance) to calculate the KBI
    save_fit : bool, default=False
        Save comparison plot of each rdf being analyzed. With the name ``plot_name``
    plot_name : str, default="kbi.png"
        If `save_fig` is true, the generated plot is saved. The ``title`` is added as a prefix to this str
    show_fig : bool, default=False
        Show figure
        
    Return
    ------
    kbi : float
        Kirkwood Buff Integral extrapolated to infinity
    kbi_se : float
        Standard error of KBI
    npoints : int
        Number of points used in the calculation
    
    """
    rinv, kbi_array = np.array(rinv), np.array(kbi_array)
    
    if rmax is None:
        rmax = np.max(1/rinv)
    ind = np.where(np.logical_and(rinv < 1/rmin, rinv > 1/rmax))
    npoints = len(ind[0])
    
    result = sps.linregress(
        rinv[ind], 
        kbi_array[ind],
    )
    kbi = result.intercept
    
    if save_fig or show_fig:
        plt.close("all")
        plt.plot(rinv, kbi_array)
        tmp = np.array([1/rmax, 1/rmin])
        plt.plot(tmp, tmp*result.slope + kbi, marker="o")
        
        plt.plot([0], [kbi], marker="s")
        plt.xlabel("1 / $r$")
        plt.ylabel("Kirkwood Buff Integral, G")
        plt.xlim((0,1/rmin))
        yminmax = np.concatenate((kbi_array[np.where(rinv < 1/rmin)], [kbi]))
        dy = np.abs(np.max(yminmax) - np.min(yminmax)) * 0.05
        plt.ylim((np.min(yminmax)-dy, np.max(yminmax)+dy))
        if save_fig:
            plt.savefig(plot_name, dpi=300)
        if show_fig:
            plt.show()
        plt.close("all")

    return kbi, result.intercept_stderr, npoints
