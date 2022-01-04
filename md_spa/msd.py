
#import read_lammps as f
import sys
import numpy as np
import warnings
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.ndimage.filters import gaussian_filter1d
from scipy.stats import linregress
from scipy.interpolate import InterpolatedUnivariateSpline

import md_spa_utils.os_manipulation as om

def debye_waller(time, msd, show_plot=False, save_plot=False, plot_name="debye-waller.png", smooth_sigma=None, verbose=False):
    """
    Analyzing the ballistic region of an MSD curve yields the debye-waller factor, which relates to the cage region that the atom experiences.

    Parameters
    ----------
    time : numpy.ndarray
        Time array of the same length at MSD
    msd : numpy.ndarray
        MSD array with one dimension
    save_plot : bool, Optional, default=False
        choose to save a plot of the fit
    show_plot : bool, Optional, default=False
        choose to show a plot of the fit
    plot_name : str, Optional, default="debye-waller.png"
        If ``save_plot==True`` the msd will be saved with the debye-waller factor marked
    smooth_sigma : float, Optional, default=None
        sigma factor used to smooth msd data with ``scipy.ndimage.filters.gaussian_filter1d``. Note that this may be useful to approximate the debye-waller factor, but another msd should be computed with additional statistics so that this feature isn't needed.
    verbose : bool, Optional, default=False
        Will print intermediate values or not
    
    Returns
    -------
    debye-waller : float

    """

    if not om.isiterable(time):
        raise ValueError("Given distances, time, should be iterable")
    else:
        time = np.array(time)
    if not om.isiterable(msd):
        raise ValueError("Given radial distribution values, msd, should be iterable")
    else:
        msd = np.array(msd)
 
    if len(msd) != len(time):
        raise ValueError("Arrays for time and msd are not of equal length.")

    if smooth_sigma != None:
        msd = gaussian_filter1d(msd, sigma=smooth_sigma)
        warnings.warn("To properly calculate the debye-waller factor the msd should be smooth. Consider using a smaller section with the ballistic region or increase your statistics.")

    spline = InterpolatedUnivariateSpline(time,msd, k=4)
    dspline = spline.derivative()
    extrema = dspline.roots().tolist()
    if len(extrema) > 2:
        dw = np.nan
        warnings.warn("Found {} extrema, consider smoothing the data with `smooth_sigma` option for an approximate value, and then increase the statistics used to calculate the msd.".format(len(extrema)))
    elif len(extrema) == 2:
        dw = spline(extrema[1])
        if verbose:
            print("Found debye waller factor to be {}".format(dw))
    else:
        warnings.warn("This msd array does not contain a caged-region")
        dw = np.nan

    if save_plot or show_plot:
        plt.figure(1)
        plt.plot(time,msd,"k",label="Data")
        if len(extrema) > 1:
            for tmp in extrema[1:]:
                plt.plot([tmp,tmp],[0,np.max(msd)])
        plt.xlabel("time")
        plt.ylabel("distance squared")
        plt.tight_layout()
        if save_plot:
            plt.savefig(plot_name,dpi=300)
        plt.figure(2)
        plt.plot(time, dspline(time),"k",linewidth=1)
        plt.tight_layout()
        if show_plot:
            plt.show()
        plt.close()

    return dw

def msd_find_diffusivity( time, msd, Nskip=1):
    """
    This function 

    # Inputs:
    # t: [np.array]
    # msd: [np.array]
    # blavg: [str] yes, Yes y or something else, will the algorithm block average the data before evaluating the diffusivity
    # fname: [str] name of output plot (commented out now)
    # Output:
    # output: [list] of length 5 or 10. The first three are always the best diffusivity, the standard error, and the slope of the loglog plot that produces this value, and the time interval used to evaluate this. If the best loglog slope exceeds .991 then the same enteries are repeated for the longest time interval with a loglog slope of at least .991. Note that if block averaging is used, the standard error will be negligible.
    # np.array([t,msd]): [list][np.arrays] temperature and msd vectors the diffusivity is taken from. These vectors will be the same as the input if there is no block averaging, or if the time and msd vectors that are input don't start at zero.

    """

    # User defined variable:
    print("1% of number of points is", int(len(t)*.01))
    #Nskip = 10
    Nskip = int(len(t)*.01)

    # NoteHere
    #Npt_bl = 10
    Npt_bl = Nskip
    print("There are", len(t)/Npt_bl, "blocks")

# Make a function to block average the msd from lammps !!!!!!!!!!!!!
    # Block Averaging
    t = np.array(t)-t[0]
    msd = np.array(msd)-msd[0]
    Nskip1 = 1
    if blavg in ["yes","y","Y","Yes"]:
        tmp_msd = [msd]
        tmp = msd
        for j in range(int(len(t)/Nskip1)-2):
            tmp = tmp[Nskip1:]-tmp[Nskip1]
            pad = [np.nan for i in range(Nskip1*j)]
            tmp_msd.append(np.array(tmp.tolist()+pad))
        tmp_msd[0] = tmp_msd[0][:len(tmp_msd[1])] # Shorten so all vectors are the same length
        msd2 = np.nanmean(np.array(tmp_msd),axis=0)

#        plt.plot(t,msd,label="Raw Data")
#        plt.plot(t[:len(msd2)],msd2,label="Block Averaged")
#        plt.legend(loc="best")
#        plt.show()        

        t = t[:len(msd2)][1:] - t[1]
        msd = msd2[1:] - msd2[1]

    # Convert data
    t_log = np.log(t)
    msd_log = np.log(msd)

    m_best=5
    interval = [0,0] # int_min, int_max
    data_size = int(len(t)*.7-len(t)*.7%Nskip)
    print("We have", data_size, "points after cutting off 30%")
    int_sizes = np.linspace(50,data_size,(data_size-50)/Nskip,endpoint=False).tolist()
    int_sizes.append(data_size)
    int_sizes = [ int(x) for x in int_sizes]

    sets = [[],[]]
    for i in int_sizes:
        int_min = 0
        int_max = i
        sets_m = 5
        for j in range(int((data_size-i)/Nskip)):
            m, inter, r_tmp, p, stder = linregress(t_log[int_min:int_max],msd_log[int_min:int_max])
            if np.abs(m-1)<.009:
                #print(m, t[int_min], t[int_max], t[0], t[-1])
                if np.abs(m-1) < np.abs(sets_m-1):
                    sets_int = [int_min,int_max]
                    sets_m = m
            if np.abs(m-1) < np.abs(m_best-1):
                interval = [int_min,int_max]
                m_best = m
            int_max+=Nskip
            int_min+=Nskip
 
        if np.abs(sets_m-1) < .009:
            sets[0].append(sets_int)
            sets[1].append(sets_m)
 
    # Best interval
    Nblocks = int(len(t[interval[0]:interval[1]])/Npt_bl)
    D = []
    for i in range(Nblocks+1):
        D6_tmp, inter,r_tmp,p,stder_tmp = linregress(t[interval[0]:interval[1]],msd[interval[0]:interval[1]])
        D.append(D6_tmp/6)
    Dfinal = np.mean(D)
    stderr = np.std(D)/np.sqrt(len(D))
 #   print("_________________________________________")
 #   print("Best Interval:", t[interval[0]], "to", t[interval[1]], "steps")
 #   print("D +- stderr in ang^2/step")
 #   print(Dfinal, stderr)
 #   print("t to the power of:", m_best)
    output = [Dfinal, stderr, m_best, t[interval[0]], t[interval[1]]]

    # Plot data
#    plt.plot(t,msd)
   # plt.loglog(t,msd)
   # plt.plot([t[interval[0]],t[interval[0]]],[0, msd.max()],"g",label="Best Interval")
   # plt.plot([t[interval[1]],t[interval[1]]],[0, msd.max()],"g")
#    plt.show()

    # Longest interval
    if len(sets[0])>0:
        interval = sets[0][-1]
        Nblocks = int(len(t[interval[0]:interval[1]])/Npt_bl)
        D = []
        for i in range(Nblocks+1):
            D6_tmp, inter,r_tmp,p,stder_tmp = linregress(t[interval[0]:interval[1]],msd[interval[0]:interval[1]])
            D.append(D6_tmp/6) 
        Dfinal = np.mean(D)
        stderr = np.std(D)/np.sqrt(len(D))
   #     print("_________________________________________")
   #     print("Longest Interval:", t[interval[0]], "to", t[interval[1]], "steps")
   #     print("D +- stderr in ang^2/step")
   #     print(Dfinal, stderr)
   #     print("t to the power of:", m_best)
        output.extend([Dfinal, stderr, m_best, t[interval[0]], t[interval[1]]])

#        plt.plot([t[interval[0]],t[interval[0]]],[0, msd.max()],"m", label="Longest Int")
#        plt.plot([t[interval[1]],t[interval[1]]],[0, msd.max()],"m")
    #    plt.xlabel("timesteps")
    #    plt.ylabel("msd $\AA^2$")
    #    plt.savefig(fname+".png")
    #    plt.close()
#        plt.legend(loc="best")
#        plt.show()

    return output, np.array([t,msd])
#
#########################################################################
##                                                                      #
##          Plot MSDa                                                   #
##                                                                      #
#########################################################################
#
#def msd_plot(paths,Rc,blavg,fnames,Nboxes):
#
#    # Input:
#    #    path [str]: path to files
#    #    Rc [float]: CG length scale
#    #    blavg [str]: says whether to block average the data
#    #    fnames [list]: List of strings for lengend
#    #    Nboxes [int]: Number of independent boxes for each system
#    # Output:
#    #    Figure of the water and proton diffusion of the systems represented by each path
#    #    .txt files of the data ussed in the figure
#
#    Nwat = 3
#
#    #CGt_conv = 0.02*10.4*1e-12 # 0.2 timestep to time units * 10.4 ps per time unit /1000 ps to seconds
#    #CGr_conv = Rc**2*1e-16 # distance units to cm^2
#
#    #CGt_conv = 0.02*10.4 # 0.2 timestep to time units * 10.4 ps per time unit /1000 ps to seconds
#    #CGr_conv = Rc**2/100 # distance units to angstrom^2
#
#    CGt_conv = 0.02 # 0.2 timestep to time units * 10.4 ps per time unit /1000 ps to seconds
#    CGr_conv = 1 # distance units to angstrom^2
#
#    print "Warning: time step is hard coded at 0.02"
#    rng = range(1,Nboxes+1)
##    style = ["","--","-.",":"]
#    style = ["","--","-.",":"]
#    print "Only enough line styles for up to 4 systems"
#
#    mpl.rc('font',family='Times New Roman')
#    lnwd=1
#    f1 = mpl.font_manager.FontProperties(family='Times New Roman', size=12)
#    f2 = mpl.font_manager.FontProperties(family='Times New Roman', size=10)
#    plt.figure(1,figsize=(3.3,2.475))
#
#    t_max = 1e32
#
#    for ii,path in enumerate(paths):
##        if ii == 0:
##            msd_owh = f.read_files([path+"box"+str(i)+"/msd_p.txt" for i in rng], np.loadtxt)
##            msd_ow  = f.read_files([path+"box"+str(i)+"/msd_w.txt" for i in rng], np.loadtxt)
##        elif ii ==1:
##            msd_owh = f.read_files([path+"box"+str(i)+"/run2/msd_p.txt" for i in rng], np.loadtxt)
##            msd_ow  = f.read_files([path+"box"+str(i)+"/run2/msd_w.txt" for i in rng], np.loadtxt)
#        msd_owh = f.read_files([path+"box"+str(i)+"/run2/msd_p.txt" for i in rng], np.loadtxt)
#        msd_ow  = f.read_files([path+"box"+str(i)+"/run2/msd_w.txt" for i in rng], np.loadtxt)
#        lenowh = min([len(msd_owh[i-1]) for i in rng])
#        lenow = min([len(msd_ow[i-1]) for i in rng])
#        for i in rng:
#            msd_ow[i-1] = msd_ow[i-1][:lenow]
#            msd_owh[i-1] = msd_owh[i-1][:lenowh]
#
#        tmp_ow = np.mean(msd_ow, axis=0)
#        tmp_owh = np.mean(msd_owh, axis=0)
#
#
#        tmp_ow  = np.array([(tmp_ow.T[0]-tmp_ow.T[0][0])*CGt_conv,(tmp_ow.T[1]-tmp_ow.T[1][0])*CGr_conv*Nwat]).T
#        tmp_owh = np.array([(tmp_owh.T[0]-tmp_owh.T[0][0])*CGt_conv,(tmp_owh.T[1]-tmp_owh.T[1][0])*CGr_conv]).T
#
#        D_owh,tmp_owh = np.array(imsd(np.array(tmp_owh).T[0],np.array(tmp_owh).T[1],blavg,"msd_p"))
#        D_ow, tmp_ow  = np.array(imsd(np.array(tmp_ow).T[0],np.array(tmp_ow).T[1],blavg,"msd_w"))
#
#        if tmp_ow[0][-1] < t_max:
#            t_max = tmp_ow[0][-1]
#            msd_max = tmp_owh[1][-1]
#
#        print t_max
#
#        plt.plot(np.array(tmp_owh[0]),np.array(tmp_owh[1]),"r"+style[ii],label=fnames[ii]+" P",linewidth=lnwd)
#        plt.plot(np.array(tmp_ow[0]),np.array(tmp_ow[1]),"k"+style[ii],label=fnames[ii]+" W",linewidth=lnwd)
#
#        ######### Save Data ##########
#        fname = path.split("/")[-2]+".txt"
#        print fname
#        with open(fname,"w") as ff:
#            tmp = np.array([tmp_owh[0], tmp_ow[1], tmp_owh[1]]).T
#            ff.write("# tau MSD_W MSD_P\n")
#            for line in tmp:
#                ff.write(" ".join(str(x) for x in line)+"\n")
#
#    plt.ylabel('MSD / $\mathregular{R_C^2}$',fontproperties=f1)
#    plt.xlabel('$\\tau$',fontproperties=f1)
#    plt.xlim(0,1e5)
#    plt.ylim(0,1e5)
#    plt.xticks(fontproperties=f2)
#    plt.yticks(fontproperties=f2)
#    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
#    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
#    plt.legend(loc='best',prop=f2)
#    plt.tight_layout(pad=0.)
#    plt.savefig('MSD.png',dpi=300)
#    plt.close()
#
#

#def plotconductivity(S,W,Rc):
#
#    # Convert to conductivity
#    print "Warning! Ensure the correct ion concentrations are defined in script for DPD simulations"
#    Nprotons = np.array([[210, 210, 210, 210],[420, 420, 420, 420],[630,630,630,630],[840,840,840,840]])
#    Volume   = np.array([[62.74**3, 64.44**3, 66.05**3, 71.83**3],[66.05**3, 69.06**3, 71.83**3, 81.19**3],[69.06**3, 73.13**3, 76.79**3, 88.78**3],[71.83**3, 76.79**3, 81.19**3, 95.27**3]])*1e-24 # cm^3 from Angstroms^3 from rdf output
#    C = Nprotons/Volume # mol/cm^3
#
#    F = 96485.33289 # C mol-1
#    R = 8.3144598 # J/(mol*K)
#    T = 298 # K
#    Na = 6.022140857e+23
#    conv = F**2/(R*T)/Na*1000 # mS from C^2/(J*mol)
#
#    kB = 1.38064852e-23   # J/K
#    e  = 1.6021766208e-19 # C
#    conv2 = e**2/(T*kB)*1000 # C^2/J
#
#    # Test from Lee
##    tmp_C = 8616./(187**3*1e-24)
##    print tmp_C, 1.1e-5*tmp_C*conv
#
#    # Simulation Data
#    c = "brgc"
#    data = np.genfromtxt("msd_output_error.csv",delimiter=",").T
#    msd = []; msd_err = []
#    for i in S:
#        msd.append([0 for x in W])
#        msd_err.append([0 for x in W])
#    for i,s in enumerate(S):
#        for j,w in enumerate(W):
#            ind_s = [ii for ii, x in enumerate(data[0]) if x == s]
#            ind_w = [ii for ii, x in enumerate(data[1]) if x == w]
#            ind = list(set(ind_s).intersection(ind_w))[0]
#            msd[i][j]     = data[4][ind]*C[i][j]*conv2
# #           print C[i][j], msd[i][j],data[4][ind]
#            msd_err[i][j] = data[5][ind]*C[i][j]*conv2
#    
