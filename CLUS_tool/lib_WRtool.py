#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import sys
import pickle
import os

from matplotlib import pyplot as plt
from numpy import linalg as LA
import netCDF4 as nc
from netCDF4 import Dataset, num2date
import cartopy.crs as ccrs
import warnings

#warnings.filterwarnings("ignore")

########
# Miscellaneous functions

def std_outname(inputs):
    name_outputs = inputs['varname'] + '{:.0f}'.format(inputs['level']) + '_' + inputs['freq'] + '_' + inputs['model_name'] + '_' + inputs['resolution'] + '_{}ens_'.format(inputs['numens']) + inputs['season'] + '_' + inputs['area'] + '_{}-{}_{}pcs'.format(inputs['start_year'], inputs['end_year'],inputs['numpcs'])

    return name_outputs

def printsep(ofile = None):
    if ofile is None:
        print('\n--------------------------------------------------------\n')
    else:
        ofile.write('\n--------------------------------------------------------\n')
    return

def str_to_bool(s):
    if s == 'True' or s == 'T':
         return True
    elif s == 'False' or s == 'F':
         return False
    else:
         raise ValueError('Not a boolean value')

def read_inputs(nomefile, key_strings, n_lines = None, itype = None, defaults = None, verbose=False):
    """
    Standard reading for input files. Searches for the keys in the input file and assigns the value to variable.
    :param keys: List of strings to be searched in the input file.
    :param defaults: Dict with default values for the variables.
    :param n_lines: Dict. Number of lines to be read after key.
    """

    keys = ['['+key+']' for key in key_strings]

    if n_lines is None:
        n_lines = np.ones(len(keys))
        n_lines = dict(zip(key_strings,n_lines))
    elif len(n_lines.keys()) < len(key_strings):
        for key in key_strings:
            if not key in n_lines.keys():
                n_lines[key] = 1

    if itype is None:
        itype = len(keys)*[None]
        itype = dict(zip(key_strings,itype))
    elif len(itype.keys()) < len(key_strings):
        for key in key_strings:
            if not key in itype.keys():
                warnings.warn('No type set for {}. Setting string as default..'.format(key))
                itype[key] = str
        warnings.warn('Not all input types (str, int, float, ...) have been specified')

    if defaults is None:
        warnings.warn('No defaults are set. Setting None as default value.')
        defaults = len(key_strings)*[None]
        defaults = dict(zip(key_strings,defaults))
    elif len(defaults.keys()) < len(key_strings):
        for key in key_strings:
            if not key in defaults.keys():
                defaults[key] = None

    variables = []
    is_defaults = []
    with open(nomefile, 'r') as infile:
        lines = infile.readlines()
        # Skips commented lines:
        lines = [line for line in lines if not line.lstrip()[:1] == '#']

        for key, keystr in zip(keys, key_strings):
            deflt = defaults[keystr]
            nli = n_lines[keystr]
            typ = itype[keystr]
            is_key = np.array([key in line for line in lines])
            if np.sum(is_key) == 0:
                print('Key {} not found, setting default value {}\n'.format(key,deflt))
                variables.append(deflt)
                is_defaults.append(True)
            elif np.sum(is_key) > 1:
                raise KeyError('Key {} appears {} times, should appear only once.'.format(key,np.sum(is_key)))
            else:
                num_0 = np.argwhere(is_key)[0][0]
                if typ == list:
                    cose = lines[num_0+1].split(',')
                    coseok = [cos.strip() for cos in cose]
                    variables.append(cose)
                elif nli == 1:
                    cose = lines[num_0+1].split()
                    if typ == bool: cose = [str_to_bool(lines[num_0+1].split()[0])]
                    if typ == str: cose = [lines[num_0+1].rstrip()]
                    if len(cose) == 1:
                        variables.append(map(typ,cose)[0])
                    else:
                        variables.append(map(typ,cose))
                else:
                    cose = []
                    for li in range(nli):
                        cos = lines[num_0+1+li].split()
                        if typ == str: cos = [lines[num_0+1+li].rstrip()]
                        if len(cos) == 1:
                            cose.append(map(typ,cos)[0])
                        else:
                            cose.append(map(typ,cos))
                    variables.append(cose)
                is_defaults.append(False)

    if verbose:
        for key, var, deflt in zip(keys,variables,is_defaults):
            print('----------------------------------------------\n')
            if deflt:
                print('Key: {} ---> Default Value: {}\n'.format(key,var))
            else:
                print('Key: {} ---> Value Read: {}\n'.format(key,var))

    return dict(zip(key_strings,variables))


#*********************************
#            precompute          *
#*********************************

def precompute(inputs):
    """
    It prepares data for the analysis:
    It reads the netCDF file of (lat, lon, lev, variable) for one or more ensemble members.
    It can select one level, a season, it can filter by running mean and compute anomalies by removing the time-mean for each time (e.g. day), then it can select an area.
    It saves the resulting netCDF field of anomalies (lat, lon, anomalies).
    """
    from readsavencfield import read4Dncfield, save3Dncfield
    from manipulate import sel_season, anomalies, sel_area

    print('**********************************************************\n')
    print('Running PRECOMPUTATION\n')
    print('**********************************************************\n')

    name_outputs  = inputs['name_outputs']  # name of the outputs
    varname       = inputs['varname']  # variable name
    level         = inputs['level']  # level to select
    wnd           = inputs['wnd']  # running mean filter time window
    numens        = inputs['numens']  # total number of ensemble members
    season        = inputs['season']  # season
    area          = inputs['area']  # area
    filenames     = inputs['filenames']  # file names (absolute paths)

    # OUTPUT DIRECTORY
    OUTPUTdir = inputs['OUTPUTdir']

    #____________Reading the netCDF file of 4Dfield, for one or more ensemble members
    var_ensList=[]

    for ens in range(numens):
        ifile=filenames[ens]
        if numens>1:
            print('\n/////////////\nENSEMBLE MEMBER %s' %ens)
        elif numens==1:
            print('\n//////////////////')

        var, level, lat, lon, dates, time_units, var_units, time_cal = read4Dncfield(ifile,lvel=level)

        # Select data for that season and average data for that season after a running mean of window=wnd
        var_season,dates_season,var_filt = sel_season(var,dates,season,wnd)

        # Compute anomalies by removing the time-mean for each time (day, month,...)
        var_anom = anomalies(var_filt,var_season, dates_season,season)#,freq)

        namefile='glob_anomaly_{0}_{1}.nc'.format(name_outputs,ens)
        ofile=OUTPUTdir+namefile
        save3Dncfield(lat,lon,var_anom,varname,var_units,dates_season,time_units,time_cal,ofile)

        # Selecting only [latS-latN, lonW-lonE] box region defineds by area
        var_area, lat_area, lon_area = sel_area(lat,lon,var_anom,area)

        namefile='area_anomaly_{0}_{1}.nc'.format(name_outputs,ens)
        ofile=OUTPUTdir+namefile
        save3Dncfield(lat_area,lon_area,var_area,varname,var_units,dates_season,time_units,time_cal,ofile)

        var_ensList.append(var_area)

    print('\nINPUT DATA:')
    print('Variable is {0}{1} ({2})'.format(varname,level,var_units))
    print('Number of ensemble members is {0}'.format(numens))
    print('Input data are read and anomaly fields for each ensemble member are computed and saved in {0}'.format(OUTPUTdir))

    if enstoselect is not None:
        print('Ensemble member number {0} is selected for the analysis'.format(enstoselect))
        var_ens=[]
        var_ens.append(var_ensList[enstoselect])
        print(len(var_ens),var_ens[0].shape)
        del var_ensList
        var_ensList=var_ens
        name_outputs=name_outputs+'_'+str(enstoselect)
        print('New name for the outputs: '+name_outputs)
        inputs['name_outputs'] = name_outputs
        inputs['numens'] = 1
    else:
        print('All the ensemble members are concatenated one after the other prior to the analysis')

    print('\n**********************************************************\n')
    print('END PRECOMPUTATION\n')
    print('************************************************************\n')

    out_precompute = [var_ensList, lat_area, lon_area, dates_season, time_units, var_units]

    return inputs, out_precompute


def read_out_precompute(OUTPUTdir, name_outputs, numens):
    """
    Reads the output of precompute from the files.
    """
    # READ from files
    anomaly_filenames='area_anomaly_'+name_outputs
    fn = [i for i in os.listdir(OUTPUTdir) if os.path.isfile(os.path.join(OUTPUTdir,i)) and anomaly_filenames in i]
    fn.sort()
    print('\nInput {0} anomaly fields netCDF files (computed by precompute.py):'.format(numens))
    print('\n'.join(fn))
    var_ensList=[]
    for ens in range(numens):
        ifile=OUTPUTdir+fn[ens]
        var_area, lat_area, lon_area, dates, time_units, var_units= read3Dncfield(ifile)
        var_ensList.append(var_area)

    out_precompute = [var_ensList, lat_area, lon_area, dates, time_units, var_units]

    return out_precompute


def compute(inputs, out_precompute = None):
    """
    It performs the EOF analysis.
    It loads the netCDF field of anomalies saved by precompute.py, if enstoselect is set to a value, it selects only that member for the analysis, otherwise all ensemble members are concatenated one after the other prior to the analysis.
    It computes empirical orthogonal functions EOFs and principal components PCs of the loaded field of anomalies weighted by the square root of the cosine of latitude.
    It plots the first EOF (multiplied by the square-root of their eigenvalues) and the first PC (divided by the square-root of their eigenvalues) and it saves the first numpcs EOFs and PCs (scaled and unscaled), jointly with the the fractional variance represented by each EOF mode.

    unscaled EOF are normalized to 1. scaled2 EOF have norm: np.sqrt(eigenvalue).

    The output of precompute has to be given as input. In case you want compute to read it from files, just give None as out_precompute.
    """

    # User-defined libraries
    from readsavencfield import read3Dncfield, read2Dncfield, save2Dncfield, save_N_2Dfields
    from EOFtool import eof_computation

    print('******************************************\n')
    print('Running COMPUTE\n'))
    print('******************************************\n')

    OUTPUTdir     = inputs['OUTPUTdir']
    name_outputs  = inputs['name_outputs']  # name of the outputs
    numens        = inputs['numens']  # total number of ensemble members
    numpcs        = inputs['numpcs']  # number of principal components
    perc          = inputs['perc']  # percentage of explained variance for EOF analysis
    enstoselect   = inputs['enstoselect']  # ensemble member to analyse

    print('\nINPUT DATA:\n')
    print('number of principal components: {0}\n'.format(numpcs))
    print('percentage of explained variance: {0}%'.format(perc))
    if (perc is None and numpcs is None) or (perc is not None and numpcs is not None):
        raise ValueError('You have to specify either "perc" or "numpcs".')
    print('ensemble member to analyse: {0}'.format(enstoselect))

    # LOAD DATA PROCESSED BY precompute.py
    #______________________________________

    if out_precompute is None:
        warnings.warn('Reading the output of precompute from files')
        out_precompute = read_out_precompute(OUTPUTdir, name_outputs, numens)

    var_ensList, lat_area, lon_area, dates, time_units, var_units = out_precompute

    syr=pd.to_datetime(dates).year[0]
    eyr=pd.to_datetime(dates).year[-1]

    print('\nINPUT DATA:')
    print('Data cover the time period from {0} to {1}'.format(syr,eyr))
    print('Data dimensions for one ensemble member is {0}'.format(var_ensList[0].shape))
    print('Number of ensemble members is {0}'.format(len(var_ensList)))

    name_outputs=name_outputs+'_{0}pcs'.format(numpcs)

    # COMPUTE AND SAVE EOFs (empirical orthogonal functions)
    # AND PCs (principal component time series)
    #_____________________________________________________________
    solver, pcs_scal1, eofs_scal2, pcs_unscal0, eofs_unscal0, varfrac = eof_computation(var_ensList,var_units,lat_area,lon_area)

    n_eofs = 25
    ind_pcs = [0,25]

    eigenval = solver.eigenvalues()
    print(len(eigenval))
    pcs = solver.pcs()
    print(np.shape(pcs))

    fig1 = plt.figure()
    #nor = np.sum(eigenval)
    plt.grid()
    plt.bar(range(0,len(eigenval)), eigenval)
    plt.xlim(-1,n_eofs)
    plt.xlabel('EOF')
    #plt.ylabel(r'$\lambda_i \div \sum_j \lambda_j$')
    plt.ylabel('Eigenvalue ({})'.format(var_units))
    plt.title('Eigenvalues of EOFs')
    plt.tight_layout()
    plt.savefig(OUTPUTdir+'eigenvalues_histoplot.pdf', format = 'pdf')

    tito = 'Showing only the pcs from {} to {}, with {} eofs..'

    fig2 = plt.figure()
    plt.xlim(-0.5,n_eofs+0.5)
    plt.ylim(ind_pcs[0]-0.5,ind_pcs[1]+0.5)
    print(tito.format(ind_pcs[0],ind_pcs[1],n_eofs))
    plt.imshow(pcs)
    gigi = plt.colorbar()
    gigi.set_label('PC value (m)')
    plt.xlabel('EOF')
    plt.ylabel('Time (seq. number)')
    plt.title('PCs')
    plt.savefig(OUTPUTdir+'PCs_colorplot.pdf', format = 'pdf')


    acc=np.cumsum(varfrac*100)
    if perc is not None:
        # Find how many PCs explain a certain percentage of variance
        # (find the mode relative to the percentage closest to perc, but bigger than perc)
        numpcs=min(enumerate(acc), key=lambda x: x[1]<=perc)[0]+1
        print('\nThe number of PCs that explain the percentage closest to {0}% of variance (but grater than {0}%) is {1}'.format(perc,numpcs))
        exctperc=min(enumerate(acc), key=lambda x: x[1]<=perc)[1]
    if numpcs is not None:
        exctperc=acc[numpcs-1]
    print('(the first {0} PCs explain the {:5.2f}% of variance)'.format(numpcs, exctperc))


    # save python object solver
    namef='{0}solver_{1}.p'.format(OUTPUTdir,name_outputs)
    pickle.dump(solver, open(namef, 'wb'), protocol=2)

    # save EOF unscaled
    varsave='EOFunscal'
    ofile='{0}EOFunscal_{1}.nc'.format(OUTPUTdir,name_outputs)
    save_N_2Dfields(lat_area,lon_area,eofs_unscal0[:numpcs],varsave,'m',ofile)

    print('The {0} EOF unscaled are saved as\n{1}'.format(numpcs,ofile))
    print('____________________________________________________________\n')

    # save EOF scaled: EOFs are multiplied by the square-root of their eigenvalues
    varsave='EOFscal2'
    ofile='{0}EOFscal2_{1}.nc'.format(OUTPUTdir,name_outputs)
    save_N_2Dfields(lat_area,lon_area,eofs_scal2[:numpcs],varsave,'m',ofile)

    print('The {0} EOF scaled2 (multiplied by the square-root of their eigenvalues)are saved as\n{1}'.format(numpcs,ofile))
    print('__________________________________________________________\n')

    # save PC unscaled
    # [time x PCi]
    namef='{0}PCunscal_{1}.txt'.format(OUTtxt,name_outputs)
    np.savetxt(namef, pcs_unscal0[:,:numpcs])

    #save PC scaled: PCs are scaled to unit variance (divided by the square-root of their eigenvalue)
    # [time x PCi]
    namef='{0}PCscal1_{1}.txt'.format(OUTtxt,name_outputs)
    np.savetxt(namef, pcs_scal1[:,:numpcs])
    print('The {0} PCs in columns are saved in\n{1}'.format(numpcs,OUTtxt))
    print('__________________________________________________________\n')

    # save varfrac: the fractional variance represented by each EOF mode
    namef='{0}varfrac_{1}.txt'.format(OUTtxt,name_outputs)
    np.savetxt(namef, varfrac, fmt='%1.10f')


    print('\n*******************************************************\n')
    print('END COMPUTE')
    print('*********************************************************\n')

    return solver


def read_out_compute(OUTPUTdir, name_outputs):
    """
    Reads the output of compute from the files.
    """

    # load solver
    namef='{0}solver_{1}.p'.format(OUTPUTdir,name_outputs)
    solver = pickle.load(open(namef, 'wb'), protocol=2)

    # name_savedvar='{0}PCunscal_{1}.txt'.format(OUTPUTdir,name_outputs)
    # PCunscal=np.loadtxt(name_savedvar)
    # print('(times, numPCs) ={0}'.format(PCunscal.shape))

    return solver


def clustering(inputs, solver = None, out_precompute = None):
    """
    Cluster analysis.
    It seeks numclus clusters via k-means algorithm for a subset of numpcs unscaled PCs.
    It saves the cluster index for each realization, the cluster centroids coordinates and the optimal variance ratio, which is the ratio between the variance among the centroid clusters and the intra-cluster variance.
    It sorts the clusters by decreasing frequency of occurrence.
    It plots and saves the cluster spatial patterns (the average of the original field of anomalies during times belonging to that cluster).

    :: solver :: is the output of compute and contains the results of the EOF computation. If None, the program looks for a saved file with standard name.
    :: out_precompute :: is the output of precompute. If None, the program looks for saved files with standard names.
    """

    import ctool
    from clus_manipulate import cluster_orderingFREQ, compute_clusterpatterns
    from readsavencfield import read3Dncfield, save_N_2Dfields

    print('***********************************************************\n')
    print('Running CLUSTERING')
    print('***********************************************************\n')

    name_outputs  = inputs['name_outputs']  # name of the outputs
    numens        = inputs['numens']  # total number of ensemble members
    numpcs        = inputs['numpcs']  # number of principal components
    perc          = inputs['perc']  # percentage of explained variance for EOF analysis
    enstoselect   = inputs['enstoselect']  # ensemble member to analyse
    numclus       = inputs['numclus']  # number of clusters

    # OUTPUT DIRECTORY
    OUTPUTdir = inputs['OUTPUTdir']

    # LOAD DATA PROCESSED BY precompute.py
    if out_precompute is None:
        warnings.warn('Reading the output of precompute from files')
        out_precompute = read_out_precompute(OUTPUTdir, name_outputs, numens)

    syr=pd.to_datetime(dates).year[0]
    eyr=pd.to_datetime(dates).year[-1]

    name_outputs=name_outputs+'_{0}pcs'.format(numpcs)

    # LOAD DATA PROCESSED BY compute.py
    if solver is None:
        solver = read_out_compute(OUTPUTdir, name_outputs)
    PCunscal = solver.pcs()
    pc=np.transpose(PCunscal)

    # k-means analysis using the subset of PCs
    #______________________________________
    printsep()
    print('number of clusters: {0}\n'.format(numclus))
    printsep()

    npart=100      #100
    varopt=[]
    indcl0=[]
    centr=[]
    print('npart ={0}'.format(npart))

    # This calculates the clustering for k=2 to 7.. useful but may insert an option to do this..
    for ncl in range(2,7):
        nfcl,indcl1,centr1,varopt1,iseed=ctool.cluster_toolkit.clus_opt(ncl,npart,pc)
        indcl0.append(np.subtract(indcl1,1))        #indcl1 starts from 1, indcl0 starts from 0
        centr.append(centr1)
        varopt.append(varopt1)
    indcl=indcl0[numclus-2]
    centr=centr[numclus-2]

    # save cluster index
    namef='{0}indcl_{1}clus_{2}.txt'.format(OUTPUTdir,numclus,name_outputs)
    np.savetxt(namef,indcl,fmt='%d')

    # save cluster centroids
    namef='{0}centr_{1}clus_{2}.txt'.format(OUTPUTdir,numclus,name_outputs)
    np.savetxt(namef,centr)

    # save cluster optimal variance ratio (this is needed for significance computation: clusters_sig.py)
    namef='{0}varopt_2to6clus_{2}.txt'.format(OUTPUTdir,numclus,name_outputs)
    np.savetxt(namef,varopt, fmt='%1.10f')


    # Cluster ordering in decreasing frequency
    #_______________________
    centrORD,indclORD=cluster_orderingFREQ(indcl,centr,numclus)

    # save cluster index
    namef='{0}indclORD_{1}clus_{2}.txt'.format(OUTPUTdir,numclus,name_outputs)
    np.savetxt(namef,indclORD,fmt='%d')

    # save cluster centroids
    namef='{0}centrORD_{1}clus_{2}.txt'.format(OUTPUTdir,numclus,name_outputs)
    np.savetxt(namef,centrORD)

    # COMPUTE AND SAVE CLUSTERS PATTERNS
    #_______________________
    # compute cluster patterns
    cluspattORD=compute_clusterpatterns(numclus,var_ensList,indclORD)
    # save cluster patterns
    varsave='cluspattern'
    ofile='{0}cluspatternORD_{1}clus_{2}.nc'.format(OUTPUTdir,numclus,name_outputs)
    save_N_2Dfields(lat_area,lon_area,np.array(cluspattORD),varsave,var_units,ofile)

    print('Cluster pattern netCDF variable (ordered by decreasing frequency of occurrence) is saved as\n{0}'.format(ofile))
    printsep()

    print('\n*******************************************************\n')
    print('END CLUSTERING')
    print('*********************************************************\n')

    return centrORD, indclORD, cluspattORD, varopt


def read_out_clustering(OUTPUTdir, name_outputs, numclus):
    """
    Reads the output of clustering from files.
    """
    # load cluster patterns
    ifile='{0}cluspatternORD_{1}clus_{2}.nc'.format(OUTPUTdir,numclus,name_outputs)
    cluspattORD,lat_area,lon_area =read_N_2Dfields(ifile)

    #load centrORD
    namef='{0}centrORD_{1}clus_{2}.txt'.format(OUTPUTdir,numclus,name_outputs)
    centrORD=np.loadtxt(namef)

    # load indclORD
    namef='{0}indclORD_{1}clus_{2}.txt'.format(OUTPUTdir,numclus,name_outputs)
    indclORD=np.loadtxt(namef)

    # Load varopt
    namef='{0}varopt_2to6clus_{2}.txt'.format(OUTPUTdir,numclus,name_outputs)
    varopt = np.loadtxt(namef)

    return centrORD, indclORD, cluspattORD, varopt


def clusters_comparison(inputs, ):
    """
    Finds the best possible match between two sets of cluster partitions.
    - It computes and saves the projections of the simulated cluster patterns and the reference cluster patterns onto the same reference EOF patterns, obtaining vectors in the PC space (numpcs dimensions).
    - It compares the two cluster partitions computing the RMS error, the phase error, the pattern correlation and the combination that minimizes the mean total RMS error, which can be used to order the simulated clusters as the reference ones.
    - It sorts the clusters as the reference ones.
    - Compares the RMS errors to those between the two reference (ERA vs NCEP) datasets.
    """

    from itertools import permutations
    from operator import itemgetter
    # User-defined libraries
    import ctool
    from readsavencfield import read_N_2Dfields, read3Dncfield, save_N_2Dfields
    from clus_manipulate import cluster_ordering_asREF, compute_pvectors, compute_pseudoPC, plot_clusters
    from pcmatch import *

    print('**********************************************************\n')
    print('Running CLUSTER COMPARISON')
    print('**********************************************************\n')

    name_outputs  = inputs['name_outputs']  # name of the outputs
    numens        = inputs['numens']  # total number of ensemble members
    numpcs        = inputs['numpcs']  # number of principal components
    perc          = inputs['perc']  # percentage of explained variance for EOF analysis
    enstoselect   = inputs['enstoselect']  # ensemble member to analyse
    numclus       = inputs['numclus']  # number of clusters

    OUTPUTdir = inputs['OUTPUTdir']
    nameREF       = sys.argv[9]
    print('REFERENCE NAME FILES: {0}'.format(nameREF))

    # LOAD DATA PROCESSED BY precompute
    if out_precompute is None:
        warnings.warn('Reading the output of precompute from files')
        out_precompute = read_out_precompute(OUTPUTdir, name_outputs, numens)

    syr=pd.to_datetime(dates).year[0]
    eyr=pd.to_datetime(dates).year[-1]

    name_outputs=name_outputs+'_{0}pcs'.format(numpcs)

    # LOAD DATA PROCESSED BY compute.py
    if solver is None:
        solver = read_out_compute(OUTPUTdir, name_outputs)
    PCunscal = solver.pcs()
    pc=np.transpose(PCunscal)

    # LOAD DATA PROCESSED BY clustering.py
    #______________________________________
    # N.B. suffix ORD refers to clusters ordered by decreasing frequency of occurrence.
    #      suffix ORDasREF refers to clusters ordered as te reference clusters (e.g. observartions)

    if out_clustering is None:
        out_clustering = read_out_clustering(OUTPUTdir, name_outputs)
    centrORD, indclORD, cluspattORD, varopt = out_clustering

    # load solverREF
    solverREF=pickle.load(open('{0}solver_{1}.p'.format(inputs,nameREF), 'rb'))























    # Compute p vector as in Dawson&Palmer 2015:
    #___________________________________________
    # The clusters are projected on the REFERENCE EOF base, given by solverREF. The RMS error is computed first in this same base.

    pvec=compute_pvectors(numpcs,solverREF,cluspattORD)
    print('pvec:\n{0}'.format(pvec))
    # save pvec
    namef='{0}pvec_{1}clus_{2}.txt'.format(OUTtxt,numclus,name_outputs)
    np.savetxt(namef, pvec)

    #load pREF
    namef='{0}pvec_{1}clus_{2}.txt'.format(OUTtxt,numclus,nameREF)
    pvecREF=np.loadtxt(namef)

    pseudoPC=compute_pseudoPC(var_ensList,numpcs,solverREF)
    print('pseudoPC:\n{0}'.format(pseudoPC))
    # save pseudoPC
    namef='{0}pseudoPC_{1}.txt'.format(OUTtxt,name_outputs)
    np.savetxt(namef,pseudoPC)


    # Find the match with the given reference clusters
    #_______________________
    pcset1 = pvec
    pcset2 = pvecREF
    #match_clusters_info(pcset1, pcset2,npcs=9)
    perm, et, ep, patcor = match_pc_sets(pcset2, pcset1)
    print('\n ----------------- \n')
    print 'RMS error = ', et
    print '(squared) Phase error =', ep
    print 'Pattern correlation (cosine of angle between) =', patcor
    print 'Optimal permutation =', perm
    print('\n ----------------- \n')

    # save et
    namef='{0}et_{1}clus_{2}.txt'.format(OUTtxt,numclus,name_outputs)
    np.savetxt(namef, et)

    # save ep
    namef='{0}ep_{1}clus_{2}.txt'.format(OUTtxt,numclus,name_outputs)
    np.savetxt(namef, ep)

    # save patcor
    namef='{0}patcor_{1}clus_{2}.txt'.format(OUTtxt,numclus,name_outputs)
    np.savetxt(namef, patcor)

    #=====================================================
    #Order as the reference clusters
    sor=list(perm[1])
    print 'Order the clusters as the reference:', sor
    #=====================================================










    # Cluster ordering as reference
    #_______________________
    centrORDasREF,indclORDasREF=cluster_ordering_asREF(indclORD,centrORD,numclus,sor)

    # save centrORDasREF
    namef='{0}centrORDasREF_{1}clus_{2}.txt'.format(OUTtxt,numclus,name_outputs)
    np.savetxt(namef,centrORDasREF)

    # save indclORDasREF
    namef='{0}indclORDasREF_{1}clus_{2}.txt'.format(OUTtxt,numclus,name_outputs)
    np.savetxt(namef,indclORDasREF, fmt='%d')

    # save freqORDasREF
    freq=[]
    for nclus in range(numclus):
        cl=list(np.where(indclORDasREF==nclus)[0])
        fr=len(cl)*100./len(indclORDasREF)
        freq.append(fr)
    freqORDasREF=np.array(freq)
    namef='{0}freq_ORDasREF_{1}clus_{2}.txt'.format(OUTtxt,numclus,name_outputs)
    np.savetxt(namef, freqORDasREF)

    # save cluspattORDasREF
    cluspattORDasREF=cluspattORD[sor]
    varsave='cluspattern'
    ofile='{0}cluspatternORDasREF_{1}clus_{2}.nc'.format(OUTnc,numclus,name_outputs)
    save_N_2Dfields(lat_area,lon_area,np.array(cluspattORDasREF),varsave,var_units,ofile)

    #if enstoselect!='no':
    #    tit='{0} {1} {2}{3} {4} {5} {6}'.format(varname,model,kind,enstoselect,res,season,area)
    #else:
    #    tit='{0} {1} {2} {3} {4} {5}'.format(varname,model,kind,res,season,area)
    #
    #ax=plot_clusters(lon_area,lat_area,cluspattORDasREF,freqORDasREF,area,numclus,obs,tit,patcor)
    #
    #namef='{0}clus_patternsORD_{1}clus_{2}.eps'.format(OUTfig,numclus,name_outputs)
    #ax.figure.savefig(namef)#bbox_inches='tight')
    #print('Clusters eps figure for {0} weather regimes is saved as\n{1}'.format(area,namef))
    #print('____________________________________________________________________________________________________________________')

    # Compute inter-clusters and intra-clusters statistics
    #if area=='EAT':
    #    clusnames=['NAO+','Blocking','Atlantic Ridge','NAO-']
    #else:

    # COMPUTE AND SAVE STATISTICS OF CLUSTER PARTITION
    #_______________________
    clusnames=['clus{0}'.format(nclus) for nclus in range(numclus)]
    print('--------------------------DIMENSIONS:------------------------------------')
    print('ANOMALY FIELD:                  [time, lat, lon]         = {0}'.format(np.concatenate(var_ensList[:]).shape))
    print('EMPIRICAL ORTHOGONAL FUNCTIONS: [num_eof, lat, lon]      = ({0}, {1}, {2})'.format(PCunscal.shape[1],len(lat_area),len(lon_area)))
    print('PRINCIPAL COMPONENTS:           [time, num_eof]          = {0}'.format(PCunscal.shape))
    print('CENTROID COORDNATES:            [num_eof, num_clusters]  = {0}'.format(centrORDasREF.shape))
    print('CLUSTER PATTERNS:               [num_clusters, lat, lon] = {0}'.format(cluspattORDasREF.shape))


    # 1) compute centroid-centroid distance in the phase space (PC space):
    print('\n==============================================================')
    print('In the PC space, here are the distances among centroids:')
    norm_centr=[]
    # permutations 2 by 2
    kperm=2
    perm=list(permutations(list(range(numclus)),kperm))
    print('{0} permutations {1} by {1}'.format(len(perm),kperm))

    i=map(itemgetter(0), perm)
    print(i)
    j=map(itemgetter(1), perm)
    print(j)

    norm_centr=[]
    for numperm in range(len(perm)):
        normcentroids=LA.norm(centrORDasREF[:,i[numperm]]-centrORDasREF[:,j[numperm]])
        norm_centr.append(normcentroids)
    print(len(norm_centr))
    print(norm_centr)

    print('\nINTER-CLUSTER DISTANCES MEAN IS {0}'.format(np.mean(norm_centr)))
    print('INTER-CLUSTER DISTANCES STANDARD DEVIATION {0}'.format(np.std(norm_centr)))
    print('==============================================================')



    # 2) compute centroid-datapoint distance in the phase space (PC space):
    print('\n==============================================================')
    print('In the PC space, here are the distances between each data point in a cluster and its centroid:')
    numdatapoints=PCunscal.shape[0]

    for nclus in range(numclus):
        print('\n******************            {0}           ***************************'.format(clusnames[nclus]))
        cl=list(np.where(indclORDasREF==nclus)[0])
        print('{0} DAYS IN THIS CLUSTER'.format(len(cl)))
        PCsintheclus=[PCunscal[i,:] for i in cl]
        #print('standard deviation of PCs relative to days belonging to this cluster={0}'.format(np.std(PCsintheclus)))

        norm=[] # [numdatapoints]
        for point in cl:
            normpoints=LA.norm(centrORDasREF[:,nclus]-PCunscal[point,:])
            norm.append(normpoints)

        print(len(norm))
        print('MAXIMUM DISTANCE FOR CLUS {0} IS {1}'.format(nclus,max(norm)))
        txt='Furthest data point to centroid of cluster {0} is day {1}'.format(nclus,list(np.where(norm == max(norm))[0]))
        print(txt)
        print('INTRA-CLUSTER DISTANCES MEAN FOR CLUS {0} IS {1}'.format(nclus,np.mean(norm)))
        print('INTRA-CLUSTER DISTANCES STANDARD DEVIATION FOR CLUS {0} IS {1}'.format(nclus,np.std(norm)))
    print('==============================================================')


    print('\n******************************************************************************')
    print('END {0}'.format(sys.argv[0]))
    print('*********************************************************************************')
