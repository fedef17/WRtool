#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import sys
import pickle
import os
import datetime

from matplotlib import pyplot as plt
from numpy import linalg as LA
import netCDF4 as nc
from netCDF4 import Dataset, num2date
import cartopy.crs as ccrs
import warnings

#warnings.filterwarnings("ignore")

########
# Miscellaneous functions

def std_outname(inputs, with_pcs = False):
    if with_pcs:
        name_outputs = inputs['varname'] + '{:.0f}'.format(inputs['level']) + '_' + inputs['freq'] + '_' + inputs['model_name'] + '_' + inputs['resolution'] + '_{}ens_'.format(inputs['numens']) + inputs['season'] + '_' + inputs['area'] + '_{}-{}_{}pcs'.format(inputs['start_year'], inputs['end_year'],inputs['numpcs'])
    else:
        name_outputs = inputs['varname'] + '{:.0f}'.format(inputs['level']) + '_' + inputs['freq'] + '_' + inputs['model_name'] + '_' + inputs['resolution'] + '_{}ens_'.format(inputs['numens']) + inputs['season'] + '_' + inputs['area'] + '_{}-{}'.format(inputs['start_year'], inputs['end_year'])

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
                try:
                    if typ == list:
                        cose = lines[num_0+1].rstrip().split(',')
                        coseok = [cos.strip() for cos in cose]
                        variables.append(coseok)
                    elif nli == 1:
                        cose = lines[num_0+1].rstrip().split()
                        #print(key, cose)
                        if typ == bool: cose = [str_to_bool(lines[num_0+1].rstrip().split()[0])]
                        if typ == str: cose = [lines[num_0+1].rstrip()]
                        if len(cose) == 1:
                            if cose[0] is None:
                                variables.append(None)
                            else:
                                variables.append(map(typ,cose)[0])
                        else:
                            variables.append(map(typ,cose))
                    else:
                        cose = []
                        for li in range(nli):
                            cos = lines[num_0+1+li].rstrip().split()
                            if typ == str: cos = [lines[num_0+1+li].rstrip()]
                            if len(cos) == 1:
                                if cos[0] is None:
                                    cose.append(None)
                                else:
                                    cose.append(map(typ,cos)[0])
                            else:
                                cose.append(map(typ,cos))
                        variables.append(cose)
                    is_defaults.append(False)
                except Exception as problemha:
                    print('Unable to read value of key {}'.format(key))
                    raise problemha

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
    enstoselect   = inputs['enstoselect']

    # OUTPUT DIRECTORY
    OUTPUTdir = inputs['OUTPUTdir']

    #____________Reading the netCDF file of 4Dfield, for one or more ensemble members
    var_ensList=[]
    var_ensList_glob=[]

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

        namefile='glob_anomaly_{}_{}.nc'.format(name_outputs,ens)
        ofile=OUTPUTdir+namefile
        save3Dncfield(lat,lon,var_anom,varname,var_units,dates_season,time_units,time_cal,ofile)
        var_ensList_glob.append(var_anom)

        # Selecting only [latS-latN, lonW-lonE] box region defineds by area
        var_area, lat_area, lon_area = sel_area(lat,lon,var_anom,area)

        namefile='area_anomaly_{}_{}.nc'.format(name_outputs,ens)
        ofile=OUTPUTdir+namefile
        save3Dncfield(lat_area,lon_area,var_area,varname,var_units,dates_season,time_units,time_cal,ofile)

        var_ensList.append(var_area)

    print('\nINPUT DATA:')
    print('Variable is {}{} ({})'.format(varname,level,var_units))
    print('Number of ensemble members is {}'.format(numens))
    print('Input data are read and anomaly fields for each ensemble member are computed and saved in {}'.format(OUTPUTdir))

    if enstoselect is not None:
        print('Ensemble member number {} is selected for the analysis'.format(enstoselect))
        var_ensList = [var_ensList[enstoselect]]
        var_ensList_glob = [var_ensList_glob[enstoselect]]
        name_outputs=name_outputs+'_'+str(enstoselect)
        print('New name for the outputs: '+name_outputs)
        inputs['name_outputs'] = name_outputs
        inputs['numens'] = 1
    else:
        print('All the ensemble members are concatenated one after the other prior to the analysis')

    print('\n**********************************************************\n')
    print('END PRECOMPUTATION\n')
    print('************************************************************\n')

    out_precompute = [var_ensList_glob, lat, lon, var_ensList, lat_area, lon_area, dates_season, time_units, var_units]

    return inputs, out_precompute


def read_out_precompute(OUTPUTdir, name_outputs, numens):
    """
    Reads the output of precompute from the files.
    """
    # READ from files
    anomaly_filenames='area_anomaly_'+name_outputs
    fn = [i for i in os.listdir(OUTPUTdir) if os.path.isfile(os.path.join(OUTPUTdir,i)) and anomaly_filenames in i]
    fn.sort()
    print('\nInput {} anomaly fields netCDF files (computed by precompute.py):'.format(numens))
    print('\n'.join(fn))
    var_ensList_area=[]
    for ens in range(numens):
        ifile=OUTPUTdir+fn[ens]
        var_area, lat_area, lon_area, dates, time_units, var_units= read3Dncfield(ifile)
        var_ensList_area.append(var_area)

    anomaly_filenames='glob_anomaly_'+name_outputs
    fn = [i for i in os.listdir(OUTPUTdir) if os.path.isfile(os.path.join(OUTPUTdir,i)) and anomaly_filenames in i]
    fn.sort()
    print('\nInput {} anomaly fields netCDF files (computed by precompute.py):'.format(numens))
    print('\n'.join(fn))
    var_ensList_glob=[]
    for ens in range(numens):
        ifile=OUTPUTdir+fn[ens]
        var_glob, lat, lon, dates, time_units, var_units= read3Dncfield(ifile)
        var_ensList_glob.append(var_glob)

    out_precompute = [var_ensList_glob, lat, lon, var_ensList_area, lat_area, lon_area, dates, time_units, var_units]

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
    print('Running COMPUTE\n')
    print('******************************************\n')

    OUTPUTdir     = inputs['OUTPUTdir']
    name_outputs  = inputs['name_outputs']  # name of the outputs
    numens        = inputs['numens']  # total number of ensemble members
    numpcs        = inputs['numpcs']  # number of principal components
    perc          = inputs['perc']  # percentage of explained variance for EOF analysis
    enstoselect   = inputs['enstoselect']  # ensemble member to analyse

    print('\nINPUT DATA:\n')
    print('number of principal components: {}\n'.format(numpcs))
    print('percentage of explained variance: {}%'.format(perc))
    if (perc is None and numpcs is None) or (perc is not None and numpcs is not None):
        raise ValueError('You have to specify either "perc" or "numpcs".')
    print('ensemble member to analyse: {}'.format(enstoselect))

    # LOAD DATA PROCESSED BY precompute.py
    #______________________________________

    if out_precompute is None:
        warnings.warn('Reading the output of precompute from files')
        out_precompute = read_out_precompute(OUTPUTdir, name_outputs, numens)
    var_ensList_glob, lat, lon, var_ensList, lat_area, lon_area, dates, time_units, var_units = out_precompute

    syr=pd.to_datetime(dates).year[0]
    eyr=pd.to_datetime(dates).year[-1]

    print('\nINPUT DATA:')
    print('Data cover the time period from {} to {}'.format(syr,eyr))
    print('Data dimensions for one ensemble member is {}'.format(var_ensList[0].shape))
    print('Number of ensemble members is {}'.format(len(var_ensList)))

    name_outputs_pcs=name_outputs+'_{}pcs'.format(numpcs)

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
        print('\nThe number of PCs that explain the percentage closest to {}% of variance (but grater than {}%) is {}'.format(perc,numpcs))
        exctperc=min(enumerate(acc), key=lambda x: x[1]<=perc)[1]
    if numpcs is not None:
        exctperc=acc[numpcs-1]
    print('(the first {} PCs explain the {:5.2f}% of variance)'.format(numpcs, exctperc))


    # save python object solver
    namef='{}solver_{}.p'.format(OUTPUTdir,name_outputs_pcs)
    pickle.dump(solver, open(namef, 'wb'), protocol=2)

    # save EOF unscaled
    varsave='EOFunscal'
    ofile='{}EOFunscal_{}.nc'.format(OUTPUTdir,name_outputs_pcs)
    save_N_2Dfields(lat_area,lon_area,eofs_unscal0[:numpcs],varsave,'m',ofile)

    print('The {} EOF unscaled are saved as\n{}'.format(numpcs,ofile))
    print('____________________________________________________________\n')

    # save EOF scaled: EOFs are multiplied by the square-root of their eigenvalues
    varsave='EOFscal2'
    ofile='{}EOFscal2_{}.nc'.format(OUTPUTdir,name_outputs_pcs)
    save_N_2Dfields(lat_area,lon_area,eofs_scal2[:numpcs],varsave,'m',ofile)

    print('The {} EOF scaled2 (multiplied by the square-root of their eigenvalues)are saved as\n{}'.format(numpcs,ofile))
    print('__________________________________________________________\n')

    # save PC unscaled
    # [time x PCi]
    namef='{}PCunscal_{}.txt'.format(OUTPUTdir,name_outputs_pcs)
    np.savetxt(namef, pcs_unscal0[:,:numpcs])

    #save PC scaled: PCs are scaled to unit variance (divided by the square-root of their eigenvalue)
    # [time x PCi]
    namef='{}PCscal1_{}.txt'.format(OUTPUTdir,name_outputs_pcs)
    np.savetxt(namef, pcs_scal1[:,:numpcs])
    print('The {} PCs in columns are saved in\n{}'.format(numpcs,OUTPUTdir))
    print('__________________________________________________________\n')

    # save varfrac: the fractional variance represented by each EOF mode
    namef='{}varfrac_{}.txt'.format(OUTPUTdir,name_outputs_pcs)
    np.savetxt(namef, varfrac, fmt='%1.10f')


    print('\n*******************************************************\n')
    print('END COMPUTE')
    print('*********************************************************\n')

    return solver


def read_out_compute(OUTPUTdir, name_outputs, numpcs):
    """
    Reads the output of compute from the files.
    """

    # load solver
    namef='{}solver_{}_{}pcs.p'.format(OUTPUTdir,name_outputs,numpcs)
    solver = pickle.load(open(namef, 'rb'))

    # name_savedvar='{}PCunscal_{}.txt'.format(OUTPUTdir,name_outputs)
    # PCunscal=np.loadtxt(name_savedvar)
    # print('(times, numPCs) ={}'.format(PCunscal.shape))

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
    from clus_manipulate import cluster_orderingFREQ, compute_clusterpatterns, compute_pvectors
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
    var_ensList_glob, lat, lon, var_ensList, lat_area, lon_area, dates, time_units, var_units = out_precompute

    syr=pd.to_datetime(dates).year[0]
    eyr=pd.to_datetime(dates).year[-1]

    name_outputs_pcs = name_outputs+'_{}pcs'.format(numpcs)

    # LOAD DATA PROCESSED BY compute.py
    if solver is None:
        solver = read_out_compute(OUTPUTdir, name_outputs, numpcs)
    PCunscal = solver.pcs()
    pc=np.transpose(PCunscal)

    # k-means analysis using the subset of PCs
    #______________________________________
    printsep()
    print('number of clusters: {}\n'.format(numclus))
    printsep()

    npart=100      #100
    varopt=[]
    indcl0=[]
    centr=[]
    print('npart ={}'.format(npart))

    # This calculates the clustering for k=2 to 7.. useful but may insert an option to do this..
    if inputs['check_sig']:
        for ncl in range(2,7):
            nfcl,indcl1,centr1,varopt1,iseed=ctool.cluster_toolkit.clus_opt(ncl,npart,pc)
            indcl0.append(np.subtract(indcl1,1))        #indcl1 starts from 1, indcl0 starts from 0
            centr.append(centr1)
            varopt.append(varopt1)
        indcl=indcl0[numclus-2]
        centr=centr[numclus-2]
    else:
        nfcl,indcl1,centr,varopt1,iseed=ctool.cluster_toolkit.clus_opt(numclus, npart, pc)
        indcl = indcl1-1
        varopt = [varopt1]

    # save cluster index
    namef='{}indcl_{}clus_{}.txt'.format(OUTPUTdir,numclus,name_outputs_pcs)
    np.savetxt(namef,indcl,fmt='%d')

    # save cluster centroids
    namef='{}centr_{}clus_{}.txt'.format(OUTPUTdir,numclus,name_outputs_pcs)
    np.savetxt(namef,centr)

    # save cluster optimal variance ratio (this is needed for significance computation: clusters_sig.py)
    if inputs['check_sig']:
        namef='{}varopt_2to6clus_{}.txt'.format(OUTPUTdir,name_outputs_pcs)
        np.savetxt(namef,varopt, fmt='%1.10f')
    else:
        print('Optimal variance ratio is {:8.4f}\n'.format(varopt1))


    # Cluster ordering in decreasing frequency
    #_______________________
    centrORD,indclORD=cluster_orderingFREQ(indcl,centr,numclus)

    # save cluster index
    namef='{}indclORD_{}clus_{}.txt'.format(OUTPUTdir,numclus,name_outputs_pcs)
    np.savetxt(namef,indclORD,fmt='%d')

    # save cluster centroids
    namef='{}centrORD_{}clus_{}.txt'.format(OUTPUTdir,numclus,name_outputs_pcs)
    np.savetxt(namef,centrORD)

    # COMPUTE AND SAVE CLUSTERS PATTERNS
    #_______________________
    # compute cluster patterns
    cluspattORD=compute_clusterpatterns(numclus,var_ensList,indclORD)
    cluspattORD = np.array(cluspattORD)
    # save cluster patterns
    varsave='cluspattern'
    ofile='{}cluspatternORD_{}clus_{}.nc'.format(OUTPUTdir,numclus,name_outputs_pcs)
    save_N_2Dfields(lat_area,lon_area,cluspattORD,varsave,var_units,ofile)

    # pvec=compute_pvectors(numpcs,solver,cluspattORD)
    # print('pvec:\n{}'.format(pvec))
    # # save pvec
    # namef='{}pvec_{}clus_{}.txt'.format(OUTPUTdir,numclus,name_outputs_pcs)
    # np.savetxt(namef, pvec)

    print('Cluster pattern netCDF variable (ordered by decreasing frequency of occurrence) is saved as\n{}'.format(ofile))
    printsep()

    print('\n*******************************************************\n')
    print('END CLUSTERING')
    print('*********************************************************\n')

    return centrORD, indclORD, cluspattORD, varopt


def read_out_clustering(OUTPUTdir, name_outputs, numpcs, numclus):
    """
    Reads the output of clustering from files.
    """
    from readsavencfield import read_N_2Dfields

    # load cluster patterns
    ifile='{}cluspatternORD_{}clus_{}_{}pcs.nc'.format(OUTPUTdir,numclus,name_outputs, numpcs)
    cluspattORD,lat_area,lon_area =read_N_2Dfields(ifile)

    #load centrORD
    namef='{}centrORD_{}clus_{}_{}pcs.txt'.format(OUTPUTdir,numclus,name_outputs, numpcs)
    centrORD=np.loadtxt(namef)

    # load indclORD
    namef='{}indclORD_{}clus_{}_{}pcs.txt'.format(OUTPUTdir,numclus,name_outputs, numpcs)
    indclORD=np.loadtxt(namef)

    # Load varopt
    namef='{}varopt_2to6clus_{}_{}pcs.txt'.format(OUTPUTdir,name_outputs, numpcs)
    varopt = np.loadtxt(namef)

    return centrORD, indclORD, cluspattORD, varopt


def read_out_cluscompare(OUTPUTdir, name_outputs, numpcs, numclus):
    """
    Reads the output of clustering from files.
    """
    from readsavencfield import read_N_2Dfields

    # load cluster patterns
    ifile='{}cluspatternORDasREF_{}clus_{}_{}pcs.nc'.format(OUTPUTdir,numclus,name_outputs, numpcs)
    cluspattORD,lat_area,lon_area =read_N_2Dfields(ifile)

    #load centrORD
    namef='{}centrORDasREF_{}clus_{}_{}pcs.txt'.format(OUTPUTdir,numclus,name_outputs, numpcs)
    centrORD=np.loadtxt(namef)

    # load indclORD
    namef='{}indclORDasREF_{}clus_{}_{}pcs.txt'.format(OUTPUTdir,numclus,name_outputs, numpcs)
    indclORD=np.loadtxt(namef)

    return centrORD, indclORD, cluspattORD

def clusters_comparison(inputs, out_precompute = None, solver = None, out_clustering = None, solver_ERA = None, out_clustering_ERA = None, out_clustering_NCEP = None):
    """
    Finds the best possible match between two sets of cluster partitions.
    - It computes and saves the projections of the simulated cluster patterns and the reference cluster patterns onto the same reference EOF patterns, obtaining vectors in the PC space (numpcs dimensions).
    - It compares the two cluster partitions computing the RMS error, the phase error, the pattern correlation and the combination that minimizes the mean total RMS error, which can be used to order the simulated clusters as the reference ones.
    - It sorts the clusters as the reference ones.
    - Compares the RMS errors to those between the two reference (ERA vs NCEP) datasets.

    pvec_ERA and pvec_NCEP contain the pcs of ERA and NCEP clusters, in order to compare with the current model.
    """

    from itertools import permutations
    from operator import itemgetter
    # User-defined libraries
    import ctool
    from readsavencfield import read_N_2Dfields, read3Dncfield, save_N_2Dfields
    from clus_manipulate import cluster_ordering_asREF, compute_pvectors, compute_pseudoPC, plot_clusters
    import pcmatch

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

    # LOAD DATA PROCESSED BY precompute
    if out_precompute is None:
        warnings.warn('Reading the output of precompute from files')
        out_precompute = read_out_precompute(OUTPUTdir, name_outputs, numens)
    var_ensList_glob, lat, lon, var_ensList, lat_area, lon_area, dates, time_units, var_units = out_precompute

    syr=pd.to_datetime(dates).year[0]
    eyr=pd.to_datetime(dates).year[-1]

    name_outputs_pcs=name_outputs+'_{}pcs'.format(numpcs)

    # LOAD DATA PROCESSED BY compute.py
    if solver is None:
        solver = read_out_compute(OUTPUTdir, name_outputs, numpcs)
    PCunscal = solver.pcs()
    pc=np.transpose(PCunscal)

    # LOAD DATA PROCESSED BY clustering.py
    #______________________________________
    # N.B. suffix ORD refers to clusters ordered by decreasing frequency of occurrence.
    #      suffix ORDasREF refers to clusters ordered as te reference clusters (e.g. observartions)

    if out_clustering is None:
        out_clustering = read_out_clustering(OUTPUTdir, name_outputs_pcs, numpcs, numclus)
    centrORD, indclORD, cluspattORD, varopt = out_clustering


    if solver_ERA is None:
        filesolver = '{}solver_{}_{}pcs.p'.format(inputs['ERA_ref_folder'], inputs['name_ERA'], inputs['numpcs'])

        if not (os.path.isfile(filesolver)):
            raise RuntimeError('EOF solver for ERA with {} pcs not found in {}.'.format(numpcs, inputs['ERA_ref_folder']))
        else:
            solver_ERA = pickle.load(open(filesolver, 'rb'))

    if out_clustering_ERA is None:
        out_clustering_ERA = read_out_clustering(inputs['ERA_ref_folder'], inputs['name_ERA'], numpcs, numclus)

    centrORD_ERA, indclORD_ERA, cluspattORD_ERA, varopt_ERA = out_clustering_ERA
    pvec_ERA = compute_pvectors(numpcs,solver_ERA,cluspattORD_ERA)
    print('pvec_ERA:\n{}'.format(pvec_ERA))

    if out_clustering_NCEP is None:
        out_clustering_NCEP = read_out_clustering(inputs['NCEP_ref_folder'], inputs['name_NCEP'], numpcs, numclus)

    centrORD_NCEP, indclORD_NCEP, cluspattORD_NCEP, varopt_NCEP = out_clustering_NCEP
    pvec_NCEP = compute_pvectors(numpcs,solver_ERA,cluspattORD_NCEP)
    print('pvec_NCEP:\n{}'.format(pvec_NCEP))


    # Compute p vector as in Dawson&Palmer 2015:
    #___________________________________________
    # The clusters are projected on the REFERENCE EOF base, given by solver_ERA. The RMS error is computed first in this same base.

    pvec=compute_pvectors(numpcs,solver_ERA,cluspattORD)
    print('pvec:\n{}'.format(pvec))
    # save pvec
    namef='{}pvec_{}clus_{}.txt'.format(OUTPUTdir,numclus,name_outputs_pcs)
    np.savetxt(namef, pvec)


    pseudoPC=compute_pseudoPC(var_ensList,numpcs,solver_ERA)
    print('pseudoPC:\n{}'.format(pseudoPC))
    # save pseudoPC
    namef='{}pseudoPC_{}.txt'.format(OUTPUTdir,name_outputs_pcs)
    np.savetxt(namef,pseudoPC)


    # Find the match with the given reference clusters
    #_______________________
    pcset1 = pvec
    pcset2 = pvec_ERA
    #match_clusters_info(pcset1, pcset2,npcs=9)
    perm, et, ep, patcor = pcmatch.match_pc_sets(pcset2, pcset1)

    pcset1 = pvec_NCEP
    pcset2 = pvec_ERA
    #match_clusters_info(pcset1, pcset2,npcs=9)
    perm_ncep, et_ncep, ep_ncep, patcor_ncep = pcmatch.match_pc_sets(pcset2, pcset1)

    printsep()
    print('Calculating errors in the {} eof ERA reference space'.format(numpcs))
    printsep()
    print('RMS error = {}'.format(et))
    #print('(squared) Phase error = {}'.format(ep))
    print('Pattern correlation (cosine of angle between) = {}'.format(patcor))
    print('Optimal permutation = {}'.format(perm))
    print('Mean RMS error = {}'.format(np.mean(et)))
    printsep()
    print('NCEP RMS error = {}'.format(et_ncep))
    print('NCEP Pattern correlation (cosine of angle between) = {}'.format(patcor_ncep))
    print('NCEP Optimal permutation = {}'.format(perm_ncep))
    print('NCEP Mean RMS error = {}'.format(np.mean(et_ncep)))
    printsep()
    rms_ratio = np.array(et)/np.array(et_ncep)
    print('Ratio to NCEP RMS error (# sigmas) = {}'.format(rms_ratio))
    print('Mean Ratio (# sigmas) = {}'.format(np.mean(rms_ratio)))
    printsep()
    printsep()

    # save et
    namef='{}et_{}clus_{}.txt'.format(OUTPUTdir,numclus,name_outputs_pcs)
    np.savetxt(namef, et)

    # save ep
    namef='{}ep_{}clus_{}.txt'.format(OUTPUTdir,numclus,name_outputs_pcs)
    np.savetxt(namef, ep)

    # save patcor
    namef='{}patcor_{}clus_{}.txt'.format(OUTPUTdir,numclus,name_outputs_pcs)
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
    namef='{}centrORDasREF_{}clus_{}.txt'.format(OUTPUTdir,numclus,name_outputs_pcs)
    np.savetxt(namef,centrORDasREF)

    # save indclORDasREF
    namef='{}indclORDasREF_{}clus_{}.txt'.format(OUTPUTdir,numclus,name_outputs_pcs)
    np.savetxt(namef,indclORDasREF, fmt='%d')

    # save freqORDasREF
    freq=[]
    for nclus in range(numclus):
        cl=list(np.where(indclORDasREF==nclus)[0])
        fr=len(cl)*100./len(indclORDasREF)
        freq.append(fr)
    freqORDasREF=np.array(freq)
    namef='{}freq_ORDasREF_{}clus_{}.txt'.format(OUTPUTdir,numclus,name_outputs_pcs)
    np.savetxt(namef, freqORDasREF)

    # save cluspattORDasREF
    cluspattORDasREF=cluspattORD[sor]
    varsave='cluspattern'
    ofile='{}cluspatternORDasREF_{}clus_{}.nc'.format(OUTPUTdir,numclus,name_outputs_pcs)
    save_N_2Dfields(lat_area,lon_area,np.array(cluspattORDasREF),varsave,var_units,ofile)

    # Loading ERA ref clusters...
    et_fs, patcor_fs = pcmatch.rms_calc_fullspace(cluspattORDasREF, cluspattORD_ERA)

    et_fs_ncep, patcor_fs_ncep = pcmatch.rms_calc_fullspace(cluspattORD_NCEP, cluspattORD_ERA)

    printsep()
    print('Calculating errors in the full space')
    printsep()
    print('RMS error = {}'.format(et_fs))
    print('Pattern correlation (cosine of angle between) = {}'.format(patcor_fs))
    print('Mean RMS error = {}'.format(np.mean(et_fs)))
    printsep()
    print('NCEP RMS error = {}'.format(et_fs_ncep))
    print('NCEP Pattern correlation (cosine of angle between) = {}'.format(patcor_fs_ncep))
    print('NCEP Mean RMS error = {}'.format(np.mean(et_fs_ncep)))
    printsep()
    rms_ratio_fs = np.array(et_fs)/np.array(et_fs_ncep)
    print('Ratio to NCEP RMS error (# sigmas) = {}'.format(rms_ratio_fs))
    print('Mean Ratio (# sigmas) = {}'.format(np.mean(rms_ratio_fs)))
    printsep()

    printsep()
    print('Showing difference in the EOF projections\n')
    eofs = solver.eofs()
    eofs_era = solver_ERA.eofs()

    cosines_2D = []
    for i in range(10):
        cosi = []
        for j in range(10):
            cosi.append(pcmatch.cosine(eofs[i], eofs_era[j]))
        cosines_2D.append(cosi)

    cosines_2D = np.array(cosines_2D)
    fig6 = plt.figure()

    plt.imshow(cosines_2D)
    plt.xlabel('EOFs {}'.format(inputs['exp_name']))
    plt.ylabel('EOFs ERA Interim')
    plt.title('Cosine of relative angle')
    plt.colorbar()
    plt.savefig(OUTPUTdir+'ERAvs{}_cosines_EOFs.pdf'.format(inputs['exp_name']), format = 'pdf')

    # Compute inter-clusters and intra-clusters statistics
    #if area=='EAT':
    #    clusnames=['NAO+','Blocking','Atlantic Ridge','NAO-']
    #else:

    # COMPUTE AND SAVE STATISTICS OF CLUSTER PARTITION
    #_______________________
    clusnames=['clus{}'.format(nclus) for nclus in range(numclus)]
    print('--------------------------DIMENSIONS:------------------------------------')
    print('ANOMALY FIELD:                  [time, lat, lon]         = {}'.format(np.concatenate(var_ensList[:]).shape))
    print('EMPIRICAL ORTHOGONAL FUNCTIONS: [num_eof, lat, lon]      = ({}, {}, {})'.format(PCunscal.shape[1],len(lat_area),len(lon_area)))
    print('PRINCIPAL COMPONENTS:           [time, num_eof]          = {}'.format(PCunscal.shape))
    print('CENTROID COORDNATES:            [num_eof, num_clusters]  = {}'.format(centrORDasREF.shape))
    print('CLUSTER PATTERNS:               [num_clusters, lat, lon] = {}'.format(cluspattORDasREF.shape))


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

    print('\nINTER-CLUSTER DISTANCES MEAN IS {}'.format(np.mean(norm_centr)))
    print('INTER-CLUSTER DISTANCES STANDARD DEVIATION {}'.format(np.std(norm_centr)))
    print('==============================================================')



    # 2) compute centroid-datapoint distance in the phase space (PC space):
    print('\n==============================================================')
    print('In the PC space, here are the distances between each data point in a cluster and its centroid:')
    numdatapoints=PCunscal.shape[0]

    for nclus in range(numclus):
        print('\n******************            {}           ***************************'.format(clusnames[nclus]))
        cl=list(np.where(indclORDasREF==nclus)[0])
        print('{} DAYS IN THIS CLUSTER'.format(len(cl)))
        PCsintheclus=[PCunscal[i,:] for i in cl]
        #print('standard deviation of PCs relative to days belonging to this cluster={}'.format(np.std(PCsintheclus)))

        norm=[] # [numdatapoints]
        for point in cl:
            normpoints=LA.norm(centrORDasREF[:,nclus]-PCunscal[point,:])
            norm.append(normpoints)

        print(len(norm))
        print('MAXIMUM DISTANCE FOR CLUS {} IS {}'.format(nclus,max(norm)))
        txt='Furthest data point to centroid of cluster {} is day {}'.format(nclus,list(np.where(norm == max(norm))[0]))
        print(txt)
        print('INTRA-CLUSTER DISTANCES MEAN FOR CLUS {} IS {}'.format(nclus,np.mean(norm)))
        print('INTRA-CLUSTER DISTANCES STANDARD DEVIATION FOR CLUS {} IS {}'.format(nclus,np.std(norm)))
    print('==============================================================')


    print('\n****************************************************\n')
    print('END CLUSTER COMPARISON')
    print('*******************************************************\n')

    return centrORDasREF, indclORDasREF, cluspattORDasREF


def clusters_plot(inputs, out_precompute = None, solver = None, out_clus_compare = None):
    """
    Produces and saves plots of weather regimes' patterns.
    """
    #from mpl_toolkits.basemap import Basemap
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import ctool
    from clus_manipulate import plot_clusters
    from readsavencfield import read3Dncfield

    print('***************************************************\n')
    print('Running clusters plots')
    print('***************************************************\n')

    name_outputs  = inputs['name_outputs']  # name of the outputs
    numens        = inputs['numens']  # total number of ensemble members
    numpcs        = inputs['numpcs']  # number of principal components
    perc          = inputs['perc']  # percentage of explained variance for EOF analysis
    enstoselect   = inputs['enstoselect']  # ensemble member to analyse
    numclus       = inputs['numclus']  # number of clusters

    OUTPUTdir = inputs['OUTPUTdir']

    # LOAD DATA PROCESSED BY precompute
    if out_precompute is None:
        warnings.warn('Reading the output of precompute from files')
        out_precompute = read_out_precompute(OUTPUTdir, name_outputs, numens)
    var_ensList_glob, lat, lon, var_ensList, lat_area, lon_area, dates, time_units, var_units = out_precompute

    syr=pd.to_datetime(dates).year[0]
    eyr=pd.to_datetime(dates).year[-1]

    name_outputs_pcs=name_outputs+'_{}pcs'.format(numpcs)

    # LOAD DATA PROCESSED BY compute.py
    if solver is None:
        solver = read_out_compute(OUTPUTdir, name_outputs, numpcs)
    PCunscal = solver.pcs()
    pc=np.transpose(PCunscal)

    # LOAD DATA PROCESSED BY clusters_comparison.py
    #______________________________________
    #load indclORDasREF
    if out_clus_compare is None:
        namef='{}indclORDasREF_{}clus_{}.txt'.format(OUTPUTdir,numclus,name_outputs_pcs)
        indclORDasREF=np.loadtxt(namef)
    else:
        centrORDasREF, indclORDasREF, cluspattORDasREF = out_clus_compare

    # PLOT AND SAVE CLUSTERS FIGURE
    # Plot of cluster patterns in one panel
    #______________________________________
    varname = inputs['varname']
    model = inputs['model_name']
    res = inputs['resolution']
    numens = inputs['numens']
    season = inputs['season']
    area = inputs['area']

    if enstoselect is not None:
        tit='{} {} ens{}\n{} {} {} ({}-{})'.format(varname,model,enstoselect,res,season,area,syr,eyr)
    else:
        tit='{} {} {}\n{} {} ({}-{})'.format(varname,model,res,season,area,syr,eyr)

    ax=plot_clusters(area,lon,lat,lon_area,lat_area,numclus,numpcs,var_ensList_glob,indclORDasREF,tit)

    namef='{}clus_patterns_{}clus_{}.eps'.format(OUTPUTdir,numclus,name_outputs_pcs)
    ax.figure.savefig(namef)#bbox_inches='tight')
    print('Clusters eps figure for {} weather regimes is saved as\n{}'.format(area,namef))
    print('____________________________________________________________________________________________________________________')


    print('\n*************************************************\n')
    print('END CLUS PLOT')
    print('***************************************************\n')

    return


def clusters_sig(inputs, solver = None, out_clustering = None):
    """
    H_0: There are no regimes ---> multi-normal distribution PDF
    Synthetic datasets modelled on the PCs of the original data are computed (synthetic PCs have the same lag-1, mean and standard deviation of the original PCs)
    SIGNIFICANCE = % of times that the optimal variance ratio found by clustering the real dataset exceeds the optimal ratio found by clustering the syntetic dataset (is our distribution more clustered than a multinormal distribution?)
    """

    import ctool
    import ctp

    print('***********************************************************\n')
    print('Running CLUSTER SIG')
    print('***********************************************************\n')

    name_outputs  = inputs['name_outputs']  # name of the outputs
    numens        = inputs['numens']  # total number of ensemble members
    numpcs        = inputs['numpcs']  # number of principal components
    perc          = inputs['perc']  # percentage of explained variance for EOF analysis
    enstoselect   = inputs['enstoselect']  # ensemble member to analyse
    numclus       = inputs['numclus']  # number of clusters
    freq = inputs['freq']
    season = inputs['season']

    OUTPUTdir = inputs['OUTPUTdir']

    name_outputs_pcs=name_outputs+'_{}pcs'.format(numpcs)

    # LOAD DATA PROCESSED BY clusters_comparison.py
    #______________________________________

    if out_clustering is None:
        out_clustering = read_out_clustering(OUTPUTdir, name_outputs_pcs)
    centrORD, indclORD, cluspattORD, varopt = out_clustering

    if solver is None:
        solver = read_out_compute(OUTPUTdir, name_outputs, numpcs)
    PCunscal = solver.pcs()
    pc=np.transpose(PCunscal)

    npart=100      #100
    nrsamp=500     #500
    print('\nNumber of synthetic PCs used for significance computation = {}'.format(nrsamp))
    if freq=='day':
        ndis=int(PCunscal.shape[0]/(30*len(season)))-1        #28 (e.g. DJF data: ndis=number of winters-1)
    elif freq=='mon':
        ndis=int(PCunscal.shape[0]/(12*len(season)))-1
    else:
        print('input data frequency is neither day nor mon')
    print('check: number of years={}'.format(ndis+1))

    # Compute significance

    #=======serial=========
    #significance=ctool.cluster_toolkit.clus_sig(nrsamp,npart,ndis,pc,varopt)
    #=======parallel=======
    start = datetime.datetime.now()
    significance=ctp.cluster_toolkit_parallel.clus_sig_p(nrsamp,npart,ndis,pc,varopt)
    end = datetime.datetime.now()
    print('significance computation took me %s seconds' %(end-start))

    print 'significance =', significance
    print '{.16f}'.format(significance[2])
    print 'significance for 4 clusters =', significance[2]

    namef='{}sig_2to6clus_{}nrsamp_{}.txt'.format(OUTPUTdir,nrsamp,name_outputs_pcs)
    np.savetxt(namef,significance)


    print('\n*******************************************************\n')
    print('END CLUSTER SIG')
    print('*********************************************************\n')

    return significance
