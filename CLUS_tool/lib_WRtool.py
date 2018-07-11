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
    If it does not exist, it creates the output directory
    It reads the netCDF file of (lat, lon, lev, variable) for one or more ensemble members (if there are more members, they should be the only files present in INPUT_PATH).
    It can select one level, a season, it can filter by running mean and compute anomalies by removing the time-mean for each time (e.g. day), then it can select an area.
    It saves the resulting netCDF field of anomalies (lat, lon, anomalies).
    """
    from readsavencfield import read4Dncfield, save3Dncfield
    from manipulate import sel_season, anomalies, sel_area

    print('**********************************************************\n')
    print('Running PRECOMPUTATION\n')
    print('**********************************************************\n')

    dir_WRtool    = inputs['dir_WRtool']  # WRTOOL directory
    dir_OUTPUT    = inputs['dir_OUTPUT']  # OUTPUT directory
    name_outputs  = inputs['name_outputs']  # name of the outputs
    varname       = inputs['varname']  # variable name
    level         = inputs['level']  # level to select
    wnd           = inputs['wnd']  # running mean filter time window
    numens        = inputs['numens']  # total number of ensemble members
    season        = inputs['season']  # season
    area          = inputs['area']  # area
    filenames     = inputs['filenames']  # file names (absolute paths)

    # User-defined libraries
    sys.path.insert(0,inputs['dir_WRtool']+'WRtool/')

    # OUTPUT DIRECTORY
    OUTPUTdir=inputs['dir_OUTPUT']+'OUTPUT/'
    OUTfig=OUTPUTdir+'OUTfig/'
    OUTnc=OUTPUTdir+'OUTnc/'
    OUTtxt=OUTPUTdir+'OUTtxt/'
    OUTpy=OUTPUTdir+'OUTpy/'
    if not os.path.exists(OUTPUTdir):
        os.mkdir(OUTPUTdir)
        print('The output directory {0} is created'.format(OUTPUTdir))
        os.mkdir(OUTfig)
        os.mkdir(OUTnc)
        os.mkdir(OUTtxt)
        os.mkdir(OUTpy)
    else:
        print('The output directory {0} already exists'.format(OUTPUTdir))

    #____________Reading the netCDF file of 4Dfield, for one or more ensemble members
    for ens in range(numens):
        ifile=filenames[ens]
        if numens>1:
            print('\n/////////////////////////////////////\nENSEMBLE MEMBER %s' %ens)
        elif numens==1:
            print('\n/////////////////////////////////////')
        var, level, lat, lon, dates, time_units, var_units, time_cal = read4Dncfield(ifile,lvel=level)
        var_season,dates_season,var_filt = sel_season(var,dates,season,wnd) # Select data for that season and average data for that season after a running mean of window=wnd
        var_anom = anomalies(var_filt,var_season, dates_season,season)#,freq)                        # Compute anomalies by removing the time-mean for each time (day, month,...)
        namefile='glob_anomaly_{0}_{1}.nc'.format(name_outputs,ens) #like anomaly_zg500_day_ECEARTH31_base_T255_DJF_EAT_0
        ofile=OUTnc+namefile
        save3Dncfield(lat,lon,var_anom,varname,var_units,dates_season,time_units,time_cal,ofile)
        var_area, lat_area, lon_area = sel_area(lat,lon,var_anom,area)      # Selecting only [latS-latN, lonW-lonE] box region defineds by area
        namefile='area_anomaly_{0}_{1}.nc'.format(name_outputs,ens) #like anomaly_zg500_day_ECEARTH31_base_T255_DJF_EAT_0
        ofile=OUTnc+namefile
        save3Dncfield(lat_area,lon_area,var_area,varname,var_units,dates_season,time_units,time_cal,ofile)

    print('\nINPUT DATA:')
    print('Variable is {0}{1} ({2})'.format(varname,level,var_units))
    print('Number of ensemble members is {0}'.format(numens))
    print('Input data are read and anomaly fields for each ensemble member are computed and saved in {0}'.format(OUTnc))

    print('\n**********************************************************\n')
    print('END PRECOMPUTATION\n')
    print('************************************************************\n')

    return
