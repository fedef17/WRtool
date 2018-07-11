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

import lib_WRtool as lwr


print('*******************************************************')
print('Running {0}'.format(sys.argv[0]))
print('********************************************************')

if len(sys.argv) > 1:
    file_input = sys.argv[1] # Name of input file (relative path)
else:
    file_input = 'input_CLUStool.in'

keys = 'dir_WRtool dir_OUTPUT dir_INPUT model_name exp_name varname level wnd season area freq filenames numpcs numclus perc enstoselect start_year end_year resolution ERA_ref NCEP_ref'
keys = keys.split()
itype = [str, str, str, str, str, str, float, int, str, str, str, list, int, int, float, int, int, int, str str str]

defaults = dict()
defaults['numclus'] = 4 # 4 clusters
defaults['numens'] = 1 # single ensemble member
defaults['enstoselect'] = None
defaults['perc'] = None
defaults['wnd'] = 5 # 5 day running mean
defaults['exp_name'] = 'xxxx'

inputs = lwr.read_inputs(file_input, keys, n_lines = n_lines, itype = itype)
#inputs['numens'] = len(inputs[filenames])

# dir_WRtool = inputs['dir_WRtool']
# dir_OUTPUT  = inputs['dir_OUTPUT']
# varname  = inputs['varname']
# level  = inputs['level']
# wnd  = inputs['wnd']
# season  = inputs['season']
# area  = inputs['area']
#
# filenames  = inputs['filenames']
# numpcs  = inputs['numpcs']
# numclus  = inputs['numclus']
# perc  = inputs['perc']
# enstoselect  = inputs['enstoselect']

inputs['numens'] = len(inputs['filenames'])

name_outputs = inputs['varname'] + inputs['level'] + '_' + inputs['freq'] + '_' + inputs['model_name'] + '_' + inputs['resolution'] + '_' + inputs['numens'] + 'ens_' + inputs['season'] + '_' + inputs['area'] + '_' + inputs['start_year'] + '_' + inputs['end_year']
inputs['name_outputs'] = name_outputs

#name_outputs="${varname}${level}_${freq}_${model}_${kind}_${res}_${numens}ens_${season}_${area}_${syr}-${eyr}"
