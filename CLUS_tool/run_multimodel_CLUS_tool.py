#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import sys
import pickle
import os
import datetime
from shutil import copy2

from matplotlib import pyplot as plt
from numpy import linalg as LA
import netCDF4 as nc
from netCDF4 import Dataset, num2date
import cartopy.crs as ccrs

import lib_WRtool as lwr
import Tkinter
import tkMessageBox
from copy import deepcopy as cp

# open our log file
logname = 'log_WRtool.log'
logfile = open(logname,'w') #self.name, 'w', 0)

# re-open stdout without buffering
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)

# redirect stdout and stderr to the log file opened above
os.dup2(logfile.fileno(), sys.stdout.fileno())
os.dup2(logfile.fileno(), sys.stderr.fileno())

print('*******************************************************')
print('Running {0}'.format(sys.argv[0]))
print('********************************************************')

if len(sys.argv) > 1:
    file_input = sys.argv[1] # Name of input file (relative path)
else:
    file_input = 'input_multi_CLUStool.in'

keys = 'dir_WRtool dir_OUTPUT dir_INPUT model_name exp_name varname level wnd season area freq filenames numpcs numclus perc enstoselect start_year end_year resolution ERA_ref_orig NCEP_ref_orig ERA_ref_folder NCEP_ref_folder check_sig run_compare separate_members'
keys = keys.split()
itype = [str, str, str, list, list, str, float, int, str, str, str, list, int, int, float, int, int, int, str, str, str, str, str, bool, bool, bool]

if len(itype) != len(keys):
    raise RuntimeError('Ill defined input keys in {}'.format(__file__))
itype = dict(zip(keys, itype))

defaults = dict()
defaults['numclus'] = 4 # 4 clusters
defaults['numens'] = 1 # single ensemble member
defaults['enstoselect'] = None
defaults['perc'] = None
defaults['wnd'] = 5 # 5 day running mean
defaults['check_sig'] = False
defaults['run_compare'] = True
defaults['separate_members'] = True

inputs = lwr.read_inputs(file_input, keys, n_lines = None, itype = itype, defaults = defaults)

inputs['numens'] = len(inputs['filenames'])
filenames_abs = []
for fil in inputs['filenames']:
    filenames_abs.append(inputs['dir_INPUT']+fil)
inputs['filenames'] = filenames_abs

inputs_era = cp(inputs)
inputs_era['model_name'] = 'ERAInterim'
name_ERA = lwr.std_outname(inputs_era, with_pcs = False)
inputs_era['name_outputs'] = name_ERA
inputs_era['numens'] = 1
inputs['name_ERA'] = name_ERA

inputs_ncep = cp(inputs)
inputs_ncep['model_name'] = 'NCEPNCAR'
name_NCEP = lwr.std_outname(inputs_ncep, with_pcs = False)
inputs_ncep['name_outputs'] = name_NCEP
inputs_ncep['numens'] = 1
inputs['name_NCEP'] = name_NCEP

# User-defined libraries
sys.path.insert(0,inputs['dir_WRtool']+'WRtool/')

#### Checks for ERA_ref and NCEP_ref, if not present, runs first the analysis on them

filesolver = '{}solver_{}_{}pcs.p'.format(inputs['ERA_ref_folder'], name_ERA, inputs['numpcs'])
#load pREF
namef='{}cluspatternORD_{}clus_{}_{}pcs.nc'.format(inputs['ERA_ref_folder'],inputs['numclus'],name_ERA,inputs['numpcs'])

if not (os.path.isfile(filesolver) and os.path.isfile(namef)):
    if not os.path.isfile(filesolver):
        print(filesolver+' not found\n')
    if not os.path.isfile(namef):
        print(namef+' not found\n')
    # Run the analysis on ERA ref
    inputs_era['filenames'] = [inputs['ERA_ref_orig']]
    inputs_era['OUTPUTdir'] = inputs['ERA_ref_folder']
    inputs_era, out_precompute_ref = lwr.precompute(inputs_era)
    solver_ERA = lwr.compute(inputs_era, out_precompute = out_precompute_ref)
    out_clustering_ref = lwr.clustering(inputs_era, solver = solver_ERA, out_precompute = out_precompute_ref)
else:
    solver_ERA = pickle.load(open(filesolver, 'rb'))

filesolver = '{}solver_{}_{}pcs.p'.format(inputs['NCEP_ref_folder'], name_NCEP, inputs['numpcs'])
namef='{}cluspatternORD_{}clus_{}_{}pcs.nc'.format(inputs['NCEP_ref_folder'],inputs['numclus'],name_NCEP,inputs['numpcs'])

if not (os.path.isfile(filesolver) and os.path.isfile(namef)):
    if not os.path.isfile(filesolver):
        print(filesolver+' not found\n')
    if not os.path.isfile(namef):
        print(namef+' not found\n')
    # Run the analysis on NCEP ref
    inputs_ncep['filenames'] = [inputs['NCEP_ref_orig']]
    inputs_ncep['OUTPUTdir'] = inputs['NCEP_ref_folder']
    inputs_ncep, out_precompute_ref = lwr.precompute(inputs_ncep)
    solver_NCEP = lwr.compute(inputs_ncep, out_precompute = out_precompute_ref)
    out_clustering_ref = lwr.clustering(inputs_ncep, solver = solver_NCEP, out_precompute = out_precompute_ref)
else:
    solver_NCEP = pickle.load(open(filesolver, 'rb'))

### RUN THE program
print('Via!\n')
print(datetime.datetime.now())

if len(inputs['filenames']) > 1:
    if inputs['separate_members']:
        print('Running separately on each member\n')
        n_runs = len(inputs['filenames'])
        filenames_run = cp(inputs['filenames'])
        tags_run = cp(inputs['exp_name'])
        models_run = cp(inputs['model_name'])
        if len(inputs['exp_name']) < n_runs or len(models_run) < n_runs:
            raise ValueError('Not enough [exp_name]s or [model_name]s for the experiments. Required one for each filename.\n')
    else:
        n_runs = 1
        inputs['exp_name'] = inputs['exp_name'][0]
        inputs['model_name'] = inputs['model_name'][0]
        #### Creating the output directory
        OUTPUTdir=inputs['dir_OUTPUT']+'OUT_'+inputs['model_name']+'_'+inputs['exp_name']+'/'

        if not os.path.exists(OUTPUTdir):
            os.mkdir(OUTPUTdir)
            print('The output directory {0} is created'.format(OUTPUTdir))

        #### Created the output directory
        inputs['OUTPUTdir'] = OUTPUTdir
        inputs['name_outputs'] = lwr.std_outname(inputs, with_pcs = False)

        print('Concatenating all ensemble members\n')

lista_dir = []
if n_runs == 1:
    lista_dir.append(inputs['OUTPUTdir'])
    inputs, out_precompute = lwr.precompute(inputs)

    solver = lwr.compute(inputs, out_precompute = out_precompute)

    out_clustering = lwr.clustering(inputs, solver = solver, out_precompute = out_precompute)

    if inputs['run_compare']:
        out_clus_compare = lwr.clusters_comparison(inputs, out_precompute = out_precompute, solver = solver, out_clustering = out_clustering, solver_ERA = solver_ERA)

        lwr.clusters_plot(inputs, out_precompute = out_precompute, solver = solver, out_clus_compare = out_clus_compare)
    else:
        lwr.clusters_plot(inputs, out_precompute = out_precompute, solver = solver, out_clus_compare = out_clustering[:-1])

    if inputs['check_sig']:
        significance = lwr.clusters_sig(inputs, solver = solver, out_clustering = out_clustering)
else:
    final_results = dict()
    for nu, fil, tag, mod in zip(range(n_runs), filenames_run, tags_run, models_run):
        inputs['filenames'] = [fil]
        inputs['exp_name'] = tag
        inputs['model_name'] = mod
        inputs['numens'] = 1
        inputs['name_outputs'] = lwr.std_outname(inputs, with_pcs = False)
        print('\n------------------------------------\n')
        print('\n------------------------------------\n')
        print('\n Model {} file: {}. exp_name: {} \n'.format(nu, fil, tag))
        print('\n------------------------------------\n')
        #### Creating the output directory
        OUTPUTdir=inputs['dir_OUTPUT']+'OUT_'+inputs['model_name']+'_'+inputs['exp_name']+'/'

        if not os.path.exists(OUTPUTdir):
            os.mkdir(OUTPUTdir)
            print('The output directory {0} is created'.format(OUTPUTdir))

        #### Created the output directory
        inputs['OUTPUTdir'] = OUTPUTdir
        lista_dir.append(inputs['OUTPUTdir'])

        inputs, out_precompute = lwr.precompute(inputs)

        solver = lwr.compute(inputs, out_precompute = out_precompute)

        out_clustering = lwr.clustering(inputs, solver = solver, out_precompute = out_precompute)

        if inputs['run_compare']:
            out_clus_compare = lwr.clusters_comparison(inputs, out_precompute = out_precompute, solver = solver, out_clustering = out_clustering, solver_ERA = solver_ERA)

            lwr.clusters_plot(inputs, out_precompute = out_precompute, solver = solver, out_clus_compare = out_clus_compare)
        else:
            lwr.clusters_plot(inputs, out_precompute = out_precompute, solver = solver, out_clus_compare = out_clustering[:-1])

        if inputs['check_sig']:
            significance = lwr.clusters_sig(inputs, solver = solver, out_clustering = out_clustering)

        final_results[(mod, tag)] = (out_clus_compare, significance)

print(datetime.datetime.now())
print('Fine')
#####
if n_runs > 1:
    pickle.dump(final_results, open(inputs['dir_OUTPUT']+'final_res_PRIMAVERA.p','w'))

logfile.close()
for direc in lista_dir:
    copy2('input_CLUStool.in', direc)
    copy2(logname, direc)
