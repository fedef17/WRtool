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
import Tkinter
import tkMessageBox
from copy import deepcopy as cp

print('*******************************************************')
print('Running {0}'.format(sys.argv[0]))
print('********************************************************')

if len(sys.argv) > 1:
    file_input = sys.argv[1] # Name of input file (relative path)
else:
    file_input = 'input_CLUStool.in'


keys = 'dir_WRtool dir_OUTPUT dir_INPUT model_name exp_name varname level wnd season area freq filenames numpcs numclus perc enstoselect start_year end_year resolution ERA_ref_orig NCEP_ref_orig ERA_ref_folder NCEP_ref_folder overwrite_output'
keys = keys.split()
itype = [str, str, str, str, str, str, float, int, str, str, str, list, int, int, float, int, int, int, str, str, str, str, str, bool]

if len(itype) != len(keys):
    raise RuntimeError('Ill defined input keys in {}'.format(__file__))

defaults = dict()
defaults['numclus'] = 4 # 4 clusters
defaults['numens'] = 1 # single ensemble member
defaults['enstoselect'] = None
defaults['perc'] = None
defaults['wnd'] = 5 # 5 day running mean
defaults['exp_name'] = 'xxxx'
defaults['overwrite_output'] = False

inputs = lwr.read_inputs(file_input, keys, n_lines = n_lines, itype = itype)

inputs['numens'] = len(inputs['filenames'])

inputs['name_outputs'] = lwr.std_outname(inputs)
print(inputs['name_outputs'])

inputs_era = cp(inputs)
inputs_era['model_name'] = 'ERAInterim'
name_ERA = lwr.std_outname(inputs_era)

inputs_ncep = cp(inputs)
inputs_ncep['model_name'] = 'NCEPNCAR'
name_NCEP = lwr.std_outname(inputs_ncep)

# User-defined libraries
sys.path.insert(0,inputs['dir_WRtool']+'WRtool/')

#### Creating the output directory

OUTPUTdir=inputs['dir_OUTPUT']+'OUT_'+inputs['model_name']+'_'+inputs['exp_name']+'/'

def kill_proc():
    raise RuntimeError('exp_name already used. Change the exp_name in the file {}'.format(file_input))

if not os.path.exists(OUTPUTdir):
    os.mkdir(OUTPUTdir)
    print('The output directory {0} is created'.format(OUTPUTdir))
else:
    print('The output directory {0} already exists'.format(OUTPUTdir))
    if inputs['overwrite_output']:
        print('Key [overwrite_output] active, writing files in the output directory {0}'.format(OUTPUTdir))
    else:
        top = Tkinter.Tk(title = 'Folder already exists', message = "Folder {} already exists. Do you want to write in the same folder? (this will overwrite file with same name) (No will kill the program and let you change exp_name)".format(OUTPUTdir))
        B1 = Tkinter.Button(top, text = 'Yes', command = top.destroy())
        B1.pack()
        B2 = Tkinter.Button(top, text = 'No', command = kill_proc)
        B2.pack()
        top.mainloop()

#### Created the output directory
inputs['OUTPUTdir'] = OUTPUTdir

#### Checks for ERA_ref and NCEP_ref, if not present, runs first the analysis on them

filesolver = '{}solver_{}_{}pcs.p'.format(inputs['ERA_ref_folder'], name_ERA, inputs['numpcs'])
#load pREF
namef='{}pvec_{}clus_{}_{}pcs.txt'.format(inputs['ERA_ref_folder'],inputs['numclus'],name_ERA,inputs['numpcs'])

if not (os.path.isfile(filesolver) and os.path.isfile(namef)):
    # Run the analysis on ERA ref
    inputs_era['filenames'] = [inputs['ERA_ref_orig']]
    inputs_era, out_precompute_ref = lwr.precompute(inputs_era)
    solver_ERA = lwr.compute(inputs_era, out_precompute = out_precompute_ref)
    out_clustering_ref = lwr.clustering(inputs_era, solver = solver_ERA, out_precompute = out_precompute_ref)

solver_ERA = pickle.load(open(filesolver, 'rb'))
pvec_ERA = np.loadtxt(namef)

filesolver = '{}solver_{}_{}pcs.p'.format(inputs['NCEP_ref_folder'], name_NCEP, inputs['numpcs'])
namef='{}pvec_{}clus_{}_{}pcs.txt'.format(inputs['NCEP_ref_folder'],inputs['numclus'],name_NCEP,inputs['numpcs'])

if not (os.path.isfile(filesolver) and os.path.isfile(namef)):
    # Run the analysis on NCEP ref
    inputs_ncep['filenames'] = [inputs['ERA_ref_orig']]
    inputs_ncep, out_precompute_ref = lwr.precompute(inputs_ncep)
    solver_NCEP = lwr.compute(inputs_ncep, out_precompute = out_precompute_ref)
    out_clustering_ref = lwr.clustering(inputs_ncep, solver = solver_NCEP, out_precompute = out_precompute_ref)

solver_NCEP = pickle.load(open(filesolver, 'rb'))
pvec_NCEP = np.loadtxt(namef)




### RUN THE program
inputs, out_precompute = lwr.precompute(inputs)

solver = lwr.compute(inputs, out_precompute = out_precompute)

out_clustering = lwr.clustering(inputs, solver = solver, out_precompute = out_precompute)
