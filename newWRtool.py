#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import sys
import os
import glob

import matplotlib
matplotlib.use('Agg') # This is to avoid the code crash if no Xwindow is available
from matplotlib import pyplot as plt

import pickle
from scipy import io
import iris
import xarray as xr

import climtools_lib as ctl
import climdiags as cd
from copy import deepcopy as copy
import itertools as itt

import warnings
warnings.simplefilter('default')
warnings.filterwarnings('default', category=DeprecationWarning)

#######################################
DEBUG = True
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14

def std_outname(tag, inputs, ref_name = False):
    """
    Gives the standard name for folders and output files.
    """
    if inputs['area'] != 'custom':
        name_outputs = '{}_{}_{}_{}clus'.format(tag, inputs['season'], inputs['area'], inputs['numclus'])
    else:
        lonW, lonE, latS, latN = inputs['custom_area']
        areatag = ''
        for cos in [ctl.convert_lon_std(lonW), ctl.convert_lon_std(lonE)]:
            if cos > 0:
                areatag += str(int(abs(cos))) + 'E'
            else:
                areatag += str(int(abs(cos))) + 'W'
        for cos in [latS, latN]:
            if cos > 0:
                areatag += str(int(abs(cos))) + 'N'
            else:
                areatag += str(int(abs(cos))) + 'S'

        name_outputs = '{}_{}_{}_{}clus'.format(tag, inputs['season'], areatag, inputs['numclus'])

    if inputs['flag_perc']:
        name_outputs += '_{}perc'.format(inputs['perc'])
    else:
        name_outputs += '_{}pcs'.format(inputs['numpcs'])


    if not ref_name:
        if inputs['year_range'] is not None:
            name_outputs += '_{}-{}'.format(inputs['year_range'][0], inputs['year_range'][1])
        else:
            name_outputs += '_allyrs'

        if inputs['use_reference_clusters']:
            name_outputs += '_refCLUS'
        elif inputs['use_reference_eofs']:
            if inputs['supervised_clustering']:
                name_outputs += '_refEOFsup'
            else:
                name_outputs += '_refEOF'
    else:
        if inputs['ref_year_range'] is not None:
            name_outputs += '_{}-{}'.format(inputs['ref_year_range'][0], inputs['ref_year_range'][1])
        else:
            name_outputs += '_allyrs'

    if inputs['detrend_local_linear']:
        name_outputs += '_dtrloc'
    elif inputs['detrend_only_global']:
        name_outputs += '_dtr'

    if not ref_name:
        if inputs['rebase_to_historical']:
            name_outputs += '_reb'

    return name_outputs

#if np.any(['log_WRtool_' in cos for cos in os.listdir('.')]):
#    os.system('rm log_WRtool_*log')

# open our log file
logname = 'log_WRtool_{}.log'.format(ctl.datestamp())
logfile = open(logname,'w') #self.name, 'w', 0)

# re-open stdout without buffering
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 1)

# redirect stdout and stderr to the log file opened above
os.dup2(logfile.fileno(), sys.stdout.fileno())
os.dup2(logfile.fileno(), sys.stderr.fileno())

print('*******************************************************')
print('Running {0}\n'.format(sys.argv[0]))
print(ctl.datestamp()+'\n')
print('********************************************************')

if len(sys.argv) > 1:
    file_input = sys.argv[1] # Name of input file (relative path)
else:
    file_input = 'input_WRtool.in'

keys = 'exp_name cart_in cart_out_general filenames model_names level season area numclus numpcs flag_perc perc ERA_ref_orig ERA_ref_folder run_sig_calc run_compare patnames patnames_short heavy_output model_tags year_range groups group_symbols reference_group use_reference_eofs obs_name filelist visualization bounding_lat plot_margins custom_area is_ensemble ens_option draw_rectangle_area use_reference_clusters out_netcdf out_figures out_only_main_figs taylor_mark_dim starred_field_names use_seaborn color_palette netcdf4_read ref_clus_order_file wnd_days show_transitions central_lat central_lon draw_grid cmip6_naming bad_matching_rule matching_hierarchy ref_year_range area_dtr detrend_only_global remove_29feb regrid_model_data single_model_ens_list plot_type custom_naming_keys pressure_levels calc_gradient supervised_clustering frac_super ignore_model_error select_area_first deg_dtr is_seasonal detrend_local_linear rebase_to_historical file_hist_rebase rebase_to_control var_long_name var_std_name var_units plot_cb_label multiple_areas multiple_seasons dump_regridded iris_read'
keys = keys.split()
itype = [str, str, str, list, list, float, str, str, int, int, bool, float, str, str, bool, bool, list, list, bool, list, list, dict, dict, str, bool, str, str, str, float, list, list, bool, str, bool, bool, bool, bool, bool, int, list, bool, str, bool, str, int, bool, float, float, bool, bool, str, list, list, str, bool, bool, bool, bool, str, list, bool, bool, bool, float, bool, bool, int, bool, bool, bool, str, bool, str, str, str, str, list, list, bool, bool]

if len(itype) != len(keys):
    raise RuntimeError('Ill defined input keys in {}'.format(__file__))
itype = dict(zip(keys, itype))

defaults = dict()
defaults['numclus'] = 4 # 4 clusters
defaults['perc'] = None
defaults['flag_perc'] = False
defaults['exp_name'] = 'WRtool'
defaults['run_sig_calc'] = False
defaults['run_compare'] = True
defaults['heavy_output'] = False
defaults['use_reference_eofs'] = False
defaults['ERA_ref_folder'] = 'ERA_ref'
defaults['obs_name'] = 'ERA'
defaults['visualization'] = 'Nstereo'
defaults['bounding_lat'] = 30.
defaults['plot_margins'] = None
defaults['is_ensemble'] = False
defaults['ens_option'] = 'all'
defaults['draw_rectangle_area'] = False
defaults['use_reference_clusters'] = False
defaults['out_netcdf'] = True
defaults['out_figures'] = True
defaults['out_only_main_figs'] = True
defaults['taylor_mark_dim'] = 100
defaults['use_seaborn'] = True
defaults['color_palette'] = 'hls'
defaults['netcdf4_read'] = False
defaults['wnd_days'] = 20
defaults['show_transitions'] = True
defaults['draw_grid'] = True
defaults['cmip6_naming'] = False
defaults['bad_matching_rule'] = 'rms_mean'
defaults['matching_hierarchy'] = None
defaults['area_dtr'] = 'global'
defaults['detrend_only_global'] = False
defaults['remove_29feb'] = True
defaults['regrid_model_data'] = False
defaults['single_model_ens_list'] = False
defaults['plot_type'] = 'pcolormesh'
defaults['custom_naming_keys'] = None
defaults['pressure_levels'] = False
defaults['calc_gradient'] = False
defaults['supervised_clustering'] = False
defaults['frac_super'] = 0.02
defaults['ignore_model_error'] = False
defaults['select_area_first'] = False
defaults['deg_dtr'] = 1
defaults['is_seasonal'] = False
defaults['detrend_local_linear'] = False
defaults['rebase_to_historical'] = False
defaults['file_hist_rebase'] = None
defaults['rebase_to_control'] = False
defaults['var_long_name'] = 'geopotential height anomaly at 500 hPa'
defaults['var_std_name'] = 'geopotential_height_anomaly'
defaults['var_units'] = 'm'
defaults['plot_cb_label'] = 'Geopotential height anomaly (m)'
defaults['multiple_areas'] = None
defaults['multiple_seasons'] = None
defaults['dump_regridded'] = False
defaults['iris_read'] = False

inputs = ctl.read_inputs(file_input, keys, n_lines = None, itype = itype, defaults = defaults)
for ke in inputs:
    print('{} : {}\n'.format(ke, inputs[ke]))

if inputs['cart_in'][-1] != '/': inputs['cart_in'] += '/'
if inputs['cart_out_general'][-1] != '/': inputs['cart_out_general'] += '/'
if inputs['ERA_ref_folder'][-1] != '/': inputs['ERA_ref_folder'] += '/'

if inputs['rebase_to_control']:
    print('Setting rebase to first model!! Check that this is the control run.')
    inputs['rebase_to_historical'] = True
    if not inputs['detrend_only_global']:
        print('WARNING!! rebasing to control without removing the global area trend could cause inconsistencies! Suggest setting detrend_only_global to True')

if inputs['rebase_to_historical']:
    if inputs['file_hist_rebase'] is None and not inputs['rebase_to_control']:
        raise ValueError('Set historical filename for rebase in file_hist_rebase or set rebase_to_control to True')

if inputs['filenames'] is None:
    if inputs['filelist'] is None:
        raise ValueError('Set either [filenames] or [filelist]')
    else:
        filo = open(inputs['filelist'], 'r')
        allfi = [fi.strip() for fi in filo.readlines()]
        filo.close()
        inputs['filenames'] = allfi

if len(inputs['filenames']) == 0:
    raise ValueError('No filenames specified. Set either [filenames] or [filelist].')

if inputs['custom_naming_keys'] is not None:
    if inputs['cmip6_naming']:
        print('WARNING! custom_naming_keys specified. Setting cmip6_naming to False')
        inputs['cmip6_naming'] = False

if inputs['model_names'] is None:
    if inputs['is_ensemble'] and not inputs['single_model_ens_list']:
        pass
    elif inputs['cmip6_naming']:
        print('Getting the model names from filenames using CMIP6 convention..\n')
        inputs['model_names'] = []
        for fi in inputs['filenames']:
            cose = fi.split('_')
            model = cose[2]
            member = cose[4]
            print('{} -> {} {}\n'.format(fi, model, member))
            inputs['model_names'].append(model + '_' + member)
    elif inputs['custom_naming_keys'] is not None:
        print('Getting the model names from filenames using custom convention..\n')
        inputs['model_names'] = []
        for fi in inputs['filenames']:
            metadata = ctl.custom_naming(fi, inputs['custom_naming_keys'], seasonal = False)
            print('{} -> {} {}\n'.format(fi, metadata['model'], metadata['member']))
            inputs['model_names'].append(metadata['model'] + '_' + metadata['member'])
    else:
        print('No filenames found, giving standard names...\n')
        n_mod = len(inputs['filenames'])
        inputs['model_names'] = ['ens_{}'.format(i) for i in range(n_mod)]

inputs['ensemble_filenames'] = None
inputs['ensemble_members'] = None

if inputs['single_model_ens_list']:
    print('List of filenames is read as single-model ensemble\n')
    if not inputs['is_ensemble']:
        print('is_ensemble is set to True')
        inputs['is_ensemble'] = True

    if not inputs['cmip6_naming'] and inputs['custom_naming_keys'] is None:
        print('cmip6 naming set to True')
        inputs['cmip6_naming'] = True

    inputs['ensemble_filenames'] = dict()
    inputs['ensemble_members'] = dict()

    mod_name = inputs['model_names'][0].split('_')[0]
    inputs['model_names'] = [mod_name]
    inputs['ensemble_filenames'][mod_name] = [inputs['cart_in']+fil for fil in inputs['filenames']]

    inputs['ensemble_members'][mod_name] = []
    for fi in inputs['filenames']:
        if inputs['cmip6_naming']:
            cose = ctl.cmip6_naming(fi.split('/')[-1], seasonal = inputs['is_seasonal'])
            if inputs['is_seasonal']:
                ens_id = cose['member'] + '_' + cose['sdate'] + '_f' + cose['dates'][0][:4]
            else:
                ens_id = cose['member'] + '_' + cose['dates'][0][:4] + '_' + cose['dates'][1][:4]
        elif inputs['custom_naming_keys'] is not None:
            cose = ctl.custom_naming(fi.split('/')[-1], inputs['custom_naming_keys'], seasonal = True)
            if inputs['is_seasonal']:
                if 'dates' in cose:
                    ens_id = cose['member'] + '_' + cose['sdate'] + '_f' + cose['dates'][0][:4]
                else:
                    ens_id = cose['member'] + '_' + cose['sdate']
            else:
                ens_id = cose['member'] + '_' + cose['dates'][0][:4] + '_' + cose['dates'][1][:4]

        inputs['ensemble_members'][mod_name].append(ens_id)

    inputs['filenames'] = ['gigi']

if np.any(['*' in fil for fil in inputs['filenames']]):
    print('Detected * in filename, setting is_ensemble to True')
    inputs['is_ensemble'] = True

if inputs['is_ensemble']:
    print('This is an ensemble run.')
    if inputs['ensemble_filenames'] is None:
        print('Expanding the * in filenames. Finding all files.. \n')
        inputs['ensemble_filenames'] = dict()
        inputs['ensemble_members'] = dict()
        # load the true file list
        set_model_names = False
        if inputs['model_names'] is None:
            inputs['model_names'] = []
            set_model_names = True

        for ifi, filgenname in enumerate(inputs['filenames']):
            # modcart = inputs['cart_in']
            lista_oks = glob.glob(inputs['cart_in']+filgenname)

            # namfilp = filgenname.split('*')
            # while '' in namfilp: namfilp.remove('')
            # if '/' in filgenname:
            #     modcart = inputs['cart_in'] + '/'.join(filgenname.split('/')[:-1]) + '/'
            #     namfilp = filgenname.split('/')[-1].split('*')
            # while '' in namfilp: namfilp.remove('')

            # print(modcart)
            # lista_all = os.listdir(modcart)
            # lista_oks = [modcart + fi for fi in lista_all if np.all([namp in fi for namp in namfilp])]
            # namfilp.append(modcart)

            if not set_model_names:
                mod_name = inputs['model_names'][ifi]
            else:
                coso = lista_oks[0]
                if inputs['cmip6_naming']:
                    cose = ctl.cmip6_naming(coso.split('/')[-1], seasonal = inputs['is_seasonal'])
                elif inputs['custom_naming_keys'] is not None:
                    cose = ctl.custom_naming(coso.split('/')[-1], inputs['custom_naming_keys'], seasonal = inputs['is_seasonal'])
                else:
                    raise ValueError('specify model names or naming standard')

                if inputs['is_seasonal']:
                    mod_name = cose['model']
                else:
                    mod_name = cose['model'] + '_' + cose['member']

                inputs['model_names'].append(mod_name)

            inputs['ensemble_filenames'][mod_name] = list(np.sort(lista_oks))
            inputs['ensemble_members'][mod_name] = []
            for coso in np.sort(lista_oks):
                if inputs['cmip6_naming']:
                    cose = ctl.cmip6_naming(coso.split('/')[-1], seasonal = inputs['is_seasonal'])
                    if inputs['is_seasonal']:
                        ens_id = cose['member'] + '_' + cose['sdate'] + '_f' + cose['dates'][0][:4]
                    else:
                        ens_id = cose['member'] + '_' + cose['dates'][0][:4] + '_' + cose['dates'][1][:4]
                elif inputs['custom_naming_keys'] is not None:
                    cose = ctl.custom_naming(coso.split('/')[-1], inputs['custom_naming_keys'], seasonal = inputs['is_seasonal'])
                    if inputs['is_seasonal']:
                        if 'dates' in cose:
                            ens_id = cose['member'] + '_' + cose['sdate'] + '_f' + cose['dates'][0][:4]
                        else:
                            ens_id = cose['member'] + '_' + cose['sdate']
                    else:
                        ens_id = cose['member'] + '_' + cose['dates'][0][:4] + '_' + cose['dates'][1][:4]
                else:
                    print('WARNING! No naming rule for ensemble. Setting filenames as ens_id.')
                    # for namp in namfilp:
                    #     coso = coso.replace(namp,'###')
                    # ens_id = '_'.join(coso.strip('###').split('###'))
                    ens_id = coso.split('/')[-1].strip('.nc')

                inputs['ensemble_members'][mod_name].append(ens_id)
            print(mod_name, inputs['ensemble_filenames'][mod_name])
            print(mod_name, inputs['ensemble_members'][mod_name])


print('filenames: ', inputs['filenames'])
print('model names: ', inputs['model_names'])

if inputs['matching_hierarchy'] is not None:
    if len(inputs['matching_hierarchy']) != inputs['numclus']:
        raise ValueError('length of matching_hierarchy should be equal to numclus')
    inputs['matching_hierarchy'] = list(map(int, inputs['matching_hierarchy']))

if not inputs['run_compare'] and len(inputs['model_names']) > 1:
    raise ValueError('Multiple models selected. Set a reference file for the observations and set run_compare to True\n')

if inputs['supervised_clustering']:
    print('WARNING!!!!! supervised_clustering is active with frac_super = {:5.2e}\n'.format(inputs['frac_super']))
    if not inputs['use_reference_eofs']:
        inputs['use_reference_eofs'] = True
        print('supervised_clustering active. Setting use_reference_eofs to True')
    if inputs['use_reference_clusters']:
        raise ValueError('supervised_clustering is incompatible with use_reference_clusters')

print(inputs['groups'])
if inputs['group_symbols'] is not None:
    for k in inputs['group_symbols']:
        inputs['group_symbols'][k] = inputs['group_symbols'][k][0]
print(inputs['group_symbols'])

if not inputs['flag_perc']:
    print('Considering fixed numpcs = {}\n'.format(inputs['numpcs']))
    inputs['perc'] = None
else:
    print('Considering pcs to explain {}% of variance\n'.format(inputs['perc']))

if inputs['groups'] is not None and inputs['reference_group'] is None:
    print('Setting default reference group to: {}\n'.format(list(inputs['groups'].keys())[0]))
    inputs['reference_group'] = list(inputs['groups'].keys())[0]

if not os.path.exists(inputs['cart_out_general']): os.mkdir(inputs['cart_out_general'])
inputs['cart_out'] = inputs['cart_out_general'] + inputs['exp_name'] + '/'
if not os.path.exists(inputs['cart_out']): os.mkdir(inputs['cart_out'])

if inputs['year_range'] is not None:
    inputs['year_range'] = list(map(int, inputs['year_range']))

if inputs['ref_year_range'] is not None:
    inputs['ref_year_range'] = list(map(int, inputs['ref_year_range']))

# dictgrp = dict()
# dictgrp['all'] = inputs['dictgroup']
# inputs['dictgroup'] = dictgrp

if inputs['area'] == 'custom':
    if inputs['custom_area'] is None:
        raise ValueError('Set custom_area or specify a default area')
    else:
        inputs['custom_area'] = list(map(float, inputs['custom_area']))
        area = inputs['custom_area']
else:
    area = inputs['area']

if inputs['multiple_areas'] is None:
    inputs['multiple_areas'] = [area]
if inputs['multiple_seasons'] is None:
    inputs['multiple_seasons'] = [inputs['season']]

allinds = list(itt.product(np.arange(len(inputs['multiple_areas'])), np.arange(len(inputs['multiple_seasons']))))
inputs['multiple_area_season'] = [(inputs['multiple_areas'][iuu], inputs['multiple_seasons'][juu]) for (iuu,juu) in allinds]

print('-------------------------------------------------\n')
print('Beginning cycle on multiple areas and seasons.\n') # This is not done in a very efficient way, data are read each time
for (area, season) in inputs['multiple_area_season']:
    print('-------------------------------------------------\n')
    print('Area: {}, season {}\n'.format(area, season))
    print('-------------------------------------------------\n')

    if inputs['area'] != 'custom':
        inputs['area'] = area
    inputs['season'] = season

    if inputs['area'] == 'EAT' and inputs['numclus'] == 4:
        if inputs['patnames'] is None:
            inputs['patnames'] = ['NAO +', 'Sc. Blocking', 'Atl. Ridge', 'NAO -']
        if inputs['patnames_short'] is None:
            inputs['patnames_short'] = ['NP', 'BL', 'AR', 'NN']
    elif inputs['area'] == 'PNA' and inputs['numclus'] == 4:
        if inputs['patnames'] is None:
            inputs['patnames'] = ['Ala. Ridge', 'Pac. Trough', 'Arctic Low', 'Arctic High']
        if inputs['patnames_short'] is None:
            inputs['patnames_short'] = ['AR', 'PT', 'AL', 'AH']

    if inputs['patnames'] is None:
        inputs['patnames'] = ['clus_{}'.format(i) for i in range(inputs['numclus'])]
    if inputs['patnames_short'] is None:
        inputs['patnames_short'] = ['c{}'.format(i) for i in range(inputs['numclus'])]

    print('patnames', inputs['patnames'])
    print('patnames_short', inputs['patnames_short'])

    if inputs['plot_margins'] is not None:
        inputs['plot_margins'] = list(map(float, inputs['plot_margins']))

    outname = 'out_' + std_outname(inputs['exp_name'], inputs) + '.p'
    nomeout = inputs['cart_out'] + outname

    if inputs['ref_year_range'] is None:
        inputs['ref_year_range'] = inputs['year_range']

    if (inputs['detrend_only_global'] or inputs['detrend_local_linear']) and not inputs['remove_29feb']:
        inputs['remove_29feb'] = True
        print('WARNING!!! Key <remove_29feb> has been changed to True. Removing 29 Feb is needed to calculate and remove global trends.')

    if inputs['run_compare']:
        ERA_ref_out = inputs['cart_out_general'] + inputs['ERA_ref_folder'] + 'out_' + std_outname(inputs['obs_name'], inputs, ref_name = True) + '.p'
        if not os.path.exists(inputs['cart_out_general'] + inputs['ERA_ref_folder']):
            os.mkdir(inputs['cart_out_general'] + inputs['ERA_ref_folder'])

    if not os.path.exists(nomeout):
        print('{} not found, this is the first run. Setting up the computation..\n'.format(nomeout))

        if inputs['run_compare']:
            if (not os.path.exists(ERA_ref_out)) or (inputs['ref_year_range'] is None):
                ERA_ref = cd.WRtool_from_file(inputs['ERA_ref_orig'], inputs['season'], area, extract_level_hPa = inputs['level'], numclus = inputs['numclus'], heavy_output = True, run_significance_calc = inputs['run_sig_calc'], sel_yr_range = inputs['ref_year_range'], numpcs = inputs['numpcs'], perc = inputs['perc'], netcdf4_read = inputs['netcdf4_read'], wnd_days = inputs['wnd_days'], area_dtr = inputs['area_dtr'], detrend_only_global = inputs['detrend_only_global'], remove_29feb = inputs['remove_29feb'], pressure_levels = inputs['pressure_levels'], calc_gradient = inputs['calc_gradient'], deg_dtr = inputs['deg_dtr'], detrend_local_linear = inputs['detrend_local_linear'])
                pickle.dump(ERA_ref, open(ERA_ref_out, 'wb'))
            else:
                print('Reference calculation already performed, reading from {}\n'.format(ERA_ref_out))
                ERA_ref = pickle.load(open(ERA_ref_out, 'rb'))

            ref_solver = ERA_ref['solver']
            ref_patterns_area = ERA_ref['cluspattern_area']
            ref_clusters_centers = ERA_ref['centroids']
        else:
            ERA_ref = None
            ref_solver = None
            ref_patterns_area = None
            ref_clusters_centers = None
            if inputs['use_reference_eofs']:
                print('No comparison with observations. Setting use_reference_eofs to False.\n')
                inputs['use_reference_eofs'] = False
            if inputs['use_reference_clusters']:
                print('No comparison with observations. Setting use_reference_clusters to False.\n')
                inputs['use_reference_clusters'] = False

        if inputs['regrid_model_data']:
            print('Regrid of model data to reference grid is set to True. Loading reference cube..')
            if inputs['netcdf4_read']:
                inputs['netcdf4_read'] = False
                print('<netcdf4_read> set to False.')
            if inputs['iris_read']:
                ref_cube = iris.load(inputs['ERA_ref_orig'])[0]
            else:
                ref_cube = xr.load_dataset(inputs['ERA_ref_orig'])
        else:
            ref_cube = None

        if inputs['rebase_to_historical']:
            if inputs['file_hist_rebase'] is not None:
                print('Loading historical climate mean from {}\n'.format(inputs['file_hist_rebase']))
                clim_rebase = dict()
                dates_clim_rebase = dict()
                results_hist, _ = ctl.load_wrtool(inputs['file_hist_rebase'])
                if inputs['cmip6_naming']:
                    # Per consistenza uso sempre il primo member dello storico
                    okmodhist = np.unique([ke.split('_')[0] for ke in results_hist])
                    for mod in okmodhist:
                        okke = [ke for ke in results_hist if ke.split('_')[0] == mod]
                        if np.any(['r1i' in ke for ke in okke]):
                            ke = [kee for kee in okke if 'r1i' in kee][0]
                        else:
                            ke = okke[0]
                        clim_rebase[mod] = results_hist[ke]['climate_mean']
                        dates_clim_rebase[mod] = results_hist[ke]['climate_mean_dates']
                else:
                    for ke in results_hist:
                        clim_rebase[ke] = results_hist[ke]['climate_mean']
                        dates_clim_rebase[ke] = results_hist[ke]['climate_mean_dates']
                del results_hist

        model_outs = dict()
        for iii, (modfile, modname) in enumerate(zip(inputs['filenames'], inputs['model_names'])):
            if not inputs['is_ensemble']:
                filin = inputs['cart_in']+modfile
            else:
                filin = inputs['ensemble_filenames'][modname]

            if inputs['rebase_to_historical']:
                if inputs['rebase_to_control']:
                    if iii == 0:
                        print('This is the control run: {}. Calculating the climate mean'.format(inputs['model_names'][0]))
                        climate_mean = None
                        dates_climate_mean = None
                    else:
                        print('Setting climate_mean to that of {}'.format(inputs['model_names'][0]))
                        climate_mean = model_outs[inputs['model_names'][0]]['climate_mean']
                        dates_climate_mean = model_outs[inputs['model_names'][0]]['climate_mean_dates']
                elif inputs['file_hist_rebase'] is not None:
                    if inputs['cmip6_naming']:
                        mod = modname.split('_')[0]
                    else:
                        mod = modname

                    if mod in clim_rebase.keys():
                        climate_mean = clim_rebase[mod]
                        dates_climate_mean = dates_clim_rebase[mod]
                    else:
                        print('{} not found in historical runs\n'.format(mod))
                        continue
            else:
                climate_mean = None
                dates_climate_mean = None

            read_data_from_p = None
            write_data_to_p = None
            if inputs['dump_regridded']:
                filraw = inputs['cart_out'] + 'rawdata_' + inputs['exp_name'] + '_' + modname + '.p'
                if os.path.exists(filraw):
                    print('Raw data exists. Reading from ' + filraw)
                    read_data_from_p = open(filraw, 'rb')
                else:
                    print('Raw data is not there. This is first run, writing to ' + filraw)
                    print('WARNING!! This could take considerable space on disk')
                    write_data_to_p = open(filraw, 'wb')

            try:
                model_outs[modname] = cd.WRtool_from_file(filin, inputs['season'], area, extract_level_hPa = inputs['level'], regrid_to_reference_cube = ref_cube, numclus = inputs['numclus'], heavy_output = inputs['heavy_output'], run_significance_calc = inputs['run_sig_calc'], ref_solver = ref_solver, ref_patterns_area = ref_patterns_area, sel_yr_range = inputs['year_range'], numpcs = inputs['numpcs'], perc = inputs['perc'], use_reference_eofs = inputs['use_reference_eofs'], use_reference_clusters = inputs['use_reference_clusters'], ref_clusters_centers = ref_clusters_centers, netcdf4_read = inputs['netcdf4_read'], wnd_days = inputs['wnd_days'], bad_matching_rule = inputs['bad_matching_rule'], matching_hierarchy = inputs['matching_hierarchy'], area_dtr = inputs['area_dtr'], detrend_only_global = inputs['detrend_only_global'], remove_29feb = inputs['remove_29feb'], pressure_levels = inputs['pressure_levels'], calc_gradient = inputs['calc_gradient'], supervised_clustering = inputs['supervised_clustering'], frac_super = inputs['frac_super'], select_area_first = inputs['select_area_first'], deg_dtr = inputs['deg_dtr'], detrend_local_linear = inputs['detrend_local_linear'], rebase_to_historical = inputs['rebase_to_historical'], climate_mean = climate_mean, dates_climate_mean = dates_climate_mean, read_from_p = read_data_from_p, write_to_p = write_data_to_p)
            except Exception as exc:
                if inputs['ignore_model_error']:
                    print('\n\n\n WARNING!!! EXCEPTION FOUND WHEN RUNNING MODEL {}: {}\n\n\n'.format(modname, exc))
                    continue
                else:
                    raise exc

            if read_data_from_p is not None: read_data_from_p.close()
            if write_data_to_p is not None: write_data_to_p.close()

        if len(list(model_outs.keys())) == 0:
            raise ValueError('NO MODEL WAS RUN. CHECK LOG FILE')

        if inputs['is_ensemble'] and inputs['is_seasonal']:
            print('This is a seasonal ensemble run, splitting up individual members..')
            model_outs = cd.extract_ensemble_results(model_outs, inputs['ensemble_members'])

        restot = dict()
        if ERA_ref is not None:
            ERA_ref_light = copy(ERA_ref)
            del ERA_ref_light['solver']
            del ERA_ref_light['var_area']
            del ERA_ref_light['var_glob']
            del ERA_ref_light['regime_transition_pcs']
            restot['reference'] = ERA_ref_light
        else:
            restot['reference'] = None
        restot['models'] = model_outs

        pickle.dump(restot, open(nomeout, 'wb'))

        try:
            json_res = copy(model_outs)
            json_res['reference'] = copy(ERA_ref)
            json_res = cd.export_results_to_json(nomeout[:-2]+'.json', json_res)
            io.savemat(nomeout[:-2]+'.mat', mdict = json_res)
            del json_res
        except:
            pass
    else:
        print('Computation already performed. Reading output from {}\n'.format(nomeout))
        restot = pickle.load(open(nomeout, 'rb'))
        if type(restot) == dict:
            model_outs = restot['models']
            ERA_ref = restot['reference']
        else: # for backward Compatibility
            model_outs, ERA_ref = restot

        if DEBUG:
            json_res = copy(model_outs)
            json_res['reference'] = ERA_ref
            json_res = cd.export_results_to_json(nomeout[:-2]+'.json', json_res)

    os.system('cp {} {}'.format(file_input, inputs['cart_out'] + std_outname(inputs['exp_name'], inputs) + '.in'))

    if inputs['area'] == 'custom':
        arearect = inputs['custom_area']
    else:
        arearect = ctl.sel_area_translate(inputs['area'])
    arearect[0] = ctl.convert_lon_std(arearect[0])
    arearect[1] = ctl.convert_lon_std(arearect[1])

    mod_out = model_outs[inputs['model_names'][0]]
    if inputs['central_lon'] is None:
        lonW, lonE, latS, latN = arearect
        if lonW < lonE:
            inputs['central_lon'] = (lonW+lonE)/2
        else:
            inputs['central_lon'] = ctl.convert_lon_std((lonW + lonE + 360)/2)
        # print(mod_out['lon_area'])
        # print(np.mean(mod_out['lon_area']))
        # inputs['central_lon'] = np.mean(mod_out['lon_area'])
    if inputs['central_lat'] is None:
        inputs['central_lat'] = np.mean(mod_out['lat_area'])
    print('Central lat, lon: {}, {}\n'.format(inputs['central_lat'], inputs['central_lon']))

    n_models = len(model_outs.keys())

    file_res = inputs['cart_out'] + 'results_' + std_outname(inputs['exp_name'], inputs) + '.dat'
    cd.out_WRtool_mainres(file_res, model_outs, ERA_ref, inputs)

    if not inputs['draw_rectangle_area']: arearect = None

    if inputs['out_figures']:
        if inputs['run_compare']:
            cd.plot_WRtool_results(inputs['cart_out'], std_outname(inputs['exp_name'], inputs), n_models, model_outs, ERA_ref, model_names = inputs['model_names'], obs_name = inputs['obs_name'], patnames = inputs['patnames'], patnames_short = inputs['patnames_short'], central_lat_lon = (inputs['central_lat'], inputs['central_lon']), groups = inputs['groups'], group_symbols = inputs['group_symbols'], reference_group = inputs['reference_group'], visualization = inputs['visualization'], bounding_lat = inputs['bounding_lat'], plot_margins = inputs['plot_margins'], draw_rectangle_area = arearect, taylor_mark_dim = inputs['taylor_mark_dim'], out_only_main_figs = inputs['out_only_main_figs'], use_seaborn = inputs['use_seaborn'], color_palette = inputs['color_palette'], show_transitions = inputs['show_transitions'], draw_grid = inputs['draw_grid'], plot_type = inputs['plot_type'], cb_label = inputs['plot_cb_label'])#, custom_model_colors = ['indianred', 'forestgreen', 'black'], compare_models = [('stoc', 'base')])
        else:
            cd.plot_WRtool_singlemodel(inputs['cart_out'], std_outname(inputs['exp_name'], inputs), mod_out, model_name = inputs['model_names'][0], patnames = inputs['patnames'], patnames_short = inputs['patnames_short'], central_lat_lon = (inputs['central_lat'], inputs['central_lon']), visualization = inputs['visualization'], bounding_lat = inputs['bounding_lat'], plot_margins = inputs['plot_margins'], draw_rectangle_area = arearect, taylor_mark_dim = inputs['taylor_mark_dim'], use_seaborn = inputs['use_seaborn'], color_palette = inputs['color_palette'], show_transitions = inputs['show_transitions'], draw_grid = inputs['draw_grid'], plot_type = inputs['plot_type'], cb_label = inputs['plot_cb_label'])

    if inputs['out_netcdf']:
        cart_out_nc = inputs['cart_out'] + 'outnc_' + std_outname(inputs['exp_name'], inputs) + '/'
        if not os.path.exists(cart_out_nc): os.mkdir(cart_out_nc)
        if inputs['is_ensemble'] and inputs['is_seasonal']:
            cd.out_WRtool_netcdf_ensemble(cart_out_nc, model_outs, ERA_ref, inputs, var_long_name = inputs['var_long_name'], var_std_name = inputs['var_std_name'], var_units = inputs['var_units'])
        else:
            cd.out_WRtool_netcdf(cart_out_nc, model_outs, ERA_ref, inputs)

    print('Completed area {}, season {}!\n'.format(area, season))

print('Check results in directory: {}\n'.format(inputs['cart_out']))
print(ctl.datestamp()+'\n')
print('Ended successfully!\n')

os.system('cp {} {}'.format(logname, inputs['cart_out']))
logfile.close()
