#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import sys
import os

import matplotlib
matplotlib.use('Agg') # This is to avoid the code crash if no Xwindow is available
from matplotlib import pyplot as plt

import pickle
from scipy import io

import climtools_lib as ctl
import climdiags as cd

import warnings
warnings.simplefilter('default')
warnings.filterwarnings('default', category=DeprecationWarning)

#######################################
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14

def std_outname(tag, inputs, ref_name = False):
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
            name_outputs += '_refEOF'
    else:
        if inputs['ref_year_range'] is not None:
            name_outputs += '_{}-{}'.format(inputs['ref_year_range'][0], inputs['ref_year_range'][1])
        else:
            name_outputs += '_allyrs'


    if inputs['detrended_anom_for_clustering'] or inputs['detrend_only_global']:
        name_outputs += '_dtr'

    return name_outputs

if np.any(['log_WRtool_' in cos for cos in os.listdir('.')]):
    os.system('rm log_WRtool_*log')

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

keys = 'exp_name cart_in cart_out_general filenames model_names level season area numclus numpcs flag_perc perc ERA_ref_orig ERA_ref_folder run_sig_calc run_compare patnames patnames_short heavy_output model_tags year_range groups group_symbols reference_group detrended_eof_calculation detrended_anom_for_clustering use_reference_eofs obs_name filelist visualization bounding_lat plot_margins custom_area is_ensemble ens_option draw_rectangle_area use_reference_clusters out_netcdf out_figures out_only_main_figs taylor_mark_dim starred_field_names use_seaborn color_palette netcdf4_read ref_clus_order_file wnd_days wnd_years show_transitions central_lat central_lon draw_grid cmip6_naming bad_matching_rule matching_hierarchy ref_year_range area_dtr detrend_only_global'
keys = keys.split()
itype = [str, str, str, list, list, float, str, str, int, int, bool, float, str, str, bool, bool, list, list, bool, list, list, dict, dict, str, bool, bool, bool, str, str, str, float, list, list, bool, str, bool, bool, bool, bool, bool, int, list, bool, str, bool, str, int, int, bool, float, float, bool, bool, str, list, list, str, bool]

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
defaults['detrended_anom_for_clustering'] = False
defaults['detrended_eof_calculation'] = False
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
defaults['wnd_years'] = 30
defaults['show_transitions'] = False
defaults['draw_grid'] = True
defaults['cmip6_naming'] = False
defaults['bad_matching_rule'] = 'rms_mean'
defaults['matching_hierarchy'] = None
defaults['area_dtr'] = 'global'
defaults['detrend_only_global'] = False


inputs = ctl.read_inputs(file_input, keys, n_lines = None, itype = itype, defaults = defaults)
for ke in inputs:
    print('{} : {}\n'.format(ke, inputs[ke]))

if inputs['area'] == 'custom':
    if inputs['custom_area'] is None:
        raise ValueError('Set custom_area or specify a default area')
    else:
        inputs['custom_area'] = list(map(float, inputs['custom_area']))
        area = inputs['custom_area']
else:
    area = inputs['area']

if inputs['cart_in'][-1] != '/': inputs['cart_in'] += '/'
if inputs['cart_out_general'][-1] != '/': inputs['cart_out_general'] += '/'
if inputs['ERA_ref_folder'][-1] != '/': inputs['ERA_ref_folder'] += '/'

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

if inputs['model_names'] is None:
    if inputs['cmip6_naming']:
        print('Getting the model names from filenames using CMIP6 convention..\n')
        inputs['model_names'] = []
        for fi in inputs['filenames']:
            cose = fi.split('_')
            model = cose[2]
            member = cose[4]
            print('{} -> {} {}\n'.format(fi, model, member))
            inputs['model_names'].append(model + '_' + member)
    else:
        print('No filenames found, giving standard names...\n')
        n_mod = len(inputs['filenames'])
        inputs['model_names'] = ['ens_{}'.format(i) for i in range(n_mod)]

if inputs['is_ensemble']:
    print('This is an ensemble run, finding all files.. \n')
    inputs['ensemble_filenames'] = dict()
    inputs['ensemble_members'] = dict()
    # load the true file list
    for filgenname, mod_name in zip(inputs['filenames'], inputs['model_names']):
        modcart = inputs['cart_in']
        namfilp = filgenname.split('*')
        while '' in namfilp: namfilp.remove('')
        if '/' in filgenname:
            modcart = inputs['cart_in'] + '/'.join(filgenname.split('/')[:-1]) + '/'
            namfilp = filgenname.split('/')[-1].split('*')
        while '' in namfilp: namfilp.remove('')

        lista_all = os.listdir(modcart)
        lista_oks = [modcart + fi for fi in lista_all if np.all([namp in fi for namp in namfilp])]
        namfilp.append(modcart)

        if inputs['ens_option'] == 'all':
            inputs['ensemble_filenames'][mod_name] = list(np.sort(lista_oks))
            inputs['ensemble_members'][mod_name] = []
            for coso in np.sort(lista_oks):
                for namp in namfilp:
                    coso = coso.replace(namp,'###')
                ens_id = '_'.join(coso.strip('###').split())
                inputs['ensemble_members'][mod_name].append(ens_id)
            print(mod_name, inputs['ensemble_filenames'][mod_name])
            print(mod_name, inputs['ensemble_members'][mod_name])

        if inputs['ens_option'] == 'member' or inputs['ens_option'] == 'year':
            raise ValueError('NOT IMPLEMENTED')
            # mem_field = inputs['starred_field_names'].index('member')
            # allfiles = list(np.sort(lista_oks))
            # all_ids = []
            # all_mems = []
            # for coso in allfiles:
            #     for namp in namfilp:
            #         coso = coso.replace(namp,' ')
            #     ens_id = '_'.join(coso.strip().split())
            #     all_ids.append(ens_id)
            #     all_mems.append(coso.strip().split()[mem_field])
            #
            # if inputs['groups'] is not None:
            #     raise ValueError('ens_option == member and groups specified are not compatible')
            # else:
            #     inputs['groups'] = dict()
            #
            # inputs['groups'][mod_name] = []
            # for mem in np.sort(np.unique(all_mems)):
            #     inputs['model_names']
            #     inputs['ensemble_filenames'][mod_name]
            #     inputs['ensemble_members'][mod_name]
            #     inputs['groups'][mod_name] = []


print('filenames: ', inputs['filenames'])
print('model names: ', inputs['model_names'])

if inputs['matching_hierarchy'] is not None:
    if len(inputs['matching_hierarchy']) != inputs['numclus']:
        raise ValueError('length of matching_hierarchy should be equal to numclus')
    inputs['matching_hierarchy'] = list(map(int, inputs['matching_hierarchy']))

if not inputs['run_compare'] and len(inputs['model_names']) > 1:
    raise ValueError('Multiple models selected. Set a reference file for the observations and set run_compare to True\n')

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

if inputs['run_compare']:
    ERA_ref_out = inputs['cart_out_general'] + inputs['ERA_ref_folder'] + 'out_' + std_outname(inputs['obs_name'], inputs, ref_name = True) + '.p'
    if not os.path.exists(inputs['cart_out_general'] + inputs['ERA_ref_folder']):
        os.mkdir(inputs['cart_out_general'] + inputs['ERA_ref_folder'])

if not os.path.exists(nomeout):
    print('{} not found, this is the first run. Setting up the computation..\n'.format(nomeout))

    if inputs['run_compare']:
        if (not os.path.exists(ERA_ref_out)) or (inputs['ref_year_range'] is None):
            ERA_ref = cd.WRtool_from_file(inputs['ERA_ref_orig'], inputs['season'], area, extract_level_hPa = inputs['level'], numclus = inputs['numclus'], heavy_output = True, run_significance_calc = inputs['run_sig_calc'], sel_yr_range = inputs['ref_year_range'], numpcs = inputs['numpcs'], perc = inputs['perc'], detrended_eof_calculation = inputs['detrended_eof_calculation'], detrended_anom_for_clustering = inputs['detrended_anom_for_clustering'], netcdf4_read = inputs['netcdf4_read'], wnd_days = inputs['wnd_days'], wnd_years = inputs['wnd_years'], area_dtr = inputs['area_dtr'], detrend_only_global = inputs['detrend_only_global'])
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

    model_outs = dict()
    for modfile, modname in zip(inputs['filenames'], inputs['model_names']):
        if not inputs['is_ensemble']:
            filin = inputs['cart_in']+modfile
        else:
            filin = inputs['ensemble_filenames'][modname]

        model_outs[modname] = cd.WRtool_from_file(filin, inputs['season'], area, extract_level_hPa = inputs['level'], numclus = inputs['numclus'], heavy_output = inputs['heavy_output'], run_significance_calc = inputs['run_sig_calc'], ref_solver = ref_solver, ref_patterns_area = ref_patterns_area, sel_yr_range = inputs['year_range'], numpcs = inputs['numpcs'], perc = inputs['perc'], detrended_eof_calculation = inputs['detrended_eof_calculation'], detrended_anom_for_clustering = inputs['detrended_anom_for_clustering'], use_reference_eofs = inputs['use_reference_eofs'], use_reference_clusters = inputs['use_reference_clusters'], ref_clusters_centers = ref_clusters_centers, netcdf4_read = inputs['netcdf4_read'], wnd_days = inputs['wnd_days'], wnd_years = inputs['wnd_years'], bad_matching_rule = inputs['bad_matching_rule'], matching_hierarchy = inputs['matching_hierarchy'], area_dtr = inputs['area_dtr'], detrend_only_global = inputs['detrend_only_global'])
        # else:
        #     model_outs[modname] = cd.WRtool_from_file(inputs['ensemble_filenames'][modname], inputs['season'], area, extract_level_hPa = inputs['level'], numclus = inputs['numclus'], heavy_output = inputs['heavy_output'], run_significance_calc = inputs['run_sig_calc'], ref_solver = ERA_ref['solver'], ref_patterns_area = ERA_ref['cluspattern_area'], sel_yr_range = inputs['year_range'], numpcs = inputs['numpcs'], perc = inputs['perc'], detrended_eof_calculation = inputs['detrended_eof_calculation'], detrended_anom_for_clustering = inputs['detrended_anom_for_clustering'], use_reference_eofs = inputs['use_reference_eofs'], use_reference_clusters = inputs['use_reference_clusters'], ref_clusters_centers = ERA_ref['centroids'], netcdf4_read = inputs['netcdf4_read'])

    pickle.dump([model_outs, ERA_ref], open(nomeout, 'wb'))
    try:
        io.savemat(nomeout[:-2]+'.mat', mdict = {'models': model_outs, 'reference': ERA_ref})
    except Exception as caos:
        print(repr(caos))
        print('Unable to produce .mat OUTPUT!!')
else:
    print('Computation already performed. Reading output from {}\n'.format(nomeout))
    [model_outs, ERA_ref] = pickle.load(open(nomeout, 'rb'))

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
        cd.plot_WRtool_results(inputs['cart_out'], std_outname(inputs['exp_name'], inputs), n_models, model_outs, ERA_ref, model_names = inputs['model_names'], obs_name = inputs['obs_name'], patnames = inputs['patnames'], patnames_short = inputs['patnames_short'], central_lat_lon = (inputs['central_lat'], inputs['central_lon']), groups = inputs['groups'], group_symbols = inputs['group_symbols'], reference_group = inputs['reference_group'], visualization = inputs['visualization'], bounding_lat = inputs['bounding_lat'], plot_margins = inputs['plot_margins'], draw_rectangle_area = arearect, taylor_mark_dim = inputs['taylor_mark_dim'], out_only_main_figs = inputs['out_only_main_figs'], use_seaborn = inputs['use_seaborn'], color_palette = inputs['color_palette'], show_transitions = inputs['show_transitions'], draw_grid = inputs['draw_grid'])#, custom_model_colors = ['indianred', 'forestgreen', 'black'], compare_models = [('stoc', 'base')])
    else:
        cd.plot_WRtool_singlemodel(inputs['cart_out'], std_outname(inputs['exp_name'], inputs), mod_out, model_name = inputs['model_names'][0], patnames = inputs['patnames'], patnames_short = inputs['patnames_short'], central_lat_lon = (inputs['central_lat'], inputs['central_lon']), visualization = inputs['visualization'], bounding_lat = inputs['bounding_lat'], plot_margins = inputs['plot_margins'], draw_rectangle_area = arearect, taylor_mark_dim = inputs['taylor_mark_dim'], use_seaborn = inputs['use_seaborn'], color_palette = inputs['color_palette'], show_transitions = inputs['show_transitions'], draw_grid = inputs['draw_grid'])

if inputs['out_netcdf']:
    cart_out_nc = inputs['cart_out'] + 'outnc_' + std_outname(inputs['exp_name'], inputs) + '/'
    if not os.path.exists(cart_out_nc): os.mkdir(cart_out_nc)
    cd.out_WRtool_netcdf(cart_out_nc, model_outs, ERA_ref, inputs)

print('Check results in directory: {}\n'.format(inputs['cart_out']))
print(ctl.datestamp()+'\n')
print('Ended successfully!\n')

os.system('cp {} {}'.format(logname, inputs['cart_out']))
logfile.close()
