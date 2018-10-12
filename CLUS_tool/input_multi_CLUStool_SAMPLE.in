Input file for run_CLUS_tool.py.

IMPORTANT NOTE: the value of the keys has to be written in the line that immediately follows the [key] label. If you don't want to set a key, comment it with a '#': #[key]. Commented lines are not considered.

NOTE (Irene): Before the weather regime analysis, data should be remapped r144x73 and the period of interest should be selected: run arrange_inputdata.sh

##################################################################
##################### REQUIRED INPUT PATHS/NAMES (have to be set!) #################

# Directory of the WRtool
[dir_WRtool]
/home/fabiano/Research/git/WRtool/CLUS_tool/

# Output data directory
[dir_OUTPUT]
/home/fabiano/Research/lavori/WeatherRegimes/OUT_WRTOOL/

# Name of this post-processing run
[exp_name]
HR_1, HR_2, HR_3, HR_4, LR_1, LR_2, LR_3, LR_4, LR_5, LR_6

# Name of the dataset (ECEARTH, ERA, NCEP)
[model_name]
ECMWF-IFS, ECMWF-IFS, ECMWF-IFS, ECMWF-IFS, ECMWF-IFS, ECMWF-IFS, ECMWF-IFS, ECMWF-IFS, ECMWF-IFS, ECMWF-IFS

# Input data directory
[dir_INPUT]
/home/fabiano/DATA/PRIMAVERA_Stream1_Z500remap/

# Input file names (files to be analyzed). If more than one file, list them in the same line, separated by commas.
[filenames]


##########################################################################
##############  reference FILES  ################

# Compare with ERA and NCEP?
[run_compare]
True

# ERA reference file (used to compare the cluster patterns).
# The program checks for existing EOF solver and clusters in ERA_ref_folder. If not found with the right number of clusters, the original file is processed in the same folder.

[ERA_ref_orig]
/home/fabiano/DATA/PRIMAVERA_Stream1_Z500remap/zg500_ERAInterim_1979-2014.nc

[ERA_ref_folder]
/home/fabiano/Research/lavori/WeatherRegimes/OUT_WRTOOL/ERA_ref_1979-2014/

# NCEP reference file (used to have a measure of the minimum error in the clustering).
# The program checks for existing EOF solver and clusters based on standard names. If not found the original file is processed.

[NCEP_ref_orig]
/home/fabiano/DATA/PRIMAVERA_Stream1_Z500remap/zg500_Aday_NCEPNCAR_2deg_1979-2014.nc

[NCEP_ref_folder]
/home/fabiano/Research/lavori/WeatherRegimes/OUT_WRTOOL/NCEP_ref_1979-2014/

##########################################################################
##############  options for EOFs/CLUSTERING  ################

# Number of EOFs to be used in the decomposition:
[numpcs]
4

# Number of clusters to be used:
[numclus]
4

# Calculate significance of clustering (run cluster_sig)
[check_sig]
True

##########################################################################
##############  options for PRECOMPUTE -- preprocessing  ################

# Name of the variable to be extracted from the fields
[varname]
zg

# Atmospheric level at which the variable is extracted (if more levels are present)
[level]
500

# Season to be selected (options: JJA, DJF, DJFM, NDJFM)
[season]
DJF

# Regional average ('EAT': Euro-Atlantic, 'PNA': Pacific North American, 'NH': Northern Hemisphere)
# Area to be selected (options: EAT, PNA, NH)
[area]
EAT

# Number of days for the running mean window (default = 5)
[wnd]
5

# Data frequency (options: day, mon, year)
[freq]
day

##########################################################################
###################  other options  ###################


# Number of ensemble member to be extracted. Comment this line with a #, if you don't want to set it. (default: all)
# [enstoselect]
# 0

# Spatial resolution of data (just for the output names)
[resolution]
144x73

# Start year (just for the output names)
[start_year]
1979

# End year (just for the output names)
[end_year]
2014
