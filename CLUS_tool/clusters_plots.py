#********************************
#        clusters_plot          *
#********************************

# Standard packages
#from netCDF4 import Dataset, num2date, datetime
import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
import os
#from mpl_toolkits.basemap import Basemap
import cartopy.crs as ccrs
import cartopy.feature as cfeature

print('**************************************************************')
print('Running {0}'.format(sys.argv[0]))
print('**************************************************************')
dir_WRtool    = sys.argv[1]  # WRTOOL directory
dir_OUTPUT    = sys.argv[2]  # OUTPUT directory
name_outputs  = sys.argv[3]  # name of the outputs
numens        = int(sys.argv[4])  # total number of ensemble members
numpcs        = sys.argv[5]  # number of principal components
perc          = sys.argv[6]  # percentage of explained variance for EOF analysis
enstoselect   = sys.argv[7]  # ensemble member to analyse
numclus       = int(sys.argv[8])  # number of clusters

if numpcs!='no':
    numpcs=int(sys.argv[5])
if perc!='no':
    perc=int(sys.argv[6])
if (perc=='no' and numpcs=='no') or (perc!='no' and numpcs!='no'):
    raise ValueError('You have to specify either "perc" or "numpcs".')
if enstoselect!='no':
    enstoselect=int(sys.argv[7])

# User-defined libraries
sys.path.insert(0,dir_WRtool+'WRtool/')
import ctool
from clus_manipulate import plot_clusters
from readsavencfield import read3Dncfield

# OUTPUT DIRECTORY
OUTPUTdir=dir_OUTPUT+'OUTPUT/'
OUTfig=OUTPUTdir+'OUTfig/'
OUTnc=OUTPUTdir+'OUTnc/'
OUTtxt=OUTPUTdir+'OUTtxt/'
OUTpy=OUTPUTdir+'OUTpy/'


# LOAD DATA PROCESSED BY precompute.py
#______________________________________
anomaly_filenames='area_anomaly_'+name_outputs
fn = [i for i in os.listdir(OUTnc) \
    if os.path.isfile(os.path.join(OUTnc,i)) and anomaly_filenames in i]
fn.sort()
print('\nInput {0} areal anomaly fields netCDF files (computed by precompute.py):'.format(numens))
print('\n'.join(fn))
for ens in range(numens):
    #print(fn[ens])
    ifile=OUTnc+fn[ens]
    var_area, lat_area, lon_area, dates, time_units, var_units= read3Dncfield(ifile)

anomaly_filenames='glob_anomaly_'+name_outputs
fn = [i for i in os.listdir(OUTnc) \
    if os.path.isfile(os.path.join(OUTnc,i)) and anomaly_filenames in i]
fn.sort()
print('\nInput {0} global anomaly fields netCDF files (computed by precompute.py):'.format(numens))
print('\n'.join(fn))
var_ensList=[]
for ens in range(numens):
    #print(fn[ens])
    ifile=OUTnc+fn[ens]
    var_glob, lat, lon, dates, time_units, var_units= read3Dncfield(ifile)
    var_ensList.append(var_glob)

syr=pd.to_datetime(dates).year[0]
eyr=pd.to_datetime(dates).year[-1]

name_outputs=name_outputs+'_{0}pcs'.format(numpcs)

if enstoselect!='no':
    print('Ensemble member number {0} is selected for the analysis'.format(enstoselect))
    var_ens=[]
    var_ens.append(var_ensList[enstoselect])
    print(len(var_ens),var_ens[0].shape)
    del var_ensList
    var_ensList=var_ens
    name_outputs=name_outputs+'_'+str(enstoselect)
    print(name_outputs)
    numens=1
else:
    print('All the ensemble members are concatenated one after the other prior to the analysis')


# LOAD DATA PROCESSED BY clusters_comparison.py
#______________________________________
#load indclORDasREF
namef='{0}indclORDasREF_{1}clus_{2}.txt'.format(OUTtxt,numclus,name_outputs)
indclORDasREF=np.loadtxt(namef)


# Plot of the nth the EOFs and PCs
#....eof_plots(neof,pcs_scal1, eofs_scal2,var,varunits,lat,lon,tit,numens):

# PLOT AND SAVE CLUSTERS FIGURE
# Plot of cluster patterns in one panel
#______________________________________
varname=name_outputs.split("_")[0]
model=name_outputs.split("_")[2]
kind=name_outputs.split("_")[3]
res=name_outputs.split("_")[4]
numens=int(name_outputs.split("_")[5][:-3].upper())
season=name_outputs.split("_")[6]
area=name_outputs.split("_")[7]

if enstoselect!='no':
    tit='{0} {1} {2} ens{3}\n{4} {5} {6} ({7}-{8})'.format(varname,model,kind,enstoselect,res,season,area,syr,eyr)
else:
    tit='{0} {1} {2} {3}\n{4} {5} ({6}-{7})'.format(varname,model,kind,res,season,area,syr,eyr)
ax=plot_clusters(area,lon,lat,lon_area,lat_area,numclus,numpcs,var_ensList,indclORDasREF,tit)

namef='{0}clus_patterns_{1}clus_{2}.eps'.format(OUTfig,numclus,name_outputs)
ax.figure.savefig(namef)#bbox_inches='tight')
print('Clusters eps figure for {0} weather regimes is saved as\n{1}'.format(area,namef))
print('____________________________________________________________________________________________________________________')

## Quick plot
#tit='cluster pattern 1'
#____________Plot the lon-lat map (Orthographic Projection)
#fig = plot_ortho(cluspatt[0], lat_area, lon_area, clat=50, clon=0, tit=tit)
#fig.show()
##plt.show(block=True)

print('\n******************************************************************************')
print('END {0}'.format(sys.argv[0]))
print('*********************************************************************************')
