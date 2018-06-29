#*********************************
#            compute             *
#*********************************

# Standard packages
from netCDF4 import Dataset, num2date
import numpy as np
import pandas as pd
import sys
import pickle
import matplotlib.pyplot as plt
#import pydoc
import os

print('*********************************************************************************')
print('Running {0}'.format(sys.argv[0]))
print('*********************************************************************************')
dir_WRtool    = sys.argv[1]  # WRTOOL directory
dir_OUTPUT    = sys.argv[2]  # OUTPUT directory
name_outputs  = sys.argv[3]  # name of the outputs
numens        = int(sys.argv[4])  # total number of ensemble members
numpcs        = sys.argv[5]  # number of principal components
perc          = sys.argv[6]  # percentage of explained variance for EOF analysis
enstoselect   = sys.argv[7]  # ensemble member to analyse

print('\nINPUT DATA:')
if numpcs!='no':
    numpcs=int(sys.argv[5])
    print('number of principal components: {0}'.format(numpcs))
if perc!='no':
    perc=int(sys.argv[6])
    print('percentage of explained variance: {0}%'.format(perc))
if (perc=='no' and numpcs=='no') or (perc!='no' and numpcs!='no'):
    raise ValueError('You have to specify either "perc" or "numpcs".')
if enstoselect!='no':
    enstoselect=int(sys.argv[7])
    print('ensemble member to analyse: {0}'.format(enstoselect))

# User-defined libraries
sys.path.insert(0,dir_WRtool+'WRtool/')
from readsavencfield import read3Dncfield, read2Dncfield, save2Dncfield, save_N_2Dfields
from EOFtool import eof_computation

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
print('\nInput {0} anomaly fields netCDF files (computed by precompute.py):'.format(numens))
print('\n'.join(fn))
var_ensList=[]
for ens in range(numens):
    #print(fn[ens])
    ifile=OUTnc+fn[ens]
    var_area, lat_area, lon_area, dates, time_units, var_units= read3Dncfield(ifile)
    var_ensList.append(var_area)

syr=pd.to_datetime(dates).year[0]
eyr=pd.to_datetime(dates).year[-1]

print('\nINPUT DATA:')
print('Data cover the time period from {0} to {1}'.format(syr,eyr))
print('Data dimensions for one ensemble member is {0}'.format(var_ensList[0].shape))
print('Number of ensemble members is {0}'.format(len(var_ensList)))

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


# COMPUTE AND SAVE EOFs (empirical orthogonal functions)
# AND PCs (principal component time series)
#_____________________________________________________________
solver, pcs_scal1, eofs_scal2, pcs_unscal0, eofs_unscal0, varfrac = eof_computation(var_ensList,var_units,lat_area,lon_area)

pickle.dump([solver, pcs_scal1, eofs_scal2, pcs_unscal0, eofs_unscal0, varfrac], open(dir_OUTPUT+'out_EOFcomputation.pic','w'))

#neof=0   # EOF to plot (neof starts from zero!)
#tit='{0} {1} {2} {3} {4} {5}'.format(varname,model,kind,res,season,area)  # field decomposed with the EOF analysis
#figPC_scal1, figEOF_scal2=eof_plots(neof,pcs_scal1, eofs_scal2,var,varunits,lat,lon,tit,numens)
## plot the PCs
#namef='{0}PCs{1}_{2}.eps'.format(OUTfig,neof+1,name_outputs)
#figPC.savefig(namef)#bbox_inches='tight')
## plot the EOFs
#namef='{0}EOFs{1}_{2}.eps'.format(OUTfig,neof+1,name_outputs)
#figEOF.savefig(namef)#bbox_inches='tight')
#print('PCs and EOFs eps figure are saved in {0}'.format(OUTfig))
#print('____________________________________________________________________________________________________________________')

acc=np.cumsum(varfrac*100)
if perc!='no':
    # Find how many PCs explain a certain percentage of variance
    # (find the mode relative to the percentage closest to perc, but bigger than perc)
    numpcs=min(enumerate(acc), key=lambda x: x[1]<=perc)[0]+1
    print('\nThe number of PCs that explain the percentage closest to {0}% of variance (but grater than {0}%) is {1}'.format(perc,numpcs))
    exctperc=min(enumerate(acc), key=lambda x: x[1]<=perc)[1]
if numpcs!='no':
    exctperc=acc[numpcs-1]
print('(the first {0} PCs explain exactly the {1}% of variance)'.format(numpcs,"%.2f" %exctperc))


# save python object solver
namef='{0}solver_{1}.p'.format(OUTpy,name_outputs)
pickle.dump(solver, open(namef, 'wb'), protocol=2)

# save EOF unscaled
varsave='EOFunscal'
ofile='{0}EOFunscal_{1}.nc'.format(OUTnc,name_outputs)
save_N_2Dfields(lat_area,lon_area,eofs_unscal0[:numpcs],varsave,'m',ofile)
#tit='EOFunscal n{0}'.format(neof+1)
#____________Plot the lon-lat map (Orthographic Projection)
#fig0 = plot_ortho(eofs_unscal0[0], lat_area, lon_area, clat=50, clon=0, tit=tit)
#fig0.show()
#plt.show(block=True)
print('The {0} EOF unscaled are saved as\n{1}'.format(numpcs,ofile))
print('____________________________________________________________________________________________________________________')

# save EOF scaled: EOFs are multiplied by the square-root of their eigenvalues
varsave='EOFscal2'
ofile='{0}EOFscal2_{1}.nc'.format(OUTnc,name_outputs)
save_N_2Dfields(lat_area,lon_area,eofs_scal2[:numpcs],varsave,'m',ofile)
#tit='EOFscal2 n{0}'.format(neof+1)
#____________Plot the lon-lat map (Orthographic Projection)
#fig2 = plot_ortho(eofs_scal2[npc], lat_area, lon_area, clat=50, clon=0, tit=tit)
#fig2.show()
#plt.show(block=True)
print('The {0} EOF scaled2 (multiplied by the square-root of their eigenvalues)are saved as\n{1}'.format(numpcs,ofile))
print('____________________________________________________________________________________________________________________')

# save PC unscaled
# [time x PCi]
namef='{0}PCunscal_{1}.txt'.format(OUTtxt,name_outputs)
np.savetxt(namef, pcs_unscal0[:,:numpcs])

#save PC scaled: PCs are scaled to unit variance (divided by the square-root of their eigenvalue)
# [time x PCi]
namef='{0}PCscal1_{1}.txt'.format(OUTtxt,name_outputs)
np.savetxt(namef, pcs_scal1[:,:numpcs])
print('The {0} PCs in columns are saved in\n{1}'.format(numpcs,OUTtxt))
print('____________________________________________________________________________________________________________________')

# save varfrac: the fractional variance represented by each EOF mode
namef='{0}varfrac_{1}.txt'.format(OUTtxt,name_outputs)
np.savetxt(namef, varfrac, fmt='%1.10f')


print('\n******************************************************************************')
print('END {0}'.format(sys.argv[0]))
print('******************************************************************************')
