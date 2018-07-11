#*********************************
#     clusters_comparison        *
#*********************************

# Standard packages
#from netCDF4 import Dataset, num2date, datetime
import numpy as np
import pandas as pd
import sys
import pickle
#import matplotlib.pyplot as plt
import os
from numpy import linalg as LA
from itertools import permutations
from operator import itemgetter

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
nameREF       = sys.argv[9]
print('REFERENCE NAME FILES: {0}'.format(nameREF))

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
from readsavencfield import read_N_2Dfields, read3Dncfield, save_N_2Dfields
from clus_manipulate import cluster_ordering_asREF, compute_pvectors, compute_pseudoPC, plot_clusters
from pcmatch import *

# OUTPUT DIRECTORY
OUTPUTdir=dir_OUTPUT+'OUTPUT/'
OUTfig=OUTPUTdir+'OUTfig/'
OUTnc=OUTPUTdir+'OUTnc/'
OUTtxt=OUTPUTdir+'OUTtxt/'
OUTpy=OUTPUTdir+'OUTpy/'


# LOAD DATA PROCESSED BY compute.py
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

print('\nINPUT DATA: {0} anomaly fields netCDF files (computed by precompute.py)'.format(numens))
print('Data cover the time period from {0} to {1}'.format(syr,eyr))
print('Data dimensions for one ensemble member are {0}'.format(var_ensList[0].shape))
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


# LOAD DATA PROCESSED BY compute.py
#______________________________________
name_savedvar='{0}varfrac_{1}.txt'.format(OUTtxt,name_outputs)
varfrac=np.loadtxt(name_savedvar)
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

# load PCunscal
name_savedvar='{0}PCunscal_{1}.txt'.format(OUTtxt,name_outputs)
PCunscal=np.loadtxt(name_savedvar)
print('(times, numPCs) ={0}'.format(PCunscal.shape))


# LOAD DATA PROCESSED BY clustering.py
#______________________________________
# N.B. suffix ORD refers to clusters ordered by decreasing frequency of occurrence.
#      suffix ORDasREF refers to clusters ordered as te reference clusters (e.g. observartions)

# load cluster patterns
ifile='{0}cluspatternORD_{1}clus_{2}.nc'.format(OUTnc,numclus,name_outputs)
cluspattORD,lat_area,lon_area =read_N_2Dfields(ifile)

# load solverREF
solverREF=pickle.load(open('{0}solver_{1}.p'.format(OUTpy,nameREF), 'rb'))

#load centrORD
namef='{0}centrORD_{1}clus_{2}.txt'.format(OUTtxt,numclus,name_outputs)
centrORD=np.loadtxt(namef)

# load indclORD
namef='{0}indclORD_{1}clus_{2}.txt'.format(OUTtxt,numclus,name_outputs)
indclORD=np.loadtxt(namef)

# Compute p vector as in Dawson&Palmer 2015:
#___________________________________________
# In order to compare model clusters to reanalysis it is necessary to be able to measure how different
# the spatial pattern of any particular modelled cluster is from its reanalysis counterpart. To do this
# an error metric is defined as the length of the vector between projections of the model cluster and
# the corresponding reanalysis cluster in a common phase space. The common phase space is produced by
# projecting the spatial pattern of both the modelled and reanalysis clusters onto the EOFs obtained
# from reanalysis. Denoting a cluster centroid map from reanalysis as the row vector c0, and the
# corresponding cluster centroid map from a model as cm, these projections are:
#            p0=c0E        pm=cmE
# Where E is a matrix containing the n retained EOFs in its columns. Each projection is simply a point
# in n-dimensional space.

pvec=compute_pvectors(numpcs,solverREF,cluspattORD)
print('pvec:\n{0}'.format(pvec))
# save pvec
namef='{0}pvec_{1}clus_{2}.txt'.format(OUTtxt,numclus,name_outputs)
np.savetxt(namef, pvec)

#load pREF
namef='{0}pvec_{1}clus_{2}.txt'.format(OUTtxt,numclus,nameREF)
pvecREF=np.loadtxt(namef)

pseudoPC=compute_pseudoPC(var_ensList,numpcs,solverREF)
print('pseudoPC:\n{0}'.format(pseudoPC))
# save pseudoPC
namef='{0}pseudoPC_{1}.txt'.format(OUTtxt,name_outputs)
np.savetxt(namef,pseudoPC)


# Find the match with the given reference clusters
#_______________________
pcset1 = pvec
pcset2 = pvecREF
#match_clusters_info(pcset1, pcset2,npcs=9)
perm, et, ep, patcor = match_pc_sets(pcset2, pcset1)
print('\n ----------------- \n')
print 'RMS error = ', et
print '(squared) Phase error =', ep
print 'Pattern correlation (cosine of angle between) =', patcor
print 'Optimal permutation =', perm
print('\n ----------------- \n')

# save et
namef='{0}et_{1}clus_{2}.txt'.format(OUTtxt,numclus,name_outputs)
np.savetxt(namef, et)

# save ep
namef='{0}ep_{1}clus_{2}.txt'.format(OUTtxt,numclus,name_outputs)
np.savetxt(namef, ep)

# save patcor
namef='{0}patcor_{1}clus_{2}.txt'.format(OUTtxt,numclus,name_outputs)
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
namef='{0}centrORDasREF_{1}clus_{2}.txt'.format(OUTtxt,numclus,name_outputs)
np.savetxt(namef,centrORDasREF)

# save indclORDasREF
namef='{0}indclORDasREF_{1}clus_{2}.txt'.format(OUTtxt,numclus,name_outputs)
np.savetxt(namef,indclORDasREF, fmt='%d')

# save freqORDasREF
freq=[]
for nclus in range(numclus):
    cl=list(np.where(indclORDasREF==nclus)[0])
    fr=len(cl)*100./len(indclORDasREF)
    freq.append(fr)
freqORDasREF=np.array(freq)
namef='{0}freq_ORDasREF_{1}clus_{2}.txt'.format(OUTtxt,numclus,name_outputs)
np.savetxt(namef, freqORDasREF)

# save cluspattORDasREF
cluspattORDasREF=cluspattORD[sor]
varsave='cluspattern'
ofile='{0}cluspatternORDasREF_{1}clus_{2}.nc'.format(OUTnc,numclus,name_outputs)
save_N_2Dfields(lat_area,lon_area,np.array(cluspattORDasREF),varsave,var_units,ofile)

#if enstoselect!='no':
#    tit='{0} {1} {2}{3} {4} {5} {6}'.format(varname,model,kind,enstoselect,res,season,area)
#else:
#    tit='{0} {1} {2} {3} {4} {5}'.format(varname,model,kind,res,season,area)
#
#ax=plot_clusters(lon_area,lat_area,cluspattORDasREF,freqORDasREF,area,numclus,obs,tit,patcor)
#
#namef='{0}clus_patternsORD_{1}clus_{2}.eps'.format(OUTfig,numclus,name_outputs)
#ax.figure.savefig(namef)#bbox_inches='tight')
#print('Clusters eps figure for {0} weather regimes is saved as\n{1}'.format(area,namef))
#print('____________________________________________________________________________________________________________________')

# Compute inter-clusters and intra-clusters statistics
#if area=='EAT':
#    clusnames=['NAO+','Blocking','Atlantic Ridge','NAO-']
#else:

# COMPUTE AND SAVE STATISTICS OF CLUSTER PARTITION
#_______________________
clusnames=['clus{0}'.format(nclus) for nclus in range(numclus)]
print('--------------------------DIMENSIONS:------------------------------------')
print('ANOMALY FIELD:                  [time, lat, lon]         = {0}'.format(np.concatenate(var_ensList[:]).shape))
print('EMPIRICAL ORTHOGONAL FUNCTIONS: [num_eof, lat, lon]      = ({0}, {1}, {2})'.format(PCunscal.shape[1],len(lat_area),len(lon_area)))
print('PRINCIPAL COMPONENTS:           [time, num_eof]          = {0}'.format(PCunscal.shape))
print('CENTROID COORDNATES:            [num_eof, num_clusters]  = {0}'.format(centrORDasREF.shape))
print('CLUSTER PATTERNS:               [num_clusters, lat, lon] = {0}'.format(cluspattORDasREF.shape))


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

print('\nINTER-CLUSTER DISTANCES MEAN IS {0}'.format(np.mean(norm_centr)))
print('INTER-CLUSTER DISTANCES STANDARD DEVIATION {0}'.format(np.std(norm_centr)))
print('==============================================================')



# 2) compute centroid-datapoint distance in the phase space (PC space):
print('\n==============================================================')
print('In the PC space, here are the distances between each data point in a cluster and its centroid:')
numdatapoints=PCunscal.shape[0]

for nclus in range(numclus):
    print('\n******************            {0}           ***************************'.format(clusnames[nclus]))
    cl=list(np.where(indclORDasREF==nclus)[0])
    print('{0} DAYS IN THIS CLUSTER'.format(len(cl)))
    PCsintheclus=[PCunscal[i,:] for i in cl]
    #print('standard deviation of PCs relative to days belonging to this cluster={0}'.format(np.std(PCsintheclus)))

    norm=[] # [numdatapoints]
    for point in cl:
        normpoints=LA.norm(centrORDasREF[:,nclus]-PCunscal[point,:])
        norm.append(normpoints)

    print(len(norm))
    print('MAXIMUM DISTANCE FOR CLUS {0} IS {1}'.format(nclus,max(norm)))
    txt='Furthest data point to centroid of cluster {0} is day {1}'.format(nclus,list(np.where(norm == max(norm))[0]))
    print(txt)
    print('INTRA-CLUSTER DISTANCES MEAN FOR CLUS {0} IS {1}'.format(nclus,np.mean(norm)))
    print('INTRA-CLUSTER DISTANCES STANDARD DEVIATION FOR CLUS {0} IS {1}'.format(nclus,np.std(norm)))
print('==============================================================')


print('\n******************************************************************************')
print('END {0}'.format(sys.argv[0]))
print('*********************************************************************************')
