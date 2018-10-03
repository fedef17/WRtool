# Standard packages
import numpy as np
import pickle
import datetime
from operator import itemgetter
import matplotlib.pyplot as plt
#from mpl_toolkits.basemap import Basemap
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from netCDF4 import Dataset
import os
from numpy import linalg as LA

#////////////////////////////////////////////////////////////////////////////////////////////////
#____________FUNCTION 1: Cluster ordering in decreasing frequency
def cluster_orderingFREQ(indcl,centroids, numclus):
    #----------------------------------------------------------------------------------------
    print('__________________________________________________________')
    print('Cluster ordering in decreasing frequency')
    #----------------------------------------------------------------------------------------

    L=[]
    for nclus in range(numclus):
        cl=list(np.where(indcl==nclus)[0])
        fr=len(cl)*100./len(indcl)
        #ttclus='Cluster{0} freq: {1}%'.format(nclus+1,"%.2f" %fr)
        #print(ttclus)
        #print('---> {0} days over {1}\n'.format(len(cl),len(indcl)))
        L.append([nclus,fr,cl])

    #Cluster labels:
    print('\nCluster labels:')
    print([L[ncl][0] for ncl in range(numclus)])
    #Cluster frequencies:
    print('\nCluster frequencies:')
    print([L[ncl][1] for ncl in range(numclus)])
    #Cluster members
    print('\nCluster members:')
    print([L[ncl][2] for ncl in range(numclus)])

    #Number of days of each cluster
    print('\nNumber of days of each cluster:')
    print([len(L[ncl][2]) for ncl in range(numclus)])
    print('Total number of days: {0}\n'.format(len(indcl)))

    # ----> Cluster order by decreasing frequency:
    L.sort(reverse=True,key=itemgetter(1))
    sor=[L[ncl][0] for ncl in range(numclus)]
    print('\nDecreasing frequency order:')
    print(sor)

    labelsord=np.copy(indcl)
    print('labels before ',labelsord[:30])

    for i,v in enumerate(labelsord):
        for ncl in range(numclus):
            if v==sor[ncl]:
                labelsord[i]=ncl
    print('Clusters has been sorted in this order {0}'.format(sor))
    print('labels after ',labelsord[:30])

    return centroids[:,sor],labelsord


#////////////////////////////////////////////////////////////////////////////////////////////////
#____________FUNCTION 2: Cluster ordering as a reference order
def cluster_ordering_asREF(indcl,centroids, numclus, sor):
    #----------------------------------------------------------------------------------------
    print('__________________________________________________________')
    print('Cluster ordering as reference (e.g. observed clusters)')
    #----------------------------------------------------------------------------------------

    print('\nReference order:')
    print(sor)

    labelsord=np.copy(indcl)
    print('labels before ',labelsord[:30])
    for i,v in enumerate(labelsord):
        for ncl in range(numclus):
            if v==sor[ncl]:
                labelsord[i]=ncl
    print('Clusters has been sorted in this order {0}'.format(sor))
    print('labels after ',labelsord[:30])

    return centroids[:,sor],labelsord

#////////////////////////////////////////////////////////////////////////////////////////////////
#____________FUNCTION 3: Compute the cluster patterns

def compute_clusterpatterns(numclus,var,indcl):
    #----------------------------------------------------------------------------------------
    print('__________________________________________________________')
    print('Compute the cluster patterns: original anomaly fieds averaged during times belonging to each cluster')
    #----------------------------------------------------------------------------------------
    var_allens=np.concatenate(var[:])
    cluspatt=[]
    for nclus in range(numclus):
        cl=list(np.where(indcl==nclus)[0])
        freq_perc=len(cl)*100./len(indcl)
        tclus='CLUSTER {0} ---> {1}%\n'.format(nclus,"%.2f" %freq_perc)
        print(tclus)
        cluspattern=np.mean(var_allens[cl,:,:],axis=0)
        cluspatt.append(cluspattern)
    return cluspatt

#////////////////////////////////////////////////////////////////////////////////////////////////
#____________FUNCTION 4: Plot the cluster patterns, all in a panel

def plot_clusters(area,lon,lat,lon_area,lat_area,numberclusters,numpcs,var_ensList,indcl,tit):
    #----------------------------------------------------------------------------------------
    print('__________________________________________________________')
    print('Plot the cluster patterns, all in a panel')
    #----------------------------------------------------------------------------------------
    # Plot the cluster patterns, all in a panel
    #central lon and lat
    clat=lat_area.min()+abs(lat_area.max()-lat_area.min())/2
    clon=lon_area.min()+abs(lon_area.max()-lon_area.min())/2
    var_allens=np.concatenate(var_ensList[:])
    proj = ccrs.Orthographic(central_longitude=clon, central_latitude=clat)
    delta=30
    if area=='PNA':
        rangecolorbar=np.arange(-270, 300, delta)
    if area=='EAT':
        rangecolorbar=np.arange(-240, 270, delta)
        #clus_ordered=['NAO+','Blocking','Altantic Ridge','NAO-']  #Order of observed clusters
        clus_ordered=['clus1','clus2','clus3','clus4']
    ax = plt.subplots(2, 2, figsize=(18,14), sharex='col', sharey='row')
    txt='Clusters for {0} PCs'.format(numpcs)
    print(txt)

    cluspatt=[]
    for nclus in range(numberclusters):
        cl=list(np.where(indcl==nclus)[0])
        freq_perc=len(cl)*100./len(indcl)
        tclus='{0} {1}%\n'.format(clus_ordered[nclus],"%.2f" %freq_perc)
        print(tclus)
        cluspattern=np.mean(var_allens[cl,:,:],axis=0)
        lonzero=cluspattern[:,0:1]
        cluspattern_ext=np.concatenate([cluspattern,lonzero],axis=1)
        lon_ext=np.append(lon,lon[-1]+(lon[1]-lon[0]))
        cluspatt.append(cluspattern)
        ax = plt.subplot(2, 2, nclus+1, projection=proj)
        ax.set_global()
        ax.coastlines()
        #ax.gridlines()
        fill = ax.contourf(lon_ext,lat,cluspattern_ext,rangecolorbar,cmap=plt.cm.RdBu_r, transform=ccrs.PlateCarree(), extend = 'both')
        plt.title(tclus, fontsize=30, fontweight='bold')
    #             ([x,y,thickness, height])
    cax = plt.axes([0.85, 0.1, 0.02, 0.8])
    cb=plt.colorbar(fill,cax=cax)#, labelsize=18)
    cb.set_label('(m)', rotation=0, fontsize=22)
    cb.ax.tick_params(labelsize=22)
    plt.suptitle(tit, fontsize=40, fontweight='bold')
    #ax.annotate(txt, xy=(.26, .875),xycoords='figure fraction',fontsize=24)
    plt.subplots_adjust(top=0.85)

    left   = 0    # the left side of the subplots of the figure
    right  = 0.8  # the right side of the subplots of the figure
    bottom = 0    # the bottom of the subplots of the figure
    top    = 0.82  # the top of the subplots of the figure
    wspace = 0    # the amount of width reserved for blank space between subplots
    hspace = 0.32   # the amount of height reserved for white space between subplots

    plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)
    #plt.show()

    return ax


#////////////////////////////////////////////////////////////////////////////////////////////////
#____________FUNCTION 5: Compute p vectors as in Dawson&Palmer 2015
#Simulated cluster pattern map projected onto observed EOF (obtaining pseudo centroid coordinates)
def compute_pvectors(numpcs,solverREF,cluspatt):
    p=solverREF.projectField(cluspatt,neofs=numpcs,eofscaling=0,weighted=True)
    return p

#____________FUNCTION 6: Compute pseudo PC
#Simulated anomaly field projected onto observed EOF (obtaining pseudo PC)
def compute_pseudoPC(var_ensList,numpcs,solverREF):
    var_allens=np.concatenate(var_ensList[:])
    len1ens=var_ensList[0].shape[0]
    numens=len(var_ensList)
    print('One ensemble member dim: {0}'.format(var_ensList[0].shape))
    print('{0} ensemble members together dim: {1}'.format(numens,var_allens.shape))
    pseudoPC=solverREF.projectField(var_allens,neofs=numpcs,eofscaling=0,weighted=True)
    return pseudoPC
