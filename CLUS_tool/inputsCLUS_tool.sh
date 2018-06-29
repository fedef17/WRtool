#!/bin/bash

## Previous of the weaher regime analysis,
## data should be remap r144x73 and the period of interest should be selected:
## run arrange_inputdata.sh

## Information required by the WRtool:

##-------------------------------about paths------------------------------------------
## WRtool directory
#WRTOOL_PATH=/home/mavilia/WEATHER_REGIMEStool/WRtool_GitHub/CLUS_tool/
WRTOOL_PATH=/home/fabiano/Desktop/Research/git/WRtool/CLUS_tool/

## Output data directory:
#OUTPUT_PATH=/home/mavilia/WEATHER_REGIMEStool/WRtool_GitHub/
OUTPUT_PATH=/home/fabiano/Desktop/Research/lavori/WeatherRegimes/OUT_WRTOOL/
## Input data directory:
##____________SIM
#INPUT_PATH=/home/mavilia/DATA/Stream1_Z500remap/
#INPUT_PATH=/home/mavilia/DATA/Z500T511regrid/Z500base/zg500mab/
#INPUT_PATH=/home/mavilia/DATA/grib_ECMWF/

##____________OBS
#INPUT_PATH=/home/mavilia/DATA/OBS/NCEP/zg500/
#INPUT_PATH=/home/fabiano/DATA/OBS/ERA/zg500/
INPUT_PATH=/home/fabiano/DATA/ECEARTHdata/SPHINX/AMIP/Z500T1279regrid/Z500base/zg500uab/

## List of input files
#filenames[0]='zg500_Aday_ERAInterim_2deg_1979-2008.nc'
#filenames[0]='zg500_Aday_EC-EARTH31_T1279base_regrid25_0_1979-2008.nc'

filenames[0]='zg500_Aday_EC-EARTH31_T1279base_regrid25_0_1979-2008.nc'
#filenames[0]='zg500_Aday_ERAInterim_2deg_1979-2008.nc'

#filenames[0]='zg500_Aday_NCEPNCAR_2deg_1979-2008.nc'
#filenames[0]='ens_ec_s4_em00_P129_L500_1981_2011.nc'
#filenames[0]='zg500_Aday_CNRM-CM6-1_TL127_regrid25_1979-2012.nc'
#filenames[0]='zg500_Aday_CMCC-CM2-VHR4_1152x768_regrid25_1979-2014.nc'

#filenames[0]='zg500_Aday_EC-EARTH31_T511base_regrid25_0_1979-2008.nc'
#filenames[1]='zg500_Aday_EC-EARTH31_T511base_regrid25_1_1979-2008.nc'
#filenames[2]='zg500_Aday_EC-EARTH31_T511base_regrid25_2_1979-2008.nc'
#filenames[3]='zg500_Aday_EC-EARTH31_T511base_regrid25_3_1979-2008.nc'
#filenames[4]='zg500_Aday_EC-EARTH31_T511base_regrid25_4_1979-2008.nc'
#filenames[5]='zg500_Aday_EC-EARTH31_T511base_regrid25_5_1979-2008.nc'
#filenames[6]='zg500_Aday_EC-EARTH31_T255stoc_regrid25_6_1979-2008.nc'
#filenames[7]='zg500_Aday_EC-EARTH31_T255stoc_regrid25_7_1979-2008.nc'
#filenames[8]='zg500_Aday_EC-EARTH31_T255stoc_regrid25_8_1979-2008.nc'
#filenames[9]='zg500_Aday_EC-EARTH31_T255stoc_regrid25_9_1979-2008.nc'

##-------------------------------about data-------------------------------------------
## Write only letters or numbers, no punctuation marks!
## If you want to leave the field empty write 'no'
varname=zg           #variable name as in the input file (zg,...)
level=500            #level to select (hPa)
freq=day             #data frequency ('day','mon',year',...)
filterwnd=5          #running mean filter time window
model=ECEARTH31     #model name ECEARTH31 NCEPNCAR ERAInterim
institute=rean       #institute name
kind=obs             #base: baseline, stoc: stochastic physics, obs: observations
res=144x73           #T255 144x73 N216L85
numens=1             #total number of ensemble members
enstoselect=no       #single ensemble member to analyse (starting from 0!)
## If you have only one realization set numens=1 and enstoselect=no
## If you have more than one ensemble member: if you set enstoselect=3, as an esample, only member 3 will be analysed;
## otherwise with enstoselect=no, all the numens ensemble member fields are concatenated in time before the analysis
## and the clusters are computed on the resulting field.

season=DJF           #season to be selected (JJA, DJF, DJFM, NDJFM)
area=EAT             #regional average (examples:'EAT':Euro-Atlantic
                     #                           'PNA': Pacific North American
                     #                           'NH': Northern Hemisphere)
syr=1979             #starting year
eyr=2008             #end year

##---------------------about cluster analysis------------------------------------------
numclus=4     #number of clusters
## Either set perc or numpcs:
perc=no                #cluster analysis is applied on a number of PCs such as they explain
                     #'perc' of total variance
numpcs=4             #number of PCs

nameREF="zg500_day_ERAInterim_obs_144x73_1ens_${season}_${area}_${syr}-${eyr}_${numpcs}pcs"      # reference name files
