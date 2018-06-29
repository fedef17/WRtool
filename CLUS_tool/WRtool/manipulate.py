# Standard packages
import sys
from netCDF4 import Dataset, num2date
import numpy as np

def sel_season(var,dates,season, wnd=0):
    #----------------------------------------------------------------------------------------
    print('____________________________________________________________________________________________________________________')
    print('Selecting only {0} data'.format(season))
    #----------------------------------------------------------------------------------------
    print('dates={0}'.format(dates))
    syr=dates[0].year
    eyr=dates[-1].year

    if season=='DJF':       #ONLY DEC-JAN-FEB
        m=[12,1,2]
        #REMOVING THE FIRST MONTHS (for the first year) because there is no previuos december
        start=int(np.where([(dates[t].year==syr and dates[t].month==m[0]-1) and dates[t].day==1 for t in range(len(dates))])[0])
        print('\nStart date: {0}, index={1}'.format(dates[start],start))
        #REMOVING THE LAST MONTHS (for the last year) because there is no following january
        end=int(np.where([(dates[t].year==eyr and dates[t].month==m[0]) and dates[t].day==1 for t in range(len(dates))])[0])
        print('End date (included): {0}, index={1}\n'.format(dates[end-1],end-1))
        print(var.shape)
        var_remove=var[start:end,:,:]
        dates_remove=dates[start:end]
        mask=np.array([((dates_remove[t].month==12) | (dates_remove[t].month==1) | (dates_remove[t].month==2)) for t in range(len(dates_remove))])
        mask1=np.array([((dates_remove[t].month==2) and (dates_remove[t].day==29)) for t in range(len(dates_remove))])
        var_season=var_remove[mask&~mask1,:,:]
        dates_season=dates_remove[mask&~mask1]
        print('dates_origin',dates)
        print('dates_remove',dates_remove)
        print('dates_season',dates_season)
    elif season=='DJFM':    #ONLY DEC-JAN-FEB-MAR
        m=[12,1,2,3]
        start=int(np.where([(dates[t].year==syr and dates[t].month==m[0]-1) and dates[t].day==1 for t in range(len(dates))])[0])
        print('\nStart date: {0}, index={1}'.format(dates[start],start))
        end=int(np.where([(dates[t].year==eyr and dates[t].month==m[0]) and dates[t].day==1 for t in range(len(dates))])[0])
        print('End date (included): {0}, index={1}\n'.format(dates[end-1],end-1))
        var_remove=var[start:end,:,:]
        dates_remove=dates[start:end]
        mask=np.array([((dates_remove[t].month==12) | (dates_remove[t].month==1) | (dates_remove[t].month==2) | (dates_remove[t].month==3)) for t in range(len(dates_remove))])
        var_season=var_remove[mask,:,:]
        dates_season=dates_remove[mask]
    elif season=='NDJFM':   #ONLY NOV-DEC-JAN-FEB-MAR
        m=[11,12,1,2,3]
        start=int(np.where([(dates[t].year==syr and dates[t].month==m[0]-1) and dates[t].day==1 for t in range(len(dates))])[0])
        print('\nStart date: {0}, index={1}'.format(dates[start],start))
        end=int(np.where([(dates[t].year==eyr and dates[t].month==m[0]) and dates[t].day==1 for t in range(len(dates))])[0])
        print('End date (included): {0}, index={1}\n'.format(dates[end-1],end-1))
        var_remove=var[start:end,:,:]
        dates_remove=dates[start:end]
        mask=np.array([((dates_remove[t].month==11) | (dates_remove[t].month==12) | (dates_remove[t].month==1) | (dates_remove[t].month==2) | (dates_remove[t].month==3)) for t in range(len(dates_remove))])
        var_season=var_remove[mask,:,:]
        dates_season=dates_remove[mask]
    elif season=='JJA':   #ONLY JUN-JUL-AUG
        m=[6,7,8]
        mask=np.array([((dates[t].month==6) | (dates[t].month==7) | (dates[t].month==8)) for t in range(len(dates))])
        var_season=var[mask,:,:]
        dates_season=dates[mask]
    else:
        print('season is not one of the following: DJF, DJFM, NDJFM, JJA')

    #----------------------------------------------------------------------------------------
    #ADDING addtimes BEFORE AND AFTER THE DATA, FOR RUNNING MEAN COMPUTATION
    addtimes=int(wnd/2)
    #e.g. for DJF: adding 29nov and 30nov at the beginning and 1mar and 2 mar at the end
    for yr in range(syr,eyr):
        print('\n::::::::::YEAR::::::::::::: {0}/{1}'.format(yr,yr+1))
        bef=int(np.where([(dates[t].year==yr and dates[t].month==m[0]) and dates[t].day==1 for t in range(len(dates))])[0])
        if m[-1]==2:
            aft=int(np.where([(dates[t].year==yr+1 and dates[t].month==m[-1] and dates[t].day==28) for t in range(len(dates))])[0])
        else:
            aft=int(np.where([(dates[t].year==yr+1 and dates[t].month==m[-1]+1 and dates[t].day==1) for t in range(len(dates))])[0])
        print('Days added for running mean:')
        print('BEFORE: from {0} to {1}'.format(dates[bef-addtimes],dates[bef-1]))
        print('AFTER:  from {0} to {1}'.format(dates[aft+1],dates[aft+1+addtimes-1]))
        var_seasonadd_yr=var[bef-addtimes:aft+1+addtimes,:,:]
        #----------------------------------------------------------------------------------------
        print('Performing the running mean (time window={0})'.format(wnd))
        #-------------------------Running mean
        var_filt_yr=np.empty([var_seasonadd_yr.shape[0]-wnd+1,var_seasonadd_yr.shape[1],var_seasonadd_yr.shape[2]])
        for c in range(var_seasonadd_yr.shape[0]-wnd+1):   #c=[0:89] for DJF
            var_filt_yr[c,:,:]=np.mean(var_seasonadd_yr[c:c+wnd,:,:],axis=0)
        print('dimensions of the {0}/{1} year variable after a running mean of window={2} times ---> {3}'.format(yr,yr+1,wnd,var_filt_yr.shape))
        if yr==syr:
            var_filt=var_filt_yr
        else:
            var_filt=np.concatenate((var_filt,var_filt_yr), axis=0)

    print('\nOriginal data dimesion---> {0}'.format(var.shape))
    print('Original data cover the time period from {0} to {1}'.format(dates[0],dates[-1]))
    print('{0} data dimension ---> {1}'.format(season,var_filt.shape))
    print('{0} data cover the time period from {1} to {2}'.format(season, dates_season[0],dates_season[-1]))

    return var_season,dates_season,var_filt

def anomalies(var_filt,var_season, dates_season,season):
    # Compute anomalies by removing the time-mean for each time (day, month,...)
    print('____________________________________________________________________________________________________________________')
    print('Computing anomalies by removing the time-mean for each time')
    if season=='DJF':       #ONLY DEC-JAN-FEB
        m=[12,1,2]
    elif season=='DJFM':    #ONLY DEC-JAN-FEB-MAR
        m=[12,1,2,3]
    elif season=='NDJFM':   #ONLY NOV-DEC-JAN-FEB-MAR
        m=[11,12,1,2,3]
    elif season=='JJA':   #ONLY JUN-JUL-AUG
        m=[6,7,8]
    var_anom=np.empty_like(var_season)
    print(var_season.shape)
    for mon in range(len(m)):
        numdays=len(np.where([(dates_season[t].year==dates_season[0].year+1 and dates_season[t].month==m[mon]) for t in range(len(dates_season))])[0])
        print('Number of days for month {0} is {1}'.format(m[mon],numdays))
        for day in range(numdays):
            #print('day: {0}, month: {1}'.format(day+1,m[mon]))
            mo=np.where([(dates_season[t].month==m[mon] and dates_season[t].day==day+1) for t in range(len(dates_season))])[0]
            #print('mo={0}, dates={1}'.format(mo,dates_season[mo]))
            #print('**********************************')
            var_anom[mo,:,:]=var_season[mo,:,:]-np.mean(var_filt[mo,:,:],axis=0)

    print('\ndimensions of the {0} variable anomalies ---> {1}'.format(season,var_anom.shape))

    del var_filt
    return var_anom


#____________Selecting only [latS-latN, lonW-lonE] box region
def sel_area(lat,lon,var,area):
    '''
    GOAL
        Selecting the area of interest
    USAGE
        var_area, lat_area, lon_area =sel_area(lat,lon,var,area)
        area can be 'EAT', 'PNA', 'NH'
    '''
    if area=='EAT':
        printarea='Euro-Atlantic'
        latN = 87.5
        latS = 30.0
        lonW =-80.0     #280
        lonE = 40.0     #40
        # lat and lon are extracted from the netcdf file, assumed to be 1D
        #If 0<lon<360, convert to -180<lon<180
        if lon.min() >= 0:
            lon_new=lon-180
            var_roll=np.roll(var,int(len(lon)/2),axis=2)
        else:
            var_roll=var
            lon_new=lon

    elif area=='PNA':
        printarea='Pacific North American'
        latN = 87.5
        latS = 30.0
        lonW = 140.0
        lonE = 280.0
        # lat and lon are extracted from the netcdf file, assumed to be 1D
        #If -180<lon<180, convert to 0<lon<360
        if lon.min() < 0:
            lon_new=lon+180
            var_roll=np.roll(var,int(len(lon)/2),axis=2)
        else:
            var_roll=var
            lon_new=lon

    elif area=='NH':
        printarea='Northern Hemisphere'
        latN = 90.0
        latS = 0.0
        lonW = lon.min()
        lonE = lon.max()
        var_roll=var
        lon_new=lon

    #----------------------------------------------------------------------------------------
    print('____________________________________________________________________________________________________________________')
    print('Selecting the area of interest: {0}'.format(printarea))
    #----------------------------------------------------------------------------------------
    #-------------------------Selecting only an area

    latidx = (lat >= latS) & (lat <= latN)
    lonidx = (lon_new >= lonW) & (lon_new <= lonE)

    var_area = var_roll[:, latidx][..., lonidx]
    print('Data dimension over the selected area ---> {0}'.format(var_area.shape))

    return var_area,lat[latidx],lon_new[lonidx]
