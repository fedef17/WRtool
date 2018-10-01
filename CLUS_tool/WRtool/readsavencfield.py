# Standard packages
from netCDF4 import Dataset, num2date, date2num
import numpy as np
import os

def read4Dncfield(ifile,**kwargs):
    '''
    GOAL
        Read netCDF file of 4Dfield
    USAGE
        var, dates = read4Dncfield(fname,level)
        fname: filname and level: level to be selected in hPa
    '''
    #----------------------------------------------------------------------------------------
    print('__________________________________________________________')
    print('Reading the 4D field [time x level x lat x lon]: \n{0}'.format(ifile))
    #----------------------------------------------------------------------------------------
    fh = Dataset(ifile, mode='r')
    variabs=[]
    for variab in fh.variables:
        variabs.append(variab)
    #print('The variables in the nc file are: ', variabs)
    lev_names = ['level', 'lev', 'pressure', 'plev', 'plev8']
    for levna in lev_names:
        if levna in variabs:
            oklevname = levna
            level = fh.variables[levna][:]
            nlevs = len(level)
            break

    try:
        lat         = fh.variables['lat'][:]
        lon         = fh.variables['lon'][:]
    except KeyError as ke:
        print(repr(ke))
        lat         = fh.variables['latitude'][:]
        lon         = fh.variables['longitude'][:]

    time        = fh.variables['time'][:]
    time_units  = fh.variables['time'].units
    time_cal    = fh.variables['time'].calendar
    var_units   = fh.variables[variabs[-1]].units

    if kwargs.get('lvel'):
        lvel=kwargs['lvel']
        if nlevs > 1:
            level_units = fh.variables[levna].units
            if level_units=='millibar' or level_units=='hPa':
                l_sel=int(np.where(level==lvel)[0])
                print('Selecting level {0} millibar'.format(lvel))
            elif level_units=='Pa':
                l_sel=int(np.where(level==lvel*100)[0])
                print('Selecting level {0} Pa'.format(lvel*100))

            level=lvel
        else:
            level = level[0]
            l_sel = 0

        var         = fh.variables[variabs[-1]][:,l_sel,:,:]
        txt='{0}{1} dimension for a single ensemble member [time x lat x lon]: {2}'.format(variabs[-1],lvel,var.shape)
    else:
        var         = fh.variables[variabs[-1]][:,:,:,:]
        txt='{0} dimension for a single ensemble member [time x lat x lon]: {1}'.format(variabs[-1],var.shape)
    #print(fh.variables)
    if var_units == 'm**2 s**-2':
        print('From geopotential (m**2 s**-2) to geopotential height (m)')   # g0=9.80665 m/s2
        var=var/9.80665
        var_units='m'
    print('calendar: {0}, time units: {1}'.format(time_cal,time_units))
    dates=num2date(time,time_units,time_cal)
    fh.close()

    print(txt)

    return var, level, lat, lon, dates, time_units, var_units, time_cal

def read3Dncfield(ifile):
    '''
    GOAL
        Read netCDF file of 3Dfield
    USAGE
        var, dates = read3Dncfield(fname)
        fname: filname
    '''
    #----------------------------------------------------------------------------------------
    print('__________________________________________________________')
    print('Reading the 3D field [time x lat x lon]: \n{0}'.format(ifile))
    #----------------------------------------------------------------------------------------
    fh = Dataset(ifile, mode='r')
    variabs=[]
    for variab in fh.variables:
        variabs.append(variab)
    #print('The variables in the nc file are: ', variabs)

    try:
        lat         = fh.variables['lat'][:]
        lon         = fh.variables['lon'][:]
    except KeyError as ke:
        print(repr(ke))
        lat         = fh.variables['latitude'][:]
        lon         = fh.variables['longitude'][:]

    time        = fh.variables['time'][:]
    time_units  = fh.variables['time'].units
    var_units   = fh.variables[variabs[-1]].units
    var         = fh.variables[variabs[-1]][:,:,:]
    txt='{0} dimension [time x lat x lon]: {1}'.format(variabs[-1],var.shape)
    #print(fh.variables)
    dates=num2date(time,time_units)
    fh.close()

    print(txt)

    return var, lat, lon, dates, time_units, var_units

def read2Dncfield(ifile):
    '''
    GOAL
        Read netCDF file of 2Dfield
    USAGE
        var = read2Dncfield(fname)
        fname: filename
    '''
    #----------------------------------------------------------------------------------------
    print('__________________________________________________________')
    print('Reading the 2D field [lat x lon]: \n{0}'.format(ifile))
    #----------------------------------------------------------------------------------------
    fh = Dataset(ifile, mode='r')
    variabs=[]
    for variab in fh.variables:
        variabs.append(variab)
    #print('The variables in the nc file are: ', variabs)

    try:
        lat         = fh.variables['lat'][:]
        lon         = fh.variables['lon'][:]
    except KeyError as ke:
        print(repr(ke))
        lat         = fh.variables['latitude'][:]
        lon         = fh.variables['longitude'][:]

    #var_units   = fh.variables[variabs[2]].units
    var         = fh.variables[variabs[2]][:,:]
    txt='{0} dimension [lat x lon]: {1}'.format(variabs[2],var.shape)
    #print(fh.variables)
    fh.close()

    #print('\n'+txt)

    return var, lat, lon


def read_N_2Dfields(ifile):
    '''
    GOAL
        Read var in ofile netCDF file
    USAGE
        read a number N of 2D fields [latxlon]
        fname: output filname
    '''
    fh = Dataset(ifile, mode='r')
    variabs=[]
    for variab in fh.variables:
        variabs.append(variab)
    #print('The variables in the nc file are: ', variabs)

    num         = fh.variables['num'][:]

    try:
        lat         = fh.variables['lat'][:]
        lon         = fh.variables['lon'][:]
    except KeyError as ke:
        print(repr(ke))
        lat         = fh.variables['latitude'][:]
        lon         = fh.variables['longitude'][:]

    var         = fh.variables[variabs[3]][:,:,:]
    txt='{0} dimension [num x lat x lon]: {1}'.format(variabs[3],var.shape)
    #print(fh.variables)
    fh.close()

    #print('\n'+txt)

    return var, lat, lon


def save2Dncfield(lats,lons,variab,varname,ofile):
    '''
    GOAL
        Save var in ofile netCDF file
    USAGE
        save2Dncfield(var,ofile)
        fname: output filname
    '''
    try:
        os.remove(ofile) # Remove the outputfile
    except OSError:
        pass
    dataset = Dataset(ofile, 'w', format='NETCDF4_CLASSIC')
    #print(dataset.file_format)

    lat = dataset.createDimension('lat', variab.shape[0])
    lon = dataset.createDimension('lon', variab.shape[1])

    # Create coordinate variables for 2-dimensions
    lat = dataset.createVariable('lat', np.float32, ('lat',))
    lon = dataset.createVariable('lon', np.float32, ('lon',))
    # Create the actual 2-d variable
    var = dataset.createVariable(varname, np.float64,('lat','lon'))

    #print('variable:', dataset.variables[varname])

    #for varn in dataset.variables.keys():
    #    print(varn)
    # Variable Attributes
    lat.units='degree_north'
    lon.units='degree_east'
    #var.units = varunits

    lat[:]=lats
    lon[:]=lons
    var[:,:]=variab

    dataset.close()

    #----------------------------------------------------------------------------------------
    print('The 2D field [lat x lon] is saved as \n{0}'.format(ofile))
    print('__________________________________________________________')
    #----------------------------------------------------------------------------------------

def save3Dncfield(lats,lons,variab,varname,varunits,dates,timeunits,time_cal,ofile):
    '''
    GOAL
        Save var in ofile netCDF file
    USAGE
        save3Dncfield(var,ofile)
        fname: output filname
    '''
    try:
        os.remove(ofile) # Remove the outputfile
    except OSError:
        pass
    dataset = Dataset(ofile, 'w', format='NETCDF4_CLASSIC')
    #print(dataset.file_format)

    time = dataset.createDimension('time', None)
    lat = dataset.createDimension('lat', variab.shape[1])
    lon = dataset.createDimension('lon', variab.shape[2])

    # Create coordinate variables for 3-dimensions
    time = dataset.createVariable('time', np.float64, ('time',))
    lat = dataset.createVariable('lat', np.float32, ('lat',))
    lon = dataset.createVariable('lon', np.float32, ('lon',))
    # Create the actual 3-d variable
    var = dataset.createVariable(varname, np.float64,('time','lat','lon'))

    #print('variable:', dataset.variables[varname])

    #for varn in dataset.variables.keys():
    #    print(varn)
    # Variable Attributes
    time.units=timeunits
    time.calendar=time_cal
    lat.units='degree_north'
    lon.units='degree_east'
    var.units = varunits

    # Fill in times.
    time[:] = date2num(dates, units = timeunits, calendar = time_cal)#, calendar = times.calendar)
    print(time_cal)
    print('time values (in units {0}): {1}'.format(timeunits,time[:]))
    print(dates)

    #print('time values (in units %s): ' % time)

    lat[:]=lats
    lon[:]=lons
    var[:,:,:]=variab

    dataset.close()

    #----------------------------------------------------------------------------------------
    print('The 3D field [time x lat x lon] is saved as \n{0}'.format(ofile))
    print('__________________________________________________________')
    #----------------------------------------------------------------------------------------

def save_N_2Dfields(lats,lons,variab,varname,varunits,ofile):
    '''
    GOAL
        Save var in ofile netCDF file
    USAGE
        save a number N of 2D fields [latxlon]
        fname: output filname
    '''
    try:
        os.remove(ofile) # Remove the outputfile
    except OSError:
        pass
    dataset = Dataset(ofile, 'w', format='NETCDF4_CLASSIC')
    #print(dataset.file_format)

    num = dataset.createDimension('num', variab.shape[0])
    lat = dataset.createDimension('lat', variab.shape[1])
    lon = dataset.createDimension('lon', variab.shape[2])

    # Create coordinate variables for 3-dimensions
    num = dataset.createVariable('num', np.int32, ('num',))
    lat = dataset.createVariable('lat', np.float32, ('lat',))
    lon = dataset.createVariable('lon', np.float32, ('lon',))
    # Create the actual 3-d variable
    var = dataset.createVariable(varname, np.float64,('num','lat','lon'))

    #print('variable:', dataset.variables[varname])

    #for varn in dataset.variables.keys():
    #    print(varn)
    # Variable Attributes
    lat.units='degree_north'
    lon.units='degree_east'
    var.units = varunits

    num[:]=np.arange(variab.shape[0])
    lat[:]=lats
    lon[:]=lons
    var[:,:,:]=variab

    dataset.close()

    #----------------------------------------------------------------------------------------
    print('The {0} 2D fields [num x lat x lon] are saved as \n{1}'.format(variab.shape[0], ofile))
    print('__________________________________________________________')
    #----------------------------------------------------------------------------------------
