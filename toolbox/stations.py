"""

A module for handling statons
A station is an object with known location, code and long name. 

"""

import numpy as np
import codecs
from matplotlib import pyplot as plt
import matplotlib as mpl
import netCDF4 as nc4

class StationError(Exception):
    pass


class Station:
    def __init__(self, code, name, lon, lat, height=0.0, area_type='', dominant_source='', ifVerbose=False):

        
        self.lon = float(lon)
        self.lat = float(lat)
        self.name = name
        self.code = code        
        self.area = area_type.lower()
        self.source = dominant_source.lower()
        self.hgt = height
        self.ifVerbose = ifVerbose
        if ifVerbose: print(self)

    def __str__(self):
        return '%s lo=%.2f la=%.2f %s' % (self.code, self.lon, self.lat, self.name.encode('utf-8'))
        
    def __hash__(self): 
        return self.code.__hash__()
    
    def __eq__(self, other): 
        return self.code == other.code 

    def __ne__(self, other):
        return not(self == other)
    
    def __gt__(self, other):
        return self.code > other.code

    def __lt__(self, other):
        return self.code < other.code

    def toFile(self, file, separator=u' '):
        fields = [format%value for format, value in (('%s', self.code), 
                                                     ('%f', self.lon),
                                                     ('%f', self.lat),
                                                     ('%.1f', self.hgt),
                                                     ('%s', self.name), 
                                                     ('%s', self.area),
                                                     ('%s', self.source))]
#        file.write(separator.join(fields).encode('utf-8') + '\n'.encode('utf-8'))
        file.write(separator.join(fields) + '\n')

    def inGrid(self, grid):
        # checks if the station is inside the given grid
        fX, fY = grid.geo_to_grid(self.lon, self.lat)
        return (fX >= -0.5 and fX < grid.nx-0.5 and fY >= -0.5 and fY < grid.ny-0.5)
        

#**********************************************************************************
#
# Reading & writing stations
#
#**********************************************************************************
    
def readStations(file, columns=None, separator=None, have_full=False):
    if columns:
        ind_lat = columns['latitude']
        ind_lon = columns['longitude']
        ind_name = columns['name']
        ind_code = columns['code']
        if 'source' in columns:
            ind_source = columns['source']
        else:
            ind_source = None
        if 'area' in columns:
            ind_area = columns['area']
        else:
            ind_area = None
        if 'altitude' in columns:
            ind_alt = columns['altitude']
        else:
            ind_alt = None
    elif separator == ' ':
        ind_lat = 1
        ind_lon = 2
        ind_name = 4
        ind_code = 0
        if have_full:
            ind_alt = 3
            ind_area = -2
            ind_source = -1
        else:
            ind_area = None
            ind_source = None
            ind_alt = None
    else:
        ind_lat = 2
        ind_lon = 1
        ind_name = 4
        ind_code = 0
        ind_area = 5
        ind_source = 6
        ind_alt = 3

    stations = {}
    try:
        file.read
        fh = file
        Fname = None
    except AttributeError:
        Fname = file
        fh = codecs.open(file,'rb') #, encoding='utf-8')  #'latin-1')   #'utf-8')
    linenum = 0


    try:
        for l in fh:
#            print (l)
            linenum += 1
            try:
                line=codecs.decode(l,"utf-8")
            except UnicodeDecodeError:
                print(Fname,":Non-unicode line:", l)
                line=codecs.decode(l,"latin1")
            if line.startswith('#') or line.isspace(): 
                continue
            dat = line.rstrip().split(separator)

            if len(dat) >= 4:
                area = dat[ind_area] if ind_area is not None else ''
                source = dat[ind_source] if ind_source is not None else ''
                alt = float(dat[ind_alt]) if ind_alt is not None else 0.0
                station = Station(dat[ind_code], dat[ind_name], 
                                  dat[ind_lon], dat[ind_lat], alt, area, source)
                stations[dat[ind_code]] = station
            elif (len(dat) == 3):
                # No name given
                stations[dat[ind_code]] = Station(dat[ind_code], '', dat[ind_lon], dat[ind_lat])
            else:
                raise ValueError('Invalid station record: ' + line)

    except:
        print('Parse error at %s:%i' % (file, linenum))
        raise
        #raise ValueError()
    if fh is not file:
        fh.close()
    return stations


#=================================================================================

def stations_to_nc(lstStations, outf):
    #
    # Writes a list of stations to open nc file, as 
    #
    # Dimensions and variables might have been created already
    #
    strlen=64
    for st in lstStations:
        strlen = max(strlen, len(st.name), len(st.area), len(st.source))

    try: slen = outf.createDimension("name_strlen", strlen)
    except: pass
    try: station = outf.createDimension("station",len(lstStations))
    except: pass
    # longitude
    try:
        lon = outf.createVariable("lon","f4",("station",), zlib=True, complevel=4)
        lon.standard_name = "longitude"
        lon.long_name = "station longitude"
        lon.units = "degrees_east"
    except: lon = outf.variables['lon']
    # latitude
    try:
        lat = outf.createVariable("lat","f4",("station",), zlib=True, complevel=4)
        lat.standard_name = "latitude"
        lat.long_name = "station latitude"
        lat.units = "degrees_north"
    except:
        lat = outf.variables['lat']
    # altitude
    try:
        alt= outf.createVariable("alt","f4",("station",), zlib=True, complevel=4)
        alt.standard_name = "surface_altitude"
        alt.long_name = "station altitude asl"
        alt.units = "m"
        alt.positive = "up";
        alt.axis = "Z";
    except:
        alt = outf.variables['alt']

    # Other station attributes: relevant for stations-only, could not be already there
    stcode = outf.createVariable("station_code","c",("station","name_strlen"), zlib=True, complevel=4)
    stcode.long_name = "station code"
    stcode.cf_role = "timeseries_id";
    stname = outf.createVariable("station_name","c",("station","name_strlen"), zlib=True, complevel=4)
    stname.long_name = "station name"
    starea = outf.createVariable("area_type","c",("station","name_strlen"), zlib=True, complevel=4)
    starea.long_name = "area_type"
    stsource = outf.createVariable("source_type","c",("station","name_strlen"), zlib=True, complevel=4)
    stsource.long_name = "source_type"
    #
    # Store the stations themselves
    #
    charTmp = np.zeros(shape=(4, len(lstStations), strlen), dtype='S1')
    for ist,st in enumerate(lstStations):
        charTmp[0,ist,:] = nc4.stringtoarr( st.code, strlen,dtype='S')
        charTmp[1,ist,:] = nc4.stringtoarr( st.name, strlen,dtype='S')
        charTmp[2,ist,:] = nc4.stringtoarr( st.area, strlen,dtype='S')
        charTmp[3,ist,:] = nc4.stringtoarr( st.source, strlen,dtype='S')

    stcode[:,:] = charTmp[0,:,:]
    stname[:,:] = charTmp[1,:,:]
    starea[:,:] = charTmp[2,:,:]
    stsource[:,:] = charTmp[3,:,:]
    lon[:] = list( (st.lon for st in lstStations))
    lat[:] = list( (st.lat for st in lstStations))
    alt[:] = list( (st.hgt for st in lstStations))
    return outf
    

#=================================================================================

def stations_from_nc(nc):
    #
    # Get stations from netcdf file, which should be already open
    #
    stlist = nc4.chartostring(nc.variables['station_code'][:])
    if stations2read is None:     # need indices of stations to get?
        idxStat = range(len(stlist))
    else:
        idxStat = np.searchsorted(stlist, stations2read, sorter=np.argsort(stlist))
        if np.any(idxStat == len(stlist)):
            for ist in range(len(idxStat)):
                if idxStat[ist] == len(stlist): 
                    print('Requested station does not exist: ', stations2read[ist])
            raise
    stnames = nc4.chartostring(nc.variables['station_name'][idxStat])
    stcodes = nc4.chartostring(nc.variables['station_code'][idxStat])
    stareas = nc4.chartostring(nc.variables['area_type'][idxStat])
    stsources = nc4.chartostring(nc.variables['source_type'][idxStat])
    stx = nc.variables['lon'][idxStat]
    sty = nc.variables['lat'][idxStat]
    stz = nc.variables['alt'][idxStat]

    obsStations = []
    for i in range(len(idxStat)):
        obsStations.append(Station(stcodes[i], stnames[i], stx[i], sty[i], 
                                   height=stz[i], area_type=stareas[i], 
                                   dominant_source=stsources[i]))
    return obsStations

    
#=================================================================================

def filtered_stations(stations, area_types=None, dominant_sources=None):
    """
    Return an iterator over a collection of stations selecting those matching 
    the given area type and dominant source.
    """
    for station in stations:
        if area_types and not station.area in area_types:
            continue
        if dominant_sources and not station.source in dominant_sources:
            continue
        yield station 


#==================================================================================

def distance_matrix(stLst):
    #
    # Calculates the distance between the sations, each to each
    # The ilst can be long, so do it efficiently
    # We shall use Haversine formula from spherical geometry
    #
    p = 3.14159265358979323/180.
    R = 6371.
    lons = np.zeros(shape=len(stLst))
    lats = np.zeros(shape=len(stLst))
    dlon = np.zeros(shape=len(stLst))
    a = np.zeros(shape=len(stLst))
    codes = [''] * len(stLst)
    distMatr = np.zeros(shape=(len(stLst),len(stLst)))
    distMatr_plane = np.zeros(shape=(len(stLst),len(stLst)))
    
    for ist in range(len(stLst)):     # prepare the vectors out of the station list
        lons[ist] = stLst[ist].lon
        lats[ist] = stLst[ist].lat
        codes[ist] = stLst[ist].code
    # Haversine  formula for distance on the sphere
    for ist in range(len(stLst)):
        dlon[:] = np.abs(lons-lons[ist])
        dlon[:] = np.where(dlon > 180, -dlon+360, dlon)
        distMatr[ist,:] = np.arcsin(np.sqrt(np.cos(lats[ist] * p) * 
                                            np.cos(lats * p) * 
                                            (1. - np.cos(dlon * p)) / 2 - 
                                            np.cos((lats - lats[ist]) * p) / 2 + 0.5))  * R * 2.
        # for checking, distance of sphere projected to plane:
        distMatr_plane[ist,:] = np.sqrt(np.square((lats[ist]-lats)*p) + 
                                        np.square(np.cos((lats[ist] + lats)*p*0.5) * 
                                                  (lons[ist]-lons)*p)) * R
    diff = distMatr_plane - distMatr
    return(distMatr, distMatr_plane, distMatr_plane - distMatr)


#==========================================================================

def downsample_to_grid(stations, gridTarget):
    #
    # Projects the stations to the given gridTarget. Returns new set of stations
    #
    # Get the coordinates and project to the target grid
    lons = np.array(list((s.lon for s in stations)))
    lats = np.array(list((s.lat for s in stations)))
    fx, fy = gridTarget.geo_to_grid(lons, lats)
    idxOK = np.logical_and(fx >=-0.5, 
                           np.logical_and(fy >= -0.5,
                                          np.logical_and(fx < gridTarget.nx - 0.5,
                                                         fy < gridTarget.ny - 0.5))) 
    arStations = np.array(stations)[idxOK]  # initial list of stations inside the grid
    # gridded locations (redundant initial list here but very complicated otherwise)
    ix = np.round(fx[idxOK]).astype(np.int)
    iy = np.round(fy[idxOK]).astype(np.int)
    grdLons, grdLats = gridTarget.grid_to_geo(ix, iy) # centres of the cells
    # New station codes come from cell indices
    stGrdCodesAll = np.array(list(('%04g_%04g' % (i, j) for i,j in zip(ix,iy))))
    # Compressed set of station codes
    stGrdCodesSortedUniq = np.unique(stGrdCodesAll)
    # compression rule
    idxInUniq = np.searchsorted(stGrdCodesSortedUniq, stGrdCodesAll)
    # Reserve space
    grdStat = []
    # Collect the stations into the new sets
    for iSt, stCode in enumerate(stGrdCodesSortedUniq):
        idxSt = idxInUniq == iSt  # indices of initial stations falling into this grid cell
        grdStat.append(Station(stCode,
                               '_'.join(list((s.code for s in arStations[idxSt]))),
                               grdLons[idxSt][0], grdLats[idxSt][0],
                               np.nanmean(list((s.hgt for s in arStations[idxSt]))),
                               '_'.join(list((s.area for s in arStations[idxSt]))),
                               '_'.join(list((s.source for s in arStations[idxSt])))))
    return grdStat


#==========================================================================

def downsample_to_subdomains(stations, gridFrom, arWinStart, nWinSize):
    #
    # Projects the stations to the given set of subdomains defined for the gridFrom.
    # The domains have same size nWinSize but different starting points arWinStart.
    # Domains can overlap
    # Returns new set of "stations"
    #
    # Get the coordinates and project to the grid
    lons = np.array(list((s.lon for s in stations)))
    lats = np.array(list((s.lat for s in stations)))
    fx, fy = gridFrom.geo_to_grid(lons, lats)
    ix = np.round(fx).astype(np.int32)
    iy = np.round(fy).astype(np.int32)
    #
    # Since the subdmwin windows are irregular, the same station can be in several
    # Have to go brute-force checking window by window
    #
    arWinEnd = arWinStart + nWinSize # endpoints of the subdomain windows
    #
    # Output: station-cells indexed as the subdomains. Only those with
    # stations are included
    #
    dmnStat = []
    for ixWin in range(arWinStart.shape[0]):
        for iyWin in range(arWinStart.shape[1]):
            ifInside = np.logical_and((ix - arWinStart[ixWin,iyWin,0]) * 
                                      (arWinEnd[ixWin,iyWin,0] - 1 - ix) >= 0,
                                      (iy - arWinStart[ixWin,iyWin,1]) * 
                                      (arWinEnd[ixWin,iyWin,1] - 1 - iy) >= 0)
            # Any station in this subdomain?
            if not np.any(ifInside): continue
            # Location of the window centre point
            winLon, winLat = gridFrom.grid_to_geo(arWinStart[ixWin,iyWin,0] + nWinSize / 2.0,
                                                  arWinStart[ixWin,iyWin,1] + nWinSize / 2.0)
            # A station-cell for this subdomain
            dmnStat.append(Station('%03g_%03g' % (ixWin, iyWin),   # code
                                   '_'.join(list((s.code for s in stations[ifInside]))),
                                   winLon, winLat,
                                   np.nanmean(list((s.hgt for s in stations[ifInside]))),
                                   '_'.join(list((s.area for s in stations[ifInside]))),
                                   '_'.join(list((s.source for s in stations[ifInside])))))
    return dmnStat


#**********************************************************************************
#
# Graphics with matplotlib
#
#**********************************************************************************


def stations2map(stations, map, marker='bo',markersize=1, label='', text=None):
    x = np.zeros(len(stations), float)
    y = np.zeros(len(stations), float)
    for i in range(len(x)):
        x[i] = stations[i].lon
        y[i] = stations[i].lat
    X,Y = map(x,y)
    map.plot(X,Y, marker,ms=markersize, label=label)
    if text == 'code':
        for s,xpt,ypt in zip(stations, X, Y):
            plt.text(xpt, ypt, str(s.code))
    elif text == 'name':
        for s,xpt,ypt in zip(stations, X, Y):
            plt.text(xpt, ypt, str(s.name))


def stationScatter(stations, map, data, cmap):
    x = np.zeros(len(stations), float)
    y = np.zeros(len(stations), float)
    for i in range(len(x)-1):
        x[i] = stations[i].lon
        y[i] = stations[i].lat
    sc = gammaScaler(data,16)
  
    collection = map.scatter(x,y,c=data,marker='o',cmap=cm)
    plt.colorbar(collection)


#####################################################################################

if __name__ == '__main__':
    # make test for the distance
    print('hello')
    sts = readStations('d:\\data\\measurements\\pollen\\EAN\\stations\\Stations_SILAM_20161212_Perm_nograv_noRhod_AIA.txt', 
                          columns={'latitude':4,'longitude':3,'code':2,'name':2})  # used to be name:7
#    print(sts)
    
    stLst = []
    for key in sorted(sts.keys()):
        stLst.append(sts[key])
    print(len(stLst))
        
    import supplementary as spp
    timer = spp.SFK_timer()
    timer.start_timer('dist')
    for t in range(10):
        dHaversine, dPlane, dError  = distance_matrix(stLst)
    timer.stop_timer('dist')
    timer.report_timers()

    
#    for ist in range(len(stLst)):
#        for ist2 in range(ist, len(stLst)):
#            print(stLst[ist].code, '(',stLst[ist].lon, stLst[ist].lat,')', 
#                  stLst[ist2].code, '(',stLst[ist2].lon, stLst[ist2].lat,')', 
#                  dHaversine[ist,ist2], dPlane[ist,ist2], dError[ist,ist2])

