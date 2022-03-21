"""
A module for handling SILAM output in grads and netcdf formats. The functions are as follows:
- the grid and vertical definition is parsed from the super-ctl/netcdf file
- a projection is defined using the grid information
- the variable descriptions are parsed.
"""

import numpy as np
from toolbox import projections
from toolbox import namelist, gradsfile, gridtools, verticals, ncreader, util
from support import netcdftime
import netCDF4 as netcdf
#from support import pupynere as netcdf
import datetime as dt
from _ast import Raise
try:
    import netCDF4 as nc4
    from toolbox import nc4reader
except:
    nc4 = None
    
from os import path, listdir

class SilamfileError(Exception):
    pass

class SilamLatLonGrid(gridtools.Grid):
    """A gridtools.Grid subclass defining the SILAM latlon grid with
    interfaces to the namelist objects"""
    def __init__(self, *args):
        gridtools.Grid.__init__(self, *args)
        # print(self.proj.south_pole)
        try:
            self.proj.south_pole
            self.grid_type = 'LON_LAT'
        except AttributeError:
            raise ValueError('The projection does not appear to be lat/lon')        
                        
    def as_namelist(self):
        hash = {'grid_type' : self.grid_type,
                'grid_title' : '',
                'lon_s_pole' : self.proj.south_pole[0],
                'lat_s_pole' : self.proj.south_pole[1],
                'nx' : self.nx,
                'ny' : self.ny,
                'dx' : self.dx,
                'dy' : self.dy,
                'resol_flag' : 128,
                'ifReduced' : 0,
                'wind_component' : 0,
                'reduced_nbr_str' : 0,
                'lat_pole_stretch' : 0.0,
                'lon_pole_stretch' : 0.0,
                'lon_start' : self.x0,
                'lat_start' : self.y0
                }
        nl = namelist.Namelist('grid')
        for key, val in hash.iteritems():
            nl.put(key, val)
        return nl

    @staticmethod
    def from_namelist(nl):
        dx, dy, lon_start, lat_start = (float(nl.get_uniq(key))
                                        for key in ('dx', 'dy', 'lon_start', 'lat_start'))
        try:
            nx, ny = int(nl.get_uniq('nx')), int(nl.get_uniq('ny'))
        except KeyError:
            lon_end, lat_end = float(nl.get_uniq('lon_end')), float(nl.get_uniq('lat_end'))
            nx, ny = (lon_end-lon_start) / dx + 1, (lat_end-lat_start) / dy + 1
            nx, ny = int(nx), int(ny)
            
        sp_lon, sp_lat = float(nl.get_uniq('lon_s_pole')), float(nl.get_uniq('lat_s_pole'))
        proj = projections.LatLon(sp_lat, sp_lon)
        grid = SilamLatLonGrid(lon_start, dx, nx, lat_start, dy, ny, proj)
        return grid

LatLonGrid = SilamLatLonGrid
    
#############################################################################

class SilamVertical:
    """Definitions for the verticals appearing in SILAM output. So far
    only the simple cases are considered - the vertical is assumed to be
    described by a 1d list of numbers."""
    def number_of_levels(self):
        return len(self.values)

    def as_namelist(self):
        nl = namelist.Namelist('vertical')
        if self.number_of_levels() > 0:
            nl.put('number_of_levels', self.number_of_levels())
            nl.put(self.__class__.values_label, ' '.join('%g' % x for x in self.values))
        nl.put('vertical_method', self.__class__.vertical_method.upper())
        if self.__class__.level_type:
            nl.put('level_type', self.__class__.level_type.upper())
        return nl

    @classmethod
    def from_namelist(cls, nl):
        # This function defines the vertical from a namelist. The
        # attributes taken depend on the subclass.
        levtype, method = nl.get_uniq('level_type'), nl.get_uniq('vertical_method')
        values = [float(x) for x in nl.get_uniq(cls.values_label).split()]
        return cls(values)

    def num_levels(self):
        return len(self.values)

    def unit(self):
        return self.__class__.unit

    def values(self):
        return self.values

#-----------------------------------------------------------------------
    
class SilamHybrid: # Nothing to inherit from SilamVertical
    # To create a vertical class, one defines these three class
    # attributes. They are used for the common operations defined in
    # the SilamVertical class.
    vertical_method = 'custom_layers'
    level_type = 'hybrid'
    values_label = 'hybrid_coefficients_bottom'
    values_top_label = 'hybrid_coefficients_domain_top'
    unit = 'Pa'
    
    def __init__(self, a_half, b_half):
        self.a_half = tuple(a_half)
        self.b_half = tuple(b_half)

    def number_of_levels(self):
        return len(a_half) - 1  #Level boundaries stored

    def as_namelist(self):
        nlevs = self.number_of_levels()
        nl = namelist.Namelist('vertical')

        if nlevs > 0:
            nl.put('number_of_levels', nlevs)
            for i in range(nlevs):
                  nl.put(self.__class__.values_label, '%d %g %g' %(i,a_half[i], b_half[i]))
            nl.put(self.__class__.values_top_label, '%d %g %g' %(i,a_half[nlevs+1], b_half[nlevs+1]))

        nl.put('vertical_method', self.__class__.vertical_method.upper())
        if self.__class__.level_type:
            nl.put('level_type', self.__class__.level_type.upper())
        return nl

    @classmethod
    def from_namelist(cls, nl):
        # This function defines the vertical from a namelist. The
        # attributes taken depend on the subclass.
        levtype, method = nl.get_uniq('level_type'), nl.get_uniq('vertical_method')
        if levtype.lower() == "hybrid":
            print("Hybrid vertical")
        else:
            print("SilamHybrid initializing with not 'hybrid' level type")
            raise SilamfileError
            return None

        a_half = []
        b_half = []
        for l in nl.get(cls.values_label):
                n, a, b = l.split()
                a_half.append(a)
                b_half.append(b)
        l=nl.get_uniq(cls.values_top_label)
        a, b = l.split()
        a_half.append(a)
        b_half.append(b)
        return SilamHybrid(a_half,b_half)


    def num_levels(self):
        return len(self.a_half) - 1 

    def unit(self):
        return self.__class__.unit

    def values(self):
        return tuple([ (self.a_half[i] + self.a_half[i+1])*0.5 + (self.b_half[i] + self.b_half[i+1])*0.5*101325
                       for i in range(self.number_of_levels())])

    def thickness(self):
        return tuple([ (self.a_half[i] - self.a_half[i+1]) + (self.b_half[i] - self.b_half[i+1])*101325
                       for i in range(self.number_of_levels())])

    def boundaries(self):
        return tuple([ self.a_half[i] + self.b_half[i]*101325 for i in range(self.number_of_levels()+1)])

    def midpoints(self):
        return self.values

#-----------------------------------------------------------------------

class SilamHeightLevels(SilamVertical):
    # To create a vertical class, one defines these three class
    # attributes. They are used for the common operations defined in
    # the SilamVertical class.
    vertical_method = 'custom_levels'
    level_type = 'height_from_surface'
    values_label = 'levels'
    unit = 'm'
    
    def __init__(self, midpoints):
        self._midpoints = tuple(midpoints)
        self.values = self._midpoints

    def thickness(self):
        return verticals.thickness(self._midpoints)

    def boundaries(self):
        return verticals.boundaries(self.thickness())

    def midpoints(self):
        return self.values
    
#-----------------------------------------------------------------------

class SilamHeightLayers(SilamVertical):
    vertical_method = 'custom_layers'
    level_type = 'height_from_surface'
    values_label = 'layer_thickness'
    unit = 'm'
    
    def __init__(self, thickness):
        self._thickness = tuple(thickness)
        self.values = self._thickness
        
    def boundaries(self):
        return verticals.boundaries(self._thickness)
        #raise SilamfileError('not implemented')

    def midpoints(self):
        return verticals.midpoint(self._thickness)

    def thickness(self):
        return self._thickness
      
#-----------------------------------------------------------------------

class SilamSurfaceVertical(SilamVertical):
    vertical_method = 'surface_level'
    level_type = None
    values_label = None
    unit = ''
    def __init__(self, *args):
        self.values = 0.0,
    def thickness(self):
        raise ValueError()
    def boundaries(self):
        raise ValueError()
    @staticmethod
    def from_namelist(nl):
        return SilamSurfaceVertical()
    
#-----------------------------------------------------------------------

def get_vertical_class(vertical_method, level_type):
    # Return the correct class for vertical definition based on the
    # two values (typically from a namelist).
    classes = SilamHeightLevels, SilamHeightLayers, SilamSurfaceVertical, SilamHybrid
    for cls in classes:
        if cls.vertical_method == vertical_method.lower() and (cls.level_type == level_type.lower()
                                                               or not cls.level_type):
            return cls
    raise SilamfileError('Unsupported vertical: %s %s' % (vertical_method, level_type))

#############################################################################

class ProjectedReader(gradsfile.GriddedDataReader, gradsfile.GenericWrapper):
    """
    A GriddedDataReader with a nontrivial projection, defined by the
    gridtools.Grid instance. Can wrap any GriddedDataReader object,
    and overloads only the meshgrid and indices methods. These methods
    will now be based on the geographic coordinates, while x() and y()
    will return the grid-native coordinates. This serves the two
    common uses: plotting maps and extracting timeseries.
    """
    def __init__(self, reader, grid):
        methods =  '_get_geo_coords', 'coordinates', 'meshgrid', 'indices' #, 'nx', 'ny'
        gradsfile.GenericWrapper.__init__(self, reader, methods)
        self._orig_descr = reader.describe()
        self.grid = grid
        self._get_geo_coords()
        
    def _get_geo_coords(self):
        # midpoints
        X, Y = gradsfile.GriddedDataReader.coordinates(self)
        shape = X.shape
        lon, lat = self.grid.proj_to_geo(X.ravel(), Y.ravel())
        self.lon = lon.reshape(shape)
        self.lat = lat.reshape(shape)
        # vertices
        X, Y = gradsfile.GriddedDataReader.coordinates(self, True)
        shape = X.shape
        lon, lat = self.grid.proj_to_geo(X.ravel(), Y.ravel())
        self.lon_vx = lon.reshape(shape)
        self.lat_vx = lat.reshape(shape)

    def coordinates(self, vertices=False):
        if vertices:
            return self.lon_vx, self.lat_vx
        else:
            return self.lon, self.lat
        
    def meshgrid(self, vertices=False):
        warnings.warn('meshgrid() should be replaced by coordinates()')
        if vertices:
            return self.lon_vx.T, self.lat_vx.T
        else:
            return self.lon.T, self.lat.T
        
    def indices(self, lon, lat):
        # Project lon, lat into the native coordinates and return the indices
        x, y = self.grid.geo_to_grid(lon, lat)
        #return int(x+0.5), int(y+0.5)
        # better when x, y can be negative:
        return np.round(x).astype(int), np.round(y).astype(int)

    def describe(self):
        description = 'PROJ(%s)' % self._orig_descr
        return description

#    def nx(self): return len(self.lon)
#    def ny(self): return len(self.lat)

#############################################################################

class BaseSilamfile:
    def _check_match(self, metadata, key, request, get_value):
        # the attribute matching rules are here:
        match = True
        try:
            request.__call__
            request_callable = True
        except AttributeError:
            request_callable = False
        testval = get_value(metadata, key).split()[0]
        if request_callable:
            try:
                match = match and request(testval)
            except ValueError:
                return False
        else:
            reqval = request
            valtype = reqval.__class__
            try:
                testval = valtype(testval)
            except ValueError:
                return False
            if isinstance(testval, float):
                match = match and np.abs(testval-reqval) < 1e-6
            else:
                match = match and testval == reqval

        return match

    def get_variable(self, **conditions):
        """ A convenience version of get_variables: return a single variable, and if
        more than one matches, raise an error. """
        
        selected = self.get_variables(**conditions)
        if len(selected) > 1:
            raise SilamfileError('More than one variable matches')
        return selected[0]

#############################################################################
    
class Silamfile(BaseSilamfile):
    """A class for parsing and describing the information in a silam super
    ctl file: the grid, vertical and variable descriptions. Does not read
    the data, that is handled by requesting a reader object with a given
    grads variable expression.

    Arguments:
    superctl : the superctl file path
    relocate : modify the path of the ctl file. If True, assume that
               the ctl is in the same directory as the superctl. If a string,
               use the given directory. Also the binary files will be relocated
               accordingly."""
    
    def __init__(self, superctl, relocate=None):
        nlgrp = namelist.NamelistGroup.fromfile(superctl)
        nl_general = nlgrp.get('general')
        grid = SilamLatLonGrid.from_namelist(nl_general)
        vertical_class = {('custom_layers', 'height_from_surface') : SilamHeightLayers,
                          ('custom_levels', 'height_from_surface') : SilamHeightLevels}
        if not nl_general.has_key('level_type'):
            vertical = None
        else:
            vert_method, vert_type = (nl_general.get_uniq('vertical_method'),
                                      nl_general.get_uniq('level_type'))
            vertical_class = get_vertical_class(vert_method, vert_type)
            vertical = vertical_class.from_namelist(nl_general)

        ctlpath_original = nl_general.get_uniq('ctl_file_name')
        if ctlpath_original.startswith('^'):
            ctlpath_original = path.join(path.dirname(superctl), ctlpath_original[1:])
        if relocate is True:
            ctlpath = path.join(path.dirname(superctl), path.basename(ctlpath_original))
            self.gradsdescriptor = gradsfile.GradsDescriptor(ctlpath)
            self.gradsdescriptor.relocate()
        elif relocate:
            ctlpath = path.join(relocate, path.basename(ctlpath_original))
            self.gradsdescriptor = gradsfile.GradsDescriptor(ctlpath)
            self.gradsdescriptor.relocate(relocate)
        else:
            ctlpath = ctlpath_original
            self.gradsdescriptor = gradsfile.GradsDescriptor(ctlpath)
            
        self.grid = grid
        self.vertical = vertical
        self.nlgrp = nlgrp
        self.ctlpath = ctlpath
        
    def get_ctl(self):
        return self.ctlpath
        
    def get_variables(self, **conditions):
        """Find variables from the superctl by matching a set of values given
        as keyword arguments. If a given variable does not have the specified
        attribute, it is never selected.
        
        The value read from the namelist is coerced to the type of the
        respective argument, if possible. This means that if the
        argument is a string, the values are compared as strings, same
        for floats and integers. If the comparison is impossible, the
        variable is ignored. Only the first word of the value is taken
        - units are ignored.
        
        Examples:
        
        # Find all PM variables (depositions, concentration, all modes)
        variables = silamf.get_variables(substance_name='PM')
        # Find all PM concentration with a mean diameter of 0.5 micron
        variables = silamf.get_variables(substance_name='PM', fix_diam_mode_mean_diameter=0.5)
        # This will probably not work
        variables = silamf.get_variables(substance_name='PM', fix_diam_mode_mean_diameter='0.5')
        
        """
        selected = []
        for nl in self.nlgrp.lists.values():
            if nl.name == 'general':
                continue
            match = True
            for key, request in conditions.items():
                if not nl.has_key(key):
                    match = False
                    break
                match = match and self._check_match(nl, key, request, get_value=lambda m, k: m.get_uniq(key))
            if match:
                selected.append(nl.name)
        return selected
    

    def get_reader(self, grads_expression, ind_level=None, projected=True, mask_mode='first',
                   grads_class=gradsfile.MMGradsfile):
        """Return a ProjectedReader reading the given grads variable expression on a given level."""
        grfile = grads_class(self.gradsdescriptor, grads_expression, ind_level, mask_mode)
        if projected:
            return ProjectedReader(grfile, self.grid)
        else:
            return grfile

    def get_var_namelist(self, var):
        for nl in self.nlgrp.lists.values():
            if nl.name == var:
                return nl
        raise SilamfileError('Variable not found: %s' % var)

    def get_general_param(self, key):
        nl_general = self.nlgrp.lists['general']
        return nl_general.get(key)

    def get_attribute(self, variable, key):
        nl = self.get_var_namelist(variable)
        value = nl.get_uniq(key)
        return value
        
#############################################################################
    
class SilamNCFile(BaseSilamfile):
    """
    Class for parsing variables from silam netcdf output. Netcdf-4 requires the NetCDF4
    python library and the nc4reader toolbox module. Variables can be selected by matching
    the attributes. For more info see Silamfile.
    """
    
    def __init__(self, input_file):

        """
        Create a parser object - either from a netcdf file or a .ctl file. In the latter
        case, the first file pointed by the ctl will be examined, and if a reader is
        requested, it will include the whole dataset.

        Now can detect if the file is .ctl or NetCDF. nc4 used in all cases now.
        """
        nc_class = nc4 and nc4.Dataset

        chType = self._detect_nc_type(input_file)
        if chType == 'ctl':
            self.descr = ncreader.NCDescriptor.fromfile(input_file)
            netcdf_file = self.descr.get_file(self.descr.times()[0])
            self.descr_path = input_file
            self.reader_class = nc4 and nc4reader.NC4Dataset
        elif chType == 'template':         # a collection of files described by a template
            timelist = self.get_timedim_from_file_set(input_file, nc_class)
            self.descr = ncreader.NCDescriptor(file=None, shift=None, template=input_file, 
                                               time_dim=None, timelist=timelist)
            netcdf_file = self.descr.get_file(self.descr.times()[0])
            self.descr_path = None
            self.reader_class = nc4 and nc4reader.NC4Dataset
        else:
            netcdf_file = input_file
            self.netcdf_path = netcdf_file
            self.descr_path = None
            self.descr = None
            self.reader_class = nc4 and nc4reader.NC4Reader
            self.reader_class_expr = nc4 and nc4reader.NC4Expression
        try:
            nc_file = nc_class(netcdf_file, 'r')
        except:
            print ('Failed to open:', netcdf_file)
            raise

        self._parse_grid_vert(nc_file)
        self._parse_attributes(nc_file)
        nc_file.close()


    def _detect_nc_type(self, fname):
        if '%' in fname:
            return 'template'
        with open(fname,'rb') as f:
            l=f.read(4)
        if l == b'DSET':
            return 'ctl'
        elif l == b'CDF\x01':
            return 'nc3'
        elif l == b'CDF\x02':
            return 'nc3' ## 64bit offset
        elif l == b'\211HDF':
            return 'nc4'
        else:
            print (l) ##123
            raise ValueError("Unknown magic for file %s"%(fname))


    def get_timedim_from_file_set(self, input_file, nc_class):
        # get all files described by the template, find the first and last in time
        lstFiles, file_1st, file_last = gradsfile.files_and_times_4_template(input_file)
        # open the first and identify the time step and start time
        timelist = []
        for fnm in lstFiles:
            nc_file = nc_class(fnm, 'r')
            tvals = nc_file.variables['time'][:]
            tunits = nc_file.variables['time'].units
            ### JP for CAMS files
            try:
                if ((not 'since' in nc_file.variables['time'].long_name) and 
                    ('FORECAST' in nc_file.variables['time'].long_name or 
                     'ANALYSIS' in nc_file.variables['time'].long_name)):
                    dcurr = nc_file.variables['time'].long_name.split()[-1]
                    dyyr = dcurr[0:4]
                    dmon = dcurr[4:6]
                    dday = dcurr[6:]
                    newUnits = '%s since %s-%s-%s 00:00:00 UTC'%(tunits,dyyr,dmon,dday)
                    timelist += list(netcdftime.num2date(tvals, newUnits, calendar='standard'))
                else:
            #### JP.....ended
                    timelist += list(netcdftime.num2date(tvals, tunits))
            except:
                timelist += list(netcdftime.num2date(tvals, tunits))
        return sorted(timelist)
#        if len(hrlist1) == 1:  # one time step per file
#            return (hrlist[0],                # first time
#                    len(lstFiles),            # n_times
#                    (hrlist2[0] - hrlist1[0]) / (len(lstFiles) - 1))  # timestep
#        else:     # several steps in one file
#            return (hrlist1[0], 
#                    (hrlist2[-1] - hrlist1[0]) / (hrlist1[1] - hrlist1[0]) + 1, 
#                    hrlist1[1] - hrlist1[0])
        
        
    def _parse_grid_vert(self, nc_file):
        rotated = False
        if 'lon' in nc_file.dimensions and 'lat' in nc_file.dimensions:
            lonname='lon'
            latname='lat'
        elif 'longitude' in nc_file.dimensions and 'latitude' in nc_file.dimensions:
            lonname='longitude'
            latname='latitude'
        elif 'rlon' in nc_file.dimensions and 'rlat' in nc_file.dimensions:
            lonname='rlon'
            latname='rlat'
            rotated = True
        else:
            raise ValueError('Cannot determine grid')

        try:
          if nc_file.grid_projection != 'lonlat':
            raise ValueError('Projection not supported: %s' % nc_file.grid_projection)
        except AttributeError:
            print ("File has no grid_projection attribute, assuming lonlat and hope for the best")
            pass
            
        if rotated:
            try:
                sp_lon, sp_lat = nc_file.pole_lon, nc_file.pole_lat
            except AttributeError:
                rpvar=nc_file.variables['rp']
                assert rpvar.grid_mapping_name == "rotated_latitude_longitude"
                sp_lon = (rpvar.grid_north_pole_longitude + 360.) % 360.  - 180.
                sp_lat = - rpvar.grid_north_pole_latitude
            proj = projections.LatLon(sp_lat, sp_lon)
        else:
            proj = projections.LatLon()
        lon, lat = nc_file.variables[lonname][:], nc_file.variables[latname][:]
        nx, ny = len(lon), len(lat)
        # Take care of the grid with 360 deg jump of the longitude
        delta = lon[1] - lon[0]
        if not np.all(util.almost_eq(lon[1:]-lon[:-1], delta, 1e-4)):    # if non-uniform
            # try to roll +- 360 degrees to eliminate possible jump
            lon[lon > 180] -= 360
            lon[lon < -180] += 360
            if not np.all(util.almost_eq(lon[1:]-lon[:-1], delta, 1e-4)):
                print('Strange file: cannot get continuous longitude')
                idxMax = np.argmax(np.abs(lon[1:]-lon[:-1]))
                print(lon[max(0,idxMax-10):min(len(lon),idxMax+10)], '\n Uniformity 1e-4:',
                      util.almost_eq(lon[1:]-lon[:-1], delta, 1e-4)[max(0,idxMax-10):min(len(lon),idxMax+10)])
                if not np.all(util.almost_eq(lon[1:]-lon[:-1], delta, 1e-2)):
                    print('REALLY strange file: cannot get continuous longitude even roughly')
                    print(lon, '\n Uniformity 1e-2:', util.almost_eq(lon[1:]-lon[:-1], delta, 1e-2))
                    raise ValueError
        # Try to recover the accuracy
        dx, dy = (lon[-1]-lon[0])/(nx-1), (lat[-1]-lat[0])/(ny-1)
        dx = 1e-5*round(dx*1e5)
        dy = 1e-5*round(dy*1e5)
        
        self.grid = SilamLatLonGrid(lon[0], dx, nx, lat[0], dy, ny, proj)

        if 'height' in nc_file.dimensions:
           hgt = nc_file.variables['height'][:]
           self.vertical = SilamHeightLevels(hgt)
        elif 'hybrid' in nc_file.dimensions:
           try:
               self.vertical = SilamHybrid(nc_file.variables['a_half'][:],nc_file.variables['b_half'][:])
           except Exception as e:
               print(str(e))
               print("Failed to parse hybrid vertical pretending it is height")
               # Pretend that it is still height -- does not matter for the stations
               hgt = nc_file.variables['hybrid'][:]
               self.vertical = SilamHeightLevels(hgt)
               pass
        else:
           self.vertical=None

    _GLOB = 'glob'
    def _parse_attributes(self, nc_file):
        attr = {}
        for var_name, ncvar in list(nc_file.variables.items()) + [(self._GLOB, nc_file)]:
            if not var_name in attr:
                attr[var_name] = {}
            for fldname in dir(ncvar):
#                print(1, fldname)
                if fldname.startswith('__'):
                    continue
                fldval = getattr(ncvar, fldname)
#                print(2, fldval, type(fldval), isinstance(fldval, str) or isinstance(fldval, float) or isinstance(fldval, int)
#                      or isinstance(fldval, np.dtype))
                # take all float, int or character attributes.
                if not (isinstance(fldval, str) or isinstance(fldval, float) or isinstance(fldval, int)
                        or isinstance(fldval, np.dtype) or isinstance(fldval, np.float32)
                        or isinstance(fldval, np.float64)):
                    continue
                attr[var_name][fldname] = fldval
#                print(3, var_name, fldname, fldval)
        self._attrs = attr


    def get_variables(self, **conditions):
        selected = []
        for var_name, var_attrs in self._attrs.items():
            match = True
            for key, request in conditions.items():
                if not key in var_attrs:
                    match = False
                    break
                match = match and self._check_match(var_attrs, key, request, get_value=lambda a, k: a[k])
            if match:
                selected.append(var_name)
        return selected

    def get_reader(self, expression, ind_level=None, mask_mode='first'):
#        print('################### ordinary reader for a change')
        projected = True # Projected reader used always. 
#        if self.descr is not None:
#            reader = self.reader_class(self.descr, expression, ind_level, mask_mode=mask_mode)
#        elif ncreader.NCExpression.is_expression(expression):
#            reader = self.reader_class_expr(self.netcdf_path, expression, ind_level, mask_mode=mask_mode)
#        else:
#            reader = self.reader_class(self.netcdf_path, expression, ind_level, mask_mode=mask_mode)
        #
        # The name of nc variable can contain - or whatever. No reason to reply on this
        # Just check if the name is recognizable
        #
        try:
            reader = self.reader_class(self.descr, expression, ind_level, mask_mode=mask_mode)
        except:
            try:
                reader = self.reader_class(self.netcdf_path, expression, ind_level, mask_mode=mask_mode)
            except:
                reader = self.reader_class_expr(self.netcdf_path, expression, ind_level, mask_mode=mask_mode)                
        if projected:
            return ProjectedReader(reader, self.grid)
        else:
            return reader

    def get_general_param(self, key):
        return self._attrs[self._GLOB][key]

    def get_attribute(self, variable, key):
        return self._attrs[variable][key]
    
    def fill_value(self, variable):
        try:  # if fill_value is explicit
#            print('silamNCfile: Trying explicit fill value')
#            print(self._attrs[variable])
            return self._attrs[variable]['_FillValue']
        except:  # have to use default
            try: 
                return self._attrs[variable]._FillValue
            except:
                try:
                    print('silamNCfile: No explicit fill value, trying NetCDF default')
    #                print('Available attributes are: ', self._attrs[variable])
                    return netcdf.default_fillvals[{np.dtype('float32'):'f4',
                                                    np.dtype('float64'):'f8'}[self._attrs[variable]['dtype']]]
                except:
                    print('silamNCfile: Failed to find any fill value, do nothing and hope for the best')
                

#############################################################################
#
# Writing the netcdf file
#

def open_ncF_out(fNm, nctype, grid, vertical, anTime, arrTime, vars3d, vars2d, units, fill_value,
                 ifCompress, ppc=None, hst=None): 
    print('opening nc file ' + fNm)
    try:
        f = netcdf.Dataset(fNm, 'w', format=nctype.upper())  # e.g. "NETCDF4"
    except:
        print('>>>> Failed opening the nc file', fNm)
        try:
            print('>>>> list of open files:', os.listdir('/proc/self/fd'))
        except:
            print('>>>> Cannot print the list of open files')
        print('>>>> Second try, see the exception')
        f = netcdf.Dataset(fNm, 'w', format=nctype.upper())  # e.g. "NETCDF4"

    if hst: f.history = hst

    f.createDimension('lon', grid.nx)
    lon = f.createVariable('lon', 'f', ('lon',))
    lon[:] = grid.x()
    lon.units = 'degrees_east'
    lon.axis = "X"

    f.createDimension('lat', grid.ny)
    lat = f.createVariable('lat', 'f', ('lat',))
    lat[:] = grid.y()
    lat.units = 'degrees_north'
    lat.axis = "Y"

    if vertical is not None:
        f.createDimension('height', vertical.num_levels())
        level = f.createVariable('height', 'f', ('height',))
        level[:] = vertical.values
        level.units = 'height_from_surface'
        level.axis = "Z"

    f.createDimension('time', None)   #len(arrTime))  # to make it unlimited record dimension
    time = f.createVariable('time', 'i', ('time',))
    time[:] = [ (h - anTime).total_seconds() for h in arrTime ]  #    arrTime
    time.calendar = 'standard'
    time.units = anTime.strftime('seconds since %Y-%m-%d %H:%M:%S UTC')
    time.long_name = 'time'
    time.standard_name = 'time'
    time.axis="T"

    for varNm in vars3d:
        if ifCompress:
            f.createVariable(varNm, 'f', ('time', 'height', 'lat', 'lon'), zlib=True, complevel=5, 
#                             chunksizes=(1,1,min(1000,grid.ny),min(1000,grid.nx)),
                             chunksizes=(1,1,grid.ny,grid.nx),
                             least_significant_digit=ppc, fill_value=fill_value)
        else:
            f.createVariable(varNm, 'f', ('time', 'height', 'lat', 'lon'), fill_value=fill_value)
        f.variables[varNm].units = units[varNm]

    for varNm in vars2d:
        if ifCompress:
            f.createVariable(varNm, 'f', ('time', 'lat', 'lon'), 
                             zlib=True, complevel=5, chunksizes=(1,grid.ny,grid.nx), 
#                             zlib=True, complevel=5, chunksizes=(1,min(1000,grid.ny),min(1000,grid.nx)), 
                             least_significant_digit=ppc, fill_value=fill_value)
        else:
            f.createVariable(varNm, 'f', ('time', 'lat', 'lon'), fill_value=fill_value)
        f.variables[varNm].units = units[varNm]
    return f


#============================================================================

def write_grid_to_nc4(outF, grid, gridName):
    #
    # Writes the grid into an open netCDF4 file: uses the Group to pass the attributes
    # The grid is written in the CDO format
    #
    gridGroup = outF.createGroup('grid')
    gridGroup.setncattr('gridName', gridName)
    nlGrid = grid.toCDOgrid()
    for key in nlGrid.keys():
        gridGroup.setncattr(key, nlGrid.get_uniq(key))
    return

#============================================================================

def read_grid_from_nc4(inF):
    #
    # Writes the grid into an open netCDF4 file: uses the Group to pass the attributes
    # The grid is written in the CDO format
    #
    nl = namelist.Namelist('')
#    print(inF.groups['grid'].ncattrs())
    for attr in inF.groups['grid'].ncattrs():
        nl.add(attr, inF.groups['grid'].getncattr(attr))
    return (gridtools.fromCDOnamelist(nl), inF.groups['grid'].getncattr('gridName'))



#############################################################################

class SILAMASCIIfile(BaseSilamfile):
    """
    Class for parsing variables from silam ASCII files. For more info see Silamfile.
    """
    
    def __init__(self, input_file):
        """
        Create a parser object -
        """
        nlIn = namelist.Namelist.fromfile(input_file)
        
        if nlIn.get_uniq('grid_type').upper() != 'LON_LAT':
            print('Only lon_lat grids are allowed')
            return
        # get the grid
        lonStart = float(nlIn.get_uniq('lon_start'))
        latStart = float(nlIn.get_uniq('lat_start'))
        try: 
            dx = nlIn.get_float('dx')[0]
            dy = nlIn.get_float('dy')[0]
            nx = nlIn.get_int('nx')[0]
            ny = nlIn.get_int('ny')[0]
        except:
            lonEnd = nlIn.get_float('lon_end')[0]
            latEnd = nlIn.get_float('lat_end')[0]
            try:
                nx = nlIn.get_int('nx')[0]
                ny = nlIn.get_int('ny')[0]
                dx = (lonEnd - lonStart) / nx
                dy = (latEnd - latStart) / ny
            except:
                dx = nlIn.get_float('dx')[0]
                dy = nlIn.get_float('dy')[0]
                nx = int(round((lonEnd - lonStart) / dx)) + 1
                ny = int(round((latEnd - latStart) / dy)) + 1
        try:
            proj = projections.LatLon(nlIn.get_float('lat_s_pole')[0], 
                                      nlIn.get_float('lon_s_pole')[0])
        except:
            proj = projections.LatLon(-90., 0.)

        self.grid = SilamLatLonGrid(lonStart, dx, nx, latStart, dy, ny, proj)
        
        # Now, the vertical. Note that ASCII field is a single-layer or level
        if nlIn.get_uniq('vertical_method').upper() == 'SURFACE_LEVEL':
           self.vertical = SilamHeightLevels([0.0])
        elif nlIn.get_uniq('vertical_method').upper() == 'height':
           levs = nlIn.get('level')
           for i in len(levs): levs[i] = float(levs[i])
           self.vertical = SilamHeightLevels(levs)
        else:
           self.vertical=None
        
        # other features
        self.ifGeoCoord = nlIn.get_uniq('coordinate_of_values').upper() == 'GEOGRAPHICAL'
        self.quantity = nlIn.get_uniq('quantity_name')
        self.substance = nlIn.get_uniq('substance_name')
        self.unit = nlIn.get_uniq('unit')
        self.nominal_mode_diameter = nlIn.get_uniq('fix_diam_mode_mean_diameter')
        self.missing = nlIn.get_float('missing_value')[0]
        anal_time_str = nlIn.get_uniq('analysis_time').split()
        fcst_str = nlIn.get_uniq('forecast_length').split()
        self.anal_time = dt.datetime(int(anal_time_str[0]),int(anal_time_str[1]),int(anal_time_str[2]),
                                int(anal_time_str[3]),int(anal_time_str[4]))
        self.fcst = dt.timedelta(minutes = int(fcst_str[0]))
        if fcst_str[1] == 'min':
            pass
        elif fcst_str[1] == 'hr':
            self.fcst = self.fcst * 60.0
        elif fcst_str[1] == 'sec':
            self.fcst = self.fcst / 60.0
        elif   fcst_str[1] == 'day':
            self.fcst = self.fcst * 24.0 * 60.0
        else:
            print('Unknown forecast line: ', fcst_str)
            return
        # Finally, values
        print(nx, ny)
        self.vals = np.ones(shape=(nx,ny)) * self.missing
        if self.ifGeoCoord:
            for valline in nlIn.get('val'):
                flds = valline.split()
                lon = float(flds[0])
                lat = float(flds[1])
                ix, iy = self.grid.geo_to_grid(lon,lat)
                self.vals[int(np.round(ix)),int(np.round(iy))] = float(flds[2])
        else:
            for valline in nlIn.get('val'):
                flds = valline.split()
                self.vals[int(flds[0]),int(flds[1])] = float(flds[2])

    #================================================================================

    def read(self):
        return self.vals




#######################################################################################
            
if __name__ == '__main__':
    superctl = '/home/vira/silam/output/blh/blh_ALL_SRCS_20101001.grads.super_ctl'
    superctl = '/home/vira/work/macc_ii/r-eda-env/wrk/test_cb4.02/iteration/gradient_96.grads.super_ctl'
    silamf = Silamfile(superctl)
    variables = silamf.get_variables(substance_name='O3')
    print(variables)
    #expr = '+'.join(variables[:-1])
    expr = '2*%s' % variables[0]
    reader = silamf.get_reader(expr, 0)
