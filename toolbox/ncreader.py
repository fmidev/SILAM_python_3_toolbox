"""
ncreader.py - gradsfile.GriddedDatareader interface to netcdf files.

This allows reading netcdf with an interface similar to Gradsfiles. Depending the situation, 
the following classes can be used:

- NCReader - read single variable from single netcdf file
- NCExpression - read an expression of variables from a single netcdf file.
- NCDescriptor + NCDataset - read a variable or expression from a 
  collection of netcdf files referred by a grads-like descriptor file.
  
The netcdf support is far from complete, for instance only relative time axes are supported, 
etc.
"""

import numpy as np, datetime as dt, re, warnings

from toolbox import gradsfile, util
from support import pupynere as netcdf
from support import netcdftime

from os import path

def parse_epoch(unit_str, return_unit=False):
    words = unit_str.split()
    timeunit = words[0]
    if not words[0] in ('seconds', 'hours'):
        raise NCError('Time unit must be hours or seconds')

    fmt_date, fmt_time = '%Y-%m-%d', '%H:%M:%S'
    if len(words) == 4:
        fmt_zone = ''
    elif len(words) == 5:
        fmt_zone = '%Z'
    else:
        raise NCError('Cannot handle time unit: %s' % unit_str)
    time = words[3]
    if '.' in time:
        fmt_msec = '.%f' # in python 2.6
    else:
        fmt_msec = ''
    fmt = '%s %s%s %s' % (fmt_date, fmt_time, fmt_msec, fmt_zone)
    if not return_unit:
        return dt.datetime.strptime(' '.join(words[2:]), fmt.rstrip())
    else:
        return dt.datetime.strptime(' '.join(words[2:]), fmt.rstrip()), timeunit
    
def make_epoch(ref_time, incr_unit):
    fmt_date, fmt_time = '%Y-%m-%d', '%H:%M:%S'
    datestr, timestr = ref_time.strftime(fmt_date), ref_time.strftime(fmt_time)
    epochstr = '%s since %s %s' % (incr_unit, datestr, timestr)
    return epochstr

class NCError(Exception):
    pass
class SeekPastFileError(NCError):
    pass


class NCDescriptor(gradsfile.GradsDescriptor):
    """ A descriptor aggregating several netcdf files split by time. Created from a file
    using the factory method below.
    """

    def __init__(self, file=None, shift=None, template=None, time_dim=None, timelist=None):
        self.file = file
        self.shift = shift
        self.template = template
        self.time_dim = time_dim
        if timelist is not None:
            gradsfile.GradsDescriptor.times = timelist.copy()
            self.timelist = sorted(timelist).copy()
            self.n_times = len(self.timelist)
            self.first_time = self.timelist[0]
            if self.n_times > 2:
                if np.all(np.array(self.timelist[1:]) - 
                          np.array(self.timelist[:-1]) == self.timelist[1]-self.timelist[0]):
                    self.timestep = self.timelist[1]-self.timelist[0]
                    self.const_time = True
                else:
                    self.const_time = False
            elif self.n_times == 2:
                self.timestep = self.timelist[1]-self.timelist[0]
                self.const_time = True
            else:
                self.const_time = False  # want to rely solely on timelist
            

    @staticmethod
    def fromfile(file_name):
        descr = NCDescriptor()
        descr.file = file_name
        descr.shift = None
        with open(file_name) as input:
            for line in input:
                lower = line.lower().lstrip()
                words = line.split()
                if lower.startswith('dset'):
                    dset = words[1]
                    if dset.startswith('^'):
                        dset = path.join(path.dirname(file_name), dset[1:])
                    descr.template = dset
                elif lower.startswith('tdef') and len(words) == 6:
                    # with the dimension name
                    descr.time_dim = words[1]
                    descr.n_times = int(words[2])
                    if not words[3].lower() == 'linear':
                        raise NCError('Only linear tdef is supported')
                    descr.first_time = gradsfile.parseGradsTime(words[4])
                    descr.timestep = gradsfile.parseGradsInterval(words[5])
                elif lower.startswith('tdef') and len(words) == 5:
                    # without
                    descr.time_dim = None
                    descr.n_times = int(words[1])
                    if not words[2].lower() == 'linear':
                        raise NCError('Only linear tdef is supported')
                    descr.first_time = gradsfile.parseGradsTime(words[3])
                    descr.timestep = gradsfile.parseGradsInterval(words[4])
                    
                elif lower.startswith('template_shift'):
                    value, unit = int(words[1]), words[2]
                    if unit != 'hr':
                        raise ValueError('shift unit must be hr')
                    descr.shift = dt.timedelta(hours=value)
        if not hasattr(descr, 'timestep'):
            raise ValueError('Failed to parse descriptor: %s' % file_name)
        else:
            if descr.timestep is None:
                raise ValueError('Failed to parse descriptor, None timetep: %s' % file_name)

        descr.const_time = descr.timestep is not gradsfile.GradsDescriptor.MONTHLY
        if descr.const_time:
            descr.timelist = [descr.first_time + ii*descr.timestep 
                              for ii in range(descr.n_times)]
        else:
            descr.timelist = list(util.iterate_monthly(descr.first_time, descr.n_times))
        return descr
    
    def get_file(self, valid_time):
        if self.shift is not None:
            return gradsfile.expandGradsTemplate(self.template, valid_time + self.shift)
        else:
            return gradsfile.expandGradsTemplate(self.template, valid_time)

    def times(self):                    
        return self.timelist
    
# The following list maps the possible dimension names in to the canonical ones used 
# in this module. Note that currently, a dimension is always required to be a variable. 

_dim_map = {'lon':'x', 'x':'x', 'lat':'y', 'y':'y', 'height':'z', 'z':'z',
           'time':'time', 'longitude' : 'x', 'latitude' : 'y', '2d_vert' : 'z', 'hybrid' : 'z',
           'lev' : 'z', 'level':'z', 'rlat':'y', 'rlon':'x',}

        
class NCReader(gradsfile.GriddedDataReader, object):
    no_flip = None
    transpose = 't'

    @classmethod
    def get_variables(cls, *args):
        return cls._get_variables(*args)
    
    @classmethod
    def _get_variables(cls, nc_path):
        ncf = netcdf.netcdf_file(nc_path, 'r')
        variables = list(ncf.variables.keys())
        non_dim = [var for var in variables if not var in ncf.dimensions]
        ncf.close()
        return non_dim
    
    def __init__(self, nc_file_or_descr, var_name, level=None, select=None, dimensions=_dim_map, mmap=True,
                 mask_mode='full', apply_scale_factor=True, apply_add_offset=True):
        if isinstance(nc_file_or_descr, str):
            self.nc_file = netcdf.netcdf_file(nc_file_or_descr, mmap=mmap)
            self.path = nc_file_or_descr
            self._close_on_close = True
        else:
            try:
                nc_file_or_descr.variables
                self.nc_file = nc_file_or_descr
            except AttributeError:
                raise NCError('The first argument is of wrong type')
            try:
                self.path = self.nc_file.filename
            except AttributeError:
                self.path = 'unknown'
            # If called with a nc_file object, the object will not be closed on close()
            self._close_on_close = False

        if not var_name in self.nc_file.variables.keys():
            raise NCError('No such variable: %s' % var_name)
        self.ncvar = self.nc_file.variables[var_name]
        if apply_scale_factor and hasattr(self.ncvar, 'scale_factor'):
            self._scale_factor = self.ncvar.scale_factor
        else:
            self._scale_factor = None
        if apply_add_offset and hasattr(self.ncvar, 'add_offset'):
            self._add_offset = self.ncvar.add_offset
        else:
            self._add_offset = None
        self.var_name = var_name
        
        self.have_vertical = False

        # Map the required dimensions into the netcdf ones. The
        # canonical names in this module are x, y, z and time. The
        # dimensions dictionary is to map the netcdf dimension names
        # into these.
        ncdims = self.ncvar.dimensions
        for ind, dim in enumerate(ncdims):
            if not dim in dimensions:
                raise NCError('Strange dimension: %s' % dim)
            if not self.have_vertical:
                self.have_vertical = dimensions[dim] == 'z'
            # Set some class attributes to allow easy access to the
            # indices and dimension names.
            self.__setattr__('_'+ dimensions[dim] + '_dim', dim)
            self.__setattr__('_ind_'+dimensions[dim] + '_dim', ind)
            #print(dimensions[dim], dim, ind)
        #
        # Time dimension of the variable might not exist
        #
        calendar='standard'
        try:
            self.ifTimeDim = True
            time_var = self.nc_file.variables[self._time_dim]
            if hasattr(time_var, "calendar"): calendar = time_var.calendar
            try:
                ### JP for CAMS pollen fc
                if (not 'since' in time_var.units) and ('FORECAST' in time_var.long_name or 
                                                        'ANALYSIS' in time_var.long_name):
                    dcurr = time_var.long_name.split()[-1]
                    dyyr = dcurr[0:4]
                    dmon = dcurr[4:6]
                    dday = dcurr[6:]
                    newUnits = '%s since %s-%s-%s 00:00:00 UTC'%(time_var.units,dyyr,dmon,dday)
                    self._times = netcdftime.num2date(time_var[:], newUnits, calendar=calendar)
                #### JP.....ended
                else:
                    self._times = netcdftime.num2date(time_var[:], time_var.units, calendar=calendar)
            except:
                self._times = netcdftime.num2date(time_var[:], time_var.units, calendar=calendar)
        except:
            self.ifTimeDim = False
            # time is not a dimension of the variable. Fake a single-time dimension
            time_var = self.nc_file.variables['time']
            self._times = netcdftime.num2date([time_var[0]], time_var.units, calendar=calendar)
#            self._ind_time_dim = len(ncdims)

        one_second=dt.timedelta(seconds=1)
        have_const_timestep = (len(self._times) > 1)
        if have_const_timestep:
          tstep = self._times[1] - self._times[0]
          have_const_timestep = all(np.abs((self._times[1:] - self._times[:-1])-tstep) < one_second)
        if  have_const_timestep:
          self.timestep = tstep 
          self.timestep_sec = tstep.total_seconds()

        else:
           self.timestep = self.timestep_sec = None
           if len(self._times) == 1:
                self.timestep_sec = 0.0
                self.timestep = dt.timedelta(seconds=self.timestep_sec)
            
        self.have_const_timestep = have_const_timestep
        self.first_time = self._times[0]
        self._time = self._times[0]
        self.n_times = len(self._times)
        self._last_time = self._times[-1]
        self._at_eof = False #self._time == self._last_time
        self.mask_mode = mask_mode
        self._first_mask = None # for the 'first' mask_mode
        #
        # time sorter
        if self.first_time < self._last_time:
            self.time_sorter = None
        else:
            self.time_sorter = np.arange(self.n_times, dtype=np.int64)[::-1]
            
        
        # Define the 4D dimension slice: to read a field, substitute
        # the time index in the slot ind_time_dim. If a level has been
        # requested, ind_z_dim is fixed at this point. In the
        # end, the field is read simply as field = variable[self._slice].
        if self.ifTimeDim:
            if self.have_vertical:
                self._slice = [slice(None), slice(None), slice(None), slice(None)]
                if level is not None:
                    self._slice[self._ind_z_dim] = level
            else:
                self._slice = [slice(None), slice(None), slice(None)]
        else:
            if self.have_vertical:
                self._slice = [slice(None), slice(None), slice(None)]
                if level is not None:
                    self._slice[self._ind_z_dim] = level
            else:
                self._slice = [slice(None), slice(None)]
            
        self._x = self.nc_file.variables[self._x_dim][:]
        delta = self._x[1] - self._x[0]
        if not np.all(util.almost_eq(self._x[1:]-self._x[:-1], delta, 1e-4)):    # if non-uniform
            # try to roll +- 360 degrees to eliminate possible jump
            self._x[self._x > 180] -= 360
            self._x[self._x < -180] += 360
        self._y = self.nc_file.variables[self._y_dim][:]
        if self.have_vertical and level is not None:
            self._z = np.array([self.nc_file.variables[self._z_dim][level]])
        elif self.have_vertical:
            self._z = self.nc_file.variables[self._z_dim][:]
        else:
            self._z = np.array([-99])

#### RK: touching grid dimensions without altering grid is _bad_ idea.
        # Check that horizontal coordinates increase
###        if np.any(self._x[1:] - self._x[:-1] < 0):
###            print 'Flipped X-dimension'
###            self._flip_ew = lambda x: x[::-1,...]
###            self._x = self._x[::-1]
###        else:
###            self._flip_ew = None
###        if np.any(self._y[1:] - self._y[:-1] < 0):
###            print 'Flipped Y-dimension'
###            self._flip_ns = lambda x: x[:,::-1,...]
###            self._y = self._y[::-1]
###        else:
###            self._flip_ns = None
        

        # Check whether the output needs to be transposed or perhaps otherwise flipped:
        vert_before_horiz = (not self.have_vertical
                             or self._ind_z_dim < self._ind_y_dim and self._ind_z_dim < self._ind_x_dim)
        y_before_x = self._ind_y_dim < self._ind_x_dim
        in_c_order = y_before_x and (not self.have_vertical or vert_before_horiz or level is not None)
        in_fortran_order = not y_before_x and (not self.have_vertical
                                               or not vert_before_horiz
                                               or level is not None)
        if in_fortran_order:
            #print('fortran')
            self._flip = None
        elif in_c_order:
            #print('c')
            # C order at least in the dimensions taken from the file.
            self._flip = lambda x: x.T 
        else:
            # if want to support something else, create a function to do the flipping - as
            # above.
            print('ind_x: %i, ind_y: %i' % (self._ind_x_dim, self._ind_y_dim))
            if self.have_vertical:
                print('ind_z: %i' % (self._ind_z_dim))
            raise NCError('This dimension configuration not is supported')
        self.vars = var_name
        try:
            self._undef = self.ncvar._FillValue
        except AttributeError:
            self._undef = np.nan
        try:
            is_single = self.ncvar.datatype == 'f4'
        except AttributeError:
            is_single = self.ncvar.typecode() == 'f'
        if is_single:
            self._eps = 1e-5
        else:
            self._eps = 1e-12

    def describe(self):
        varstr = self.vars
        if self._scale_factor is not None:
            varstr = '%s*%g' % (varstr, self._scale_factor)
        if self._add_offset is not None:
            varstr = '%s + %g' % (varstr, self._add_offset)
        what = '(%s:%s)' % (self.path, varstr)
        return what
        
    def _set_space_slice(self, level):
        if level is None:
            self._space_slice = None
            return
        dimensions_no_time = [dim for dim in self.ncvar.dimensions if not dim == self._time_dim]
        if len(dimensions_no_time) > 3:
            raise NCError('The variable has too many dimensions')
        self._space_slice = [slice(None) for dim in dimensions_no_time]
        try:
            ind_level = dimensions_no_time.index(self._z_dim)
        except ValueError:
            raise NCError('Level requested but not available')
        self._space_slice[ind_level] = level

    def _get_time_ind_const(self):
        if self.timestep_sec == 0.0:
            return 0
        delta = self._time - self.first_time
        seconds = 24*3600*delta.days + delta.seconds
        time_ind = seconds / self.timestep_sec
        return int(time_ind)

    def _get_time_ind_var(self):
        time_ind = list(self._times).index(self._time)
        return time_ind
    
    def _set_time_slice(self, time_ind, time_ind_end=None):
        if time_ind_end and time_ind_end > time_ind:
            if time_ind_end > self.n_times:
                raise SeekPastFileError()
            self._slice[self._ind_time_dim] = slice(time_ind, time_ind_end)
        else:
            self._slice[self._ind_time_dim] = slice(time_ind)

    def read(self, n_steps=1, squeeze=True):
        if self._at_eof:
            raise gradsfile.EndOfFileException()
        if self.ifTimeDim:
            if self.have_const_timestep:
                ind_time = self._get_time_ind_const()
            else:
                ind_time = self._get_time_ind_var()
            self._set_time_slice(ind_time, ind_time + n_steps)
        else:
            ind_time = 0  # if no time dimension for the var, never leave the slot 0
        values = self.ncvar[self._slice].copy()
        #values = self.ncvar[self._slice]
        #1/0
        #mask = np.zeros(values.shape, dtype=np.bool)
        mask = False
        # at_eof true if last step read
        self._at_eof = ind_time + n_steps >= len(self._times)
        if not self._at_eof:
            self.seek(n_steps)
        else:
            self._time = self._last_time
        #return values
        #if squeeze:
        values = values.squeeze()
        if self._flip:
            # order of dimensions
            values = self._flip(values)

#Flip does the wrong job with projected reader (default one)! 
###        if self._flip_ew:
###            # change to scan east to west
###            values = self._flip_ew(values)
###        if self._flip_ns:
###            # change to scan south to north
###            values = self._flip_ns(values)
            
        if not squeeze:
            shape = (len(self._x), len(self._y), len(self._z), n_steps, 1)
            values = values.reshape(shape, order='f')
        if self.mask_mode == 'full':
            #print(values-self._undef)
            mask = np.abs(values - self._undef) < self._eps
        elif self.mask_mode == 'first':
            if self._first_mask is None:
                if np.isnan(self._undef):
                    mask = np.isnan(values)
                else:
                    mask = np.abs(values - self._undef) < self._eps
                self._first_mask = mask
            else:
                mask = self._first_mask
        elif self.mask_mode == False:
            mask = False
        else:
            raise ValueError('Unsupported mask mode')

        # must take care to promote values to right type. values *= scale doesn't work if
        # (when) values is integer type.
#        print(values.dtype, self._scale_factor, self._add_offset)
        if (issubclass(values.dtype.type, np.floating)):

            if self._scale_factor is not None:
                values *= self._scale_factor
            if self._add_offset is not None:
                values += self._add_offset
        else: #Integer offsets might be tricky
            if self._scale_factor is not None:
                #print('before', values.dtype)
                values = values[...] * self._scale_factor
                #values *= self._scale_factor
                #print('after', values.dtype)
            if self._add_offset is not None:
                if self._scale_factor is not None:
                    # avoid copying
                    values += self._add_offset
                    # Force stuff that is within a descrete from zero to zero
                    values[np.abs(values)< np.abs(self._scale_factor)] = 0
                else:
                    values = values[...] + self._add_offset

        if self.mask_mode == False: return values
        else: return np.ma.MaskedArray(values, mask)
        
    def seek(self, n):
        # Seeks n-th time moment from the current time
        if self._at_eof:
            raise SeekPastFileError()
        if self.have_const_timestep:
            new_time = self._time + n*self.timestep
        else:
            ind_time = self._get_time_ind_var()
            new_time = self._times[ind_time + n]
        self.goto(new_time)

    def seek_abs(self, n):
        # Unlike seek, goes straight to the given index
        try:
            self._time = self._times[n]
        except:
            print(self.path)
            print('Size of the time vector:', self._times.size)
            raise NCError('No such time index in file: %s' % n)

    def rewind(self):
        self.goto(self.first_time)
        #self._time = self.first_time
        #self.
        
    def goto(self, time):
        idxTime = np.searchsorted(self._times, time, sorter=self.time_sorter)
        if self._times[idxTime] != time: 
#        if not time in self._times:
            print(self.path)
            print(self._times)
            raise NCError('No such time in file: %s' % time)
        self._time = time
        #self._at_eof = time == self._last_time
        self._at_eof = False
        
    def close(self):
        if self._close_on_close:
            self.nc_file.close()

    def x(self):
        return self._x.copy()

    def y(self):
        return self._y.copy()

    def z(self):
        return self._z.copy()

    def t(self):
        return sorted(self._times)

    def nx(self):
        return len(self._x)

    def ny(self):
        return len(self._y)

    def nz(self):
        return len(self._z)

    def nt(self):
        return len(self._times)

    def dt(self):
        return self.timestep

    def ndims(self):
        return 3 if self.have_vertical else 2

    def nvars(self):
        return 1

    def undef(self):
        return self._undef
    
    def tell(self):
        if self._at_eof:
            raise gradsfile.EndOfFileException()
        return self._time

    def ncattr(self, attr_name, glob=False):
        if glob:
            return getattr(self.nc_file, attr_name)
        else:
            return getattr(self.ncvar, attr_name)

class NCExpression(gradsfile.GriddedDataReader):
    """Instances of this class wrap one or more NCReaders to evaluate
    a multi-variable expression at read time. The interface is
    identical to NCReader."""
    
    @staticmethod
    def is_expression(it):
        return '+' in it or '-' in it or '*' in it or '/' in it
    
    def __init__(self, nc_file, expression, level=None, select=None,
                 dimensions=_dim_map, mask_mode='full',
                 apply_scale_factor=True, apply_add_offset=True):
        if isinstance(nc_file, str):
            nc_obj = netcdf.netcdf_file(nc_file)
        elif isinstance(nc_file, netcdf.netcdf_file):
            nc_obj = nc_file
        nc_vars = nc_obj.variables.keys()
        var_list, self.expression = self._parse_expression(expression, nc_vars)
        self.str_expression = expression
        
        if not all(nc_obj.variables[varname].dimensions == nc_obj.variables[var_list[0]].dimensions
                   for varname in var_list):
            raise NCError('The variables in expression have different dimensions')
        
        self.ncfiles = []
        for var in var_list:
            self.ncfiles.append(NCReader(nc_obj, var, level, select, dimensions, True,
                                         mask_mode, apply_scale_factor, apply_add_offset))
        self._nc_obj = nc_obj
        self._undef = self.ncfiles[0].undef()
        self.path = self.ncfiles[0].path
        self.mask_mode = mask_mode
        self._first_mask = None # for the 'first' mask_mode
        
    def read(self, n=1, squeeze=True):
        fields = [ncf.read(n, squeeze) for ncf in self.ncfiles]
        result = self.expression(fields)

        if isinstance(result, np.ndarray): ## No mask here
            return result

        if self.mask_mode == 'full':
            mask = reduce(np.logical_or, (field.mask for field in fields))
        elif self.mask_mode == False:
            mask = False
        elif self.mask_mode == 'first':
            if self._first_mask is None:
                mask = reduce(np.logical_or, (field.mask for field in fields))
                self._first_mask = mask
            else:
                mask = self._first_mask
        else:
            raise ValueError('Unsupported mask_mode: %s' % self.mask_mode)
        result.mask = mask
        return result

    def describe(self):
        what = '(%s:%s)' % (self.path, self.str_expression)
        return what
    
    def _parse_expression(self, expression, available_variables):
        var_list = []
        ivar = -1
        expr_parsed = expression
        for var in available_variables:
            varex = r'(^|\W)' + var + r'(\W|$)'
            if re.search(varex, expression) is not None:
                ivar += 1
                expr_parsed = re.sub(varex, '\\1@[%i]\\2' % ivar, expr_parsed)
                var_list.append(var)
        expr_parsed = expr_parsed.replace('@', '_X')
        #print('NCExpression:', expr_parsed)
        #print('Variables:', var_list)

        try:
            evaluate = eval('lambda _X: %s' % expr_parsed)
        except SyntaxError:
            print('expr_parsed:', expr_parsed, var_list)
            raise NCError('Syntax error in expression: %s' % expression)
        except NameError:
            print('Available variables:', available_variables)
            raise NCError('Unknown identifier in expression: %s' % expression)
        except:
            raise
        
        # Test the expression:
        test_fields = [np.ones((2,2,2)) for var in var_list]
        try:
            evaluate(test_fields)
        except NameError:
            print('Parsed expression: ', expr_parsed)
            raise NCError('Unknown identifier in expression: %s' % expression)
        except:
            raise NCError('Invalid expression: %s' % expression)
        
        return var_list, evaluate


    def seek(self, n):
        for ncf in self.ncfiles:
            ncf.seek(n)
    def rewind(self):
        for ncf in self.ncfiles:
            ncf.rewind()
    def goto(self, time):
        for ncf in self.ncfiles:
            ncf.goto(time)
    def close(self):
        self._nc_obj.close()
        for ncf in self.ncfiles:
            ncf.close()
    def x(self):
        return self.ncfiles[0].x()
    def y(self):
        return self.ncfiles[0].y()
    def z(self):
        return self.ncfiles[0].z()
    def t(self):
        return self.ncfiles[0].t()
    def dt(self):
        return self.ncfiles[0].dt()
    def ndims(self):
        return self.ncfiles[0].ndims()
    def nvars(self):
        return 1
    def undef(self):
        return self._undef
    def tell(self):
        return self.ncfiles[0].tell()
    def ncattr(self, which, glob=False):
        return self.ncfiles[0].ncattr(which, glob)


class NCDataset(gradsfile.GriddedDataReader):
    _nc_cls_expr = NCExpression
    _nc_cls_noexpr = NCReader
    @staticmethod
    def get_variables(descr_path):
        descriptor = NCDescriptor.fromfile(descr_path)
        return NCReader.get_variables(descriptor.get_file(descriptor.times()[0]))
    
    def __init__(self, descriptor, variable, level=None, mask_mode='full', dimensions=_dim_map):
        if isinstance(descriptor, NCDescriptor):
            self.descr = descriptor
        elif isinstance(descriptor, str):
            self.descr = NCDescriptor.fromfile(descriptor)
        else:
            raise NCError('descriptor needs to be NCDescriptor or filepath')
        self._ncf = None
        self._ncfilename = None
        self.variable = variable
        self._dimensions = dimensions
        self.mask_mode = mask_mode
        # Choose the reader class depending whether the variable is "simple" or not:
        self._nc_class = self._nc_cls_expr if NCExpression.is_expression(variable) else self._nc_cls_noexpr
        self.level = level
        try:
            self.last_time = self.descr.timelist[-1]
        except:
            self.descr.timelist = list( (self.descr.first_time + n * self.descr.timestep 
                                         for n in range(self.descr.n_times)) )
            self.last_time = self.descr.timelist[-1]
        self.rewind()
        try:
            self._switch_file(self.descr.first_time)
        except OSError: #, err:
            raise NCError('Failed to open Netcdf file for the first time: %s' % OSError)
        self._x = np.array(self._ncf.x())
        self._y = np.array(self._ncf.y())
        self._z = np.array(self._ncf.z())
        if len(self._z) > 1:
            self._shape = (len(self._x), len(self._y), len(self._z))
        else:
            self._shape = (len(self._x), len(self._y))
        self.vars = [variable]
        
    def describe(self):
        what = '(%s:%s)' % (self.descr.file, self.variable)
        return what
    
    def _switch_file(self, time):
        new_file = self.descr.get_file(time)
        if self._ncfilename != new_file:
            if self._ncf:
                self._ncf.close()
            self._ncf = self._nc_class(new_file, self.variable, self.level, mask_mode=self.mask_mode,
                                       dimensions=self._dimensions)
            self._ncfilename = new_file

    def rewind(self):
        self.goto(self.descr.first_time)
        
    def _read_one_step(self):
        self._switch_file(self._time)
        try:
            if not self._ncf.tell() == self._time:
                self._ncf.goto(self._time)
        except gradsfile.EndOfFileException:
            # raised by tell() if self._ncf is at eof
            self._ncf.goto(self._time)
        #assert self._time == self._ncf._time
        values = self._ncf.read(1, squeeze=False)
        self.seek(1)
        return values
        
    def read(self, n_times=1, squeeze=True):
        if self._at_eof:
            raise gradsfile.EndOfFileException()
        if n_times == 1:
            values = self._read_one_step()
        else:
            if self._ind_time + n_times - 1 > len(self.descr.timelist):
                raise gradsfile.EndOfFileException()
            read_end_time = self.descr.timelist[self._ind_time + n_times - 1]
            shape_full = (len(self._x), len(self._y), len(self._z), n_times, 1)
            values = np.empty(shape_full)
            for ind_time in range(n_times):
                values[...,ind_time,0] = self._read_one_step().squeeze()           
        if squeeze:
            values = values.squeeze()
        return values

    def seek(self, n):
        # Searches n steps from the current one
        self._ind_time += n
        if self._ind_time < 0 or self._ind_time >= self.descr.n_times:
            self._at_eof = True
            self._time = None
        else:
            self._time = self.descr.timelist[self._ind_time]
            self._at_eof = False

    def seek_abs(self, n):
        if n >= len(self.descr.timelist):
            raise gradsfile.EndOfFileException()
        self._time = self.descr.timelist[n]
        self._ind_time = n
        self._at_eof = False

    def goto(self, time):   # searches for time
        self._time = time
        self._ind_time = np.searchsorted(self.descr.timelist, time)
        self._at_eof = False

    def close(self):
        self._ncf.close()

    def x(self):
        return self._x

    def y(self):
        return self._y

    def z(self):
        return self._z

    def t(self):
        return np.array(self.descr.times())

    def nx(self):
        return len(self._x)

    def ny(self):
        return len(self._y)

    def nz(self):
        return len(self._z)

    def nt(self):
        return len(self.descr.times())

    def dt(self):
        try: return self.descr.timestep
        except: return None

    def ndims(self):
        return len(self._shape)

    def nvars(self):
        return 1

    def tell(self):
        return self._time

    def undef(self):
        return self._ncf.undef()

    def ncattr(self, attr_name):
        return self._ncf.ncattr(attr_name)
