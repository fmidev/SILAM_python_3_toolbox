# A small wrapper to allow using the netcdf4 interface together with ncreaders.

import numpy as np, datetime as dt, re, warnings

from toolbox import gradsfile, util, ncreader
#from ncreader import NCError, SeekPastFileError
#from support import pupynere

import netCDF4 as nc4 

from os import path

class NC4Reader(ncreader.NCReader):
    @classmethod
    def _get_variables(cls, nc_path):
        ncf = nc4.Dataset(nc_path, 'r')
        variables = list(ncf.variables.keys())
        non_dim = [var for var in variables if not var in ncf.dimensions]
        ncf.close()
        return non_dim

    def __init__(self, nc_file_or_descr, *args, **kwargs):
        if isinstance(nc_file_or_descr, nc4.Dataset): #  or isinstance(nc_file_or_descr, pupynere.netcdf_file):
            # pupynere files are allowed in order to pass a test...they are no harm anyway
            self._nc4_file = nc_file_or_descr
            # attn. if use a given nc4.Dataset, mask and scale setting (below) will not be changed!
            filepath = 'unknown' # for now. NCReader.__init__ might be able to find this.
            self._close_on_close = False
        elif isinstance(nc_file_or_descr, ncreader.NCDescriptor):
            filepath = nc_file_or_descr.template
            self._nc4_file = nc4.Dataset(nc_file_or_descr, 'r')
            self._nc4_file.set_auto_maskandscale(False)
        elif isinstance(nc_file_or_descr, str):
            filepath = nc_file_or_descr
            self._nc4_file = nc4.Dataset(nc_file_or_descr, 'r')
            self._nc4_file.set_auto_maskandscale(False)
#            for variable in self._nc4_file.variables.values():
#                variable.set_auto_maskandscale(False)
            self._close_on_close = True
        else:
            raise ncreader.NCError('The first argument is of wrong type')
        
        ncreader.NCReader.__init__(self, self._nc4_file, *args, **kwargs)
        if self.path == 'unknown':
            # try still:
            try:
                self.path = self._nc4_file.filepath()
            except AttributeError:
                # we got a pupynere object
                self.path = self._nc4_file.filename
            except ValueError:
                # the version of nc4 doesn't support the filepath() method
                self.path = filepath

class NC4Expression(ncreader.NCExpression):
    def __init__(self, nc_file_or_descr, expression, level=None, select=None,
                 dimensions=ncreader._dim_map, mask_mode='full',
                 apply_scale_factor=True, apply_add_offset=True):
        if isinstance(nc_file_or_descr, str):
            nc_obj = nc4.Dataset(nc_file_or_descr, 'r')
            nc_obj.set_auto_maskandscale(False)
        else:
            # The given object will be used as such. Note: on close(), nc_obj will be closed.
            nc_obj = nc_file_or_descr
            
        nc_vars = nc_obj.variables.keys()
        var_list, self.expression = self._parse_expression(expression, nc_vars)
        self.str_expression = expression
        
        if not all(nc_obj.variables[varname].dimensions == nc_obj.variables[var_list[0]].dimensions
                   for varname in var_list):
            raise ncreader.NCError('The variables in expression have different dimensions')
        
        self.ncfiles = []
        for var in var_list:
            self.ncfiles.append(NC4Reader(nc_obj, var, level, select, dimensions, True,
                                          mask_mode, apply_scale_factor, apply_add_offset))
        self.path = self.ncfiles[0].path
        self._nc_obj = nc_obj
        self._undef = self.ncfiles[0].undef()
        self.mask_mode = mask_mode
        self._first_mask = None # used if mask_mode = 'first'

    
        
class NC4Dataset(ncreader.NCDataset):
    _nc_cls_expr = NC4Expression
    _nc_cls_noexpr = NC4Reader
