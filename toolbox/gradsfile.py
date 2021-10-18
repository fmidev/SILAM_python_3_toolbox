import datetime as dt, calendar
import numpy as np
from numpy import ma
import sys, copy, glob
from os import path
import re

from toolbox import util

import warnings

# check if we have any() builtin (as it is for python 2.5)
try: 
    any
except NameError:
    def any(a):
        for element in a:
            if element: return True
        return False

""" This module contains classes for reading GrADS files, and more
generally, gridded datasets. This is implemented through the interface
in class GriddedDataReader. Only use these methods to access the data! 

The following classes are available: 
- GriddedDataReader : abstract base class 
- Gradsfile : a Gradsfile reader 
- CompositeGradsfile : a griddedDataReader returning the sum of several 
  variables across several files (ie. griddedDataReader objects)
- AveragedReader : wrapper around a griddedDataReader returning time
  averages in given windows.
"""

grads_templates = ['%y4', '%y2', '%m2', '%m1', '%h2', '%h1', '%d2', '%d1']
grads_template_length = {'%y4' : 4, '%y2' : 2, '%m2' : 2, '%m1' : 1, '%h2' : 2, '%h1' : 1, '%d2' : 2, '%d1' : 1}


class GenericWrapper(object):
    def __init__(self, wrapped, my_methods):
        self.wrapped = wrapped
        self.own_methods = tuple(my_methods) + ('describe',)
        
    def __getattribute__(self, attr):
        if attr in object.__getattribute__(self, 'own_methods') or not attr in GriddedDataReader.__dict__:
            return object.__getattribute__(self, attr)
        return getattr(object.__getattribute__(self, 'wrapped'), attr)

class GriddedDataReader:
    """ An "abstract" reader object for gridded data. Currently implemented
    for gradsfile and compositeGradsfile. Only use these methods
    outside this module!

    """
    def read(self, n=1):
        raise NotImplementedError()
    
    def seek(self, n=1):
        raise NotImplementedError()

    def seek_abs(self, n=1):
        raise NotImplementedError()

    def rewind(self):
        raise NotImplementedError()
    
    def goto(self, time):
        if not time in self.t():
            raise GradsfileError('Only exact times are allowed')
        if time > self.tell():
            self.rewind()
        while self.time < time:
            self.seek()

    def close(self):
        raise NotImplementedError()
    
    # Return x, y, z and t axes
    def x(self): raise NotImplementedError()
    def y(self): raise NotImplementedError()
    def z(self): raise NotImplementedError()
    def t(self): raise NotImplementedError()
    def nx(self): raise NotImplementedError()
    def ny(self): raise NotImplementedError()
    def nz(self): raise NotImplementedError()
    def nt(self): raise NotImplementedError()
    def var_name(self): raise NotImplementedError()
    
    def meshgrid(self, vertices=False):
        # This method calls the np.meshgrid function. I don't like it anymore, because
        # then X[i,j] corresponds to x[j], y[i] ie. the axes are flipped.
        warnings.warn('The meshgrid() method is depreceated, use coordinates() instead')
        if not vertices:
            X, Y = np.meshgrid(self.x(), self.y())
        else:
            x = util.points_to_edges(self.x())
            y = util.points_to_edges(self.y())
            X, Y = np.meshgrid(x, y)
        return X, Y

    def coordinates(self, vertices=False):
        """
        Return the coordinates of grid points in 2D arrays:

        X, Y = reader.coordinates()
        x[i] <-> X[i,j] <-> data[i,j]
        y[j] <-> Y[i,j] <-> data[i,j]
        
        The optional argument specifies if we want the coordinates of vertices of the
        grid (eg. for a meshgrid picture).
        
        """
        
        if not vertices:
            X, Y = np.meshgrid(self.x(), self.y())
        else:
            x = util.points_to_edges(self.x())
            y = util.points_to_edges(self.y())
            X, Y = np.meshgrid(x, y)
        return X.T, Y.T        
    
    # Return timestep
    def dt(self): raise NotImplementedError()
    def ndims(self): raise NotImplementedError()
    def nvars(self): raise NotImplementedError()
    def undef(self): raise NotImplementedError()
    # Return current time
    def tell(self): raise NotImplementedError()
    def indices(self, x, y):
        return self._ind_x(x), self._ind_y(y)

    def shape(self):
        return len(self.x()), len(self.y()), len(self.z()), self.nvars()
    
    # Return grid indices for coordinates
    def _ind_x(self, x):
        X = self.x()
        x0 = X[0]
        dx = X[1]-X[0]
        return int(round((x - x0) / dx))
    
    def _ind_y(self, y):
        Y = self.y()
        y0 = Y[0]
        dy = Y[1]-Y[0]
        return int(round((y - y0) / dy))

    def describe(self):
        """
        Tell what is returned by read()
        """
        

class GradsfileError(Exception):
    pass

class EndOfFileException(Exception):
    """
    The "end-of-file" has slightly different meaning than in the usual C-sense. The position
    in file, returned by tell() is the timestamp corresponding to the data that would be read 
    next. When all the data is consumed, this becomes undefined, and EndOfFileExecption 
    is raised.
    """
    pass

class Gradsfile(GriddedDataReader):
    """A class for handling grads gridded data files.
    """
    size_of_float = 4
        
    def parseExpression(self, expr):
        isExpression = '+' in expr or '-' in expr or '*' in expr or '/' in expr
        expr_parsed = expr 
        import re

        if isExpression:
            # Var can be defined as an expression of variables in the file. In this case,
            # the expression is converted into a python function which takes the full field as
            # an argument. As an intermediate step, the argument is denoted by @, since this 
            # should not appear in grads variables.
            ivar = -1
            varlist = []
            for var in self.desc.vars:
                varex = var + '(\W|$)'
                if re.search(varex, expr) is not None:
                    ivar += 1
                    expr_parsed = re.sub(varex, '@[:,:,:,%i,:]\\1' % ivar, expr_parsed)
                    varlist.append(var)
            expr_parsed = expr_parsed.replace('@', 'X')
            try:
                self.expression = eval('lambda X: %s' % expr_parsed)
            except SyntaxError:
                raise GradsfileError('Syntax error in expression: %s' % expr)
            except NameError:
                raise GradsfileError('Unknown identifier in expression: %s' % expr)
            except:
                raise
#                raise GradsfileError('Invalid expression: %s' % expr)
            
            # Test the expression:
            testfld = np.ones((1, 1, 1, len(varlist), 1))
            try:
                self.expression(testfld)
            except NameError:
                raise GradsfileError('Unknown identifier in expression: %s' % expr)
            except:
                raise GradsfileError('Invalid expression: %s' % expr)
            
            self.vars = varlist
            
        else:
            self.vars = [expr]
            self.expression = None
                
        
    def __str__(self):
        str = 'gradsfile: %s' % (self.desc.file)
        return str

    def describe(self):
        if self.expression:
            what = '(%s:%s)' % (self.desc.file, self._var_request)
        else:
            what = '(%s:%s)' % (self.desc.file, ' '.join(self.vars))
        return what
    
    def __init__(self, ctl, var, level=None, mask_mode=None):

        """ Gradsfile(ctl, var) -> Gradsfile object

        Open a grads dataset for reading. The data are described by
        ctl, which can be either path to the descriptor file or a
        GradsDescriptor object. The variables must be specified either
        as string for single or sequence for multiple variables. The
        variables must have same dimension.

        Handling undefined values is controlled by the argument mask_mode.
        It can be one of the following:
        - 'full' -> compare the values to the undefined value at every read
          and return a ma.MaskedArray
        - 'first' -> compare the values only at the first read. This could 
          save time if the mask is static.
        - 'false' -> return a MaskedArray, but set the mask to False regardless
          of the values.
        - 'none' -> do not perform any checks, return a regular array.

        CAVEAT: the variables are returned in the order they are in
        the gradsfile, not in the order they are in the input list.

        LIMITATIONS: not all features available for the grads
        descriptors are supported: xdef, ydef and tdef must be of type
        LINEAR, while zdef must be LEVELS. The timestep must be either
        days, hours or minutes.

        Templates (most) and byteorder conversions are handled.
               
        """
        if mask_mode is None:
                mask_mode='full'
        try: 
            nx = ctl.nx
            self.desc = ctl
        except AttributeError: # not a descriptor object
            self.desc = GradsDescriptor(ctl)
   
        self.fileNm = None
        self.fileHandle = None
        self.position = 0
        #self._time = self.desc.t0
        self.step = 0
        self._times = self.desc.times
        self.timeEnd = self._times[-1]
        self.ifSwap = (self.desc.byteorder != sys.byteorder)
        self.step = 0
        self._at_eof = False
        # We can have a single variable or a list of them. The limitation is
        # that the list may not contain complex expressions. However, if the list 
        # has only one element, we can still allow it.

        if isinstance(var, str):
            # Is var an expression?
            self.parseExpression(var)
        else:
            # Check if sequence
            try:
                iter(var)
            except TypeError:
                raise TypeError('VAR has to be either sequence of strings or a string.')
            if len(var) > 1:
                self.vars = [v for v in var]
                self.expression = None
            else:
                self.parseExpression(var[0])
        self._var_request = var
        
        allowed = ('full', 'first', 'false', 'none')
        if not mask_mode.lower() in allowed:
            raise GradsfileError('mask_mode must be one of following: ' + ' '.join(allowed))
        self.mask_mode = mask_mode.lower() 
        if self.mask_mode == 'first':
            self._mask = None
            
        
        self.ifReadVar = [v in self.vars for v in self.desc.vars]
        if not any(self.ifReadVar):
                raise GradsfileError('None of the variables exist in file')
        # The variables must have same z dimension
        nzVars = np.array([self.desc.nz[vTmp] for vTmp in self.vars])
        if not np.all(nzVars == nzVars[0]):
            raise GradsfileError('Variables must have the same z dimension')
        self.ifReadVar = [v in self.vars for v in self.desc.vars]
        if not any(self.ifReadVar):
                raise GradsfileError('None of the variables exist in file')
            
        self._nvars = len(self.vars)
        self.ifOpen = True
        self.nz = nzVars[0]
        if level is not None:
            if level < 0 or level > self.nz-1 or (self.nz == 1 and level > 0):
                raise GradsfileError('Invalid level requested: %s' % str(level))
            self.level = level
            self.nz = 1
        else:
            self.level = None
        self.fieldSz = self.desc.nx*self.desc.ny*self.nz
        
    def _read_many(self, n):
        r = np.empty((self.desc.nx, self.desc.ny, self.nz, self._nvars, n), dtype=np.float32)
        
        if self.level is not None:
            nzFull = self.desc.nz[self.vars[0]]
            seek_before = self.level * self.fieldSz * Gradsfile.size_of_float
            seek_after = (nzFull - self.level - 1) * self.fieldSz * Gradsfile.size_of_float
        else:
            seek_before = 0
            seek_after = 0

        for step in range(n):
            now = self._times[self.step]
            if now > self.timeEnd:
                break

            fileNm = expandGradsTemplate(self.desc.template, now)
            if self.fileNm != fileNm:
                self.switchFile(fileNm)

            iReadVar = 0
            
            for iVar in range(self.desc.nvars):
                if self.ifReadVar[iVar]:
                    #print('Will seek at %i' % self.fileHandle.tell())
                    if seek_before > 0: 
                        self.fileHandle.seek(seek_before, 1)
                    #print('Will read at %i' % self.fileHandle.tell())
                    fld = np.fromfile(self.fileHandle, dtype=np.float32,
                                      count=self.fieldSz)
                    r[:,:,:,iReadVar,step] = fld.reshape((self.desc.nx, self.desc.ny, self.nz),
                                                         order='F')                                    
                    if seek_after > 0: 
                        self.fileHandle.seek(seek_after, 1)
                    iReadVar += 1
                    
                else:
                    thisNz = self.desc.nz[self.desc.vars[iVar]]
                    self.fileHandle.seek(self.desc.nx*self.desc.ny*thisNz 
                                         * Gradsfile.size_of_float, 1)
            self.step += 1
            if self.step == self.desc.nt:
                break
        return r
            
    def _read_one_var(self):
        if self.level is not None:
            nzFull = self.desc.nz[self.vars[0]]
            seek_before = self.level * self.fieldSz * Gradsfile.size_of_float
            seek_after = (nzFull - self.level - 1) * self.fieldSz * Gradsfile.size_of_float
        else:
            seek_before = 0
            seek_after = 0
        shape = (self.desc.nx, self.desc.ny, self.nz, 1, 1)
        fileNm = expandGradsTemplate(self.desc.template, self._times[self.step])
        if self.fileNm != fileNm:
            self.switchFile(fileNm)
        for ind_var in range(self.desc.nvars):
            if self.ifReadVar[ind_var]:
                if seek_before:
                    # seek to the correct level
                    self.fileHandle.seek(seek_before, 1)
                field = np.fromfile(self.fileHandle, dtype=np.float32, count=self.fieldSz)
                if seek_after:
                    self.fileHandle.seek(seek_after, 1)
                break
            else:
                thisNz = self.desc.nz[self.desc.vars[ind_var]]
                # seek to the next variable
                self.fileHandle.seek(self.desc.nx*self.desc.ny*thisNz 
                                     * Gradsfile.size_of_float, 1)
        self.step += 1
        nxny = self.desc.ny*self.desc.nx
        sof = Gradsfile.size_of_float
        seek_to_next = sum(self.desc.nz[self.desc.vars[ind]]*nxny*sof
                           for ind in range(ind_var+1, self.desc.nvars))

        # seek to the next timestep
        if seek_to_next:
            self.fileHandle.seek(seek_to_next, 1)
        return field.reshape(shape, order='f')
        
    def read(self, n = 1, squeeze=True):
        """x = gf.read(n)
        
        Read the next n timesteps from the GrADS dataset into a numpy array.
        
        Parameters 
        ---------- 
        
        n : integer, optional 
           Number of timesteps to read. By default,
           read one timestep.
           
        Returns 
        ------- 
        
        x : masked nd-array 
        The data. Depending on n and number of
        variables to read, x can be 2-5 dimensional with dimensions (nx,
        ny, [nz], [nvars], [nt]). Undefined values are masked.

        """
        ifSqueeze = squeeze
        if n < 0:
            raise GradsfileError('Number of steps has be nonnegative')
        elif n == 0:
            return np.array([], dtype=np.float32)

        if self.step + n > self.desc.nt:
            raise EndOfFileException()
        
        if n == 1 and self._nvars == 1:
            r = self._read_one_var()
        else:
            r = self._read_many(n)
        if self.step > self.desc.nt-1:
            self._at_eof = True
        #if (self.step > self.desc.nt-1): self.step = self.desc.nt-1
        if self.ifSwap: r.byteswap(True)

        # Handle the mask before applying the expression!
        if self.mask_mode == 'full':
            mask = np.abs(r-self.desc.undef) < 1e-5
            #print(np.any(mask))
        elif self.mask_mode == 'first':
            if self._mask is None:
                self._mask = np.abs(r[:,:,:,:,0]-self.desc.undef) < 1e-5
                #print(np.mean(r[...]), np.any(self._mask))
            if n == 1:
                mask = self._mask
            else:
                mask = np.tile(self._mask, (1,1,1,1,n))
                #print(mask.shape)
        elif self.mask_mode == 'false':
            mask = False
        elif self.mask_mode == 'none':
            mask = None
            
        # If we have an expression, apply it now:
        if self.expression:
            # Expression will kill the 'variable' dimension. But we don't want to do that yet.
            r = self.expression(r).reshape((self.desc.nx, self.desc.ny, self.nz, 1, n))
            if mask is not None and mask is not False:
                mask = np.any(mask, 3)

        if squeeze: 
            r = r.squeeze()
            if mask is not None and mask is not False:
                mask = mask.squeeze()
               
        # apply mask
        if mask is not None:
            r = ma.MaskedArray(r, mask)
        return r

    def switchFile(self, newFile):
        if not (self.fileHandle == None or self.fileHandle.closed):
                self.fileHandle.close()
        self.fileHandle = open(newFile, 'rb')
        self.fileNm = newFile

    #****************************************************************************************************
            
    def seek(self, n):
        """ Advance n timesteps in the file without reading. """
        if (n < 0):
            raise GradsfileError('Cannot seek backwards')
        elif (n == 0):
            return
        
        #now = self._time + n*self.desc.dt
        step = self.step + n
        if step >= self.desc.nt:
            self._at_eof = True
        #if now > self.timeEnd:
            self.step = self.desc.nt-1
            self.fileNm = None
            if self.fileHandle is not None and not self.fileHandle.closed:
                self.fileHandle.close()
            self.fileHandle = None
            return
        if step < 0:#now < self.desc.t0:
            self.step = 0
            return
        now = self._times[step]
            
        # how many bytes is a timestep?
        tStepSz = sum(self.desc.nz.values()) * self.desc.nx*self.desc.ny * Gradsfile.size_of_float

        #Pick file
        newfile=expandGradsTemplate(self.desc.template, now)
        if self.fileNm != newfile:
           while self.fileNm != newfile: # Need to look for a file
                 self.step += 1
                 self.fileNm = expandGradsTemplate(self.desc.template, self._times[self.step])
           #print("Switching to new file", newfile)
           self.switchFile(newfile) # Open file once
        
        #get offset
        seekbytes=0
        while self.step < step:
            self.step += 1
            seekbytes += tStepSz
        #print('Seeking file %s from %i by %i' % (self.fileNm, self.fileHandle.tell(),seekbytes))
        self.fileHandle.seek(seekbytes, 1)
      

    def goto(self, time):
        """ Move the file pointer to the timestep corresponding to 
        the Datetime object time. The time must match exactly some 
        timestep in the file.

        """
        times = self._times        
        try:
            ind_time = times.index(time)
        except ValueError:
            if (time > self.timeEnd or time < self.desc.t0):
                raise GradsfileError('Cannot move past the file')
            else:
                raise GradsfileError('Only exact grads file times are allowed')
        
        if self._at_eof or time < self._times[self.step]:
            self.rewind()
        
        file_to_be = expandGradsTemplate(self.desc.template, time)
        
        if file_to_be == self.fileNm:
            steps_to_seek = ind_time - self.step
            if steps_to_seek < 0:
                print(self.step, ind_time, time, self.t())
            self.seek(steps_to_seek)
            return
        
        for ind,timestep in enumerate(times):
            if expandGradsTemplate(self.desc.template, timestep) == file_to_be:
                break

        self.switchFile(file_to_be)
        self.step = ind
        steps_to_seek = ind_time - ind
        self.seek(steps_to_seek)
        
    def rewind(self):
        """ Return to the first timestep."""
        self.step = 0
        self._at_eof = False
        file = expandGradsTemplate(self.desc.template, self._times[0])
        #if file != self.fileNm:
        self.switchFile(file)
        
    
    def close(self):
        if not self.ifOpen: return
        self.fileNm = None
        if self.fileHandle != None and not self.fileHandle.closed:
            self.fileHandle.close()
        self.fileHandle = None
        self.ifOpen = False

    def x(self): return self.desc.x
    def y(self): return self.desc.y
    def z(self): 
        if self.ndims() > 2:
            return self.desc.z
        else:
            return np.array([np.nan])

    def t(self):
        return self._times
        #return self.desc.times()
        #T = [self.desc.t0 + self.desc.dt*i for i in range(0,self.desc.nt)]
        #return T
    def nx(self): return self.desc.nx
    def ny(self): return self.desc.ny
    def nz(self): return self.nz
    def nt(self): return self.desc.nt
    
    def dt(self):
        return self.desc.dt

    def undef(self): return self.desc.undef
    
    def ndims(self): 
        if self.nz > 1:
            return 3
        else:
            return 2
    
    def nvars(self): 
        if self.expression:
            return 1
        else:
            return self._nvars

    def tell(self):
        if self.step < self.desc.nt:
            return self._times[self.step]
        else:
            raise EndOfFileException()

class CompositeGradsfile(GriddedDataReader):
    """ A virtual grads file consisting of several files and variables.

    Define a "metafile" object with same methods as Gradsfile by including
    them with using the method addGradsfile. Example:
    
    gfSO24 = Gradsfile('foo.grads.ctl',['so2','so4'])
    gfSO4 = Gradsfile('bar.grads.ctl','h2so4')
    cgf = compositeGradsfile('S')
    cgf.addGradsfile(gfSO24)
    cgf.addGradsfile(gfSO4)
    x = cgf.read(1)
    # x now has so2 + so3 + so4

    It is a bad idea to read or seek files like gfSO4 while they
    are connected to a composite gradsfile.
    
    NOTE. no gradsfile specific fields/methods are used. Clean for any
    griddedDataReader.

    """
    def __init__(self, varName):
        self.vars = [varName]
        self.X = None
        self.Y = None
        self.Z = None
        self.T = None
        self._nvars = 1
        self.nfiles = 0
        self.nz = None
        self.files = []
        self._ndims = None
        # add nz 
        
    def addGradsfile(self, gf):
        if self.X is not None:
            if len(self.X) != len(gf.x()):
                raise ValueError('The file has incompatible x axis')
        else:
            self.X = gf.x()

        if self.Y is not None:
            if len(self.Y) != len(gf.y()):
                raise ValueError('The file has incompatible y axis')
        else:
            self.Y = gf.y()

        if self.Z is not None:
            if len(self.Z) != len(gf.z()) or np.any(np.abs(self.Z - gf.z()) > 1e-6):
                raise ValueError('The file has incompatible z axis')
        else:
            self.nz = len(gf.z())
            self.Z = gf.z()

        if self.T is not None:
            if len(self.T) != len(gf.t()):
                raise ValueError('The file has incompatible t axis')
        else:
            self.T = gf.t()
            self.DT = gf.dt()

        if self._ndims is not None:
            if self._ndims != gf.ndims():
                raise ValueError('The file has different number of dimensions')
        else:
            self._ndims = gf.ndims()

        self.files.append(gf)
        self.nfiles += 1

    def describe(self):
        description = 'COMPOSITE(%s)' % ('+'.join(gf.describe() for gf in self.files))
        return description
    
    def read(self, n=1, squeeze=True):
        if self.nfiles == 0:
            raise ValueError('No files chosen')
        if n == 0:
            return np.array([])
        field = np.zeros((len(self.X), len(self.Y), self.nz, 1, n), dtype=np.float32)
        mask = np.zeros(field.shape, dtype=np.bool)
        for gf in self.files:
            readFld = gf.read(n, False) # do not squeeze the array.
            assert(readFld.ndim >= field.ndim)
            if readFld.ndim > field.ndim: 
                # Sum along the variable dimension
                field += np.sum(readFld.data,-2)
                mask = mask | np.any(readFld.mask, -2)
           
            else:
                field += readFld.data.reshape(field.shape)
                mask = mask | readFld.mask.reshape(field.shape)
                 
        if squeeze: field = field.squeeze()
        return ma.MaskedArray(field, mask)

    def seek(self, n):
        for gf in self.files:
            gf.seek(n)
            
    def goto(self, time):
        for gf in self.files:
            gf.goto(time)

    def rewind(self):
        for gf in self.files:
            gf.rewind()

    def close(self):
        for gf in self.files:
            gf.close()

    def x(self): return self.X
    def y(self): return self.Y
    def z(self): return self.Z
    def t(self): return self.T
    def nx(self): return len(self.X)
    def ny(self): return len(self.Y)
    def nz(self): return len(self.Z)
    def nt(self): return len(self.T)
    def dt(self): return self.DT
    def undef(self): return self.files[0].undef()
    def tell(self): return self.files[0].tell()
    def nvars(self): return 1
    def ndims(self): return self._ndims
        
def expandGradsTemplate(templ, time):
    out = templ
    # only partial support at the moment
    formats = {'%y4' : '%04i',
               '%y2' : '%s',
               '%m2' : '%02i',
               '%m1' : '%i',
               '%h2' : '%02i',
               '%h1' : '%i',
               '%d2' : '%02i',
               '%d1' : '%i'
               }
    y2 = str(time.year)[2:4]
    out = out.replace('%y2', y2)
    out = out.replace('%y4', '%04i' % time.year)
    
    for ch in ('%m2', '%m1'):
        out = out.replace(ch, formats[ch] % time.month)
    for ch in ('%h2', '%h1'):
        out = out.replace(ch, formats[ch] % time.hour)
    for ch in ('%d2', '%d1'):
        out = out.replace(ch, formats[ch] % time.day)
    return out


def all_files_for_template(tIn):
    templTmp = tIn
    while templTmp.find('%') > 0:
        ifFound = False
        for t in grads_templates:
            if templTmp.find(t) > 0:
                templTmp = templTmp.replace(t,'%s' % (''.join(['?']*grads_template_length[t])))
                ifFound = True
        if ifFound == False:
            print('Unknown GrADS template in the file name: ',templTmp)
            print('Allowed GrADS teomplates are:', grads_templates)
            raise ValueError
    return glob.glob(templTmp)  # all files satisfying the template at some time


def exists(tIn):
    return len(all_files_for_template(tIn)) > 0

def get_active_templates(templIn):
    active_templates = []
    for t in grads_templates:
        tIdx = templIn.find(t) # position in the template name
        if tIdx > 0: 
            active_templates.append([t, tIdx])
    # Have to find the position of each tempalte in real-name file
    # Do it dumb way: efficiency is not an issue here
    for t in active_templates:
        delta = 0
        for t2 in active_templates:
            if t[1] > t2[1]: 
                delta += grads_template_length[t2[0]] - len(t2[0])
        t[1] += delta
    return active_templates

def files_and_times_4_template(templIn):
    # get all files satisfying the template for any time
    # Get files satisfying the template and tempalte elements in the templIn
    start_end_time_pieces = [[],[]]  # start and end of the interval
    start_end_time = []
    for s_e_idx in [0,-1]:
        active_templates = get_active_templates(templIn)
        FNmSearch = templIn
        lstFiles = all_files_for_template(FNmSearch)
        if len(lstFiles) < 1:
            print('No files found for:', FNmSearch)
            raise ValueError
        # Find the earliest and the latest times as written in the file names
#        timePieces = {}
        while len(active_templates) > 0:
            t, tIdx = active_templates[0]
#            print(lstFiles[0], t, tIdx, grads_template_length[t])
            timePieces = []
            for fnm in lstFiles:
                fTmp = fnm[tIdx:tIdx + grads_template_length[t]]
                timePieces.append(fnm[tIdx:tIdx + grads_template_length[t]])
            timePieces = sorted(timePieces)
#            timePieces = sorted(list( (int(fnm[tIdx:tIdx + grads_template_length[t]]) 
#                                       for fnm in lstFiles) ))
            start_end_time_pieces[s_e_idx].append(timePieces[s_e_idx])
            FNmSearch = FNmSearch.replace(t,str(start_end_time_pieces[s_e_idx][-1]))
            lstFiles = all_files_for_template(FNmSearch)
            active_templates = get_active_templates(FNmSearch)
        if len(lstFiles) != 1:
            print('All templates have been used up but single file is not found:', lstFiles)
            raise ValueError
        # Now need to create datetime object with who-knows-how-many-items-defined
        d0 = int(start_end_time_pieces[s_e_idx][0])
        d1 = 1
        d2 = 1
        d3 = 0
        d4 = 0
        if len(start_end_time_pieces[s_e_idx]) > 1:
            d1 = int(start_end_time_pieces[s_e_idx][1])
            if len(start_end_time_pieces[s_e_idx]) > 2:
                d2 = int(start_end_time_pieces[s_e_idx][2])
                if len(start_end_time_pieces[s_e_idx]) > 3:
                    d3 = int(start_end_time_pieces[s_e_idx][3])
                    if len(start_end_time_pieces[s_e_idx]) > 4:
                        d4 = int(start_end_time_pieces[s_e_idx][4])
        start_end_time.append(dt.datetime(d0,d1,d2,d3,d4))
            
    return (all_files_for_template(templIn),                  # all files addressed by the template
            expandGradsTemplate(templIn, start_end_time[0]),  # first-time file
            expandGradsTemplate(templIn, start_end_time[1]))  # last-time file


class GradsDescriptor:
    MONTHLY = '__mo__'
    def __init__(self, file=None):
        if file == None: return
        self.file = file
        fhCtl = open(file, 'r')
        line = fhCtl.readline()
        self.byteorder = sys.byteorder
        while line != '':
            line = line.lstrip()
            linelc = line.lower()
            if linelc.startswith('dset'):
                filepath = line[4:].lstrip()
                self.template = filepath.rstrip()
                if (self.template.startswith('^')):
                        ftemplate=self.template[1:]
                        self.template = path.join(path.dirname(file),ftemplate)
                
            elif linelc.startswith('options'):
                option = linelc.split()[1]
                if option == 'big_endian':
                    self.byteorder = 'big'
                elif option == 'little_endian':
                    self.byteorder = 'little'
                elif option == 'template':
                    self.ifTemplate = True
                else:
                    raise GradsfileError('Unsupported option: %s' % option)
                    
            elif linelc.startswith('undef'):
                self.undef = float(linelc.split()[1])
                
            elif linelc.startswith('xdef'):
                fields = linelc.split()
                if not fields[2] == 'linear':
                    raise GradsfileError('Only linear xdef is supported')
                self.nx = int(fields[1])
                self.x0 = float(fields[3])
                self.dx = float(fields[4])
                self.x = np.array([self.x0+i*self.dx for i in range(0, self.nx)])
                
            elif linelc.startswith('ydef'):
                fields = linelc.split()
                if not fields[2] == 'linear':
                    raise GradsfileError('Only linear ydef is supported')
                self.ny = int(fields[1])
                self.y0 = float(fields[3])
                self.dy = float(fields[4])
                self.y = np.array([self.y0+i*self.dy for i in range(0, self.ny)])
                
            elif linelc.startswith('zdef'):
                fields = linelc.split()
                if not fields[2] == 'levels':
                    raise GradsfileError('Only levels type zdef is supported')
                self._nz = int(fields[1])
                self.z0 = float(fields[3])
                self.z = np.array([float(val) for val in fields[3:]])
                
            elif linelc.startswith('tdef'):
                fields = linelc.split()
                #print(fields)
                if not fields[2] == 'linear':
                    raise GradsfileError('Only linear tdef is supported')
                self.nt = int(fields[1])
                self.t0 = parseGradsTime(fields[3])
                self.dt = parseGradsInterval(fields[4])
                if self.dt is GradsDescriptor.MONTHLY:
                    self.times = list(util.iterate_monthly(self.t0, self.nt))
                else:
                    self.times = [self.t0 + ii*self.dt for ii in range(self.nt)]
                
            elif linelc.startswith('vars'):
                fields = linelc.split()
                self.nvars = int(fields[1])
                self.vars = []
                self.longVars = {}
                self.nz = {}
                line = fhCtl.readline()
                linelc = line.lower()
                while linelc.find('endvars') < 0:
                    fields = line.split()
                    self.vars.append(fields[0])
                    nzVar = int(fields[1])
                    if nzVar == 0: nzVar = 1
                    if nzVar != 1 and nzVar != self._nz:
                        raise GradsfileError('Inconsistent vertical definition')
                    self.nz[fields[0]] = nzVar
                    longname = (' '.join(fields[5:])).strip()
                    self.longVars[fields[0]] = longname
                    #ix = linelc.find(fields[5])
                    #self.longvars.append(line[ix:].strip())
                    line = fhCtl.readline()
                    linelc = line.lower()

            # next line from ctl
            line = fhCtl.readline()
                    
        fhCtl.close()
        if self.nvars != len(self.vars):
            raise GradsfileError('Inconsistent number of variables')

        # duplicated variable names are dangerous
        for variable in self.vars:
            name_count = sum(v == variable for v in self.vars)
            if name_count > 1:
                raise GradsfileError('Duplicated variable: %s' % variable)
        
    def relocate(self, directory=None):
        assert(directory or self.file)
        if not directory: 
            directory = path.dirname(self.file)
        self.template = path.join(directory, path.basename(self.template))

    def times(self):
        return self.times


class _MmapCache:
    def __init__(self, size):
        self.cachesize = size
        self._maplst = []
        self._mmaps = {}
    
    def _add(self, filename, shape):
        if len(self._maplst) >= self.cachesize:
            self._remove(self._maplst[0])
        try:
            mm = np.memmap(filename, mode='r', dtype=np.float32, shape=shape, order='F')
        except:
            print('Error creating memorymap for file:', filename)
            raise
        # the last element tells how many times the map has been requested or released
        self._mmaps[filename] = (mm, shape, [0])
        self._maplst.append(filename)

    def _remove(self, filename):
        arr_count = self._mmaps[filename][2]
        arr_count[0] -= 1
        if arr_count[0] == 0:
            del(self._mmaps[filename])
            ind = self._maplst.index(filename)
            del(self._maplst[ind])
        
    def get_mmap(self, filename, shape):
        filename = path.realpath(filename)
        if not filename in self._mmaps:
            self._add(filename, shape)
        mmap, shape_cur, arr_count = self._mmaps[filename]
        if shape_cur != shape:
            raise ValueError('Shape requested differs from cached')
        arr_count[0] += 1
        #print('get mm', arr_count[0])
        return mmap
        
    def release(self, filename):
        real = path.realpath(filename)
        if not real in self._mmaps:
            return
        self._remove(real)
        
MMCACHE_SIZE = 5
mmcache = _MmapCache(MMCACHE_SIZE)

def next_month(time):
    if time.month == 12:
        return time.replace(year=time.year+1, month=1)
    else:
        return time.replace(month=time.month+1)
        

class MMGradsfile(Gradsfile):
    MASK_TOL = 1e-5
    mmcache = mmcache
    def _parse_expression(self, expr):
        self._var_request = expr
        chars_expr = '+-*/()'
        isExpression = any(char in expr for char in chars_expr)
        expr_parsed = expr 
        if isExpression:
            # Var can be defined as an expression of variables in the file. In this case,
            # the expression is converted into a python function which takes the list of fields as
            # an argument. As an intermediate step, the argument is denoted by @, since this 
            # should not appear in grads variables.
            ivar = -1
            varlist = []
            for var in self.desc.vars:
                varex = var + '(\W|$)'
                if re.search(varex, expr) is not None:
                    ivar += 1
                    expr_parsed = re.sub(varex, '@[%i]\\1' % ivar, expr_parsed)
                    varlist.append(var)
            expr_parsed = expr_parsed.replace('@', 'X')
            try:
                self._evaluation = eval('lambda X: %s' % expr_parsed)
            except SyntaxError:
                raise GradsfileError('Syntax error in expression: %s' % expr)
            except NameError:
                raise GradsfileError('Unknown identifier in expression: %s' % expr)
            except:
                raise
#                raise GradsfileError('Invalid expression: %s' % expr)
            
            # Test the expression:
            testfld = [np.ones((2,2))]*len(varlist)
            try:
                self._evaluation(testfld)
            except NameError:
                raise GradsfileError('Unknown identifier in expression: %s' % expr)
            except:
                raise GradsfileError('Invalid expression: %s' % expr)
            
            self.vars = varlist
            
        else:
            self.vars = [expr]
            self._evaluation = None

    def describe(self):
        if self._evaluation:
            what = '(%s:%s)' % (self.desc.file, self._var_request)
        else:
            what = '(%s:%s)' % (self.desc.file, ' '.join(self.vars))
        return what
            
            
    def _set_offsets(self):
        if not all(var in self.desc.vars for var in self.vars):
            raise GradsfileError('Requsted variable is not available')
        self._var_offsets = []
        self._var_vert_sizes = []
        offset = 0
        for var in self.desc.vars:
            nz = self.desc.nz[var]
            if var in self.vars:
                self._var_offsets.append(offset)
                self._var_vert_sizes.append(nz)
            offset += nz
        assert len(self._var_offsets) == len(self.vars)
        nz_first = self._var_vert_sizes[0]
        if not all(nz == nz_first for nz in self._var_vert_sizes):
            raise GradsfileError('Variables must have same z dimensions')
        if nz_first == 1 and self.ind_level is not None:
            raise GradsfileError('Requesting level for a 2D variable')
        if self.ind_level is None:
            self.nz = nz_first
        else:
            self.nz = 1
            
    def _set_file_lims(self):
        self._num_fields_in_file = sum(self.desc.nz.values())
        cur_start = 0
        filenow = None
        self._times = self.desc.times()
        self._glob_lims_for_file = {}
        for ind_time, time in enumerate(self._times):
            prev_start = cur_start
            fileprev = filenow
            filenow = expandGradsTemplate(self.desc.template, time)
            changed = fileprev and filenow != fileprev
            if changed:
                cur_start = ind_time
                prev_end = ind_time
                self._glob_lims_for_file[fileprev] = (prev_start, prev_end)
                #print('filename, limits', fileprev, prev_start, prev_end)
        self._glob_lims_for_file[filenow] = (cur_start, ind_time+1)
        
    def __init__(self, ctl, var, level=None, mask_mode='full'):
        allowed_mask_modes = 'full', 'first', 'false', 'none'
        if not mask_mode in allowed_mask_modes:
            raise ValueError('Invalid mask_mode')
        self.mask_mode = mask_mode
        try: 
            ctl.nx
            self.desc = ctl
        except AttributeError: # not a descriptor object
            self.desc = GradsDescriptor(ctl)
        
        assert isinstance(var, basestring)
        self._parse_expression(var)
        self.ind_level = level
        self._set_offsets()
        self._set_file_lims()
        self._shape3d = (self.desc.nx, self.desc.ny, self.nz)
        self._shape2d = (self.desc.nx, self.desc.ny)
        
        self._need_byteswap = (self.desc.byteorder != sys.byteorder)

        self._mm = None        
        self.end = self.t()[-1]
        self.begin = self.t()[0]
        self.now = self.t()[0]
        self._ind_glob = self._ind_loc = 0
        self.eof = False
        self._file_cur = None
        self._first_mask = None
        self.goto(self.begin)
        self.ones = np.ones(self._shape3d, order='F')
        
    def close(self):
        if self._file_cur:
            assert self._mm is not None
            self.mmcache.release(self._file_cur)
           
    def _switch_file(self, file_new):
        if not file_new in self._glob_lims_for_file:
            raise GradsfileError('File not available: %s' % file_new)
        lims = self._glob_lims_for_file[file_new]
        steps_in_file = lims[1] - lims[0]
        file_shape = self._shape2d + (self._num_fields_in_file, steps_in_file)
        if self._file_cur:
            self.mmcache.release(self._file_cur)
        self._mm = self.mmcache.get_mmap(file_new, file_shape)
        self._ind_loc = 0
        self._file_cur = file_new
        
    def goto(self, when):
        if when < self.begin or when > self.end:
            raise GradsfileError('Time not covered: %s' % str(when))
        file_now = expandGradsTemplate(self.desc.template, when)
        #print('goto', when)
        if file_now != self._file_cur:
            self._switch_file(file_now)
        lims = self._glob_lims_for_file[file_now]
        found = False
        for ind_glob in xrange(*lims):
            #print('ig', ind_glob)
            ind_loc = ind_glob - lims[0]
            if self._times[ind_glob] == when:
                found = True
                break
        if not found:
            raise GradsfileError('Time not available: %s' % str(when))
        self._ind_loc = ind_loc
        self._ind_glob = ind_glob
        self.now = self._times[ind_glob]
        self.eof = False
        
    def rewind(self):
        self.goto(self.begin)
    
    def tell(self):
        if self.eof:
            raise EndOfFileException()
        return self._times[self._ind_glob]
    
    def seek(self, num_steps):
        if num_steps == 0:
            return
        if num_steps < 0:
            raise ValueError('Cannot seek backwards')
        ind_glob_new = self._ind_glob + num_steps
        #print('seek: ind_glob_new', ind_glob_new, len(self._times))
        if ind_glob_new >= len(self._times):
            raise EndOfFileException()    
        self.goto(self._times[ind_glob_new])
        assert self._ind_glob == ind_glob_new

    def _get_field_slice(self, ind_var_local):
        # argument: index in variables to read
        var_offset = self._var_offsets[ind_var_local]
        if self.ind_level is None:
            field_slice = slice(var_offset, var_offset+self._var_vert_sizes[ind_var_local])
        else:
            field_slice = slice(var_offset+self.ind_level, var_offset+self.ind_level+1)
        return field_slice

    
    def _read_one(self, ind_var=0):
        field_slice = self._get_field_slice(ind_var)
        data = self._mm[:,:,field_slice,self._ind_loc]
        if self._need_byteswap:
            # False:#self._need_byteswap:
            data = data.byteswap()
        return data
    
    def _eval_mask(self, mask_fields):
        if self.mask_mode == 'first' and self._first_mask is not None:
            return self._first_mask
        if len(mask_fields) == 1:
            mask = np.abs(mask_fields[0] - self.desc.undef) < self.MASK_TOL
        else:
            near_undef = [np.abs(field - self.desc.undef) < self.MASK_TOL for field in mask_fields]
            mask = reduce(np.logical_or, near_undef)
        if self._first_mask is None:
            self._first_mask = mask
        return mask

    def read(self, num_steps=1, squeeze=True):
        if self.eof:
            raise EndOfFileException()
        num_vars_out = 1
        readbufr = np.empty(self._shape3d + (num_vars_out, num_steps), np.float32)
        need_mask = self.mask_mode == 'full' or self.mask_mode == 'first' 
        if need_mask:
            maskbufr = np.empty(readbufr.shape, dtype=bool)
            mask_fields = []
        for ind_step in range(num_steps):
            eval_fields = []
            for ind_var in range(len(self.vars)):
                selection = self._read_one(ind_var)
                if self._evaluation:
                    eval_fields.append(selection)
                else:
                    readbufr[...,ind_var,ind_step] = selection[...]
                if need_mask:
                    mask_fields.append(selection)
            if self._evaluation:
                result = self._evaluation(eval_fields)
                readbufr[...,0,ind_step] = result
            if need_mask:
                maskbufr[...,0,ind_step] = self._eval_mask(mask_fields)
            if self._ind_glob == len(self._times) - 1:
                self.eof = True
            else:
                self.seek(1)

        if squeeze:
            readbufr = readbufr.squeeze()
        if need_mask:
            return ma.MaskedArray(readbufr, maskbufr)
        elif self.mask_mode == 'false':
            return ma.MaskedArray(readbufr, False)
        else:
            return readbufr

    def nvars(self):
        # no support for multiple variables in one reader
        return 1


class MemCache:
    def __init__(self, size, dtype=np.float32):
        self.cachesize = size
        self._arrlst = []
        self._arrays  = {}
        self.dtype = dtype
        
    def _add(self, filename, shape):
        if len(self._arrlst) >= self.cachesize:
            self._remove(self._arrlst[0])
        data = np.fromfile(filename, dtype=self.dtype).reshape(shape, order='F')
        # the last element tells how many times the map has been requested or released
        self._arrays[filename] = (data, shape, [0])
        self._arrlst.append(filename)

    def _remove(self, filename):
        arr, shape_cur, arr_count = self._arrays[filename]
        arr_count[0] -= 1
        if arr_count[0] == 0:
            del(self._arrays[filename])
            ind = self._arrlst.index(filename)
            del(self._arrlst[ind])

    def get_mmap(self, filename, shape):
        filename = path.realpath(filename)
        if not filename in self._arrays:
            self._add(filename, shape)
        arr, shape_cur, arr_count = self._arrays[filename]
        if shape_cur != shape:
            raise ValueError('Shape requested differs from cached')
        arr_count[0] += 1
        return arr

    def release(self, filename):
        real = path.realpath(filename)
        if not real in self._arrays:
            return
        self._remove(real)

mcache_size = 1
mcache = MemCache(mcache_size)
class BrutalGradsfile(MMGradsfile):
    mmcache = mcache
        
def parseGradsTime(timestr):
    if len(timestr) == 12:
        (hr, d, mon, year) = (timestr[0:2], timestr[3:5], timestr[5:8], timestr[8:12])
        min = 0
    elif len(timestr) == 15:
        (hr, min, d, mon, year) = (timestr[0:2], timestr[3:5], timestr[6:8],
                                   timestr[8:11], timestr[11:15])
    else:
        raise GradsfileError('Unable to parse time: %s' % timestr)
    
    months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep',
              'oct', 'nov', 'dec']
    month = months.index(mon.lower()) + 1
    return dt.datetime(int(year), month, int(d), int(hr), int(min))

def makeGradsTime(time):
    mon = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep',
              'oct', 'nov', 'dec'][time.month-1]
    fmt = '%H:%MZ%d' + mon + '%Y'
    return time.strftime(fmt)

def parseGradsInterval(intStr):
    unit = intStr[-2:]
    value = int(intStr[0:-2])
    if unit == 'hr':
        return dt.timedelta(hours=value)
    elif unit == 'mn':
        return dt.timedelta(minutes=value)
    elif unit == 'dy':
        return dt.timedelta(days=value)
    elif unit == 'mo':
        return GradsDescriptor.MONTHLY
    else:
        raise GradsfileError('Sorry, time unit '+unit+' is not supported')

def makeGradsInterval(timedelta):
    seconds = timedelta.total_seconds()
    if seconds % 60 != 0:
        raise ValueError('Timestep needs to be in full minutes')
    minutes = seconds / 60
    if minutes % 60 == 0:
        hours = minutes / 60
        return '%ihr' % hours
    else:
        return '%imn' % minutes
