import numpy as np
from numpy import cos, sin, arcsin, arctan2, sqrt
from toolbox import namelist, projections

R_earth = 6378.1e3 
deg2rad = np.pi / 180. 
geo_proj = projections.LatLon()  # just to define the standard geographical projection


def xyzFromLonLat(lons,lats):
    #geographic degrees to 3D cartesian in km, assuming sperical earth
    x = R_earth * np.cos(lats * deg2rad) * np.cos(lons * deg2rad)
    y = R_earth * np.cos(lats * deg2rad) * np.sin(lons * deg2rad)
    z = R_earth * np.sin(lats * deg2rad)
    return x,y,z

def lonlatFromXYZ(x,y,z):
    lons = np.arctan2(y, x) / deg2rad
    lats = np.arctan2(z, np.sqrt(x**2 + y**2))/deg2rad
    return lons,lats

def phithetaFromXYZ(x,y,z):
    phi = np.arctan2(y, x) 
    theta = np.arctan2(z, np.sqrt(x**2 + y**2))
    return phi, theta


def gc_distance(x1,x2,y1,y2, r=6378.0):
    xr1 = x1*np.pi/180
    xr2 = x2*np.pi/180
    yr1 = y1*np.pi/180
    yr2 = y2*np.pi/180
    
    dx = np.abs(xr1-xr2)
    dsg = (np.sin((yr2-yr1)/2))**2 + np.cos(yr1)*np.cos(yr2)*(np.sin(dx/2))**2
    dsg = 2*np.arcsin(np.sqrt(dsg))
    return r*dsg

def dd(a, dim=0, dx = 1.0):
    # derivative along dimension dim
    d = np.zeros(a.shape)    
    if dim > 2:
        raise ValueError('Not so many dims!')
    if dim == 0:
        d[1:-1,...] = (a[2:,...]-a[0:-2,...]) / (2*dx)
        d[0,...] = (a[1,...]-a[0,...]) / dx
        d[-1,...] = (a[-1,...]-a[-2,...]) / dx
    if dim == 1:
        d[:,1:-1,...] = (a[:,2:,...]-a[:,:-2,...]) / (2*dx)
        d[:,0,...] = (a[:,1,...]-a[:,0,...]) / dx
        d[:,-1,...] = (a[:,-1,...]-a[:,-2,...]) / dx
    if dim == 2:
        d[:,:,1:-1,...] = (a[:,:,2:,...]-a[:,:,:-2,...]) / (2*dx)
        d[:,:,0,...] = (a[:,:,1,...]-a[:,:,0,...]) / dx
        d[:,:,-1,...] = (a[:,:,-1,...]-a[:,:,-2,...]) / dx
    return d

def divergence(u,v,w=None, d=None):
    if d == None:
        d = np.ones(u.ndim)
    else:
        d = np.array(d)
    # sanity check
    if w != None:
        assert(u.ndim == 3)
        fields = (u,v,w)
    else:
        assert(u.ndim == 2 and v.ndim == 2)
        fields = (u,v)

    div = np.zeros(u.shape)
    for i in range(0,len(fields)):
        div += dd(fields[i], i, d[i])
    return div

def grad(a, d = None):
    if d == None:
        d = np.ones(a.ndim)
    else:
        d = np.array(d)
    g = [dd(a,i,d[i]) for i in range(0,a.ndim)]
    return g

def curl(u, v, w=None, d=None):
    if u.ndim > 2:
        raise ValueError('Not implemented')

    if w is not None:
        raise ValueError('Not implemented')
    else:
        c = dd(v, 0, d[0]) - dd(u, 1, d[1])

    return c

def curl2d(psi, dx, dy):
    return dd(psi, 1, dy), -dd(psi, 0, dx)


def aave(F, lats, lons):
    # Area weighted average of F
    # lats, lons are assumed to represent a regular grid.
    
    dx = lons[1] - lons[0]
    dy = lats[1] - lats[0]

    if not all(lats[1:] - lats[0:-1] == dy) \
       or not all(lons[1:] - lons[0:-1] == dx):
        raise ValueError('Sorry, the grid has to be regular.')
    
    area = R**2 * dy*deg2rad * dx*deg2rad * np.cos(lats*deg2rad)
    area = np.outer(np.ones(lons.size), area)

    area_total = R**2 * deg2rad * (lons[-1] - lons[0])\
        * (sin(deg2rad*lats[1]) - sin(deg2rad*lats[0]))
    
    return np.sum(F * area) / area_total
    

def area(lats, lons):
    # Area of gridpoints defined by lats & lons.
    lats = np.asarray(lats)
    lons = np.asarray(lons)
    dx = abs(lons[1] - lons[0])
    dy = abs(lats[1] - lats[0])
    area = R_earth**2 * dy*deg2rad * dx*deg2rad * np.cos(lats*deg2rad)
    area = np.outer(area, np.ones(lons.size))
    return area.T


class Grid:
    def __init__(self, x0, dx, nx, y0, dy, ny, projection):
        self.x0 = x0
        self.y0 = y0
        self.dx = dx
        self.dy = dy
        self.nx = nx
        self.ny = ny
        self.proj = projection
        self.shape = (nx, ny)

    def grid_to_geo(self, X, Y):
        x = self.x0 + X*self.dx
        y = self.y0 + Y*self.dy
        lon, lat = self.proj.to_geo(x,y)
        return lon, lat
    
    def geo_to_grid(self, lon, lat):
        if self.proj == geo_proj: # speedup
            x = lon
            y = lat
        else:
            x, y = self.proj.to_proj(lon, lat)
        X = (x-self.x0) / self.dx
        # for the case of global map, check +- one revolution
        X = np.where(X<-0.5, X+360./self.dx, X)
        X = np.where(X>=self.nx-0.5, X-360./self.dx, X)
        Y = (y-self.y0) / self.dy
        return X, Y

    def proj_to_grid(self, x, y):
        X = (x-self.x0) / self.dx
        Y = (y-self.y0) / self.dy
        return X, Y

    def proj_to_geo(self, x, y):
        return self.proj.to_geo(x, y)

    def x(self):
        return self.x0 + np.arange(self.nx)*self.dx
    
    def y(self):
        return self.y0 + np.arange(self.ny)*self.dy
    
    def cell_dimensions(self):
        Y, X = np.meshgrid(np.arange(self.ny), np.arange(self.nx))
        lon, lat = self.grid_to_geo(X.ravel(), Y.ravel())
        dsx = self.proj.ds_dxi(lat, lon)*self.dx
        dsy = self.proj.ds_dnu(lat, lon)*self.dy
        return dsx, dsy

    def toCDOgrid(self):   # acually, returns a namelist
        nl = namelist.Namelist('')
        nl.set('gridtype' , 'lonlat')  # projection ???
        nl.set('gridsize' ,  self.nx * self.ny)
        nl.set('xsize'    ,  self.nx)
        nl.set('ysize'    ,  self.ny)
        nl.set('xname'    ,  'lon')
        nl.set('xlongname',  "longitude")
        nl.set('xunits'   ,  "degrees_east")
        nl.set('yname'    ,  'lat')
        nl.set('ylongname',  "latitude")
        nl.set('yunits'   ,  "degrees_north")
        nl.set('xfirst'   ,  self.x0)
        nl.set('xinc'     ,  self.dx)
        nl.set('yfirst'   ,  self.y0)
        nl.set('yinc'     ,  self.dy)
        return nl


def fromCDOnamelist(nlIn):
    x0 = nlIn.get_float('xfirst')[0]
    nx = nlIn.get_int('xsize')[0]
    dx = nlIn.get_float('xinc')[0]
    y0 = nlIn.get_float('yfirst')[0]
    ny = nlIn.get_int('ysize')[0]
    dy = nlIn.get_float('yinc')[0]
    if dx < 0:
        x0 += dx * (nx-1)
        dx = -dx
    if dy < 0:
        y0 += dy * (ny-1)
        dy = -dy
    return Grid(x0, dx, nx, y0, dy, ny, 
                {'LONLAT':projections.LatLon()}[nlIn.get_uniq('gridtype').upper()])

def fromCDOgrid(chCDOfile):    
    # a wrapper for the namelist initialization
    nlIn = namelist.Namelist.fromfile(chCDOfile)
    return fromCDOnamelist(nlIn)

def gems_grid():
    import projections
    return Grid(-17.0, 0.2, 265, 33.0, 0.2, 195, projections.LatLon())

def gems_grid_ext():
    import projections
    return Grid(-25.0, 0.2, 351, 30.0, 0.2, 220, projections.LatLon())

def latLonFromPoints(lats, lons):
    import projections
    dx = lons[1]-lons[0]
    dy = lats[1]-lats[0]
    (nx, ny) = (len(lons), len(lats))
    return Grid(lons[0], dx, nx, lats[0], dy, ny, projections.LatLon())

