#from numpy import *
#import numpy as np
from numpy import sin, cos, tan, pi, empty, isscalar, tensordot, dot, array, arcsin, arctan2, all, zeros

R = 6370.0e3
deg_to_rad = pi / 180
rad_to_deg = 180 / pi
rot_void_R = zeros(shape=(3,3))
for i in range(3): rot_void_R[i,i] = 1

def sec(x):
    return 1.0 / cos(x)

def csc(x):
    return 1.0 / sin(x)

def cot(x):
    return 1.0 / tan(x)

class Projection:
   
    def metric(self, lat, lon):
        # dx_dlon; dy_dlat
        m = empty(2)
        m[0] = R * cos(deg_to_rad*lat)
        m[1] = R 
        return m

    def __eq__(self, other):
        pass

    def to_proj(self, lon, lat):
        return self.xi(lat, lon), self.nu(lat, lon)
    
    def to_geo(self, x, y):
        return self.lon(x, y), self.lat(x, y)

    def xi(self, lat, lon):
        pass
    
    def nu(self, lat, lon):
        pass

    def lat(self, x, y):
        pass
    
    def lon(self, x, y):
        pass
    
    def dxi_dlat(self, lat, lon):
        pass

    def dnu_dlat(self, lat, lon):
        pass
    
    def dxi_dlon(self, lat, lon):
        pass
    
    def dnu_dlon(self, lat, lon):
        pass

    def jacobian(self, lat, lon):
        if isscalar(lat):
            shape = ()
        else:
            shape = lat.shape

        J = zeros(shape + (2,2))
        J[...,0,0] = self.dxi_dlon(lat, lon)
        J[...,0,1] = self.dxi_dlat(lat, lon)
        J[...,1,0] = self.dnu_dlon(lat, lon)
        J[...,1,1] = self.dnu_dlat(lat, lon)
        return J

    def dxi_dx(self, lat, lon):
        dx_dlon = R * cos(deg_to_rad*lat)
        return self.dxi_dlon(lat, lon) / dx_dlon

    def dnu_dy(self, lat, lon):
        dy_dlat = R
        return self.dnu_dlat(lat, lon) / dy_dlat

    def ds_dxi(self, lat, lon):
        dx_dlon = R * cos(deg_to_rad*lat)
        dy_dlat = R
        J = self.jacobian(lat, lon)
        Jinv = linalg.inv(J)
        dlon_dxi = Jinv[0,0]
        #dlon_dnu = Jinv[0,1]
        dlat_dxi = Jinv[1,0]
        #dlat_dnu = Jinv[1,1]
        dx_dxi = dx_dlon * dlon_dxi
        dy_dxi = dy_dlat * dlat_dxi
        return sqrt(dx_dxi**2 + dy_dxi**2)
    
    def ds_dnu(self, lat, lon):
        dx_dlon = R * cos(deg_to_rad*lat)
        dy_dlat = R
        J = self.jacobian(lat, lon)
        det = J[...,0,0]*J[...,1,1] - J[...,0,1]*J[...,1,0]

        Jinv = linalg.inv(J)
        
        dlon_dnu = -J[...,0,1] / det
        dlat_dnu = J[...,0,0] / det

        #dlon_dxi = Jinv[0,0]
        #dlon_dnu = Jinv[...,0,1]
        #dlat_dxi = Jinv[1,0]
        #dlat_dnu = Jinv[...,1,1]
        dx_dnu = dx_dlon * dlon_dnu
        dy_dnu = dy_dlat * dlat_dnu
        return sqrt(dx_dnu**2 + dy_dnu**2)

    def rotation(self, lat, lon):
        # latlon -> projected
        J = self.jacobian(lat, lon)
        M = diag([self.ds_dxi(lat, lon), self.ds_dnu(lat, lon)])
        return dot(M, dot(J, diag(self.metric(lat, lon)**-1)))


class NoProjection(Projection):
    def xi(self, lat, lon):
        return lon
    def nu(self, lat, lon):
        return lat
    def __eq__(self, other):
        return True
    def lat(self, x, y):
        return y
    def lon(self, x, y):
        return x
    def dxi_dlat(self, lat, lon):
        return 0.0
    def dnu_dlat(self, lat, lon):
        return 1.0
    def dxi_dlon(self, lat, lon):
        return 1.0
    def dnu_dlon(self, lat, lon):
        return 0.0
    
    
class LatLon(Projection):
    
    def __init__(self, lat_southpole=-90.0, lon_southpole=0.0, gamma=0.0):
        self.rot_3d = Rotation(lat_southpole, lon_southpole, gamma)
        self.south_pole = (lon_southpole, lat_southpole)
        
    def __eq__(self, other):
        return self.south_pole == other.south_pole and self.rot_3d == other.rot_3d
    
    def xi(self, lat, lon):
        return self.rot_3d.latLonToRotated(lat, lon)[1]
    
    def nu(self, lat, lon):
        return self.rot_3d.latLonToRotated(lat, lon)[0]

    def lat(self, x, y):
        try:
            if all(self.rot_3d.R.ravel() == rot_void_R.ravel()): return y
        except: pass
        return self.rot_3d.latLonToWorld(y, x)[0]

    def lon(self, x, y):
        try:
            if all(self.rot_3d.R.ravel() == rot_void_R.ravel()): return x
        except: pass
        return self.rot_3d.latLonToWorld(y, x)[1]
    
    def ds_dxi(self, lat, lon):
        latmod = self.nu(lat, lon)
        return deg_to_rad * R * cos(deg_to_rad*latmod)
    
    def ds_dnu(self, lat, lon):
        return deg_to_rad * R

    def rotation(self, lat_geo, lon_geo):
        lat = self.nu(lat_geo, lon_geo)
        lon = self.xi(lat_geo, lon_geo)
        # This is from world to rotated!!
        # latdeg, londeg *already* in rotated crd
        #lat = latdeg * deg_to_rad
        #lon = londeg * deg_to_rad
        #print lat, lon
        lat2, lon2 = (lat_geo*deg_to_rad, lon_geo*deg_to_rad)
        # Basis in the reference frame
        B1 = get_basis(lat, lon)
        #print B1
        # Rotate B1 into modified frame
        B1 = dot(self.rot_3d.R, B1)
        # Basis in the rotated frame
        B2 = get_basis(lat_geo, lon_geo)
        #print B2
        # Project the vector in modified B1-basis into B2 basis
        r = dot(B2.T, B1)
        return r.T


class LambertConformalConic(Projection):
    
    def __init__(self, phi0, phi1, phi2, lam0):
        (self.phi0, self.phi1, self.phi2, self.lam0) = (deg_to_rad*phi0, deg_to_rad*phi1, 
                                                        deg_to_rad*phi2, deg_to_rad*lam0)
        
        self.n = (log(cos(self.phi1) * sec(self.phi2))
                  / log(tan(pi/4 + self.phi2/2) * cot(pi/4 + self.phi1/2)))
       
        self.F = cos(self.phi1) * tan(pi/4 + self.phi1/2)**self.n / self.n
        self.rho0 = self.F * cot(pi/4 + self.phi0/2)**self.n

    def __eq__(self, oth):
        return self.lam0 == oth.lam0 and self.F == oth.F and self.n == oth.n and self.rho == oth.rho

    def xi(self, lat, lon):
        rho = self.F * cot(pi/4 + deg_to_rad*lat/2)**self.n
        return rho * sin(self.n * (lon*deg_to_rad - self.lam0))
    
    def nu(self, lat, lon):
        rho = self.F * cot(pi/4 + deg_to_rad*lat/2)**self.n
        return self.rho0 - rho * cos(self.n*(lon*deg_to_rad - self.lam0))

    def lat(self, xi, nu):
        rho = sign(self.n) * sqrt(xi**2 + (self.rho0-nu)**2)
        phi = 2 * arctan((self.F / rho) ** (1/self.n)) - pi/2
        return phi * rad_to_deg

    def lon(self, xi, nu):
        theta = arctan(xi / (self.rho0 - nu))
        lam = self.lam0 + theta / self.n
        return lam * rad_to_deg

    def dxi_dlon(self, lat, lon):
        rho = self.F * cot(pi/4 + deg_to_rad*lat/2)**self.n
        return self.n * rho * cos(self.n * (lon*deg_to_rad - self.lam0))

    def dxi_dlat(self, lat, lon):
        drho_dlat = (-self.F * self.n * cot(pi/4 + deg_to_rad*lat/2)**(self.n-1) 
                     *csc(pi/4 + deg_to_rad*lat/2)**2) * 0.5
        return drho_dlat * sin(self.n * (lon*deg_to_rad - self.lam0))

    def dnu_dlon(self, lat, lon):
        rho = self.F * cot(pi/4 + deg_to_rad*lat/2)**self.n
        return rho * self.n * sin(self.n * (deg_to_rad*lon - self.lam0))

    def dnu_dlat(self, lat, lon):
        drho_dlat = (-self.F * self.n * cot(pi/4 + deg_to_rad*lat/2)**(self.n-1) 
                     *csc(pi/4 + deg_to_rad*lat/2)**2) * 0.5
        return -drho_dlat * cos(self.n * (deg_to_rad*lon - self.lam0))
    
    
#
# Spherical geometry & rotations
#

def get_basis(latdeg, londeg):
    lat = latdeg * deg_to_rad
    lon = londeg * deg_to_rad
    
    B = empty((3,2))
    
    B[0,0] = -sin(lon)
    B[1,0] = cos(lon)
    B[2,0] = 0.
    B[0,1] = -cos(lon)*sin(lat)
    B[1,1] = -sin(lon)*sin(lat)
    B[2,1] = cos(lat)

    return B


def get_rotation(lat_pole, lon_pole):
    # Right multiplication -> world to rotated
    # Left multiplication -> rotated to world 
   
    latr = deg_to_rad * lat_pole
    lonr = deg_to_rad * lon_pole

    alpha = (pi/2. + lonr)
    beta = pi/2. - latr
    gamma = 0.0#-alpha

    ca = cos(alpha)
    cb = cos(beta)
    cg = cos(gamma)
    sa = sin(alpha)
    sb = sin(beta)
    sg = sin(gamma)

    R = empty((3,3))

    R[0,0] = ca*cg - sa*cb*sg
    R[1,0] = sa*cg + ca*cb*sg
    R[2,0] = sb*sg
    R[0,1] = -ca*sg - sa*cb*cg
    R[1,1] = -sa*sg + ca*cb*cg
    R[2,1] = sb*cg
    R[0,2] = sb*sa
    R[1,2] = -sb*ca
    R[2,2] = cb

    return R

def get_rotation_v2(lat_pole, lon_pole, gamma=0.0):
    # Right multiplication -> world to rotated
    # Left multiplication -> rotated to world 
    
    latr = deg_to_rad * lat_pole
    lonr = deg_to_rad * lon_pole

    alpha = lonr
    beta = pi/2 + latr
    gamma = deg_to_rad * gamma
    
    ca = cos(alpha)
    cb = cos(beta)
    cg = cos(gamma)
    sa = sin(alpha)
    sb = sin(beta)
    sg = sin(gamma)

    R = empty((3,3))

    R[0,0] = ca*cb*cg - sa*sg
    R[1,0] = sa*cb*cg + ca*sg
    R[2,0] = sb*cg
    R[0,1] = -ca*cb*sg - sa*cg
    R[1,1] = -sa*cb*sg + ca*cg
    R[2,1] = -sb*sg
    R[0,2] = -ca*sb
    R[1,2] = -sa*sb
    R[2,2] = cb

    return R


class Rotation:
    def __init__(self, lat_pole, lon_pole, gamma=0.0):
        # SOUTH POLE
        self.R = get_rotation_v2(lat_pole, lon_pole, gamma)
        self.lat_pole = lat_pole * deg_to_rad
        self.lon_pole = lon_pole * deg_to_rad

    def __eq__(self, oth):
        return all(self.R == oth.R) and self.lat_pole == oth.lat_pole and self.lon_pole == oth.lon_pole

    def _transform_latlon(self, latdeg, londeg, R):
        lat = latdeg * deg_to_rad
        lon = londeg * deg_to_rad
        if isscalar(latdeg):
            shape = ()
        else:
            shape = latdeg.shape

        
        r1 = empty((3,) + shape)
        r1[0,...] = cos(lon)*cos(lat)
        r1[1,...] = sin(lon)*cos(lat)
        r1[2,...] = sin(lat)

#        print "-------------"
#        print "R=", R 
#        print R.shape
#        print "r1=", r1
#        print r1.shape
        r2 = tensordot(R, r1, axes=(1,0))

        ###Numerics causes problems in poles (very rarely)...
        #Dirty hack to clean it
        nlatd = arcsin(r2[2,...]*0.9999999) / deg_to_rad

        nlond = arctan2(r2[1,...], r2[0,...]) / deg_to_rad
        
        return nlatd, nlond
 
    def vector3ToLatLon(self, r):
        nlatd = arcsin(r[2]) / deg_to_rad
        nlond = arctan2(r[1], r[0]) / deg_to_rad
        return nlatd, nlond

    def latLonToVector3(self, latdeg, londeg):
        lat = latdeg * deg_to_rad
        lon = londeg * deg_to_rad
        
        r1 = empty(3)
        r1[0] = cos(lon)*cos(lat)
        r1[1] = sin(lon)*cos(lat)
        r1[2] = sin(lat)

        return r1

    def latLonToWorld(self, latdeg, londeg):
        return self._transform_latlon(latdeg, londeg, self.R)

    def latLonToRotated(self, latdeg, londeg):
        return self._transform_latlon(latdeg, londeg, self.R.T)

    def vector3ToWorld(self, vector3):
        return dot(self.R, vector3)
    
    def vector3ToRotated(self, vector3):
        return dot(self.R.T, vector3)

    def transform_wind(self, u, v, latdeg, londeg, R):
        # This is from world to rotated!!
        # latdeg, londeg *already* in rotated crd
        lat = latdeg * deg_to_rad
        lon = londeg * deg_to_rad
        
        lat2, lon2 = self.transform_latlon(latdeg, londeg, R)
        # Basis in the reference frame
        B1 = get_basis(lat2, lon2)
        # Rotate B1 into modified frame
        B1 = dot(R.T, B1)
        # Basis in the rotated frame
        B2 = get_basis(latdeg, londeg)
        # Project the vector in modified B1-basis into B2 basis
        r = dot(B2.T, B1)
        V = array([u,v])
        return dot(r[0,:], V), dot(r[1,:], V)

    
    def windToRotated(self, u, v, latdeg, londeg):
        return transform_wind(u, v, latdeg, londeg, self.R)
    
    def windToWorld(self, u, v, latdeg, londeg):
        return transform_wind(u, v, latdeg, londeg, self.R.T)



if __name__ == '__main__':     
    proj = LambertConformalConic(0, 50, 60, 0)

    dx_m = 80.0 * 1e3
    dy_m = 80.0 * 1e3

    nx = 30
    ny = 30

    lat0 = 40
    lon0 = -5

    dxi = dx_m / proj.ds_dxi(lat0, lon0) #proj.dxi_dx(lat0, lon0) * dx_m
    dnu = dy_m / proj.ds_dnu(lat0, lon0) #proj.dnu_dy(lat0, lon0) * dy_m

    xi, nu = meshgrid(proj.xi(lat0,lon0) + arange(nx)*dxi, proj.nu(lat0, lon0) + arange(ny)*dnu)

    import matplotlib as mpl
    from mpl_toolkits.basemap import Basemap
    import figs
    import pylab
    pylab.figure(1)
    pylab.clf()

    lat = proj.lat(xi, nu)
    lon = proj.lon(xi, nu)
    mp = Basemap(projection='cyl',llcrnrlon=lon0-10, urcrnrlat=lat0+30,
                 urcrnrlon=lon0+40, llcrnrlat=lat0-20,resolution='l')
    u = array([1,0])
    up = zeros((nx*ny,2))
    v = array([0,1])
    vp = up.copy()
    for i in range(nx*ny):
        up[i,:] = dot(proj.rotation(lat.ravel()[i], lon.ravel()[i]).T, u)
        vp[i,:] = dot(proj.rotation(lat.ravel()[i], lon.ravel()[i]).T, v)

    mp.scatter(lon.ravel(), lat.ravel(),c='r')
    mp.quiver(lon.ravel(), lat.ravel(), up[:,0], up[:,1])
    mp.quiver(lon.ravel(), lat.ravel(), vp[:,0], vp[:,1])
    figs.finishMap(mp, 5, 5, 'k')

    pylab.figure(2)
    pylab.clf()

    proj2 = LatLon(-30, 0)

    dx_m = 80.0 * 1e3
    dy_m = 80.0 * 1e3

    nx = 30
    ny = 30

    lat0 = 40
    lon0 = -5

    dxi = dx_m / proj2.ds_dxi(lat0, lon0) #proj2.dxi_dx(lat0, lon0) * dx_m
    dnu = dy_m / proj2.ds_dnu(lat0, lon0) #proj2.dnu_dy(lat0, lon0) * dy_m

    xi, nu = meshgrid(proj2.xi(lat0,lon0) + arange(nx)*dxi, proj2.nu(lat0, lon0) + arange(ny)*dnu)

    import matplotlib as mpl
    from mpl_toolkits.basemap import Basemap
    import figs

    lat = proj2.lat(xi.ravel(), nu.ravel())
    lon = proj2.lon(xi.ravel(), nu.ravel())
    mp = Basemap(projection='cyl',llcrnrlon=lon0-10, urcrnrlat=lat0+30,
                 urcrnrlon=lon0+40, llcrnrlat=lat0-20,resolution='l')
    u = array([1,0])
    up = zeros((nx*ny,2))
    v = array([0,1])
    vp = up.copy()
    for i in range(nx*ny):
        up[i,:] = dot(proj2.rotation(lat.ravel()[i], lon.ravel()[i]).T, u)
        vp[i,:] = dot(proj2.rotation(lat.ravel()[i], lon.ravel()[i]).T, v)

    mp.scatter(lon.ravel(), lat.ravel(),c='b')
    mp.quiver(lon.ravel(), lat.ravel(), up[:,0], up[:,1])
    mp.quiver(lon.ravel(), lat.ravel(), vp[:,0], vp[:,1])
    figs.finishMap(mp, 5, 5, 'k')


