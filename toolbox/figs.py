import matplotlib as mpl
import pylab
import numpy as np
#import figs
try:
    from mpl_toolkits import basemap
    from mpl_toolkits.basemap import Basemap
except ImportError:
    print ('Warning: basemap not available')

clevs_default = [0, 0.05, 0.1, 0.2, .5, 1., 2., 5., 10., 20., 50., 100., 200., 500., 1000., 2000.]

#clevs_vars = {'cnc_so2': [0, 3e-9, 1e-8, 3e-8, 1e-7, 3e-7, 1e-6, 3e-6]}
clevs_vars = {'cnc_no2' : clevs_default[:-2],
              'cnc_so2' : clevs_default[:-2],
              'cnc_sslt' : clevs_default[:-4],
              'ems_so2' : clevs_default[:-2],
              'ems_cor001' : [0, 0.25, 0.5, 0.75, 1., 1.25, 1.5, 1.75, 2],
              'cnc_pm2_5' : clevs_default[:-2],
              'cnc_pm10' : clevs_default[:-2],
              'cnc_pm_m_50' : clevs_default[:-6],
              'cnc_pm_m6_0' : clevs_default[:-6],
              'cnc_o3' : [0, 20., 40., 60., 80, 100, 120, 140, 150],
              'cnc_co' : [ 50, 100, 150, 200, 300, 400, 500, 700, 1000, 1500, 2000, 3000],
              'cnc_no' : clevs_default[:-4],
              'cnc_nh3' : clevs_default[:-4],
              }

ccols_vars = {'ems_cor001' : [mpl.cm.RdBu(clevs_vars['ems_cor001'][i] / 2) for i in range(0, 9)]}
gamma_vars = {'cnc_so2': 1.0, 'cnc_no2' : 1.2, 'cnc_seasalt' : 1.2}
gamma_default = 1.2

default_cm = mpl.cm.jet

def gammaScaler(data, gamma):
    a = np.min(data)
    b = np.max(data)
    f = lambda x: ((x-a) / np.abs(b-a))**gamma
    return f

def clevs(varNm, nlevs = 24):
    try:
        levs = clevs_vars[varNm.lower()]
    except KeyError:
        levs = clevs_default
    
    if len(levs) != nlevs and nlevs > 0:
        x = np.linspace(0,1,len(levs))
        xi = np.linspace(0,1,nlevs)
        levs = np.interp(xi, x, levs)

    return np.concatenate(([-1e-12],levs))

def cmap(varNm, cmbase=default_cm):
    levs = clevs(varNm)[1:]
    cols = ccols(varNm, baseCM=cmbase)
    norm = mpl.colors.Normalize(0,levs[-1])
    cdict = {'red' : [], 'green' : [], 'blue' : []}
    for l,c in zip(levs, cols):
        cdict['red'].append([norm(l),c[0],c[0]])
        cdict['green'].append([norm(l),c[1],c[1]])
        cdict['blue'].append([norm(l),c[2],c[2]])
    return mpl.colors.LinearSegmentedColormap(varNm, cdict, 512)

def cmap_and_norm_clevs(cmapname, clevs):
   # Input -- name of mpl colormap
   #          np.array of cleca
    #Generates colormap and norm with given clevs 
    # As in grads 
    basecm=pylab.cm.get_cmap(cmapname)
    nlevs=len(clevs)
    cols=basecm(np.linspace(0, 1, nlevs+1))
    norm = mpl.colors.BoundaryNorm(clevs, nlevs-1)
    #cm_out = pylab.cm.from_list("%s_discr"%cmapname, cols[1:-1], nlevs-1)
    cm_out = mpl.colors.ListedColormap(cols[1:-1], name="%s_discr"%cmapname)
    cm_out.set_under(cols[0])
    cm_out.set_over(cols[-1])
    cm_out.set_bad([0,0,0,0]) #Transparent
    return cm_out, norm



def norm(varNm):
    normn = mpl.colors.Normalize(0,clevs(varNm)[-1])
    return normn

def ccols(varNm, nlevs = 24, baseCM=default_cm):   
    if varNm in ccols_vars:
        ccols = ccols_vars[varNm]
        return ccols
    if varNm in gamma_vars:
        gamma = gamma_vars[varNm]
    else:
        gamma = gamma_default
    levs = clevs(varNm)
    sc = gammaScaler([0,1], gamma)
    points = np.linspace(0,1.,nlevs)
    clrs = baseCM(sc(points))
    clrs[...,3] = 0.5
    return clrs

def contourf_raw(x, y, F, clevs, ccols):
    fig = mpl.figure.Figure()
    ax = fig.add_axes([0,0,1,1])
    cont = ax.contourf(x, y, F.T, clevs, colors=ccols)
    for c in cont.collections:
        c.set_antialiased(False)
    for child in ax.get_children():
        if isinstance(child, mpl.patches.Rectangle): child.set_visible(0)
    ax.xaxis.set_visible(0)
    ax.yaxis.set_visible(0)
    #ax.axesFrame.set_visible(0)
    ax.set_frame_on(0)
    return fig

# def contourf_raw(x, y, F):
#     fig = mpl.figure.Figure()
#     ax = fig.add_axes([0,0,1,1])
#     ax.pcolor(x, y, F.T)
#     for child in ax.get_children():
#         if isinstance(child, mpl.patches.Rectangle): child.set_visible(0)
#     ax.xaxis.set_visible(0)
#     ax.yaxis.set_visible(0)
#     ax.axesFrame.set_visible(0)
#     return fig

def colorbarFig(varNm, cmbase=default_cm, ax=None, text=None):
    fig = mpl.figure.Figure(figsize=(1,8))
    
    if text != None and ax == None:
        ax = fig.add_axes([0, 0.05, 0.4, 0.85])
        axTx = fig.add_axes([0, 0.9, 1.0, 0.1])
        axTx.xaxis.set_visible(0)
        axTx.yaxis.set_visible(0)
        axTx.axesFrame.set_visible(0)
        axTx.text(0.05, 0.5, text)
    elif ax == None:
        ax = fig.add_axes([0, 0.05, 0.4, 0.9])
   

    cb = mpl.figure.cbar.ColorbarBase(ax, cmap(varNm, cmbase), 
                                      boundaries=clevs(varNm)[1:], 
                                      format='%2.1f')
    cb.draw_all()
    return fig


def map4file(gradsfile, resolution='l'):
    (x0,x1,y0,y1) = (gradsfile.x()[0], gradsfile.x()[-1],
                     gradsfile.y()[0], gradsfile.y()[-1])
    return Basemap(projection='cyl',llcrnrlon=x0, urcrnrlat=y1,
                   urcrnrlon=x1, llcrnrlat=y0,resolution=resolution)

def map4grid(grid, resolution='l'):
    (x0,x1,y0,y1) = (grid.x0, grid.x0+(grid.nx-1)*grid.dx,
                     grid.y0, grid.y0+(grid.ny-1)*grid.dy)
    return Basemap(projection='cyl',llcrnrlon=x0, urcrnrlat=y1,
                   urcrnrlon=x1, llcrnrlat=y0,resolution=resolution)

def worldMap(resolution='l'):
    x0, x1, y0, y1 = -180, 180, -90, 90
    return Basemap(projection='cyl',llcrnrlon=x0, urcrnrlat=y1,
                   urcrnrlon=x1, llcrnrlat=y0,resolution=resolution)

def NHMap(resolution='l'):
    x0, x1, y0, y1 = -180, 180, 0, 90
    return Basemap(projection='cyl',llcrnrlon=x0, urcrnrlat=y1,
                   urcrnrlon=x1, llcrnrlat=y0,resolution=resolution)

def finishMap(mp, dx=None, dy=None, color='w', labels='both', thicklines=False, fontsize=8):
    linewidth = 1.0 if thicklines else 0.5
    mp.drawcoastlines(color=color, linewidth=linewidth)
    mp.drawcountries(color=color)
    mp.drawmapboundary(color=color)
    deltas=np.array([1.,5.,10, 20, 30.])
    if mp.projection in  ['spstere','npstere']:
            meridians=np.arange(0,360, 30)
            # lrtb
            labelslist = [1,0,0,1]
            mp.drawmeridians(meridians,  color=color,
                 labels=labelslist, fontsize=fontsize, yoffset=0.5, rotation=0.0)
            parallels = np.arange(-60, 30, 90)
            mp.drawparallels(parallels, 
                     color=color, labels=labelslist, fontsize=fontsize, xoffset=0.3)
    else:
        if dx == None:
            xr = abs(mp.urcrnrlon - mp.llcrnrlon)
            dx = deltas[max(1, np.searchsorted(deltas,xr/3.))-1] # 
        if dy == None:
            yr = abs(mp.urcrnrlat - mp.llcrnrlat)
            dy = deltas[max(1, np.searchsorted(deltas,yr/3.))-1] # 
        if dy > 0:
            if labels.lower() == 'x' or labels.lower() == 'both':
                labelslist = [0,0,0,1]
            else:
                labelslist = [0,0,0,0]
            meridians=np.arange(np.floor(mp.llcrnrlon/dx)*dx, mp.urcrnrlon,dx)
            mp.drawmeridians(meridians,  color=color,
                    labels=labelslist, fontsize=fontsize, yoffset=0.5, rotation=0.0)
        if dx > 0:
            if labels.lower() == 'y' or labels.lower() == 'both':
                labelslist = [1,0,0,0]
            else:
                labelslist = [0,0,0,0]
            parallels = np.arange(np.floor(mp.llcrnrlat/dy)*dy , mp.urcrnrlat, dy)
            mp.drawparallels(parallels, 
                     color=color, labels=labelslist, fontsize=fontsize, xoffset=0.3)

def color_scatter(x, y, values, clevs, ccols=None, cmap=None, plotter=pylab):
    if cmap is None and ccols is None:
        raise ValueError('Must define either cmap or ccols')
    
    if ccols is None:
        ccols = cmap(np.linspace(0, 1, len(clevs)))
    colors = mpl.colors.ListedColormap(ccols)
    xx = []
    yy = []
    colorlist = []
    for xp, yp, val in zip(x, y, values):
        color = None
        for i in range(len(clevs)):
            if val > clevs[i]:
                color = colors.colors[i]
        if color is not None:
            colorlist.append(color)
            xx.append(xp)
            yy.append(yp)
    return plotter.scatter(xx, yy, c=colorlist)
    #1/0

def clevs_pcolor(X, Y, Z, clevs, cmap=None, ccols=None, plotter=pylab):
    # 
    # For the contourf type images, one can define a set of contour
    # levels and their colors. For pcolor plots, this is not
    # immediately possible, but can be achieved by making a listed
    # colormap, and plotting an array of indices to it. The colorbar
    # then needs to be created using the command below.
    #

    if cmap is None and ccols is None:
        raise ValueError('Must define either cmap or ccols')
    
    if ccols is None:
        ccols = cmap(np.linspace(0, 1, len(clevs)+1))
            
    # Make a listed colormap for the clevs
    
    colors = mpl.colors.ListedColormap(ccols)
    
    # Make an array indexing the colors. Initially everything clips.

    I = np.ones(Z.shape, dtype=np.int32) * 0
    for i in range(len(clevs)):
        I[Z > clevs[i]] = i+1

    # do pcolor
    
    try:
        I = np.ma.MaskedArray(I,Z.mask)
    except:
        I = np.ma.MaskedArray(I,np.logical_not(np.isfinite(Z)))
    
    return plotter.pcolormesh(X, Y, I, norm=mpl.colors.NoNorm(), cmap=colors)

def clevs_imshow(X, Y, Z, clevs, cmap=None, ccols=None, plotter=pylab):
    # 
    # For the contourf type images, one can define a set of contour
    # levels and their colors. For pcolor plots, this is not
    # immediately possible, but can be achieved by making a listed
    # colormap, and plotting an array of indices to it. The colorbar
    # then needs to be created using the command below.
    #

    if cmap is None and ccols is None:
        raise ValueError('Must define either cmap or ccols')
    
    if ccols is None:
        ccols = cmap(np.linspace(0, 1, len(clevs)+1))
            
    # Make a listed colormap for the clevs
    
    colors = mpl.colors.ListedColormap(ccols)
    
    # Make an array indexing the colors. Initially everything clips.

    I = np.ones(Z.shape, dtype=np.int32) * 0
    for i in range(len(clevs)):
        I[Z > clevs[i]] = i+1

    # do pcolor
    
    try:
        I = np.ma.MaskedArray(I,Z.mask)
    except:
        pass
    extent = [X[0], X[-1], Y[-1], Y[0]]
    plotter.imshow(I.T, norm=mpl.colors.NoNorm(), cmap=colors, interpolation='nearest', extent=extent)
    
def clevs_colorbar(clevs, orientation='horizontal', format=None, shrink=1.0, ticks=None,
                   rotate=None, fontsize=None, mappable=None):
    #import colorbar_jv
    clevs_ = [clevs[0]-100]+list(clevs)+[clevs[-1]+100]# + [10*clevs[-1]]
    #clevs_ = clevs
    #norm = mpl.colors.BoundaryNorm(clevs, cmap.N)
    if ticks is None:
        ticks = clevs
    cb = pylab.colorbar(mappable=mappable, boundaries=clevs_, values=range(0,len(clevs)+1), 
                        orientation=orientation, format=format, extend='both', shrink=shrink, ticks=ticks)
    labels = cb.ax.get_xticklabels()
    if rotate:
        mpl.pyplot.setp(labels, rotation=rotate)
    if fontsize:
        mpl.pyplot.setp(labels, fontsize=fontsize)
    #for t in cb.ax.get_xticklabels(): t.set_fontsize(7)
    #cb = pylab.colorbar(boundaries=clevs_, norm=norm, cmap=cmap,
    #                    orientation=orientation, format=format, extend='both', shrink=shrink)
    
    return cb

def clevs_colorbar_rev(clevs, orientation='horizontal', format=None, shrink=1.0, mappable=None):
    clevs_ = [-100*clevs[0]]+list(clevs)+[100*clevs[-1]]# + [10*clevs[-1]]
    cb = pylab.colorbar(mappable=mappable, boundaries=clevs_, values=range(len(clevs),-1,-1), 
                        orientation=orientation, format=format, extend='both', shrink=shrink,
                        ticks=clevs)
    for t in cb.ax.get_xticklabels(): t.set_fontsize(6)
    return cb


def pcolor_coordinates(x, x0):
    crd = np.empty(len(x)+1)
    crd[0] = x0
    for i in range(1,len(crd)):
        crd[i] = 2*x[i-1]-crd[i-1]
        if not crd[i] > crd[i-1]:
            raise ValueError('Inconsistent x values')

    return crd

def draw_grid(grid, map_, style='r', label=None):
    ypt = grid.y()
    xpt = grid.x()

    # N
    y = (grid.ny-1)*np.ones(grid.nx)
    lon, lat = grid.grid_to_geo(np.arange(grid.nx), y)
    map_.plot(lon, lat, style, label=label)
    #raise Exception('')
    # S
    y = np.zeros(grid.nx)
    lon, lat = grid.grid_to_geo(np.arange(grid.nx), y)
    map_.plot(lon, lat, style)
    # E
    x = (grid.nx-1)*np.ones(grid.ny)
    lon, lat = grid.grid_to_geo(x, np.arange(grid.ny))
    map_.plot(lon, lat, style)
    # W
    x = np.zeros(grid.ny)
    lon, lat = grid.grid_to_geo(x, np.arange(grid.ny))
    map_.plot(lon, lat, style)

    finishMap(map_, color='k', dx=10, dy=10)
    
class StackedAreaPlot:
    def __init__(self, x, ax=None):
        if not ax:
            ax = pylab
        self.ax = ax
        self.to_draw = []
        self.x = np.array(x)
        
    def add_component(self, data, color, label):
        self.to_draw.append((data, color, label))
        
    def draw(self):
        lower = np.zeros(self.x.shape)
        for curve, color, label in self.to_draw:
            upper = lower + curve
            self.ax.fill_between(self.x, lower, upper, facecolor=color)
            lower = upper
        
    def legend(self, ax=None, kwargs={}):
        rectangles = []
        labels = []
        if not ax:
            ax = self.ax
        for curve, color, label in self.to_draw:
            r = pylab.Rectangle((0, 0), 1, 1, fc=color)
            rectangles.append(r)
            labels.append(label)
        ax.legend(rectangles, labels, **kwargs)
        
if __name__ == '__main__':
    x = np.linspace(0,1,20)
    stacked = StackedAreaPlot(x)
    y1, y2, y3 = pylab.rand(20), pylab.rand(20)+0.5, pylab.rand(20)+2.5
    for y, col, label in zip((y1, y2, y3), ('red', 'green', 'blue'), ('first', 'second', 'third')):
        stacked.add_component(y, col, label)
    stacked.draw()
    pylab.plot(x, y1+y2+y3, 'kx--')
    stacked.legend(kwargs={'ncol':3})
    pylab.show()
    
    
        
def plot_sparse_timeseries(series, time_format, plotargs):
    times = series.times()
    timenodes = []
    values = []
    for time in times:
        timenodes.append(time - series.duration(time))
        timenodes.append(time)
        value = series[time]
        values.append(value)
        values.append(value)

    pylab.plot(timenodes, values, **plotargs)
    timeval = pylab.date2num(np.unique(timenodes))
    labels = [t.strftime(time_format) for t in timeval]
    xticks(timeval, labels)
    
def make_cross_section(data3d, x, y, z, **kwargs):
    try:
        start_point, end_point = kwargs['start_point'], kwargs['end_point']
        npoints = kwargs['npoints']
        xout = np.linspace(start_point[0], end_point[0], npoints)
        yout = np.linspace(start_point[1], end_point[1], npoints)
        return3 = True
    except KeyError:
        try:
            xout = kwargs['xout']
            yout = kwargs['yout']
            npoints = len(xout)
            return3 = False
        except KeyError:
            raise ValueError('Must define either xout and yout or start_point, end_point and npoints')
    section = np.empty((npoints, len(z)))
    for iz in range(len(z)):
        # note the flip of dimensions
        line = basemap.interp(data3d[:,:,iz].T, x, y, xout, yout)
        section[:,iz] = line
    if return3:
        return xout, yout, section
    else:
        return section

