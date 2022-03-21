'''
Created on 24.6.2020

@author: sofievm
'''

from mpl_toolkits.basemap import Basemap
from matplotlib import pyplot as plt
import numpy as np
import matplotlib as mpl
import os, io
from zipfile import ZipFile
from toolbox import supplementary as spp, MyTimeVars

def get_geo_lines_step(range_deg):
    if range_deg < 1: d = 0.1
    elif range_deg < 10: d = 1.
    elif range_deg < 40: d = 5.
    elif range_deg < 90: d = 10.
    elif range_deg < 190: d = 20.
    else: d = 30.
    return d


#==============================================================================
    
def bar_plot(ax, data, data_stdev=None, colors=None, total_width=0.8, single_width=1, 
             legend=(True, 'upper right', 7, None), group_names=None):   # if needed, location, fontsize
    """Draws a bar plot with multiple bars per data point.

    Parameters
    ----------
    ax : matplotlib.pyplot.axis
        The axis we want to draw our plot on.

    data: dictionary
        A dictionary containing the data we want to plot. Keys are the names of the
        data, the items is a list of the values.

        Example:
        data = {
            "x":[1,2,3,4],
            "y":[1,2,3,4],
            "z":[1,2,3,4],
        }
    "x" - "z" go to legend, 1-4 form 4 groups of bars, 3 bars in each group
    The group names are in group_names

    colors : array-like, optional
        A list of colors which are used for the bars. If None, the colors
        will be the standard matplotlib color cyle. (default: None)

    total_width : float, optional, default: 0.8
        The width of a bar group. 0.8 means that 80% of the x-axis is covered
        by bars and 20% will be spaces between the bars.

    single_width: float, optional, default: 1
        The relative width of a single bar within a group. 1 means the bars
        will touch eachother within a group, values less than 1 will make
        these bars thinner.

    legend: bool, optional, default: True
        If this is set to true, a legend will be added to the axis.
    """

    # Check if colors where provided, otherwhise use the default color cycle
    if colors is None:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # Number of bars per group
    n_bars = len(data)

    # The width of a single bar
    bar_width = total_width / n_bars

    # List containing handles for the drawn bars, used for the legend
    bars = []

    # Iterate over all data
    for i, (name, values) in enumerate(data.items()):
        # The offset in x direction of that bar
        x_offset = (i - n_bars / 2) * bar_width + bar_width / 2

        if data_stdev:
            # Draw a bar for every value of that type
            for x, y in enumerate(values):
                colr = colors[i % len(colors)]
                bar = ax.bar(x + x_offset, y, width=bar_width * single_width, 
                             color=colr, yerr=data_stdev[name][x],
                             error_kw={'elinewidth':0.5, 'ecolor':'black'})  # colr})
        else:
            # Draw a bar for every value of that type
            for x, y in enumerate(values):
                bar = ax.bar(x + x_offset, y, width=bar_width * single_width, 
                             color=colors[i % len(colors)])

        # Add a handle to the last drawn bar, which we'll need for the legend
        bars.append(bar[0])

    # Draw legend if we need
    if legend[0]:
        ax.legend(bars, data.keys(), loc=legend[1], fontsize=legend[2], bbox_to_anchor=legend[3])  #(1.05,1.0))

    if not group_names is None:
        ax.set_xticks(range(len(group_names)))
        ax.set_xticklabels(group_names, fontsize=10, rotation=90)

    return (bars, data.keys())


#====================================================================================

def multicolor_label(ax, list_of_strings, list_of_colors, axis='x', anchorpad=0, **kw):
    """this function creates axes labels with multiple colors
    ax specifies the axes object where the labels should be drawn
    list_of_strings is a list of all of the text items
    list_if_colors is a corresponding list of colors for the strings
    axis='x', 'y', or 'both' and specifies which label(s) should be drawn"""
    from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, HPacker, VPacker

    # x-axis label
    if axis=='x' or axis=='both':
        boxes = [TextArea(text, textprops=dict(color=color, ha='left',va='bottom',**kw)) 
                    for text,color in zip(list_of_strings,list_of_colors) ]
        xbox = HPacker(children=boxes,align="center",pad=0, sep=0)  #5)
        anchored_xbox = AnchoredOffsetbox(loc=3, child=xbox, pad=anchorpad,frameon=False,bbox_to_anchor=(0.2, -0.09),
                                          bbox_transform=ax.transAxes, borderpad=0.)
        ax.add_artist(anchored_xbox)

    # y-axis label
    if axis=='y' or axis=='both':
        boxes = [TextArea(text, textprops=dict(color=color, ha='left',va='bottom',rotation=90,**kw)) 
                     for text,color in zip(list_of_strings[::-1],list_of_colors[::-1]) ]
        ybox = VPacker(children=boxes,align="center", pad=0, sep=0) #5)
        anchored_ybox = AnchoredOffsetbox(loc=3, child=ybox, pad=anchorpad, frameon=False, bbox_to_anchor=(-0.10, 0.2), 
                                          bbox_transform=ax.transAxes, borderpad=0.)
        ax.add_artist(anchored_ybox)


#====================================================================================

def draw_map_with_points(arTitles_, lonsStat, latsStat, grid, outDir, chFNmOut, 
                         vals_points=None, vals_map=None, chUnit='', 
                         numrange=(np.nan,np.nan), ifNegatives=False, zipOut=None, cmap='jet'):
    #
    # Draws a map with points in a common color palette.
    # Can draw a multipanel picture. The 0-th dimension of vals_map then gives the pictures
    #
    if vals_map is None: ifMultiMap = False
    else: ifMultiMap = len(vals_map.shape) == 3
    # Get the proportion of the map
    latScale = np.maximum(np.minimum(np.cos(spp.degrees_2_radians * 
                                            (grid.y0 + grid.ny * grid.dy / 2)), 1), 0.3)
    # when ready to rectangular grid
#    fig_X_vs_Y = grid.nx * latScale / grid.ny
    # or square grid of cylindrical projection
    fig_X_vs_Y = 1
    # How many maps?
    if ifMultiMap:
        if len(arTitles_) != vals_map.shape[0]:
            print('list of titles has different size then values 0-th dimension:',
                  len(arTitles_), vals_map.shape)
            raise ValueError
        arTitles = arTitles_
        nxAx = np.ceil(np.sqrt(len(arTitles))).astype(np.int)
        nyAx = np.ceil(len(arTitles) / nxAx).astype(np.int)
        # Tupla of figure size
        B = np.sqrt(40. / (fig_X_vs_Y * grid.nx * grid.ny))
        figSz = (B * grid.nx * fig_X_vs_Y * nxAx, B * grid.ny * nyAx)
    else:
        # Old calls of this sub can have arTitles as a string - for drawing a single map
        if isinstance(arTitles_, list) or isinstance(arTitles_,np.ndarray): arTitles = arTitles_
        else: arTitles = [arTitles_]
        nxAx = 1
        nyAx = 1
        # Tupla of figure size
        B = np.sqrt(100. / (fig_X_vs_Y * grid.nx * grid.ny))
        figSz = (B * grid.nx * fig_X_vs_Y, B * grid.ny)
            
    # Crete th geographical objects
    bmap = Basemap(projection='cyl', resolution='i', 
                   llcrnrlon = grid.x0, 
                   urcrnrlat = max(grid.y0 + (grid.ny-1) * grid.dy,grid.y0),
                   urcrnrlon = grid.x0 + (grid.nx-1) * grid.dx, 
                   llcrnrlat = min(grid.y0 + (grid.ny-1) * grid.dy,grid.y0))
    # identify the min and max values
    if numrange: min_, max_ = numrange
    else: min_, max_ = (np.nan, np.nan)
    minlog = min_
    if min_ is None: min_= np.nan
    if max_ is None: max_= np.nan

    if np.isnan(min_): 
        if not vals_map is None: 
            min_= np.nanmin(vals_map)
            if np.any(vals_map > 0):
                minlog = np.nanmin(vals_map[vals_map > 0])
            else: minlog = 0
        if not vals_points is None: 
            if np.isfinite(min_): min_= min(min_, np.nanmin(vals_points))
            else: min_= np.nanmin(vals_points)
            if np.any(vals_map > 0):
                minlog = min(minlog, np.nanmin(vals_points[vals_points > 0]))
            else: minlog = 0
    if np.isnan(max_): 
        if not vals_map is None: max_ = np.percentile(vals_map[np.isfinite(vals_map)], 99)  # max_= np.nanmax(vals_map)
        if not vals_points is None: max_ = np.percentile(vals_points[np.isfinite(vals_points)], 99)  # max_= np.nanmax(vals_map)
    minlog = max(min_, 1e-5)
    maxlog = max(max_, 1e-5)
    if min_ < max_ / 100.: 
        if not ifNegatives: min_=0.0
    if not np.all(np.isfinite([min_, max_, minlog, maxlog])):
        print('############ min-max are not definable: min_, max_, minlog, maxlog', min_, max_,minlog, maxlog)
        return
    step = (max_ - min_) / 10.
    steplog = (np.log(maxlog) - np.log(minlog)) / 10.
    print('min_, max_,minlog, maxlog, step, steplog', min_, max_,minlog, maxlog, step, steplog)
    if step == 0.0: return 
    mpl.rc('image', cmap=cmap)
    plt.rcParams['image.cmap'] = cmap
    levsLog = np.exp(np.arange(np.log(minlog), np.log(maxlog), steplog))
    #
    # Plot in linear and logarithmic scales
    #
    for scale, norm, clevs in [('_lin', plt.Normalize(min_, max_), np.arange(min_, max_, step)),
                               ('_log', mpl.colors.LogNorm(minlog, maxlog, clip=False), levsLog)]:
        cmap = cmap
        #
        # Make individual plots: each model plus obs
        #
        # create figure and axes instances, if needed
        #
        if ifMultiMap:
            fig, axes = mpl.pyplot.subplots(nxAx, nyAx, figsize=figSz)  #(6.5 * nxAx, 6. * nyAx))
        else:
            fig, ax = mpl.pyplot.subplots(1,1, figsize=figSz)   #(11, 8.5))
            axes = [ax]
        for iPanel in range(len(arTitles)):
            ax = np.array(axes).ravel()[iPanel]
            mpl.pyplot.sca(ax)
            # draw coastlines, state and country boundaries, edge of map.
            bmap.drawcoastlines(linewidth=0.5)
            bmap.drawcountries(linewidth=0.4)
            # draw parallels and meridians
            dlo = get_geo_lines_step(grid.nx*grid.dx)
            dla = get_geo_lines_step(grid.ny*abs(grid.dy))
            lomi, loma = np.round(np.array([grid.x0, grid.x0+(grid.nx-1)*grid.dx]) / dlo) * dlo
            if grid.dy > 0:
                lami, lama = np.round(np.array([grid.y0, grid.y0+(grid.ny-1)*grid.dy]) / dla) * dla
            else:
                lami, lama = np.round(np.array([grid.y0+(grid.ny-1)*grid.dy, grid.y0]) / dla) * dla
            bmap.drawmeridians(np.arange(lomi, loma, dlo), labels=[0,0,0,1],fontsize=10)
            bmap.drawparallels(np.arange(lami, lama, dla), labels=[1,0,0,0],fontsize=10)
            if ifMultiMap:
                if not vals_map is None:
                    lons_map, lats_map = bmap.makegrid(vals_map.shape[1], vals_map.shape[2]) # get lat/lons of ny by nx evenly space grid.
                    xmap, ymap = bmap(lons_map, lats_map) # compute map proj coordinates.
                    # draw filled contours.
        #            cs = bmap.contourf(xmap, ymap, vals_map.T, clevs, clip=False, cmap=cmap, norm=norm)
                    cs = bmap.imshow(vals_map[iPanel,:,:].T, norm=norm, cmap=cmap)
                    cbar = bmap.colorbar(cs,location='bottom',pad="5%")
                if not vals_points is None:
                    # The scatter now
                    xStat,yStat = bmap(lonsStat, latsStat)
                    scat = bmap.scatter(xStat,yStat, c=vals_points[iPanel,:], s=50, linewidth=0.75,
                                        edgecolor='black', norm=norm, cmap=cmap) #vmin=0, vmax=10000)
                    cbar = mpl.pyplot.colorbar(scat,ax=ax, orientation="horizontal", pad=0.05)
            else:
                if not vals_map is None:
                    lons_map, lats_map = bmap.makegrid(vals_map.shape[0], vals_map.shape[1]) # get lat/lons of ny by nx evenly space grid.
                    xmap, ymap = bmap(lons_map, lats_map) # compute map proj coordinates.
                    # draw filled contours.
        #            cs = bmap.contourf(xmap, ymap, vals_map.T, clevs, clip=False, cmap=cmap, norm=norm)
                    cs = bmap.imshow(vals_map.T, norm=norm, cmap=cmap)
                    cbar = bmap.colorbar(cs,location='bottom',pad="5%")
                if not vals_points is None:
                    # The scatter now
                    xStat,yStat = bmap(lonsStat, latsStat)
                    scat = bmap.scatter(xStat,yStat, c=vals_points, s=50, linewidth=0.75, edgecolor='black', 
                                        norm=norm, cmap=cmap) #vmin=0, vmax=10000)
                    cbar = mpl.pyplot.colorbar(scat,ax=ax, orientation="horizontal", pad=0.05)

            cbar.set_label(chUnit)
            ax.set_title(arTitles[iPanel],fontsize=12)
        # Clean the empty space for unused panels
        for iPanel in (range(len(arTitles), nxAx * nyAx)):
            np.array(axes).ravel()[iPanel].set_axis_off()
            
        
    #    print 'Coarse picture **************************************************'
        print(os.path.join(outDir, os.path.split(chFNmOut)[-1] + scale + '.png'))
        if zipOut is None:
            plt.savefig(os.path.join(outDir, os.path.split(chFNmOut)[-1] + scale + '.png'), 
                        dpi=300) #, bbox_inches='tight')
        else:
            streamOut = io.BytesIO()
            plt.savefig(streamOut, dpi=300)
            with zipOut.open(os.path.split(chFNmOut)[-1] + scale + '.png', 'w') as pngOut:
                pngOut.write(streamOut.getbuffer())
            streamOut.close()
        plt.clf()
        plt.close()


#====================================================================================

def draw_weekly_cycle(FRP_weekdays, FRP_std_weekdays, FRP_median_weekdays, nFires, 
                      station, chLU, chTrigger, chOutDir):
    fig, ax = mpl.pyplot.subplots(1,1, figsize=(6,6))
    ax2 = ax.twinx()
    bp1 = ax.bar(np.arange(-0.3,6.7,1.0), FRP_weekdays, yerr=FRP_std_weekdays, width=0.3, 
                 label='FRE fraction', color='dodgerblue')
    bp2 = ax.bar(np.arange(0.,7.,1.0), FRP_median_weekdays, yerr=FRP_std_weekdays, width=0.3, 
                 label='FRE median', color='g')
    bp3 = ax2.bar(np.arange(0.3, 7.3, 1.), nFires, width=0.3, label='Nbr of fire cases', color='orange')
    plt.xticks(range(7), 'Mon Tue Wed Thu Fri Sat Sun'.split())
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc=3, fontsize=8)
    ax.set_title('Variability (%g,%g) %s, trigger: %s' % (station.lon, station.lat, chLU, chTrigger))
    spp.multicolor_label(ax, ['FRE sum. norm ', 'FRE median, norm'], ['dodgerblue','g'], 'y', anchorpad=-1)
#    ax.set_ylabel('FRE norm.fraction',color='dodgerblue')
    ax2.set_ylabel('Nbr of days with fires',color='orange')
    plt.savefig(os.path.join(chOutDir,'cell_weekly_cycle_%s_%s.png' % (chLU, station.code)), 
                dpi=200, bbox_inches='tight')
    plt.clf()
    plt.close()


#====================================================================================

def draw_FRP_FDL_tsMatrices(arTSM_, nFiresMin, arNames, arQuality, outDir, outFNm_templ, 
                            tStart=None, tEnd=None, ifPrintToZip=True):
    #
    # Draws time series for all locations of tsMatrices, which must be identical
    # The matrices can be already in memory or in files
    #
    # Read the files if needed
    if isinstance(arTSM_[0], str):
        arTSM = []
        for fnm in arTSM_:
            arTSM.append(MyTimeVars.TsMatrix.fromNC(fnm, tStart, tEnd))
    else:
        if tStart is not None:  itStart = np.searchsorted(arTSM_[0].times, tStart)
        else: itStart = 0 
        if tEnd is not None: itEnd = np.searchsorted(arTSM_[0].times, tEnd)
        else: itEnd = len(arTSM_[0].times)
        arTSM = arTSM_
    nTSM = len(arTSM)
    # Stupidity check:
    for its in range(1, nTSM):
        if not (np.all(arTSM[0].times == arTSM[its].times) and
                np.all(arTSM[0].stations == arTSM[its].stations)):
            print('tsMatrices are not with identical dimensions, tsm[0] and tsm[%g]' % its)
            print('Dimensions:', arTSM[0].vals.shape, arTSM[its].vals.shape)
            raise ValueError
    # ...and a bit more sophisticated
    idxFRPpositive = arTSM[0].vals[np.isfinite(arTSM[0].vals)] >= 0.0
    if np.sum(idxFRPpositive) == 0:
        print('No positive FRP values')
        return
    else:
        scales = np.ones(shape=(nTSM))
#        FRP_mean = np.nanmean((arTSM[0].vals[np.isfinite(arTSM[0].vals)])[idxFRPpositive])
#        for i in range(nTSM):
#            scales[i] = FRP_mean / np.nanmean((arTSM[2].vals[np.isfinite(arTSM[0].vals)])[idxFRPpositive])
    # colors
    cols_hist = list( ((np.mod(i,3)/3.0+0.33, np.mod(i,4)/4.0+0.25, i/(nTSM+1.0), 1) 
                       for i in range(nTSM)) )
    # couple of indices for convenience
    times_general = arTSM[0].times[itStart:itEnd]
    #
    # Drawing goes cell by cell. 
    # There will be many files, so we pack them into an archive
    #
    if ifPrintToZip: 
        zipOut = ZipFile(os.path.join(outDir,'%s_%s.zip_tmp' % (outFNm_templ,
                                                                arTSM[its].variables[0])),'w')
    # main cycle
    for icell in range(len(arTSM[0].stations)):
        idxFRP_OK = np.isfinite(arTSM[0].vals[itStart:itEnd, icell])
        # If too few fires, no reason to draw anything
        if np.sum(arTSM[0].vals[idxFRP_OK,icell]>0) < nFiresMin: 
            print('Too few fires for drawing (%gE,%gN), %i' % (arTSM[0].stations[icell].lon,
                  arTSM[0].stations[icell].lat, np.sum(arTSM[0].vals[idxFRP_OK,icell]>0)))
            continue
        arR = []
        fig = mpl.pyplot.figure(figsize=(12,12))   #subplots(1,1, figsize=(12,6))
        gs = fig.add_gridspec(2,2)
        #
        # The time series plot: upper half of the page
        #
        ax = fig.add_subplot(gs[0,:])
        axTW = ax.twinx()
        # First, plot the index predictions
        FDImax = 0.0
        for its in range(1, nTSM):
            chTitle = '%s (%gE,%gN)' % (outFNm_templ, arTSM[its].stations[icell].lon,
                                      arTSM[its].stations[icell].lat)
#            chTitle = '%s (%gN,%gE)' % (arTSM[its].variables[0], arTSM[its].stations[icell].lon,
#                                      arTSM[its].stations[icell].lat)
            axTW.plot_date(times_general, arTSM[its].vals[:, icell] * scales[its], 
                           label=arNames[its], linestyle='-', linewidth=1, marker='')
            # make the array to correlate
#            arR.append(spp.nanCorrCoef(np.maximum(arTSM[0].vals[idxFRP_OK, icell], 0.0), 
#                                       np.where(np.logical_and(arTSM[0].vals[idxFRP_OK, icell] < 0, 
#                                                               arTSM[its].vals[idxFRP_OK, icell] * 
#                                                               scales[its] < 
#                                                               -arTSM[0].vals[idxFRP_OK, icell]), 
#                                                0.0, arTSM[its].vals[idxFRP_OK, icell])))
            FDImax = max(np.max(arTSM[its].vals[:, icell] * scales[its]), FDImax)
            axTW.set_ylabel(chTitle.split('(')[0].replace('_cmp_',''))
        #
        # FRP is to be plotted twice: actually observed FRP as red and detection limit as blue
        times = times_general[idxFRP_OK]
        ifPos = arTSM[0].vals[idxFRP_OK, icell] > 0
        ifNeg = arTSM[0].vals[idxFRP_OK, icell] < 0
        ax.plot_date(times[ifPos], (arTSM[0].vals[idxFRP_OK, icell])[ifPos], 
                     label=arNames[0], linestyle='', marker='o', markersize=4, color='r')
        ax.plot_date(times[ifNeg], -(arTSM[0].vals[idxFRP_OK, icell])[ifNeg], 
                     label=arNames[0] + ' detect.lim', linestyle='', marker='.', 
                     markersize=1, color='b')
        # Finalise the pictures
        ax.set_ylim(-1, 1.1*np.max(arTSM[0].vals[idxFRP_OK, icell]))
        axTW.set_ylim(-0.01*FDImax, FDImax * 1.1)
        ax.legend(fontsize=10, loc=2)
        axTW.legend(fontsize=10, loc=1)
        ax.set_ylabel('FRP, DL, [MW]')
        ax.set_title('%s, correl to %s: %s' % (chTitle, arTSM[0].variables[0], 
                                               '  '.join(list(('%4.3f' % r for r in arQuality)))))
        #
        # Now, two scatterplots
        # Limits should be the same for both plots
        xlim = np.max(arTSM[0].vals[idxFRP_OK, icell])
        ylim = np.max(np.array(list((np.max(arTSM[its].vals[idxFRP_OK, icell]) for its in [1,2]))))
        if xlim < 3 * ylim and xlim > ylim / 3:
            xlim = max(xlim,ylim)
            ylim=xlim
        # Draw
        for its in [1,2]:
            ax2 = fig.add_subplot(gs[1,its-1])
            # actual FRP vs prediction
            ax2.scatter((arTSM[0].vals[idxFRP_OK, icell])[ifPos],
                        (arTSM[its].vals[idxFRP_OK, icell])[ifPos] * scales[its],
                        marker='o', color='r',label='FRP obs>0 vs predict')
            # Observations below detection can correspond to predictions
            # below and above detection, i.e. still-good and bad predictions
            idxLowLow = np.logical_and(ifNeg, 
                                       arTSM[its].vals[idxFRP_OK, icell] * scales[its] < 
                                       -arTSM[0].vals[idxFRP_OK, icell])
            idxLowHigh = np.logical_and(ifNeg, 
                                       arTSM[its].vals[idxFRP_OK, icell] * scales[its] >= 
                                       -arTSM[0].vals[idxFRP_OK, icell])
            # below-detection FRP vs above-detection prediction
            ax2.scatter(-(arTSM[0].vals[idxFRP_OK, icell])[idxLowHigh],
                        (arTSM[its].vals[idxFRP_OK, icell])[idxLowHigh] * scales[its],
                        marker='.', s=2, color='b', label='FRP obs=0 vs predict > det.lim')
            # below-detection FRP vs below-detection prediction
            # Both are zero
            ax2.scatter(np.maximum(0.,arTSM[0].vals[idxFRP_OK, icell])[idxLowLow],
                        (arTSM[its].vals[idxFRP_OK, icell])[idxLowLow] * scales[its],
                        marker='.', s=3, color='limegreen', label='FRP obs=0 vs predict < det.lim')
            ax2.set_xlabel('Observed FRP, MW')
            ax2.set_ylabel(arNames[its])
            ax2.set_xlim(-0.01*xlim, 1.01*xlim)
            ax2.set_ylim(-0.01*ylim, 1.01*ylim)
            ax2.set_title(arNames[its])
            ax2.legend(loc=4)

        print(os.path.join(outDir,'%s_%s_%s.png' % (outFNm_templ,arTSM[its].variables[0], 
                                                          arTSM[its].stations[icell].code)))
        if ifPrintToZip:
            streamOut = io.BytesIO()
            plt.savefig(streamOut, dpi=300)
            with zipOut.open('%s_%s_%s.png' % (outFNm_templ,arTSM[its].variables[0], 
                                               arTSM[its].stations[icell].code), 'w') as pngOut:
                pngOut.write(streamOut.getbuffer())
            streamOut.close()
        else:
            plt.savefig(os.path.join(outDir,'%s_%s_%s.png' % (outFNm_templ,arTSM[its].variables[0], 
                                                              arTSM[its].stations[icell].code)), 
                                                              dpi=300)
        plt.clf()
        plt.close()
    
    if ifPrintToZip:
        chTmp = zipOut.filename
        nameFinal = zipOut.filename.replace('_tmp','')    # final name for archive
        if os.path.exists(nameFinal):          # already exists?
            if os.path.exists(nameFinal + '_prev'): os.remove(nameFinal + '_prev')  # ... previous one?
            os.rename(nameFinal, nameFinal + '_prev')   # existing to backup
        zipOut.close()   # Accurate closure of the zip file
        os.rename(chTmp, nameFinal)   # final name for the zip


#====================================================================================

def draw_tser_FRP_prediction_tsM(arTSM_, arNames, outDir, arTitle, metadata, tStart=None, tEnd=None, 
                                 ifNormalise=True, ifPrintToZip=True, ifTextDump=False, 
                                 ifSecondAxis=False):
    #
    # Draws time series for all locations of tsMatrices, which must be identical
    # The matrices can be already in memory or in files
    #
    # Read the files if needed
    if isinstance(arTSM_[0], str):
        arTSM = []
        for fnm in arTSM_:
            arTSM.append(MyTimeVars.TsMatrix.fromNC(fnm, tStart, tEnd))
    else:
        arTSM = arTSM_
    nTSM = len(arTSM)
    cols_hist = list( ((np.mod(i,3)/3.0+0.33, np.mod(i,4)/4.0+0.25, i/(nTSM+1.0), 1) 
                       for i in range(nTSM)) )
    # Stupidity check
    for its in range(1, nTSM):
        if not (  #np.all(arTSM[0].times == arTSM[its].times) and
                np.all(arTSM[0].stations == arTSM[its].stations)):
            print('tsMatrices are not with identical dimensions, tsm[0] and tsm[%g]' % its)
            print('Dimensions:', arTSM[0].vals.shape, arTSM[its].vals.shape)
            raise ValueError
    #
    # Selecting the time period
    if tStart is None: 
        itStartObs = 0
        itStartMdl = 0
    else: 
        itStartObs = np.searchsorted(arTSM[0].times, tStart)
        itStartMdl = np.searchsorted(arTSM[1].times, tStart)
    if tEnd is None: 
        itEndObs = len(arTSM[0].times)
        itEndMdl = len(arTSM[1].times)
    else: 
        itEndObs = np.searchsorted(arTSM[0].times, tEnd)
        itEndMdl = np.searchsorted(arTSM[1].times, tEnd)
    timesObs = arTSM[0].times[itStartObs:itEndObs]
    timesMdl = arTSM[1].times[itStartMdl:itEndMdl]
    #
    # For comparison, have to get the common time interval
    timesCommon = sorted(list(set(timesObs).intersection(set(timesMdl))))
    idxTimesCommon_obs = np.searchsorted(arTSM[0].times, timesCommon)
    idxTimesCommon_mdl = np.searchsorted(arTSM[1].times, timesCommon)
    # 
    # Drawing goes cell by cell
    #
    if not ifNormalise: fNorm = 1.0
    rOut = []
    nFiresOut = []
    if ifPrintToZip: 
        # Careful: if program stopped before closing zip it will be unreadable
        # use zip_tmp to mark that
        zipOut = ZipFile(os.path.join(outDir,'%s_%s.zip_tmp' % (arTSM[its].variables[0],
                                                                '_'.join(arTitle))),'w')
    for icell in range(len(arTSM[0].stations)):
        vMax = -1.
        arR = []
        arN = []
        fig = mpl.pyplot.figure(constrained_layout=True, figsize=(20,9))
        gs = fig.add_gridspec(10,1)
        ax0 = fig.add_subplot(gs[:9,0])
        if ifSecondAxis: ax1 = ax0.twinx()
        else: ax1 = ax0
#        fig, ax0 = mpl.pyplot.subplots(1,1, figsize=(20,8))
        keyCellMetadata = arTSM[0].stations[icell].code
        if not keyCellMetadata in metadata.keys(): keyCellMetadata = 'rest'

#        ax1 = ax0.twinx()
        # draw the observations: first, positive FRP, then negative detection limit
        # positive: FRP
        idxOK = np.isfinite(arTSM[0].vals[itStartObs:itEndObs, icell])
        idxPos = arTSM[0].vals[itStartObs:itEndObs, icell][idxOK] >= 0
        for its in range(1, nTSM):
            arR.append(spp.nanCorrCoef(arTSM[0].vals[idxTimesCommon_obs, icell], 
                                        arTSM[its].vals[idxTimesCommon_mdl, icell]))
            arN.append(np.sum(arTSM[0].vals[idxTimesCommon_obs, icell]
                       [np.isfinite(arTSM[0].vals[idxTimesCommon_obs, icell])] > 0))
            if ifNormalise:
                fNorm = np.nanmax(np.abs(arTSM[its].vals[itStartMdl:itEndMdl, icell]))
                if fNorm == 0: fNorm = 1
            ax1.plot_date(timesMdl, arTSM[its].vals[itStartMdl:itEndMdl, icell] / fNorm, 
                          color='darkgoldenrod', label=arNames[its], linestyle='-', 
                          linewidth=1, marker='')
            vMax = np.maximum(vMax, np.max(arTSM[its].vals[itStartObs:itEndObs, icell] / fNorm))
        #
        # Now, the observations
        try: vMax_0 = np.max(arTSM[0].vals[itStartObs:itEndObs, icell][idxOK][idxPos])
        except: vMax_0 = 0
        if ifNormalise: fNorm = vMax_0
        if fNorm == 0: fNorm = 1 
        ax0.plot_date(timesObs[idxOK][idxPos], 
                      arTSM[0].vals[itStartObs:itEndObs, icell][idxOK][idxPos] / fNorm, 
                      color='r', label=arNames[0] + ' FRP', linestyle='', linewidth=0, marker='.', 
                      markersize=5)
        # negative: detection limit
        idxNeg = arTSM[0].vals[itStartObs:itEndObs, icell][idxOK] < 0
        ax0.plot_date(timesObs[idxOK][idxNeg], 
                      -arTSM[0].vals[itStartObs:itEndObs, icell][idxOK][idxNeg] / fNorm, 
                      color='b', label=arNames[0] + ' Detection Limit', linestyle='', linewidth=0, 
                      marker='.', markersize=1)
        # General stuff
        rOut.append(arR)
        nFiresOut.append(arN)
        ax0.legend(fontsize=10, loc=2)
        ax0.set_ylabel(arTSM[0].variables[0] + ' , ' + arTSM[0].units[0], fontsize=12)
        if ifSecondAxis:
            ax1.set_ylim(0.0, np.max(arTSM[1].vals[itStartObs:itEndObs, icell] / fNorm))
            ax1.set_ylabel('Predicted ' + arTSM[0].variables[0])
            ax1.yaxis.label.set_color('darkgoldenrod')
            ax1.tick_params(axis='y', colors='darkgoldenrod')
        vMax = np.maximum(vMax, vMax_0 / fNorm)
        ax0.set_ylim(-0.01*vMax, vMax * 1.01)
#        ax0.set_xticklabels(fontsize=12)
#        ax0.set_yticklabels(fontsize=12)
        ax0.set_title('%s, %s, (%gE,%gN), %s, r=%s' % (arTSM[its].variables[0], 
                                                         arTSM[its].stations[icell].code,
                                                         arTSM[its].stations[icell].lon, 
                                                         arTSM[its].stations[icell].lat, 
                                                         '  '.join(arTitle), 
                                                         '  '.join(list(('%4.3f' % r for r in arR)))),
                      fontsize=14)
        #
        # The grid cell metadata
        #
        ax_txt = fig.add_subplot(gs[9,0])
        # First, basic parameters oft eh run, excluding those already in the title
        ax_txt.text(0, 1, metadata['general_setup'])
        jPos = 0.5
        iPos = 0
        # one by one, parameters
        keys = metadata[keyCellMetadata].keys()
        for iKey, key in enumerate(keys):
            if key == 'general_setup' : continue
            ax_txt.text(iPos, jPos, '%s:  %g' % (key, metadata[keyCellMetadata][key]))
            jPos -= 0.27
            if jPos < 0: 
                iPos += 0.2
                jPos = 1
            
            
        ax_txt.spines["top"].set_visible(False)
        ax_txt.spines["right"].set_visible(False)
        ax_txt.spines["left"].set_visible(False)
        ax_txt.spines["bottom"].set_visible(False)
        ax_txt.set_xticklabels('')
        ax_txt.set_yticklabels('')
        ax_txt.tick_params(axis='x',which='both',bottom=False,top=False,labelbottom=False)
        ax_txt.tick_params(axis='y',which='both',left=False,right=False,labelbottom=False)
#        mpl.pyplot.show()
       
        #
        # Store the image
        #
        if ifPrintToZip:
            streamOut = io.BytesIO()
            plt.savefig(streamOut, dpi=200, bbox_inches='tight')
            with zipOut.open('%s_%s_%s.png' % (arTSM[its].variables[0], '_'.join(arTitle),
                                               arTSM[its].stations[icell].code), 'w') as pngOut:
                pngOut.write(streamOut.getbuffer())
            streamOut.close()
            # We can also drop the time series to text dump
            if ifTextDump:
                txtIO = io.StringIO()
                txtIO.write('# YYYYMMDD MODIS_FRP_or_DL_MW FFM_predictions_MW\n')
                for iTime, tTime in enumerate(timesCommon):
                    txtIO.write(tTime.strftime('%Y%m%d ') + 
                                ('%g ' % arTSM[0].vals[idxTimesCommon_obs[iTime], icell]) +
                                ' '.join(list(('%g' % arTSM[i].vals[idxTimesCommon_mdl[iTime], icell]
                                                for i in range(1, nTSM)))) + '\n')  #.encode('utf-8'))
                with zipOut.open('%s_%s_%s.txt' % (arTSM[its].variables[0], '_'.join(arTitle),
                                                   arTSM[its].stations[icell].code), 'w') as txtOut:
                    txtOut.write(txtIO.getvalue().encode('utf-8'))
                txtOut.close()
                
        else:
            plt.savefig(os.path.join(outDir, '%s_%s_%s.png' % (arTSM[its].variables[0], 
                                                               '_'.join(arTitle),
                                                               arTSM[its].stations[icell].code)), 
                                                               dpi=200, bbox_inches='tight')
        plt.clf()
        plt.close()

    if ifPrintToZip:
        chTmp = zipOut.filename
        nameFinal = zipOut.filename.replace('_tmp','')    # final name for archive
        if os.path.exists(nameFinal):          # already exists?
            if os.path.exists(nameFinal + '_prev'): os.remove(nameFinal + '_prev')  # ... previous one?
            os.rename(nameFinal, nameFinal + '_prev')   # existing to backup
        zipOut.close()  # temporatry file
        os.rename(chTmp, nameFinal) # rename to actual zip

    return (rOut,nFiresOut)


#====================================================================================

def draw_tser_two_tsMatrices(arTSM_, arNames, outDir, outFNm_templ, 
                             tStart=None, tEnd=None, ifNormalise=True, ifSameRank=False):
    #
    # Draws time series for all locations of tsMatrices, which must be identical
    # The matrices can be already in memory or in files
    #
    # Read the files if needed
    if isinstance(arTSM_[0], str):
        arTSM = []
        for fnm in arTSM_:
            arTSM.append(MyTimeVars.TsMatrix.fromNC(fnm, tStart, tEnd))
    else:
        arTSM = arTSM_
    nTSM = len(arTSM)
    cols_hist = list( ((np.mod(i,3)/3.0+0.33, np.mod(i,4)/4.0+0.25, i/(nTSM+1.0), 1) 
                       for i in range(nTSM)) )
    # Stupidity check
    for its in range(1, nTSM):
        if not (np.all(arTSM[0].times == arTSM[its].times) and
                np.all(arTSM[0].stations == arTSM[its].stations)):
            print('tsMatrices are not with identical dimensions, tsm[0] and tsm[%g]' % its)
            print('Dimensions:', arTSM[0].vals.shape, arTSM[its].vals.shape)
            raise ValueError
    #
    # Selecting the time period
    times = arTSM[0].times
    if tStart is not None: 
        itStart = np.searchsorted(arTSM[0].times, tStart)
    else:
        itStart = 0 
    if tEnd is not None: 
        itEnd = np.searchsorted(arTSM[0].times, tEnd)
    else:
        itEnd = len(arTSM[0].times)
    if itStart >= itEnd:
        raise ValueError('Start and end indices conflict:' + 
                         str(itStart) + ',' + str(itEnd))
    times = times[itStart:itEnd]
    colors = ['r','b','g','y','m','gold','firebrick']
    
    # 
    # Drawing goes cell by cell
    #
    if not ifNormalise: fNorm = 1.0
    for icell in range(len(arTSM[0].stations)):
        arR = []
        fig, ax0 = mpl.pyplot.subplots(1,1, figsize=(10,5))
        if nTSM > 1: ax1 = ax0.twinx()
        else: ax1 = None
        for its in range(nTSM):
            color=colors[its]
            if its == 0: 
                ax = ax0
                if ifSameRank:     # if the first time series is of the same type as others
                    chMarker = ''
                    linewidth=2
                    linestyle = '-'
                else:
                    chMarker = 'o'
                    linewidth=0
                    linestyle = ''
            else: 
                chMarker = ''
                ax = ax1
                linewidth = 2
                linestyle='-'
            if len(arTSM[its].variables) > 1:
                for ivar in range(len(arTSM[its].variables)):
                    chTitle = '%s_%s' % (arTSM[its].variables[ivar], arTSM[its].stations[icell].code)
                    if its>0: arR.append(spp.nanCorrCoef(arTSM[0].vals[itStart:itEnd, icell, ivar], 
                                                         arTSM[its].vals[itStart:itEnd, icell, ivar]))
                    if ifNormalise: 
                        fNorm = np.max(np.abs(arTSM[its].vals[itStart:itEnd, icell, ivar]))
                        if fNorm == 0: fNorm = 1 
                    ax.plot_date(times, arTSM[its].vals[itStart:itEnd, icell, ivar] / fNorm, 
                                 label=arNames[its], linestyle='-', marker=chMarker)
            else:
                chTitle = '%s_%s' % (arTSM[its].variables[0], arTSM[its].stations[icell].code) 
                if its>0: arR.append(spp.nanCorrCoef(arTSM[0].vals[itStart:itEnd, icell], 
                                                     arTSM[its].vals[itStart:itEnd, icell]))
                if ifNormalise: 
                    fNorm = np.nanmax(np.abs(arTSM[its].vals[itStart:itEnd, icell]))
                    if fNorm == 0: fNorm = 1
                ax.plot_date(times, arTSM[its].vals[itStart:itEnd, icell] / fNorm, color=color,
                             label=arNames[its], linestyle=linestyle, linewidth=linewidth,
                             marker=chMarker)
        if nTSM == 1: ax0.legend(fontsize=10) #, loc=1)
        else:
            lines0, labels0 = ax0.get_legend_handles_labels()
            lines1, labels1 = ax1.get_legend_handles_labels()
            ax0.legend(lines0 + lines1, labels0 + labels1, fontsize=10)
        ax0.set_ylabel(arNames[0], color='r')
        if nTSM == 1:
            ax0.set_title(chTitle, fontsize=14)
        else:
            ax1.set_ylabel(' '.join(labels1), color=colors[1])
#            multicolor_label(ax1, arNames[1:], colors[1:len(arNames)], axis='y')
            ax0.set_title('%s, correl to %s: %s' % (chTitle, arTSM[0].variables[0], 
                                                          '  '.join(list(('%4.3f' % r for r in arR)))))
        plt.savefig(os.path.join(outDir,outFNm_templ + '_' + chTitle + '.png'), dpi=300)
        plt.clf()
        plt.close()


#====================================================================================

def plot_overall_scores_barchart(skills, runNms, chFNmTempl):
    #
    # Draws a bar chart with skill scores for the optimiser
    #
    runs = list(skills.keys())
    runs.remove('percentiles')
    grids = list(skills[runs[0]].keys())
    LUs = sorted(list(skills[runs[0]][grids[0]].keys()))
    FDIs = list(skills[runs[0]][grids[0]][LUs[0]].keys())
    #
    # Each cost function makes own set of runs, scores between them not comparable
    # Each grid can be also plotted separately - but in the same chart
    # We shall use the bar_plot of supplementary module
    #
    for run in runs:
        fig, axes = mpl.pyplot.subplots(len(grids),1, figsize=(10, 6*len(grids)))
        if len(grids) == 1: axes = [axes]
        for iax, ax in enumerate(axes):
            data = {}
            for LU in LUs:
                try: data[LU] = list((-skills[run][grids[iax]][LU][FDI] for FDI in FDIs))
                except: data[LU] = [np.nan] * len(FDIs)
            spp.bar_plot(ax, data, group_names = FDIs) #, data_stdev=None, colors=None, total_width=0.8, single_width=1, 
                                   # legend=(True, None, 7)):   # if needed, location, fontsize
            ax.set_title('Run %s, %s, grid %s '  % (run, runNms[run], grids[iax]))
            ax.set_ylim(0, -skills['percentiles'][1] * 1.1)    # 99-th percentile: allow for an outlier
        plt.savefig(chFNmTempl + '_' + run + '_' + runNms[run] + '.png', dpi=200, bbox_inches='tight')
        plt.clf()
        plt.close()

# More options to get tight lyout
#
#plt.gca().set_axis_off()
#plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
#plt.margins(0,0)
#plt.gca().xaxis.set_major_locator(plt.NullLocator())
#plt.gca().yaxis.set_major_locator(plt.NullLocator())
#plt.savefig("filename.pdf", bbox_inches = 'tight', pad_inches = 0)
#

#====================================================================================

def plot_overall_scores_barchart_commonMetric(dicSkills, costNms, chTitle, outFNm):
    #
    # Draws a bar chart with skill scores for the optimiser. 
    # skills are: skills_all[k][FDI_type], where k is the run description
    #
    grids = list(dicSkills.keys())
    runs = list(dicSkills[grids[0]].keys())     # e.g. r07_a1_bF1_bD_1
    runID = list(('%s\n%s' % (run, costNms[run][0]) for run in runs))
    #
    # Each cost function makes own set of runs, scores between them not comparable
    # Each grid can be also plotted separately - but in the same chart
    # We shall use the bar_plot of supplementary module
    #
    data = {}
    dataStd = {}
    fig, ax = mpl.pyplot.subplots(1,1, figsize=(12,6))
    for grd in grids:
        # A list of skills of the given run cost fiunction and FDIs
        # For the mean value, few-fire case must be suppressed, use weightted mean
#        data[grd] = list((np.mean(dicSkills[grd][run]) for run in runs))
#        dataStd[grd] = list((np.std(dicSkills[grd][run]) for run in runs))
        data[grd] = list((spp.weighted_mean(np.array(dicSkills[grd][run][0]), 
                                            np.array(dicSkills[grd][run][1])) for run in runs))
        dataStd[grd] = list((spp.weighted_std(np.array(dicSkills[grd][run][0]),
                                              np.array(dicSkills[grd][run][1])) for run in runs))
    spp.bar_plot(ax, data, group_names = runID, data_stdev=dataStd) #, colors=None, total_width=0.8, single_width=1, 
                                   # legend=(True, None, 7)):   # if needed, location, fontsize
    ax.set_title(chTitle)
    ax.set_ylim(-0.1, 0.5)
    ax.grid(True, axis='y', color='lightgrey')
    plt.savefig(outFNm, dpi=200, bbox_inches='tight')
    plt.clf()
    plt.close()


#====================================================================================

def draw_quantile_plot(percentiles, prcObs, prcMdl, prcObs_fit, prcMdl_fit, prcMdl_scaled, 
                       chCellTitle, dicDumpZip, outDir, outFNm):
    #
    # Quantile plot
    #
    fig = mpl.pyplot.figure(constrained_layout=True, figsize=(9,10))
    gs = fig.add_gridspec(3, 20)
    ax1 = fig.add_subplot(gs[0:2, 1:19])
    
#    fig, ax = mpl.pyplot.subplots(1,1, figsize=(6,7))
    chart = ax1.scatter(prcObs, prcMdl, marker='o', label='obs-mdl', c=range(len(prcObs)), 
                       cmap=mpl.cm.get_cmap('jet'), s=100)
    chart2 = ax1.scatter(prcObs_fit, prcMdl_fit, marker='.', c='black', label='fitObs-fitMdl')
    ax2 = ax1.twinx()
#    chart3 = ax1.scatter(prcObs, prcMdl_scaled, marker='x', c='green', label='obs-mdl_scaled')
    chart3 = ax2.plot(prcObs, prcMdl_scaled, marker='.', linestyle='-', color='green', label='obs-mdl_scaled')
    ax1.set_title(chCellTitle)
    ax1.set_xlabel('Obs; obs_fit percentile')
    ax1.set_ylabel('mdl percentile')
    ax2.set_ylabel('scaled_Mdl percentile', color='green')
    ax1.legend(loc=4)
    ax2.legend(loc=5)
    cbar = fig.colorbar(chart, orientation='vertical', cax = fig.add_subplot(gs[0:2,0]), 
                        ticks=range(0,1001,100))
    cbar.ax.set_yticklabels(range(0,101,10))
    
    ax3 = fig.add_subplot(gs[2, 1:10])
    chart4 = ax3.scatter(percentiles, prcObs, marker='o', label='Observed')
    chart5 = ax3.scatter(percentiles, prcObs_fit, marker='.', label='Observed_fit')
    ax3.set_title('Percentiles Observed')
    ax3.legend()
    
    ax4 = fig.add_subplot(gs[2, 10:19])
    char6 = ax4.scatter(percentiles, prcMdl, marker='o', label='Predicted')
    chart7 = ax4.scatter(percentiles, prcMdl_fit, marker='.', label='Predicted_fit')
    ax4.set_title('Percentiles Modelled')
    ax4.legend()
    # Store figure
    if dicDumpZip is None:
        mpl.pyplot.savefig(os.path.join(outDir, outFNm))
    else:
        if not 'quantile' in dicDumpZip.keys():
            dicDumpZip['quantile'] = ZipFile(os.path.join(outDir, outFNm.replace('.png','_et_al.zip_tmp')),
                                             'w')
        streamOut = io.BytesIO()
        mpl.pyplot.savefig(streamOut, dpi=300)
        with dicDumpZip['quantile'].open(outFNm, 'w') as pngOut: 
            pngOut.write(streamOut.getbuffer())
        streamOut.close()
    mpl.pyplot.clf()
    mpl.pyplot.close()

    
#if __name__ == '__main__':
#
#        fig = mpl.pyplot.figure(constrained_layout=True, figsize=(2,4))
#        gs = fig.add_gridspec(10,1)
#        ax0 = fig.add_subplot(gs[:9,0])
#        ax_txt = fig.add_subplot(gs[9,0])
#        ax_txt.text(0,0,'try')
#        ax_txt.spines["top"].set_visible(False)
#        ax_txt.spines["right"].set_visible(False)
#        ax_txt.spines["left"].set_visible(False)
#        ax_txt.spines["bottom"].set_visible(False)
#        ax_txt.set_xticklabels('')
#        ax_txt.set_yticklabels('')
#        ax_txt.tick_params(axis='x',which='both',bottom=False,top=False,labelbottom=False)
#        ax_txt.tick_params(axis='y',which='both',left=False,right=False,labelbottom=False)
#        mpl.pyplot.show()
        
    
