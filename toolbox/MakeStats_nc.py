#!/usr/bin/env python

import matplotlib
matplotlib.use('Agg')
#from opertools import tsdb
from mpl_toolkits.basemap import Basemap

from toolbox import timeseries as ts

import datetime as dt
import numpy as np
from toolbox import MyTimeVars
import matplotlib.pyplot as plt
import pickle
import gzip
import sys
import os
import stations_opev
import figs


try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    mpisize = comm.size
    mpirank = comm.Get_rank()
except:
    mpisize = 1
    mpirank = 0
    comm = None
    pass

flipBias=False ### One obs, many measurements

#mods="NODAlr-fit2-1h-ntp NODAlr-op55-ntp NODAlr-fit2clf-1h-ntp NODAlr-fit2clg-1h-ntp".split(); 
#outdir="Stats"
#mods="NODAlr-ECKZ-ntp NODAlr-fit2bis-1h-ntp NODAlr-fitTV15a-ntp NODAlr-opHyf-ntp NODAlr-ERA5-ntp NODAlr-fit2c-1h-ntp NODAlr-fitTV15b-ntp NODAlr-opHyh-ntp NODAlr-afBVOC-1h-ntp NODAlr-fit2cld-1h-ntp NODAlr-fitTV15c-ntp NODAlr-opN10a-ntp NODAlr-afBVOC01-1h-ntp NODAlr-fit2clf-1h-ntp NODAlr-fix15FR-1h-ntp NODAlr-opN20-ntp NODAlr-fit2-1h-ntp NODAlr-fit2clg-1h-ntp NODAlr-fix15IER-1h-ntp NODAlr-opN50a-ntp NODAlr-fit2-ntp NODAlr-fit2clh-1h-ntp NODAlr-fix15NL-1h-ntp NODAlr-opN50b-ntp NODAlr-fit2GFM-1h-ntp NODAlr-fit2d-1h-ntp NODAlr-fix15PT-1h-ntp NODAlr-opUst-ntp NODAlr-fit2GFMa-1h-ntp NODAlr-fit2hino2-1h-ntp NODAlr-op55-ntp NODAlr-oper-ntp NODAlr-fit2a-1h-ntp NODAlr-fit2nph-1h-ntp NODAlr-op55n-ntp NODAlr-fit2b-1h-ntp NODAlr-fitTV15-ntp NODAlr-opHyb-ntp".split()
#mods="lr-op56-lotol-ntp lr-op56ECKZ-ntp lr-op56hr-lotol-ntp lr-op56-ntp lr-op56-off2d-ntp lr-op56SilKZ-ntp".split()

outdir="Stats"
mods=sys.argv[1:]




species="NO2 O3 PM10 PM25 SO2 OX NOX CO NO".split()
#species="CO".split()
#species="NO2 O3 PM10 PM25 SO2 OX NOX".split()
#species_noObs="g_stomatal Kz_1m BLH lai".split()



fig = plt.figure(figsize=(10,7)) 
# Put all the data to a big matrix



units="EU ug/m3"

spprev=""
cnt = 0

mapBoxes= {"CO":dict(llcrnrlon = -10, llcrnrlat = 34 , urcrnrlon = 30, urcrnrlat = 60),
           "NO2":dict(llcrnrlon = -19, llcrnrlat = 35 , urcrnrlon = 32, urcrnrlat = 68),
           }
statlimits = {
              ('NO2', 'Bias') : (-20, 20),
              ('NO2', 'RMS') : (0,20),
              ('NOX', 'Bias') : (-20, 20),
              ('NOX', 'RMS') : (0,20),
              ('NO', 'Bias') : (-20, 20),
              ('NO', 'RMS') : (0,20),
              ('CO', 'Bias') : (-500, 500),
              ('CO', 'RMS') : (0, 500),
              ('O3', 'Bias') : (-50, 50),
              ('O3', 'RMS') : (0, 50),
              ('OX', 'Bias') : (-50, 50),
              ('OX', 'RMS') : (0, 50),
              ('PM25', 'Bias') : (-20, 20),
              ('PM25', 'RMS') : (0, 20),
              ('PM10', 'Bias') : (-20, 20),
              ('PM10', 'RMS') : (0, 20),
              ('SO2', 'Bias') : (-10, 10),
              ('SO2', 'RMS') : (0, 20),
              }
def limits_for_statistic(statistic, species):
    if statistic == "Corr":
        return (0,1)
    key = (species, statistic)
    if key in statlimits:
        return statlimits[key]
    else:
        print statistic, species
        raise ValueError("Unknown norm for statistics")



statlist="Bias RMS N Corr meanObs meanMod stdObs stdMod MORatio MOstdRatio".split()
for  sp in species:

      input_dir="../OBS_INPUT"
      stlistvali=stations_opev.getStationSet(input_dir, sp, "validation")
      stlistassi=stations_opev.getStationSet(input_dir, sp, "assimilation")

      for mod in mods:


          cnt += 1
          if (cnt % mpisize != mpirank):
              continue

          if sp != spprev:
              ncfile="../TimeVars/%s/%s_%s.nc"%("obs","obs",sp,)
              print "Getting obs from ", ncfile
              try:
                obsMatr = MyTimeVars.TsMatrix.fromNC(ncfile)
              except IOError:
                break
              spprev = sp
          ncfile="../TimeVars/%s/%s_%s.nc"%(mod,mod,sp,)
          if not os.path.exists(ncfile):
            continue
          print "Getting mod from ", ncfile
          case=mod
          modMatr = MyTimeVars.TsMatrix.fromNC(ncfile)

          stations = obsMatr.stations
          hrlist  =  obsMatr.times

          seasonalStats = MyTimeVars.SeasonalStats(obsMatr, modMatr, mod)
          statdir="%s/%s"%(outdir,mod)
          os.system("mkdir -p %s"%(statdir,))

          statlogname ="%s/Stats_%s_%s.log"%(statdir,sp,mod,)
          with open(statlogname,'w') as statfile:

            for season in seasonalStats.seasons:
              ifHead = True
              stat = seasonalStats.stats[season]
              for area in "all ruralbg opstations assi vali".split():
                  stidxSelected=[]
                  for ist,st in enumerate(stations):
                        if area == "all":
                             stidxSelected.append(ist)
                        elif area == "assi":
                            if st.code in stlistassi:
                               stidxSelected.append(ist)
                        elif area == "vali":
                            if st.code in stlistvali:
                               stidxSelected.append(ist)
                        elif area == "opstations":
                          if st.code in stations_opev.lst:
                             stidxSelected.append(ist)
                        elif area == "ruralbg":
                          if (st.source.startswith("background") and st.area.startswith('rural')):
                             stidxSelected.append(ist)

                  for statproc in "mean median prc95".split():
                      stat.printStats(sys.stdout, statlist,  statproc, area, ifHead, stidxSelected)
                      stat.printStats(statfile, statlist,  statproc, area, ifHead, stidxSelected)
                      ifHead=False

                  ##
                  ## Make picture
                  for stname in "RMS Bias Corr".split():
                        ax = fig.add_axes([0.05,0.15,0.9,0.8])
                        axcb = fig.add_axes([0.2,0.05,0.6,0.05])

                        bmapargs=dict(projection='cyl', resolution='l', ax=ax)
                        bmapargs.update(mapBoxes['NO2'])

                        basemap = Basemap(**bmapargs)

                        norm = matplotlib.colors.Normalize(*limits_for_statistic(stname, sp))
                        sct = stat.plot2Map(basemap,  stname, 'jet', norm, stidxSelected)
                        plt.colorbar(sct,cax=axcb, extend='both', format="%g", orientation='horizontal')

                        #pylab.colorbar(mappable=sct, orientation='horizontal')
                        ax.set_title('%s %s, model: %s, season: %s, stations:%s' % (stname, sp, mod, season, area))
                        figs.finishMap(basemap, dx=10, dy=10, color='k')
                        fileout = '%s/%s.%s.%s.%s.%s.map.png' % (statdir, mod, sp, season, area, stname)
                        fig.savefig(fileout)
                        fig.clf()










