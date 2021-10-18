#!/usr/bin/env python
# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use('Agg')

import  os, sys, argparse

from toolbox import silamfile, timeseries
import datetime as dt
import numpy as np
from toolbox import MyTimeVars

os.umask(0002)


parser = argparse.ArgumentParser('extract_ts2nc.py')
parser.add_argument('input_file',help='path to .ctl file')
parser.add_argument('out_file', help='Path to the netcdf timeseries file')
parser.add_argument('substance', help='variable')

args  = parser.parse_args()


## FIXME reference station file
obsfile='../TimeVars/obs/obs_CO.nc'

print "Getting stations", obsfile
tv = MyTimeVars.TsMatrix.fromNC(obsfile)
stationdic={}
for st in tv.stations:
    stationdic[st.code] = st

stlist = sorted(stationdic.keys())

stidx={}
for i,st in enumerate(stlist):
    stidx[st]=i

ntimes,nst =  tv.vals.shape

#
# Times
#
hrlist=tv.times
timeidx={}
for i,t in enumerate(hrlist):
    timeidx[t]=i

#
# Valmatr
#
valmatr= np.float32(np.nan) * np.empty((ntimes,nst),dtype=np.float32)
ncfile=silamfile.SilamNCFile(descriptor_file=args.input_file)

try:
    serlist = timeseries.fromGradsfile(tv.stations, ncfile.get_reader(args.substance,mask_mode=False), 
                remove_if_outside=True, verbose=False)
except ValueError: ##raised on files with vertical
    serlist = timeseries.fromGradsfile(tv.stations, ncfile.get_reader(args.substance,mask_mode=False,ind_level=0), 
                remove_if_outside=True, verbose=False)
for ser in serlist:
       ist = stidx[ser.station.code]
       for t, v in zip(ser.times(), ser.data()):
          try:
            valmatr[timeidx[t],ist] = v
          except KeyError:
             pass
it=(np.isfinite(np.nanmean(valmatr,axis=1)))
print "%d time steps"%(np.sum(it),)



tsmatr=MyTimeVars.TsMatrix(hrlist,tv.stations,valmatr, "ug/m3")

tsmatr.to_nc(args.out_file)
print args.out_file, "done!"
    



