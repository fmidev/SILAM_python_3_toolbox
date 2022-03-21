#!/usr/bin/env python3

my_description="""
    Handles timeseries for bunch of stsations, calculates and plots timevars

    Enjoy! (Roux)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import sys # Not needed normally
import netCDF4 as nc4
from toolbox import stations as ST ## not to be confused with stations arrays
from toolbox import gridtools
from toolbox import silamfile
import os
import datetime as dt
from support import netcdftime 
import socket

try:
    myhost = os.uname()[1]
    if myhost.startswith("nid") or  myhost.startswith("voima") or myhost.startswith("teho") or  myhost.startswith("eslogin"):
        VoimaBug=True
    else:
        VoimaBug=False
except:
    VoimaBug=False
#from matplotlib import streamplot
#import string


class TimeVarData:
    """
                Calculate daily, weekly and monthly quartiles
    """
    weekhours = np.arange(24*7)
    hours    = np.arange(24)
    seasonmonths = [[12, 1,2], [3,4,5], [6,7,8], [9,10,11]]
    weekdays = np.arange(7) 
    months    = np.arange(12)


    def  __init__(self,times, vals, timeFactors = None, axis=None):
        # Vals -- np.array [itime, istation]
        # or 
        #  np.array [itime]
        self.weeklyvar =  np.nan * np.empty((24*7,4,))  # time, low,median,high,mean
        self.hourvar   =  np.nan * np.empty((24,4))
        self.seasonHourVar = np.nan * np.empty((4,24,4))
        self.dowvar    =  np.nan * np.empty((7,4))
        self.monvar    =  np.nan * np.empty((12,4))
        self.defined = False
        if  timeFactors == None :
            timeHours, timeDOW, timeMon = zip(*[(t.hour, t.isoweekday(), t.month) for t in times])
            timeHours = np.array(timeHours)
            timeDOW   = np.array(timeDOW)
            timeMon   = np.array(timeMon)
        else:
            timeHours, timeDOW, timeMon = timeFactors
#                 print np.sum(np.isfinite(vals))
                 
        if len(vals.shape) == 2:
            (self.ntimes,self.nstations)=vals.shape
        elif len(vals.shape) == 1:
            self.ntimes = len(vals)
            self.nstations = 0
        else:
            raise ValueError("TimeVarData got wrong-shaped value argument"+str( vals.shape))



        if np.sum(np.isfinite(vals)) < 200:
            print("Warning! Not enough vals for Timevar")
            return None

        for day in self.weekdays:
            for hour in self.hours:
                select = np.logical_and(timeHours==hour, timeDOW==day+1)
                if (self.nstations !=0):
                    v=vals[select,:].flat
                else:
                    v=vals[select]
                v=v[np.isfinite(v)]
#                                 try:
                if len(v) > 0 :
                    self.weeklyvar[day*24+hour,0:3] = np.percentile(v,[25,50,75],axis=axis)
                    self.weeklyvar[day*24+hour,3] = np.mean(v)
#                                  except:
#                                    print  vals.shape,  sum(select)
#                                    print v
 #                                   exit()

        for day in self.weekdays:
            select = (timeDOW==day+1)
#                         print "dow select:", np.where(select)
            if (self.nstations !=0):
                v=vals[select,:].flat
            else:
                v=vals[select]
            v=v[np.isfinite(v)]
            if len(v) > 0 :
                self.dowvar[day,0:3] = np.percentile(v,[25,50,75],axis=axis)
                self.dowvar[day,3] = np.mean(v)
                 #sys.exit()

        for hour in self.hours:
            select = (timeHours==hour)
            if (self.nstations !=0):
                v=vals[select,:].flat
            else:
                v=vals[select]
            v=v[np.isfinite(v)]
            if len(v) > 0 :
                self.hourvar[hour,0:3] = np.percentile(v,[25,50,75],axis=axis)
                self.hourvar[hour,3] = np.mean(v)

        for iseason, months in enumerate(self.seasonmonths):
            for hour in self.hours:
                select = np.logical_and(timeHours==hour, (timeMon==months[0])+(timeMon==months[1])+(timeMon==months[2])) 
                if (self.nstations !=0):
                    v=vals[select,:].flat
                else:
                    v=vals[select]
                v=v[np.isfinite(v)]
                if len(v) > 0 :
                    self.seasonHourVar[iseason, hour,0:3] = np.percentile(v,[25,50,75],axis=axis)
                    self.seasonHourVar[iseason, hour,3] = np.mean(v)

        for month in self.months:
            select = (timeMon==month+1)
            if (self.nstations !=0):
                v=vals[select,:].flat
            else:
                v=vals[select]
            v=v[np.isfinite(v)]
            if len(v) > 0 :
                self.monvar[month,0:3] = np.percentile(v,[25,50,75],axis=axis)
                self.monvar[month,3] = np.mean(v)
                 
        self.defined = True
        return None

class EmisCorrection:
    """
        Calculates emission correction factors from ratios of medians
    """
    def  __init__(self,obsVar, modVar):
        self.hourcorr = obsVar.hourvar[:,1]/modVar.hourvar[:,1]
        self.dowcorr  = obsVar.dowvar[:,1]/modVar.dowvar[:,1]
        self.moncorr  = obsVar.monvar[:,1]/modVar.monvar[:,1]
        return None
    def dict(self):
        return {"hourcorr":self.hourcorr, "dowcorr":self.dowcorr, "moncorr":self.moncorr }



def PlotTimevars(fig, timevars,  labels=None, title=None, units=None, plotNeg=False):
    """
        plots array of timevars (array of timevar instances)
    """
    axweek=fig.add_subplot(3,1,1)
    axweek.set_xlim([0,24*7])
    axweek.set_xticks(range(0,24*7,24))
    axweek.xaxis.set_major_formatter(ticker.NullFormatter())
    axweek.xaxis.set_minor_locator(ticker.FixedLocator(np.arange(7)*24+12))
    axweek.xaxis.set_minor_formatter(ticker.FixedFormatter('Mon Tue Wed Thu Fri Sat Sun'.split()))

    seasonnames = "djf mam jja son".split()
    if title:
        axweek.set_title(title)

    axhour = fig.add_subplot(3,4,5)
    axhour.set_xlim([0,24])
    axhour.set_xticks(range(0,24,6))
    axdow = fig.add_subplot(3,4,6)
    axdow.set_xlim([0,7])
    axdow.xaxis.set_major_formatter(ticker.NullFormatter())
    axdow.xaxis.set_minor_locator(ticker.FixedLocator(np.arange(7)+0.5))
    axdow.xaxis.set_minor_formatter(ticker.FixedFormatter('Mo Tu We Th Fr Sa Su'.split()))
    axmon = fig.add_subplot(3,4,7)
    axmon.set_xlim([0,12])
    axmon.xaxis.set_major_locator(ticker.FixedLocator(np.arange(11)+1))
    axmon.xaxis.set_major_formatter(ticker.NullFormatter())
    axmon.xaxis.set_minor_locator(ticker.FixedLocator(np.arange(12)+0.5))
    axmon.xaxis.set_minor_formatter(ticker.FixedFormatter('J F M A M J J A S O N D'.split()))
    axlegend = fig.add_subplot(3,4,8)
    axes = [axweek, axhour, axdow, axmon]
    for iseason,season in enumerate(seasonnames):
        axes.append(fig.add_subplot(3,4,9+iseason))
        axes[-1].set_title(season)
        axes[-1].set_xlim([0,24])
        axes[-1].set_xticks(range(0,24,6))

    colors="red blue green cyan  magenta black".split()
    maxy = max([np.nanmax(v.weeklyvar[:,:]) for v in filter(None, timevars) ])
		
    if plotNeg:
        miny = min([np.nanmin(v.weeklyvar[:,:]) for v in filter(None, timevars) ])
        for ax in axes:
            ax.plot([0,150],[0,0], color='k', ls='-') # Zero line
    else:
        if not maxy > 0 :
            print("Maxy is wrong ", maxy)
            raise ValueError("Maxy is wrong for strictly positive quantity") 
        miny = 0

    nvars=len(timevars)
    if nvars > len(colors):
        raise ValueError("Too many variables for Timevar")

    if labels==None:
        labels = ["%d"%i for i in range(nvars)]
    for ivar in range(nvars):
        var=timevars[ivar]
        if not var:
            continue
        myxvals = [var.weekhours,var.hours, var.weekdays, var.months, var.hours, var.hours, var.hours, var.hours,]  
        myvars  = [var.weeklyvar, var.hourvar, var.dowvar, var.monvar,  var.seasonHourVar[0, :,:],  var.seasonHourVar[1, :,:],  var.seasonHourVar[2, :,:],  var.seasonHourVar[3, :,:] ]
        for ax,x,v in zip(axes,myxvals,myvars): 
            ax.plot(x+0.5,v[:,1], color=colors[ivar], label=labels[ivar]) #Median
            ax.plot(x+0.5,v[:,3], color=colors[ivar],  ls = '--', dashes=(5, 2)) #mean
            ax.fill_between(x+0.5,v[:,0], v[:,2], color=colors[ivar], alpha=0.2)
            ax.set_ylim([miny,maxy])
            ax.set_label(units)
            ax.grid()

    ## Legend axis
    axlegend.set_xlim([-2,-1])
    axlegend.axis('off')
    for ivar in range(nvars):
        axlegend.plot(range(2), range(2), color=colors[ivar], label=labels[ivar])
    axlegend.plot(range(2), range(2), color='grey',  ls = '-', label="Median")
    axlegend.plot(range(2), range(2), color='grey',  ls = '--', dashes=(5, 2), label="Mean")
    axlegend.fill_between(range(2), range(2), color='grey',  alpha=0.2,  label="25th-75th prcntl")
    axlegend.legend( loc='center', frameon=True)                    
    fig.tight_layout()


class SpatialStats:
    """
       Contain Spatial statistics 
    """
    def __init__(self, times, valsobs, valsmod, flipBias = False):
#        print np.sum(np.isfinite(vals))
        (self.ntimes,self.nstations)=valsobs.shape
        if (valsmod.shape != valsobs.shape):
            print(valsmod.shape, valsobs.shape)
            print("SpatialStats got different-shape arrays")
            raise ValueError

        self.times = times
        self.bias = np.nanmean(valsmod - valsobs, axis=1)
        self.FracB = 2*np.nanmean((valsmod - valsobs)/(valsmod + valsobs + 0.1), axis=1)
        self.FGerr = 2*np.nanmean(abs(valsmod - valsobs)/(valsmod + valsobs + 0.1), axis=1)

        if flipBias:
            self.bias *= -1
            self.FracB *= -1
        self.RMS  = np.sqrt(np.nanmean((valsmod - valsobs)*(valsmod - valsobs), axis=1)) 

        self.N = np.nan * np.empty((self.ntimes))
        self.corr = np.nan * np.empty((self.ntimes))
        self.meanOBS = np.nan * np.empty((self.ntimes))
        self.meanMOD = np.nan * np.empty((self.ntimes))
        self.stdOBS = np.nan * np.empty((self.ntimes))
        self.stdMOD = np.nan * np.empty((self.ntimes))

        for it in range(self.ntimes):
            idx=np.isfinite(valsmod[it,:] - valsobs[it,:])
            N = np.sum(idx)
            self.N[it] = N
            if (N<5): continue


            modmean = np.mean(valsmod[it, idx]) 
            obsmean = np.mean(valsobs[it, idx]) 
            self.meanOBS[it] = obsmean
            self.meanMOD[it] = modmean

            self.stdOBS[it] = np.sqrt(np.mean( (valsobs[it, idx] - obsmean)**2 ))
            self.stdMOD[it] = np.sqrt(np.mean( (valsmod[it, idx] - modmean)**2 ))

            self.corr[it] = np.mean(( valsmod[it, idx] - modmean )*(valsobs[it, idx] - obsmean))
            self.corr[it] /= self.stdOBS[it]*self.stdMOD[it]


    def getTV(self,stat):
        if stat == "RMSE":
            return TimeVarData(self.times,self.RMS)
        elif stat == "FGerr":
            return TimeVarData(self.times,self.FGerr)
        elif stat == "Corr":
            return TimeVarData(self.times,self.corr)
        elif stat == "Bias":
            return TimeVarData(self.times,self.bias)
        elif stat == "FracB":
            return TimeVarData(self.times,self.FracB)
        elif stat == "N":
            return TimeVarData(self.times,self.N)
        elif stat == "meanOBS":
            return TimeVarData(self.times,self.meanOBS)
        elif stat == "meanMOD":
            return TimeVarData(self.times,self.meanMOD)
        elif stat == "stdOBS":
            return TimeVarData(self.times,self.stdOBS)
        elif stat == "stdMOD":
            return TimeVarData(self.times,self.stdMOD)
        else:
            print("timevar can be one of RMSE Corr Bias")
            raise ValueError


class TemporalStats:

    """
       Contain Spatial statistics 
    """
    def __init__(self, lons, lats, case, valsobs, valsmod, flipBias = False):

#        print np.sum(np.isfinite(vals))
        (self.ntimes,self.nstations)=valsobs.shape
        if (valsmod.shape != valsobs.shape):
            print(valsmod.shape, valsobs.shape)
            print("TemporalStats got different-shape arrays")
            raise ValueError

        assert (len(lons) == self.nstations)
        assert (len(lats) == self.nstations)
        self.lons  = lons
        self.lats  = lats
        self.case  = case ##To appear in the table header and in the title

        self.validstats="Bias RMS N Corr meanObs meanMod stdObs stdMod MORatio MOstdRatio FracB FGerr".split()

        self.Bias = np.nanmean(valsmod - valsobs, axis=0)
        self.RMS  = np.sqrt(np.nanmean((valsmod - valsobs)*(valsmod - valsobs), axis=0)) 
        self.FracB = 2*np.nanmean((valsmod - valsobs)/(valsmod + valsobs + 0.1), axis=0)
        self.FGerr = 2*np.nanmean(abs(valsmod - valsobs)/(valsmod + valsobs + 0.1), axis=0)

        self.N = np.nan * np.empty((self.nstations))
        self.Corr = np.nan * np.empty((self.nstations))
        self.meanObs = np.nan * np.empty((self.nstations))
        self.meanMod = np.nan * np.empty((self.nstations))
        self.stdObs = np.nan * np.empty((self.nstations))
        self.stdMod = np.nan * np.empty((self.nstations))

        for ist in range(self.nstations):
            idx=np.isfinite(valsmod[:,ist] - valsobs[:,ist])
            N = np.sum(idx)
            self.N[ist] = N
            if (N<5): continue
            modmean = np.mean(valsmod[idx, ist]) 
            obsmean = np.mean(valsobs[idx, ist]) 
            self.meanObs[ist] = obsmean
            self.meanMod[ist] = modmean

            self.stdObs[ist] = np.sqrt(np.mean( (valsobs[idx,ist] - obsmean)**2 ))
            self.stdMod[ist] = np.sqrt(np.mean( (valsmod[idx,ist] - modmean)**2 ))

            self.Corr[ist] = np.mean(( valsmod[idx,ist] - modmean )*(valsobs[idx,ist] - obsmean))
            self.Corr[ist] /= self.stdObs[ist]*self.stdMod[ist]

        self.MORatio    = self.meanMod/self.meanObs
        self.MOstdRatio = self.stdMod/self.stdObs

    def getstat(self, statname):
        if statname in self.validstats:
            return getattr(self, statname)
        else:
            print("getstat alled with statname = '%s'"%(statname))
            raise ValueError("statlist can be one of "+",".merge(self.validstats))



    def printStats(self, outf, statlist,  statproc, rawHead, ifHead, idxStations=None):

        #case -- string 
        # statlist list of strings of statistics
        if ifHead:
            outf.write("\n\n%s\n"%(self.case,))
            outf.write("%25s"%("",))
            for st in statlist:
                outf.write("%10s"%(st,))
            outf.write("\n")

        statfuns=dict(
                    mean = np.nanmean, 
                    median = np.nanmedian, 
                    prc95 = lambda x : np.nanpercentile(x,95),
                    first = lambda x : x[0],  ##First station from the list
                    )
        try:
            stfun = statfuns[statproc]
        except KeyError:
            raise KeyError("statproc argument can be one of "+",".merge(statfuns.keys()))

        Nstations=self.nstations
        if idxStations:
            Nstations = np.sum(np.isfinite(self.meanObs[idxStations]))

        rawtitle="%25s"%("%s %s N=%d"%(rawHead,statproc,Nstations))
        outf.write(rawtitle)
        for st in statlist:
            statvals = self.getstat(st)
            if idxStations:
                outf.write("%10.2f"%(stfun(statvals[idxStations])))
            else:
                outf.write("%10.2f"%(stfun(statvals)))
        outf.write("\n")

    def plot2Map(self, basemap,  stat, cmap, norm, idxStations):
        starr=self.getstat(stat)
        return basemap.scatter(self.lons[idxStations], self.lats[idxStations], 
                    c=starr[idxStations], s=30, cmap=cmap, norm=norm, linewidths=0)
        

###############################################################

class SeasonalStats:

    """
       Contain Temporal statistics over Seasons
    """
    def __init__(self, obsMatr, modMatr, case):
        stations = obsMatr.stations
        hrlist  =  obsMatr.times

        valsobs = obsMatr.vals
        valsmod = modMatr.vals

        lons = np.array([st.lon for st in stations])
        lats = np.array([st.lat for st in stations])

#        print np.sum(np.isfinite(vals))
        (self.ntimes,self.nstations)=valsobs.shape
        if (valsmod.shape != valsobs.shape):
            print(valsmod.shape, valsobs.shape)
            print("TemporalStats got different-shape arrays")
            raise ValueError


        self.seasons="all djf mam jja son".split()
        self.stats={}
        seasonmonths = dict(djf=[12, 1,2], mam=[3,4,5], jja=[6,7,8], son=[9,10,11], all=range(1,13))

        self.stats["all"] = TemporalStats(lons, lats, case, valsobs, valsmod)
        timeMon = np.array([t.month for t in hrlist])
        for season in "djf mam jja son".split():
            months = seasonmonths[season]
            idxtime = (timeMon==months[0])+(timeMon==months[1])+(timeMon==months[2])
            self.stats[season] = TemporalStats(lons, lats, case+", "+season, valsobs[idxtime,:], valsmod[idxtime,:])


#
# Read the data
#
class PollutantModobs:
    """
    Class to contain synchronous data, station info, time axis etc.
    with finctions to select stations by ID etc
    """
    def __init__(self, ncpol, ncnames, titles=None, airbasefile=""):
        """
        ncnames -- list of filenames, 
        """
        if titles==None:  titles=ncnames

        # Read the data
        self.titles=[]
        self.data={}
        stnames = None
        self.ncpol=ncpol # Variable name in .nc files
        for infile, title in zip(ncnames,titles):
            if stnames==None: # First file (presumably, observations)
                self.titles.append(title)
                nc=netcdf.netcdf_file(infile,"r")
                stnames = np.array([''.join(nc.variables['stn_id'].data[ist,:]) for ist in  range(nc.dimensions["station"])])
                mdlTime = netcdftime.utime(nc.variables['time'].units)
                self.times = mdlTime.num2date(nc.variables['time'].data) 
                #for i in range(1000):
                #        print self.times[i].strftime("%a %F %T")
                                
                # Valid satation indices
                nodataidx = [ np.all(np.isnan(nc.variables[ncpol].data[:,i]))   for i in range(len(stnames)) ]
                self.stselect = np.logical_not(np.array(nodataidx))
                self.stid=np.array(stnames[self.stselect]) # station ID
                self.units=nc.variables[ncpol].units
                self.data[title]=nc.variables[ncpol].data[:,self.stselect]
                nc.close()               
            else:
                self.AddData(infile,title)
                
        airbasetypes = getAirbaseTypes(airbasefile)

        stidx = np.array( [airbasetypes["idx"][name] 
                           for name in self.stid]
                           )
        self.sttype = np.array(["%s_%s"%(airbasetypes["station_type_of_area"][i],
                                         airbasetypes["type_of_station"][i].lower(),
                                         ) for i in stidx ]
                                         )
        self.countrycode = np.array([airbasetypes["country_iso_code"][i] for i in stidx])
        self.country = np.array([airbasetypes["country_name"][i] for i in stidx])
        #for i in range(10):
        #       print "%10s %10s %5.2f"%(self.stid[i], self.country[i],nc.variables[ncpol].data[5,i])
                
        timeHours, timeDOW, timeMon = zip(*[(t.hour, t.isoweekday(), t.month) for t in self.times])
        self.timeHours=np.array(timeHours)
        self.timeDOW=np.array(timeDOW)
        self.timeMon=np.array(timeMon)

    def AddData(self,ncname,title): # adds one more ncfile to the existing dataset
        self.titles.append(title)
        nc=netcdf.netcdf_file(ncname,"r")
        mdlTime = netcdftime.utime(nc.variables['time'].units)
        mytimes = mdlTime.num2date(nc.variables['time'].data)
        if (mytimes != self.times).any():
            print("Error: incompatible times in %s"%(ncname))
            print("OBS", self.times[0].strftime("%F %T"), self.times[-1].strftime("%F %T"))
            print("mod", mytimes[0].strftime("%F %T"), mytimes[-1].strftime("%F %T"))
            exit (-1)
        mystnames = np.array([''.join(nc.variables['stn_id'].data[ist,:]) for ist in  range(nc.dimensions["station"])])
        mystid = mystnames[self.stselect]
        if (np.any(mystid != self.stid)):
            print("Error: incompatible stations")
            print(mystid)
            print(stid)
            exit(-1)
        self.data[title]=nc.variables[self.ncpol].data[:,self.stselect]
        nc.close()

    def DelData(self,title): # removes one ncfile from the existing dataset
        self.titles.remove(title)
        del self.data[title]
    
    def SelectStations(self,names=None, country=None, ccode=None, sttype=None, minstations = 1):
        """
        return a bool array  selecting  stations
        """
        sel=np.ones((len(self.stid)),dtype=bool)
        if names != None:
            sel *= np.in1d(self.stid, names)
            
        if ccode != None:
            #sel *= np.array([c in ccode for c in   self.countrycode])
            sel *= np.in1d(self.countrycode,ccode)

        if country != None:
#               sel *= np.array([c in country for c in  self.country])
            sel *= np.in1d(self.country, country)

        if sttype != None:
            #sel *= np.array([t in sttype for t in  self.sttype])
            sel *= np.in1d(self.sttype, sttype)

        if np.sum(sel) >= minstations:
            return sel
        else:
            return None

###############################################################################
#
#  Class tsMatrix: a 2-D and 3-D timeseries representatoin as (times, stations,variable[s])
#  For the simple case of just one variable the dimension is dropped in vals array
#
###############################################################################

class TsMatrix:

    def __init__(self, times, stations, variables, vals, units, 
                 fill_value = -999999, timezone='UTC'):
        # All dimension must be sorted in accordance with the types
        # Otherwise manipulations and searches are slow or impossible
        stcodes=[ st.code for st in stations ]
        if np.any(sorted(times) != 
                  times) or np.any(sorted(stcodes) != 
                                   stcodes) or np.any(sorted(variables) != 
                                                       variables):
            print('Sorting problem: times / stations/  variables:', 
                  np.any(sorted(times) != times), 
                  np.any(sorted(stcodes) != stcodes),
                  np.any(sorted(variables) != variables))
            raise ValueError('Some dimensions are not sorted properly')

        if len(vals.shape) == 2:  # just a single variable, 2D case
            (self.ntimes, self.nstations) = vals.shape
            if ((len(times),len(stations)) != vals.shape):
                print((len(times),len(stations),len(variables)), "!=", vals.shape)
                raise ValueError("TsMatrix got incompatibel-shape arrays")
            self.vals = vals
            assert (len(variables) == 1)
        else:                         # must be a list or numpy array. 3D case
            (self.nvariables, self.ntimes, self.nstations) = vals.shape
            if ((len(variables),len(times),len(stations)) != vals.shape):
                print(len(variables),(len(times),len(stations)), "!=", vals.shape)
                raise ValueError("TsMatrix got incompatibel-shape arrays")
            if (self.nvariables == 1):
                self.vals = vals[0,:,:]  ## make 2d array 
            else:
                self.vals = vals


        if len(units) != len(variables):
            print('Variables:',variables)
            print('Units:',units)
            print('Units and variables have different lengths')
            raise ValueError("len(units) != len(variables)")
        self.times = times
        self.stations = stations
        self.variables = np.array(variables)
        self.units = np.array(units)
        self.fill_value = fill_value
        self.timezone = timezone

    #==========================================================================
    def to_dataframe(self, var=None, copy=None):
        #
        # Converts given variable of tsmatrix to dataframe with station codes 
        # as column headers 
        #
        import pandas as pd

        columns = [ st.code for st in self.stations ]
        if  len(self.vals.shape) == 2 and (var is None or var == self.variables[0]):
            return pd.DataFrame(data=self.vals, index=self.times, columns=columns, copy=copy)
        else:
            ivar = np.where(self.variables == var)[0][0]
            return pd.DataFrame(data=self.vals[ivar,:,:], index=self.times, columns=columns, copy=copy)

    #==========================================================================

    def to_nc(self, filename):
        #
        # Stores the tsMatrix object to the nc file
        #
        # dimensions
        if  len(self.vals.shape) == 3:
            nvar,nt,nst = len(self.variables), len(self.times), len(self.stations)
        else:
            nt,nst = len(self.times), len(self.stations)
            
        # start time #some versions of ncdump have issues parsing minutes in reftime...
        refdate=self.times[0].date()
        reftime=dt.datetime(refdate.year, refdate.month, refdate.day)

        tmpfilename = "%s.%s.pid%08d"%(filename, socket.gethostname(), os.getpid())


        with nc4.Dataset(tmpfilename , "w", format="NETCDF4") as outf:

            outf.featureType = "timeSeries";
            strlen=64
            
            # dimensions
            time = outf.createDimension("time", None)
            slen = outf.createDimension("name_strlen", strlen)
            station = outf.createDimension("station",nst)
            if len(self.vals.shape) == 3:
                variable = outf.createDimension("variable",nvar)
                valdims = ("variable","time","station")
            else:
                valdims = ("time","station")

            # time
            t = outf.createVariable("time","i4",("time",))
            t.standard_name="time"
            t.long_name="end of averaging period"
            t.calendar="standard"
            t.units = reftime.strftime("minutes since %Y-%m-%d %H:%M:%S UTC")
            t[:] = [ (h - reftime).total_seconds()/60 for h in self.times ]
            t.timezone = self.timezone 

            # longitude
            lon = outf.createVariable("lon","f4",("station",), zlib=True, complevel=4)
            lon.standard_name = "longitude"
            lon.long_name = "station longitude"
            lon.units = "degrees_east"

            # latitude
            lat = outf.createVariable("lat","f4",("station",), zlib=True, complevel=4)
            lat.standard_name = "latitude"
            lat.long_name = "station latitude"
            lat.units = "degrees_north"

            # altitude
            alt= outf.createVariable("alt","f4",("station",), zlib=True, complevel=4)
            alt.standard_name = "surface_altitude"
            alt.long_name = "station altitude asl"
            alt.units = "m"
            alt.positive = "up";
            alt.axis = "Z";
            
            if len(self.vals.shape) == 3:
                # variables
                v = outf.createVariable("variable","c",("variable",'name_strlen'), zlib=True, complevel=4)
                v.long_name = "variable name"
                # units - the size of variables
                u = outf.createVariable("unit","c",("variable",'name_strlen'), zlib=True, complevel=4)
                u.long_name = "variable unit"
                for iq in range(len(self.variables)):
                    v[iq,:] = nc4.stringtoarr( self.variables[iq], strlen, dtype='S')
                    u[iq,:] = nc4.stringtoarr( self.units[iq], strlen, dtype='S')

            # stations
            stcode = outf.createVariable("station_code","c",("station","name_strlen"), 
                                         zlib=True, complevel=4)
            stcode.long_name = "station code"
            stcode.cf_role = "timeseries_id";
            stname = outf.createVariable("station_name","c",("station","name_strlen"), 
                                         zlib=True, complevel=4)
            stname.long_name = "station name"
            starea = outf.createVariable("area_type","c",("station","name_strlen"), 
                                         zlib=True, complevel=4)
            starea.long_name = "area_type"
            stsource = outf.createVariable("source_type","c",("station","name_strlen"), 
                                           zlib=True, complevel=4)
            stsource.long_name = "source_type"
            
            # values
            val= outf.createVariable("val","f4", valdims, 
                                     fill_value = self.fill_value, zlib=True, complevel=4)
            val.coordinates = "lat lon alt station_code"
            if len(self.vals.shape) == 3:
                val.units = ''
            else:
                val.units = self.units
                try: 
                    val.long_name = self.variables[0]  # this is vairable name, e.g. cnc_O3_gas
                    # FIXME  This is ny no means standard_name, and should not be used this way 
                    val.standard_name = self.variables[0]  # this is vairable name, e.g. cnc_O3_gas##
                except: pass

            outdat = self.vals
            
            charTmp = np.zeros(shape=(len(self.stations), strlen), dtype='S1')
            for stvar, attname in [(stcode, 'code'), (stname, 'name'), (starea, 'area'), (stsource, 'source'),]:
                for ist,st in enumerate(self.stations):
                    charTmp[ist,:] = nc4.stringtoarr( getattr(st, attname) , strlen,dtype='S')
                stvar[:,:] = charTmp

            lon[:] = list( (st.lon for st in self.stations))
            lat[:] = list( (st.lat for st in self.stations))
            alt[:] = list( (st.hgt for st in self.stations))

            
            # main dataset: replace nan with own fill_value
            outdat[np.isnan(outdat)] = self.fill_value
            val[:] = outdat

        # Rename the temporary file. On Windows, unlike Unix, os.rename cannot replace existing file
        # but replace does, albeit it is not an atomic operation there (in Unix it should be atomic)
        os.replace(tmpfilename, filename)  


    #==========================================================================
    @classmethod
    def fromNC(self, ncfile, tStart=None, tEnd=None, stations2read_=None, variables2read_=None):
        #
        # Reads the pre-made observation NC file and puts the data to a 
        # matrix station-time. Note that tStart, tEnd are only limiters, the
        # whole interval is not necessarily filled-in.
        #
        if not stations2read_ is None: stations2read = sorted(stations2read_)
        else: stations2read = None
        if not variables2read_ is None: variables2read = sorted(variables2read_)
        else: variables2read = None
         
        with nc4.Dataset(ncfile) as nc:
            nc.set_auto_maskandscale(False) ## Never mask, never scale
            # Do we have the "variable" dimension?
            ifMultiVar = ('variable' in nc.variables.keys())
            if ifMultiVar:
                variables = nc4.chartostring(nc.variables['variable'][:])  # array of  names for timeseries variables
                if variables2read:
                    idxVar = np.searchsorted(variables, variables2read, sorter=np.argsort(variables))
                    if np.any(idxVar == len(variables)): # some variable does not exist in the file
                        for ivar in range(len(variables)):
                            if idxVar[ivar] == len(variables):
                                print('Requested variable does not exist: ', variables2read[ivar])
                    raise ValueError("Variable Not Found in the file")
                else:
                    idxVar = slice(None,None,None)
                units = nc4.chartostring(nc.variables['unit'][:])
            else:
                try:
                    variables = [nc.variables['val'].long_name]  # just a string but make a list
                except: variables = ['']   # for old tsMatrices
                try:
                    units = [nc.variables['val'].units]
                except AttributeError:
                    print("Unknown units in %s, ug/m3 assumed"%(ncfile))
                    units=["ug/m3"]

            # times
            tvals = nc.variables['time'][:]
            ### added check, because of possible problem with cdftime in hrlist 
            if hasattr(tvals,'mask'): tvals = tvals.filled()
            tunits = nc.variables['time'].units
            # Get stations
            stlist = nc4.chartostring(nc.variables['station_code'][:])
            if stations2read is None:     # need indices of stations to get?
                idxStat = slice(None,None,None)
            else:
                idxStat = np.searchsorted(stlist, stations2read, sorter=np.argsort(stlist))
                if np.any(idxStat == len(stlist)):
                    for ist in range(len(idxStat)):
                        if idxStat[ist] == len(stlist): 
                            print('Requested station does not exist: ', stations2read[ist])
                    raise
            stnames = nc4.chartostring(nc.variables['station_name'][idxStat])
            stcodes = nc4.chartostring(nc.variables['station_code'][idxStat])
            stareas = nc4.chartostring(nc.variables['area_type'][idxStat])
            stsources = nc4.chartostring(nc.variables['source_type'][idxStat])
            stx = nc.variables['lon'][idxStat]
            sty = nc.variables['lat'][idxStat]
            stz = nc.variables['alt'][idxStat]

            obsStations = []
            for i in range(len(stnames)):
                obsStations.append(ST.Station(stcodes[i], stnames[i], stx[i], sty[i], 
                                                    height=stz[i], area_type=stareas[i], 
                                                    dominant_source=stsources[i]))

            ##
            ## Time slice
            itstart = None
            itend = None
            hrlistFull = netcdftime.num2date(tvals,tunits)
            if tStart: 
                itstart = np.searchsorted(hrlistFull, tStart)
            if tEnd:
                itend = np.searchsorted(hrlistFull, tEnd, side='right')
            selection_time = slice(itstart, itend, 1)
            hrlist = hrlistFull[selection_time]

            if ifMultiVar:
                # Three axes
                if type(idxStat) == list:
                    valmatr = nc.variables['val'][idxVar,selection_time,:][:,:,idxStat] ## station is inner dimension
                else:
                    # no station selection -- can just read it in one kick
                    valmatr = nc.variables['val'][idxVar,selection_time,:]
            else:
                    valmatr = nc.variables['val'][selection_time,idxStat]
            #
            # Fill value
            #
            try:  # if fill_value is explicit
                fill_v = nc.variables['val']._FillValue
                valmatr = np.where(valmatr == fill_v, np.nan, valmatr)
            except:  # have to use default
                try:
                    flil_v = nc4.default_fillvals[{np.dtype('float32'):'f4',
                                                   np.dtype('float64'):'f8'}[nc.variables['val'].dtype]]
                    valmatr = np.where(np.logical_and(np.abs(valmatr-fill_v) > np.abs(1e-5*fill_v),
                                                      np.isfinite(valmatr)), valmatr, np.nan)
                except:
                    print('Failed to find any fill value, do nothing and hope for the best')
                    fill_v = -999999  # pretty much anything, it is not used anyway
            #
            # Timezone
            try: timezone =  nc.variables['time'].timezone
            except: timezone = 'UTC'
        
        return self(hrlist, obsStations, variables, valmatr, units, fill_v, timezone)

    #==========================================================================
    
    @classmethod
    def extract_from_fields(cls, stations, times, variables, inFieldNCFNm, refheights=None):
        #
        # Reads the basic or processed model output - maps in netcdf format - and
        # extracts stations for the given time period or for the entire duration
        # the oberved period.
        # Returns the tsMatrix object, single-variable
        # refheights  == reference heights for station extraction in _model_ vertical
        #
        print('Extracting from ', inFieldNCFNm)
        #
        # The source file
        #
        if isinstance(variables, list):
            ncvars = variables
        else:
            ncvars = [ variables ]

        ncfile = silamfile.SilamNCFile(inFieldNCFNm)
        izs  = None
        if ncfile.vertical is None:
            ind_level = None
            if not refheights  is None:
                raise ValueError("Attempt to read 3D from a file that has only 2D")
        elif refheights  is None:
            ind_level = 0
        else:
            ind_level = None ## 3D reader
            bnds = ncfile.vertical.boundaries()
            nz = ncfile.vertical.number_of_levels()
            izs = np.searchsorted(bnds[1:],refheights) ## Underground stations to lowest layer



            idxabove = np.where(izs >= nz) ##Above the domain top
            if len(idxabove) > 0:
           #    for ist in idxabove:
            #    import pdb; pdb.set_trace()
                print("%d of %d refheights above the domain top! Forcing them in."%(len(idxabove),len(refheights)))

                izs[idxabove] =  nz - 1 


        readers = []
        for v in ncvars:
            readers.append(ncfile.get_reader(v, mask_mode=False, ind_level=ind_level))
        #
        nStat = len(stations)
        # Times
        #
        if times is None:
            hrlist = readers[0].t()
        else:
            hrlist = np.array(times)
        ntimes = len(hrlist)
#        timeidx = {}
#        for i,t in enumerate(hrlist): 
#            timeidx[t]=i
        try: timezone = ncfile.variables['time'].timezone
        except: timezone = 'UTC'
        #
        # Valmatr
        #
        valmatr= np.ones((len(ncvars),ntimes,nStat),dtype=np.float32) * np.nan
        #
        # checklist for locations of stations
        #
        nx = ncfile.grid.nx  ### IF reading expressioon form netcdf file directly reader.nx() triggers "NotImplemented error"
        ny = ncfile.grid.ny  ## I could not figure why, but ncfile.grid seems to be a good alternative. R
        timesIn = np.array(readers[0].t())
        # are all needed times present in the file?
        idxTimes = np.searchsorted(timesIn, hrlist)
        idxTimes[idxTimes >= len(timesIn)] = len(timesIn)-1  
        idxTimesOK = timesIn[idxTimes] == hrlist
        if not np.all(idxTimesOK): print('Not all needed times are in the file')
        # stations
        stOK = np.array([False] * nStat)
        coords = []
        nOutside = 0
        stlons = np.array([s.lon for s in stations])
        stlats = np.array([s.lat for s in stations])
        ixs, iys = readers[0].indices(stlons, stlats)
        stOK = (ixs >= 0) * (ixs < nx) * (iys >= 0) * (iys < ny)

        nOutside = np.sum(np.logical_not(stOK))
        #for i in  np.where(np.logical_not(stOK))[0]: ##where returns 1-element tuple 
        #    print(stations[i], ixs[i], iys[i], ' vs ', nx, ny, 'Station', stations[i].code, ' is outside the domain')
        print('Stations outside the domain:', nOutside)
       # exit
        # note that reader is projected readers, i.e. the actual reader is wrapped
#        print(reader)
#        try: print(reader.wrapped.units)
#        except: print('failed reader.wrapped.units')
#        try: print(reader.wrapped.vars.units)
#        except: print('failed reader.wrapped.vars.units')
        try:
            units = [ r.wrapped.ncattr('units') for r in readers ]
        except AttributeError:
            try:
                units = [ r.wrapped.vars[0].units for r in readers ]
            except AttributeError:
                try:
                    units = [ r.wrapped.ncvar.units for r in readers ]
                except AttributeError:
                    print("Unknown units in %s" % (inFieldNCFNm))
                    units = [ 'unknown' for r in readers ]

        # Is times big? Do we report the progress?
        if ntimes > 10000: counter = 1
        else: counter = None

        if izs is None:
            coords = (ixs[stOK], iys[stOK]) ## 2D fields to come
        else:
            coords = (ixs[stOK], iys[stOK], izs[stOK]) ## 3D fields to come

        for it, t in enumerate(hrlist):  #range(ntimes):
            if counter:
                if counter % 5000 == 0: print('Reading: ' + str(hrlist[it]))
                counter += 1
            if idxTimesOK[it]:
                for iv, v in enumerate(ncvars):
                    readers[iv].seek_abs(idxTimes[it])   #goto(t)     #hrlist[it])
                    field = readers[iv].read(1) ## Reader might return strange things
                    # masked or not?                    
                    if isinstance(field, np.ndarray):
                        valmatr[iv,it,stOK] = field[coords]
                    elif isinstance(field, np.ma.MaskedArray):
                        valmatr[iv,it,stOK] = field.data[coords]
                    else:
                        raise TypeError("Reader returned %s"%(type(field)))
            else:
                print('Time ', hrlist[it], ' is not in the input nc file')

        # whatever fill value is in the input file, here it is nan
        valmatr = np.where(valmatr == ncfile.fill_value(ncvars[0]), np.nan, valmatr)
        if np.sum(np.logical_not(np.isfinite(valmatr))) > 0:
            print('TSM Bad values:', np.sum(np.logical_not(np.isfinite(valmatr))))

        it=(np.any(np.isfinite(valmatr),axis=1))
        print("%d valid time steps, %d (of %d) stations" % (np.sum(it), np.sum(stOK), len(stations)))

        return cls(hrlist, stations, ncvars, valmatr, units , -999999, timezone)


    #==========================================================================
    
    @classmethod
    def fromTimeSeries(self, tseries, unit, time_tag='end'):
        #
        # convert timeseries objects to tsMatrix objects
        # Requirement: identical time step in all series
        #
        stations = list( (ts.station for ts in tseries) )
        # A problem: time series do not store missing data. Have to figure out the regular time axis
        tStart = tseries[0].times()[0]
        tEnd = tseries[0].times()[-1]
        tStep = tseries[0].durations()[0]
        # find the earliest and latest times, and check the averaging step
        for ts in tseries:
            if len(ts.times()) == 0: continue
            if np.any(np.array(ts.durations()) != tStep):
                raise ValueError('Non-constant duration of time series %s' % ts.station.code)
            if tStart > ts.times()[0]: tStart = ts.times()[0]
            if tEnd < ts.times()[-1]: tEnd = ts.times()[-1]
        # timeseries put the time tag at the beginning of the averaging interval
        # SILAM - to the end. CDO assumes middle but allows overriding. Here, the default is
        # as SILAM does, but alsp allows overriding.
        if time_tag == 'beginning': tShift = dt.timedelta(hours=0)
        elif time_tag == 'end':     tShift = tStep
        elif time_tag == 'middle':  tShift = tStep / 2.
        else: raise ValueError('Unknown time_tag: ' + str(time_tag))
        tsM_times = np.array(list( (tStart + tShift + tStep * i
                                    for i in range(np.int32(np.round((tEnd-tStart)/tStep))))))
        # create the matrix
        valmatr= np.float32(np.nan) * np.empty((len(tsM_times),len(stations)), 
                                               dtype=np.float32)
        # Fill it in
        idxSort = np.argsort(stations)
        for iSt in range(len(stations)):
            for it, t in enumerate(tsM_times):
                try: valmatr[it,idxSort[iSt]] = tseries[idxSort[iSt]][t]
                except: pass
        return self(tsM_times, sorted(stations), [tseries[0].quantity], valmatr, [unit], timezone='UTC')

    @classmethod
    def fromSilamLog(cls, infile):
        descr="""
            converts silam logfile to tsmatrix with TOTAL MASS REPORT + emissions
            species appear as variables, budget components appear as "stations"
        """

        if infile == "-":
            inf=sys.stdin
            closeInf = False
        else:
            if infile.endswith(".bz2"):
                inf = bz2.open(infile,'rt')
            elif infile.endswith(".gz"):
                inf = gzip.open(infile,'rt')
            else:
                inf = open(infile,'rt')
            closeInf = True

        specieslines = {}
        emstot = {}
        times=[]

        prevdate=""
        date=""
        replen=0 ## Number of words in a report line
        for l0 in inf:
            l=l0.lstrip() ## Get to after init
            if l.startswith("Last output time now:"):
                date=l.split("now:")[1][:-1]
                continue

            # Increment emissions
            if l.startswith("Total injected emission mass as counted from sources"):
                for l1 in inf:
                    l1 = l1.lstrip()
                    a=l1.split()
                    if a[0].endswith("__src"):
                        emstot[a[0][0:-5]] += float(a[-1])
                    else:
                        break
                continue

            ##
            ## Read mass report
            if l.startswith("========== TOTAL MASS REPORT ============"):
                if prevdate >=  date:  ## Several massreports per timestep, only the last one counts
                    continue
                prevdate = date
                times.append(date)

                l=next(inf)
                head=next(inf).lstrip()
                assert head.startswith("Species")
                replen = len(head.split())
                    
                for l1 in inf:
                    l1 = l1.lstrip()
                    if l1.startswith("Grand"): ## End of report
                        break
                    a=l1.split()
                    if len(a) != replen:
                        print(head)
                        print(l1)
                        print(replen)
                        raise ValueError("Strange number of words")
                    if not a[0] in specieslines:
                        specieslines[a[0]] = []
                        emstot[a[0]] = 0.
                    specieslines[a[0]].append(l1[:-1]+ " %10.5e"%(emstot[a[0]],)) 

        if closeInf:
            inf.close()

        ## Parse strings
        species = sorted(emstot.keys())
        fields = head.split()[1:] + ['Emis']
        timestamps = np.array([dt.datetime.strptime( t, "%Y-%m-%d %H:%M:%S.0") for t in times])
        nt,nsp,nf = len(timestamps), len(species), len(fields)
        valmatr = np.zeros((nsp,nt,nf), dtype=np.float32) ## species -> variable (might have different units)
                                       ## field  -> stations

        stations = [ ST.Station( fld, 'X', np.nan, np.nan, np.nan, "", "") for fld in sorted(fields) ]
        storder = np.argsort(fields)
        for isp, sp in enumerate(species):
            for it, l in enumerate(specieslines[sp]):
                linevals = np.array ([ float(v) for v in l.split()[1:] ]) ## 
                valmatr[isp,it,:] = linevals[storder]  ##In alphabetic order


        return cls(timestamps,stations, species, valmatr, ['mass_unit']*nsp)        


    #==========================================================================

    def to_map(self, grid, fill_value, outFNm, timeStart=None, timeEnd=None):
        # converts the tsMatrix to a map and stores into a file
        # open the output file
        if timeStart is None: idxStart = 0
        else: ixdStart = np.searchsorted(self.times, timeStart)
        if timeEnd is None: idxEnd = len(self.times)
        else: idxEnd = max(np.searchsorted(self.times, timeStart) + 1, len(self.times))
        if idxStart >= idxEnd:
            print('Cannot store the tsMatrix: start-end indices in conflict',idxStart, idxEnd)
            raise
        outF = silamfile.open_ncF_out(outFNm, 
                                      'NETCDF4', grid, 
                                      silamfile.SilamSurfaceVertical(),
                                      self.times[0], self.times, # anal_time, times, 
                                      [],           # vars3d 
                                      self.variables, 
                                      dict(((v, u) for v, u in zip(self.variables, self.units))),
                                      fill_value,
                                      True, 3,  # ifCompress, ppc
                                      'From tsMatrix')     # hst
        # Make a temporary single-time-step map
        mapTmp = np.ones(shape=(grid.ny,grid.nx)) * fill_value
        fx, fy = grid.geo_to_grid(np.array(list(((s.lon for s in self.stations)))),
                                  np.array(list(((s.lat for s in self.stations)))))
        ix = np.round(fx).astype(np.int32)
        iy = np.round(fy).astype(np.int32)
        # Time cycle
        for it, t in enumerate(self.times):
            mapTmp[iy,ix] = self.vals[it,:]   # overwrite the cells with this tsM values
            mapTmp[np.isnan(mapTmp)] = fill_value
            outF.variables[self.variables[0]][it,:] = mapTmp[:,:]
            mapTmp[:,:] = fill_value
        outF.variables['time'].timezone = self.timezone
        outF.close()
            

    #==========================================================================
    
    def join(self, tsMatrToAdd):
        return concatenate([self,tsMatrToAdd]) 
    
    #==========================================================================
    
    @classmethod
    def concatenate(cls, arMatrices):
        #
        # Concatenates matrices in time and/or station dimension
        # Can do it for both statoin dimension and for time dimension
        # Overlapping times / stations are not added. Sorting of times is
        # verified, sorting of stations is not.

        if len(arMatrices) == 1:
            return arMatrices[0]

        timezone = arMatrices[0].timezone
        variables = arMatrices[0].variables
        units = arMatrices[0].units
        
        stationdic={}
        timedic={}
        for tsm in arMatrices:
            for st in tsm.stations:
                stationdic[st.code] = st

            for t in tsm.times:
                timedic[t] = 1

            if timezone !=  tsm.timezone:
                print(timezone, '!=',  tsm.timezone)
                raise ValueError("Different timezone in tsmatrices to concatenate")

            if np.any(variables != tsm.variables):
                print('variables = ',    variables )
                print('tsm.variables = ', tsm.variables )
                raise ValueError('join is only allowed for same set of variables')
        
            if np.any(units != tsm.units):
                print('units = ',    units )
                print('tsm.units = ', tsm.units )
                raise ValueError('join is only allowed for same set of units')
        
        timesNew = sorted(timedic.keys())
        codesNew = np.array(sorted(stationdic.keys()))
        stationsNew = [ stationdic[code] for code in codesNew ]

        #
        nV, nT, nSt = len(variables), len(timesNew), len(stationsNew)

        if nV == 1:
            values = np.ones(shape=(nT, nSt), dtype=np.float32) * np.nan
        else:
            values = np.ones(shape=(nV, nT, nSt), dtype=np.float32) * np.nan


        for tsm in arMatrices:
            stcodes = np.array([ st.code for st in tsm.stations ])
                        
            idxStatInNew = np.searchsorted(codesNew, stcodes)
            idxTimesInNew = np.searchsorted(timesNew, tsm.times)
            if nV == 1:
                for iTime in range(len(idxTimesInNew)):
                    values[idxTimesInNew[iTime], idxStatInNew] = tsm.vals[iTime,:]
            else:
                for iTime in range(len(idxTimesInNew)):
                    values[:,idxTimesInNew[iTime], idxStatInNew] = tsm.vals[:,iTime,:]
            
        return cls(timesNew, stationsNew, variables, values, units , -999999, timezone)

 
    #==========================================================================
    
    def add_variables(self, arMatricesNew):
        # Old interface backward compatible
        # spoils "self"
        self = merge_variables([self] + arMatrices)
        return self

    #==========================================================================
    
    @classmethod
    def merge_variables(self, arMatrices):
        #
        # Merging multi-variables but
        # otherwise identical matrices
        #
        mold = arMatrices[0]
        nt = len(mold.times)
        nst = len(mold.stations)
        variables = set(mold.variables)

        for M in arMatrices[1:]:
            if mold.timezone !=  M.timezone:
                print('### Cannot add variables from TSM with a different time zone')
                print('Me: ', mold.variables, mold.timezone)
                print('To add: ', M.variables, M.timezone)
                raise ValueError
            if np.any(mold.times != M.times):
                print('Cannot merge variables for different times')
                print('Me times: ', mold.variables, mold.times)
                print('To add times: ', M.variables, M.times)
                raise ValueError
            if np.any(mold.stations != M.stations):
                print('Cannot merge variables for different stations')
                print('Me stations: ', mold.variables, mold.stations)
                print('To add stations: ', M.variables, M.stations)
                raise ValueError
            variables = variables.union(set(M.variables))
        
        variables = np.array(sorted(list(variables)))
        nVar = len(variables)

        vals = np.ones(shape=(nVar, nt, nst), dtype = np.float32) * np.nan
        units = np.array([''] * nVar)
        # add  amtrices
        for M in arMatrices:
            idxV = np.searchsorted(variables, M.variables)
            units[idxV] = M.units[:] ## Units always array
            if len(M.vals.shape) == 2:
                idxV = idxV[0]
            vals[idxV,:,:] = M.vals
            
        return self(mold.times, mold.stations, variables, vals, units, mold.fill_value, mold.timezone)
            
        
    #==========================================================================
    
    def subset(self, timesIn, stationsIn, variablesIn=None):
        #
        # Subsets the self down to the given set of stations and times. The input
        # can have stations/times non-existing in self, those will be ignored.
        # Note that re-sorting is not needed: we just remove some elelemts
        #
        # Pick only stations and times from self  
        if variablesIn is None:
            variablesNew = self.variables
        else:
            variablesNew = sorted(np.array(set(variablesIn).intersection(set(self.variables))))
        stationsNew = sorted(np.array(list(set(stationsIn).intersection(set(self.stations)))))
        timesNew = sorted(np.array(list(set(timesIn).intersection(set(self.times)))))
        #
        # There might be a chance for shortcuts if In dataset actually the same as self
        #
        if (len(timesNew) == len(self.times) and len(stationsNew) == len(self.stations) and 
            len(variablesNew) == len(self.variables)):
            return self
        # work has to be done. Prepare a new tsMatrix
        # Work is strongly different for single- and multi-variable cases
        if len(variablesNew) == 1:
            arTmp = np.ones(shape=(len(timesNew), len(stationsNew))) * np.nan
            # Again trying to use shortcuts...
            if len(timesNew) == len(self.times): 
                # same times, mind that they are always sorted in tsMatrices. Stations differ
                arTmp[:,:] = self.vals[:,np.searchsorted(self.stations, stationsNew)] 
            elif len(stationsNew) == len(self.stations): 
                # same stations, mind that they are always sorted in tsMatrices. Times differ
                arTmp[:,:] = self.vals[np.searchsorted(self.times,timesNew), :]
            else:
                idxSt = np.searchsorted(self.stations, stationsNew)
                idxTime = np.searchsorted(self.times, timesNew)
                for iT, idxT in enumerate(idxTime):
                    arTmp[iT,:] = self.vals[idxT,idxSt]
        else:
            arTmp = np.ones(shape=(len(variablesNew), len(timesNew), len(stationsNew))) * np.nan
            # Again trying to use shortcuts...
            if variablesNew == self.variables:  # chance for simplification
                if len(timesNew) == len(self.times): 
                    # same times, mind that they are always sorted in tsMatrices. Statoins differ
                    arTmp[:,:,:] = self.vals[:,:,np.searchsorted(self.stations, stationsNew)] 
                elif len(stationsNew) == len(self.stations): 
                    # same times, mind that they are always sorted in tsMatrices. Statoins differ
                    arTmp[:,:,:] = self.vals[:,np.searchsorted(self.times, timesNew),:]
                else:
                    idxSt = np.searchsorted(self.stations, stationsNew)
                    idxTime = np.searchsorted(self.times, timesNew)
                    for iT, idxT in enumerate(idxTime):
                        arTmp[:,iT,:] = self.vals[:,idxT,idxSt]
            else:
                # go variable by variable
                idxNewVars = np.searchsorted(self.variables, variablesNew)
                for iv, idxV in idxNewVars:
                    if len(timesNew) == len(self.times): 
                        # same times, mind that they are always sorted in tsMatrices. Statoins differ
                        arTmp[iv,:,:] = self.vals[idxV,:,np.searchsorted(self.stations, stationsNew)] 
                    elif len(stationsNew) == len(self.stations): 
                        # same times, mind that they are always sorted in tsMatrices. Statoins differ
                        arTmp[iv,:,:] = self.vals[idxV,np.searchsorted(self.times, timesNew),:]
                    else:
                        idxSt = np.searchsorted(self.stations, stationsNew)
                        idxTime = np.searchsorted(self.times, timesNew)
                        for iT, idxT in enumerate(idxTime):
                            arTmp[iv,iT,:] = self.vals[idxV,idxT,idxSt]

        return TsMatrix(timesNew, stationsNew, variablesNew, arTmp, self.units, self.fill_value, self.timezone)
            
        
    #==========================================================================

    def fromUTC_to_LocalSolarTime(self, ifCheck=True):
        #
        # reshuffles the matrix so that the times become in Local Solar Time
        # rather than UTC by default
        #
        one_hour = dt.timedelta(hours=1)
        # Stupidity check: time series must be hourly
        if ifCheck:
            if not np.all(np.array(self.times[1:]) - np.array(self.times[:-1]) == one_hour):
                print('Only hourly time series can be turned into local time. We have:')
                print(np.array(self.times[1:]) - np.array(self.times[:-1]))
                raise ValueError
        #
        # Get the shift in hours from the station locations
        shifts_hrs = np.array(list(( np.round(s.lon / 15.0) for s in self.stations)), dtype=np.int16)
        shifts_hrs[shifts_hrs > 12] -= 24   # shift is to be within [-12 : +12]
        arTmp = self.vals.copy()  # Have to use temporary array
        self.vals.fill(np.nan)  # fill the self vals with nans
        #
        # Scan stations and shift each array to the given number of steps
        if len(self.variables) == 1:
            for i in range(len(self.stations)):
                if shifts_hrs[i] > 0:
                    self.vals[shifts_hrs[i]:,i] = arTmp[:-shifts_hrs[i],i]
                elif shifts_hrs[i] < 0:
                    self.vals[:shifts_hrs[i],i] = arTmp[-shifts_hrs[i]:,i]
                else:
                    self.vals[:,i] = arTmp[:,i]
        else:
            for i in range(len(self.stations)):
                if shifts_hrs[i] > 0:
                    self.vals[:,shifts_hrs[i]:,i] = arTmp[:,:-shifts_hrs[i],i]
                elif shifts_hrs[i] < 0:
                    self.vals[:,:shifts_hrs[i],i] = arTmp[:,-shifts_hrs[i]:,i]
                else:
                    self.vals[:,:,i] = arTmp[:,:,i]

#        # Get the shift in hours from the station locations
#        shifts_hrs = np.array(list(( np.round(s.lon / 15.0) for s in self.stations)), dtype=np.int16)
#        shifts_hrs[shifts_hrs > 12] -= 24   # shift is to be within [-12 : +12]
#        idxPos = shifts_hrs >= 0   # array of true-s and false-s
#        idxNeg = shifts_hrs < 0  
#        #
#        # Shift the data series
#        arTmp = self.vals.copy()  # Have to use temporary array
#        self.vals.fill(np.nan)  # fill the self vals with nans
#        if len(self.variables) == 1:
#            self.vals[shifts_hrs:,idxPos] = arTmp[:-shifts_hrs,idxPos]
#            self.vals[:-shifts_hrs,idxNeg] = arTmp[shifts_hrs:,idxNeg]
#        else:
#            self.vals[:,shifts_hrs:,idxPos] = arTmp[:,:-shifts_hrs,idxPos]
#            self.vals[:,:-shifts_hrs,idxNeg] = arTmp[:,shifts_hrs:,idxNeg]
        # Do not forget the mark the time zone
        self.timezone = 'LST'
        return self
        
        
    #==========================================================================

    def average(self, chTargetAveraging, time_tag='end'):
        #
        # Takes the given TSM and performs an averaging as required. So-far, only some
        # averaging types are introduced.
        # DAILY averaging of all kinds is the only one implemented here
        #
        # Stupidity check
        #
        tStep = self.times[1] - self.times[0]
        if not np.all(np.array(self.times[1:]) - np.array(self.times[:-1]) == tStep):
            print('Non-constant time step:', tStep, 
                  np.array(self.times[1:]) - np.array(self.times[:-1]))
            raise ValueError
        #
        # Get the size of the averaged array, check for start-end tails of incomplete days
        # Incomplete days are removed
        #
        one_day = dt.timedelta(days=1)
        one_hour = dt.timedelta(hours=1)
        one_minute = dt.timedelta(minutes=1)

        if tStep != one_hour:
            print('For %s target averaging only hourly input allowed, not sStep=' % 
                  chTargetAveraging, tStep)
            raise ValueError
        idxStart = 0
        while(self.times[idxStart].hour != 0): idxStart += 1
#        idxAvStart = max(0, int(np.ceil(idxStart/24.0)))
        idxEnd = len(self.times)
        while(self.times[idxEnd-1].hour != 23): idxEnd -= 1
#        idxAvEnd = max(0, int(np.ceil((len(self.times) - idxEnd) / 24.0)))
        # size of averaged array
        nDays = (idxEnd - idxStart) // 24  #+ idxAvStart + idxAvEnd
        #
        # Averaged times. Note: the time tag is at the MIDDLE of the averaging interval,
        # i.e. yr-mon-day 11:30 is the daily average for the day. This is to make it same as
        # what CDO would make
        # But one can overwrite this: SILAM rather has a time tag at the end of the period,
        # whereas MMAS standard puts it to the beginning.
        #
        if chTargetAveraging in 'daymean daymin daymax daysum MDA8 M817'.split():
            if time_tag == 'middle': avTime0 = self.times[idxStart] + 12 * one_hour #+ 30 * one_minute
            elif time_tag == 'beginning': avTime0 = self.times[idxStart]
            elif time_tag == 'end': avTime0 = self.times[idxStart]  + one_day
            else: raise ValueError('TSM average: Strange time_tag: ' + time_tag)
        else:
            raise ValueError('TSM average: strange averaging type: ' + chTargetAveraging)
            
        avTimes = [ avTime0 + one_day * i for i in range(nDays) ] 

        if len(self.variables) == 1:
            nhr, nst = self.vals.shape
            valsAv_shape = (nDays, nst)
            shapex = (nDays,24, nst)
            if chTargetAveraging == 'daymean':                # From hourly to daily mean
                valsAv = np.nanmean(self.vals[idxStart:idxEnd,:].reshape(shapex),
                                    axis=1).reshape(valsAv_shape)
            elif chTargetAveraging == 'daymin':                # From hourly to daily min
                valsAv = np.nanmin(self.vals[idxStart:idxEnd,:].reshape(shapex),
                                   axis=1).reshape(valsAv_shape)
            elif chTargetAveraging == 'daymax':                # From hourly to daily max
                valsAv = np.nanmax(self.vals[idxStart:idxEnd,:].reshape(shapex),
                                   axis=1).reshape(valsAv_shape)
            elif chTargetAveraging == 'daysum':                # From hourly to daily sum
                valsAv = np.sum(self.vals[idxStart:idxEnd,:].reshape(shapex),
                                axis=1).reshape(valsAv_shape)   ### NaNs make day NaN
            elif chTargetAveraging == 'MDA8': # Daily max of 8-hour mean
                import bottleneck as bn
                arr = bn.move_mean(self.vals, window=8, axis=1, min_count=1)  ## over time
                valsAv = arr[idxStart:idxEnd,:].reshape(shapex).max(axis=1).reshape(valsAv_shape)
            elif chTargetAveraging == 'M817':  ## Mean of 8-17 
                valsAv = np.nanmean(self.vals[idxStart:idxEnd,:].reshape(shapex)[:,8:17,:],
                                    axis=1).reshape(valsAv_shape)
            else:
                print('Unknown target averagind type:', chTargetAveraging)
                raise ValueError
        else:
            nvars, nhr, nst =  self.vals.shape
            valsAv_shape = (nvars, nDays, nst)
            shapex = (nvars, nDays, 24, nst)
            if chTargetAveraging == 'daymean':                # From hourly to daily mean
                valsAv = np.nanmean(self.vals[:,idxStart:idxEnd,:].reshape(shapex),
                                    axis=2).reshape(valsAv_shape)
            elif chTargetAveraging == 'daymin':                # From hourly to daily min
                valsAv = np.nanmin(self.vals[:,idxStart:idxEnd,:].reshape(shapex),
                                   axis=2).reshape(valsAv_shape)
            elif chTargetAveraging == 'daymax':                # From hourly to daily max
                valsAv = np.nanmax(self.vals[:,idxStart:idxEnd,:].reshape(shapex),
                                   axis=2).reshape(valsAv_shape)
            elif chTargetAveraging == 'daysum':                # From hourly to daily sum
                valsAv = np.sum(self.vals[:,idxStart:idxEnd,:].reshape(shapex),
                                axis=2).reshape(valsAv_shape)   ### NaNs must make daysum == NaN
            elif chTargetAveraging == 'MDA8':                  # Daily max of 8-hour mean
                import bottleneck as bn
                arr = bn.move_mean(self.vals, window=8, axis=1, min_count=1)  ## over time
                valsAv = arr[:,idxStart:idxEnd,:].reshape(shapex).max(axis=2).reshape(valsAv_shape)
            elif chTargetAveraging == 'M817':                  # Mean of 8-17 
                valsAv = np.nanmean(self.vals[:,idxStart:idxEnd,:].reshape(shapex)[:,8:17,:],
                                    axis=2).reshape(valsAv_shape)
            else:
                print('Unknown target averagind type:', chTargetAveraging)
                raise ValueError

        return TsMatrix(avTimes, self.stations, self.variables, valsAv, self.units, 
                        self.fill_value, self.timezone)


    #==========================================================================

    def downsample_to_grid(self, gridTarget):
        #
        # Projects the tsMatrix stations to the given gridTarget and averages the
        # data of stations falling into the same grid cell. Returns new tsMatrix
        #
        # Create new stations. Strictly speaking, there is a similar function for
        # stations only but it does not return the projection rules, so we repeat it here
        # Get the coordinates and project to the target grid
        lons = np.array(list((s.lon for s in self.stations)))
        lats = np.array(list((s.lat for s in self.stations)))
        fx, fy = gridTarget.geo_to_grid(lons, lats)
        idxOK = np.logical_and(fx >=-0.5, 
                               np.logical_and(fy >= -0.5,
                                              np.logical_and(fx < gridTarget.nx - 0.5,
                                                             fy < gridTarget.ny - 0.5))) 
        arStations = np.array(self.stations)[idxOK]  # initial list of stations inside the grid
        if len(self.variables) == 1:
            valsTmp = self.vals[:,idxOK]       # speedup!
        else:
            valsTmp = self.vals[:,:,idxOK]     # speedup!
        # gridded locations (redundant initial list here but very complicated otherwise)
        ix = np.round(fx[idxOK]).astype(np.int)
        iy = np.round(fy[idxOK]).astype(np.int)
        grdLons, grdLats = gridTarget.grid_to_geo(ix, iy) # centres of the cells
        # New station codes come from cell indices
        stGrdCodesAll = np.array(list(('%04g_%04g' % (i, j) for i,j in zip(ix,iy))))
        # Compressed set of station codes
        stGrdCodesSortedUniq = np.unique(stGrdCodesAll)
        # compression rule
        idxInUniq = np.searchsorted(stGrdCodesSortedUniq, stGrdCodesAll)
        # Reserve space
        grdStat = []
        if len(self.variables) == 1:
            valsGrd = np.zeros(shape=(len(self.times), len(stGrdCodesSortedUniq)))
        else:
            valsGrd = np.zeros(shape=(len(self.variables), len(self.times), len(stGrdCodesSortedUniq)))
        # Collect the stations and data into the new sets
        for iSt, stCode in enumerate(stGrdCodesSortedUniq):
            idxSt = idxInUniq == iSt  # indices of initial stations falling into this grid cell
            grdStat.append( ST.Station(stCode, ##FIXME
                                            '_'.join(list((s.code for s in arStations[idxSt]))),
                                            grdLons[idxSt][0], grdLats[idxSt][0],
                                            np.nanmean(list((s.hgt for s in arStations[idxSt]))),
                                            '_'.join(list((s.area for s in arStations[idxSt]))),
                                            '_'.join(list((s.source for s in arStations[idxSt])))))
            if len(self.variables) == 1:
                valsGrd[:,iSt] = np.nanmean(valsTmp[:,idxSt])
            else:
                valsGrd[:,:,iSt] = np.nanmean(valsTmp[:,:,idxSt])

        return TsMatrix(self.times, grdStat, self.variables, valsGrd, self.units, 
                        self.fill_value, self.timezone)


    #==========================================================================

    def grid_mask(self, gridTarget):
        #
        # Removes stations not belonging to the given gridTarget
        #
        # Mask
        arStatInGrid = np.array(list((s.inGrid(gridTarget) for s in self.stations)))
        
        if len(self.variables) == 1:
            return (arStatInGrid,
                    TsMatrix(self.times, np.array(self.stations)[arStatInGrid], self.variables, 
                             self.vals[:,arStatInGrid], self.units, self.fill_value, self.timezone))
        else:
            return (arStatInGrid,
                    TsMatrix(self.times, np.array(self.stations)[arStatInGrid], self.variables, 
                             self.vals[:,:,arStatInGrid], self.units, self.fill_value, self.timezone))


    #==========================================================================
    
    def verify(self, chFNmIn=None, log=None):
        #
        # Just a convenience encapsuation of a generic finction.
        #
        return verify(self, chFNmIn, log)
    
    #==========================================================================

    def timestep(self):
        #
        # Convenience encapsulation of the timestep of the tsM
        #
        if len(self.times) < 2: return None
        else: return self.times[1] - self.times[0]

###############################################################################
    
def verify(tsM_, chFNmIn, log):
    #
    # Verifies the given tsMatrix, either tsM or taking it from the file. 
    # Does not generate errors, only reports the content of the matrix, and returns 
    # the number of nan-s
    # The report is written to the log file if it is provided. Printed otherwise
    #
    if chFNmIn is None:
        tsM = tsM_
    else:
        tsM = TsMatrix.fromNC(chFNmIn)

    nNan = np.sum(np.isnan(tsM.vals))
    nFinite = np.sum(np.isfinite(tsM.vals))
    vMin = np.nanmin(tsM.vals)
    vMean = np.nanmean(tsM.vals)
    vMax = np.nanmax(tsM.vals)
    vMed = np.nanmedian(tsM.vals)
    if nFinite == tsM.vals.size and np.all(np.isfinite([vMin,vMean,vMax,vMed])): 
        chValsOK = 'vals_OK'
    else: chValsOK = '###>>_suspicious_vals'
    if len(tsM.variables) == 1:
        ifDimsOK = (len(tsM.times) == tsM.vals.shape[0] and 
                    len(tsM.stations) == tsM.vals.shape[1] and
                    len(tsM.vals.shape) == 2)
    else:  
        ifDimsOK = (len(tsM.times) == tsM.vals.shape[1] and 
                    len(tsM.stations) == tsM.vals.shape[2] and
                    len(tsM.variables) == tsM.vals.shape[0] and
                    len(tsM.vals.shape) == 3)
    if ifDimsOK: chDimsOK = 'dims_OK'
    else: chDimsOK = '###>>_problematic_dimensions'
    if log is None:
        print('tsMatrix %s %s: nTimes=%g nStations=%g nVars=%g vals_dims=%s units=%s' % 
              (chDimsOK, chValsOK, len(tsM.times), len(tsM.stations), len(tsM.variables),
               str(tsM.vals.shape), ' '.join(tsM.units)) +
              ' fill_value=%g timezone=%s' % (tsM.fill_value, tsM.timezone) +
              ' n_NaN=%g n_finite=%g n_bad_rest=%g min=%g mean=%g max=%g median=%g' % 
              (nNan, nFinite, tsM.vals.size - nFinite - nNan, 
               np.nanmin(tsM.vals), np.nanmean(tsM.vals), np.nanmax(tsM.vals), np.nanmedian(tsM.vals)))
    else:
        log.log('tsMatrix %s %s: nTimes=%g nStations=%g nVars=%g vals_dims=%s units=%s' % 
              (chDimsOK, chValsOK, len(tsM.times), len(tsM.stations), len(tsM.variables),
               str(tsM.vals.shape), ' '.join(tsM.units)) +
              ' fill_value=%g timezone=%s' % (tsM.fill_value, tsM.timezone) +
              ' n_NaN=%g n_finite=%g n_bad_rest=%g min=%g mean=%g max=%g median=%g file=%s' % 
              (nNan, nFinite, tsM.vals.size - nFinite - nNan, 
               np.nanmin(tsM.vals), np.nanmean(tsM.vals), np.nanmax(tsM.vals), np.nanmedian(tsM.vals), chFNmIn))
    return ifDimsOK and chValsOK == 'vals_OK'
        
    
    
