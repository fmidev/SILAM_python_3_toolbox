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
from toolbox import stations, gridtools
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
from matplotlib import streamplot
import string


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
        # dimensions?
        if len(variables) == 1:  # just a single variable, 2D case
            (self.ntimes, self.nstations) = vals.shape
            if ((len(times),len(stations)) != vals.shape):
                print((len(times),len(stations),len(variables)), "!=", vals.shape)
                raise ValueError("TsMatrix got incompatibel-shape arrays")
            self.dimension = 2
        else:                         # must be a list or numpy array. 3D case
            (self.nvariables, self.ntimes, self.nstations) = vals.shape
            if ((len(variables),len(times),len(stations)) != vals.shape):
                print(len(variables),(len(times),len(stations)), "!=", vals.shape)
                raise ValueError("TsMatrix got incompatibel-shape arrays")
            self.dimension = 3
        if len(units) != len(variables):
            print('Variables:',variables)
            print('Units:',units)
            print('Units and variables have different lengths')
            raise ValueError("len(units) != len(variables)")
        self.times = times
        self.stations = stations
        self.variables = variables
        self.vals = vals
        self.units = units
        self.fill_value = fill_value
        self.timezone = timezone


    #==========================================================================
    def to_nc(self, filename):
        #
        # Stores the tsMatrix object to the nc file
        #
        # dimensions
        if self.dimension == 3:
            nvar,nt,nst = len(self.variables), len(self.times), len(self.stations)
        else:
            nt,nst = len(self.times), len(self.stations)
            nvar = 0
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
            if self.dimension == 3:
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
            
            if self.dimension == 3:
                # variables
                v = outf.createVariable("variable","c",("variable",), zlib=True, complevel=4)
                v.standard_name = "variable"
                v.long_name = "variable"
                v.units = ""
                # units - the size of variables
                u = outf.createVariable("unit","c",("variable",), zlib=True, complevel=4)
                u.standard_name = "unit"
                u.long_name = "variable unit"
                u.units = ""
                for iq in range(len(self.quantities)):
                    u[:] = nc4.stringtoarr( self.units[iq], strlen, dtype='S')

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
            if self.dimension == 3:
                val.units = ''
            else:
                val.units = self.units
                try: 
                    val.long_name = self.variables[0]  # this is vairable name, e.g. cnc_O3_gas
                    val.standard_name = self.variables[0]  # this is vairable name, e.g. cnc_O3_gas
                except: pass

            ## .01 absolute precision
            ###outdat = np.around(self.vals,  decimals=2)
            outdat = self.vals
            
            charTmp = np.zeros(shape=(len(self.stations), strlen), dtype='S1')
            for stvar, attname in [(stcode, 'code'), (stname, 'name'), (starea, 'area'), (stsource, 'source'),]:
                for ist,st in enumerate(self.stations):
                    charTmp[ist,:] = nc4.stringtoarr( getattr(st, attname) , strlen,dtype='S')
                stvar[:,:] = charTmp

            lon[:] = list( (st.lon for st in self.stations))
            lat[:] = list( (st.lat for st in self.stations))
            alt[:] = list( (st.hgt for st in self.stations))

            precision=500 ##Relative precision
            keepbits=np.int(np.ceil(np.log2(precision)))
            maskbits=20 -keepbits
            mask=(0xFFFFFFFF >> maskbits)<<maskbits
            b=outdat.view(dtype=np.int32)
            b &= mask
            
            # main dataset: replace nan with own fill_value
            val[:] = np.where(np.isfinite(outdat), outdat, self.fill_value)
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
            # Do we have the "variable" dimension?
            if 'variable' in nc.variables.keys():
                ifMultiVar = True
                variables = nc4.chartostring(nc.variables['variable'][:])  # array of variables / quantities
                if variables2read:
                    idxVar = np.searchsorted(variables, variables2read, sorter=np.argsort(variables))
                    if np.any(idxVar == len(variables)): # some variable does not exist in the file
                        for ivar in range(len(variables)):
                            if idxVar[ivar] == len(variables):
                                print('Requested variable does not exist: ', variables2read[ivar])
                    raise
                else:
                    idxVar = range(len(variables))
                units = nc4.chartostring(nc.variables['unit'][:])
            else:
                ifMultiVar = False
                try:
                    variables = [nc.variables['val'].standard_name]  # just a string but make a list
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
                idxStat = range(len(stlist))
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
            for i in range(len(idxStat)):
                obsStations.append(stations.Station(stcodes[i], stnames[i], stx[i], sty[i], 
                                                    height=stz[i], area_type=stareas[i], 
                                                    dominant_source=stsources[i]))
            #
            # Reading vals can be a long task, so all cuts are implemented separately
            # Selection can speed-up things very significantly, so allow separate legs too
            #
            if ifMultiVar:
                #
                # Three axes with possible selections 
                #
                if tStart or tEnd:
                    hrlistFull = netcdftime.num2date(tvals,tunits)
                    if tStart: tStart_ = tStart
                    else: tStart_ = hrlistFull[0]
                    if tEnd: tEnd_ = tEnd
                    else: tEnd_ = hrlistFull[-1]
                    selection_time = np.logical_and(np.logical_not(hrlistFull < tStart_), 
                                                    np.logical_not(hrlistFull > tEnd_))
                    hrlist = hrlistFull[selection_time]
                    try:
                        valmatr = nc.variables['val'][idxVar,:,:][:,selection_time,:].data[:,:,idxStat]
                    except:
                        valmatr = nc.variables['val'][idxVar,:,:][:,selection_time,:][:,:,idxStat]
                else:
                    # no reduction, take full file
                    hrlist = netcdftime.num2date(tvals,tunits)
                    try:
                        valmatr = nc.variables['val'][idxVar,:,:][:,:,idxStat].data
                    except:
                        valmatr = nc.variables['val'][idxVar,:,:][:,:,idxStat]
            else:
                #
                # Selection can speed-up things very significantly, so allow a separate leg
                #
                if tStart or tEnd:
                    hrlistFull = netcdftime.num2date(tvals,tunits)
                    if tStart: tStart_ = tStart
                    else: tStart_ = hrlistFull[0]
                    if tEnd: tEnd_ = tEnd
                    else: tEnd_ = hrlistFull[-1]
                    selection_time = np.logical_and(np.logical_not(hrlistFull < tStart_), 
                                                    np.logical_not(hrlistFull > tEnd_))
                    hrlist = hrlistFull[selection_time]
                    try:
                        valmatr = nc.variables['val'][selection_time,:].data[:,idxStat]
                    except:
                        valmatr = (nc.variables['val'][selection_time,:])[:,idxStat]
                else:
                    # no reduction, take full file
                    hrlist = netcdftime.num2date(tvals,tunits)
                    try:
                        valmatr = nc.variables['val'][:,idxStat].data
                    except:
                        valmatr = nc.variables['val'][:,idxStat]
            #
            # Fill value
            #
            try:  # if fill_value is explicit
                fill_v = nc.variables['val']._FillValue
                valmatr = np.where(valmatr == fill_v, np.nan, valmatr)
            except:  # have to use default
                try:
#                    print(nc4.default_fillvals)
#                    print(nc.variables['val'].dtype)
#                    print({np.dtype('float32'):'f4',
#                           np.dtype('float64'):'f8'}[nc.variables['val'].dtype])
                    flil_v = nc4.default_fillvals[{np.dtype('float32'):'f4',
                                                   np.dtype('float64'):'f8'}[nc.variables['val'].dtype]]
#                    print('default fill value for', {np.dtype('float32'):'f4',
#                           np.dtype('float64'):'f8'}[nc.variables['val'].dtype])
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
    def extract_from_fields(self, stations, times, variable, inFieldNCFNm):
        #
        # Reads the basic or processed model output - maps in netcdf format - and
        # extracts stations for the given time period or for the entire duration
        # the oberved period.
        # Returns the tsMatrix object, single-variable
        #
        print('Extracting from ', inFieldNCFNm)
        #
        # The source file
        #
        ncfile = silamfile.SilamNCFile(inFieldNCFNm)
        if ncfile.vertical is None:
            reader = ncfile.get_reader(variable, mask_mode=False)
        else:
            reader = ncfile.get_reader(variable, mask_mode=False, ind_level=0)
        #
        # Stations
        #
        stats = sorted(stations)
        stationdic={}
        stlist = []
        for st in stats:
            stationdic[st.code] = st
            stlist.append(st.code)
        nStat = len(stlist)
        stidx={}
        for i,st in enumerate(stlist): stidx[st]=i
        #
        # Times
        #
        if times is None:
            hrlist = reader.t()
        else:
            hrlist = np.array(times)
        ntimes = len(hrlist)
        timeidx = {}
        for i,t in enumerate(hrlist): 
            timeidx[t]=i
        try: timezone = ncfile.variables['time'].timezone
        except: timezone = 'UTC'
        #
        # Valmatr
        #
        valmatr= np.ones((ntimes,nStat),dtype=np.float32) * np.nan
        #
        # checklist for locations of stations
        #
        nx = ncfile.grid.nx  ### IF reading expressioon form netcdf file directly reader.nx() triggers "NotImplemented error"
        ny = ncfile.grid.ny  ## I could not figure why, but ncfile.grid seems to be a good alternative. R
        timesIn = np.array(reader.t())
        # are all needed times present in the file?
        idxTimes = np.searchsorted(timesIn, hrlist)
        idxTimes[idxTimes >= len(timesIn)] = len(timesIn)-1  
        idxTimesOK = timesIn[idxTimes] == hrlist
        if not np.all(idxTimesOK): print('Not all needed times are in the file')
        # stations
        stOK = np.array([False] * nStat)
        coords = []
        nOutside = 0
        for i in range(nStat):
            s = stationdic[stlist[i]]
            ix, iy = reader.indices(s.lon, s.lat)
            stOK[i] = ix >= 0 and ix < nx  and iy >= 0 and iy < ny
            # append only coords to extract
            if stOK[i]:
                coords.append((ix,iy))
            else:  
              #  print(s, ix, iy, ' vs ', nx, ny, 'Station', s.code, ' is outside the domain')
                nOutside += 1
        print('Stations outside the domain:', nOutside)
       # exit

        # find the x and y indices (z assumed 0)
        if3d = len(reader.z()) > 1
        # Is times big? Do we report the progress?
        if ntimes > 10000: counter = 1
        else: counter = None

        selst = np.array(stOK)

        #turn array of tuples into tuple of arrays
        (ixs,iys) = zip(*coords)
        coords = (np.array(ixs),np.array(iys))

        for it, t in enumerate(hrlist):  #range(ntimes):
            if counter:
                if counter % 5000 == 0: print('Reading: ' + str(hrlist[it]))
                counter += 1
            if idxTimesOK[it]:
                reader.seek_abs(idxTimes[it])   #goto(t)     #hrlist[it])
                field = reader.read(1) ## Reader might return strange things
                if isinstance(field, np.ndarray):
                    valmatr[it,selst] = field[coords]
                elif isinstance(field, np.ma.MaskedArray):
                    valmatr[it,selst] = field.data[coords]
                else:
                    raise TypeError("Reader returned %s"%(type(field)))
            else:
                print('Time ', hrlist[it], ' is not in the input nc file')

        # whatever fill value is in the input file, here it is nan
        valmatr = np.where(valmatr == ncfile.fill_value(variable), np.nan, valmatr)
        if np.sum(np.logical_not(np.isfinite(valmatr))) > 0:
            print('TSM Bad values:', np.sum(np.logical_not(np.isfinite(valmatr))))

        it=(np.any(np.isfinite(valmatr),axis=1))
        print("%d valid time steps, %g stations" % (np.sum(it), len(stats)))
        # note that reader is projected readers, i.e. the actual reader is wrapped
#        print(reader)
#        try: print(reader.wrapped.units)
#        except: print('failed reader.wrapped.units')
#        try: print(reader.wrapped.vars.units)
#        except: print('failed reader.wrapped.vars.units')
        try:
            tsmatr = self(hrlist, stats, [variable], valmatr, [reader.wrapped.ncvar.units],
                          -999999, timezone)
        except:
            try:
                tsmatr = self(hrlist, stats, [variable], valmatr, [reader.wrapped.vars[0].units],
                              -999999, timezone)
            except:
                print("Unknown units in %s" % (inFieldNCFNm))
                tsmatr = self(hrlist, stats, [variable], valmatr, ['unknown'], -999999, timezone)
    
        return tsmatr


    #==========================================================================
    
    def fromTimeSeries(self, tseries, unit):
        # convert timeseries objects to tsMatrix objects
        stations = [(stations.station for ts in tseries)]
        valmatr= np.float32(np.nan) * np.empty((len(tseries.times()), len(stations)), 
                                               dtype=np.float32)
        for iSt in range(len(stations)):
            valmatr[:,iSt] = tseries[iSt].data()
        return self(tseries.times(), stations, [tseries.quantity], valmatr, unit, timezone='UTC')


    #==========================================================================

    def to_map(self, grid, fill_value, outFNm):
        # converts the tsMatrix to a map and stores into a file
        # open the output file
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
        #
        # Extends the current matrix with the additional one
        # Can do it for both statoin dimension and for time dimension
        # Overlapping times / stations are not added. Sorting of times is
        # verified, sorting of stations is not.
        #
        if self.timezone !=  tsMatrToAdd.timezone:
            print('### Cannot add variables from TSM with a different time zone')
            raise ValueError
        #
        # Build new dimensions
        #
        timesNew = sorted(list(set(self.times).union(set(tsMatrToAdd.times))))   # new time dimension
        stationsNew = sorted(list(set(self.stations).union(set(tsMatrToAdd.stations)))) # new stations
        print('timesNes ', len(timesNew), ',  stationsNew = ', len(stationsNew))
        if np.any(self.variables != tsMatrToAdd.variables):
            print('join is only allowed for same set of variables')
            raise ValueError
        #
        # Save own data to a temporary and define new size of the array
        # resize does not work here
        #
        arTmp = self.vals.copy()  # temporary
        if len(self.variables) == 1:
            self.vals = np.ones(shape=(len(timesNew), len(stationsNew))) * np.nan
        else:
            self.vals = np.ones(shape=(len(self.variables), len(timesNew), len(stationsNew))) * np.nan
        #
        # get indices of the own stations and own times in the new lists
        # To it sequentially for both matrices. Note that if they overlap, the 
        # itnersection will be written twice and the tsMatrix to add will overwrite
        # the initial values
        #
        # self tsMatrix to new vals
        idxStatInNew = np.searchsorted(stationsNew, self.stations)
        idxTimesInNew = np.searchsorted(timesNew, self.times)
        if len(self.variables) == 1:
            for iTime in range(len(idxTimesInNew)):
                self.vals[idxTimesInNew[iTime], idxStatInNew] = arTmp[iTime,:]  # own values are back
        else:
            for iTime in range(len(idxTimesInNew)):
                self.vals[:,idxTimesInNew[iTime], idxStatInNew] = arTmp[:,iTime,:]  # own values are back
        # tsMatrix to add to the new vals
        idxStatInNew = np.searchsorted(stationsNew, tsMatrToAdd.stations)
        idxTimesInNew = np.searchsorted(timesNew, tsMatrToAdd.times)
        if len(self.variables) == 1:
            for iTime in range(len(idxTimesInNew)):
                self.vals[idxTimesInNew[iTime], idxStatInNew] = tsMatrToAdd.vals[iTime,:]  # new values added
        else:
            for iTime in range(len(idxTimesInNew)):
                self.vals[:,idxTimesInNew[iTime], idxStatInNew] = tsMatrToAdd.vals[:,iTime,:]  # new values added
        self.times = timesNew.copy()
        self.stations = stationsNew.copy()
            
        return self        

 
    #==========================================================================
    
    def add_variables(self, arMatricesNew):
        #
        # A mirroring way of joining the matrices: merging multi-variables but
        # otherwise identical matrices
        #
        variablesNew = set(self.variables.copy())
        for M in arMatricesNew:
            if self.timezone !=  M.timezone:
                print('### Cannot add variables from TSM with a different time zone')
                print('Me: ', self.variables, self.timezone)
                print('To add: ', M.variables, M.timezone)
                raise ValueError
            if np.any(self.times != M.times):
                print('Cannot merge variables for different times')
                print('Me times: ', self.variables, self.times)
                print('To add times: ', M.variables, M.times)
                raise ValueError
            if np.any(self.stations != M.stations):
                print('Cannot merge variables for different stations')
                print('Me stations: ', self.variables, self.stations)
                print('To add stations: ', M.variables, M.stations)
                raise ValueError
            variablesNew = variablesNew.union(set(M.variables))
        if len(variablesNew) == 1: return  # nothing to do
        
        variablesNew = np.array(sorted(list(variablesNew)))
        # start merging the data
        arTmp = self.vals.copy()  # temporary
        U = self.units.copy()
        self.vals = np.ones(shape=(len(variablesNew),len(self.times),len(self.stations))) * np.nan
        self.units = [''] * len(variablesNew)
        # put own dat back
        idxV = np.searchsorted(variablesNew, self.variables)
        if len(self.variables) == 1:
            self.vals[idxV[0],:,:] = arTmp[:,:]
            self.units[idxV[0]] = U[0]
        else:
            self.vals[idxV,:,:] = arTmp[:,:,:]
            self.units[idxV] = U[:]
        # add other amtrices
        for M in arMatricesNew:
            idxV = np.searchsorted(variablesNew, M.variables)
            if len(M.variables) == 1:
                self.vals[idxV[0],:,:] = M.vals[:,:]
                self.units[idxV[0]] = U[0]
            else:
                self.vals[idxV,:,:] = M.vals[:,:,:]
                self.units[idxV] = U[:]
            
        self.variables = variablesNew
        return self
            
        
    #==========================================================================
    
    def intercept(self, timesIn, stationsIn, variablesIn=None):
        #
        # Cuts the self down to the given set of statoins and times
        # The result has the intercept of stations and times
        # Note that resorting is not needed: we just remove some elelemts
        #
        if variablesIn is None:
            variablesNew = self.variables
        else:
            variablesNew = sorted(np.array(set(variablesIn).intersection(set(self.variables))))
        stationsNew = sorted(np.array(set(stationsIn).intersection(set(self.stations))))
        timesNew = sorted(np.array(set(timesIn).intersection(set(self.times))))
        #
        # There might be a chance for shortcuts if In dataset actually the same as self
        #
        if (len(timesNew) == len(self.times) and len(stationsNew) == len(self.stations) and 
            len(variablesNew) == len(self.variables)):
            return self
        # work has to be done. Save and redefine own array
        arTmp = self.vals.copy()  # temporary
        # Work is strongly different for single- andmulti-variable cases
        if len(variablesNew) == 1:
            self.vals = np.ones(shape=(len(timesNew), len(stationsNew))) + np.nan
            # Again trying to use shortcuts...
            if len(timesNew) == len(self.times): 
                # same times, mind that they are always sorted in tsMatrices. Statoins differ
                self.vals[:,:] = arTmp[:,np.searchsorted(stationsNew, self.stations) < len(stationsNew)] 
            elif len(stationsNew) == len(self.stations): 
                # same times, mind that they are always sorted in tsMatrices. Statoins differ
                self.vals[:,:] = arTmp[np.searchsorted(timesNew, self.times) < len(timesNew), :]
            else:
                istOK = np.searchsorted(stationsNew, self.stations) < len(stationsNew)
                itimeOK = np.searchsorted(timesNew, self.times) < len(timesNew)
                itOut = 0
                for iT in range(len(timesNew)):
                    if itimeOK[iT]:
                        self.vals[itOut,:] = arTmp[iT,istOK]
                        iiOut += 1
        else:
            self.vals = np.ones(shape=(len(variablesNew), len(timesNew), len(stationsNew))) * np.nan
            # Again trying to use shortcuts...
            if variablesNew == self.variables:  # chance for simplification
                if len(timesNew) == len(self.times): 
                    # same times, mind that they are always sorted in tsMatrices. Statoins differ
                    self.vals[:,:,:] = arTmp[:,:,np.searchsorted(stationsNew, self.stations) < len(stationsNew)] 
                elif len(stationsNew) == len(self.stations): 
                    # same times, mind that they are always sorted in tsMatrices. Statoins differ
                    self.vals[:,:,:] = arTmp[:,np.searchsorted(timesNew, self.times) < len(timesNew),:]
                else:
                    istOK = np.searchsorted(stationsNew, self.stations) < len(stationsNew)
                    itimeOK = np.searchsorted(timesNew, self.times) < len(timesNew)
                    itOut = 0
                    for iT in range(len(timesNew)):
                        if itimeOK[iT]:
                            self.vals[:,itOut,:] = arTmp[:,iT,istOK]
                            iiOut += 1
            else:
                idxNewVars = np.searchsorted(self.variables, variablesNew)
                for iv in idxNewVars:
                    if len(timesNew) == len(self.times): 
                        # same times, mind that they are always sorted in tsMatrices. Statoins differ
                        self.vals[iv,:,:] = arTmp[iv,:,np.searchsorted(stationsNew, 
                                                                       self.stations) < len(stationsNew)]
                    elif len(stationsNew) == len(self.stations):
                        # same times, mind that they are always sorted in tsMatrices. Statoins differ
                        self.vals[iv,:,:] = arTmp[iv, np.searchsorted(timesNew, self.times) < len(timesNew), :]
                    else:
                        istOK = np.searchsorted(stationsNew, self.stations) < len(stationsNew)
                        itimeOK = np.searchsorted(timesNew, self.times) < len(timesNew)
                        itOut = 0
                        for iT in range(len(timesNew)):
                            if itimeOK[iT]:
                                self.vals[iv,itOut,:] = arTmp[iv,iT,istOK]
                                iiOut += 1
        return self
            
        
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

    def average(self, chTargetAveraging):
        #
        # Takes the given TSM and performs an averaging as required. So-far, only some
        # averaging types are introduced.
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
        # i.e. yr-mon-day 11:30 is the daily average for the day. This is to make it same
        # CDO would make
        #
        
        avTime0 = self.times[idxStart]  + 11 * one_hour + 30 * one_minute 
        avTimes = [ avTime0 + one_day * i for i in range(nDays) ] 

        if len(self.variables) == 1:
            nhr, nst = self.vals.shape
            valsAv_shape = (nDays, nst)
            shapex = (nDays,24, nst)
            avax = 1
        else:
            nvars, nhr, nst =  self.vals.shape
            valsAv_shape = (nvars, nDays, nst)
            shapex = ( nvars, nDays, 24, nst)
            avax = 2

        #
        # Do the required averaging
        #
        if chTargetAveraging == 'daymean':
            # From hourly to daily mean
            valsAv = np.nanmean(self.vals[idxStart:idxEnd].reshape(shapex),   axis=avax).reshape(valsAv_shape)
        elif chTargetAveraging == 'daymin':
            # From hourly to daily min
            valsAv = np.nanmin(self.vals[idxStart:idxEnd].reshape(shapex), axis=avax).reshape(valsAv_shape)
        elif chTargetAveraging == 'daymax':
            # From hourly to daily max
            valsAv = np.nanmax(self.vals[idxStart:idxEnd].reshape(shapex), axis=avax).reshape(valsAv_shape)
        elif chTargetAveraging == 'daysum':
            # From hourly to daily sum
            valsAv = self.vals[idxStart:idxEnd].reshape(shapex).sum(axis=avax).reshape(valsAv_shape)   ### NaNs make day NaN
        elif chTargetAveraging == 'MDA8': # Daily max of 8-hour mean
            import bottleneck as bn
            arr = bn.move_mean(self.vals, window=8, axis=(avax-1), min_count=1)  ## over time
            valsAv = arr[idxStart:idxEnd].reshape(shapex).max(axis=avax).reshape(valsAv_shape)
        elif chTargetAveraging == 'M817':  ## Mean of 8-17 
            if avax == 1:
                valsAv = np.nanmean( self.vals[idxStart:idxEnd].reshape(shapex)[:,8:17,:], axis=avax).reshape(valsAv_shape)
            else:
                valsAv = np.nanmean( self.vals[idxStart:idxEnd].reshape(shapex)[:,:,8:17,:], axis=avax).reshape(valsAv_shape)
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
            grdStat.append(stations.Station(stCode,
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


    #============================================================================
    
    @classmethod
    def verify(self, chFNmIn, log):
        #
        # Verifies the given tsMatrix, either self or taking it from the file. 
        # Does not generate errors, only reports the content of the matrix, and returns 
        # the number of nan-s
        # The report is written to the log file if it is provided. Printed otherwise
        #
        if chFNmIn is None:
            nNan = np.sum(np.isnan(self.vals))
            nFinite = np.sum(np.isfinite(self.vals))
            vMin = np.nanmin(self.vals)
            vMean = np.nanmean(self.vals)
            vMax = np.nanmax(self.vals)
            vMed = np.nanmedian(self.vals)
            if len(self.variables) == 1:
                ifDimsOK = (self.times.size == self.vals.shape[0] and 
                            len(self.stations) == self.vals.shape[1] and
                            len(self.vals.shape) == 2)
            else:  
                ifDimsOK = (self.times.size == self.vals.shape[1] and 
                            len(self.stations) == self.vals.shape[2] and
                            len(self.variables) == self.vals.shape[0] and
                            len(self.vals.shape) == 3)
            if ifDimsOK: chDimsOK = 'dims_OK'
            else: chDimsOK = '###>>_problematic_dimensions'
            if nFinite == self.vals.size and np.all(np.isfinite([vMin,vMean,vMax,vMed])):
                chValsOK = 'vals_OK'
            else: chValsOK = '###>>_suspicious_vals'
             
            if log is None:
                print('tsMatrix %s %s: nTimes=%g nStations=%g nVars=%g vals_dims=%s units=%s' % 
                      (chDimsOK, chValsOK, self.times.size, len(self.stations), len(self.variables),
                       str(self.vals.shape), ' '.join(self.units)) +
                      ' fill_value=%g timezone=%s' % (self.fill_value, self.timezone) +
                      ' n_NaN=%g n_finite=%g n_bad_rest=%g min=%g mean=%g max=%g median=%g' % 
                      (nNan, nFinite, self.vals.size - nFinite - nNan, 
                       np.nanmin(self.vals), np.nanmean(self.vals), np.nanmax(self.vals), np.nanmedian(self.vals)))
            else:
                log.log('tsMatrix %s %s: nTimes=%g nStations=%g nVars=%g vals_dims=%s units=%s' % 
                      (chDimsOK, chValsOK, self.times.size, len(self.stations), len(self.variables),
                       str(self.vals.shape), ' '.join(self.units)) +
                      ' fill_value=%g timezone=%s' % (tsM.fill_value, tsM.timezone) +
                      ' n_NaN=%g n_finite=%g n_bad_rest=%g min=%g mean=%g max=%g median=%g' % 
                      (nNan, nFinite, self.vals.size - nFinite - nNan, 
                       np.nanmin(self.vals), np.nanmean(self.vals), np.nanmax(self.vals), np.nanmedian(self.vals)))
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
                ifDimsOK = (tsM.times.size == tsM.vals.shape[0] and 
                            len(tsM.stations) == tsM.vals.shape[1] and
                            len(tsM.vals.shape) == 2)
            else:  
                ifDimsOK = (tsM.times.size == tsM.vals.shape[1] and 
                            len(tsM.stations) == tsM.vals.shape[2] and
                            len(tsM.variables) == tsM.vals.shape[0] and
                            len(tsM.vals.shape) == 3)
            if ifDimsOK: chDimsOK = 'dims_OK'
            else: chDimsOK = '###>>_problematic_dimensions'
            if log is None:
                print('tsMatrix %s %s: nTimes=%g nStations=%g nVars=%g vals_dims=%s units=%s' % 
                      (chDimsOK, chValsOK, tsM.times.size, len(tsM.stations), len(tsM.variables),
                       str(tsM.vals.shape), ' '.join(tsM.units)) +
                      ' fill_value=%g timezone=%s' % (tsM.fill_value, tsM.timezone) +
                      ' n_NaN=%g n_finite=%g n_bad_rest=%g min=%g mean=%g max=%g median=%g' % 
                      (nNan, nFinite, tsM.vals.size - nFinite - nNan, 
                       np.nanmin(tsM.vals), np.nanmean(tsM.vals), np.nanmax(tsM.vals), np.nanmedian(tsM.vals)))
            else:
                log.log('tsMatrix %s %s: nTimes=%g nStations=%g nVars=%g vals_dims=%s units=%s' % 
                      (chDimsOK, chValsOK, tsM.times.size, len(tsM.stations), len(tsM.variables),
                       str(tsM.vals.shape), ' '.join(tsM.units)) +
                      ' fill_value=%g timezone=%s' % (tsM.fill_value, tsM.timezone) +
                      ' n_NaN=%g n_finite=%g n_bad_rest=%g min=%g mean=%g max=%g median=%g file=%s' % 
                      (nNan, nFinite, tsM.vals.size - nFinite - nNan, 
                       np.nanmin(tsM.vals), np.nanmean(tsM.vals), np.nanmax(tsM.vals), np.nanmedian(tsM.vals), chFNmIn))
        return ifDimsOK and chValsOK == 'vals_OK'
            
        
        
