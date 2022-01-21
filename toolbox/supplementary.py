# -*- coding: ISO-8859-1 -*-
'''
A few useful subs for the SFM and others

Created on 15.4.2017
@author: sofievm
'''

import os, sys, datetime as dt
import types
import time, glob
import numpy as np
from matplotlib import pyplot as plt
import netCDF4 as nc4
from toolbox import gridtools

#
# MPI may be tricky: in puhti it loads fine but does not work
# Therefore, first check for slurm loader environment. Use it if available
#
try:
    # slurm loader environment
    mpirank = int(os.getenv("SLURM_PROCID",None))
    mpisize = int(os.getenv("SLURM_NTASKS",None))
    chMPI = '_mpi%03g' % mpirank
    comm = None
    print('SLURM pseudo-MPI activated', chMPI, 'tasks: ', mpisize)
except:
    # not in pihti - try usual way
    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        mpisize = comm.size
        mpirank = comm.Get_rank()
        chMPI = '_mpi%03g' % mpirank
        print ('MPI operation, mpisize=', mpisize, chMPI)
    except:
        print ("mpi4py failed, single-process operation")
        mpisize = 1
        mpirank = 0
        chMPI = ''
        comm = None

###########################################################################

one_day = dt.timedelta(days=1)
one_hour = dt.timedelta(hours=1)
one_minute = dt.timedelta(minutes=1)
zero_interval = dt.timedelta(hours=0)

radians_2_degrees =  57.29577951
degrees_2_radians =  0.01745329252


################################################################# 

def solar_zenith_angle(lon_deg, lat_deg, julian_day, hour, minutes):
    #
    # Computes the solar zenith angle from basic parameters and UTC time.
    # To account for the local time, we use longitude
    #
    # Declination of the sun
    #
    d = 23.45 * np.pi / 180. * np.sin(2. * np.pi * (284. + julian_day) / 365.)
    #
    # Equation of time in minutes
    # http://solardat.uoregon.edu/SolarRadiationBasics.html
    #
    Eqt = np.where(julian_day < 106,
                   -14.2 * np.sin(np.pi * (julian_day + 7.) / 111.),
                   np.where(julian_day < 166,
                            4.0 * np.sin(np.pi * (julian_day - 106.) / 59.),
                            np.where(julian_day < 246,
                                     -6.5 * np.sin(np.pi * (julian_day - 166.) / 80.),
                                     16.4 * np.sin(np.pi * (julian_day - 247.) / 113.))))
    #
    # Solar time in hours. Longsm -s the longitude of the "standard meridian" for the given time zone,
    # while longLocal is the actual longitude needed. The differrence is then less than 1 hour.
    # If we count from Greenwich, Longsm=0, of course
    #
    Tsolar = hour + (minutes+Eqt) / 60. + lon_deg / 15.
    #
    # Hour angle is:
    #
    w = np.pi * (12. - Tsolar) / 12.  # in radians
    #
    # Cosine of zenith angle
    #
    return np.arccos(np.sin(lat_deg * degrees_2_radians) * np.sin(d) + 
                   np.cos(lat_deg * degrees_2_radians) * np.cos(d) * np.cos(w)) * radians_2_degrees


#########################################################################################

# Write a text message to the log file and print it to the screen
# Need a class encapsulating the log file and making it pickle-compatible
#
class log:
    def __init__(self, fInLog='run.log', wrkDir='', default_template=''):  # Checks the log file handler, creates if needed
        if isinstance(fInLog, str):
            if default_template == '':
                self.logfile = open(os.path.join(wrkDir, fInLog),'w')
            else:
                self.logfile = open(os.path.join(wrkDir, dt.datetime.now().strftime(default_template)),'w')
        elif isinstance(fInLog, log):   # an instance of this very log class
            self.logfile = fInLog.logfile
        else:
            print('Strange fInType, neither file name nor log object:', fInLog, type(fInLog))
            sys.exit()
        self.filename = self.logfile.name
        self.logOK = True
    
    def log(self, msg):
        self.logfile.write(msg + '\n')
        print (msg)
        
    def error(self, msg):
        self.logfile.write('#### ERROR. ' + msg + '\n')
        print (msg)
        
    def write(self, msg):
        self.logfile.write(msg + '\n')
        
    def get_fname(self):
        return self.logfile.name
        
    def __getstate__(self):
        state = self.__dict__.copy()   # save the self status
        if 'logfile' in state.keys():
            del state['logfile']           # eliminate the unpicklable file
            state['logOK'] = False
        return state
    
    def __setstate__(self, state):
#        try:
#            if self.logOK: return
#        except: pass
        self.__dict__.update(state)   # update everything except for file instance
#        ifOK = False
#        if os.path.exists(self.filename):
#            try:
#                newfile = open(self.filename, 'a')    # open the log file in append mode
#                ifOK = True
#            except: pass
#        if not ifOK:
#            try:
#               newfile = open(self.filename, 'w')  # open the log file in append mode
#            except: pass
#        if not ifOK:
#            try:
#                print 'Failed to reopen the log file, start new in the current directory: ', os.getcwd()
#                newfile = open(os.path.basename(self.filename), 'w')
#                newfile.write('Failed to reopen the log file, start new in the current directory %s\n' % os.getcwd())
#            except:
#                print 'Failed to open any log file. You must reopen it'
#                return
#        self.logfile = newfile                   # store the file handler.
#        self.filename = self.logfile.name
#        self.logOK = True
    
    def changelog(self, newLog, wrkDir='', default_template =''):
        try: self.logfile.close()   # who knows, what this file status will be...
        except: pass
        if type(newLog) == types.StringType:
            if default_template == '':
                self.logfile = open(os.path.join(wrkDir, newLog),'w')
            else:
                self.logfile = open(os.path.join(wrkDir, dt.datetime.now().strftime(default_template)),'w')
        elif type(newLog) == types.InstanceType:
            self.logfile = newLog.logfile
        else:
            print ('Strange fInType, neither file name nor log object:', newLog, type(newLog))
            sys.exit()
        self.filename = self.logfile.name
        
    def close(self):
        self.logfile.close()

################################################
#
# Reports fatal error and stops the program
#
def fatal(log, msg, chPlace):
    log.log('############## FATAL error in %s ###########' % chPlace)
    log.log(msg)
    sys.exit()


# Check the condition, if false, break the run
#
def check(log, condition_true, msg, place, ifSoft = False):
    if not condition_true:
        if ifSoft:
            log.log(msg + ', place= ' + place)
        else:
            fatal(log, msg, place)
    return condition_true

# Elements of Grads-type template
#
def process_template(strTempl, day, dayAn=None):  # valid time and analysis time
    keysAn = '%ay4 %am2 %ad2 %ah2 %f2 %f3'.split()
    keys_vals = [('%y4','%04i' % day.year),
                 ('%m2','%02i' % day.month),
                 ('%d2','%02i' % day.day),
                 ('%h2','%02i' % day.hour)]
    if dayAn is None:
        if any((kAn in strTempl for kAn in keysAn)): 
            fatal('No analysis time but required for: ' + strTempl,'spp_process_template')
    else:
        keys_vals += [('%ay4','%04i' % dayAn.year),
                      ('%am2','%02i' % dayAn.month),
                      ('%ad2','%02i' % dayAn.day),
                      ('%ah2','%02i' % dayAn.hour),
                      ('%f2','%02i' % int((day-dayAn).seconds/3600.0 + (day-dayAn).days*24.0)),
                      ('%f3','%03i' % int((day-dayAn).seconds/3600.0 + (day-dayAn).days*24.0))]
    strOut = strTempl
    for keyVal in keys_vals:
        if keyVal[0] in strTempl:
            strOut = strOut.replace(keyVal[0], keyVal[1]) 
    return strOut

#
# A multiple replacement function for a string. May be not too elegant and may depend on 
# order of replacements but functional enough for majority of cases
#
def multireplace(text, dicReplacements):
    for item, repl in dicReplacements.items():
        text = text.replace(item, repl)
    return text

##############################################################################
#
# Class SFM_timer 
#
##############################################################################

class SFK_timer:
    def __init__(self):
        self.tStart = {'zz_overall_time': time.time()} # internal label, nobody else uses it
        self.timers = {}
        return
    
    def start_timer(self, chTimerName):
        if chTimerName in self.tStart.keys():
            print('Timer ' + chTimerName + ' has already been started', 'SFM-timer__start_timer')
        self.tStart[chTimerName] = time.time()
        return True
    
    def stop_timer(self, chTimerName):
        if not chTimerName in self.tStart.keys():
            print('Timer ' + chTimerName + ' has not been started', 'SFM-timer__stop_timer')
        if chTimerName in self.timers.keys():
            self.timers[chTimerName] += time.time() - self.tStart[chTimerName]    
        else:
            self.timers[chTimerName] = time.time() - self.tStart[chTimerName]
        self.tStart.pop(chTimerName)       # timer can be stopped only once
        return self.timers[chTimerName]
    
    def report_timers(self, fOut=None, chTimerName=None):   # if timer name is not given, all will be reported
        if chTimerName is None:
            ifStopZZ = 'zz_overall_time' in self.tStart.keys() 
            if ifStopZZ: self.stop_timer('zz_overall_time')  # internal label, nobody else uses it
            chTimerName = sorted(self.timers.keys())
        else:
            if not chTimerName in self.timers.keys():
               print('Timer ' + chTimerName + ' has not been started', 'SFM-timer__report_timers')
               return
            chTimerName = [chTimerName]   # turn it to array
        for chT in chTimerName:
            strOut = 'Timer ' + chT
            if self.timers[chT] > 86400: strOut += ' %i day' % int(self.timers[chT] / 86400)
            if self.timers[chT] > 3600: strOut += ' %i hr' % np.mod(int(self.timers[chT] / 3600), 24)
            if self.timers[chT] > 60: strOut += ' %i min' % np.mod(int(self.timers[chT] / 60), 60)
            strOut += ' %f sec' % np.mod(self.timers[chT],60)
            if fOut: fOut.write(strOut + '\n')
            print(strOut)
        if ifStopZZ: self.start_timer('zz_overall_time')
        return


#======================================================================

def decode_geogr_coord(chIn):
    #
    # Reads geographical coordinates from the striong like  40°12'44.3"N  or  5°05'03.1"W   
    #
    deg = float(chIn.split('°')[0])
    try: min = float(chIn.split('°')[1].split("'")[0])
    except: min = 0.
    try: sec = float(chIn.split('°')[1].split("'")[1].split('"')[0])
    except: sec =0.
    dir = {'N':1,'S':-1,'E':1,'W':-1}[chIn.strip()[-1]]
    return dir * (deg + min / 60.0 + sec / 3600.0)
    
    
#==============================================================================

def find_file(chDir, chFNmTempl, chDateMark, date2find):
    #
    # Finds the file using the template given
    #
    chMask = chFNmTempl.replace(chDateMark,'*')
    files = glob.glob(os.path.join(chDir,chMask))
    print ("Searching in:", chDir, ", files: ", chMask)
    for i in range(10):
        dayTmp = date2find - dt.timedelta(days=i)
        print ('Checking: ', dayTmp)
        if os.path.join(chDir, dayTmp.strftime(chFNmTempl)) in files:
            print ('Found file: ', dayTmp.strftime(chFNmTempl))
            return os.path.join(chDir, dayTmp.strftime(chFNmTempl))
    print ('Failed to find the file for template: ', chFNmTempl)
    return ''




#====================================================================================

def points_to_field_corrad(grid_to,     # map grid that needs to be made 
                           lons_from, lats_from, vals_from,   # dots, from which map is made 
                           distance_method, corrad,   # rules for  
                           weight_from=None):
    #
    # Makes a field out of a bunch of points based on correlation radius
    #
    gridY, gridX = np.meshgrid(grid_to.y(), grid_to.x())
    outMap = np.zeros(shape=(grid_to.nx, grid_to.ny), dtype=np.float32)
    norm = np.zeros(shape=(grid_to.nx, grid_to.ny), dtype=np.float32)
    #
    # Make the output map by collecting the impact of all point values one-by-one 
    #
    for iP in range(len(vals_from)):
        dist = gridtools.gc_distance(gridX, lons_from[iP], gridY, lats_from[iP])
        if distance_method == 'exp':
            fWeight = np.exp(-1.0 * dist / corrad)
        elif distance_method == 'inv_dist':
            fWeight = 1.0 / (dist + corrad)
        elif distance_method == 'inv_dist_4':
            fWeight = 1.0 / (dist + corrad)**4
        else:
            print('Unknown mapping method. Allowed are exp / inv_dist / inv_dist_4')
            return None
        if weight_from is not None:
            fWeight *= weight_from
        outMap += vals_from[iP] * fWeight
        norm += fWeight
    return outMap / (norm + 1e-30)
#
#    #
#    # Draw the creature
#    #
#    bmap = Basemap(projection='cyl', 
#                   llcrnrlon=grid.x()[0], urcrnrlat=grid.y()[-1],
#                   urcrnrlon=grid.x()[-1], llcrnrlat=grid.y[0], resolution='l')
#    fOut = os.path.join(wrkDir,'scaling',
#                        'scaling_' + gridType + '_' + str(iteration) + '_' + 'corr_%04d' % corrD + '_' + dist_method + chAccN + '_' + subst + '.png')
#    print fOut
#    figs.clevs_pcolor(lon, lat, scaling.T, clevs, cmap=pylab.cm.jet)
#    figs.finishMap(bmap, dx=10.0, dy=10.0)
#    pylab.title('Scaling: ' + gridType + ', iter=' + str(iteration) + ', corr=' + str(corrD) + 'km, ' + dist_method + ' ' + subst)
#    figs.clevs_colorbar(clevs,orientation='vertical',format='%.4f') 
#    pylab.savefig(fOut)
#    pylab.clf()


#====================================================================================

def cross_correlation(A1d, B1d, max_shift):
    #
    # Computes cross-correlation of the two 1D arrays, up to 1/3 of their length
    #
    cc_span = min(max_shift, int(len(A1d) / 3.0))
    cc = np.ones(shape = (2, 1 + cc_span * 2))* np.nan
#    print(list(range(-cc_span, cc_span)))
    cc[0,:] = range(-cc_span, cc_span+1)
    lenB =len(B1d)
    for shift in range(cc_span+1):
#        print(np.nanmean(A1d[shift:] * B1d[:lenB-shift]))
#        print(np.nanmean(A1d[shift:] * np.nanmean(B1d[:lenB-shift]))) 
#        print(np.nanstd(A1d[shift:]))
#        print(np.nanstd(B1d[:lenB-shift]))
        Ashift = A1d[shift:]
        Bshift = B1d[:lenB-shift]
        idxOK = np.logical_and(np.isfinite(Ashift), np.isfinite(Bshift))
        if not np.any(idxOK): continue
        Astd = np.std(Ashift[idxOK])
        Bstd = np.std(Bshift[idxOK])
        if Astd * Bstd > 0.0:
            cc[1,cc_span-shift] = ((np.mean(Ashift[idxOK] * Bshift[idxOK]) - 
                                    np.mean(Ashift[idxOK]) * np.mean(Bshift[idxOK])) / 
                                   (Astd * Bstd))
        else:
            cc[1,cc_span-shift] = 0.0
        Ashift = A1d[:lenB-shift]
        Bshift = B1d[shift:]
        idxOK = np.logical_and(np.isfinite(Ashift), np.isfinite(Bshift))
        cc[1,cc_span+shift] = ((np.mean(Ashift[idxOK] * Bshift[idxOK]) - 
                                np.mean(Ashift[idxOK]) * np.mean(Bshift[idxOK])) / 
                               (np.std(Ashift[idxOK]) * np.std(Bshift[idxOK])+1e-10))
    return cc
    

#====================================================================================

def nanCorrCoef(A1d, B1d, indOK=None):
    #
    # Computes the correlation coefficient allowing for nan-s and trying to resove
    # division by zero.
    # procedure: 
    # - first, try np.corrcoef - if OK, just return
    # If not:
    # - going step by step with all statistics nan-robust
    # - first, compute variances. if zero, return zero correlation coefficient
    # - if variances != 0 comptue covariance and divide
    #
#    print(A1d)
#    print(B1d) 
#    if np.logical_and(np.any(A1d != 0),np.any(B1d != 0)):
#        try: 
#            if indOK is None:
#                cc = np.corrcoef(A1d.flatten(), B1d.flatten())[0,1]
#            else:
#                cc = np.corrcoef(A1d.flatten()[indOK], B1d.flatten()[indOK])[0,1]
#            if np.isfinite(cc):
#                return cc
#            else:
#                print('non-finite standard correlation. Arrays are:')
#                print(A1d)
#                print(B1d) 
#        except:
#            print('problem with standard correlation. Arrays are:')
#            print(A1d)
#            print(B1d) 
#            pass
#    print('trying better way')

    if indOK is None: indOK = np.isfinite(A1d + B1d)  # nan + anything = nan
    if np.sum(indOK) < 2: return 0
    A_mean = A1d[indOK].mean()
    B_mean = B1d[indOK].mean()
    varA = np.square(A1d[indOK] - A_mean).mean()
    varB = np.square(B1d[indOK] - B_mean).mean()
    if varA * varB == 0: return 0
    if varA < 1e-10 * A_mean * A_mean: return 0
    if varB < 1e-10 * B_mean * B_mean: return 0
    covar = (A1d[indOK] * B1d[indOK]).mean() - (A1d[indOK]).mean() * (B1d[indOK]).mean()
    denom = np.sqrt(varA * varB)
    R = covar / denom
    if np.abs(R) > 1.0:
        print('Interesting correlation coefficient:', R)
        print('A:\n', A1d)
        print('B:\n', B1d)
        print('covar, variance A, B', covar, varA, varB)
        # dangerous for dimensional variables but loss of numerical precision is a possbility
        if varA < 1e-10 or varB < 1e-10: return 0.0  
        if abs(R) < 1.0001: return 1 * np.sign(R)
        raise ValueError
    return R


#====================================================================================

def weightedCorrCoef(A1d, B1d, weights, indOK=None):
    #
    # Computes weighted correlation coefficient
    #
    if indOK is None: 
        indOK = np.logical_and(np.isfinite(A1d),np.isfinite(B1d))
    if np.sum(indOK) < 2: return 0
    # Weighted average:
    sumWeights = np.sum(weights[indOK])
    meanA = np.sum(A1d[indOK] * weights[indOK]) / sumWeights
    meanB = np.sum(B1d[indOK] * weights[indOK]) / sumWeights
    # weighted variance
    varA = np.sum(np.square(A1d[indOK] - meanA) * weights[indOK]) / sumWeights
    varB = np.sum(np.square(B1d[indOK] - meanB) * weights[indOK]) / sumWeights
    if varA * varB == 0: return 0
    # weighted covariance
    covar = np.sum(A1d[indOK] * B1d[indOK] * weights[indOK]) / sumWeights - meanA * meanB
    denom = np.sqrt(varA * varB)
    if np.abs(covar / denom) > 1.0:
        print('Interesting correlation coefficient:', covar / denom)
        print('A:\n', A1d)
        print('B:\n', B1d)
        print('covar, variance_A, _B', covar, varA, varB)
        if varA < 1e-10 or varB < 1e-10: return 0.0
        if abs(covar / denom) < 1.0001: return 1
        print('Does not look negligible. Raising exception')
        raise ValueError
    return covar / denom 

#====================================================================================

def weighted_mean(A1d, weights, indOK=None):
    #
    # Computes weighted average of an array
    #
    if indOK is None: 
        indOK = np.isfinite(A1d)
    if np.sum(indOK) < 2: return 0
    return np.sum(A1d[indOK] * weights[indOK]) / np.sum(weights[indOK])


#====================================================================================

def weighted_std(A1d, weights, indOK=None):
    #
    # Computes weighted average of an array
    #
    if indOK is None: 
        indOK = np.isfinite(A1d)
    if np.sum(indOK) < 2: return 0
    return np.sqrt(weighted_mean(A1d*A1d, weights, indOK) - weighted_mean(A1d, weights, indOK)**2) 


#====================================================================================

def shift_ax0(ar2shift, shift, fill_value):
    arTmp = ar2shift.copy()   # a temporary copy, faster than other methods
    if fill_value is None:    
        # the foirst/last element will be replicated
        if shift > 0: fill_value = arTmp[0]
        else: fill_value = arTmp[-1]
    if shift > 0:
        ar2shift[:shift] = fill_value
        ar2shift[shift:] = arTmp[:-shift]
    elif shift < 0:
        ar2shift[shift:] = fill_value
        ar2shift[:shift] = arTmp[-shift:]
    else:
        ar2shift[:] = arTmp[:]
    return ar2shift


#====================================================================================

def fu_sigmoid(x, *coefs):
    #
    # Sigmoid function: y = shift_y + scale / (1 + exp (-rate_norm * (x-shift_x)))
    # where rate_norm = rate / (maxX - minX)
    #
    scale, rate, shift_x, shift_y = coefs
#    print(coefs, np.sum(np.square(x-(shift_y + scale / (1. + np.exp(-rate / (np.nanmax(x) - np.nanmin(x)) * (x - shift_x)))))))
    return shift_y + scale / (1. + np.exp(-rate / (np.nanmax(x) - np.nanmin(x)) * (x - shift_x))) 

#====================================================================================

def sigmoid_LS(coefs, *pars):
    #
    # Sigmoid function: y = shift_y + scale / (1 + exp (-rate_norm * (x-shift_x)))
    # where rate_norm = rate / (maxX - minX)
    #
    scale, rate, shift_x, shift_y = coefs
    x, y, a, b, ifPrint = pars
    yPred = shift_y + scale / (1. + np.exp(-rate / (np.nanmax(x) - np.nanmin(x)) * (x - shift_x)))
    if ifPrint:
        print(coefs, np.sum(np.square(y-yPred)))
        fig, ax = plt.subplots(1,1, figsize=(6,7))
        chart = ax.scatter(x, y, marker='.', label='y')
        chart2 = ax.scatter(x, yPred, marker='o', label='fit')
        ax.set_title('%g ' % np.sum(np.square(y-yPred)) + str(coefs))
        ax.legend()
        plt.savefig('d:\\tmp\\sig\\sigmoid_' + str(time.time()) + '.png')
        plt.clf()
        plt.close()
#    return np.sum(np.square(y - yPred) / (a + b * y))
    return (y - yPred) / (a + b * y)
#    return shift_y + scale / (1. + np.exp(-rate / (np.nanmax(x) - np.nanmin(x)) * (x - shift_x))) 


#====================================================================================

def exp_LS(coefs, *pars):
    #
    # Sigmoid function: y = shift_y + scale * exp(rate_norm * (x-shift_x)
    # where rate_norm = rate / (maxX - minX)
    #
    scale, rate, shift_x, shift_y = coefs
    x, y, a, b, ifPrint = pars
    try:
        yPred = shift_y + scale * np.exp(rate / (np.nanmax(x) - np.nanmin(x)) * (x - shift_x))
        ifPrintLocal = False
    except:
        print('Warning: problem with supplementary.exp_LS')
        print('coefs:', coefs, 'np.nanmax(x)', np.nanmax(x), 'np.nanmin(x)', np.nanmin(x), 
              '\nx ', x, '\ny', y)
        ifPrintLocal = True
        yPred = np.ones(shape=x.shape, dtype=np.float32) * shift_y
        idxOK = rate / (np.nanmax(x) - np.nanmin(x)) * (x - shift_x) > -20
        yPred[idxOK] = shift_y + scale * np.exp(rate / (np.nanmax(x[idxOK]) - np.nanmin(x[idxOK])) *
                                                (x[idxOK] - shift_x))
    if ifPrint or ifPrintLocal:
        print(coefs, np.sum(np.square(y-yPred)))
        fig, ax = plt.subplots(1,1, figsize=(6,7))
        chart = ax.scatter(x, y, marker='.', label='y')
        chart2 = ax.scatter(x, yPred, marker='o', label='fit')
        ax.set_title('%g ' % np.sum(np.square(y-yPred)) + str(coefs))
        ax.legend()
        plt.savefig('sigmoid_' + str(time.time()) + '.png')
        plt.clf()
        plt.close()
#    return np.sum(np.square(y - yPred) / (a + b * y))
    return (y - yPred) / (a + b * y)
#    return shift_y + scale / (1. + np.exp(-rate / (np.nanmax(x) - np.nanmin(x)) * (x - shift_x))) 


#====================================================================================

def fu_exp(x, *coefs):
    #
    # Sigmoid function: y = shift_y + scale * exp(rate_norm * (x-shift_x)
    # where rate_norm = rate / (maxX - minX)
    #
    scale, rate, shift_x, shift_y = coefs
    return shift_y + scale * np.exp(rate / (np.nanmax(x) - np.nanmin(x)) * (x - shift_x))


#====================================================================================

def hyperbola_LS(coefs, *pars):
    #
    # Sigmoid function: y = shift_y + scale / (x - shift_x)
    #
    scale, shift_x, shift_y = coefs
    x, y, a, b, ifPrint = pars
    yPred = shift_y + scale / (x - shift_x)
    if ifPrint:
        print(coefs, np.sum(np.square(y-yPred)))
        fig, ax = plt.subplots(1,1, figsize=(6,7))
        chart = ax.scatter(x, y, marker='.', label='y')
        chart2 = ax.scatter(x, yPred, marker='o', label='fit')
        ax.set_title('%g ' % np.sum(np.square(y-yPred)) + str(coefs))
        ax.legend()
        plt.savefig('d:\\tmp\\hyp\\hyperbola_' + str(time.time()) + '.png')
        plt.clf()
        plt.close()
#    return np.sum(np.square(y - yPred) / (a + b * y))
    return (y - yPred) / (a + b * y)
#    return shift_y + scale / (1. + np.exp(-rate / (np.nanmax(x) - np.nanmin(x)) * (x - shift_x))) 


#====================================================================================

def fu_hyperbola(x, *coefs):
    #
    # Sigmoid function: y = shift_y + scale / (x-shift_x)
    # where rate_norm = rate / (maxX - minX)
    #
    scale, shift_x, shift_y = coefs
    return shift_y + scale / (x - shift_x)


#====================================================================================

def fu_gaussian(x, mu, sig):
    #
    # Returns Gaussian function
    #
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))


#====================================================================================

def shift_arr(arr, num, fill_value=np.nan):
    # Shifts the given array with a given index
    # preallocate empty array and assign slice by chrisaycock
    # This is the fastest way among quite a few tested by StaackOverflow
    result = np.empty_like(arr)
    if num > 0:
        result[:num] = fill_value
        result[num:] = arr[:-num]
    elif num < 0:
        result[num:] = fill_value
        result[:num] = arr[-num:]
    else:
        result[:] = arr
    return result


#====================================================================================

def MPI_join(chJoinID, tmpDir, wait_max = one_hour * 24):
    #
    # If MPI is hand-made on puhti, have to implement joining manually,
    # Each process creates a directory with chJoinID, and mpi rank
    # process zero here checks these directories. When all are at place, deletes them
    # all, which is a sign for non-zero MPI processes to continue
    # wait_max controls how long the synchronization can wait
    #
    if mpisize == 1: return
    tEndWait = dt.datetime.now() + wait_max
    # valid communicator?
    if comm is None:          # puhti case
        print('MANUAL BARRIER mpi %03i' % mpirank)
        # Announce your presence
        path2check = os.path.join(tmpDir,'%s_%03i' % (chJoinID, mpirank))
        try: os.makedirs(path2check)
        except: pass
        # MPI 0 collects the signals, others wait
        if mpirank == 0:
            # check that everyone reached the end and created the directories
            for iP in range(mpisize):
                path2check = os.path.join(tmpDir, '%s_%03i' % (chJoinID,iP))
                cnt = 0
                while not os.path.exists(path2check):
                    if cnt > 10: 
                        print('000 is waiting for MPI %03i' % iP, dt.datetime.now())
                        cnt = 0
                    time.sleep(10)
                    cnt += 1
                    if dt.datetime.now() > tEndWait: 
                        raise ValueError(('MPI_%03i is waiting too long, ' % mpirank) + 
                                         dt.datetime.now().strftime('%Y %m %d %H %M'))
            # All directories are at place. Now delete all directories
            for iP in range(mpisize):
                path2check = os.path.join(tmpDir,'%s_%03i' % (chJoinID, iP))
                os.removedirs(path2check)
        else:
            # wait until own directory is deleted
            cnt = 0
            while os.path.exists(path2check):
                if cnt > 10: 
                    print('mpi %03i is waiting' % mpirank, dt.datetime.now())
                    cnt = 0
                time.sleep(10)
                cnt += 1
                if dt.datetime.now() > tEndWait: 
                    raise ValueError(('MPI_%03i is waiting too long, ' % mpirank) + 
                                     dt.datetime.now().strftime('%Y %m %d %H %M'))
    else:
        # communicator exists. Make a barrier
        print('BARRIER')
#        iTmp = mpirank
        comm.Barrier()   #gather(iTmp, root=0)
    print('mpi %03i released' % mpirank)


#====================================================================================

def ensure_directory_MPI(chDir, ifPatience=False):
    #
    # Makes sure that the directory exists, creates if neeed
    # In MPI, only mpirank 0 is allowed to create teh directory. Others must wait
    #
    cnt = 0
    while not os.path.exists(chDir):
        try: os.makedirs(chDir)
        except: pass
        if cnt > 0:
            time.sleep(5)
            if cnt > 100: 
                if ifPatience:
                    print('Waiting for the directory ' + chDir)
                    cnt = 0   # restart waiting
                    if cnt > 1000: raise ValueError('Failed to create directoory 1: ' + chDir)
                else: raise ValueError('Failed to create directoory 2: ' + chDir)
        cnt += 1


######################################################################################

if __name__ == '__main__':
    print('Hi, MPI rank', mpirank)
    
#    fLst = glob.glob('f:\\project\\fires\\IS4FIRES_v3_0_grid_PP\\EU_2_0_LST_daily\\IS4FIRES_v3_0_EU_2_0_*_LST_daily.nc4')
#    tmr = SFK_timer()
#    tmr.start_timer('reading')
#    vSum = 0
#    for i in range(1, 500):
#        with nc4.Dataset(fLst[i]) as nc:
#            try:
#                v = nc.variables['FP_frp'][:]
#            except:
#                v = [0]
#            vSum += v[0]
#    tmr.stop_timer('reading')
#    tmr.report_timers()
    
    x = np.arange(0.0, 4*np.pi, 0.01)
    A = np.sin(x) + 1
    B_ini = np.sin(x) + 1.1
    rnd = (np.random.random(size=x.shape[0]) - 0.5)

    B = B_ini * (1 + 1 * rnd) #**2
    weights = np.ones(shape=A.shape) / (B+0.2)
    
    print(nanCorrCoef(A, B), weightedCorrCoef(A, B, weights))
    
    fig, ax = plt.subplots(1,1, figsize=(10,10))
    p2 = ax.plot(x,B,label='B', linestyle='', marker = '.', ) #linewidth=2)
    p1 = ax.plot(x,A,label='A', linestyle='-', linewidth=3)
    p3 = ax.plot(x,rnd,label='noise', linestyle='', marker='.',markersize=3)
    p4 = ax.plot(x,weights,label='weight', linestyle='', marker='.')
    ax.legend()
    ax.set_title('A_av= %4.2f, A_Wav = %4.2f, B_av= %4.2f, B_Wav = %4.2f, Corr = %4.3f, Wcorr = %4.3f' % 
                 (A.mean(), weighted_mean(A, weights), B.mean(), weighted_mean(A, weights), 
                  nanCorrCoef(A, B), weightedCorrCoef(A, B, weights)))
    plt.savefig('d:\\tmp\\corrcoef_weighted_tst.png')
#    plt.show()
    
