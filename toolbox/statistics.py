'''
A set of tools and classes to compute a variety of statistics.
The statistics can be computed for the filtered subsets of the data 
'''

import datetime, sys, numpy as np, operator
try:
    import scipy.stats.stats as stats
except ImportError:
    print ('Failed to import scipy.stats')

try:
    methodcaller = operator.methodcaller
    
except AttributeError:
    def methodcaller(method_name):
        def call(obj):
            return obj.__getattribute__(method_name)()
        return call

def is_sequence_of_strings(seq):
    try:
        seq.__iter__
    except AttributeError:
        return False
    for ss in seq:
        if not isinstance(basestring, seq):
            return False
    return True

class StatisticError(Exception):
    pass
     
class Statistic:
    """An interface for objects which compute statistical indicators. The
    input for statistics computations is a stream in form of (key,
    obs, mdl) where key is arbitrary object while obs, mdl are reals.

    The subclasses must provide a _eval(points) method, which carries
    out the compuation. The argument points is a stream of tuples
    (obs, mdl). The 'key' argument is only used by the upper-level
    calculate(itervalues) method, which by default simply ignores
    it. However, the method can be overloaded to e.g. perform
    time-based filtering or averaging.

    The input arguments are assumed to be given as iterators. This has
    implications for the computational routines, namely:
    - the len() function cannot be used (calculate the length on the fly)
    - the stream can be traversed only once (create a list if needed).

    By convention, the statistics return np.nan, if the statistic is
    undefined.
    """

    # Valid range of this statistic, if any.
    valid_min = None
    valid_max = None
    normal_max = valid_max
    normal_min = valid_min
    
    def __init__(self):
        self.name = self.__class__.__name__
        
    def calculate(self, itervalues):
        points = ((obs, mdl) for (key, obs, mdl) in itervalues)
        value = self._eval(points)
        return value
     
class VectorStatistic(Statistic):
    """
    An alternative base class for statistics: for subclasses, the _eval method
    takes as an argument two numpy arrays and computes the result.
    """
    def calculate(self, itervalues):
        arrays = [], []
        for (key, obsval, mdlval) in itervalues:
            arrays[0].append(obsval)
            arrays[1].append(mdlval)
        value = self._eval(np.array(arrays[0]), np.array(arrays[1]))
        return value
     
def make_time_filtered(statistic_class, new_name, hour):
    """
    Create a new statistic based on selecting only data for a given hour. 
    """
    class Filtered(statistic_class):
        def __init__(self):
            statistic_class.__init__(self)
            self.name = new_name
        def calculate(self, points):
            newpoints = ((triplet[1], triplet[2]) for triplet in points if triplet[0].hour == hour)
            #newpoints = self.hourfilter(points, hour)
            return self._eval(newpoints)
    return Filtered

def make_daily_mean(statistic_class, new_name):
    class ForDailyMean(statistic_class):
        def __init__(self):
            statistic_class.__init__(self)
            self.name = new_name
                
        def calculate(self, points):
            # find the daily mean
            by_day_1 = {}
            by_day_2 = {}
            count = {}
            for when, v1, v2 in points:
                day = datetime.datetime(when.year, when.month, when.day)
                if not day in by_day_1:
                    by_day_1[day] = v1
                    by_day_2[day] = v2
                    count[day] = 1
                else:
                    by_day_1[day] += v1
                    by_day_2[day] += v2
                    count[day] += 1
            return statistic_class.calculate(self, ((day, by_day_1[day]/count[day], by_day_2[day]/count[day])
                                                    for day in by_day_1))
    return ForDailyMean

def make_daily_max(statistic_class, new_name):
    """
    Create a new statistic by evaluating only daily maximum values.
    """
    class ForDailyMax(statistic_class):
        def __init__(self):
            statistic_class.__init__(self)
            self.name = new_name
        def calculate(self, points):
            # find the daily max
            by_day_1 = {}
            by_day_2 = {}
            for when, v1, v2 in points:
                day = datetime.datetime(when.year, when.month, when.day)
                if not day in by_day_1:
                    by_day_1[day] = v1
                    by_day_2[day] = v2
                else:
                    if v1 > by_day_1[day]:
                        by_day_1[day] = v1
                    if v2 > by_day_2[day]:
                        by_day_2[day] = v2
            return self._eval((by_day_1[day], by_day_2[day]) for day in by_day_1)
    return ForDailyMax


def shift_args_xx_percentile(new_name, new_prc, time_unit):
    """
    Finds out the number of time steps between the location of the percentile prc.
    For pollen, for instance, prc=0.025 will give the error of the 2.5-criterion season start
    in days if daily values are used or in hours if hourly values are used, etc
    """
    class dist_xx_prc(argPrc):
        tu = datetime.timedelta(days=0.)
        def __init__(self):
            argPrc.__init__(self,new_prc)
            self.name = new_name
            self.tu = time_unit
        def calculate(self, points):
            arrays = [], [], []
            for (key, obsval, mdlval) in points:
                arrays[0].append(obsval)
                arrays[1].append(mdlval)
                arrays[2].append(key)
            if len(arrays[0]) < 3: 
                return np.NaN
            else:
                argobs, argmdl = argPrc._eval(self, np.array(arrays[0]), np.array(arrays[1]))
#                print  (argobs, argmdl, arrays)
                if argobs < 0 or argobs >= len(arrays[0]) or argmdl < 0 or argmdl >= len(arrays[0]):
                    return np.NaN
                else:
                    return (arrays[2][argmdl] - arrays[2][argobs]).total_seconds() / self.tu.total_seconds()
    return dist_xx_prc


def args_xx_percentile_obs(new_name, new_prc):
    """
    Finds out the day number since the start of the year of the percentile prc.
    For pollen, for instance, prc=0.025 will give the date of 2.5-criterion season start
    """
    class arg_xx_prc_obs(argPrc):
        def __init__(self):
            argPrc.__init__(self,new_prc)
            self.name = new_name
        def calculate(self, points):
            arrays = [], [], []
            for (key, obsval, mdlval) in points:
                arrays[0].append(obsval)
                arrays[1].append(mdlval)
                arrays[2].append(key)
            if len(arrays[0]) < 3:
                print (self.name, ': too few points in time series, cannot make obs percentile')
                print ('keys, mdl values, obs values', arrays[2]) #, array[0],array[1] )
                return np.NaN
#            print arrays[2], arrays[0], arrays[1]
            argobs, argmdl = argPrc._eval(self, np.array(arrays[0]), np.array(arrays[1]))
            if argobs < 0 or argobs >= len(arrays[0]) or argmdl < 0 or argmdl >= len(arrays[0]):
                return np.NaN
            else:
                return arrays[2][argobs].timetuple().tm_yday
    return arg_xx_prc_obs

    
def args_xx_percentile_mdl(new_name, new_prc):
    """
    Finds out the day number since the start of the year of the percentile prc.
    For pollen, for instance, prc=0.025 will give the date of 2.5-criterion season start
    """
    class arg_xx_prc_mdl(argPrc):
        def __init__(self):
            argPrc.__init__(self,new_prc)
            self.name = new_name
        def calculate(self, points):
            arrays = [], [], []
            for (key, obsval, mdlval) in points:
                arrays[0].append(obsval)
                arrays[1].append(mdlval)
                arrays[2].append(key)
            if len(arrays[0]) < 3:
                print (self.name, ': too few points in time series, cannot make mdl percentile' )
                return np.nan
            argobs, argmdl = argPrc._eval(self, np.array(arrays[0]), np.array(arrays[1]))
            if argobs < 0 or argobs >= len(arrays[0]) or argmdl < 0 or argmdl >= len(arrays[0]):
                return np.NaN
            else:
                return arrays[2][argmdl].timetuple().tm_yday
    return arg_xx_prc_mdl

    
class argPrc(VectorStatistic):
    prc = 0.0
    def __init__(self,prc_in):
        self.prc = prc_in
    def _eval(selfself,obs,mdl):
        obs_cum = np.cumsum(obs)
        mdl_cum = np.cumsum(mdl)
        if len(obs_cum) < 3: 
            return np.NaN
        else:
            return (np.searchsorted(obs_cum, obs_cum[-1] * selfself.prc), 
                    np.searchsorted(mdl_cum, mdl_cum[-1] * selfself.prc))

    
class RMS(Statistic):
    valid_min = 0.0
    def _eval(self, pairs):
        n = 0
        rms = 0
        for a, b in pairs:
            n += 1
            rms += (a-b)**2
        if n == 0:
            return np.nan
        return np.sqrt(rms/n)

class NormBias(Statistic):
    normal_min = valid_min = -1.0
    normal_max = 1.0
    def _eval(self, pairs):
        n = 0
        nbias = 0
        for a, b in pairs:
            n += 1
        if a+b != 0.0: nbias += (a-b)/(a+b)
        if n == 0:
            return np.nan
        return 2.0*nbias/n

class FractErr(Statistic):
    normal_min = valid_min = 0.0
    normal_max = 2.0
    def _eval(self, pairs):
        n = 0
        ferr = 0
        for a, b in pairs:
            n += 1
            if a+b != 0.0: ferr += abs((a-b)/(a+b))
        if n == 0:
            return np.nan
        return 2.0*ferr/n

class Bias(Statistic):
    def _eval(self, pairs):
        n = 0
        bias = 0
        for a, b in pairs:
            n += 1
            bias += b - a
        if n == 0:
            return np.nan
        return bias/n

class Correlation(Statistic):
    normal_min = 0.0
    valid_min = -1.0
    normal_max = valid_max = 1.0
    def _eval(self, iterpairs):
        pairs = list(iterpairs)
        cov = ma = mb = va = vb = 0
        n = 0
        for a, b in pairs:
            n += 1
            ma += a
            mb += b
        if n == 0:
            return np.nan
        ma /= n
        mb /= n
        for a, b in pairs:
            cov += (a-ma)*(b-mb)
            va += (a-ma)**2
            vb += (b-mb)**2
        if va < 1e-15 or vb < 1e-15:
            return np.nan
        return cov / np.sqrt(va*vb)

class p_value_lg(Statistic):
    def _eval(self, iterpairs):
        pairs = list(iterpairs)
        ap, bp = [], []
        for a,b in pairs:
            ap.append(a)
            bp.append(b)
        if len(ap) >= 2:
            r, p = stats.pearsonr(ap, bp)
            return np.log10(p)
        else:
            print(ap,bp)
            return 0.0

class MaxModel(Statistic):
    def _eval(self, pairs):
        mx = np.nan
        for obs, mdl in pairs:
            if np.isnan(mx) or mdl > mx:
                mx = mdl
        return mx

class MaxObs(Statistic):
    def _eval(self, pairs):
        mx = np.nan
        for obs, mdl in pairs:
            if np.isnan(mx) or obs > mx:
                mx = obs
        return mx


class MeanModel(VectorStatistic):
    def _eval(self, obs, mdl):
        return np.mean(mdl)

class StdevObs(VectorStatistic):
    valid_min = 0.0
    def _eval(self, obs, mdl):
        return np.std(obs)

class StdevModel(VectorStatistic):
    valid_min = 0.0
    def _eval(self, obs, mdl):
        return np.std(mdl)

class MeanObs(Statistic):
     def _eval(self, pairs):
          mean = 0.0
          n = 0
          for obs, mdl in pairs:
               mean += obs
               n += 1
          if not n:
               return np.nan
          return mean / n

class StdevRatio(VectorStatistic):
    normal_min = valid_min = 0.0
    normal_max = 2.0
    def _eval(self, obs, mdl):
        var_mdl = np.std(mdl)
        var_obs = np.std(obs)
        if var_obs < 1e-9:
            return np.nan
        return var_mdl/var_obs

class N(Statistic):
     def _eval(self, pairs):
         nn = 0
         for pair in pairs:
             nn += 1
         return nn

class Fac2(Statistic):
    normal_min = valid_min = 0.0
    normal_max = valid_max = 1.0
    """
    The fraction within factor of 2.
    """
    def _eval(self, pairs):
        n_total = 0
        n_fac2 = 0
        for obs, mdl in pairs:
            n_total += 1
            if obs < 1e-15:
                if mdl < 1e-15:
                    n_fac2 += 1
            else:
                if 0.5 <= mdl/obs <= 2:
                    n_fac2 += 1
        if n_total > 0:
            return float(n_fac2) / n_total
        else:
            return np.nan

# def find_statistic(name):
#     for attr, val in globals().items():
#         #print val
#         try:
#             if issubclass(val, Statistic):
#                 instance = val()
#                 if instance.name == name:
#                     return val
#         except TypeError:
#             pass
        
def map_statistics():
    # make inverse mapping from names to classes. Tricky because the name is an instance
    # attribute. Should maybe make it a class one?
    name2class = {}
    for thing in globals().values():
        try:
            if issubclass(thing, Statistic):
                instance = thing()
                name2class[instance.name] = thing
        except TypeError:
            pass
    return name2class
        
Bias15UTC = make_time_filtered(Bias, 'Bias15UTC', 15)
RMS15UTC = make_time_filtered(RMS, 'RMS15UTC', 15)
MeanObs15UTC = make_time_filtered(MeanObs, 'MeanObs15UTC', 15)

Bias03UTC = make_time_filtered(Bias, 'Bias03UTC', 3)
RMS03UTC = make_time_filtered(RMS, 'RMS03UTC', 3)
MeanObs03UTC = make_time_filtered(MeanObs, 'MeanObs03UTC', 3)

shift_1_prc = shift_args_xx_percentile('shift_1p', 0.01, datetime.timedelta(days=1))
shift_2_5_prc = shift_args_xx_percentile('shift_2_5p', 0.025, datetime.timedelta(days=1))
shift_5_prc = shift_args_xx_percentile('shift_5p', 0.05, datetime.timedelta(days=1))
shift_10_prc = shift_args_xx_percentile('shift_10p', 0.1, datetime.timedelta(days=1))
shift_25_prc = shift_args_xx_percentile('shift_25p', 0.25, datetime.timedelta(days=1))
shift_50_prc = shift_args_xx_percentile('shift_50p', 0.5, datetime.timedelta(days=1))
shift_75_prc = shift_args_xx_percentile('shift_75p', 0.75, datetime.timedelta(days=1))
shift_90_prc = shift_args_xx_percentile('shift_90p', 0.9, datetime.timedelta(days=1))
shift_95_prc = shift_args_xx_percentile('shift_95p', 0.95, datetime.timedelta(days=1))
shift_97_5_prc = shift_args_xx_percentile('shift_97_5p', 0.975, datetime.timedelta(days=1))
shift_99_prc = shift_args_xx_percentile('shift_99p', 0.99, datetime.timedelta(days=1))

day_1_prc_obs = args_xx_percentile_obs('day_1p_o', 0.01) 
day_2_5_prc_obs = args_xx_percentile_obs('day_2_5p_o', 0.025) 
day_5_prc_obs = args_xx_percentile_obs('day_5p_o', 0.05) 
day_10_prc_obs = args_xx_percentile_obs('day_10p_o', 0.1) 
day_25_prc_obs = args_xx_percentile_obs('day_25p_o', 0.25) 
day_50_prc_obs = args_xx_percentile_obs('day_50p_o', 0.5) 
day_75_prc_obs = args_xx_percentile_obs('day_75p_o', 0.75) 
day_90_prc_obs = args_xx_percentile_obs('day_90p_o', 0.9) 
day_95_prc_obs = args_xx_percentile_obs('day_95p_o', 0.95) 
day_97_5_prc_obs = args_xx_percentile_obs('day_97_5p_o', 0.975) 
day_99_prc_obs = args_xx_percentile_obs('day_99p_o', 0.99) 

day_1_prc_mdl = args_xx_percentile_mdl('day_1p_m', 0.01) 
day_2_5_prc_mdl = args_xx_percentile_mdl('day_2_5p_m', 0.025) 
day_5_prc_mdl = args_xx_percentile_mdl('day_5p_m', 0.05) 
day_10_prc_mdl = args_xx_percentile_mdl('day_10p_m', 0.1) 
day_25_prc_mdl = args_xx_percentile_mdl('day_25p_m', 0.25) 
day_50_prc_mdl = args_xx_percentile_mdl('day_50p_m', 0.5) 
day_75_prc_mdl = args_xx_percentile_mdl('day_75p_m', 0.75) 
day_90_prc_mdl = args_xx_percentile_mdl('day_90p_m', 0.9) 
day_95_prc_mdl = args_xx_percentile_mdl('day_95p_m', 0.95) 
day_97_5_prc_mdl = args_xx_percentile_mdl('day_97_5p_m', 0.975) 
day_99_prc_mdl = args_xx_percentile_mdl('day_99p_m', 0.99) 

name2stat = map_statistics()
def find_statistic(name):
    """
    Return the statistic class with given name. If the names are not unique, the matching
    one is picked arbitrarily. If statistic with the given name is found, a KeyError
    results.
    """
    return name2stat[name]

