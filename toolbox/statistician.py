"""
statistician.py - a tool for computing timeseries statistics.

The module defines several statistician classes. Depending on the
subclass, the timeseries are either read from a TimeseriesDatabase or
given in-memory. Once the requested statistical parameters are set,
the computations are carried out by the collect() method.

The computed statistics are returned in a Table instance. The table
columns are various statistics, while rows represent the unaggregated
dimension: the station, time, or a forecast length.

A summary of the classes follows:

Statistician: compute statistics for each station. Timeseries given as objects.

DatabaseStatistician: as above, but read the values as a composite
timeseries from db.

SpatialStatistician: As statistician, but evaluate the parameters for
each time across stations.

SpatialDatabaseStatistician: As above, but reads from a db.

DiurnalStatistician: Compute statistics as a function time of
day. Stations are aggregated.

DiurnalDatabaseStatistician: As above but reads from a db.

ForecastStatistician: Compute values as a function of forecast
length. Only the database version exists.

"""

import datetime, sys, numpy as np, operator
from toolbox import tsdb
from toolbox import timeseries
#import tsdb
from toolbox.statistics import *
from toolbox.tables import Table
from toolbox import util

try:
    # Do we have new enough python?
    methodcaller = operator.methodcaller
except AttributeError:
    def methodcaller(method_name):
        def call(obj):
            return obj.__getattribute__(method_name)()
        return call
import json, pickle, os, datetime as dt, warnings
from os import path

class Error(Exception):
    pass


def get_seasonal_filter(*months):
    """
    Construct a function that filters times from the 3-iterator
    (below) based on the month attribute of the first tuple element.
    """
    def filt(iterator):
        for time, obs, mdl in iterator:
            if time.month in months:
                yield time, obs, mdl
    return filt

seasonal_filters = {'djf' : get_seasonal_filter(12,1,2),
                    'mam' : get_seasonal_filter(3,4,5),
                    'jja' : get_seasonal_filter(6,7,8),
                    'son' : get_seasonal_filter(9,10,11)}




class Statistician:
    """Compute statistical parameters for sets of timeseries. 
    Example usage could be as follows:

    statist = Statistician('NO2')
    statist.add_series('model1', list_model1)
    statist.add_series('model2', list_model2)
    statist.set_reference('obs', list_obs)
    statist.add_statistic(statistics.RMS)
    statist.collect()
    table = statist.tables('model1')['by_station']
    summary = statist.tables('model1')['summary'] # contains aggregated statistics
    rmse = table.get('S0001', 'RMS')
    ...


    """
    class StationFilter:
        def __init__(self):
            self.sources = ['background']
            self.areas = None
            
    def __init__(self, quantity, time_start=None, time_end=None):
        """ A Statistician instance.
        Arguments:
        quantity - indentifier for quantity the statistics are computed for
        time_start, time_end - time range (inclusive) for the statistcs to be computed.
        """
        
        self.modeldict = {}
        self.models = []
        self.statistics = []
        if time_start and not time_end or time_end and not time_start:
            raise ValueError('Either both or neither time_start and time_end must be given')
        self.first_time = time_start
        self.last_time = time_end
        self.quantity = quantity
        self.obs = None
        self.ref_source = None
        self.filter = Statistician.StationFilter()
        self.verbosity = 1
        self.station_table_key = 'code'
        self.aggregations = [(np.mean, 'Mean'), (np.median, 'Median')]
        
        
    def say(self, what):
        if self.verbosity:
            print (what)

    def add_aggregation(self, aggr_func, aggr_name):
        """
        Add an aggregation function to the summary. The aggregations
        are applied accross the stations; default ones are mean and median.
        """
        self.aggregations.append((aggr_func, aggr_name))
        
    def set_reference(self, model, series_list=None):
        """Set the reference set of series.
        model - name of the set
        series_list - list of Timeseries objects"""
        assert(isinstance(model, str))
        self.ref_source = model
        if series_list is not None:
            hash = {}
            for series in series_list:
                hash[series.station] = series
                self.modeldict[model] = hash
        # Kill the possibly cached observations
        self.obs = None
         
    def add_series(self, model, series_list):
        """Add a set of series for computations."""
        hash = {}
        for series in series_list:
            hash[series.station] = series
        self.modeldict[model] = hash
        self.models.append(model)
         
    def add_statistic(self, statistic):
        """Include the statistics.Statistic subclass into computations."""
        self.statistics.append(statistic())

    def filter_areas(self, area_types):
        if area_types is not None:
            assert(util.is_sequence_of_strings(area_types))
        self.filter.areas = area_types

    def filter_sources(self, source_types):
        if source_types is not None:
            assert(util.is_sequence_of_strings(source_types))
        self.filter.sources = source_types
         
    def _get_values(self, source):
#        print self.modeldict.keys()
        return self.modeldict[source]

    def _get_observations(self):
        if self.ref_source is None:
            raise Error('Reference source not set')
        if self.obs is None:
            self.obs = self._get_values(self.ref_source)
            
    def _timefilter_none(self, iterator):
        for time, obs, mdl in iterator:
            yield time, obs, mdl

    def _timefilter(self, iterator):
        for time, obs, mdl in iterator:
            if time >= self.first_time and time <= self.last_time:
                yield time, obs, mdl
                
    def _get_statistic(self, source, statistic):
        #print source, statistic.name
        self._get_observations()
        values = self._get_values(source)
        statistic_by_station = {}
        for station in self.obs:
            if not station in values:
                continue
            #from_itermerge = ((val[0], val[2], val[3])
            #                  for val in timeseries.iterMerge(self.obs[station], values[station]))
            #zip3 = from_itermerge
            #zip3 = timeseries.iterMerge(self.obs[station], values[station])
            #zip3 = timeseries.izip3_timeseries(self.obs[station], values[station])
            
#            print self.obs[station].times(), values[station].times()
            
            zip3 = timeseries.izip3_timeseries(self.obs[station], values[station])
            if self.first_time:
                iterator = self._timefilter(zip3)
            else:
                iterator = zip3
            statistic_by_station[station] = statistic.calculate(iterator)
#            print station, statistic_by_station[station] 
        return statistic_by_station

    def _filtered_obs_stations(self):
        self._get_observations()
        return list(tsdb.filtered_stations(self.obs.keys(),
                                           area_types=self.filter.areas,
                                           dominant_sources=self.filter.sources))
     
    def collect(self):
        """Calculate the statistical parameters. Two tables will be created
        for each model: the by_station table with values for each
        station, and the summary table."""
        
        stations = self._filtered_obs_stations()
        if not stations:
            self.say('No comparable stations found')
        statnames = [stat.name for stat in self.statistics]
        stat_tables = {}
        for model in self.models:
            table = Table(columns=statnames, value_format='%20.2f', 
                          field_width=20, first_column_width=16)
            table.row_formatter = operator.attrgetter(self.station_table_key) 
            for stat in self.statistics:
#                print model, stat.name
                statistic_by_station = self._get_statistic(model, stat)
                for station in stations:
#                    print station
                    table.add_row(station)
                    if not station in statistic_by_station:
                        table.values[(station,stat.name)] = np.nan
                    else:
                        table.values[(station,stat.name)] = statistic_by_station[station]
                    
            # check if we have rows where no statistics where obtained, probably because they
            # were outside some dataset
            rows = list(table.rows)
            for row in rows:
                if all(np.isnan(table.get_row(row))):
                    table.remove_row(row)

            stat_tables[model] = table
             
        summary_tables = {}
        for model in self.models:
            summary_table = Table(None, statnames, value_format='%20.2f', 
                                  field_width=20, first_column_width=24)
            summary_tables[model] = summary_table
            for stat in self.statistics:
                table = stat_tables[model]
                values = {}
                for station in stations:
                    key = station, stat.name
                    #if np.isnan(table.values[key]):
                    #    continue
                    if not station.area in values:
                        values[station.area] = []
                    values[station.area].append(table.values[key])
                if not values:
                    raise Error('No values anywhere!')
                for func, funcname in self.aggregations:
                    for station_type in values:
                        value = func([val for val in values[station_type] if not np.isnan(val)])
                        nvalues = len(values[station_type])
                        if self.filter.areas:
                            row = funcname
                        else:
                            row = '%s (%s, N=%i)' % (funcname, station_type, nvalues)
                        summary_table.add_row(row)
                        summary_table.values[(row, stat.name)] = value

        self.stat_tables = stat_tables
        self.summary_tables = summary_tables

    def report(self, model, filename=None):
        if filename:
            self.say('filename: ' + filename)
            stream = open(filename, 'w')
        else:
            stream = sys.stdout
        self._write_basic_header(model, stream)
            
        stream.write('\nSUMMARY\n')
        self.summary_tables[model].to_stream(stream)
        stream.write('\n')

        self.stat_tables[model].to_stream(stream)
        if filename:
            stream.close()

    def _write_basic_header(self, model, stream):
        stream.write('MODEL: %s\n' % model)
        stream.write('SPECIES: %s\n' % self.quantity)
        if self.first_time:
            stream.write('COVERAGE: %s thru %s\n' % (self.first_time, self.last_time))
        if self.filter.areas:
            areas = ' '.join(self.filter.areas)
        else:
            areas = 'all areas'
        if self.filter.sources:
            sources = ' '.join(self.filter.sources)
        else:
            sources = 'all sources'
        if self.filter.sources or self.filter.areas:
            stream.write('STATIONS: %s - %s\n' % (areas, sources))
             
    def tables(self, model):
        return {'by_station' : self.stat_tables[model], 'summary' : self.summary_tables[model]}
     
class DatabaseStatistician(Statistician):
    def __init__(self, db, dataset, quantity, time_start, time_end, timestep,
                 ref_source=None, cachedir=None):
        Statistician.__init__(self, quantity, time_start, time_end)
        self.db = db
        self.timestep = timestep
        self.shift_by_model = {}
        if ref_source:
            self.ref_source = ref_source
            self.shift_by_model[ref_source] = datetime.timedelta(0)
        else:
            self.ref_source = None
            
        self.dataset = dataset
        self.collocated = {}
        self.cachedir = cachedir
        
    def add_model(self, model, shift=datetime.timedelta(0)):
        self.models.append(model)
        self.shift_by_model[model] = shift

    def set_reference_model(self, model, shift=datetime.timedelta(0)):
        self.ref_source = model
        self.shift_by_model[model] = shift
        self.obs = None

    def set_timefilter(self, filt):
        """
        Filter the values to compute by their timestamp. By default,
        we take everything between the time_start and time_end defined
        above. A feasible choice is a seasonal filter defined in this
        module.
        """
        self._timefilter = filt

    def reset_timefilter(self):
        self._timefilter = self._timefilter_none

    def _filtered_obs_stations(self):
        stations = self.db.get_stations((self.dataset,))
        return list(tsdb.filtered_stations(stations,
                                           area_types=self.filter.areas,
                                           dominant_sources=self.filter.sources))
        
    def _get_values(self, source):
        self.say("Retrieving data for '%s'" % source)
        if source in self.modeldict:
            return self.modeldict[source]
        
        shift = self.shift_by_model[source]
        #print self.db.get_timestamps(self.dataset)
        #print self.first_time, self.last_time
        timestamps = [stamp for stamp in self.db.get_timestamps(self.dataset)
                      if self.first_time <= stamp <= self.last_time]
        by_station = {}
        stations = self._filtered_obs_stations()
        if not timestamps:
            raise UserWarning('No suitable data in dataset %s' % self.dataset)
            # nothing available (should be error?)
            pass
        elif len(timestamps) == 1:
            # only one timestamp, take and have it.
            now = list(timestamps)[0]
            series_list = self.db.get_series(self.dataset, source, now, self.quantity, stations)
            self.say('Timestep %s: %i series read ' % (now, len(series_list)))
            for series in series_list:
                by_station[series.station] = series
            if not series_list:
                warnings.warn(UserWarning('No data found for source %s' % source))
        else:
            now = self.first_time - self.timestep
            # The algorithm: scan the time nodes starting from
            # earliest. Read the series accepting values at times > timestamp + shift[model],
            # where timestamp is the tsdb key. New values will replace the old ones.
            while now < self.last_time:
                now += self.timestep
                forecast_start = now + shift
                series_list = self.db.get_series(self.dataset, source, now, self.quantity, stations)
                self.say('Timestep %s: %i series read ' % (now, len(series_list)))
                # series_list can be empty for a non-existent timenode.
                for series in series_list:
                    if not series.station in by_station:
                        hash = timeseries.Timeseries([],[],[],series.station,self.quantity)
                        by_station[series.station] = hash
                    else:
                        hash = by_station[series.station]
                    for when in series:
                        if when < forecast_start:
                            continue
                        hash.insert(when, series[when], series.duration(when))
                        #hash[when] = series[when]
        self.modeldict[source] = by_station
        self.say('stored to modeldict: %s' % source)
        return by_station


    def _date_handler(obj):
        return obj.isoformat() if hasattr(obj, 'isoformat') else obj

    def _check_cache(self, source):
        if not self.cachedir:
            return
        cachepath = path.join(self.cachedir, source)
        if not path.exists(cachepath):
            return
        infofile = path.join(cachepath, '%s.info' % self.dataset)
        if not path.exists(infofile):
            return
        with open(infofile, 'r') as infofile:
            infodct = pickle.load(infofile)
        if self.first_time < infodct['first_time']:
            self.say('Cache not covering first time')
            return
        if self.last_time > infodct['last_time']:
            self.say('Cache not covering last time')
            return
        my_shift = self.shift_by_model[source].days*3600*24 + self.shift_by_model[source].seconds
        if not 'shift' in infodct or my_shift != int(infodct['shift']):
            self.say('Cache has different shift')
            return
        mtime = self.db.modif_time(self.dataset, source, self.quantity)
        self.say('Mod times:', mtime, infodct['mtime'])
        if not mtime:
            return # nothing in db!

        if not infodct['mtime'] or mtime > infodct['mtime']:
            self.say('Source dataset changed since cached')
            return
        if mtime < infodct['mtime']:
            raise ValueError('cache has been changed in future')
        with open(path.join(cachepath, '%s.data' % self.dataset), 'r') as datafile:
            collocated = pickle.load(datafile)
        self.say('Collocated from cache:', cachepath)
        self.collocated[source] = collocated
        return collocated
    
    def _store_cache(self, source, data):
        if not self.cachedir:
            return
        cachepath = path.join(self.cachedir, source)
        if not path.exists(cachepath):
            os.makedirs(cachepath)
        infodct = {'last_time' : self.last_time,
                   'first_time' : self.first_time,
                   'mtime' : self.db.modif_time(self.dataset, source, self.quantity),
                   'shift' : self.shift_by_model[source].days*3600*24 + self.shift_by_model[source].seconds}
        with open(path.join(cachepath, '%s.info' % self.dataset), 'w') as infofile:
            pickle.dump(infodct, infofile)
        with open(path.join(cachepath, '%s.data' % self.dataset), 'w') as datafile:
            pickle.dump(data, datafile)
        self.say('Stored in cache:', cachepath)

    def _check_mem(self, source):
        if source in self.collocated:
            return self.collocated[source]
        else:
            return None
        
    def _collocate(self, source):
        self.say('collocating: %s' % source)
        obs = self._get_values(self.ref_source)
        self.say('have observations')
        mdl = self._get_values(source)
        self.say('have model')
        by_station = {}
        for station in obs:
            if not station in mdl or len(obs[station]) == 0 or len(mdl[station]) == 0:
                continue
            from_itermerge = ((val[0], val[2], val[3])
                              for val in timeseries.iterMerge(obs[station], mdl[station]))
            collocated = list(from_itermerge)
            #collocated = list(timeseries.iterMerge(obs[station], mdl[station]))
            #collocated = list(timeseries.izip3(obs[station], mdl[station]))
            by_station[station] = collocated
        self.collocated[source] = by_station
        self.say('removing from modeldict: %s' % source)
        del(self.modeldict[source])
        if len(self.modeldict) == 1:
            key = self.modeldict.keys()[0]
            assert key == self.ref_source
            self.say('collocation done, removing observations')
            del(self.modeldict[key])
        return by_station

    def _check_cache_mem(self, source):
        inmem = self._check_mem(source)
        if inmem is None:
            cached = self._check_cache(source)
            if cached is None:
                values = self._collocate(source)
                if values:
                    self._store_cache(source, values)
                #else:
                #    print 'no values'
            elif not cached:
                # in cache, but empty
                warnings.warn('source %s: in cache, but empty' % source)
                return
            else:
                values = cached
        elif not inmem:
            # in memory but empty
            return 
        else:
            values = inmem
        return values
    
    def _get_statistic(self, source, statistic):
        values = self._check_cache_mem(source)
        statistic_by_station = {}
        if not values:
            return statistic_by_station
        for station in values:#self.collocated[source]:
            zip3 = iter(values[station])#iter(self.collocated[source][station])
            if self.first_time:
                iterator = self._timefilter(zip3)
            else:
                iterator = zip3
            statistic_by_station[station] = statistic.calculate(iterator)
        return statistic_by_station
        
    def _write_basic_header(self, model, stream):
        Statistician._write_basic_header(self, model, stream)
        stream.write('FORECAST LENGTH: %s\n\n' % self.shift_by_model[model])
    
    def report(self, model, filename=None):
        if filename:
            stream = open(filename, 'w')
        else:
            stream = sys.stdout
        self._write_basic_header(model, stream)
        stream.write('SUMMARY\n')
        self.summary_tables[model].to_stream(stream)
        stream.write('\n')

        self.stat_tables[model].to_stream(stream)
        if filename:
            stream.close()

def iterate_over_common_keys(dict1, dict2):
    for key in dict1:
        if key in dict2:
            yield key, dict1[key], dict2[key]

class SpatialStatistician(Statistician):
    def _get_statistic(self, source, statistic):
        # return statistic by time
        self._get_observations()
        values = self._get_values(source)
        all_times = set()
        for series in values.itervalues():
            #1/0
            all_times.update(series.__iter__())
        statistic_by_time = {}
        for time in all_times:
            valhash, obshash = {}, {}
            for station in values:
                if not time in values[station] or not time in self.obs[station]:
                    continue
                valhash[station] = values[station][time]
                obshash[station] = self.obs[station][time]
            iterator = iterate_over_common_keys(obshash, valhash)
            statistic_by_time[time] = statistic.calculate(iterator)   
        return statistic_by_time
    
    def collect(self):
        self.timetables = {}
        self._get_observations()
        statnames = [stat.name for stat in self.statistics]
        for model in self.models:
            table = Table(columns=statnames, value_format='%20.2f',
                          field_width=20, first_column_width=24)
            table.row_formatter = methodcaller('isoformat')
            for stat in self.statistics:
                statistic_by_time = self._get_statistic(model, stat)
                for time, val in statistic_by_time.iteritems():
                    table.add_row(time)
                    table.values[(time, stat.name)] = val
            self.timetables[model] = table

    def report(self, model, filename=None):
        if filename:
            stream = open(filename, 'w')
        else:
            stream = sys.stdout
        self._write_basic_header(model, stream)
        stream.write('STATISTICS BY TIME\n')
        self.timetables[model].to_stream(stream)
        stream.write('\n')
        if filename:
            stream.close()

    def tables(self, model):
        return {'by_time' : self.timetables[model]}
                             

class SpatialDatabaseStatistician(DatabaseStatistician, SpatialStatistician):
    _get_statistic = SpatialStatistician._get_statistic
    collect = SpatialStatistician.collect
    report = SpatialStatistician.report
    tables = SpatialStatistician.tables

class DiurnalStatistician(Statistician):
    def _get_statistic(self, source, statistic):
        self._get_observations()
        stations = self._filtered_obs_stations()
        # values is indexed by station
        values = self._get_values(source)
        all_times = set()
        for series in values.itervalues():
            # series can be either timeseries or plain dictionary
            all_times.update(series.__iter__())
        # now all_time contains all times 
        hours = range(24)
        obshash = [{} for _i in hours]
        valhash = [{} for _i in hours]
        statistic_by_time = [np.nan for i in hours] #@UnusedVariable
        for time in all_times:
            for station in values:
                if not station in stations: # the filtered stations!
                    continue
                if not time in values[station] or not time in self.obs[station]:
                    continue
                key = (station,time)
                valhash[time.hour][key] = values[station][time]
                obshash[time.hour][key] = self.obs[station][time]

        for hour in hours:
            iterator = iterate_over_common_keys(obshash[hour], valhash[hour])
            statistic_by_time[hour] = statistic.calculate(iterator)
        return statistic_by_time

    def collect(self):
        self.timetables = {}
        self._get_observations()
        
        for model in self.models:
            table = Table(columns=[stat.name for stat in self.statistics],
                          first_column_width=4, field_width=16, value_format='%16.2f')
            for stat in self.statistics:
                statistic_by_hour = self._get_statistic(model, stat)
                for hour, val in enumerate(statistic_by_hour):
                    table.add_row(hour)
                    table.values[(hour,stat.name)] = val
            self.timetables[model] = table

    def report(self, model, filename=None):
        if filename:
            stream = open(filename, 'w')
        else:
            stream = sys.stdout
        self._write_basic_header(model, stream)
        stream.write('STATISTICS BY HOUR (UTC)\n')
        self.timetables[model].to_stream(stream)
        if filename:
            stream.close()
            
    def tables(self, model):
        return {'by_hour' : self.timetables[model]}


class DiurnalDatabaseStatistician(DatabaseStatistician, DiurnalStatistician):
    _get_statistic = DiurnalStatistician._get_statistic
    collect = DiurnalStatistician.collect
    report = DiurnalStatistician.report
    tables = DiurnalStatistician.tables

class ForecastStatistician(DatabaseStatistician):
    @staticmethod
    def _fmt_fclen(delta):
        hours = delta.days*24 + delta.seconds/3600.0
        #minutes = (delta.seconds - delta.seconds/3600 * 3600) / 60
        return '%.1f' % hours
    
    def __init__(self, db, dataset, quantity, time_start, time_end, timestep, ref_source=None):
        DatabaseStatistician.__init__(self, db, dataset, quantity, time_start, time_end,
                                      timestep, ref_source)
        self.fcdict = {}
        
    def _get_forecasts(self, source):
        if source in self.fcdict:
            return self.fcdict[source]

        now = self.first_time
        fchash = {}
        while now <= self.last_time:
            series_list = self.db.get_series(self.dataset, source, now, self.quantity)
            for series in series_list:
                key = now, series.station
                fchash[key]  = series
            
            now += self.timestep
        self.fcdict[source] = fchash
        return fchash

    def _get_statistic(self, source, statistic):
        self._get_observations()
        stations = self._filtered_obs_stations()
        fchash = self._get_forecasts(source)
        fcval = {}
        obsval = {}
        statistic_by_fclen = {}
        for (init_time, station), series in fchash.iteritems():
            if not station in stations:
                continue
            for valid_time in series:
                if not valid_time in self.obs[station]:
                    continue
                fclen = valid_time - init_time
                if not fclen in fcval:
                    fcval[fclen] = {}
                    obsval[fclen] = {}
                fcval[fclen][(init_time, station)] = series[valid_time]
                obsval[fclen][(init_time, station)] = self.obs[station][valid_time]
        for fclen in fcval:
            iterator = iterate_over_common_keys(obsval[fclen], fcval[fclen])
            statistic_by_fclen[fclen] = statistic.calculate(iterator)
        return statistic_by_fclen

    def collect(self):
        self.timetables = {}
        self._get_observations()
        for model in self.models:
            table = Table(columns=[stat.name for stat in self.statistics],
                          value_format='%16.2f', field_width=16, first_column_width=4)
            table.row_formatter = ForecastStatistician._fmt_fclen
            for stat in self.statistics:
                statistic_by_fclen = self._get_statistic(model, stat)
                for delta, val in statistic_by_fclen.iteritems():
                    table.add_row(delta)
                    table.values[(delta, stat.name)] = val
            self.timetables[model] = table

    def report(self, model, filename=None):
        if filename:
            stream = open(filename, 'w')
        else:
            stream = sys.stdout
        self._write_basic_header(model, stream)
        stream.write('\nSTATISTICS BY FORECAST LENGTH\n\n')
        self.timetables[model].to_stream(stream)
        stream.write('...\n')
        if filename:
            stream.close()

    def tables(self, model):
        return {'by_delta' : self.timetables[model]}
