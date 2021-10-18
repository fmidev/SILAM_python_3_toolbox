import collections
import pylab

from toolbox import tables
#from toolbox import statistician
#iport statistician
try:
    from mpl_toolkits import basemap
    from mpl_toolkits.basemap import Basemap
except ImportError:
    print ('Warning: basemap not available')
import numpy as np

def skip_nan(aggr):
    """
    Transform function aggr to ignore nans.
    """
    def nan_skipped(x):
        x = np.array(x)
        return aggr(x[~np.isnan(x)])
    return nan_skipped

class Summary:
    def __init__(self, aggregations, stationdict=None):
        self.stationdict = stationdict
        self.tables = {}
        self.sumtables = {}
        self.n_by_stype = {}
        self.columns = None
        self.aggregations = aggregations
        
    def add_table(self, model, table):
        self.tables[model] = table
        if not self.columns:
            self.columns = table.columns
        else:
            if not self.columns == table.columns:
                raise ValueError('Attempting to summarise nonhomogeneous tables')
            
    def _get_stationdict(self):
        self.stationdict = {}
        for table in self.tables.values():
            for station in table.rows:
                self.stationdict[station.code] = station

    def _get_stationtypes(self):
        types = collections.defaultdict(set)
        for station in self.stationdict.values():
            stype = (station.source, station.area)
            types[stype].add(station.code)
        self.stationtypes = types
        
    def _get_for_stype(self, stype):
        sumtable = tables.Table(columns=self.columns, first_column_width=24,
                                      value_format='%20.2f', field_width=20)
        stations = set()
        n = 0
        for model, table in self.tables.items():
            for row in table.rows:
                if ((isinstance(row, basestring) and not row in self.stationtypes[stype])
                    or (not isinstance(row, basestring) and not row.code in self.stationtypes[stype])):
                    continue
                stations.add(row)
                n += 1
        self.n_by_stype[stype] = n
        for model in self.tables:
            for aggr in self.aggregations:
                row = '%s: %s' % (aggr[0], model)
                aggregate = self.tables[model].aggregate(stations, aggr[0], aggr[1])
                for col in self.columns:
                    sumtable.set(row, col, aggregate.get(aggr[0], col))
        return sumtable
        
    def collect(self):
        if not self.stationdict:
            self._get_stationdict()
        self._get_stationtypes()
        for stype in self.stationtypes:
            self.sumtables[stype] = self._get_for_stype(stype)

    def to_stream(self, stream):
        for (stype, table) in self.sumtables.items():
            area, source = stype
            if area == '':
                area = 'unknown area'
            if source == '':
                source = 'unknown source'
            stream.write('SUMMARY - %s, %s, N = %i\n\n' % (area, source, self.n_by_stype[stype]))
            table.to_stream(stream)
            stream.write('\n')
            
        
class Histogram:
    def __init__(self, param):
        self.param = param
        self.tables = []
        
    def add(self, table, label=None, color=None):
        values = np.array(table.get_column(self.param))
        self.tables.append((values[~np.isnan(values)], label, color))

    def make(self, axes=pylab, **kwargs):
        values, labels, colors = zip(*self.tables)
        if any(colors):
            kwargs['color'] = colors
        kwargs['label'] = labels
        return axes.hist(values, **kwargs)

class Map:
    def __init__(self, station_table, stationdict=None):
        self.table = station_table
        if not stationdict:
#            assert iter(self.table.rows).next().x
            assert next(iter(self.table.rows)).lon
            self._get_stationdict()
            self._code_is_key = False
        else:
            self.stationdict = stationdict
            self._code_is_key = True
        
    def _get_stationdict(self):
        self.stationdict = {}
        for station in self.table.rows:
            self.stationdict[station.code] = station
        
    def basemap(self, llcrnrlon=None, urcrnrlat=None,
                                 urcrnrlon=None, llcrnrlat=None):
        # find the envelope for the map
        
        urcrnrlon = urcrnrlon or  int(max(s.x for s in self.stationdict.itervalues())+5)
        urcrnrlat = urcrnrlat or int(max(s.y for s in self.stationdict.itervalues())+5)
        llcrnrlat = llcrnrlat or int(min(s.y for s in self.stationdict.itervalues())-5)
        llcrnrlon = llcrnrlon or int(min(s.x for s in self.stationdict.itervalues())-5)
        mp = Basemap(projection='cyl',llcrnrlon=llcrnrlon, urcrnrlat=urcrnrlat,
                     urcrnrlon=urcrnrlon, llcrnrlat=llcrnrlat,resolution='l')
        return mp

    def values(self, param):
        stations = self.stationdict.values()
        if self._code_is_key:
            keys = [s.code for s in stations]
        else:
            keys = stations

        lons = [s.lon for s in stations]
        lats = [s.lat for s in stations]
        val = [self.table.get(key, param) for key in keys]
        return lons, lats, val
        
    def add(self, table, label=None, color=None):
        values = np.array(table.get_column(self.param))
        self.tables.append((values[~np.isnan(values)], label, color))

    def make(self, axes=pylab, **kwargs):
        values, labels, colors = zip(*self.tables)
        if any(colors):
            kwargs['color'] = colors
        kwargs['label'] = labels
        return axes.hist(values, **kwargs)

def get_timeseries(table, param):
    import datetime as dt
    from toolbox import timeseries
    """
    Read a timeseries from the SpatialStatistician output
    """
    fmt = '%Y-%m-%dT%H:%M:%S'
    values = []
    times = []
    for row in table:
        time = dt.datetime.strptime(row, fmt)
        value = table.get(row, param)
        times.append(time)
        values.append(value)
    series = timeseries.Timeseries(values, times, '1 hr', quantity=param, validator=timeseries.alwaysValid)
    return series
    
