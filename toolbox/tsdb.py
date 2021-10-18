"""
tsdb - a primitive database for Timeseries objects.

This module contains a class for managing a database of timeseries, implemented as simple
directory structure. Functions include storing and retrieving series and stations, 
and some simple aggregations of them.

The db is structured into datasets, which, at each given time, share the same
set of stations. Inside a dataset, series can belong to one of several sources. Finally, every
series has a timestamp, eg. for different forecasts. The series must have a defined quantity.

The db implements a locking routine; concurrent access is not supported. The connection should 
preferably be released explicitly, see below.  

"""
from toolbox import timeseries
from os import path
import os, datetime, shutil
import glob

class TimeseriesDatabaseError(Exception):
    pass
class DatabaseLockedError(TimeseriesDatabaseError):
    """
    This error is raised if the database is locked by another connection.
    """
    pass
class DatabaseDisconnectedError(TimeseriesDatabaseError):
    """
    Raised if an access is attempted throgh a disconnected handle.
    """
    pass

class DatasetNotWritableError(TimeseriesDatabaseError):
    """
    Raised if dataset is not writable (eg. a virtual dataset).
    """

def _closed(*args):
    raise DatabaseDisconnectedError()

    
class TimeseriesDatabase:
    """
    db = TimeseriesDatabase(directory, force)
    A handle to a filesystem-based timeseries database.
    Constructor arguments:
    directory : (string) the root directory for storage
    force : (boolean) ignore locking - user's risk.
    
    """
    _date_format = '%Y%m%d_%H'
    _station_format = {'code' : 0,
                       'latitude' : 1, 
                       'longitude' : 2,
                       'altitude' : 3,
                       'name' : 4,
                       'area' : 5,
                       'source' : 6}
    _station_separator = ';'
    _series_format = {'code' : 0,
                      'quantity' : 1,
                      'year' : 2,
                      'month' : 3,
                      'day' : 4,
                      'hour' : 5,
                      'minute' : 6,
                      'duration' : 7,
                      'value' : 8}
    _lockname = '.lock'
    
    def lock(self):
        self.lock_mode = True
        self._setlock()

    def _setlock(self):
        if not self.lock_mode:
            return
        try:
            os.makedirs(self._lock)
        except os.error:
            if path.exists(self._lock):
                raise DatabaseLockedError()
            else: 
                raise

    def _unlock(self):
        if not self.lock_mode:
            return
        if path.exists(self._lock):
            os.rmdir(self._lock)

    def flush(self):
        # observationDatabse, below, implements this.
        pass
            
    def __init__(self, dir, force=False, lock=True):
        self.root = dir
        self._lock = path.join(dir, TimeseriesDatabase._lockname)

        if not path.exists(self.root):
            os.makedirs(self.root)
        self.lock_mode = lock
        if force:
            self._unlock()

        self._setlock()
        self.stations = {}
        self._open = True
    
    def __del__(self):
        import os
        if os.path.exists(self._lock):
            os.rmdir(self._lock)
    
   
    
    def release(self):
        """
        Disconnect from the db and make it available. Further attempt to access the
        db through the current handle raises an error.
        """
        self._unlock()
        methods = 'put_series', 'get_series', 'get_stations', 'get_sequence', 'export_series'
        for method in methods:
            self.__dict__[method] = _closed
    
    def purge(self):
        for dset in self._get_datasets():
            shutil.rmtree(path.join(self.root, dset))
        self.stations = {}
    def destroy(self):
        self.release()
        shutil.rmtree(self.root)
        
    def put_stations(self, dataset, timestamp, stations):
        """
        Insert a collection of stations into this dataset and timestamp.
        """
        for station in stations:
            self._add_station(dataset, timestamp, station)
        
    def put_series(self, series, timestamp, source, dataset):
        """
        Insert this series into db at the node defined by source, dataset and timestamp.
        The series must have a valid station.
        """
        if not series.station:
            raise TimeseriesDatabaseError('Undefined station')
        if not series.quantity:
            raise TimeseriesDatabaseError('Undefined quantity')
        if dataset == TimeseriesDatabase._lockname:
            raise TimeseriesDatabaseError('Name %s reserved for internal use'
                                         % TimeseriesDatabase._lockname)
        
        self._create_node(source, dataset)
        self._add_station(dataset, timestamp, series.station)
        self._add_series(dataset, source, timestamp, series)

    def _create_node(self, source, dataset):
        dataset_dir = path.join(self.root, dataset)
        if not path.exists(dataset_dir):
            os.makedirs(path.join(dataset_dir, 'stations'))
            os.makedirs(path.join(dataset_dir, 'series'))
            
        source_dir = path.join(dataset_dir, 'series', source)
        if not path.exists(source_dir):
            os.makedirs(source_dir)
        
    def _add_station(self, dataset, timestamp, station):
        # We want to avoid re-reading the station file each time a series is added.
        # Hence, the set of stations for every timestamp-dataset combination is
        # kept in the memory once read for the first time. 
        station_key = (dataset, timestamp)
        station_file = self._get_station_file(dataset, timestamp)
        
        filehandle = None
        
        if not station_key in self.stations:
            # Not stored into this key yet. Read the existing stations, if any.
            stations = set()
            if path.exists(station_file):
                # The file exists: the node has been created earlier. Read
                # the existing stations.
                stations_read \
                = timeseries.readStations(station_file, 
                                          columns=TimeseriesDatabase._station_format,
                                          separator=TimeseriesDatabase._station_separator)
                stations.update(stations_read.itervalues())
                filehandle = open(station_file, 'a')
            else:
                # File does not exist: entirely fresh key.
                filehandle = open(station_file, 'w')
            if not station in stations:
                station.toFile(filehandle, TimeseriesDatabase._station_separator)
            stations.add(station)
            self.stations[station_key] = stations
            
        else:
            # The file has been touched in this session. If the station is in cache, 
            # do nothing, otherwise store into cache and to the file.
            if not station in self.stations[station_key]:
                # This station not yet stored into db. Include into the cache self.stations
                # and append immediately to the file (which must exist).
                filehandle = open(station_file, 'a')
                self.stations[station_key].add(station)
                station.toFile(filehandle, TimeseriesDatabase._station_separator)
        
        if filehandle:
            filehandle.close()
            
    def _get_station_file(self, dataset, timestamp):
        station_file = path.join(self.root, dataset, 'stations', self._get_timestr(timestamp))
        return station_file
    
    def _get_series_file(self, dataset, source, quantity, timestamp):
        series_file = path.join(self.root, dataset, 'series', source, 
                                '%s_%s' % (quantity, self._get_timestr(timestamp)))
        return series_file
    
    def _add_series(self, dataset, source, timestamp, series):
        # Call create_node first.
        series_file = self._get_series_file(dataset, source, series.quantity, timestamp)
        fh = open(series_file, 'a')
        series.toFile(fh, columns=TimeseriesDatabase._series_format)
        fh.close()

    def modif_time(self, dataset, source, quantity):
        stamps = self.get_timestamps(dataset)
        time_last_modif = None
        for stamp in stamps:
            series_file = self._get_series_file(dataset, source, quantity, stamp)
            if not path.exists(series_file):
                continue
            mtime = datetime.datetime.utcfromtimestamp(path.getmtime(series_file))
            print (series_file, mtime)
            if not time_last_modif or mtime > time_last_modif:
                time_last_modif = mtime
        return time_last_modif
    
    def get_series(self, dataset, source, timestamp, quantity, stations=None, forecast_length=None):
        """
        Retrieve a (possibly empty) list of series defined by dataset, source, timestamp
        and quantity. If stations is omitted, all stations are accepted. A forecast length can be given
        - in this case, only read data between timestamp and timestamp+forecast_length. 
        """
        series_file = self._get_series_file(dataset, source, quantity, timestamp)
        if not path.exists(series_file):
            return []
        fh = open(series_file, 'r')
        if stations is None:
            station_file = self._get_station_file(dataset, timestamp)
            if not station_file:
                return []
            stations_read \
                = timeseries.readStations(station_file,
                                          columns=TimeseriesDatabase._station_format,
                                          separator=TimeseriesDatabase._station_separator)
            stations = stations_read.values()
        if forecast_length:
            t2 = timestamp + forecast_length
        else:
            t2 = None
        series_list = timeseries.fromFile_minutes(stations, series_file, quantity,
                                                  validator=timeseries.alwaysValid, t2=t2)
        return series_list
    
    def export_series(self, dataset, source, timestamp, quantity, destination):
        """
        Copy the series into a text file.
        """
        
        series_file = self._get_series_file(dataset, source, quantity, timestamp)
        if not path.exists(series_file):
            raise TimeseriesDatabaseError('No series for source %s in set %s at %s' 
                                          % (source, dataset, timestamp))
            return   
        shutil.copy(series_file, destination)

    def _get_timestr(self, stamp):
        return stamp.strftime(self._date_format)
    
    def _get_timestamps(self, dataset):
        filenames = os.listdir(path.join(self.root, dataset, 'stations'))
        stamps = []
        for filename in filenames:
            stamps.append(datetime.datetime.strptime(filename, self.__class__._date_format))
        return stamps

    get_timestamps = _get_timestamps
    
    def _get_datasets(self):
        ls = os.listdir(self.root)
        lockname = self.__class__._lockname
        #datasets = [entry for entry in ls if not entry.startswith('.') and path.isdir(entry)]
        datasets = [entry for entry in ls 
                    if path.isdir(path.join(self.root, entry)) and entry != lockname]
        return datasets

    def _get_sources(self, dataset):
        directory = path.join(self.root, dataset, 'series')
        ls = os.listdir(directory)
        sources = [entry for entry in ls 
                   if path.isdir(path.join(directory, entry)) and entry != self.__class__._lockname]
        return sources

    def walk(self):
        for dataset in self._get_datasets():
            for source in self._get_sources(dataset):
                yield dataset, source
    
    def _read_stations(self, filename):
        if not path.exists(filename):
            return {}
        stations_read \
            = timeseries.readStations(filename,
                                      columns=TimeseriesDatabase._station_format,
                                      separator=TimeseriesDatabase._station_separator)
        return stations_read
    
    def get_stations(self, datasets=None, timestamps=None):
        """
        Get set of stations, potentially limited to those defined in
        the given datasets and timestamps.
        """
        if not datasets:
            datasets = self._get_datasets()
        if not timestamps:
            timestamps = set()
            for dset in datasets:
                print( dset)
                timestamps.update(self._get_timestamps(dset))
        stations = set()
        for dataset in datasets:
            for timestamp in timestamps:
                station_file = self._get_station_file(dataset, timestamp)
                if not path.exists(station_file):
                    continue
                stations_read = self._read_stations(station_file).itervalues()
                stations.update(stations_read)
        
        return stations 

    def get_sequence(self, dataset, source, times, stations, quantity, forecast_length=None):
        """
        Return a structure in format of list = dict[station], where each element of 
        the list are timeseries at a given station belonging to the given dataset, source
        and quantity. 
        """
        series_dict = {}
        for timestamp in times:
            series_list = self.get_series(dataset, source, timestamp, quantity, list(stations), forecast_length)
            for series in series_list:
                if series.station in series_dict:
                    series_dict[series.station].append(series)
                else:
                    series_dict[series.station] = [series]
        return series_dict

    
    

def filtered_stations(stations, area_types=None, dominant_sources=None):
    """
    Return an iterator over a collection of stations selecting those matching 
    the given area type and dominant source.
    """
    for station in stations:
        if area_types and not station.area in area_types:
            continue
        if dominant_sources and not station.source in dominant_sources:
            continue
        yield station

class ObservationDatabase(TimeseriesDatabase):
    # a refined version of the above. Allows more general observations,
    # vertobs.VerticalObservations mainly. Rest of the interface is compatible with
    # TimeseriesDatabase, but the databases themselves are not necessarily.
    
#_date_format = '%Y%m%d_%H%M%S%f'
    _date_format = '%Y%m%d'
    
    def __init__(self, *args, **kwargs):
        self._append = kwargs['append'] if 'append' in kwargs else True
        self.ts_format = kwargs['timeseries_format'] if 'timeseries_format' in kwargs else 'netcdf'
        for key in ['timeseries_format', 'append']:
            if key in kwargs:
                del(kwargs[key])

        TimeseriesDatabase.__init__(self, *args, **kwargs)

        #TimeseriesDatabase.__init__(self, *args, **kwargs)
       
        self._active_file = None, None
        self._files_created = set()
        self._tswriters = {}

        if 'prefer_backend' in kwargs:
            raise ValueError('prefer_backend is not yet implemented')
            self.prefer_backend = kwargs['prefer_backend']
        else:
            self.prefer_backend = None
        self.compress = 'compress' in kwargs and kwargs['compress']
        
    def _get_obs_file(self, dataset, source, quantity, timestamp, shape=None):
        try:
            timestr = self._get_timestr(timestamp)
        except AttributeError:
            timestr = timestamp            
        series_file = path.join(self.root, dataset, 'series', source, '%s_%s' % (quantity, timestr))
        shapestr = '.' + '_'.join(str(ii) for ii in shape) if shape else ''
        series_file += shapestr
        return series_file
    
    def _add_observation(self, dataset, source, timestamp, observation):
        obs_file = self._get_obs_file(dataset, source, observation.quantity, timestamp, observation.shape)
        if self._active_file[0] != obs_file:
            if self._active_file[0] is not None:
                self._active_file[1].close()
            # Appending: if true, always append. Otherwise, only append if this file was
            # created during the session.
            if path.exists(obs_file) and not self._append and not obs_file in self._files_created:
                print( 'unlink: ', obs_file)
                os.unlink(obs_file)
            try:
                self._active_file = obs_file, observation.__class__.get_writer(obs_file,
                                                                               append=True)#,
                                                                               #backend=nc_backend)
            except AttributeError:
                self._active_file = obs_file, open(obs_file, 'a')
                raise
            self._files_created.add(obs_file)
        fh = self._active_file[1]
        observation.toFile(fh)

    def _time_from_name(self, file_path, have_shape=True):
        name = path.basename(file_path)
        if have_shape:
            quant_date, shape = name.split('.')
        else:
            quant_date = name
        datestr = quant_date.split('_')[-1]
        return datetime.datetime.strptime(datestr, self.__class__._date_format)
        
    def _get_timestamps(self, dataset):
        series_ptrn = self._get_obs_file(dataset, '*', '*', '*', '*')
        timeformat = self._get_obs_file(dataset, '*', '*', self.__class__._date_format, '*')
        print( series_ptrn)
        files = glob.glob(series_ptrn)
        have_shape = True
        if not files:
            # try without shape
            series_ptrn = self._get_obs_file(dataset, '*', '*', '*', None)
            files = glob.glob(series_ptrn)
            have_shape = False
        stamps = [self._time_from_name(name, have_shape) for name in files]
        return set(stamps)

    get_timestamps = _get_timestamps

    def _add_series(self, *args):
        dataset = args[0]
        if self.is_virtual(dataset):
            raise DatasetNotWritableError('Dataset %s is virtual' % dataset)
        if self.ts_format == 'netcdf':
            self._add_series_nc(*args)
        else:
            TimeseriesDatabase._add_series(self, *args)
        
    
    def _add_series_nc(self, dataset, source, timestamp, series):
        series_file = self._get_series_file(dataset, source, series.quantity, timestamp)
        if not series_file in self._tswriters:
            self._tswriters[series_file] = timeseries.NCIO(series_file, 'w', timestamp)
        writer  = self._tswriters[series_file]
        writer.add(series)
    
    def put_observation(self, observation, timestamp, source, dataset):
        if self.is_virtual(dataset):
            raise DatasetNotWritableError('Dataset %s is virtual' % dataset)
        self._create_node(source, dataset)
        self._add_observation(dataset, source, timestamp, observation)

    def get_series(self, dataset, source, timestamp, quantity, stations=None, forecast_length=None):
        if self.ts_format == 'netcdf':
            return self._get_series_nc(dataset, source, timestamp, quantity, stations, forecast_length)
        else:
            return TimeseriesDatabase.get_series(dataset, source, timestamp, stations, forecast_length)
        

    def _get_series_nc(self, dataset, source, timestamp, quantity, stations=None, forecast_length=None):
        """
        Retrieve a (possibly empty) list of series defined by dataset, source, timestamp
        and quantity. If stations is omitted, all stations are accepted. A forecast length can be given
        - in this case, only read data between timestamp and timestamp+forecast_length. 
        """
        series_file = self._get_series_file(dataset, source, quantity, timestamp)
        if not path.exists(series_file):
            return []
        if stations is None:
            station_file = self._get_station_file(dataset, timestamp)
            if not station_file:
                return []
            stations_read \
                = timeseries.readStations(station_file,
                                          columns=TimeseriesDatabase._station_format,
                                          separator=TimeseriesDatabase._station_separator)
            stations = stations_read.values()
        if forecast_length:
            t2 = timestamp + forecast_length
        else:
            t2 = None
        reader = timeseries.NCIO(series_file, 'r')
        # If stations is None, NCIO will get stations from
        # the netcdf and disregard the virtual dataset!
        assert stations is not None 
        series_list = list(reader.iter_series(stations, t2=t2))
        return series_list
        
    def get_observations(self, dataset, source, timestamp, quantity, obs_class,
                         forecast_length=None, obs_kwargs={}):
        if self.is_virtual(dataset):
            raise TimeseriesDatabaseError('Cannot get_observations for virtual dataset')
        series_file_ptrn = self._get_obs_file(dataset, source, quantity, timestamp, ('*',))
        series_files = glob.glob(series_file_ptrn)
        #print series_files
        if forecast_length:
            time_end = timestamp + forecast_length
        else:
            time_end = None

        series_list = []
        for series_file in series_files:
            print( 'observations from file: ', series_file, time_end, obs_kwargs)
            series_list.extend(obs_class.fromFile(series_file, time_end=time_end, **obs_kwargs))
            #print len(series_list)
        return series_list
    
    def flush(self):
        filenames = self._tswriters.keys()
        for filename in filenames:
            writer = self._tswriters[filename]
            writer.close()
            del(self._tswriters[filename])
            
    
    def release(self):
        if self._active_file[0]:
            self._active_file[1].close()
        self.flush()
        TimeseriesDatabase.release(self)

    def create_virtual_dataset(self, dataset_virt, dataset_real):
        """
        Create a virtual dataset: the data files link to another set, but the stations are
        a subset. Each subset must be defined separately for the each timestamp.
        """
        dataset_dir = path.join(self.root, dataset_virt)
        if path.exists(dataset_dir):
            if self.is_virtual(dataset_virt) and self._get_real_dataset(dataset_virt) != dataset_real:
                raise TimeseriesDatabaseError('Virtual dataset %s exists but links to %s' %
                                              (dataset_virt, self._get_real_dataset(dataset_virt)))
            elif not self.is_virtual(dataset_virt):
                raise TimeseriesDatabaseError('Non-virtual dataset %s exists' %
                                              (dataset_virt))
            else:
                return # all ok

        os.makedirs(path.join(dataset_dir, 'stations'))
        os.symlink(path.join('..', dataset_real, 'series'), path.join(dataset_dir, 'series'))
        #os.symlink(path.join('..', dataset_real, 'stations'), path.join(dataset_dir, 'stations'))
        #with open(path.join(dataset_dir, 'station_subset'), 'w') as fileout:
        #    for station in stations:
        #        station.toFile(fileout, separator=self._station_separator)
        with open(path.join(dataset_dir, 'real_dataset'), 'w') as fileout:
            fileout.write(dataset_real)

    def set_stations(self, dataset_virt, timestamp, stations):
        if not self.is_virtual(dataset_virt):
            raise TimeseriesDatabaseError('Dataset is not virtual: %s' % dataset_virt)
        dataset_real = self._get_real_dataset(dataset_virt)
        stations_real = self.get_stations([dataset_real], [timestamp])
        for station in stations:
            if not station in stations_real:
                raise TimeseriesDatabaseError('Station not found: %s' % station.code)
            self._add_station(dataset_virt, timestamp, station)
    
    def is_virtual(self, dataset):
        station_list_file = path.join(self.root, dataset, 'real_dataset')
        return path.exists(station_list_file)

    def _get_real_dataset(self, dataset_virt):
        real_file = path.join(self.root, dataset_virt, 'real_dataset')
        with open(real_file, 'r') as filein:
            dataset_real = filein.next()
        return dataset_real
    
    # def _get_stations(self, dataset, timestamp):
    #     if self.is_virtual(dataset):
    #         subset = set(self._get_stations_subset(dataset).itervalues())
    #         dataset_virt = dataset
    #         dataset = self._get_real_dataset(dataset_virt)
    #     else:
    #         subset = None
    #     station_file = self._get_station_file(dataset, timestamp)
    #     if not path.exists(station_file):
    #         return set()
    #     stations_read = self._read_stations(station_file).itervalues()
    #     if subset is not None:
    #         stations = set(station for station in stations_read if station in subset)
    #     else:
    #         stations = set(stations_read)
    #     return stations
            
    # def get_stations(self, datasets=None, timestamps=None):
    #     if not datasets:
    #         datasets = self._get_datasets()
    #     if not timestamps:
    #         timestamps = set()
    #         for dset in datasets:
    #             timestamps.update(self._get_timestamps(dset))
    #     stations = set()
    #     for dataset in datasets:
    #         for timestamp in timestamps:
    #             stations.update(self._get_stations(dataset, timestamp))
    #     return stations
                                
if __name__ == '__main__':
    import pylab, numpy as np
    tsdb = TimeseriesDatabase('./tsdb')
    tsdb.purge()
    tsdb2 = TimeseriesDatabase('./tsdb', 1)
    length = 3*24
    data = pylab.rand(length)
    #print data  
    time_0 = datetime.datetime(2008,1,2)
    one_hour = datetime.timedelta(hours=1)
    one_day = datetime.timedelta(days=1)
    stamp = datetime.datetime(2008,1,2)
    
    stations = (timeseries.Station('s1', 'station 1', 25.0, 56.0, 
                                 area_type='urban', dominant_source='industrial'),
                timeseries.Station('s2', 'station 2', 25.0, 52.0, 
                                 area_type='urban', dominant_source='background'),
                timeseries.Station('s3', 'station 3', 28.0, 53.0, 
                                 area_type='urban', dominant_source='background'))
    sources = 'src1', 'src2'
    species = 'CO', 'NO2'
    datasets = 'dset1', 'dset2'

    for itime in xrange(4):
        if itime == 2:
            continue
        times = [time_0 + itime*one_day + i*one_hour for i in xrange(length)]
        stamp = time_0 + itime*one_day
        for ds in datasets:
            for src in sources:
                for q in species:
                    for st in stations:
                        data = pylab.rand(length)
                        series = timeseries.Timeseries(data, times, st, quantity=q)
                        tsdb.put_series(series, stamp, src, ds)
    
    ss = tsdb.get_series('dset1', 'src1', time_0+one_day, 'CO', (stations[0],))
    #print ss
    stations = tsdb.get_stations()
    selected = list(filtered_stations(stations, dominant_sources=('background',)))
    #selected = tsdb.get_stations(station_types=('urban background',))
    print (selected)
    seq = tsdb.get_sequence('dset1', 'src1', (time_0+i*one_day for i in xrange(4)), selected,
                            'CO')
    st = seq.keys()[0]
    for ss in seq[st]:
        ss.plot()
    tsdb.export_series('dset1', 'src1', time_0+one_day, 'NO2', 'exp1')

    #tsdb.release()
