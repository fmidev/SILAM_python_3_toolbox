import datetime as dt, numpy as np

def almost_eq(x, y, eps=1e-15):
    return np.abs(x-y) < eps

def dt2sec(delta):
    return delta.days*24*3600 + delta.seconds

zero_interval = dt.timedelta(0)
one_day = dt.timedelta(days=1)
one_hour = dt.timedelta(hours=1)

try:
    from matplotlib import cbook
    is_sequence_of_strings = cbook.is_scalar_or_string  #(val).is_sequence_of_strings
except ImportError:
    def is_sequence_of_strings(seq):
        try:
            seq.__iter__
        except AttributeError:
            return False
        for ss in seq:
            if not isinstance(basestring, ss):
                return False
        return True


def iterate_monthly(first_time, count):
    now = first_time
    for counter in xrange(count):
        yield now
        if now.month < 12:
            now = now.replace(month=now.month+1)
        else:
            now = now.replace(year=now.year+1, month=1)
        
    
def iterate_year_hourly(year):
    now = dt.datetime(year, 1, 1)
    while now.year == year:
        yield now
        now += one_hour
def iterate_year_daily(year):
    now = dt.datetime(year, 1, 1)
    while now.year == year:
        yield now
        now += one_day

def next_month(date):
    if date.month < 12:
        return dt.datetime(date.year, date.month+1, 1)
    else:
        return dt.datetime(date.year+1, 1, 1)
def prev_month(date):
    if date.month > 1:
        return dt.datetime(date.year, date.month-1, 1)
    else:
        return dt.datetime(date.year-1, 12, 1)
        
def time_range(first, end, step):
    now = first
    while now < end:
        yield now
        now += step

def points_to_edges(pt):
    pt = np.asarray(pt)
    delta = pt[1]-pt[0]
    if not np.all(almost_eq(pt[1:]-pt[:-1], delta, 1e-4)):
        print('########## Midpoints not equidistant. Delta=', delta)
        print(pt)
        print(pt[1:]-pt[:-1])
        raise ValueError('Midpoints not equidistant')
    return np.arange(pt[0]-delta/2, pt[-1]+delta, delta)

def get_chunk(array, ind_chunk, num_chunks):
    if ind_chunk >= num_chunks:
        raise ValueError('ind_chunk >= num_chunks')
    arr_size = len(array)
    if num_chunks > arr_size:
        raise ValueError('num_chunks > arr_size')
    
    chunk_size = arr_size / num_chunks
    remainder = arr_size - num_chunks*chunk_size
    if ind_chunk < remainder:
        chunk_size += 1
        ind_start = ind_chunk*chunk_size
    else:
        ind_start = remainder*(chunk_size+1) + (ind_chunk-remainder)*chunk_size
    return array[ind_start:ind_start + chunk_size]

class Workshare:
    def __init__(self, string=None):
        if string is None:
            self.ind_task, self.num_tasks = 0, 1
            return
        
        args = string.split(':')
        self.ind_task, self.num_tasks = int(args[0]), int(args[1])
        if self.ind_task < 0:
            raise ValueError('Index of task cannot be < 0')
        if self.num_tasks < 0:
            raise ValueError('Number of tasks cannot be < 0')
        if self.ind_task >= self.num_tasks:
            raise ValueError('Index of task cannot be >= number of tasks')
        
    def __getitem__(self, ind):
        # keep compatibility with the old style (below).
        if ind == 0:
            return self.ind_task
        elif ind == 1:
            return self.num_tasks
        else:
            raise IndexError('Bad index %i' % ind)

    def get_chunk(self, array):
        return get_chunk(array, self.ind_task, self.num_tasks)
        
def parse_workshare(string):
    args = string.split(':')
    ind_chunk, num_chunks = int(args[0]), int(args[1])
    return ind_chunk, num_chunks


