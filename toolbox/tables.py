import collections, numpy as np
from toolbox import timeseries
import copy
class Table:
    """ This class defines a 2D array with named columns and rows. The
    column and row keys can be arbitrary hashable objects. For
    non-string keys, the row_formatter and column_formatter attributes
    need to be set.

    The implemetation includes methods for extracting individual
    values, rows and columns."""
    
    def __init__(self, rows=None, columns=None, value_format='%-12s', field_width=12,
                 first_column_width=None, missing=np.nan):
        """Create a Table. While the column and row keys can be set here, they
        can be added on the fly as well."""
        if not columns:
            self.columns = []
        else:
            self.columns = list(columns)
        if not rows:
            self.rows = set()
        else:
            self.rows = set(rows)
        if not first_column_width:
            self.first_column_width = field_width
        else:
            self.first_column_width = first_column_width
        self.values = collections.defaultdict(lambda: missing)
        self.value_format = value_format
        self.field_width = field_width
        self.row_formatter = str
        self.column_formatter = str
         
    def to_stream(self, stream):
        """Write the table into a writable object (e.g. file)."""
        stream.write(self.first_column_width*' ')
        string_format = '%%%is' % self.field_width
        string_format_wide = '%%-%is' % self.first_column_width
        for column in self.columns:
            stream.write(string_format % self.column_formatter(column))
        stream.write('\n')
        for row in self.sorted():
            stream.write(string_format_wide % self.row_formatter(row))
            for column in self.columns:
                stream.write(self.value_format % self.values[(row,column)])
            stream.write('\n')

    @staticmethod
    def fromfile(filename, dtype=float):
        """Read a table from a given file. The values will be coarced into dtype."""
        handle = open(filename)
        columns = handle.next().split()
        table = Table(columns=columns)
        for line in handle:
            stripped = line.strip()
            if stripped.startswith('#'):
                continue
            split = stripped.split()
            row = split[0]
            table.add_row(row)
            for col, val in zip(columns, split[1:]):
                table.values[(row, col)] = dtype(val)
        handle.close()
        return table
             
    def add_row(self, row):
        self.rows.add(row)

    def add_column(self, column):
        self.columns.append(column)

    def remove_row(self, row):
        self.rows.remove(row)
        ifRepeat = False
        while ifRepeat:
            ifRepeat = False
            keyslist = self.values.keys()
            for key in keyslist:
                if key[0] == row:
                    del(self.values[key])
                    ifRepeat = True
        
    def get(self, row, column):
        return self.values[(row, column)]
         
    def get_column(self, column):
        """Return the values in a column, in a list sorted by the row key."""
        return [self.values[(row,column)] for row in self.sorted()]

    def get_row(self, row):
        """Return the values in a row, in a list sorted by the column key."""
        return [self.values[(row,column)] for column in self.sorted_columns()]

    def sorted(self):
        """Return the rows sorted after applying self.row_formatter."""
        return sorted(self.rows, key=self.row_formatter)

    def sorted_columns(self):
        """Return the columns sorted after applying self.column_formatter."""
        return sorted(self.columns, key=self.column_formatter)
     
    def transpose(self):
        """Return a Table with columns and rows swapped."""
        transp = Table(self.columns, self.rows, self.value_format, 
                       self.field_width)
        for row, col in self.values:
            transp.values[(col,row)] = self.values[(row,col)]
        return transp

    def aggregate(self, rows, aggregation, aggregator):
        aggregate = Table(columns=self.columns, rows=aggregation)
        for col in self.columns:
            values = [self.values[(row, col)] for row in rows]
            value = aggregator(values)
            aggregate.values[(aggregation, col)] = value
        return aggregate

    def __iter__(self):
        """Iterate over rows, in arbitrary order."""
        return self.rows.__iter__()

    def set(self, row, column, value):
        """Set a value for a given row and column. The corresponding keys are
        added if necessary."""
        if not row in self.rows:
            self.add_row(row)
        if not column in self.columns:
            self.add_column(column)
        self.values[(row, column)] = value

    def get(self, row, column):
        return self.values[(row, column)]

    def select(self, rows=None, columns=None):
        """
        Return a Table with the given rows and columns from this table.
        """

        if rows is None:
            rows = self.rows
        if columns is None:
            columns = self.columns
        
        selected = Table(set(rows), set(columns), self.value_format, self.field_width)
        selected.column_formatter = self.column_formatter
        selected.row_formatter = self.row_formatter
        for (col, row), val in self.values.items():
            selected.values[(col,row)] = val
            #foo = self.values[(col,row)]
        return selected


    
