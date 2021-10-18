import re, os

"""
namlist.py - handle SILAM namelists

The module reads and writes SILAM namelists. The following formats are considered:

- only named lines in a file: use Namelist.fromfile to read a single namelist

- namelist group: multiple namelists surrounded by LIST = / END_LIST =, with unique list
  names: use NamelistGroup.fromfile to read a namelist group

- namelist group with repeating list names: use Namelist.lists_from_lines, see NamelistGroup for an example

- namelists with arbitrary limits: use Namelist.fromlines with stop_line and start_line. 



"""

# With escape backslash -- intelligent
#confparser = re.compile(r"^\s*([a-zA-Z_][a-zA-Z_0-9]*)\s*=\s*((?:[^\\#!\r]|\\.)+)")
# should be modified to handle quoted strings?

#without escape backslash -- faster
#confparser = re.compile(r"^\s*([a-zA-Z_][a-zA-Z_0-9]*)\s*=\s*(|[^#^!^\r]*[^#^!^ ^\n^\r]).*$")
#confparser = re.compile(r"^\s*([a-zA-Z_][a-zA-Z_0-9]*)\s*=\s*([^#^!^\r^\n]*\W)\s*$")
#  attr4 = ! also blank


# allow empty values, avoid trailing newlines or blanks  
#confparser = re.compile(r"^\s*([a-zA-Z_][a-zA-Z_0-9]*)\s*=\s*(|[^#^!]*[^#^!^ ^\n])\s*(?:[!#].*|$)")
#confparser = re.compile(r"^\s*([a-zA-Z_][a-zA-Z_0-9]*)\s*=\s*(|[^#^!]*[^#^!^\S])\s*(?:[!#].*|$)")
# WRONG: confparser = re.compile("^\s*([a-zA-Z_][a-zA-Z_0-9]*)\s*=\s*(\\^?[^#^!]*[^#^!^ ^\r^\n]|).*$")
confparser = re.compile(r"^\s*([a-zA-Z_][a-zA-Z_0-9]*)\s*=\s*([^#!]*[^#! \r\n]|).*$")
# key, vale = confparser.match(line).groups()
# strips comments

class NamelistGroup:
    """
    A group of namelists. Contrary to namelist itself, duplicate keys are not allowed.
    """
    @staticmethod
    def fromfile(filein, nl=None, groupname=None):
        """
        Create a namelist group from an iterable.
        """
        if isinstance(filein, str):
            nlfile = open(filein, 'r')
            filename = filein
            close = True
        else:
            nlfile = filein
            close = False
            try:
                filename = filein.name
            except AttributeError:
                filename = ''
            
        if not groupname:
            groupname = filename
        nlGrp = NamelistGroup(groupname)
        line = True
        count = 0

        for nl in Namelist.lists_from_lines(nlfile):
            count += 1
            if nl.name in nlGrp.lists:
                raise NamelistError('List already in group: %s' % nl.name)
            nlGrp.put(nl.name, nl)

        if close:
            nlfile.close()
        return nlGrp
    
    def __init__(self, name):
        self.name = name
        self.lists = {}

    def put(self, nlname, nl):
        self.lists[nlname] = nl

    def get(self, nlname):
        return self.lists[nlname]

    def keys(self):
        return self.lists.keys()

    def values(self):
        return self.lists.values()


class NoListFoundError(Exception):
    pass

class Namelist:
    def _update_from_lines(self, read_lines, stop_line=None, ignore_line=None, forbidden_line=None):
        # update the list reading lines from iterable
        empty = True
        for line in read_lines:
            if ignore_line and line.lstrip().startswith(ignore_line):
                continue
            if stop_line and line.lstrip().startswith(stop_line):
                return line
            if forbidden_line and line.lstrip().startswith(forbidden_line):
                raise NamelistError('Invalid entry in this context: %s' % line)
            while '${' in line:
                env_var_idxStart = line.find('${')
                env_var_idxEnd = line.find('}', env_var_idxStart)
                envKey = line[env_var_idxStart+2 : env_var_idxEnd]
                env_var = os.getenv(env_key)
                if env_var:
                    line = line.replace(env_key, env_var)  #[:env_var_idxStart] + env_var + line[env_var_idxEnd+1:]
                else:
                    line = line.replace(env_key,'')  # [:env_var_idxStart] + line[env_var_idxEnd+1:]
            match = confparser.match(line)
            if not match:
                continue
            key, val = match.groups()
            self.add(key, val)
            empty = False
        if stop_line and not empty:
            raise NamelistError('stop_line given but not found: %s' % stop_line)
        return not empty
        
    @staticmethod
    def fromlines(read_lines, nlname=None, stop_line=None, start_line=None):
        """
        Initialize namelist from an iterable. If given, stop reading when the line startswith
        stop_line. Lines starting with start_line are ignored.
        """
        nl = Namelist(nlname)
        if not nl._update_from_lines(iter(read_lines), stop_line, ignore_line=start_line):
            raise NoListFoundError()
        return nl
    
    @staticmethod
    def fromfile(inputfile, nlname=None, stop_line=None, start_line=None):
        """
        Create a single namelist (optionally with name given) from a file (path or
        object). Other arguments:
        - stop_line - stop reading when line.startswith(stop_line)
        - start_line - ignore this line while reading.
        """
        if isinstance(inputfile, str):
            nlfile = open(inputfile, 'r')
            close = True
        else:
            nlfile = inputfile
            close = False
        nl = Namelist.fromlines(nlfile, nlname, stop_line, start_line)
        if close:
            nlfile.close()
        return nl
        
    @staticmethod
    def lists_from_lines(read_lines):
        """
        Iterate over namelists limited by LIST = , END_LIST = given in a file.
        """
        # read_lines must be iterator: if it is eg. list, the nested fors don't work correctly! 
        read_lines = iter(read_lines)
        count = 0
        for line in read_lines:
            count += 1
            match = confparser.match(line)
            if not match:
                continue
            # find the LIST = line
            key, val = match.groups()
            if not key == 'LIST':
                raise NamelistError('Item outside group: %s' % val)
            nl = Namelist(val)
            end_list_line = nl._update_from_lines(read_lines, stop_line='END_LIST', forbidden_line='LIST')
            match_end_list = confparser.match(end_list_line)
            if not match:
                raise NamelistError('Bad END_LIST line: %s' % end_list_line)
            key, val = match_end_list.groups()
            if not key == 'END_LIST':
                # shouldn't happen because END_LIST = stop_line
                raise NamelistError('Really strange END_LIST line: %s' % end_list_line)
            if not val == nl.name:
                raise RunawayNamelistError(nl.name, val)
            yield nl
            
    
    def __init__(self, name):
        self.name = name
        self.hash = {}

    def __contains__(self, key):
        return key in self.hash
    
    def __iter__(self):
        return iter(self.hash)

    def __len__(self):
        return len(self.hash)
    
    def put(self, key, val):
        # now just alias for set.
        self.set(key, val)
        #self.hash[key] = [val]

    def set(self, key, val):
        """
        Assign key -> val, overwriting any previous definition.
        """
        if isinstance(val, str):
            self.hash[key] = [val]
        else:
            try:
                # maybe val is iterable
                self.hash[key] = list(val)
            except TypeError:
                # probably still scalar
                self.hash[key] = [val]
    def add(self, key, val):
        """
        Assign key -> val, extending the previous definition.
        """
        if not key in self.hash:
            self.hash[key] = [val]
        else:
            self.hash[key].append(val)

    def get(self, key):
        try: return self.hash[key]
        except: return []

    def get_uniq(self, key):
        val = self.hash[key]
        if len(val) > 1:
            raise NamelistError('Single required but multiple defined for %s' % key)
        return val[0]

    def get_float(self, key):
        return [float(val) for val in self.hash[key]]

    def get_int(self, key):
        return [int(val) for val in self.hash[key]]

    def keys(self):
        return self.hash.keys()

    def values(self):
        return self.hash.values()

    def has_key(self, key):
        return key in self.hash

    def extend(self, namelist2):
        """ Join namelist 2 into this list. """
        for key in namelist2.keys():
            for val in namelist2.get(key):
                self.add(key, val)
    
    def substitute(self, substitutions):
        # Replaces all occurrences of each expression in all namelist items
        for nlKey, nlVals in self.hash.items():  # nlVals is a list responding to the nlKey
            newNlVals = []
            for nlVal in nlVals:                 # individual value in namelist
                for (substKey, substVal) in substitutions.items():  # anything from substitution?
                    if substKey in nlVal:
#                        print(substKey, substVal)
                        nlVal = nlVal.replace(substKey, str(substVal))
#                        print(nlVal)
                newNlVals.append(nlVal)
            self.hash[nlKey] = newNlVals

    def tofile(self, outputfile, listname=None, append=False, exceptKey=None):
        if append: 
            mode = 'a'
        else:
            mode = 'w'

        try:
            outputfile.write
            fh = outputfile
        except AttributeError:
            fh = open(outputfile, mode)
        
        if listname is not None:
            fh.write('LIST = %s\n' % listname)
        for key, val in sorted(self.hash.items()):
            if exceptKey is not None:
                if key == exceptKey: continue
            for entry in val:
                fh.write('%s = %s\n' % (key, entry))
        if listname is not None:
            fh.write('END_LIST = %s\n\n' % listname)

    def tostr(self):
        lines = []
        for key, val in self.hash.items():
            for entry in val:
                lines.append('%s = %s' % (key, entry))
        return '\n'.join(lines)

    def todictionary(self):
        dic = {}
        for key, val in sorted(self.hash.items()):
            dic[key] = []
            for entry in val:
                dic[key].append(entry)
        return dic
        
        
class NamelistError(Exception):
    pass

class RunawayNamelistError(NamelistError):
    def __init__(self, needed, given):
        NamelistError('List %s terminated with END_LIST = %s' % (needed, given))


