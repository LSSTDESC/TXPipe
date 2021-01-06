import uuid
import datetime
import socket
import getpass
import warnings
import pathlib
import yaml
import shutil
import pickle

class FileValidationError(Exception):
    pass

class DataFile:
    """
    A class representing a DataFile to be made by pipeline stages
    and passed on to subsequent ones.

    DataFile itself should not be instantiated - instead subclasses
    should be defined for different file types.

    These subclasses are used in the definition of pipeline stages
    to indicate what kind of file is expected.  The "suffix" attribute,
    which must be defined on subclasses, indicates the file suffix.

    The open method, which can optionally be overridden, is used by the 
    machinery of the PipelineStage class to open an input our output
    named by a tag.

    """
    supports_parallel_write = False
    suffix = None
    def __init__(self, path, mode, extra_provenance=None, validate=True, **kwargs):
        self.path = path
        self.mode = mode
        self.file = self.open(path, mode, **kwargs)

        if validate and mode == 'r':
            self.validate()

        if mode == 'w':
            self.provenance = self.generate_provenance(extra_provenance)
            self.write_provenance()
        else:
            self.provenance = self.read_provenance()

    @staticmethod
    def generate_provenance(extra_provenance=None):
        """
        Generate provenance information - a dictionary
        of useful information about the origina 
        """
        UUID = uuid.uuid4().hex
        creation = datetime.datetime.now().isoformat()
        domain = socket.getfqdn()
        username = getpass.getuser()

        # Add other provenance and related 
        provenance = {
            'uuid': UUID,
            'creation': creation,
            'domain': domain,
            'username': username,
            }

        if extra_provenance:
            provenance.update(extra_provenance)
        return provenance

    def write_provenance(self):
        """
        Concrete subclasses (for which it is possible) should override
        this method to save the dictionary self.provenance to the file.
        """
        pass

    def add_provenance(self, key, value):
        """
        Concrete subclasses (for which it is possible) should override
        this method to save the a new string key/value pair to file
        """
        pass


    def read_provenance(self):
        """
        Concrete subclasses for which it is possible should override
        this method and read the provenance information from the file.

        Other classes will return this dictionary of UNKNOWNs
        """
        provenance = {
            'uuid':     "UNKNOWN",
            'creation': "UNKNOWN",
            'domain':   "UNKNOWN",
            'username': "UNKNOWN",
        }

        return provenance


    def validate(self):
        """
        Concrete subclasses should override this method
        to check that all expected columns are present.
        """
        pass

    @classmethod
    def open(cls, path, mode):
        """
        Open a data file.  The base implementation of this function just
        opens and returns a standard python file object.

        Subclasses can override to either open files using different openers
        (like fitsio.FITS), or, for more specific data types, return an
        instance of the class itself to use as an intermediary for the file.

        """
        return open(path, mode)

    def close(self):
        self.file.close()

    @classmethod
    def make_name(cls, tag):
        if cls.suffix:
            return f'{tag}.{cls.suffix}'
        else:
            return tag

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

class HDFFile(DataFile):
    supports_parallel_write = True
    """
    A data file in the HDF5 format.
    Using these files requires the h5py package, which in turn
    requires an HDF5 library installation.

    """
    suffix = 'hdf5'
    required_datasets = []

    @classmethod
    def open(cls, path, mode, **kwargs):
        # Suppress a warning that h5py always displays
        # on import
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import h5py
        # Return an open h5py File
        return h5py.File(path, mode, **kwargs)

    def write_provenance(self):
        """
        Write provenance information to a new group,
        called 'provenance'
        """
        if self.mode == 'r':
            raise ValueError("Cannot write provenance to a file opened in read-only mode")

        # This method *must* be called by all the processes in a parallel
        # run.  
        self._provenance_group = self.file.create_group('provenance')

        # Call the sub-method to do each item
        for key, value in self.provenance.items():
            self._provenance_group.attrs[key] = value

    def read_provenance(self):
        try:
            group = self.file['provenance']
            attrs = group.attrs
        except KeyError:
            group = None
            attrs = {}
        
        provenance = {
            'uuid':     attrs.get('uuid', "UNKNOWN"),
            'creation': attrs.get('creation', "UNKNOWN"),
            'domain':   attrs.get('domain', "UNKNOWN"),
            'username': attrs.get('username', "UNKNOWN"),
        }

        self._provenance_group = group

        return provenance



    def validate(self):
        missing = [name for name in self.required_datasets if name not in self.file]
        if missing:
            text = "\n".join(missing)
            raise FileValidationError(f"These data sets are missing from HDF file {self.path}:\n{text}")

    def close(self):
        self.file.close()



class FitsFile(DataFile):
    """
    A data file in the FITS format.
    Using these files requires the fitsio package.
    """
    suffix = 'fits'
    required_columns = []

    @classmethod
    def open(cls, path, mode, **kwargs):
        import fitsio
        # Fitsio doesn't have pure 'w' modes, just 'rw'.
        # Maybe we should check if the file already exists here?
        if mode == 'w':
            mode = 'rw'
        return fitsio.FITS(path, mode=mode, **kwargs)

    def missing_columns(self, columns, hdu=1):
        """
        Check that all supplied columns exist
        and are in the chosen HDU
        """
        ext = self.file[hdu]
        found_cols = ext.get_colnames()
        missing_columns = [col for col in columns if col not in found_cols]
        return missing_columns


    def write_provenance(self):
        """
        Write provenance information to a new group,
        called 'provenance'
        """
        # Call the sub-method to do each item
        if self.mode == 'r':
            raise ValueError("Cannot write provenance to a file opened in read mode")

        for key, value in self.provenance.items():
            if isinstance(value, str) and '\n' in value:
                values = value.split("\n")
                for i,v in enumerate(values):
                    self.file[0].write_key(key+f"_{i}", v)
            else:
                self.file[0].write_key(key, value)


    def read_provenance(self):
        header = self.file[0].read_header()
        provenance = {
            'uuid':     header.get('uuid', 'UNKNOWN'),
            'creation': header.get('creation', 'UNKNOWN'),
            'domain':   header.get('domain', 'UNKNOWN'),
            'username': header.get('username', 'UNKNOWN'),
        }
        return provenance



    def validate(self):
        """Check that the catalog has all the required columns and complain otherwise"""
        # Find any columns that do not exist in the file
        missing = self.missing_columns(self.required_columns)

        # If there are any, raise an exception that lists them explicitly
        if missing:
            text = "\n".join(missing)
            raise FileValidationError(f"These columns are missing from FITS file {self.path}:\n{text}")

    def close(self):
        self.file.close()


class TextFile(DataFile):
    """
    A data file in plain text format.
    """
    suffix = 'txt'

class YamlFile(DataFile):
    """
    A data file in yaml format.
    The top-level object in TXPipe YAML
    files should always be a dictionary.
    """
    suffix = 'yml'

    def __init__(self, path, mode, extra_provenance=None, validate=True, load_mode='full'):
        self.path = path
        self.mode = mode
        self.file = self.open(path, mode)

        if mode == "r":
            if load_mode == 'safe':
                self.content = yaml.safe_load(self.file)
            elif load_mode == 'full':
                self.content = yaml.full_load(self.file)
            elif load_mode == 'unsafe':
                self.content = yaml.unsafe_load(self.file)
            else:
                raise ValueError(f"Unknown value {yaml_load} of load_mode. "
                                  "Should be 'safe', 'full', or 'unsafe'")
            # get provenance
            self.provenance = self.read_provenance()

        else:
            self.provenance = self.generate_provenance(extra_provenance)
            self.write_provenance()

    def read(self, key):
        return self.content[key]

    def write(self, d):
        if not isinstance(d, dict):
            raise ValueError("Only dicts should be passed to YamlFile.write")
        yaml.dump(d, self.file)

    def write_provenance(self):
        d = {'provenance': self.provenance}
        self.write(d)

    def read_provenance(self):
        prov = self.content.pop('provenance', {})
        req_provenance = {
                'uuid':     prov.get('uuid', "UNKNOWN"),
                'creation': prov.get('creation', "UNKNOWN"),
                'domain':   prov.get('domain', "UNKNOWN"),
                'username': prov.get('username', "UNKNOWN"),
        }
        prov.update(req_provenance)
        return prov


class Directory(DataFile):
    suffix = ''

    @classmethod
    def open(self, path, mode):
        p = pathlib.Path(path)

        if mode == "w":
            if p.exists():
                shutil.rmtree(p)
            p.mkdir(parents=True)
        else:
            if not p.is_dir():
                raise ValueError(f"Directory input {path} does not exist")
        return p

    def write_provenance(self):
        """
        Write provenance information to a new group,
        called 'provenance'
        """
        # This method *must* be called by all the processes in a parallel
        # run.  
        if self.mode == 'r':
            raise ValueError("Cannot write provenance to a directory opened in read-only mode")

        self._provenance_file = open(self.file / 'provenance.yml', 'w')

        # Call the sub-method to do each item
        yaml.dump(self.provenance, self._provenance_file)
        self._provenance_file.flush()


    def read_provenance(self):
        try:
            f = open(self.file / 'provenance.yml')
            attrs = yaml.load(f)

        except KeyError:
            f = None
            attrs = {}

        self._provenance_file = f
        
        provenance = {
            'uuid':     attrs.get('uuid', "UNKNOWN"),
            'creation': attrs.get('creation', "UNKNOWN"),
            'domain':   attrs.get('domain', "UNKNOWN"),
            'username': attrs.get('username', "UNKNOWN"),
        }

        return provenance

class FileCollection(Directory):
    """
    Represents a grouped bundle of files, for cases where you don't
    know the exact list in advance.
    """
    suffix = ''

    def write_listing(self, filenames):
        """
        Write a listing file in the directory recording
        (presumably) the filenames put in it.
        """
        fn = self.path_for_file('txpipe_listing.txt')
        with open(fn, 'w') as f:
            yaml.dump(filenames, f)

    def read_listing(self):
        """
        Read a listing file from the directory.
        """
        fn = self.path_for_file('txpipe_listing.txt')
        with open(fn, 'w') as f:
            filenames = yaml.safe_load(f)
        return filenames

    def path_for_file(self, filename):
        """
        Get the path for a file inside the collection.
        Does not check if the file exists or anything like
        that.
        """
        return str(self.file / filename)


class PNGFile(DataFile):
    suffix = 'png'

    @classmethod
    def open(self, path, mode, **kwargs):
        import matplotlib
        import matplotlib.pyplot as plt
        if mode != "w":
            raise ValueError("Reading existing PNG files is not supported")
        return plt.figure(**kwargs)


    def close(self):
        import matplotlib.pyplot as plt
        self.file.savefig(self.path, metadata=self.provenance)
        plt.close(self.file)

    def write_provenance(self):
        # provenance is written on closing the file
        pass

    def read_provenance(self):
        raise ValueError("Reading existing PNG files is not supported")

class PickleFile(DataFile):
    suffix = "pkl"

    @classmethod
    def open(self, path, mode, **kwargs):
        return open(path, mode + "b")

    def write_provenance(self):
        self.write(self.provenance)

    def read_provenance(self):
        return self.read()

    def write(self, obj):
        if self.mode != "w":
            raise ValueError("Tried to write to read-only pickle file")
        pickle.dump(obj, self.file)

    def read(self):
        if self.mode != "r":
            raise ValueError("Tried to read from a write-only pickle file")
        return pickle.load(self.file)

