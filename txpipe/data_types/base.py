import uuid
import datetime
import socket
import getpass
import warnings

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
    def __init__(self, path, mode, validate=True, write_identifier=True, **kwargs):
        self.path = path
        self.mode = mode
        self.file = self.open(path, mode, **kwargs)

        if validate and mode == 'r':
            self.validate()

        if mode == 'w':
            self.provenance = self.generate_provenance()
            self.write_provenance()
        else:
            self.provenance = self.read_provenance()

    @staticmethod
    def generate_provenance():
        """
        Generate provenance information - a dictionary
        of useful information about the origina 
        """
        UUID = uuid.uuid4().hex
        creation_date = datetime.datetime.now().isoformat()
        domain = socket.getfqdn()
        user = getpass.getuser()

        # Add other provenance and related 
        return {
            'uuid': UUID,
            'creation': creation_date,
            'domain': domain,
            'username': user,
            }

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
        """
        pass

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
        self._provenance_group = self.file.create_group('provenance')

        # Call the sub-method to do each item
        for key, value in self.provenance.items():
            self.add_provenance(key, value)

    def add_provenance(self,  key, value):
        """
        Add a single item of provenance data to the file
        """
        if self.mode == 'r':
            raise ValueError("Cannot write provenance to a file opened in read-only mode")

        if self._provenance_group is None:
            warnings.warn(f"Unable to save provenance information to file {self.path}")
            return
        print(f"Writing provenance item {key} = {value}")
        self._provenance_group.attrs[key] = value

    def read_provenance(self):
        try:
            group = self.file['provenance']
            attrs = group.attrs
        except KeyError:
            group = None
            attrs = {}

        self._provenance_group = group
        
        provenance = {
            'uuid':     attrs.get('uuid', "UNKNOWN"),
            'creation': attrs.get('creation_date', "UNKNOWN"),
            'domain':   attrs.get('domain', "UNKNOWN"),
            'username': attrs.get('user', "UNKNOWN"),
        }

        print(f"Read provenance {provenance} for file {self.path}")
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
        for key, value in self.provenance.items():
            self.add_provenance(key, value)

    def add_provenance(self,  key, value):
        """
        Add a single item of provenance data to the file
        """
        if self.mode == 'r':
            raise ValueError("Cannot write provenance to a file opened in read mode")

        self.file[0].write_key(key, value)

    def read_provenance(self):
        header = self.file[0].read_header()
        provenance = {
            'uuid':     header.get('uuid', 'UNKNOWN'),
            'creation': header.get('creation_date', 'UNKNOWN'),
            'domain':   header.get('domain', 'UNKNOWN'),
            'username': header.get('user', 'UNKNOWN'),
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
    """
    suffix = 'yml'

