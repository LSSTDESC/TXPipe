from .base import FitsFile, HDFFile, DataFile, YamlFile, TextFile, Directory, PNGFile

def metacalibration_names(names):
    """
    Generate the metacalibrated variants of the inputs names,
    that is, variants with _1p, _1m, _2p, and _2m on the end
    of each name.
    """
    suffices = ['1p', '1m', '2p', '2m']
    out = []
    for name in names:
        out += [name + '_' + s for s in suffices]
    return out

class ShearCatalog(HDFFile):
    """
    A generic shear catalog 
    """
    # These are columns

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._catalog_type = None
    
    def read_catalog_info(self):
        try:
            group = self.file['shear']
            info = dict(group.attrs)
        except:
            raise ValueError(f"Unable to read shear catalog")
        shear_catalog_type = info.get('catalog_type')
        return shear_catalog_type

    @property
    def catalog_type(self):
        if self._catalog_type is not None:
            return self._catalog_type

        if 'catalog_type' in self.file['shear'].attrs:
            t = self.file['shear'].attrs['catalog_type']
        elif 'mcal_g1' in self.file['shear'].keys():
            t = 'metacal'
        elif 'c1' in self.file['shear'].keys():
            t = 'lensfit'
        else:
            raise ValueError("Could not figure out catalog format")

        self._catalog_type = t
        return t


class TomographyCatalog(HDFFile):
    required_datasets = []

    def read_zbins(self, bin_type):
        """
        Read saved redshift bin edges from attributes
        """
        d = dict(self.file['tomography'].attrs)
        nbin = d[f'nbin_{bin_type}']
        zbins = [(d[f'{bin_type}_zmin_{i}'], d[f'{bin_type}_zmax_{i}']) for i in range(nbin)]
        return zbins

    def read_nbin(self, bin_type):
        d = dict(self.file['tomography'].attrs)
        return d[f'nbin_{bin_type}']





class RandomsCatalog(HDFFile):
    required_datasets = ['randoms/ra', 'randoms/dec', 'randoms/e1', 'randoms/e2']




class MapsFile(HDFFile):
    required_datasets = []

    def read_healpix(self, map_name, return_all=False):
        import healpy
        import numpy as np
        group = self.file[f'maps/{map_name}']
        nside = group.attrs['nside']
        npix = healpy.nside2npix(nside)
        m = np.repeat(healpy.UNSEEN, npix)
        pix = group['pixel'][:]
        val = group['value'][:]
        m[pix] = val
        if return_all:
            return m, pix, nside
        else:
            return m

    def read_map_info(self, map_name):
        group = self.file[f'maps/{map_name}']
        info = dict(group.attrs)
        if not 'pixelization' in info:
            raise ValueError(f"Map '{map_name}' not found, "
                             f"or not saved properly in file {self.path}")
        return info

    def read_map(self, map_name):
        info = self.read_map_info(map_name)
        pixelization = info['pixelization']
        if pixelization == 'gnomonic':
            m = self.read_gnomonic(map_name)
        elif pixelization == 'healpix':
            m = self.read_healpix(map_name)
        else:
            raise ValueError(f"Unknown map pixelization type {pixelization}")
        return m

    def read_mask(self):
        mask = self.read_map('mask')
        mask[mask<0] = 0
        return mask


    def write_map(self, map_name, pixel, value, metadata):
        """
        Save an output map to an HDF5 subgroup.

        The pixel numbering and the metadata are also saved.

        Parameters
        ----------

        group: H5Group
            The h5py Group object in which to store maps
        name: str
            The name of this map, used as the name of a subgroup in the group where the data is stored.
        pixel: array
            Array of indices of observed pixels
        value: array
            Array of values of observed pixels
        metadata: mapping
            Dict or other mapping of metadata to store along with the map
        """
        if not 'pixelization' in metadata:
            raise ValueError("Map metadata should include pixelization")
        subgroup = self.file['maps'].create_group(map_name)
        subgroup.attrs.update(metadata)
        subgroup.create_dataset("pixel", data=pixel)
        subgroup.create_dataset("value", data=value)


    def plot_healpix(self, map_name, view='cart', **kwargs):
        import healpy
        import numpy as np

        m, pix, nside = self.read_healpix(map_name, return_all=True)
        lon,lat=healpy.pix2ang(nside,pix,lonlat=True)
        npix=healpy.nside2npix(nside)
        if len(pix)==0:
            print(f"Empty map {map_name}")
            return
        if len(pix)==len(m):
            w = np.where((m!=healpy.UNSEEN)&(m!=0))
        else:
            w = None
        lon_range = [lon[w].min()-0.1, lon[w].max()+0.1]
        lat_range = [lat[w].min()-0.1, lat[w].max()+0.1]
        m[m==0] = healpy.UNSEEN
        title = kwargs.pop('title', map_name)
        if view == 'cart':
            healpy.cartview(m, lonra=lon_range, latra=lat_range, title=title, hold=True, **kwargs)
        elif view == 'moll':
            healpy.mollview(m, title=title, hold=True, **kwargs)
        else:
            raise ValueError(f"Unknown Healpix view mode {mode}")

    def read_gnomonic(self, map_name):
        import numpy as np
        group = self.file[f'maps/{map_name}']
        info = dict(group.attrs)
        nx = info['nx']
        ny = info['ny']
        m = np.zeros((ny,nx))
        m[:,:] = np.nan

        pix = group['pixel'][:]
        val = group['value'][:]
        w = np.where(pix!=-9999)
        pix = pix[w]
        val = val[w]
        x = pix % nx
        y = pix // nx
        m[y,x] = val
        return m

    def plot_gnomonic(self, map_name, **kwargs):
        import matplotlib.pyplot as plt
        import numpy as np
        info = self.read_map_info(map_name)
        ra_min, ra_max = info['ra_min'], info['ra_max']
        if ra_min > 180 and ra_max<180:
            ra_min -= 360
        ra_range = (ra_max, ra_min)
        dec_range = (info['dec_min'],info['dec_max'])

        # the view arg is needed for healpix but not gnomonic
        kwargs.pop('view')
        m = self.read_gnomonic(map_name)
        extent = list(ra_range) + list(dec_range)
        title = kwargs.pop('title', map_name)
        plt.imshow(m, aspect='equal', extent=extent, **kwargs)
        plt.title(title)
        plt.colorbar()

    def plot(self, map_name, **kwargs):
        info = self.read_map_info(map_name)
        pixelization = info['pixelization']
        if pixelization == 'gnomonic':
            m = self.plot_gnomonic(map_name, **kwargs)
        elif pixelization == 'healpix':
            m = self.plot_healpix(map_name, **kwargs)
        else:
            raise ValueError(f"Unknown map pixelization type {pixelization}")
        return m


class LensingNoiseMaps(MapsFile):
    required_datasets = [
        ]

    def read_rotation(self, realization_index, bin_index):
        g1_name = f'rotation_{realization_index}/g1_{bin_index}'
        g2_name = f'rotation_{realization_index}/g2_{bin_index}'

        g1 = self.read_map(g1_name)
        g2 = self.read_map(g2_name)

        return g1, g2

    def number_of_realizations(self):
        info = self.file['maps'].attrs
        lensing_realizations = info['lensing_realizations']
        return lensing_realizations

class ClusteringNoiseMaps(MapsFile):
    def read_density_split(self, realization_index, bin_index):
        rho1_name = f'split_{realization_index}/rho1_{bin_index}'
        rho2_name = f'split_{realization_index}/rho2_{bin_index}'
        rho1 = self.read_map(rho1_name)
        rho2 = self.read_map(rho2_name)
        return rho1, rho2

    def number_of_realizations(self):
        info = self.file['maps'].attrs
        clustering_realizations = info['clustering_realizations']
        return clustering_realizations


class PhotozPDFFile(HDFFile):
    required_datasets = []

class CSVFile(DataFile):
    suffix = 'csv'
    def save_file(self,name,dataframe):
        dataframe.to_csv(name)

class SACCFile(DataFile):
    suffix = 'sacc'

    @classmethod
    def open(cls, path, mode, **kwargs):
        import sacc
        if mode == 'w':
            raise ValueError("Do not use the open_output method to write sacc files.  Use sacc.write_fits")
        return sacc.Sacc.load_fits(path)

    def read_provenance(self):
        meta = self.file.metadata
        provenance = {
            'uuid':     meta.get('provenance/uuid', "UNKNOWN"),
            'creation': meta.get('provenance/creation', "UNKNOWN"),
            'domain':   meta.get('provenance/domain', "UNKNOWN"),
            'username': meta.get('provenance/username', "UNKNOWN"),
        }

        return provenance


    def close(self):
        pass






class NOfZFile(HDFFile):

    # Must have at least one bin in
    required_datasets = []

    def get_nbin(self, kind):
        return self.file['n_of_z'][kind].attrs['nbin']

    def get_n_of_z(self, kind, bin_index):
        group = self.file['n_of_z'][kind]
        z = group['z'][:]
        nz = group[f'bin_{bin_index}'][:]
        return (z, nz)

    def get_n_of_z_spline(self, bin_index, kind='cubic', **kwargs):
        import scipy.interpolate
        z, nz = self.get_n_of_z(bin_index)
        spline = scipy.interpolate.interp1d(z, nz, kind=kind, **kwargs)
        return spline

    def save_plot(self, filename, **fig_kw):
        import matplotlib.pyplot as plt
        plt.figure(**fig_kw)
        self.plot()
        plt.legend()
        plt.savefig(filename)
        plt.close()

    def plot(self, kind):
        import matplotlib.pyplot as plt
        for b in range(self.get_nbin(kind)):
            z, nz = self.get_n_of_z(kind, b)
            plt.plot(z, nz, label=f'Bin {b}')
