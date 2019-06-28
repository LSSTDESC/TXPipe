from .base import FitsFile, HDFFile, DataFile, YamlFile, TextFile, Directory

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

class MetacalCatalog(HDFFile):
    """
    A metacal output catalog
    """
    # These are columns
    required_datasets = ['metacal/mcal_g1', 'metacal/mcal_g1_1p', 
        'metacal/mcal_g2', 'metacal/mcal_flags', 'metacal/ra',
        'metacal/mcal_T']

    # Add methods for handling here ...


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




class DiagnosticMaps(HDFFile):
    required_datasets = [
        'maps/depth/value',
        'maps/depth/pixel',
        ]

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


    def display_healpix(self, map_name, **kwargs):
        import healpy
        import numpy as np
        m, pix, nside = self.read_healpix(map_name, return_all=True)
        lon,lat=healpy.pix2ang(nside,pix,lonlat=True)
        npix=healpy.nside2npix(nside)
        lon_range = [lon.min()-0.1, lon.max()+0.1]
        lat_range = [lat.min()-0.1, lat.max()+0.1]
        title = kwargs.pop('title', map_name)
        healpy.cartview(m,lonra=lon_range, latra=lat_range, title=title, **kwargs)

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


    def display_gnomonic(self, map_name, **kwargs):
        import pylab
        import numpy as np
        info = self.read_map_info(map_name)
        ra_min, ra_max = info['ra_min'], info['ra_max']
        if ra_min > 180 and ra_max<180:
            ra_min -= 360
        ra_range = (ra_min, ra_max)
        dec_range = (info['dec_min'],info['dec_max'])

        m = self.read_gnomonic(map_name)
        extent = list(ra_range) + list(dec_range)
        title = kwargs.pop('title', map_name)
        pylab.imshow(m, aspect='equal', extent=extent, **kwargs)
        pylab.title(title)
        pylab.colorbar()
        pylab.show()



class PhotozPDFFile(HDFFile):
    required_datasets = []

class CSVFile():
    suffix = 'csv'
    def save_file(self,name,dataframe):
        dataframe.to_csv(name)

class SACCFile(HDFFile):
    suffix = 'sacc'


class NOfZFile(HDFFile):
    # Must have at least one bin in
    required_datasets = ['n_of_z/lens/z', 'n_of_z/source/bin_0']

    def validate(self):
        super().validate()

        for kind in ('lens', 'source'):
            nbin = self.get_nbin(kind)
            for b in range(nbin):
                col_name = 'bin_{}'.format(b)
                if not col_name in self.file[f'n_of_z/{kind}']:
                    raise FileValidationError(f"Expected to find {nbin} bins in NOfZFile but was missing at least {col_name}")

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

    def plot(self):
        import matplotlib.pyplot as plt
        for b in range(self.get_nbin()):
            z, nz = self.get_n_of_z(b)
            plt.plot(z, nz, label=f'Bin {b}')
