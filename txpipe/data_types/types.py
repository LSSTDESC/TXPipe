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
    
    def read_catalog_info(self):
        try:
            group = self.file['shear']
            info = dict(group.attrs)
        except:
            raise ValueError(f"Unable to read shear catalog")
        shear_catalog_type = info.get('catalog_type')
        return shear_catalog_type

    def validate(self):
        
        shear_catalog_type = self.read_catalog_info()
        
        if shear_catalog_type=='metacal':
            required_datasets_mcal = ['shear/mcal_g1', 'shear/mcal_g1_1p', 
        'shear/mcal_g2', 'shear/mcal_flags', 'shear/ra',
        'shear/mcal_T']
        elif shear_catalog_type=='lensfit':
            required_datasets_lensfit = ['shear/g1', 
        'shear/g2', 'shear/flags', 'shear/ra',
        'shear/T']
        else:
            raise ValueError(f"Unrecognized shear catalog type {shear_catalog_type}")
        super().validate()


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

    def get_nbins(self):
        group = self.file['maps']
        info = dict(group.attrs)
        nbin_lens = info.get('nbin_lens', 0)
        nbin_source = info.get('nbin_source', 0)
        return nbin_source, nbin_lens


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

    def validate(self):
        
        nsrc = self.get_nbin('source')
        nlens = self.get_nbin('lens')
        if nsrc and nlens > 0:
            required_datasets = ['n_of_z/lens/z', 'n_of_z/source/bin_0']
        elif nsrc > 0: 
            required_datasets = ['n_of_z/source/bin_0']
        else:
            required_datasets = ['n_of_z/source/bin_0']
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

    def plot(self, kind):
        import matplotlib.pyplot as plt
        for b in range(self.get_nbin(kind)):
            z, nz = self.get_n_of_z(kind, b)
            plt.plot(z, nz, label=f'Bin {b}')
