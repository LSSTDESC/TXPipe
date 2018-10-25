from descformats import FitsFile, HDFFile, DataFile, YamlFile

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

class MetacalCatalog(FitsFile):
    """
    A metacal output catalog
    """
    # These are columns
    metacal_columns = [
        'mcal_g', 'mcal_g_cov',  'mcal_pars',  'mcal_pars_cov', 
        'mcal_T', 'mcal_T_err', 'mcal_T_r', 'mcal_s2n_r',]

    other_columns = ['mcal_flux_cov', 'mcal_weight', 'mcal_flux', 
        'mcal_flux_s2n', 'mcal_mag', 'mcal_gpsf', 'mcal_logsb', 'mcal_Tpsf']

    # The parent class will check these columns exist.
    required_columns = ( metacal_columns 
                        + metacalibration_names(metacal_columns) 
                        + other_columns )

    # Add methods for handling here ...


class TomographyCatalog(HDFFile):
    required_datasets = []

    def read_zbins(self):
        """
        Read saved redshift bin edges from attributes
        """
        d = dict(self.file['tomography'].attrs)
        tomo.close()
        nbin = d['nbin']
        zbins = [(d[f'zmin_{i}'], d[f'zmax_{i}']) for i in range(nbin)]
        return zbins




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

    def read_tangential(self, map_name):
        import numpy as np
        group = self.file[f'maps/{map_name}']
        info = dict(group.attrs)
        nx = info['nx']
        ny = info['ny']
        print(ny,nx)
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
        ra_min, ra_max = info['ra_min'], info['ra_max']
        print(ra_min, ra_max)
        if ra_min > 180 and ra_max<180:
            ra_min -= 360
        ra_range = (ra_min, ra_max)
        dec_range = (info['dec_min'],info['dec_max'])
        return m, ra_range, dec_range

    def display_tangential(self, map_name, **kwargs):
        import pylab
        import numpy as np
        m, ra_range, dec_range = self.read_tangential(map_name)
        extent = list(ra_range) + list(dec_range)
        title = kwargs.pop('title', map_name)
        pylab.imshow(m, aspect='equal', extent=extent, **kwargs)
        pylab.title(title)
        pylab.colorbar()
        pylab.show()



class PhotozPDFFile(HDFFile):
    required_datasets = []


class SACCFile(HDFFile):
    suffix = 'sacc'

