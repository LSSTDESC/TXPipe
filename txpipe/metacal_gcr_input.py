from .base_stage import PipelineStage
from .data_types import MetacalCatalog, HDFFile
from .utils.metacal import metacal_band_variants, metacal_variants
import numpy as np
import glob
import re

class TXMetacalGCRInput(PipelineStage):
    """
    This stage simulates metacal data and metacalibrated
    photometry measurements, starting from a cosmology catalogs
    of the kind used as an input to DC2 image and obs-catalog simulations.

    This is mainly useful for testing infrastructure in advance
    of the DC2 catalogs being available, but might also be handy
    for starting from a purer simulation.
    """
    name='TXMetacalGCRInput'

    inputs = []

    outputs = [
        ('shear_catalog', MetacalCatalog),
        ('photometry_catalog', HDFFile),
    ]

    config_options = {
        'cat_name': str,
    }

    def run(self):
        import GCRCatalogs
        import h5py
        # Open input data.  We do not treat this as a formal "input"
        # since it's the starting point of the whol pipeline and so is
        # not in a TXPipe format.
        cat_name = self.config['cat_name']
        cat = GCRCatalogs.load_catalog(cat_name)

        # Total size is needed to set up the output file,
        # although in larger files it is a little slow to compute this.
        n = len(cat)
        print(f"Total catalog size = {n}")  

        available = cat.list_all_quantities()
        bands = []
        for b in 'ugrizy':
            if f'mcal_mag_{b}' in available:
                bands.append(b)

        # Columns that we will need.
        shear_cols = (['id', 'ra', 'dec', 'mcal_psf_g1', 'mcal_psf_g2', 'mcal_psf_T_mean', 'mcal_flags']
            + metacal_variants('mcal_g1', 'mcal_g2', 'mcal_T', 'mcal_s2n')
            + metacal_band_variants(bands, 'mcal_mag', 'mcal_mag_err')
        )

        photo_cols = ['id', 'ra', 'dec']
        # Photometry columns (non-metacal)
        for band in 'ugrizy':
            photo_cols.append(f'{band}_mag')
            photo_cols.append(f'{band}_mag_err')
            photo_cols.append(f'snr_{band}_cModel')

        # eliminate duplicates
        cols = list(set(shear_cols + photo_cols))

        # We will rename this band
        for band in 'ugrizy':
            photo_cols.append(f'snr_{band}')

        start = 0
        shear_output = None
        photo_output = None

        # Loop through the data, as chunke natively by GCRCatalogs
        for data in cat.get_quantities(cols, return_iterator=True):
            #
            self.rename_columns(data)
            # First chunk of data we use to set up the output
            # It is easier this way (no need to check types etc)
            # if we change the column list
            if shear_output is None:
                shear_output = self.setup_shear_output(data, shear_cols, n)
                photo_output = self.setup_photo_output(data, photo_cols, n)


            # Write out this chunk of data to HDF
            end = start + len(data['ra'])
            print(f"    Saving {start} - {end}")
            self.write_output(shear_output['metacal'],    shear_cols, start, end, data)
            self.write_output(photo_output['photometry'], photo_cols, start, end, data)
            start = end

        # All done!
        photo_output.close()
        shear_output.close()

    def rename_columns(self, data):
        for band in 'ugrizy':
            data[f'snr_{band}'] = data[f'snr_{band}_cModel']
            del data[f'snr_{band}_cModel']


    def setup_shear_output(self, cat, cols, n):
        import h5py
        filename = self.get_output('shear_catalog')
        f = h5py.File(filename, "w")
        g = f.create_group('metacal')
        for name, col in cat.items():
            if name in cols:
                g.create_dataset(name, shape=(n,), dtype=col.dtype)
        return f

    def setup_photo_output(self, cat, cols, n):
        import h5py
        filename = self.get_output('photometry_catalog')
        f = h5py.File(filename, "w")
        g = f.create_group('photometry')
        for name, col in cat.items():
            if name in cols:
                g.create_dataset(name, shape=(n,), dtype=col.dtype)
        return f


    def write_output(self, g, cols, start, end, data):
        for name in cols:
            if name.endswith('_cModel'):
                name = name[:-len('_cModel')]
            g[name][start:end] = data[name]

