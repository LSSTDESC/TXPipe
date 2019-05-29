from ceci import PipelineStage
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
        ('shear_catalog', HDFFile),
    ]

    config_options = {
        'cat_name': str,
    }

    def run(self):
        import GCRCatalogs
        # Open input data.  We do not treat this as a formal "input"
        # since it's the starting point of the whol pipeline and so is
        # not in a TXPipe format.
        cat_name = self.config['cat_name']
        cat = GCRCatalogs.load_catalog(cat_name)

        # Total size is needed to set up the output file,
        # although in larger files it is a little slow to compute this.
        n = len(cat)
        print(f"Total catalog size = {n}")

        # Columns that we will need.
        cols = (['objectId', 'ra', 'dec', 'mcal_psf_g1', 'mcal_psf_g2', 'mcal_psf_T_mean']
            + metacal_variants('mcal_g1', 'mcal_g2', 'mcal_T', 'mcal_s2n')
            + metacal_band_variants('mcal_mag', 'mcal_mag_err')
        )

        start = 0
        outfile = None

        # Loop through the data, as chunke natively by GCRCatalogs
        for data in cat.get_quantities(cols, return_iterator=True):

            # First chunk of data we use to set up the output
            # It is easier this way (no need to check types etc)
            # if we change the column list
            if outfile is None:
                outfile = self.setup_output(data, n)

            # Write out this chunk of data to HDF
            end = start + len(data['ra'])
            self.write_output(outfile, start, end, data)
            start = end

        # All done!
        outfile.close()


    def setup_output(self, cat, n):
        import h5py
        filename = self.get_output('shear_catalog')
        f = h5py.File(filename, "w")
        g = f.create_group('metacal')
        for name, col in cat.items():
            g.create_dataset(name, shape=(n,), dtype=col.dtype)
        return f

    def write_output(self, f, start, end, data):
        g = f['metacal']
        print(f"    Saving {start} - {end}")
        for name, col in data.items():
            g[name][start:end] = col

