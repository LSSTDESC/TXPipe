from ceci import PipelineStage
from .data_types import MetacalCatalog, HDFFile
import numpy as np
import glob
import re

class TXGCRInput(PipelineStage):
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
    ]

    config_options = {
        'repo':'',
        'model': 'gauss'

    }

    def run(self):

        butler_iterator = self.iterate_metacal_butler(data)

        # We need the first chunk of data so that we
        # know which columns are available.  Later we
        # may want to strip this down to reduce the file
        # size by just using the ones we know we nee.
        cat = next(butler_iterator)
        f = self.setup_output(filename, cat, n)

        # Write out the first data chunk, and keep
        # track of our position
        start = 0
        end = start + len(cat['id'])
        self.write_output(f, start, end, cat)
        start = end

        # Now write out the remaining chunks of data
        for cat in butler_iterator:
            end = start + len(cat)
            self.write_output(f, start, end, cat)
            # Keep track of cursor position
            start = end

        f.close()


    def setup_output(self, cat, n):
        filename = self.get_output('shear_catalog')
        f = h5py.File(filename, "w")
        g = f.create_group('metacal')
        for name in cat.columns:
            g.create_dataset(name, shape=(n,), dtype=cat[name].dtype)    
        return f

    def write_output(self, f, start, end, cat):
        g = f['metacal']
        print(f"    Saving {start} - {end}")
        for name in cat.colnames:
            g[name][start:end] = cat[name]
            

    def find_metacal_tracts_and_patches(self, repo):
        # This seems to be the only way to pull out the list of available
        # tracts and patches
        files = glob.glob(f'{repo}/deepCoadd-results/merged/*/*/mcalmax-deblended-*-*.fits')    
        p = re.compile("mcalmax\-deblended\-(?P<tract>[0-9]+)\-(?P<patch>[0-9]+,[0-9]+)\.fits")

        results = []
        for f in files:
            m = p.search(f)
            if m is None:
                continue
            tract = int(m['tract'])
            patch = m['patch']
            results.append((f, tract, patch))
        return sorted(results)



    def total_length(self, data):
        n = 0
        for f, _, _ in data:
            m = fits.open(f)[1].header['NAXIS2']
            n += m
        return n


    def iterate_metacal_butler(self, data):
        model = self.config['model']
        m = len(f'mcal_{model}')

        for _, tract, patch in data:
            print(f"Loading {tract} {patch}")

            # This returns a SourceCatalog object
            try:
                cat = butler.get('deepCoadd_mcalmax_deblended',
                    dataId={'tract': tract, 'patch': patch, 'filter':'merged'})
            except dp.NoResults:
                continue

            # This is the only way I couldd find to pull out the
            # column names from this object
            colnames = cat.schema.extract("*").keys()

            # Build up a dictionary of the columns,
            data = {}
            for name in colnames:
                # Strip the 'mcal_gauss' prefix and replace with
                # just 'mcal_', as the rest of the code is expecting
                if name.startswith(f"mcal_{model}"):
                    name = 'mcal_' + name[m:]

                data[name] = cat[name]

            yield data
