from ceci import PipelineStage
from .data_types import MetacalCatalog, HDFFile
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
    ]

    config_options = {
        'repo': str,
        'model': 'gauss'

    }

    def run(self):


        inputs = self.find_metacal_tracts_and_patches()
        butler_iterator = self.iterate_metacal_butler(inputs)

        n = self.total_length(inputs)
        print(f"Total catalog size = {n}")
        # We need the first chunk of data so that we
        # know which columns are available.  Later we
        # may want to strip this down to reduce the file
        # size by just using the ones we know we nee.
        data = next(butler_iterator)
        outfile = self.setup_output(data, n)

        # Write out the first data chunk, and keep
        # track of our position
        start = 0
        end = start + len(data['id'])
        self.write_output(outfile, start, end, data)
        start = end

        # Now write out the remaining chunks of data
        for data in butler_iterator:
            end = start + len(data['id'])
            self.write_output(outfile, start, end, data)
            # Keep track of cursor position
            start = end

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
            

    def find_metacal_tracts_and_patches(self):
        # This seems to be the only way to pull out the list of available
        # tracts and patches
        repo = self.config['repo']

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
        from astropy.io import fits
        n = 0
        for f, _, _ in data:
            m = fits.open(f)[1].header['NAXIS2']
            n += m
        return n


    def iterate_metacal_butler(self, data):
        import lsst.daf.persistence as dp
        
        repo = self.config['repo']
        butler = dp.Butler(repo)

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
                    new_name = 'mcal' + name[m:]
                else:
                    new_name = name

                data[name] = cat[name]

            yield data
