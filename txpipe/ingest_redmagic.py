from .base_stage import PipelineStage
from .data_types import HDFFile, FitsFile
import numpy as np



class TXIngestRedmagic(PipelineStage):
    name = "TXIngestRedmagic"
    inputs = [
        ('redmagic_catalog', FitsFile),
    ]

    outputs = [
        ('lens_catalog', HDFFile),
        ('lens_tomography_catalog', HDFFile),
        ('lens_photoz_stack', HDFFile),
    ]

    config_options = {
        'lens_zbin_edges':[float],
        'chunk_rows': 100_000,
        'zmin': 0.0,
        'zmax': 3.0,
        'dz': 0.01,
        'bands': 'grizy',
    }

    def run(self):


        # Count number of objects
        f = self.open_input('redmagic_catalog')
        n = f[1].get_nrows()
        f.close()

        chunk_rows = self.config['chunk_rows']
        # mag is hopefully grizy
        bands = self.config['bands']
        zbin_edges = self.config['lens_zbin_edges']
        nbin_lens = len(zbin_edges) - 1

        cat = self.open_output('lens_catalog')
        tomo = self.open_output('lens_tomography_catalog')

        zmin = self.config['zmin']
        zmax = self.config['zmax']
        dz = self.config['dz']
        z_grid = np.arange(zmin, zmax, dz)
        nz_grid = np.zeros((nbin_lens, z_grid.size))
        nz = len(z_grid)




        g = cat.create_group('lens')
        g.create_dataset('ra', (n,), dtype=np.float64)
        g.create_dataset('dec', (n,), dtype=np.float64)
        g.create_dataset('chisq', (n,), dtype=np.float64)
        g.create_dataset('zredmagic', (n,), dtype=np.float64)
        for b in 'grizy':
            g.create_dataset(f'mag_{b}', (n,), dtype=np.float64)
            g.create_dataset(f'mag_err_{b}', (n,), dtype=np.float64)

        h = tomo.create_group('tomography')
        h.create_dataset('lens_bin', (n,), dtype=np.int32)
        h.create_dataset('lens_counts', (nbin_lens,), dtype='i')
        h.attrs['nbin_lens'] = nbin_lens
        h.attrs[f'lens_zbin_edges'] = zbin_edges



        counts = np.zeros(nbin_lens, dtype=np.int64)

        cols = ['ra', 'dec', 'zredmagic', 'mag', 'mag_err', 'chisq']


        for (s, e, data) in self.iterate_fits('redmagic_catalog', 1, cols, chunk_rows):

            n = data['ra'].size
            z = data['zredmagic']
            weight = np.repeat(1.0, n)

            # work out the redshift bin for each object, if any.
            # do any other selection here
            zbin = np.digitize(z, zbin_edges) - 1
            zbin[zbin == nbin_lens] = -1

            # can select on any other criterion here, e.g.
            # mag or chisq.  This is an example
            sel = data['chisq'] < 10
            # deselect these objects
            zbin[~sel] = -1



            # Build up the count of the n(z) histograms per-bin
            z_grid_index = np.floor((z - zmin) / dz).astype(int)
            for i, (i_z, b) in enumerate(zip(z_grid_index, zbin)):
                if (b >= 0):
                    nz_grid[b][i_z] += weight[i]

            # Build up the counts
            any_bin = zbin >= 0
            counts += np.bincount(zbin[any_bin], minlength=nbin_lens)


            g['ra'][s:e] = data['ra']
            g['dec'][s:e] = data['dec']
            g['chisq'][s:e] = data['chisq']
            g['zredmagic'][s:e] = data['zredmagic']

            for i, b in enumerate(bands):
                g[f'mag_{b}'][s:e] = data['mag'][:, i]
                g[f'mag_err_{b}'][s:e] = data['mag_err'][:, i]

            h['lens_bin'][s:e] = zbin


        stack = self.open_output('lens_photoz_stack')
        # Finally save the n(z) values we have built up
        k = stack.create_group(f"n_of_z/lens")

        # HDF has "attributes" which are for small metadata like this
        k.attrs["nbin"] = nbin_lens
        k.attrs["nz"] = nz_grid

        # Save the redshift sampling
        k.create_dataset("z", data=z_grid)
        
        # And all the bins separately
        for b in range(nbin_lens):
            k.attrs[f"count_{b}"] = counts[b]
            k.create_dataset(f"bin_{b}", data=nz_grid[b])

        stack.close()

        # write out photo-z stack

