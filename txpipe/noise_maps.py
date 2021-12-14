from .base_stage import PipelineStage
from .maps import TXBaseMaps
from .data_types import ShearCatalog, TomographyCatalog, MapsFile, \
                        LensingNoiseMaps, ClusteringNoiseMaps, HDFFile
import numpy as np
from .utils.mpi_utils import mpi_reduce_large
from .utils import choose_pixelization
from .utils.calibration_tools import read_shear_catalog_type

class TXNoiseMaps(PipelineStage):
    """
    Generate a suite of random noise maps by randomly
    rotating individual galaxy measurements.

    """
    # TODO rewrite this as a TXBaseMaps subclass
    # like the two below    
    name='TXNoiseMaps'
    
    inputs = [
        ('shear_catalog', ShearCatalog),
        ('lens_tomography_catalog', TomographyCatalog),
        ('shear_tomography_catalog', TomographyCatalog),
        # We get the pixelization info from the diagnostic maps
        ('mask', MapsFile),
        ('lens_maps', MapsFile),
    ]

    outputs = [
        ('source_noise_maps', LensingNoiseMaps),
        ('lens_noise_maps', ClusteringNoiseMaps),
    ]

    config_options = {
        'chunk_rows': 100000,
        'lensing_realizations': 30,
        'clustering_realizations': 1,
        'shear_catalog_type': 'mcal',
        'mask_in_weights': False,
    }        

    def run(self):
        from .utils import choose_pixelization

        # get the number of bins.
        nbin_source, nbin_lens, ngal_maps, mask, map_info = self.read_inputs()
        
        pixel_scheme = choose_pixelization(**map_info)
        lensing_realizations = self.config['lensing_realizations']
        clustering_realizations = self.config['clustering_realizations']

        # The columns we will need
        if self.config["shear_catalog_type"] == "metacal":
            shear_cols = ['ra', 'dec', 'weight', 'mcal_g1', 'mcal_g2']
        else:
            shear_cols = ['ra', 'dec', 'weight', 'g1', 'g2']
        # Make the iterators
        chunk_rows = self.config['chunk_rows']

        it = self.combined_iterators(chunk_rows,
                'shear_catalog', 'shear', shear_cols,
                'shear_tomography_catalog','tomography', ['source_bin'],
                'lens_tomography_catalog','tomography', ['lens_bin'],
            )

        # Get a mapping from healpix indices to masked pixel indices
        # This reduces memory usage.  We could use a healsparse array
        # here, but I'm not sure how to do that best with our
        # many realizations.  Possiby a recarray?
        index_map = np.zeros(pixel_scheme.npix, dtype=np.int64) - 1
        c = 0
        for i in range(pixel_scheme.npix):
            if mask[i] > 0:
                index_map[i] = c
                c += 1

        # Number of unmasked pixels
        npix = c

        if self.rank == 0:
            nmaps = nbin_source * (2 * lensing_realizations + 1) + nbin_lens * clustering_realizations * 2
            nGB = (npix * nmaps * 8) / 1000.**3
            print(f"Allocating maps of size {nGB:.2f} GB")

        # lensing g1, g2
        G1 = np.zeros((npix, nbin_source, lensing_realizations))
        G2 = np.zeros((npix, nbin_source, lensing_realizations))
        # lensing weight
        GW = np.zeros((npix, nbin_source))
        # clustering map - n_gal to start with
        ngal_split = np.zeros((npix, nbin_lens, clustering_realizations, 2), dtype=np.int32)
        # TODO: Clustering weights go here



        # Loop through the data
        for (s, e, data) in it:
            print(f"Rank {self.rank} processing rows {s} - {e}")
            source_bin = data['source_bin']
            lens_bin = data['lens_bin']
            ra = data['ra']
            dec = data['dec']
            orig_pixels = pixel_scheme.ang2pix(ra, dec)
            pixels = index_map[orig_pixels]
            n = e - s

            w = data['weight']
            g1 = data[shear_cols[-2]] * w
            g2 = data[shear_cols[-1]] * w

            # randomly select a half for each object
            split = np.random.binomial(1, 0.5, (n, clustering_realizations))

            # random rotations of the g1, g2 values
            phi = np.random.uniform(0, 2*np.pi, (n, lensing_realizations))
            c = np.cos(phi)
            s = np.sin(phi)
            g1r =  c * g1[:, np.newaxis] + s * g2[:, np.newaxis]
            g2r = -s * g1[:, np.newaxis] + c * g2[:, np.newaxis]

            for i in range(n):
                # convert to the index in the partial space
                pix = pixels[i]

                if pix < 0:
                    continue

                sb = source_bin[i]
                lb = lens_bin[i]
                # build up the rotated map for each bin
                if sb >= 0:
                    G1[pix, sb, :] += g1r[i]
                    G2[pix, sb, :] += g2r[i]
                    GW[pix, sb] += w[i]
                # Build up the ngal for the random half for each bin
                for j in range(clustering_realizations):
                    if lb >= 0:
                        ngal_split[pix, lb, j, split[i]] += 1
                    # TODO add to clustering weight too


        # Sum everything at root
        if self.comm is not None:
            mpi_reduce_large(G1, self.comm)
            mpi_reduce_large(G2, self.comm)
            mpi_reduce_large(GW, self.comm)
            mpi_reduce_large(ngal_split, self.comm)
            if self.rank != 0:
                del G1, G2, GW, ngal_split


        if self.rank==0:
            print("Saving maps")
            outfile = self.open_output('source_noise_maps', wrapper=True)

            # The top section has the metadata in
            group = outfile.file.create_group("maps")
            group.attrs['nbin_source'] = nbin_source
            group.attrs['lensing_realizations'] = lensing_realizations

            metadata = {**self.config, **map_info}

            pixels = np.where(mask>0)[0]

            for b in range(nbin_source):
                for i in range(lensing_realizations):

                    bin_mask = np.where(GW[:, b]>0)

                    g1 = G1[:, b, i] / GW[:, b]
                    g2 = G2[:, b, i] / GW[:, b]

                    outfile.write_map(f"rotation_{i}/g1_{b}", 
                        pixels[bin_mask], g1[bin_mask], metadata)

                    outfile.write_map(f"rotation_{i}/g2_{b}", 
                        pixels[bin_mask], g2[bin_mask], metadata)

            outfile = self.open_output('lens_noise_maps', wrapper=True)
            group = outfile.file.create_group("maps")
            group.attrs['nbin_lens'] = nbin_lens
            group.attrs['clustering_realizations'] = clustering_realizations

            for b in range(nbin_lens):
 
                for i in range(clustering_realizations):
                    half1 = np.zeros(npix)
                    half2 = np.zeros_like(half1)

                    if self.config['mask_in_weights']:
                        half1 = ngal_split[:, b, i, 0]
                        half2 = ngal_split[:, b, i, 1]
                    else:
                        half1 = (ngal_split[:, b, i, 0])/mask[pixels]
                        half2 = (ngal_split[:, b, i, 1])/mask[pixels]

                    # Convert to overdensity.  I thought about
                    # using half the mean from the full map to reduce
                    # noise, but thought that might add covariance
                    # to the two maps, and this shouldn't be that noisy
                    #mu1 = np.average(half1, weights=mask[reverse_map])
                    #mu2 = np.average(half2, weights=mask[reverse_map])

                    mu1 = np.mean(half1)
                    mu2 = np.mean(half2)

                    # This will produce some mangled sentinel values
                    # but they will be masked out
                    rho1 = (half1 - mu1) / mu1
                    rho2 = (half2 - mu2) / mu2

                    # Write both overdensity and count maps
                    # for each bin for each split
                    outfile.write_map(f"split_{i}/rho1_{b}", 
                        pixels, rho1, metadata)
                    outfile.write_map(f"split_{i}/rho2_{b}", 
                        pixels, rho2, metadata)
                    # counts
                    outfile.write_map(f"split_{i}/ngal1_{b}", 
                        pixels, half1, metadata)
                    outfile.write_map(f"split_{i}/ngal2_{b}", 
                        pixels, half2, metadata)
                    



    def read_inputs(self):

        with self.open_input('mask', wrapper=True) as f:
            mask = f.read_map('mask')
            # pixelization etc
            map_info = f.read_map_info('mask')

        with self.open_input('lens_maps', wrapper=True) as f:
            nbin_lens = f.file['maps'].attrs['nbin_lens']
            ngal_maps = [f.read_map(f'ngal_{b}') for b in range(nbin_lens)]

        with self.open_input('shear_tomography_catalog', wrapper=True) as f:
            nbin_source = f.file['tomography'].attrs['nbin_source']


        return nbin_source, nbin_lens, ngal_maps, mask, map_info



class TXSourceNoiseMaps(TXBaseMaps):
    name='TXSourceNoiseMaps'
    
    inputs = [
        ('shear_catalog', ShearCatalog),
        ('shear_tomography_catalog', TomographyCatalog),
        # We get the pixelization info from the diagnostic maps
        ('mask', MapsFile),
    ]

    outputs = [
        ('source_noise_maps', LensingNoiseMaps),
    ]

    config_options = {
        'chunk_rows': 100000,
        'lensing_realizations': 30,
    }

    # instead of reading from config we match the basic maps
    def choose_pixel_scheme(self):
        with self.open_input("mask", wrapper=True) as maps_file:
            pix_info = maps_file.read_map_info("mask")

        return choose_pixelization(**pix_info)

    def prepare_mappers(self, pixel_scheme):
        read_shear_catalog_type(self)

        with self.open_input("mask", wrapper=True) as maps_file:
            mask = maps_file.read_map("mask")

        with self.open_input('shear_tomography_catalog', wrapper=True) as f:
            nbin_source = f.file['tomography'].attrs['nbin_source']

        # Mapping from 0 .. nhit - 1 to healpix indices
        reverse_map = np.where(mask>0)[0]
        # Get a mapping from healpix indices to masked pixel indices
        # This reduces memory usage.  We could use a healsparse array
        # here, but I'm not sure how to do that best with our
        # many realizations.  Possiby a recarray?
        index_map = np.zeros(pixel_scheme.npix, dtype=np.int64) - 1
        index_map[reverse_map] = np.arange(reverse_map.size)

        # Number of unmasked pixels
        npix = reverse_map.size
        lensing_realizations = self.config["lensing_realizations"]

        # lensing g1, g2
        G1 = np.zeros((npix, nbin_source, lensing_realizations))
        G2 = np.zeros((npix, nbin_source, lensing_realizations))
        # lensing weight
        GW = np.zeros((npix, nbin_source))

        return (npix, G1, G2, GW, index_map, reverse_map, nbin_source)

    def data_iterator(self):

        if self.config["shear_catalog_type"] == "metacal":
            shear_cols = ['ra', 'dec', 'weight', 'mcal_g1', 'mcal_g2']
        else:
            shear_cols = ['ra', 'dec', 'weight', 'g1', 'g2']

        it = self.combined_iterators(self.config["chunk_rows"],
                'shear_catalog', 'shear', shear_cols,
                'shear_tomography_catalog','tomography', ['source_bin'],
            )
        return it

    def accumulate_maps(self, pixel_scheme, data, mappers):
        npix, G1, G2, GW, index_map, _, _ = mappers
        lensing_realizations = self.config['lensing_realizations']

        if self.config['shear_catalog_type'] == 'metacal':
            data['g1'] = data['mcal_g1']
            data['g2'] = data['mcal_g2']

        source_bin = data['source_bin']

        # Get the pixel index for each object and convert
        # to the reduced index
        ra = data['ra']
        dec = data['dec']
        orig_pixels = pixel_scheme.ang2pix(ra, dec)
        pixels = index_map[orig_pixels]

        # Pull out some columns we need
        n = len(ra)
        w = data['weight']
        # Pre-weight the g1 values so we don't have to
        # weight each realization again
        g1 = data['g1'] * w
        g2 = data['g2'] * w

        # random rotations of the g1, g2 values
        phi = np.random.uniform(0, 2*np.pi, (n, lensing_realizations))
        c = np.cos(phi)
        s = np.sin(phi)
        g1r =  c * g1[:, np.newaxis] + s * g2[:, np.newaxis]
        g2r = -s * g1[:, np.newaxis] + c * g2[:, np.newaxis]

        for i in range(n):
            sb = source_bin[i]

            # Skip objects we don't use
            if sb < 0:
                continue

            # convert to the index in the partial space
            pix = pixels[i]

            # The sentinel value for pixels is -1
            if pix < 0:
                continue

            # build up the rotated map for each bin
            G1[pix, sb, :] += g1r[i]
            G2[pix, sb, :] += g2r[i]
            GW[pix, sb] += w[i]


    def finalize_mappers(self, pixel_scheme, mappers):
        # only one mapper here - we call its finalize method
        # to collect everything
        npix, G1, G2, GW, index_map, reverse_map, nbin_source = mappers
        lensing_realizations =  self.config["lensing_realizations"]

        # Sum everything at root
        if self.comm is not None:
            mpi_reduce_large(G1, self.comm)
            mpi_reduce_large(G2, self.comm)
            mpi_reduce_large(GW, self.comm)
            if self.rank != 0:
                del G1, G2, GW

        # build up output
        maps = {}

        # only master gets full stuff
        if self.rank != 0:
            return maps

        for b in range(nbin_source):
            for i in range(lensing_realizations):

                bin_mask = np.where(GW[:, b]>0)

                g1 = G1[:, b, i] / GW[:, b]
                g2 = G2[:, b, i] / GW[:, b]

                maps["source_noise_maps", f"rotation_{i}/g1_{b}"] = (
                    reverse_map[bin_mask], g1[bin_mask]
                )

                maps["source_noise_maps", f"rotation_{i}/g2_{b}"] = (
                    reverse_map[bin_mask], g2[bin_mask]
                )
        return maps


class TXExternalLensNoiseMaps(TXBaseMaps):
    name='TXExternalLensNoiseMaps'
    
    inputs = [
        ('lens_tomography_catalog', TomographyCatalog),
        ('lens_catalog', HDFFile),
        ('mask', MapsFile),
    ]

    outputs = [
        ('lens_noise_maps', ClusteringNoiseMaps),
    ]

    config_options = {
        'chunk_rows': 100000,
        'clustering_realizations': 1,
        'mask_in_weights': False,
    }

    # instead of reading from config we match the basic maps
    def choose_pixel_scheme(self):
        with self.open_input("mask", wrapper=True) as maps_file:
            pix_info = maps_file.read_map_info("mask")

        return choose_pixelization(**pix_info)

    def prepare_mappers(self, pixel_scheme):

        with self.open_input("mask", wrapper=True) as maps_file:
            mask = maps_file.read_map("mask")

        with self.open_input('lens_tomography_catalog', wrapper=True) as f:
            nbin_lens = f.file['tomography'].attrs['nbin_lens']

        # Mapping from 0 .. nhit - 1  to healpix indices
        reverse_map = np.where(mask>0)[0]
        # Get a mapping from healpix indices to masked pixel indices
        # This reduces memory usage.  We could use a healsparse array
        # here, but I'm not sure how to do that best with our
        # many realizations.  Possiby a recarray?
        index_map = np.zeros(pixel_scheme.npix, dtype=np.int64) - 1
        index_map[reverse_map] = np.arange(reverse_map.size)

        # Number of unmasked pixels
        npix = reverse_map.size
        clustering_realizations = self.config["clustering_realizations"]

        ngal_split = np.zeros((npix, nbin_lens, clustering_realizations, 2), dtype=np.int32)
        # TODO: Clustering weights go here

        return (npix, ngal_split, index_map, reverse_map, mask, nbin_lens)

    def data_iterator(self):
        it = self.combined_iterators(self.config["chunk_rows"],
                'lens_catalog','lens', ['ra', 'dec'],
                'lens_tomography_catalog','tomography', ['lens_bin'],
            )
        return it

    def accumulate_maps(self, pixel_scheme, data, mappers):
        npix, ngal_split, index_map, _, _, _ = mappers
        clustering_realizations = self.config['clustering_realizations']

        # Tomographic bin
        lens_bin = data['lens_bin']

        # Get the pixel index for each object and convert
        # to the reduced index
        ra = data['ra']
        dec = data['dec']
        orig_pixels = pixel_scheme.ang2pix(ra, dec)
        pixels = index_map[orig_pixels]
        n = len(ra)

        # randomly select a half for each object
        split = np.random.binomial(1, 0.5, (n, clustering_realizations))

        for i in range(n):
            lb = lens_bin[i]

            # Skip objects we don't use
            if lb < 0:
                continue

            # convert to the index in the partial space
            pix = pixels[i]

            # The sentinel value for pixels is -1
            if pix < 0:
                continue

            for j in range(clustering_realizations):
                ngal_split[pix, lb, j, split[i]] += 1


    def finalize_mappers(self, pixel_scheme, mappers):
        # only one mapper here - we call its finalize method
        # to collect everything
        npix, ngal_split, index_map, reverse_map, mask, nbin_lens = mappers
        clustering_realizations = self.config["clustering_realizations"]
        # Sum everything at root
        if self.comm is not None:
            mpi_reduce_large(ngal_split, self.comm)
            if self.rank != 0:
                del ngal_split

        # build up output
        maps = {}

        # only master gets full stuff
        if self.rank != 0:
            return maps

        for b in range(nbin_lens):
            for i in range(clustering_realizations):
                # We have computed the first half already,
                # and we have the total map from an earlier stage
                half1 = np.zeros(npix)
                half2 = np.zeros_like(half1)

                if mask_in_weights:
                    half1 = ngal_split[:, b, i, 0]
                    half2 = ngal_split[:, b, i, 1]
                else:
                    half1 = (ngal_split[:, b, i, 0])/mask[reverse_map>0]
                    half2 = (ngal_split[:, b, i, 1])/mask[reverse_map>0]

                # Convert to overdensity.  I thought about
                # using half the mean from the full map to reduce
                # noise, but thought that might add covariance
                # to the two maps, and this shouldn't be that noisy
                #mu1 = np.average(half1, weights=mask[reverse_map])
                #mu2 = np.average(half2, weights=mask[reverse_map])
            
                mu1 = np.mean(half1)
                mu2 = np.mean(half2)
               
                # This will produce some mangled sentinel values
                # but they will be masked out
                rho1 = (half1 - mu1) / mu1
                rho2 = (half2 - mu2) / mu2

                # Save four maps - density splits and ngal splits
                maps['lens_noise_maps', f"split_{i}/rho1_{b}"] = (
                    reverse_map, rho1)
                maps['lens_noise_maps', f"split_{i}/rho2_{b}"] = (
                    reverse_map, rho2)
                maps['lens_noise_maps', f"split_{i}/ngal1_{b}"] = (
                    reverse_map, half1)
                maps['lens_noise_maps', f"split_{i}/ngal2_{b}"] = (
                    reverse_map, half2)

        return maps
