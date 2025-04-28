from .base_stage import PipelineStage
from .maps import TXBaseMaps
from .data_types import (
    ShearCatalog,
    TomographyCatalog,
    MapsFile,
    LensingNoiseMaps,
    ClusteringNoiseMaps,
    HDFFile,
)
import numpy as np
from .utils.mpi_utils import mpi_reduce_large
from .utils import (
    choose_pixelization,
    Calibrator,
    read_shear_catalog_type,
    rename_iterated,
)


map_config_options = {
    "chunk_rows": 100000,  # The number of rows to read in each chunk of data at a time
    "pixelization": "healpix",  # The pixelization scheme to use, currently just healpix
    "nside": 0,  # The Healpix resolution parameter for the generated maps. Only req'd if using healpix
    "sparse": True,  # Whether to generate sparse maps - faster and less memory for small sky areas,
    "ra_cent": np.nan,  # These parameters are only required if pixelization==tan
    "dec_cent": np.nan,
    "npix_x": -1,
    "npix_y": -1,
    "pixel_size": np.nan,  # Pixel size of pixelization scheme
}


class TXSourceNoiseMaps(PipelineStage):
    """
    Generate source noise maps directly from binned *calibrated* shear 
    catalogs using the same approach as TXSourceMaps. Previous version 
    generated this from unbinned and uncalibrated shear catalogs.  
    """
    name = "TXSourceNoiseMaps"
    dask_parallel = True

    inputs = [
        ("binned_shear_catalog", HDFFile),
    ]
    outputs = [
        ("source_noise_maps", LensingNoiseMaps),
    ]
    config_options = {
        "block_size": 0,
        "lensing_realizations": 30,
        **map_config_options
    }
        
    def run(self):
        import dask
        import dask.array as da
        import healpy
        from tqdm import tqdm

        # Configuration options
        pixel_scheme = choose_pixelization(**self.config)
        nside = self.config["nside"]
        npix = healpy.nside2npix(nside)
        block_size = self.config["block_size"]

        if block_size == 0:
            block_size = "auto"
        lensing_realizations = self.config["lensing_realizations"]

        # We have to keep this open throughout the process, because
        # dask will internally load chunks of the input hdf5 data.
        f = self.open_input("binned_shear_catalog")
        nbin = f['shear'].attrs['nbin_source']

        # The "all" bin is the non-tomographic case.
        bins = list(range(nbin)) + ["all"]
        output = {}

        for i in bins:
            ra = da.from_array(f[f"shear/bin_{i}/ra"], block_size)
            dec = da.from_array(f[f"shear/bin_{i}/dec"], block_size)
            g1 = da.from_array(f[f"shear/bin_{i}/g1"], block_size)
            g2 = da.from_array(f[f"shear/bin_{i}/g2"], block_size)
            weight = da.from_array(f[f"shear/bin_{i}/weight"], block_size)

            g1 = g1-da.mean(g1)
            g2 = g2-da.mean(g2)
            
            pix = pixel_scheme.ang2pix(ra, dec)

            # For the other map we use bincount with weights - these are the
            # various maps by pixel. bincount gives the number of objects in each
            # vaue of the first argument, weighted by the weights keyword, so effectively
            # it gives us
            # p_i = sum_{j} x[j] * delta_{pix[j], i}
            # which is out map
            for seed in range(lensing_realizations):
                phi = da.random.uniform(0, 2 * np.pi, len(ra))
                c = da.cos(phi)
                s = da.sin(phi)
                g1r = c * g1 + s * g2
                g2r = -s * g1 + c * g2

                weight_map = da.bincount(pix, weights=weight, minlength=npix)
                g1_map = da.bincount(pix, weights=weight * g1r, minlength=npix)
                g2_map = da.bincount(pix, weights=weight * g2r, minlength=npix)
                
                # normalize by weights to get the mean map value in each pixel
                g1_map /= weight_map
                g2_map /= weight_map

                # slight change in output name
                if i == "all": i = "2D"

                # replace nans with UNSEEN.  The NaNs can occur if there are
                # no objects in a pixel, so the value is undefined.
                g1_map[da.isnan(g1_map)] = healpy.UNSEEN
                g2_map[da.isnan(g2_map)] = healpy.UNSEEN

                # Save all the stuff we want here.
                output[f"rotation_{seed}/g1_{i}"] = g1_map
                output[f"rotation_{seed}/g2_{i}"] = g2_map

            output[f"lensing_weight_{i}"] = weight_map

        # mask is where a pixel is hit in any of the tomo bins
        mask = da.zeros(npix, dtype=bool)
        for i in bins:
            if i == "all":
                i = "2D"
            mask |= output[f"lensing_weight_{i}"] > 0

        output["mask"] = mask

        # Everything above is lazy - this forces the computation.
        # It works out an efficient (we hope) way of doing everything in parallel
        output, = dask.compute(output)
        f.close()

        # collate metadata
        metadata = {
            key: self.config[key]
            for key in map_config_options
        }
        metadata['nbin'] = nbin
        metadata['nbin_source'] = nbin

        pix = np.where(output["mask"])[0]
        
        # write the output maps
        with self.open_output("source_noise_maps", wrapper=True) as out:
            for seed in range(lensing_realizations):
                for i in bins:
                    # again rename "all" to "2D"
                    if i == "all":
                        i = "2D"

                    # We save the pixels in the mask - i.g. any pixel that is hit in any
                    # tomographic bin is included. Some will be UNSEEN.
                    for key in "g1", "g2":
                        out.write_map(f"rotation_{seed}/{key}_{i}", pix, output[f"rotation_{seed}/{key}_{i}"][pix], metadata)

            out.file['maps'].attrs.update(metadata)


class TXLensNoiseMaps(TXBaseMaps):
    """
    Generate lens density noise realizations using random splits

    This randomly assigns each galaxy to one of two bins and uses the
    different between the halves to get a noise estimate.
    """
    name = "TXLensNoiseMaps"

    inputs = [
        ("lens_tomography_catalog", TomographyCatalog),
        ("photometry_catalog", HDFFile),
        ("mask", MapsFile),
    ]

    outputs = [
        ("lens_noise_maps", ClusteringNoiseMaps),
    ]

    config_options = {
        "chunk_rows": 100000,
        "clustering_realizations": 1,
        "mask_in_weights": False,
    }

    # instead of reading from config we match the basic maps
    def choose_pixel_scheme(self):
        with self.open_input("mask", wrapper=True) as maps_file:
            pix_info = maps_file.read_map_info("mask")

        return choose_pixelization(**pix_info)

    def prepare_mappers(self, pixel_scheme):

        with self.open_input("mask", wrapper=True) as maps_file:
            mask = maps_file.read_map("mask")

        with self.open_input("lens_tomography_catalog", wrapper=True) as f:
            nbin_lens = f.file["tomography"].attrs["nbin"]

        # Mapping from 0 .. nhit - 1  to healpix indices
        reverse_map = np.where(mask > 0)[0]
        # Get a mapping from healpix indices to masked pixel indices
        # This reduces memory usage.  We could use a healsparse array
        # here, but I'm not sure how to do that best with our
        # many realizations.  Possiby a recarray?
        index_map = np.zeros(pixel_scheme.npix, dtype=np.int64) - 1
        index_map[reverse_map] = np.arange(reverse_map.size)

        # Number of unmasked pixels
        npix = reverse_map.size
        clustering_realizations = self.config["clustering_realizations"]

        ngal_split = np.zeros(
            (npix, nbin_lens, clustering_realizations, 2), dtype=np.int32
        )
        # TODO: Clustering weights go here

        return (npix, ngal_split, index_map, reverse_map, mask, nbin_lens)

    def data_iterator(self):
        it = self.combined_iterators(
            self.config["chunk_rows"],
            "photometry_catalog",
            "photometry",
            ["ra", "dec"],
            "lens_tomography_catalog",
            "tomography",
            ["bin"],
        )
        return it

    def accumulate_maps(self, pixel_scheme, data, mappers):
        npix, ngal_split, index_map, _, _, _ = mappers
        clustering_realizations = self.config["clustering_realizations"]

        # Tomographic bin
        lens_bin = data["bin"]

        # Get the pixel index for each object and convert
        # to the reduced index
        ra = data["ra"]
        dec = data["dec"]
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

                if self.config["mask_in_weights"]:
                    half1 = ngal_split[:, b, i, 0]
                    half2 = ngal_split[:, b, i, 1]
                else:
                    half1 = (ngal_split[:, b, i, 0]) / mask[reverse_map]
                    half2 = (ngal_split[:, b, i, 1]) / mask[reverse_map]

                # Convert to overdensity.  I thought about
                # using half the mean from the full map to reduce
                # noise, but thought that might add covariance
                # to the two maps, and this shouldn't be that noisy
                # half1 and half2 are already weighted by the mask, so we just need the average
                mu1 = np.average(half1[mask[reverse_map] > 0])
                mu2 = np.average(half2[mask[reverse_map] > 0])

                # This will produce some mangled sentinel values
                # but they will be masked out
                rho1 = (half1 - mu1) / mu1
                rho2 = (half2 - mu2) / mu2

                # Save four maps - density splits and ngal splits
                maps["lens_noise_maps", f"split_{i}/rho1_{b}"] = (reverse_map, rho1)
                maps["lens_noise_maps", f"split_{i}/rho2_{b}"] = (reverse_map, rho2)
                maps["lens_noise_maps", f"split_{i}/ngal1_{b}"] = (reverse_map, half1)
                maps["lens_noise_maps", f"split_{i}/ngal2_{b}"] = (reverse_map, half2)

        return maps


class TXExternalLensNoiseMaps(TXLensNoiseMaps):
    """
    Generate lens density noise realizations using random splits of an external catalog

    This randomly assigns each galaxy to one of two bins and uses the
    different between the halves to get a noise estimate.
    """
    name = "TXExternalLensNoiseMaps"

    inputs = [
        ("lens_tomography_catalog", TomographyCatalog),
        ("lens_catalog", HDFFile),
        ("mask", MapsFile),
    ]

    def data_iterator(self):
        it = self.combined_iterators(
            self.config["chunk_rows"],
            "lens_catalog",
            "lens",
            ["ra", "dec"],
            "lens_tomography_catalog",
            "tomography",
            ["bin"],
        )
        return it


# These functions will be jitted and used in the TXNoiseMapsJax class below.
# Note that, quoting the JAX docs:
#   Unlike NumPy in-place operations such as x[idx] += y, if multiple indices
#   refer to the same location, all updates will be applied (NumPy would only
#   apply the last update, rather than applying all updates.)
#   So this is not just the raw equivalent of GN[masked_pixels, masked_source_bin] += masked_gnr
# This is better than original numpy!
def GN_add(GN, masked_pixels, masked_source_bin, masked_gnr):
    return GN.at[masked_pixels, masked_source_bin, :].add(masked_gnr)


def GW_add(GW, masked_pixels, masked_source_bin, masked_weights):
    return GW.at[masked_pixels, masked_source_bin].add(masked_weights)


def ngal_split_add(
    ngal_split, pixels_lb_mask, clustering_realizations, split_lb_mask, lens_bin_lb_mask
):
    from jax import numpy as jnp

    return ngal_split.at[
        pixels_lb_mask,
        lens_bin_lb_mask,
        jnp.arange(clustering_realizations),
        split_lb_mask,
    ].add(1)


class TXNoiseMapsJax(PipelineStage):
    """
    Generate noise realisations of lens and source maps using JAX

    This is a JAX/GPU version of the noise map stages.

    Need to update to stop assuming lens and source are the same
    and split into two stages.

    """

    name = "TXNoiseMapsJax"
    inputs = [
        ("shear_catalog", ShearCatalog),
        ("lens_tomography_catalog", TomographyCatalog),
        ("shear_tomography_catalog", TomographyCatalog),
        # We get the pixelization info from the diagnostic maps
        ("mask", MapsFile),
        ("lens_maps", MapsFile),
    ]
    outputs = [
        ("source_noise_maps", LensingNoiseMaps),
        ("lens_noise_maps", ClusteringNoiseMaps),
    ]

    config_options = {
        "chunk_rows": 4000000,
        "lensing_realizations": 30,
        "clustering_realizations": 1,
        "seed": 0,
    }

    def run(self):
        from jax import numpy as jnp
        from jax.ops import index
        from jax import random, jit, device_get, device_put
        from .utils import choose_pixelization

        raise ValueError("This code needs rewriting because source_bin and lens_bin now have the same name in the tomo files.")

        # get the number of bins.
        nbin_source, nbin_lens, ngal_maps, mask, map_info = self.read_inputs()
        pixel_scheme = choose_pixelization(**map_info)
        lensing_realizations = self.config["lensing_realizations"]
        clustering_realizations = self.config["clustering_realizations"]

        # The columns we will need
        shear_cols = ["ra", "dec", "weight", "mcal_g1", "mcal_g2"]

        # Make the iterators
        chunk_rows = self.config["chunk_rows"]

        it = self.combined_iterators(
            chunk_rows,
            "shear_catalog",
            "shear",
            shear_cols,
            "shear_tomography_catalog",
            "tomography",
            ["bin"],
            "lens_tomography_catalog",
            "tomography",
            ["bin"],
        )

        # Get a mapping from healpix indices to masked pixel indices
        index_map = np.zeros(pixel_scheme.npix, dtype=jnp.int32) - 1
        counter = 0
        for i in range(pixel_scheme.npix):
            if mask[i] > 0:
                index_map[i] = counter
                counter += 1

        # Number of unmasked pixels
        npix = counter

        # The memory usage of this class can get high, so we report what is expected here, so
        # if a crash happens a few moments later it's clear why.
        if self.rank == 0:
            nmaps = (
                nbin_source * (2 * lensing_realizations + 1)
                + nbin_lens * clustering_realizations * 2
            )
            nGB = (npix * nmaps * 8) / 1000.0**3
            print(f"Allocating maps of size {nGB:.2f} GB")

        # lensing g1, g2. To start with we accumalate these, and normalize them later
        G1 = jnp.zeros((npix, nbin_source, lensing_realizations))
        G2 = jnp.zeros((npix, nbin_source, lensing_realizations))

        # lensing weights per pixel, which we later use to normalize g1, g2
        GW = jnp.zeros((npix, nbin_source))

        # clustering map - we start by generating a random split in the number count
        # maps, and later convert this to overdensity maps
        ngal_split = jnp.zeros(
            (npix, nbin_lens, clustering_realizations, 2), dtype=np.int32
        )
        # TODO: Clustering weights go here

        # Initialize PRNG key for Jax with a seed, which can either be
        # chosen by the user or generated with numpy
        if self.config["seed"] == 0:
            seed = np.random.randint(2**32)
        else:
            seed = self.config["seed"]

        # ensure that every MPI rank has a different seed, and set up the JAX RNG
        # system
        seed += self.rank
        key = random.PRNGKey(seed)

        # apply the just-in-time compilation to these functions; this means that
        # they are compiled on first use, for the data types given to them, then
        # subsequent times the compiled version is used. They are used on each chunk
        # of the data as we loop through it.
        GN_add_jit = jit(GN_add)
        GW_add_jit = jit(GW_add)
        ngal_split_add_jit = jit(ngal_split_add, static_argnums=(2,))

        # Loop through the data
        # TODO: this whole bit should be a single jax.jit kernel for speed
        for (s, e, data) in it:
            # Number of objects in this chunk
            n = e - s
            print(f"Rank {self.rank} processing rows {s} - {e}")

            # Send data to GPU
            source_bin = device_put(data["bin"])
            lens_bin = device_put(data["bin"])
            weights = device_put(data["weight"])
            g1 = device_put(data["mcal_g1"]) * weights
            g2 = device_put(data["mcal_g2"]) * weights

            # Compute which pixel each object is in
            ra = data["ra"]
            dec = data["dec"]
            orig_pixels = device_put(pixel_scheme.ang2pix(ra, dec))
            pixels = device_put(index_map[orig_pixels])

            # This is how you do RNG with JAX. We use subkey for this RNG operation
            # and then key is passed forward for the next operation
            key, subkey = random.split(key)

            # randomly select a half for each lens bin object
            # random.bernoulli returns True/False arrays. Convert that
            # to an integer array (both on the GPU) by multiplying by 1
            split = 1 * random.bernoulli(subkey, 0.5, (n, clustering_realizations))

            # random rotations of the g1, g2 values
            key, subkey = random.split(key)
            phi = random.uniform(
                subkey, shape=(lensing_realizations, n), minval=0, maxval=2 * jnp.pi
            )
            cos = jnp.cos(phi)
            sin = jnp.sin(phi)
            g1r = jnp.transpose(cos * g1 + sin * g2)
            g2r = jnp.transpose(-sin * g1 + cos * g2)

            # masks showing which pixels to fill in
            pix_mask = pixels >= 0
            sb_mask = (source_bin >= 0) & pix_mask

            # jax.jit doesn't like masks inside masks so we have to calculate these in
            # advance instead of doing that within the jitted functions
            masked_pixels = pixels[sb_mask]
            masked_source_bin = source_bin[sb_mask]
            masked_g1r = g1r[sb_mask]
            masked_g2r = g2r[sb_mask]
            masked_weights = weights[sb_mask]
            lb_mask = sb_mask & (lens_bin >= 0)
            pixels_lb_mask = pixels[lb_mask]
            lens_bin_lb_mask = lens_bin[lb_mask]
            split_lb_mask = split[lb_mask]

            # Accumulate into the total noise maps. Under JAX this can't be an in-place
            # operation, so we have to replace G1 each time. Under the hood this may
            # be happening in-place, I think it depends.
            G1 = GN_add_jit(G1, masked_pixels, masked_source_bin, masked_g1r)
            G2 = GN_add_jit(G2, masked_pixels, masked_source_bin, masked_g2r)
            GW = GW_add_jit(GW, masked_pixels, masked_source_bin, masked_weights)
            ngal_split = ngal_split_add_jit(
                ngal_split,
                pixels_lb_mask,
                clustering_realizations,
                split_lb_mask,
                lens_bin_lb_mask,
            )
            # TODO: Currently breaks with clustering_realizations > 1

        # Now we have finished looping through the data, we sum everything over the
        # different processes to the root process
        if self.comm is not None:
            import mpi4jax
            from mpi4py import MPI

            G1, token = mpi4jax.reduce(G1, MPI.SUM, root=0)
            G2, token = mpi4jax.reduce(G2, MPI.SUM, root=0, token=token)
            G2, token = mpi4jax.reduce(GW, MPI.SUM, root=0, token=token)
            ngal_split, token = mpi4jax.reduce(ngal_split, MPI.SUM, root=0, token=token)
            if self.rank != 0:
                del G1, G2, GW, ngal_split

        # Save the maps on the root processor
        if self.rank == 0:
            print("Saving maps")

            # First we save the source noise maps
            outfile = self.open_output("source_noise_maps", wrapper=True)

            # The top section has the metadata in it
            group = outfile.file.create_group("maps")
            # TODO: sort out nbin vs nbin_source, nbin_lens
            group.attrs["nbin_source"] = nbin_source
            group.attrs["nbin"] = nbin_source
            group.attrs["lensing_realizations"] = lensing_realizations

            # Get outputs from GPU
            G1 = device_get(G1)
            G2 = device_get(G2)
            GW = device_get(GW)
            metadata = {**self.config, **map_info}

            # We save only the hit pixels
            pixels = np.where(mask > 0)[0]

            # Loop through each realization of each bin
            for b in range(nbin_source):
                for i in range(lensing_realizations):

                    # Normalize this bin with the weights
                    bin_mask = np.where(GW[:, b] > 0)
                    g1 = G1[:, b, i] / GW[:, b]
                    g2 = G2[:, b, i] / GW[:, b]

                    # and save g1 and g2 maps to the file.
                    outfile.write_map(
                        f"rotation_{i}/g1_{b}", pixels[bin_mask], g1[bin_mask], metadata
                    )

                    outfile.write_map(
                        f"rotation_{i}/g2_{b}", pixels[bin_mask], g2[bin_mask], metadata
                    )

            # Similar for the lensing noise maps
            outfile = self.open_output("lens_noise_maps", wrapper=True)
            group = outfile.file.create_group("maps")
            group.attrs["nbin_lens"] = nbin_lens
            group.attrs["clustering_realizations"] = clustering_realizations

            for b in range(nbin_lens):
                for i in range(clustering_realizations):
                    # We have computed the first half already,
                    # and we have the total map from an earlier stage
                    half1 = ngal_split[:, b, i, 0]
                    half2 = ngal_split[:, b, i, 1]

                    # Convert to overdensity.  I thought about
                    # using half the mean from the full map to reduce
                    # noise, but thought that might add covariance
                    # to the two maps, and this shouldn't be that noisy
                    mu1 = np.average(half1, weights=mask[pixels])
                    mu2 = np.average(half2, weights=mask[pixels])
                    # This will produce some mangled sentinel values
                    # but they will be masked out
                    rho1 = (half1 - mu1) / mu1
                    rho2 = (half2 - mu2) / mu2

                    # Write both overdensity and count maps
                    # for each bin for each split
                    outfile.write_map(f"split_{i}/rho1_{b}", pixels, rho1, metadata)
                    outfile.write_map(f"split_{i}/rho2_{b}", pixels, rho2, metadata)
                    # counts
                    outfile.write_map(f"split_{i}/ngal1_{b}", pixels, half1, metadata)
                    outfile.write_map(f"split_{i}/ngal2_{b}", pixels, half2, metadata)

    def read_inputs(self):

        with self.open_input("mask", wrapper=True) as f:
            mask = f.read_map("mask")
            # pixelization etc
            map_info = f.read_map_info("mask")

        with self.open_input("lens_maps", wrapper=True) as f:
            nbin_lens = f.file["maps"].attrs["nbin"]
            ngal_maps = [f.read_map(f"ngal_{b}") for b in range(nbin_lens)]

        with self.open_input("shear_tomography_catalog") as f:
            nbin_source = f["tomography"].attrs["nbin"]
            sz1 = f["tomography/bin"].size

        with self.open_input("lens_tomography_catalog") as f:
            sz2 = f["tomography/bin"].size

        if sz1 != sz2:
            raise ValueError(
                "Lens and source catalogs are different sizes in "
                "TXNoiseMaps. In this case run TXSourceNoiseMaps "
                "and TXLensNoiseMaps separately."
            )

        return nbin_source, nbin_lens, ngal_maps, mask, map_info
