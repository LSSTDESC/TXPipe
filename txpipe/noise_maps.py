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


class TXSourceNoiseMaps(TXBaseMaps):
    """
    Generate realizations of shear noise maps with random rotations

    This takes the shear catalogs and tomography and randomly spins the
    shear values in it, removing the shear signal and leaving only shape noise
    """
    name = "TXSourceNoiseMaps"

    inputs = [
        ("shear_catalog", ShearCatalog),
        ("shear_tomography_catalog", TomographyCatalog),
        # We get the pixelization info from the diagnostic maps
        ("mask", MapsFile),
    ]

    outputs = [
        ("source_noise_maps", LensingNoiseMaps),
    ]

    config_options = {
        "chunk_rows": 100000,
        "lensing_realizations": 30,
        "true_shear": False,
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

        with self.open_input("shear_tomography_catalog", wrapper=True) as f:
            nbin_source = f.file["tomography"].attrs["nbin_source"]

        # Mapping from 0 .. nhit - 1 to healpix indices
        reverse_map = np.where(mask > 0)[0]
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

        with self.open_input("shear_catalog", wrapper=True) as f:
            shear_cols, renames = f.get_primary_catalog_names(self.config["true_shear"])

        it = self.combined_iterators(
            self.config["chunk_rows"],
            "shear_catalog",
            "shear",
            shear_cols,
            "shear_tomography_catalog",
            "tomography",
            ["source_bin"],
        )
        return rename_iterated(it, renames)

    def accumulate_maps(self, pixel_scheme, data, mappers):
        npix, G1, G2, GW, index_map, _, _ = mappers
        lensing_realizations = self.config["lensing_realizations"]
        source_bin = data["source_bin"]

        # Get the pixel index for each object and convert
        # to the reduced index
        ra = data["ra"]
        dec = data["dec"]
        orig_pixels = pixel_scheme.ang2pix(ra, dec)
        pixels = index_map[orig_pixels]

        # Pull out some columns we need
        n = len(ra)
        w = data["weight"]
        # Pre-weight the g1 values so we don't have to
        # weight each realization again
        g1 = data["g1"] * w
        g2 = data["g2"] * w

        # random rotations of the g1, g2 values
        phi = np.random.uniform(0, 2 * np.pi, (n, lensing_realizations))
        c = np.cos(phi)
        s = np.sin(phi)
        g1r = c * g1[:, np.newaxis] + s * g2[:, np.newaxis]
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
        lensing_realizations = self.config["lensing_realizations"]

        # Sum everything at root
        if self.comm is not None:
            mpi_reduce_large(
                G1, self.comm, max_chunk_count=2**26
            )  # fiducial is 2**30
            mpi_reduce_large(G2, self.comm, max_chunk_count=2**26)
            mpi_reduce_large(GW, self.comm, max_chunk_count=2**26)
            if self.rank != 0:
                del G1, G2, GW

        # build up output
        maps = {}

        # only master gets full stuff
        if self.rank != 0:
            return maps

        # We need to calibrate the shear maps
        cal, _ = Calibrator.load(self.get_input("shear_tomography_catalog"))

        for b in range(nbin_source):
            for i in range(lensing_realizations):

                bin_mask = np.where(GW[:, b] > 0)

                g1 = G1[:, b, i] / GW[:, b]
                g2 = G2[:, b, i] / GW[:, b]

                g1, g2 = cal[b].apply(g1, g2, subtract_mean=False)

                maps["source_noise_maps", f"rotation_{i}/g1_{b}"] = (
                    reverse_map[bin_mask],
                    g1[bin_mask],
                )

                maps["source_noise_maps", f"rotation_{i}/g2_{b}"] = (
                    reverse_map[bin_mask],
                    g2[bin_mask],
                )
        return maps


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
            nbin_lens = f.file["tomography"].attrs["nbin_lens"]

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
            ["lens_bin"],
        )
        return it

    def accumulate_maps(self, pixel_scheme, data, mappers):
        npix, ngal_split, index_map, _, _, _ = mappers
        clustering_realizations = self.config["clustering_realizations"]

        # Tomographic bin
        lens_bin = data["lens_bin"]

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
            ["lens_bin"],
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
            ["source_bin"],
            "lens_tomography_catalog",
            "tomography",
            ["lens_bin"],
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
            source_bin = device_put(data["source_bin"])
            lens_bin = device_put(data["lens_bin"])
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
            group.attrs["nbin_source"] = nbin_source
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
            nbin_lens = f.file["maps"].attrs["nbin_lens"]
            ngal_maps = [f.read_map(f"ngal_{b}") for b in range(nbin_lens)]

        with self.open_input("shear_tomography_catalog") as f:
            nbin_source = f["tomography"].attrs["nbin_source"]
            sz1 = f["tomography/source_bin"].size

        with self.open_input("lens_tomography_catalog") as f:
            sz2 = f["tomography/lens_bin"].size

        if sz1 != sz2:
            raise ValueError(
                "Lens and source catalogs are different sizes in "
                "TXNoiseMaps. In this case run TXSourceNoiseMaps "
                "and TXLensNoiseMaps separately."
            )

        return nbin_source, nbin_lens, ngal_maps, mask, map_info
