from .base_stage import PipelineStage
from .data_types import (
    MapsFile,
    YamlFile,
    RandomsCatalog,
    TomographyCatalog,
    HDFFile,
    FiducialCosmology,
)
from .utils import choose_pixelization, Splitter
import numpy as np

class TXRandomCat(PipelineStage):
    """
    Generate a catalog of randomly positioned points

    This accounts for the depth being different in each pixel, but probably
    does still need updates, and testing.
    """
    name = "TXRandomCat"
    inputs = [
        ("aux_lens_maps", MapsFile),
        ("lens_photoz_stack", HDFFile),
        ("fiducial_cosmology", FiducialCosmology),
    ]
    outputs = [
        ("random_cats", RandomsCatalog),
        ("binned_random_catalog", RandomsCatalog),
    ]
    config_options = {
        "density": 100.0,  # number per square arcmin at median depth depth.  Not sure if this is right.
        "Mstar": 23.0,  # Schecther distribution Mstar parameter
        "alpha": -1.25,  # Schecther distribution alpha parameter
        "chunk_rows": 100_000,
        "method":"quadrilateral", #method should be "quadrilateral" or "spherical_projection"
    }

    def run(self):
        import scipy.special
        import scipy.stats
        import healpy
        import pyccl
        from . import randoms
        from .utils.hdf_tools import BatchWriter
        import healpix

        # Load the input depth map
        with self.open_input("aux_lens_maps", wrapper=True) as maps_file:
            depth = maps_file.read_map("depth/depth")
            info = maps_file.read_map_info("depth/depth")
            nside = info["nside"]
            scheme = choose_pixelization(**info)

        pz_stack = self.open_input("lens_photoz_stack")

        # We also generate comoving distances under a fiducial cosmology
        # for each random, for use in Rlens type metrics
        with self.open_input("fiducial_cosmology", wrapper=True) as f:
            cosmo = f.to_ccl()


        # Cut down to pixels that have any objects in
        pixel = np.where(depth > 0)[0]
        depth = depth[pixel]
        npix = depth.size

        if len(pixel) == 1:
            raise ValueError("Only one pixel in depth map!")

        ### This will likely need to be changed with the density Schechter function and density
        ### being calculated individually for each redshift bin.  When it comes time to update the code
        ### to do that, move the lines below (down to the multi hash line) into the Ntomo loop

        # Read configuration values
        Mstar = self.config["Mstar"]
        alpha15 = 1.5 + self.config["alpha"]
        density_at_median = self.config["density"]
        method = self.config["method"]
        allowed_methods = ["quadrilateral","spherical_projection"]
        assert method in allowed_methods

        # Work out the normalization of a Schechter distribution
        # with the given median depth
        median_depth = np.median(depth)
        x_med = 10.0 ** (0.4 * (Mstar - median_depth))
        phi_star = density_at_median / scipy.special.gammaincc(alpha15, x_med)

        # Work out the number density in each pixel based on the
        # given Schecter distribution
        x = 10.0 ** (0.4 * (Mstar - depth))
        density = phi_star * scipy.special.gammaincc(alpha15, x)

        # Pixel geometry - area in arcmin^2
        pix_area = scheme.pixel_area(degrees=True) * 60.0 * 60.0
        if method == "quadrilateral":
            vertices = scheme.vertices(pixel)
        else:
            vertices = None

        ##################################################################################

        ### I think this is redundant if your pz_stack file has separated lens bins

        ### The current file in 'pz_stack' only has 1 lens bin, but zbins loads 4 bins!!
        Ntomo = len(pz_stack["n_of_z"]["lens"].keys()) - 1
        z_photo_arr = pz_stack["n_of_z"]["lens"]["z"][:]

        ### Loop over the tomographic bins to find number of galaxies in each pixel/zbin
        ### When the density changes per redshift bin, this can go into the main Ntomo loop
        numbers = np.zeros((Ntomo, npix), dtype=int)
        if self.rank == 0:
            for j in range(Ntomo):
                # Poisson distribution about mean
                numbers[j] = scipy.stats.poisson.rvs(density * pix_area, 1)

        # give all processors the same values
        if self.comm is not None:
            self.comm.Bcast(numbers)

        # total number of objects in each bin
        bin_counts = numbers.sum(axis=1)

        # start index of the total number in the global bin
        bin_starts = np.concatenate([[0], np.cumsum(bin_counts)])[:-1]

        # The starting index of each pixel in the array, so we can parallelize
        pix_starts = np.zeros((Ntomo, npix), dtype=int)
        for j in range(Ntomo):
            pix_starts[j] = np.concatenate([[0], np.cumsum(numbers[j])])[:-1]

        ### Get total number of randoms in all zbins
        ### Once the density gets updated per redshift bin, the output file will need to
        ### combine all the tomographic bins in a bit more clever/convenient way than currently
        n_total = numbers.sum()
        if self.rank == 0:
            print(f"Generating {n_total} randoms")
            for j in range(Ntomo):
                print(f"  - {bin_counts[j]} in bin {j}")

        # First output is the all of the
        output_file = self.open_output("random_cats", parallel=True)
        group = output_file.create_group("randoms")
        ra_out = group.create_dataset("ra", (n_total,), dtype=np.float64)
        dec_out = group.create_dataset("dec", (n_total,), dtype=np.float64)
        z_out = group.create_dataset("z", (n_total,), dtype=np.float64)
        chi_out = group.create_dataset("comoving_distance", (n_total,), dtype=np.float64)
        bin_out = group.create_dataset("bin", (n_total,), dtype=np.int16)

        # Second output is specific to an individual bin, so we can just load
        # a single bin as needed
        binned_output = self.open_output("binned_random_catalog", parallel=True)
        binned_group = binned_output.create_group("randoms")

        subgroups = []
        binned_group.attrs["nbin"] = Ntomo
        for i in range(Ntomo):
            g = binned_group.create_group(f"bin_{i}")
            g.create_dataset("ra", (bin_counts[i],))
            g.create_dataset("dec", (bin_counts[i],))
            g.create_dataset("z", (bin_counts[i],))
            g.create_dataset("comoving_distance", (bin_counts[i],))
            subgroups.append(g)

        pixels_per_proc = npix // self.size

        for j in range(Ntomo):
            ### Load pdf of ith lens redshift bin pz
            n_hist = pz_stack[f"n_of_z/lens/bin_{j}"][:]

            ### Make cdf and normalise
            z_cdf = np.cumsum(n_hist)
            z_cdf_norm = z_cdf / np.float(max(z_cdf))

            subgroup = subgroups[j]
            # Generate the random points in each pixel
            ndone = 0

            nvertex = npix #number of verticies is the same as number of pixels
            my_nvertex = int(np.ceil(nvertex / self.size))
            start_vertex = self.rank * my_nvertex
            end_vertex = min(start_vertex + my_nvertex, nvertex)

            # These two classes batch up chunks of output to be done in large
            # sets, so that whatever the size of the randoms in this bin it will
            # still work.
            batch1 = BatchWriter(
                group,
                {
                    "ra": np.float64,
                    "dec": np.float64,
                    "z": np.float64,
                    "comoving_distance": np.float64,
                    "bin": np.int16,
                },
                offset=bin_starts[j] + pix_starts[j, start_vertex],
                max_size=self.config["chunk_rows"],
            )
            batch2 = BatchWriter(
                subgroup,
                {
                    "ra": np.float64,
                    "dec": np.float64,
                    "z": np.float64,
                    "comoving_distance": np.float64,
                },
                offset=pix_starts[j, start_vertex],
                max_size=self.config["chunk_rows"],
            )

            if method == "quadrilateral":
                for i in range(start_vertex, end_vertex):
                    vertices_i = vertices[i]
                    if ndone % 1000 == 0:
                        print(
                            f"Rank {self.rank} done {ndone:,} of its {pixels_per_proc:,} pixels for bin {j}"
                        )  # Use the pixel vertices to generate the points
                    ### This likely wont work for curved sky maps since healpy pixels aren't
                    ### fully quadrilateral... not sure how big of a difference (if any) this
                    ### will make
                    N = numbers[j, i]
                    p1, p2, p3, p4 = vertices_i.T
                    P = randoms.random_points_in_quadrilateral(p1, p2, p3, p4, N)
                    # Convert to RA/Dec
                    # This is not healpy-dependent so we just use it as a convenience function
                    ra, dec = healpy.vec2ang(P, lonlat=True)

                    bin_index = np.repeat(j, N)

                    ### Create random values [0,1] equal to the number of galaxies per pixel
                    cdf_rand_val = np.random.uniform(0, 1.0, N)
                    ### Interpolate those random values to a redshift value given by the cdf
                    # z_photo_rand = np.interp(cdf_rand_val,z_cdf_norm,z_photo_arr)
                    z_interp_func = scipy.interpolate.interp1d(z_cdf_norm, z_photo_arr)
                    # Sometimes we don't quite go down to z - deal with that
                    cdf_rand_val = cdf_rand_val.clip(z_cdf_norm.min(), z_cdf_norm.max())
                    z_photo_rand = z_interp_func(cdf_rand_val)
                    distance = pyccl.comoving_radial_distance(
                        cosmo, 1.0 / (1 + z_photo_rand)
                    )

                    # Save output to the generic non-binned output
                    batch1.write(
                        ra=ra,
                        dec=dec,
                        z=z_photo_rand,
                        comoving_distance=distance,
                        bin=bin_index,
                    )

                    # Save to the bit that is specific to this bin
                    batch2.write(ra=ra, dec=dec, z=z_photo_rand, comoving_distance=distance)

                    ndone += 1
            elif method == "spherical_projection":
                #use the same batch/chunks as the quadrilateral method
                #though i dont think it is likely to be neccesary with this method

                #pixel id for each random objects in this chunk
                pix_catalog = np.repeat(pixel[start_vertex:end_vertex], numbers[j,:][start_vertex:end_vertex])
                
                #generate a random location within the pixel using healpix 
                ra, dec = healpix.randang(nside, pix_catalog, lonlat=True)
                
                N = len(pix_catalog)
                bin_index = np.repeat(j, N)

                ### Create random values [0,1] equal to the number of galaxies per pixel
                cdf_rand_val = np.random.uniform(0, 1.0, N)
                ### Interpolate those random values to a redshift value given by the cdf
                # z_photo_rand = np.interp(cdf_rand_val,z_cdf_norm,z_photo_arr)
                z_interp_func = scipy.interpolate.interp1d(z_cdf_norm, z_photo_arr)
                # Sometimes we don't quite go down to z - deal with that
                cdf_rand_val = cdf_rand_val.clip(z_cdf_norm.min(), z_cdf_norm.max())
                z_photo_rand = z_interp_func(cdf_rand_val)
                distance = pyccl.comoving_radial_distance(
                    cosmo, 1.0 / (1 + z_photo_rand)
                )

                # Save output to the generic non-binned output
                batch1.write(
                    ra=ra,
                    dec=dec,
                    z=z_photo_rand,
                    comoving_distance=distance,
                    bin=bin_index,
                )

                # Save to the bit that is specific to this bin
                batch2.write(ra=ra, dec=dec, z=z_photo_rand, comoving_distance=distance)
            else:
                raise RuntimeError('method must be one of {0}'.format(allowed_methods))


            batch1.finish()
            batch2.finish()

        if self.comm is not None:
            self.comm.Barrier()
        output_file.close()
        binned_output.close()


if __name__ == "__main__":
    PipelineStage.main()
