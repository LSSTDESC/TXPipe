from .base_stage import PipelineStage
from .data_types import (
    MapsFile,
    YamlFile,
    RandomsCatalog,
    TomographyCatalog,
    QPNOfZFile,
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
        ("mask", MapsFile),
        ("lens_photoz_stack", QPNOfZFile),
        ("fiducial_cosmology", FiducialCosmology),
    ]
    outputs = [
        ("random_cats", RandomsCatalog),
        ("binned_random_catalog", RandomsCatalog),
        ("binned_random_catalog_sub", RandomsCatalog),
    ]
    config_options = {
        "density": 100.0,  # number per square arcmin at median depth depth.  Not sure if this is right.
        "Mstar": 23.0,  # Schecther distribution Mstar parameter
        "alpha": -1.25,  # Schecther distribution alpha parameter
        "chunk_rows": 100_000,
        "method":"quadrilateral", #method should be "quadrilateral" or "spherical_projection"
        "sample_rate": 0.5,  # fraction of random catalog to be retained in the sub-sampled catalog
                             # This should be larger than ~1.11*sqrt(Ndata/Nrandom) to maintain the same shot noise precision 
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

        # Load the input mask 
        # TODO: add option to use higher resolution healsparse map to draw randoms from
        with self.open_input("mask", wrapper=True) as maps_file:
            mask = maps_file.read_map("mask")

        with self.open_input("lens_photoz_stack", wrapper=True) as f:
            # This is a QP object
            n_of_z_object = f.read_ensemble()
            Ntomo = n_of_z_object.npdf - 1 # ensemble includes the non-tomo 2D n(z)

        # We also generate comoving distances under a fiducial cosmology
        # for each random, for use in Rlens type metrics
        with self.open_input("fiducial_cosmology", wrapper=True) as f:
            cosmo = f.to_ccl()

        # Cut down to pixels that have any objects in
        pixel = np.where(mask > 0)[0]
        depth = depth[pixel]
        frac = mask[pixel]
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

        # Pixel geometry - area of a single pixel in arcmin^2
        full_pix_area = scheme.pixel_area(degrees=True) * 60.0 * 60.0
        pix_area = frac*full_pix_area

        if method == "quadrilateral":
            vertices = scheme.vertices(pixel)
        else:
            vertices = None

        ##################################################################################

        ### I think this is redundant if your pz_stack file has separated lens bins

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

                    # Generate random redshifts in the distribution using QP's tools
                    z_photo_rand = n_of_z_object.rvs(size=N)[j]

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

                # Generate random redshifts in the distribution using QP's tools
                z_photo_rand = n_of_z_object.rvs(size=N)[j]

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

        print('Sub-sampling randoms at rate {0}'.format(self.config["sample_rate"]))
        self.subsample_randoms(binned_output)

        if self.comm is not None:
            self.comm.Barrier()
        output_file.close()
        binned_output.close()


    def subsample_randoms(self,binned_output):
        """Randomly subsample the binned random catalog and saves catalog

        This can be used within the 2-point clustering stage for the RR term, to speed up the 
        calculation without losing precision

        Currently reloads the binned randoms and saves them
        """
        from .utils.hdf_tools import BatchWriter

        sample_rate = self.config["sample_rate"]

        #get number of tomographic bins in binned random catalog
        Ntomo = binned_output['randoms'].attrs["nbin"]

        # Only save the binned random catalog for this stage
        binned_output_sub = self.open_output("binned_random_catalog_sub", parallel=True)
        binned_group_sub = binned_output_sub.create_group("randoms")
        subgroups = []
        binned_group_sub.attrs["nbin"] = Ntomo

        for j in range(Ntomo):

            #load the columns from full catalog in this tomo bin
            #We could add a feature to loop over chunks of data here, if random catalogs ever get really large
            ra = binned_output[f"randoms/bin_{j}/ra"][:]
            dec = binned_output[f"randoms/bin_{j}/dec"][:]
            z = binned_output[f"randoms/bin_{j}/z"][:]
            comoving_distance = binned_output[f"randoms/bin_{j}/comoving_distance"][:]

            #create subsampling array
            ntotal = len(ra)
            nsub = int(sample_rate*ntotal)
            select_sub = np.random.choice(np.arange(len(ra)),size=nsub,replace=False)

            #create hdf group
            subgroup = binned_group_sub.create_group(f"bin_{j}")
            subgroup.create_dataset("ra", (nsub,))
            subgroup.create_dataset("dec", (nsub,))
            subgroup.create_dataset("z", (nsub,))
            subgroup.create_dataset("comoving_distance", (nsub,))
            subgroups.append(subgroup)

            # batch up chunks of output to be done in large
            # sets, so that whatever the size of the randoms in this bin it will
            # still work.
            batch2 = BatchWriter(
                subgroup,
                {
                    "ra": np.float64,
                    "dec": np.float64,
                    "z": np.float64,
                    "comoving_distance": np.float64,
                },
                offset=0, #check this is right
                max_size=self.config["chunk_rows"],
            )

            batch2.write(   ra=ra[select_sub], 
                            dec=dec[select_sub], 
                            z=z[select_sub], 
                            comoving_distance=comoving_distance[select_sub]
                            )

            batch2.finish()

        if self.comm is not None:
            self.comm.Barrier()
        binned_output_sub.close()


class TXSubsampleRandoms(PipelineStage):
    """
    Randomly subsample the binned random catalog and save catalog
    This can be used within the 2-point clustering stage for the RR term, to speed up the 
    calculation without losing precision

    The subsampling is already run by default in TXRandomCat
    Use this subclass if you are loading your randoms from elsewhere and need to subsample
    """
    name = "TXSubsampleRandoms"
    inputs = [
        ("binned_random_catalog", HDFFile),
    ]
    outputs = [
        ("binned_random_catalog_sub", RandomsCatalog),
    ]
    config_options = {
        "chunk_rows": 100_000,
        "sample_rate": 0.5,  # fraction of random catalog that should be retained in the subsampled catalog
    }

    def run(self):
        from . import randoms
        from .utils.hdf_tools import BatchWriter

        sample_rate = self.config["sample_rate"]

        #get number of tomographic bins in binned random catalog
        with self.open_input("binned_random_catalog") as f:
            Ntomo = f['randoms'].attrs["nbin"]

        # Only save the binned random catalog for this stage
        binned_output = self.open_output("binned_random_catalog_sub", parallel=True)
        binned_group = binned_output.create_group("randoms")
        subgroups = []
        binned_group.attrs["nbin"] = Ntomo

        for j in range(Ntomo):

            #load the columns from full catalog in this tomo bin
            with self.open_input("binned_random_catalog") as f:
                ra = f[f"randoms/bin_{j}/ra"][:]
                dec = f[f"randoms/bin_{j}/dec"][:]
                z = f[f"randoms/bin_{j}/z"][:]
                comoving_distance = f[f"randoms/bin_{j}/comoving_distance"][:]

            #create subsampling array
            ntotal = len(ra)
            nsub = int(sample_rate*ntotal)
            select_sub = np.random.choice(np.arange(len(ra)),size=nsub,replace=False)

            #create hdf group
            subgroup = binned_group.create_group(f"bin_{j}")
            subgroup.create_dataset("ra", (nsub,))
            subgroup.create_dataset("dec", (nsub,))
            subgroup.create_dataset("z", (nsub,))
            subgroup.create_dataset("comoving_distance", (nsub,))
            subgroups.append(subgroup)

            # batch up chunks of output to be done in large
            # sets, so that whatever the size of the randoms in this bin it will
            # still work.
            batch2 = BatchWriter(
                subgroup,
                {
                    "ra": np.float64,
                    "dec": np.float64,
                    "z": np.float64,
                    "comoving_distance": np.float64,
                },
                offset=0, #check this is right
                max_size=self.config["chunk_rows"],
            )

            batch2.write(   ra=ra[select_sub], 
                            dec=dec[select_sub], 
                            z=z[select_sub], 
                            comoving_distance=comoving_distance[select_sub]
                            )

            batch2.finish()

        if self.comm is not None:
            self.comm.Barrier()
        binned_output.close()

if __name__ == "__main__":
    PipelineStage.main()
