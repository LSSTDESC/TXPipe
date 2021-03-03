from .base_stage import PipelineStage
from .data_types import HDFFile, MapsFile, TextFile, SACCFile
from .utils import choose_pixelization, multi_where
import re
import time
import numpy as np
import itertools


class HDFSplitter:
    """
    Helper class to write out data that is split into bins

    """
    def __init__(self, group, name, columns, bins, initial_size, dtypes=None):
        self.bins = bins
        self.group = group
        self.columns = columns
        self.index = {b: 0 for b in bins}
        self.sizes = {b: initial_size for b in bins}
        self.subgroups = {b: group.create_group(f"{name}_{b}") for b in bins}

        for i, b in enumerate(bins):
            group.attrs[f'bin_{i}'] = b

        dtypes = dtypes or {}

        for b in bins:
            sub = self.subgroups[b]
            for col in columns:
                dt = dtypes.get(col, 'f8')
                sub.create_dataset(col, (initial_size,), maxshape=(None,), dtype=dt, chunks=True)

    def write(self, data, bins):

        wheres = multi_where(bins, self.bins)

        for b in self.bins:
            # Get the index of objects that fall in this bin
            w = wheres[b]
            if w.size == 0:
                continue

            # Make the right subsets of the data according to this index
            bin_data = {col: data[col][w] for col in self.columns}

            # Save this chunk of data
            self.write_bin(bin_data, b)

        return wheres

    def check_enlarge(self, b, e):
        if e > self.sizes[b]:
            group = self.subgroups[b]
            sz = int(1.5 * e)
            self.sizes[b] = sz
            for col in self.columns:
                group[col].resize((sz,))


    def write_bin(self, data, b):
        # Length of this chunk
        n = len(data[self.columns[0]])
        # Group where we will write the data
        group = self.subgroups[b]
        # Indices of this output
        s = self.index[b]
        e = s + n

        # Enlarge columns if needed
        self.check_enlarge(b, e)

        # Write to columns
        for col in self.columns:
            group[col][s:e] = data[col]

        # Update overall index
        self.index[b] = e

    def finalize(self):
        for b, sub in self.subgroups.items():
            for col in self.columns:
                sub[col].resize((self.index[b],))




class CMIngestHalosCosmoDC2(PipelineStage):
    name = "CMIngestHalosCosmoDC2"
    parallel = False
    inputs = []
    outputs = [("cluster_mag_halo_catalog", HDFFile)]
    config_options = {
        "cat_name": "cosmoDC2_v1.1.4_image",
        "halo_mass_min": 0.5e13,
        "initial_size": 100_000,
        "ra_range": [50.0, 73.1],
        "dec_range": [-45.0, -27.0],
    }

    def run(self):
        import GCRCatalogs

        # Configuration options
        mass_min = self.config["halo_mass_min"]
        cat_name = self.config["cat_name"]
        sz = self.config["initial_size"]

        # Open the cosmoDC2 catalog
        overwrite = {
            "check_md5": False,
            "check_size": False,
            "ensure_meta_consistent": False,
        }
        cat = GCRCatalogs.load_catalog(cat_name, config_overwrite=overwrite)

        # Selection of data we will read from it below
        cols = ["halo_mass", "redshift", "ra", "dec", "halo_id"]
        filters = [
            f"halo_mass > {mass_min}",
            "is_central == True",
            f"ra > {ra_range[0]}",
            f"ra < {ra_range[1]}",
            f"dec > {dec_range[0]}",
            f"dec < {dec_range[1]}",
        ]

        # Create output data file with extensible data sets
        f = self.open_output("cluster_mag_halo_catalog")
        g = f.create_group("halos")
        g.create_dataset("halo_mass", (sz,), maxshape=(None,), dtype="f8", chunks=True)
        g.create_dataset("redshift", (sz,), maxshape=(None,), dtype="f8", chunks=True)
        g.create_dataset("ra", (sz,), maxshape=(None,), dtype="f8", chunks=True)
        g.create_dataset("dec", (sz,), maxshape=(None,), dtype="f8", chunks=True)
        g.create_dataset("halo_id", (sz,), maxshape=(None,), dtype="i8", chunks=True)

        # Prepare the iterator to loop through GCR
        it = cat.get_quantities(cols, filters=filters, return_iterator=True)

        # s is the start index for the next data chunk
        s = 0
        for data in it:
            # e is the end index for this data chunk
            e = s + data["ra"].size
            print(f"Read data chunk {s:,} - {e:,}")

            # Expand the data sets if we are exceeding the current
            # size.  Grow by 50% each time.
            if e > sz:
                sz = int(1.5 * e)
                print(f"Resizing data to {sz:,}")
                for col in cols:
                    g[col].resize((sz,))

            # Output this chunk of data to the file
            for col in cols:
                g[col][s:e] = data[col]

            # Update the starting index for the next chunk
            s = e

        print(f"Ingestion complete. Resizing to final halo count {e:,}")
        # Now we have finished we can truncate any
        # excess space in the output data
        for col in cols:
            g[col].resize((e,))

        # And that's all.
        f.close()

class CMSelectHalos(PipelineStage):
    name = "CMSelectHalos"
    parallel = False
    inputs = [("cluster_mag_halo_catalog", HDFFile)]
    outputs = [("cluster_mag_halo_tomography", HDFFile)]
    config_options = {
        "zedge": [0.2, 0.4, 0.6, 0.8, 1.0, 1.2],
        "medge": [20, 30, 45, 70, 120, 220]

    }
    def run(self):
        
        zedge = np.array(self.config['zedge'])
        # where does this number 45 come from?
        medge = np.array(self.config['medge']) * (1e14 / 45)

        nz = len(zedge) - 1
        nm = len(medge) - 1

        # add infinities to either end to catch objects that spill out
        zedge = np.concatenate([[-np.inf], zedge, [np.inf]])
        medge = np.concatenate([[-np.inf], medge, [np.inf]])

        # all pairs of z bin, m bin indices
        bins = list(itertools.product(range(nz), range(nm)))
        bin_names = [f"{i}_{j}" for i,j in bins]

        # my_bins = [i, pair for pair in self.split_tasks_by_rank(enumerate(bins))]
        cols = ["halo_mass", "redshift", "ra", "dec"]
        chunk_rows = 100_000
        it = self.iterate_hdf("cluster_mag_halo_catalog", "halos", cols, chunk_rows)

        f = self.open_output("cluster_mag_halo_tomography")
        g = f.create_group("tomography")
        initial_size = 100_000
        splitter = HDFSplitter(g, "bin", cols, bin_names, initial_size)

        for _, _, data in it:
            n = len(data["redshift"])
            zbin = np.digitize(data['redshift'], zedge)
            mbin = np.digitize(data['halo_mass'], medge)

            # Find which bin each object is in, or None
            sel_bins = []
            for i in range(n):
                zi = zbin[i]
                mi = mbin[i]
                if (zi == 0) or (zi == nz + 1) or (mi == 0) or (mi == nm + 1):
                    b = None
                else:
                    b = f"{zi-1}_{mi-1}"
                sel_bins.append(b)

            # Save data to the correct subgroup
            splitter.write(data, sel_bins)

        # Truncate arrays to correct size
        splitter.finalize()

        # Save metadata
        for (i, j), name in zip(bins, bin_names):
            metadata = splitter.subgroups[name].attrs
            metadata['mass_min'] = medge[i+1]
            metadata['mass_max'] = medge[i+2]
            metadata['z_min'] = zedge[i+1]
            metadata['z_max'] = zedge[i+2]

        f.close()



class CMBackgroundSelector(PipelineStage):
    name = "CMBackgroundSelector"
    parallel = False
    inputs = [("photometry_catalog", HDFFile)]
    outputs = [("cluster_mag_background", HDFFile), ("cluster_mag_footprint", MapsFile)]

    # Default configuration settings
    config_options = {
        "ra_range": [50.0, 73.1],
        "dec_range": [-45.0, -27.0],
        "mag_cut": 26.0,
        "zmin": 1.5,
        "nside": 2048,
        "initial_size": 100_000,
        "chunk_rows": 100_000,
    }

    def run(self):
        import healpy

        #  Count the max number of objects we will look at in total
        with self.open_input("photometry_catalog") as f:
            N = f["photometry/ra"].size

        #  Open and set up the columns in the output
        f = self.open_output("cluster_mag_background")
        g = f.create_group("sample")
        sz = self.config["initial_size"]
        ra = g.create_dataset("ra", (sz,), maxshape=(None,))
        dec = g.create_dataset("dec", (sz,), maxshape=(None,))

        # Get values from the user configutation.
        # These can be set on the command line, in the config file,
        #  or use the default values above
        ra_min, ra_max = self.config["ra_range"]
        dec_min, dec_max = self.config["dec_range"]
        mag_cut = self.config["mag_cut"]
        zmin = self.config["zmin"]
        nside = self.config["nside"]
        chunk_rows = self.config["chunk_rows"]

        # We will keep track of a hit map to help us build the
        #  random catalog later.  Make it zero now; every time a pixel
        #  hits it we will set a value to one
        npix = healpy.nside2npix(nside)
        hit_map = np.zeros(npix)

        # Prepare an iterator that will loop through the data
        it = self.iterate_hdf(
            "photometry_catalog",
            "photometry",
            ["ra", "dec", "mag_i", "redshift_true"],
            chunk_rows,
        )

        s = 0
        # Loop through the data.  The indices s1, e1 refer to the full catalog
        # start/end, that we are selecting from. The indices s and e refer to
        #  the data we have selected
        for s1, e1, data in it:
            #  make selection
            sel = (
                (data["ra"] > ra_min)
                & (data["ra"] < ra_max)
                & (data["dec"] > dec_min)
                & (data["dec"] < dec_max)
                & (data["mag_i"] < mag_cut)
                & (data["redshift_true"] > 1.5)
            )

            #  Pull out the chunk of data we would like to select
            ra_sel = data["ra"][sel]
            dec_sel = data["dec"][sel]

            # Mark any hit pixels
            pix = healpy.ang2pix(nside, ra_sel, dec_sel, lonlat=True)
            hit_map[pix] = 1

            #  Number of selected objects
            n = ra_sel.size

            # Print out our progress
            frac = n / (e1 - s1)
            print(
                f"Read data chunk {s1:,} - {e1:,} and selected {n:,} objects ({frac:.1%})"
            )

            e = s + n
            if e > sz:
                print(f"Resizing output to {sz}")
                sz = int(1.5 * e)
                ra.resize((sz,))
                dec.resize((sz,))

            # write output.
            ra[s:e] = ra_sel
            dec[s:e] = dec_sel

            # update start for next point
            s = e

        # Chop off any unused space
        print(f"Final catalog size {e:,}")
        ra.resize((e,))
        dec.resize((e,))
        f.close()

        #  Save the footprint map.  Select all the non-zero pixels
        pix = np.where(hit_map > 0)[0]
        # We just want a binary map for now, but can upgrade this to
        #  a depth map later
        val = np.ones(pix.size, dtype=np.int8)
        metadata = {"pixelization": "healpix", "nside": nside}

        with self.open_output("cluster_mag_footprint", wrapper=True) as f:
            f.write_map("footprint", pix, val, metadata)


class CMRandoms(PipelineStage):
    name = "CMRandoms"
    inputs = [
        ("cluster_mag_halo_catalog", HDFFile),
        ("cluster_mag_footprint", MapsFile),
    ]
    outputs = [("cluster_mag_randoms", HDFFile)]
    config_options = {"density": 30.0}  #  per sq arcmin

    def run(self):
        import scipy.stats
        from .randoms import random_points_in_quadrilateral
        import healpy

        # open the footprint catalog and read in the things we need
        # from it - the list of hit pixels, and the map scheme
        with self.open_input("cluster_mag_footprint", wrapper=True) as f:
            hit_map = f.read_map("footprint")
            info = f.read_map_info("footprint")
            nside = info["nside"]
            scheme = choose_pixelization(**info)

        #  Hit pixels and the coordinates of their vertices
        pix = np.where(hit_map > 0)[0]
        vertices = scheme.vertices(pix)

        # Randomly select the number of objects per pixel, using a poisson distribution with
        #  the specified mean density, for now
        area = scheme.pixel_area(degrees=True) * 60.0 * 60.0
        density = np.repeat(self.config["density"], pix.size)
        counts = scipy.stats.poisson.rvs(density * area, 1)

        # Use the same counts for all the processors
        if self.comm is not None:
            counts = self.comm.bcast(counts)

        # total number of objects to be generated over all the pixels
        total_count = counts.sum()

        # The starting index of each pixel in the array, so we can parallelize
        starts = np.concatenate([[0], np.cumsum(counts)])

        # Open the output data and create the necessary columns
        output_file = self.open_output("cluster_mag_randoms", parallel=True)
        group = output_file.create_group("randoms")
        ra_out = group.create_dataset("ra", (total_count,), dtype=np.float64)
        dec_out = group.create_dataset("dec", (total_count,), dtype=np.float64)

        # Each processor now does a subset of the pixels, generating and saving
        #  points in each
        for i, vertex in self.split_tasks_by_rank(enumerate(vertices)):
            if (i % (10_000 * self.size)) == self.rank:
                print(
                    f"Rank {self.rank} done {i//self.size:,} of its {counts.size//self.size:,} pixels"
                )
            # Generate random points in this pixel
            p1, p2, p3, p4 = vertex.T
            N = counts[i]
            P = random_points_in_quadrilateral(p1, p2, p3, p4, N)

            # This is not healpy-specific so we just use it as a convenience function
            ra, dec = healpy.vec2ang(P, lonlat=True)

            # Write output
            s = starts[i]
            e = starts[i + 1]
            ra_out[s:e] = ra
            dec_out[s:e] = dec

        output_file.close()


class CMPatches(PipelineStage):
    """

    This is currently copied from the in-progress treecorr-mpi branch.
    Think later how to
    """

    name = "CMPatches"
    inputs = [("cluster_mag_randoms", HDFFile)]
    outputs = [("cluster_mag_patches", TextFile)]
    config_options = {
        "npatch": 32,
        "every_nth": 100,
    }

    def run(self):
        import treecorr

        input_filename = self.get_input("cluster_mag_randoms")
        output_filename = self.get_output("cluster_mag_patches")

        # Build config info
        npatch = self.config["npatch"]
        every_nth = self.config["every_nth"]
        config = {
            "ext": "randoms",
            "ra_col": "ra",
            "dec_col": "dec",
            "ra_units": "degree",
            "dec_units": "degree",
            "every_nth": every_nth,
            "npatch": npatch,
        }

        # Create the catalog
        cat = treecorr.Catalog(input_filename, config)

        # Generate and write the output patch centres
        print(f"generating {npatch} centers")
        cat.write_patch_centers(output_filename)


class CMRedshifts(PipelineStage):
    name = "CMRedshifts"
    pass


class CMCorrelations(PipelineStage):
    name = "CMCorrelations"
    inputs = [
        ("cluster_mag_halo_tomography", HDFFile),
        ("cluster_mag_background", HDFFile),
        ("cluster_mag_patches", TextFile),
        ("cluster_mag_randoms", HDFFile),
    ]
    outputs = [("cluster_mag_correlations", SACCFile),]
    config_options = {
        "min_sep": 0.5,
        "max_sep": 300.0,
        "nbins": 9,
        "bin_slop": 0.1,
        "sep_units": "arcmin",
        "cores_per_task": 32,
        "verbose": 1,
        "var_method": "jackknife",
    }

    def run(self):
        import treecorr
        import sacc

        #  the names of the input files we will need
        background_file = self.get_input("cluster_mag_background")
        randoms_file = self.get_input("cluster_mag_randoms")
        patch_centers = self.get_input("cluster_mag_patches")

        with self.open_input("cluster_mag_halo_tomography") as f:
            metadata = dict(f['tomography'].attrs)
            halo_bins = []
            for key, value in metadata.items():
                if re.match('bin_[0-9]', key):
                    halo_bins.append(value)
        print("Using halo bins: ", halo_bins)


        with self.open_input("cluster_mag_background") as f:
            sz = f['sample/ra'].size
            print(f"Background catalog size: {sz}")

        with self.open_input("cluster_mag_randoms") as f:
            sz = f['randoms/ra'].size
            print(f"Random catalog size: {sz}")



        # create background catalog
        bg_cat = treecorr.Catalog(
            background_file,
            self.config,
            patch_centers=patch_centers,
            ext="sample",
            ra_col="ra",
            dec_col="dec",
            ra_units="degrees",
            dec_units="degrees",
        )

        #  randoms catalog.  For now we use a single randoms file for both catalogs, but may want to change this.
        ran_cat = treecorr.Catalog(
            randoms_file,
            self.config,
            patch_centers=patch_centers,
            ext="randoms",
            ra_col="ra",
            dec_col="dec",
            ra_units="degrees",
            dec_units="degrees",
        )




        S = sacc.Sacc()
        S.add_tracer('misc', 'background')
        for halo_bin in halo_bins:
            S.add_tracer('misc', f'halo_{halo_bin}')

        t = time.time()
        print("Computing randoms x randoms")
        random_random = self.measure(ran_cat, ran_cat)
        t1 = time.time()
        print(f"took {t1 - t:.1f} seconds")

        t = time.time()
        print("Computing random x background")
        random_bg = self.measure(ran_cat, bg_cat)
        t1 = time.time()
        print(f"took {t1 - t:.1f} seconds")


        print("Computing background x background")
        t = time.time()
        bg_bg = self.measure(bg_cat, bg_cat)
        t1 = time.time()
        print(f"took {t1 - t:.1f} seconds")

        bg_bg.calculateXi(random_random, random_bg)

        self.add_sacc_data(S, 'background', 'background', "galaxy_density_xi", bg_bg)

        comb = [bg_bg]

        for halo_bin in halo_bins:
            print(f"Computing with halo bin {halo_bin}")
            tracer1 = f'halo_{halo_bin}'
            tracer2 = f'background'
            halo_halo, halo_bg, metadata = self.measure_halo_bin(bg_cat, ran_cat, halo_bin, random_random, random_bg)
            # We build up the comb list to get the covariance of it later
            # in the same order as our data points
            comb.append(halo_halo)
            comb.append(halo_bg)

            self.add_sacc_data(S, tracer1, tracer2, "halo_halo_density_xi", halo_halo, **metadata)
            self.add_sacc_data(S, tracer1, tracer2, "halo_galaxy_density_xi", halo_bg, **metadata)


        cov = treecorr.estimate_multi_cov(comb, self.config['var_method'])

        S.add_covariance(cov)


        S.save_fits(self.get_output("cluster_mag_correlations"))


    def add_sacc_data(self, S, tracer1, tracer2, corr_type, corr, **tags):
        theta = np.exp(corr.meanlogr)
        npair = corr.npairs
        weight = corr.weight

        xi = corr.xi
        err = np.sqrt(corr.varxi)
        n = len(xi)
        for i in range(n):
            S.add_data_point(corr_type, (tracer1, tracer2), xi[i],
                theta=theta[i], error=err[i], weight=weight[i], **tags)


    def measure_halo_bin(self, bg_cat, ran_cat, halo_bin, random_random, random_bg):
        import treecorr
        halo_tomo_file = self.get_input("cluster_mag_halo_tomography")
        patch_centers = self.get_input("cluster_mag_patches")


        with self.open_input("cluster_mag_halo_tomography") as f:
            sz = f[f'tomography/bin_{halo_bin}/ra'].size
            metadata = dict(f[f'tomography/bin_{halo_bin}'].attrs)
            print(f"Halo bin {halo_bin} catalog size: {sz}")
            print("Metadata: ", metadata)

        # create foreground catalog using a specific foreground bin
        halo_cat = treecorr.Catalog(
            halo_tomo_file,
            self.config,
            patch_centers=patch_centers,
            ext=f"tomography/bin_{halo_bin}",
            ra_col="ra",
            dec_col="dec",
            ra_units="degrees",
            dec_units="degrees",
        )

        t = time.time()
        print(f"Computing {halo_bin} x {halo_bin}")
        halo_halo = self.measure(halo_cat, halo_cat)

        print(f"Computing {halo_bin} x randoms")
        halo_random = self.measure(halo_cat, ran_cat)

        print(f"Computing {halo_bin} x background")
        halo_bg = self.measure(halo_cat, bg_cat)
        t = time.time() - t
        print(f"Bin {halo_bin} took {t:.1f} seconds")

        # Use these combinations to calculate the correlations functions
        halo_halo.calculateXi(random_random, halo_random)
        halo_bg.calculateXi(random_random, halo_random, random_bg)

        return halo_halo, halo_bg, metadata


    def measure(self, cat1, cat2):
        import treecorr
        # Get any treecorr-related params from our config, while leaving out any that are intended
        # for this code
        config = {
            x: y
            for x, y in self.config.items()
            if x in treecorr.NNCorrelation._valid_params
        }

        p = treecorr.NNCorrelation(**config)
        p.process(cat1, cat2, low_mem=False)
        return p
