from .base_stage import PipelineStage
from .data_types import HDFFile, MapsFile, TextFile
import numpy as np
import GCRCatalogs


class CLIngestHalosCosmoDC2(PipelineStage):
    name = "CLIngestHalosCosmoDC2"
    parallel = False
    inputs = []
    outputs = [("cluster_mag_halo_catalog", HDFFile)]
    config_options = {
        "cat_name": "cosmoDC2_v1.1.4_image",
        "halo_mass_min": 0.5e13,
        "initial_size": 100_000,
    }

    def run(self):
        # Configuration options
        mass_min = self.config['halo_mass_min']
        cat_name = self.config['cat_name']
        sz = self.config['initial_size']

        # Open the cosmoDC2 catalog
        overwrite = {'check_md5': False, 'check_size': False, 'ensure_meta_consistent': False}
        cat = GCRCatalogs.load_catalog(cat_name, config_overwrite=overwrite)
        
        # Selection of data we will read from it below
        cols = ['halo_mass', 'redshift','ra', 'dec', 'halo_id']
        filters = [f'halo_mass > {mass_min}','is_central==True']

        # Create output data file with extensible data sets 
        f = self.open_output("cluster_mag_halo_catalog")
        g = f.create_group("halos")
        g.create_dataset('halo_mass', (sz,), maxshape=(None,), dtype='f8', chunks=True)
        g.create_dataset('redshift', (sz,), maxshape=(None,), dtype='f8', chunks=True)
        g.create_dataset('ra', (sz,), maxshape=(None,), dtype='f8', chunks=True)
        g.create_dataset('dec', (sz,), maxshape=(None,), dtype='f8', chunks=True)
        g.create_dataset('halo_id', (sz,), maxshape=(None,), dtype='i8', chunks=True)

        # Prepare the iterator to loop through GCR
        it = cat.get_quantities(cols, filters=filters, return_iterator=True)

        # s is the start index for the next data chunk
        s = 0
        for data in it:
            # e is the end index for this data chunk
            e = s + data['ra'].size
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




class CLMagnificationBackgroundSelector(PipelineStage):
    parallel = False
    name = "CLMagnificationBackgroundSelector"
    inputs = [("photometry_catalog", HDFFile)]
    outputs = [("cluster_mag_background", HDFFile), ("cluster_mag_footprint", MapsFile)]
    config_options = {
        "ra_range": [50.0, 73.1],
        "dec_range": [-45.0, -27.0],
        "mag_cut": 1.5.
        "zmin": 1.5.
        "nside": 2048,
        "initial_size": 100_000
    }

    def run(self):

        # Count the max number of objects we will look at
        with self.open_input("photometry_catalog") as f:
            N = f['photometry/ra'].size

        # Open and set up the columns in the output
        f = self.open_output("cluster_mag_background", "w", parallel=True)
        g = f.create_group("sample")
        sz = self.config['initial_size']
        ra = g.create_dataset("ra", (sz,), maxshape=(None,))
        dec = g.create_dataset("dec", (sz,), maxshape=(None,))

        # Extract inputs from the user configutation
        ra_min, ra_max = self.config['ra_range']
        dec_min, dec_max = self.config['dec_range']
        mag_cut = self.config['mag_cut']
        zmin = self.config['zmin']
        nside = self.config['nside']

        npix = healpy.nside2npix(nside)
        hit_map = np.zeros(npix)

        # Prepare an iterator
        it = self.iterate_hdf5("photometry_catalog", "photometry", ["ra", "dec", "mag_i", "redshift_true"])

        s = 0
        # Loop through the data
        for _, _, data in it:
            # make selection
            sel = (
                    (data['ra'] > ra_min)
                    & (data['ra'] < ra_max)
                    & (data['dec'] > dec_min) 
                    & (data['dec'] < dec_max)
                    & (data['mag_i'] < mag_cut)
                    & (data['redshift_true'] > 1.5)
                )

            ra_sel = data['ra'][sel]
            dec_sel = data['dec'][sel]
            pix = healpy.ang2pix(nside, ra_sel, dec_sel, lonlat=True)
            hit_map[pix] = 1

            e = s + ra_sel.size
            if e > sz:
                sz = int(1.5 * e)
                ra.resize((sz,))
                dec.resize((sz,))

            # write output.
            ra[s:e] = ra_sel
            dec[s:e] = dec_sel

            # update start for next point
            s = e

        f.close()

        # Collate the overall footprint from all the processors
        if self.comm is not None:
            hit_map = self.comm.Reduce(hit_map).clip(0, 1)

        # Save the footprint on one processor
        if self.rank == 0:
            pix = np.where(hit_map > 0)[0]
            val = np.ones(footprint_pix.size, dtype=np.int8)

            f = self.open_output("cluster_mag_footprint", wrapper=True)
                f.write_map("footprint", pix, val, self.config)





class CLMagnificationRandoms(PipelineStage):
    name = "CLMagnificationRandoms"
    inputs = [("cluster_mag_halo_catalog", HDFFile), ("cluster_mag_footprint", MapsFile)]
    outputs = [("cluster_mag_randoms", HDFFile)]
    config_options = {
        "density": 30 # per sq arcmin
    }
    def run(self):

        # open the footprint catalog and read in the things we need
        # from it - the list of hit pixels, and the map scheme
        with self.open_input("cluster_mag_footprint") as f:
            hit_map = f.read_map("footprint")
            info = f.read_map_info("footprint")
            nside = info['nside']
            scheme = choose_pixelization(**info)

        # Hit pixels and the coordinates of their vertices
        pix = np.where(hit_map > 0)[0]
        vertices = scheme.vertices(pix)

        # Randomly select the number of objects per pixel, using a poisson distribution with
        # the specified mean density, for now
        area = scheme.pixel_area(degrees=True) * 60. * 60.
        density = np.repeat(self.config['density'], pix.size)
        counts = scipy.stats.poisson.rvs(density*area, 1)

        # Use the same counts for all the processors
        if self.comm is not None:
            counts = self.comm.bcast(counts)

        # total number of objects to be generated over all the pixels
        total_count = counts.sum()

        # The starting index of each pixel in the array, so we can parallelize
        starts = np.concatenate([0, np.cumsum(counts)])

        # Open the output data and create the necessary columns
        output_file = self.open_output('cluster_mag_randoms')
        group = output_file.create_group('randoms')
        ra_out = group.create_dataset('ra', (total_count,), dtype=np.float64)
        dec_out = group.create_dataset('dec', (total_count,), dtype=np.float64)


        # Each processor now does a subset of the pixels, generating and saving
        # points in each
        for i, vertex in self.split_tasks_by_rank(enumerate(vertices)):
            # Generate random points in this pixel
            p1, p2, p3, p4 = vertex.T
            P = randoms.random_points_in_quadrilateral(p1, p2, p3, p4, N)

            # This is not healpy-specific so we just use it as a convenience function
            ra, dec = healpy.vec2ang(P, lonlat=True)

            # Write output
            s = starts[i]
            e = starts[i + 1]
            ra_out[s:e] = ra
            dec_out[s:e] = dec

        output_file.close()

class CLMagnificationPatches(PipelineStage):
    """

    This is currently copied from the in-progress treecorr-mpi branch.
    Think later how to 
    """
    inputs = [("cluster_mag_randoms", HDFFile)]
    outputs = [("cluster_mag_patches", TextFile)]
    config_options = {
        'npatch' : 32,
        'every_nth': 100,
    }

    def run(self):
        import treecorr
        import matplotlib
        matplotlib.use('agg')

        input_filename = self.get_input('random_cats')
        output_filename = self.get_output('patch_centers')

        # Build config info
        npatch = self.config['npatch']
        every_nth = self.config['every_nth']
        config = {
            'ext': 'randoms',
            'ra_col': 'ra',
            'dec_col': 'dec',
            'ra_units': 'degree',
            'dec_units': 'degree',
            'every_nth': every_nth,
            'npatch': npatch,
        }

        #Create the catalog
        cat = treecorr.Catalog(input_filename, config)

        # Generate and write the output patch centres
        print(f"generating {npatch} centers")
        cat.write_patch_centers(output_filename)


class CLMagnificationCorrelations(PipelineStage):
    name = "CLMagnificationCorrelations"
    inputs = [
        ("cluster_mag_halo_catalog", HDFFile),
        ("cluster_mag_background", HDFFile),
        ("cluster_mag_patches", TextFile),
    ]
    outputs = []
    config_options = {
        'min_sep':0.5,
        'max_sep':300.,
        'nbins':9,
        'bin_slop':0.1,
        'sep_units':'arcmin',
        'flip_g2':True,
        'cores_per_task':32,
        'verbose':1,
        'var_method': 'jackknife',
        }

    def run(self):
        import treecorr

        halo_file = self.get_input("cluster_mag_halo_catalog")
        source_file = self.get_input("cluster_mag_background")
        patch_centers = self.get_input("cluster_mag_patches")

        # create foreground catalog
        halo_cat = treecorr.Catalog(halo_file, self.config, patch_centers=patch_centers)
        source_cat = treecorr.Catalog(source_file, self.config, patch_centers=patch_centers, ext="randoms")
        random_cat = 
        # create background catalog

        # create random catalog

        # run under MPI
    cat_rand_halo =  rand_cat(cat_halo, Nobj = cat_halo.ra.size, patch_centers=patch_centers)
    
    ls = treecorr.NNCorrelation(**bin_dict)
    ls.process(cat_halo, cat)
    
    ll = treecorr.NNCorrelation(**bin_dict)
    ll.process(cat_halo, cat_halo)
    
    lr = treecorr.NNCorrelation(**bin_dict)
    lr.process(cat_halo, rand)
    
    xi, varxi = ls.calculateXi(rr, lr, rd)
    r = np.exp(ls.meanlogr)
    sigxi = np.sqrt(varxi)
    covxi = ls.estimate_cov(bin_dict['var_method'])
    
    ls_rand = treecorr.NNCorrelation(**bin_dict)
    ls_rand.process(cat_rand_halo, cat)
    
    lr_rand = treecorr.NNCorrelation(**bin_dict)
    lr_rand.process(cat_rand_halo, rand)
    
    xi_rand, varxi_rand = ls_rand.calculateXi(rr, lr_rand, rd)
    r_rand = np.exp(ls_rand.meanlogr)
    sigxi_rand = np.sqrt(varxi_rand)
    covxi_rand = ls_rand.estimate_cov(bin_dict['var_method'])



