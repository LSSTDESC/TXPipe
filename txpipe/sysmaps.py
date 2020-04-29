from .base_stage import PipelineStage
from .data_types import MetacalCatalog, TomographyCatalog, DiagnosticMaps, HDFFile, PNGFile, YamlFile
import numpy as np
from .utils.theory import theory_3x2pt
from .utils import dilated_healpix_map

SHEAR_SHEAR = 0
SHEAR_POS = 1
POS_POS = 2


class TXDiagnosticMaps(PipelineStage):
    """
    For now, this Pipeline Stage computes a depth map using the DR1 method,
    which takes the mean magnitude of objects close to 5-sigma S/N.

    In the future we will add the calculation of other diagnostic maps
    like airmass for use in systematics tests and covariance mode projection.

    DM may in the future provide tools we can use in place of the methods
    used here, but not on the DC2 timescale.


    """
    name='TXDiagnosticMaps'

    # We currently take everything from the shear catalog.
    # In the long run this may become DM output
    inputs = [
        ('photometry_catalog', HDFFile),
        ('shear_catalog', HDFFile),
        ('tomography_catalog', TomographyCatalog),
    ]

    # We generate a single HDF file in this stage
    # containing all the maps
    outputs = [
        ('diagnostic_maps', DiagnosticMaps),
        ('tracer_metadata', HDFFile),
    ]

    # Configuration information for this stage
    config_options = {
        'pixelization': 'healpix', # The pixelization scheme to use, currently just healpix
        'nside':0,   # The Healpix resolution parameter for the generated maps. Only req'd if using healpix
        'snr_threshold':float,  # The S/N value to generate maps for (e.g. 5 for 5-sigma depth)
        'snr_delta':1.0,  # The range threshold +/- delta is used for finding objects at the boundary
        'chunk_rows':100000,  # The number of rows to read in each chunk of data at a time
        'sparse':True,   # Whether to generate sparse maps - faster and less memory for small sky areas,
        'ra_cent':np.nan,  # These parameters are only required if pixelization==tan
        'dec_cent':np.nan,
        'npix_x':-1,
        'npix_y':-1,
        'pixel_size':np.nan, # Pixel size of pixelization scheme
        'depth_band' : 'i',
        'true_shear' : False,
        'flag_exponent_max': 8,
        'dilate': True,
    }


    def run(self):
        """
        Run the analysis for this stage.

         - choose the pixelization scheme for the map
         - loop through chunks of the photometry catalog (in paralllel if enabled)
         - build up the map gradually
         - the master process saves the map
        """
        from .mapping import DepthMapperDR1, Mapper, FlagMapper
        from .utils import choose_pixelization

        # Read input configuration informatiomn
        config = self.config


        # Select a pixelization scheme based in configuration keys.
        # Looks at "pixelization as the main thing"
        pixel_scheme = choose_pixelization(**config)
        config.update(pixel_scheme.metadata)

        chunk_rows = config['chunk_rows']
        band = config['depth_band']

        # These are the columns we're going to need from the various files
        phot_cols = ['ra', 'dec', f'snr_{band}', f'{band}_mag']

        if config['true_shear']:
            shear_cols = ['true_g']
        else:
            shear_cols = ['mcal_g1', 'mcal_g2', 'mcal_psf_g1', 'mcal_psf_g2']
        shear_cols += ['mcal_flags', 'weight']
        bin_cols = ['source_bin', 'lens_bin']
        m_cols = ['R_gamma']

        T = self.open_input('tomography_catalog')
        d = dict(T['/tomography'].attrs)
        T.close()

        source_bins = list(range(d['nbin_source']))
        lens_bins = list(range(d['nbin_lens']))


        # Make three mapper classes, one for the signal itself
        # (shear and galaxy count), another for the depth
        # calculation, and a third one for the PSF
        mapper = Mapper(pixel_scheme, lens_bins, source_bins, sparse=config['sparse'])
        mapper_psf = Mapper(pixel_scheme, lens_bins, source_bins, sparse=config['sparse'])
        depth_mapper = DepthMapperDR1(pixel_scheme,
                                      config['snr_threshold'],
                                      config['snr_delta'],
                                      sparse = config['sparse'],
                                      comm = self.comm)
        flag_mapper = FlagMapper(pixel_scheme, config['flag_exponent_max'], sparse=config['sparse'])


        # Build some "iterators".  Iterators are things you can loop through,
        # but the good thing here is that they don't get created all at once
        # (which here would mean loading in a great number of data columns)
        # but instead are "lazy" - at this point all that happens is that
        # we prepare to read these columns from these different files.
        # These methods by default yield trios of (start, end, data),
        # but in this case because we are agregating we don't need the "start" and
        # "end" numbers.  So we re-define to ignore them
        shear_it = self.iterate_hdf('shear_catalog', 'metacal', shear_cols, chunk_rows)

        phot_it = self.iterate_hdf('photometry_catalog', 'photometry', phot_cols, chunk_rows)
        phot_it = (d[2] for d in phot_it)

        bin_it = self.iterate_hdf('tomography_catalog','tomography', bin_cols, chunk_rows)
        bin_it = (d[2] for d in bin_it)

        m_it = self.iterate_hdf('tomography_catalog','multiplicative_bias', m_cols, chunk_rows)
        m_it = (d[2] for d in m_it)

        # Now, we actually start loading the data in.
        # This thing below just loops through all the files at once
        iterator = zip(shear_it, phot_it, bin_it, m_it)
        for (s,e,shear_data), phot_data, bin_data, m_data in iterator:
            print(f"Process {self.rank} read data chunk {s:,} - {e:,}")
            # Pick out a few relevant columns from the different
            # files to give to the depth mapper.
            depth_data = {
                'mag': phot_data[f'{band}_mag'],
                'snr': phot_data[f'snr_{band}'],
                'bins': bin_data['lens_bin'],
                'ra': phot_data['ra'],
                'dec': phot_data['dec'],
            }


            # Get either the true shears or the measured ones,
            # depending on options
            if config['true_shear']:
                shear_tmp = {'g1': shear_data['true_g1'], 'g2': shear_data['true_g2']}
            else:
                shear_tmp = {'g1': shear_data['mcal_g1'], 'g2': shear_data['mcal_g2']}
                
            # In either case we need the PSF g1 and g2 to map as well
            shear_psf_tmp = {'g1': shear_data['mcal_psf_g1'], 'g2': shear_data['mcal_psf_g2']}

            shear_tmp['ra'] = phot_data['ra']
            shear_tmp['dec'] = phot_data['dec']
            shear_tmp['weight'] = shear_data['weight']

            # Should we use weights in the PSF mapping as well?
            # Yes: the point of these maps is as a diagnostic to compare
            # with shear maps, and the weighting should be the same.
            shear_psf_tmp['ra'] = phot_data['ra']
            shear_psf_tmp['dec'] = phot_data['dec']
            shear_psf_tmp['weight'] = shear_data['weight']

            # And add these data chunks to our maps
            depth_mapper.add_data(depth_data)
            mapper.add_data(shear_tmp, bin_data, m_data)
            mapper_psf.add_data(shear_psf_tmp, bin_data, m_data) # Same?
            flag_mapper.add_data(phot_data['ra'], phot_data['dec'], shear_data['mcal_flags'])

        # Collect together the results across all the processors
        # and combine them to get the final results
        if self.rank==0:
            print("Finalizing maps")
        depth_pix, depth_count, depth, depth_var = depth_mapper.finalize(self.comm)
        map_pix, ngals, g1, g2, var_g1, var_g2, weights_g = mapper.finalize(self.comm)
        map_pix_psf, ngals_psf, g1_psf, g2_psf, var_g1_psf, var_g2_psf, _ = mapper_psf.finalize(self.comm)
        flag_pixs, flag_maps = flag_mapper.finalize(self.comm)

        # Only the root process saves the output
        if self.rank==0:
            print("Saving maps")
            # Open the HDF5 output file
            outfile = self.open_output('diagnostic_maps')
            # Use one global section for all the maps
            group = outfile.create_group("maps")
            # Save each of the maps in a separate subsection
            self.save_map(group, "depth", depth_pix, depth, config)
            self.save_map(group, "depth_count", depth_pix, depth_count, config)
            self.save_map(group, "depth_var", depth_pix, depth_var, config)

            # I'm expecting this will one day call off to a 10,000 line
            # library or something.
            mask_pix, mask = self.compute_mask(pixel_scheme, depth_pix, depth_count)
            self.save_map(group, "mask", mask_pix, mask, config)
            npix = len(mask_pix)


            # Do a very simple centroid calculation.
            # This is not robust, and will not cope with
            # maps that corss
            ra, dec = pixel_scheme.pix2ang(mask_pix, radians=False, theta=False)
            ra_centroid = ra.mean()
            dec_centroid = dec.mean()

            # Save some other handy map info that will be useful later
            area = pixel_scheme.pixel_area(degrees=True) * npix
            group.attrs['area'] = area
            group.attrs['area_unit'] = 'sq deg'
            group.attrs['nbin_source'] = len(source_bins)
            group.attrs['nbin_lens'] = len(lens_bins)
            group.attrs['flag_exponent_max'] = config['flag_exponent_max']
            group.attrs['centroid_ra'] = ra_centroid
            group.attrs['centroid_dec'] = dec_centroid

            # Now save all the lens bin galaxy counts, under the
            # name ngal
            for b in lens_bins:
                self.save_map(group, f"ngal_{b}", map_pix, ngals[b], config)
                self.save_map(group, f"psf_ngal_{b}", map_pix_psf, ngals_psf[b], config)

            for b in source_bins:
                self.save_map(group, f"g1_{b}", map_pix, g1[b], config)
                self.save_map(group, f"g2_{b}", map_pix, g2[b], config)
                self.save_map(group, f"var_g1_{b}", map_pix, var_g1[b], config)
                self.save_map(group, f"var_g2_{b}", map_pix, var_g2[b], config)
                self.save_map(group, f"lensing_weight_{b}", map_pix, weights_g[b], config)
                # PSF maps
                self.save_map(group, f"psf_g1_{b}", map_pix_psf, g1_psf[b], config)
                self.save_map(group, f"psf_g2_{b}", map_pix_psf, g2_psf[b], config)
                self.save_map(group, f"psf_var_g1_{b}", map_pix_psf, var_g1_psf[b], config)
                self.save_map(group, f"psf_var_g2_{b}", map_pix_psf, var_g2_psf[b], config)

            for i,(p, m) in enumerate(zip(flag_pixs, flag_maps)):
                f = 2**i
                t = m.sum()
                print(f"Map shows total {t} objects with flag {f}")
                self.save_map(group, f"flag_{f}", p, m, config)

            self.save_metadata_file(area)


    def compute_mask(self, pixel_scheme, index, count):
        import healpy

        # the index and count may be either partial or
        # full sky.  Make a full-sky map, either way,
        # with UNSEEN where the pixel has no objects
        mask = np.repeat(healpy.UNSEEN, pixel_scheme.npix)
        mask[index] = count

        # Compress down to UNSEEN/1
        mask[mask <= 0] = healpy.UNSEEN
        mask[mask > 0] = 1


        # optionally expand the UNSEENs by one pixel
        if self.config['dilate']:
            print("Dilating mask")
            mask = dilated_healpix_map(mask)

        # Pull out observed pixels.
        # This doesn't make that much sense here, because our
        # mask is just 0/1 so the mask we output is 1 everywhere.
        # But later our mask will have non-binary values.
        mask_pix = np.where(mask>0)
        mask = mask[mask_pix]
        print(mask_pix)
        print(mask.max())
        return mask_pix, mask



    def save_map(self, group, name, pixel, value, metadata):
        """
        Save an output map to an HDF5 subgroup.

        The pixel numbering and the metadata are also saved.

        Parameters
        ----------

        group: H5Group
            The h5py Group object in which to store maps
        name: str
            The name of this map, used as the name of a subgroup in the group where the data is stored.
        pixel: array
            Array of indices of observed pixels
        value: array
            Array of values of observed pixels
        metadata: mapping
            Dict or other mapping of metadata to store along with the map
        """
        subgroup = group.create_group(name)
        subgroup.attrs.update(metadata)
        subgroup.create_dataset("pixel", data=pixel)
        subgroup.create_dataset("value", data=value)

    def save_metadata_file(self, area):
        area_sq_arcmin = area * 60**2
        tomo_file = self.open_input('tomography_catalog')
        meta_file = self.open_output('tracer_metadata')
        def copy(in_section, out_section, name):
            x = tomo_file[f'{in_section}/{name}'][:]
            meta_file.create_dataset(f'{out_section}/{name}', data=x)

        def copy_attrs(name, out_name):
            for k,v in tomo_file[name].attrs.items():
                meta_file[out_name].attrs[k] = v


        copy('multiplicative_bias', 'tracers', 'R_gamma_mean')
        copy('multiplicative_bias', 'tracers', 'R_S')
        copy('multiplicative_bias', 'tracers', 'R_total')
        copy('tomography', 'tracers', 'N_eff')
        copy('tomography', 'tracers', 'lens_counts')
        copy('tomography', 'tracers', 'sigma_e')
        copy('tomography', 'tracers', 'source_counts')
        N_eff = tomo_file['tomography/N_eff'][:]
        n_eff = N_eff / area_sq_arcmin
        lens_counts = tomo_file['tomography/lens_counts'][:]
        source_counts = tomo_file['tomography/source_counts'][:]
        lens_density = lens_counts / area_sq_arcmin
        source_density = source_counts / area_sq_arcmin
        meta_file.create_dataset('tracers/n_eff', data=n_eff)
        meta_file.create_dataset('tracers/lens_density', data=lens_density)
        meta_file.create_dataset('tracers/source_density', data=source_density)
        meta_file['tracers'].attrs['area'] = area
        meta_file['tracers'].attrs['area_unit'] = 'sq deg'
        copy_attrs('tomography', 'tracers')

        meta_file.close()


class FakeTracer:
    def __init__(self, mu, sigma):
        self.z = np.arange(0.0, 3.0, 0.01)
        self.nz = np.exp(-0.5*(self.z - mu)**2 / sigma**2) / np.sqrt(2*np.pi) / sigma
        self.nsample = len(self.z)

class TXFakeMaps(TXDiagnosticMaps):
    """
    Generate fake maps using healpy
    """
    name='TXFakeMaps'

    # We have no inputs.  Everything is fake.
    inputs = [('fiducial_cosmology', YamlFile),]

    # Same outputs as the real map
    outputs = [
        ('diagnostic_maps', DiagnosticMaps),
        ('tracer_metadata', HDFFile),
        ('photoz_stack', HDFFile),
    ]

    # Configuration information for this stage
    config_options = {
        'pixelization': 'healpix', # The pixelization scheme to use, currently just healpix
        'nside':512,   # The Healpix resolution parameter for the generated maps. Only req'd if using healpix
        'sparse':True,   # Whether to generate sparse maps - faster and less memory for small sky areas,
        'source_bin_centers': [0.5, 0.7, 0.9, 1.1],
        'source_bin_widths': [0.1, 0.1, 0.1, 0.1],
        'sigma_e': [0.2, 0.2, 0.2, 0.2],
        'n_eff': [10.0, 10.0, 10.0, 10.0],
        'lens_bin_centers': [0.35, 0.55],
        'lens_bin_widths': [0.1, 0.1],
        'lens_counts':[10.0, 10.0],
        'dec_min': -70.,
        'dec_max': -10.,
        'ra_min': 0.0,
        'ra_max': 90.,
        'flag_exponent_max': 8,
    }


    def run(self):
        """
        Run the analysis for this stage.

         - choose the pixelization scheme for the map
         - loop through chunks of the photometry catalog (in paralllel if enabled)
         - build up the map gradually
         - the master process saves the map
        """
        from .utils import choose_pixelization

        # Read input configuration informatiomn
        config = self.config

        # Select a pixelization scheme based in configuration keys.
        # Looks at "pixelization as the main thing"
        if config['pixelization']!= 'healpix':
            raise ValueError("Only writing faker for healpix")
        pixel_scheme = choose_pixelization(**config)
        config.update(pixel_scheme.metadata)



        nbin_source = len(config['source_bin_centers'])
        nbin_lens = len(config['lens_bin_centers'])
        source_bins = list(range(nbin_source))
        lens_bins = list(range(nbin_lens))
        npix_full = config['npix']

        # We cut down to a simple rectangular region for testing.
        pix_full = np.arange(config['npix'])
        ra, dec = pixel_scheme.pix2ang(pix_full)
        region = (
              (dec > config['dec_min']) 
            & (dec < config['dec_max'])
            & (ra > config['ra_min'])
            & (ra < config['ra_max'])
        )

        pix = pix_full[region]
        npix = len(pix)

        
        depth_pix = pix
        depth = np.repeat(25.0, npix)
        depth_count = np.repeat(30, npix)
        depth_var = np.repeat(0.05**2, npix)


        map_pix = pix
        source_maps, lens_maps = self.simulate_gaussian_maps(nbin_source, nbin_lens)
        g1 = [s[0][pix] for s in source_maps]
        g2 = [s[1][pix] for s in source_maps]
        ngals = [m[pix] for m in lens_maps]
        var_g1 = [np.zeros_like(g) for g in g1]
        var_g2 = [np.zeros_like(g) for g in g2]

        map_pix_psf = pix
        ngals_psf = np.repeat(10.0, npix)
        zero = np.repeat(0.0, npix)
        g1_psf = [zero for i in range(nbin_source)]
        g2_psf = [zero for i in range(nbin_source)]
        var_g1_psf = [zero for i in range(nbin_source)]
        var_g2_psf = [zero for i in range(nbin_source)]

        flag_pixs = pix
        flag_maps = [zero for i in range(config['flag_exponent_max'])]

        print("Saving maps")
        # Open the HDF5 output file
        outfile = self.open_output('diagnostic_maps')
        # Use one global section for all the maps
        group = outfile.create_group("maps")
        # Save each of the maps in a separate subsection
        self.save_map(group, "depth", depth_pix, depth, config)
        self.save_map(group, "depth_count", depth_pix, depth_count, config)
        self.save_map(group, "depth_var", depth_pix, depth_var, config)

        # I'm expecting this will one day call off to a 10,000 line
        # library or something.
        mask_pix, mask = self.compute_mask(pixel_scheme, depth_pix, depth_count)
        npix = len(mask_pix)
        self.save_map(group, "mask", mask_pix, mask, config)

        # Save some other handy map info that will be useful later
        area = pixel_scheme.pixel_area(degrees=True) * npix
        group.attrs['area'] = area
        group.attrs['area_unit'] = 'sq deg'
        group.attrs['nbin_source'] = len(source_bins)
        group.attrs['nbin_lens'] = len(lens_bins)
        group.attrs['flag_exponent_max'] = config['flag_exponent_max']

        # Now save all the lens bin galaxy counts, under the
        # name ngal
        for b in lens_bins:
            self.save_map(group, f"ngal_{b}", map_pix, ngals[b], config)
            self.save_map(group, f"psf_ngal_{b}", map_pix_psf, ngals_psf[b], config)

        for b in source_bins:
            self.save_map(group, f"g1_{b}", map_pix, g1[b], config)
            self.save_map(group, f"g2_{b}", map_pix, g2[b], config)
            self.save_map(group, f"var_g1_{b}", map_pix, var_g1[b], config)
            self.save_map(group, f"var_g2_{b}", map_pix, var_g2[b], config)
            # PSF maps
            self.save_map(group, f"psf_g1_{b}", map_pix_psf, g1_psf[b], config)
            self.save_map(group, f"psf_g2_{b}", map_pix_psf, g2_psf[b], config)
            self.save_map(group, f"psf_var_g1_{b}", map_pix_psf, var_g1_psf[b], config)
            self.save_map(group, f"psf_var_g2_{b}", map_pix_psf, var_g2_psf[b], config)

        for i,(p, m) in enumerate(zip(flag_pixs, flag_maps)):
            f = 2**i
            t = m.sum()
            print(f"Map shows total {t} objects with flag {f}")
            self.save_map(group, f"flag_{f}", p, m, config)

            self.save_metadata_file(area, nbin_source, nbin_lens)
            self.save_photoz_stack(nbin_source, nbin_lens)

    def generate_tracers(self, nbin_source, nbin_lens):
        tracers = {}
        for i in range(nbin_source):
            mu = self.config['source_bin_centers'][i]
            sigma = self.config['source_bin_widths'][i]
            tracers[f'source_{i}'] = FakeTracer(mu, sigma)

        for i in range(nbin_lens):
            mu = self.config['lens_bin_centers'][i]
            sigma = self.config['lens_bin_widths'][i]
            tracers[f'lens_{i}'] = FakeTracer(mu, sigma)
        return tracers

    def save_photoz_stack(self, nbin_source, nbin_lens):
        tracers = self.generate_tracers(nbin_source, nbin_lens)

        outfile = self.open_output('photoz_stack')

        for name, n in [('source', nbin_source), ('lens', nbin_lens)]:
            group = outfile.create_group(f"n_of_z/{name}")
            group.attrs["nbin"] = n
        
            for i in range(n):
                tracer = tracers[f'source_{i}']
                group.create_dataset(f"bin_{i}", data=tracer.nz)

                # TODO: make and save counts for each bin
                # group.attrs["count_{}"] = ...

                if i==0:
                    group.create_dataset("z", data=tracer.z)
                    group.attrs["nz"] = len(tracer.z)
        outfile.close()



    def generate_theory_cl(self, nbin_source, nbin_lens):
        import scipy.interpolate
        tracers = self.generate_tracers(nbin_source, nbin_lens)
        cosmo_file = self.get_input("fiducial_cosmology")
        theory_cl = theory_3x2pt(cosmo_file, tracers, nbin_source, nbin_lens)

        # This gives us theory with log-spaced ell values, but healpy
        # wants all the ell values.  So we interpolate.
        ell_sample = theory_cl['ell']
        ell_max = min(3*self.config['nside'] - 1, ell_sample[-1])
        ell_grid = np.arange(1, ell_max+1)
        theory_full_grid = {'ell': ell_grid}
        for key, val in theory_cl.items():
            if key == 'ell':
                new_val = ell_grid
            else:
                s = scipy.interpolate.InterpolatedUnivariateSpline(ell_sample, val)
                new_val = s(ell_grid)

            theory_full_grid[key] = new_val

        return ell_max, theory_full_grid

    def simulate_alm(self, theory_cl, nbin_source, nbin_lens, ell_max):
        import healpy as hp
        # healpy wants a specific ordering for C_ell, which we re-create here
        # first, define a function mapping from the healpy ordering
        # to the dict keys we use in theory_cl
        def pair_to_key(i, j):
            i_is_source = i < nbin_source
            j_is_source = j < nbin_source
            i1 = i if i_is_source else i - nbin_source
            j1 = j if j_is_source else j - nbin_source
            if i_is_source and j_is_source:
                k = SHEAR_SHEAR
            elif (not i_is_source) and (not j_is_source):
                k = POS_POS
            else:
                k = SHEAR_POS
            return (i1, j1, k)
        
        # Now put things into this new order
        ntot = nbin_source + nbin_lens
        cl = []
        for i in range(ntot):
            for j in range(ntot):
                if j<i:
                    continue
                key = pair_to_key(i,j)
                cl.append(theory_cl[key])

        # And finally simulate alm. new=False refers to the fact
        # that healpy is going to change the ordering scheme at some point
        alm = hp.synalm(cl, new=False, lmax=ell_max)
        return alm

    def simulate_gaussian_maps(self, nbin_source, nbin_lens):
        import healpy as hp
        ell_max, theory_cl = self.generate_theory_cl(nbin_source, nbin_lens)
        alm = self.simulate_alm(theory_cl, nbin_source, nbin_lens, ell_max)

        # For the sources we generate g1 and g2.
        source_maps = []
        for i in range(nbin_source):
            # assume E-mode only
            ee = alm[i]
            bb = np.zeros_like(ee)
            g1, g2 = hp.alm2map_spin([ee,bb], self.config['nside'], spin=2, lmax=ell_max)
            source_maps.append((g1,g2))

        # For lens maps we just generate the density.
        # This actually makes maps that go negative, because
        # we are pretending everything is Gaussian.
        lens_maps = []
        for i in range(nbin_lens):
            cl = alm[nbin_source + i]
            m = hp.alm2map(cl, nside=self.config['nside'], lmax=ell_max)
            n_mean = self.config['lens_counts'][i]
            lens_maps.append((1+m)*n_mean)

        self.add_noise(source_maps, lens_maps)

        return source_maps, lens_maps

    def add_noise(self, source_maps, lens_maps):
        import healpy as hp
        pix_area_deg2 = hp.nside2pixarea(self.config['nside'], degrees=True)
        pix_area_arcmin2 = pix_area_deg2 * 3600

        for i, (g1, g2) in enumerate(source_maps):
            sigma_e = self.config['sigma_e'][i]
            n_eff_arcmin2 = self.config['n_eff'][i]
            n_eff_pixel = n_eff_arcmin2 * pix_area_arcmin2
            sigma_pixel = sigma_e / np.sqrt(n_eff_pixel)
            
            g1 += np.random.normal(size=g1.size) * sigma_pixel
            g2 += np.random.normal(size=g2.size) * sigma_pixel


        for i, m in enumerate(lens_maps):
            n_arcmin2 = self.config['lens_counts'][i]
            n_pixel = n_arcmin2 / pix_area_arcmin2
            sigma_pixel = 1.0 / np.sqrt(n_eff_pixel)
            print(i, sigma_pixel)   
            
            m += np.random.normal(size=m.size) * sigma_pixel




    def save_metadata_file(self, area, nbin_source, nbin_lens):
        area_sq_arcmin = area * 60**2
        meta_file = self.open_output('tracer_metadata')
        group = meta_file.create_group('tracers')
        group.create_dataset('R_gamma_mean', data = np.zeros((nbin_source, 2, 2)))
        group.create_dataset('R_S', data = np.zeros((nbin_source, 2, 2)))
        group.create_dataset('R_total', data = np.zeros((nbin_source, 2, 2)))

        n_eff = np.array(self.config['n_eff'])
        N_eff = n_eff * area_sq_arcmin
        lens_density = np.array(self.config['lens_counts'])
        lens_counts = lens_density * area_sq_arcmin

        # we make the same outputs that are in the standard mapper.
        group.create_dataset('N_eff', data = N_eff)
        group.create_dataset('lens_counts', data=lens_counts)
        group.create_dataset('sigma_e', data=self.config['sigma_e'])
        group.create_dataset('source_counts', data=np.zeros(nbin_source))

        group.create_dataset('n_eff', data=n_eff)
        group.create_dataset('lens_density', data=lens_density)
        group.create_dataset('source_density', data=np.zeros(nbin_source))


        group.attrs['area'] = area
        group.attrs['area_unit'] = 'sq deg'
        group.attrs['nbin_source'] = nbin_source
        group.attrs['nbin_lens'] = nbin_lens

        meta_file.close()



class TXMapPlots(PipelineStage):
    """
    """
    name='TXMapPlots'

    inputs = [
        ('diagnostic_maps', DiagnosticMaps),
    ]
    outputs = [
        ('depth_map', PNGFile),
        ('ngal_lens_map', PNGFile),
        ('g1_map', PNGFile),
        ('g2_map', PNGFile),
        ('flag_map', PNGFile),
        ('mask_map', PNGFile),
    ]
    config = {}

    def run(self):
        # PSF tests
        import matplotlib
        matplotlib.use('agg')
        import matplotlib.pyplot as plt
        m = self.open_input("diagnostic_maps", wrapper=True)

        fig = self.open_output('depth_map', wrapper=True, figsize=(5,5))
        m.plot('depth', view='cart')
        fig.close()

        nbin_source, nbin_lens = m.get_nbins()
        flag_exponent_max = m.file['maps'].attrs['flag_exponent_max']

        fig = self.open_output('ngal_lens_map', wrapper=True, figsize=(5*nbin_lens, 5))
        for i in range(nbin_lens):
            plt.subplot(1, nbin_lens, i+1)
            m.plot(f'ngal_{i}', view='cart')
        fig.close()

        fig = self.open_output('g1_map', wrapper=True, figsize=(5*nbin_source, 5))
        for i in range(nbin_source):
            plt.subplot(1, nbin_source, i+1)
            m.plot(f'g1_{i}', view='cart')
        fig.close()

        fig = self.open_output('g2_map', wrapper=True, figsize=(5*nbin_source, 5))
        for i in range(nbin_source):
            plt.subplot(1, nbin_source, i+1)
            m.plot(f'g2_{i}', view='cart')
        fig.close()

        fig = self.open_output('flag_map', wrapper=True, figsize=(5*flag_exponent_max, 5))
        for i in range(flag_exponent_max):
            plt.subplot(1, flag_exponent_max, i+1)
            f = 2**i
            m.plot(f'flag_{f}', view='cart')
        fig.close()

        fig = self.open_output('mask_map', wrapper=True, figsize=(5,5))
        m.plot('mask', view='cart')
        fig.close()




if __name__ == '__main__':
    PipelineStage.main()

