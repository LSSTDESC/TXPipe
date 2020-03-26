from .base_stage import PipelineStage
from .data_types import MetacalCatalog, TomographyCatalog, RandomsCatalog, YamlFile, SACCFile, DiagnosticMaps, HDFFile, PhotozPDFFile
import numpy as np
import collections
from .utils import choose_pixelization, HealpixScheme, GnomonicPixelScheme, ParallelStatsCalculator
from .utils.theory import theory_3x2pt
import sys
import warnings


# Using the same convention as in twopoint.py
SHEAR_SHEAR = 0
SHEAR_POS = 1
POS_POS = 2



NAMES = {SHEAR_SHEAR:"shear-shear", SHEAR_POS:"shear-position", POS_POS:"position-position"}

Measurement = collections.namedtuple(
    'Measurement',
    ['corr_type', 'l', 'value', 'win', 'i', 'j'])

class TXTwoPointFourier(PipelineStage):
    """This Pipeline Stage computes all auto- and cross-correlations
    for a list of tomographic bins, including all galaxy-galaxy,
    galaxy-shear and shear-shear power spectra. Sources and lenses
    both come from the same shear_catalog and tomography_catalog objects.

    The power spectra are computed after deprojecting a number of
    systematic-dominated modes, represented as input maps.

    In the future we will want to include the following generalizations:
     - TODO: specify which cross-correlations in particular to include
             (e.g. which bin pairs and which source/lens combinations).
     - TODO: include flags for rejected objects. Is this included in
             the tomography_catalog?
     - TODO: ell-binning is currently static.
    """
    name = 'TXTwoPointFourier'
    inputs = [
        ('photoz_stack', HDFFile),  # Photoz stack
        ('diagnostic_maps', DiagnosticMaps),
        ('fiducial_cosmology', YamlFile),  # For the cosmological parameters
        ('tracer_metadata', TomographyCatalog),  # For density info
    ]
    outputs = [
        ('twopoint_data_fourier', SACCFile)
    ]

    config_options = {
        "mask_threshold": 0.0,
        "bandwidth": 0,
        "apodization_size": 3.0,
        "apodization_type": "C1",  #"C1", "C2", or "Smooth"
        "flip_g1": False,
        "flip_g2": False,
    }

    def run(self):
        import pymaster
        import healpy
        import sacc
        import pyccl
        config = self.config

        if self.comm:
            self.comm.Barrier()

        self.setup_results()


        # Generate namaster fields
        pixel_scheme, f_d, f_wl, nbin_source, nbin_lens, f_sky = self.load_maps()
        if self.rank==0:
            print("Loaded maps and converted to NaMaster fields")

        # Get the complete list of calculations to be done,
        # for all the three spectra and all the bin pairs.
        # This will be used for parallelization.
        calcs = self.select_calculations(nbin_source, nbin_lens)


        # Load in the per-bin auto-correlation noise levels and 
        # mean response values
        tomo_info = self.load_tomographic_quantities(nbin_source, nbin_lens, f_sky)


        # Binning scheme, currently chosen from the geometry.
        # TODO: set ell binning from config
        ell_bins = self.choose_ell_bins(pixel_scheme, f_sky)

        # Load the n(z) values, which are both saved in the output
        # file alongside the spectra, and then used to calcualate the
        # fiducial theory C_ell, which is used in the deprojection calculation
        tracers = self.load_tracers(nbin_source, nbin_lens)
        theory_cl = theory_3x2pt(
            self.get_input('fiducial_cosmology'),
            tracers,
            nbin_source, nbin_lens,
            fourier=True)

        # If we are rank zero print out some info
        if self.rank==0:
            nband = ell_bins.get_n_bands()
            ell_effs = ell_bins.get_effective_ells()
            print(f"Chosen {nband} ell bin bands with effective ell values and ranges:")
            for i in range(nband):
                leff = ell_effs[i]
                lmin = ell_bins.get_ell_min(i)
                lmax = ell_bins.get_ell_max(i)
                print(f"    {leff:.0f}    ({lmin:.0f} - {lmax:.0f})")


        # Run the compute power spectra portion in parallel
        # This splits the calculations among the parallel bins
        # It's not the most optimal way of doing it
        # as it's not dynamic, just a round-robin assignment.
        for i, j, k in self.split_tasks_by_rank(calcs):
            self.compute_power_spectra(
                pixel_scheme, i, j, k, f_wl, f_d, ell_bins, tomo_info, theory_cl)

        if self.rank==0:
            print(f"Collecting results together")
        # Pull all the results together to the master process.
        self.collect_results()

        # Write the collect results out to HDF5.
        if self.rank == 0:
            self.save_power_spectra(tracers, nbin_source, nbin_lens)
            print("Saved power spectra")



    def load_maps(self):
        import pymaster as nmt
        import healpy
        # Parameters that we need at this point
        apod_size = self.config['apodization_size']
        apod_type = self.config['apodization_type']


        # Load the various input maps and their metadata
        map_file = self.open_input('diagnostic_maps', wrapper=True)
        pix_info = map_file.read_map_info('mask')
        area = map_file.file['maps'].attrs["area"]

        nbin_source = map_file.file['maps'].attrs['nbin_source']
        nbin_lens = map_file.file['maps'].attrs['nbin_lens']

        # Choose pixelization and read mask and systematics maps
        pixel_scheme = choose_pixelization(**pix_info)

        # Load the mask. It should automatically be the same shape as the
        # others, based on how it was originally generated.
        # We remove any pixels that are at or below our threshold (default=0)
        mask = map_file.read_map('mask')
        mask[np.isnan(mask)] = 0
        mask[mask==healpy.UNSEEN] = 0

        # Using a flat mask as the clustering weight for now, since I need to know
        # how to turn the depth map into a weightm
        clustering_weight = mask

        f_sky = area / 41253.
        if self.rank == 0:
            print(f"Unmasked area = {area}, fsky = {f_sky}")

        # Load all the maps in.
        # TODO: make it possible to just do a subset of these
        ngal_maps = [map_file.read_map(f'ngal_{b}') for b in range(nbin_lens)]
        g1_maps = [map_file.read_map(f'g1_{b}') for b in range(nbin_source)]
        g2_maps = [map_file.read_map(f'g2_{b}') for b in range(nbin_source)]
        lensing_weights = [map_file.read_map(f'lensing_weight_{b}') for b in range(nbin_source)]

        # Mask any pixels which have the healpix bad value
        for (g1, g2, lw) in zip(g1_maps, g2_maps, lensing_weights):
            lw[g1 == healpy.UNSEEN] = 0
            lw[g2 == healpy.UNSEEN] = 0

        # When running on the CosmoDC2 mocks I've found I need to flip
        # both g1 and g2 in order to get both positive galaxy-galaxy lensing
        # and shear-shear spectra.
        if self.config['flip_g1']:
            for g1 in g1_maps:
                w = np.where(g1!=healpy.UNSEEN)
                g1[w]*=-1

        if self.config['flip_g2']:
            for g2 in g2_maps:
                w = np.where(g2!=healpy.UNSEEN)
                g2[w]*=-1

        # TODO: load systematics maps here, once we are making them.
        # maybe from Eli or Zilong's code
        syst_nc = None
        syst_wl = None

        map_file.close()

        # THe gnomonic (tangent plane) mapping versions need the map width
        # I'm thinking of deleting support for this since it adds a whole
        # set of options to maintain.
        if pixel_scheme.name == 'gnomonic':
            lx = np.radians(pixel_scheme.size_x)
            ly = np.radians(pixel_scheme.size_y)

        # For the lensing mask we optionall apodize the mask.
        # TODO: include the masked object fraction in here.
        # TODO: ask about including the depth map in here
        if apod_size > 0:
            if self.rank==0:
                print(f"Apodizing clustering weights map with size {apod_size} deg and method {apod_type}")

            # NaMaster has different functions for apodizing the two different maps
            if pixel_scheme.name == 'gnomonic':
                clustering_weight = nmt.mask_apodization_flat(clustering_weight, lx, ly, apod_size, apotype=apod_type)
            elif pixel_scheme.name == 'healpix':
                clustering_weight = nmt.mask_apodization(clustering_weight, apod_size, apotype=apod_type)
            else:
                raise ValueError(f"Pixelization scheme {pixel_scheme.name} not supported by NaMaster")

        # Set any unseen pixels to zero weight.
        for ng in ngal_maps:
            clustering_weight[ng==healpy.UNSEEN] = 0

        d_maps = []
        for ng in ngal_maps:
            # Convert the number count maps to overdensity maps.
            # First compute the overall mean object count per bin.
            # Maybe we should do this in the mapping code itself?
            # mean clustering galaxies per pixel in this map
            mu = ng[clustering_weight>0].mean()
            # and then use that to convert to overdensity
            dmap = ng.copy()
            dmap[clustering_weight>0] -= mu
            dmap[clustering_weight>0] /= mu
            # and re-masking, just in case
            dmap[~(clustering_weight>0)] = 0
            d_maps.append(dmap)

        # We now convert these maps, which are either healpix arrays or 2x2 maps,
        # to the NaMaster internal type, which also knows about the 
        density_fields = []
        lensing_fields = []

        # Now convert these maps and masks into NaMaster Field objects.
        # First the over-density maps.
        for i,d in enumerate(d_maps):
            if self.rank == 0:
                print(f"Generating density field {i}")

            # There are two classes depending on which pixel scheme we're using.
            if pixel_scheme.name == 'gnomonic':
                field = nmt.NmtFieldFlat(lx, ly, clustering_weight, [d], templates=syst_nc) 
            elif pixel_scheme.name == 'healpix':
                field = nmt.NmtField(clustering_weight, [d], templates=syst_nc)
            else:
                raise ValueError(f"Pixelization scheme {pixel_scheme.name} not supported by NaMaster")

            density_fields.append(field)


        # And then the lensing maps
        for i, (g1, g2, lw) in enumerate(zip(g1_maps, g2_maps, lensing_weights)):
            if self.rank == 0:
                print(f"Generating lensing field {i}")
            # Same again.  NaMaster knows whether these are spin 0 or spin 2
            # from the length of the maps list we pass it (1 or 2).
            if pixel_scheme.name == 'gnomonic':
                field = nmt.NmtFieldFlat(lx, ly, lw, [g1,g2], templates=syst_wl) 
            elif pixel_scheme.name == 'healpix':
                field = nmt.NmtField(lw, [g1, g2], templates=syst_wl) 
            else:
                raise ValueError(f"Pixelization scheme {pixel_scheme.name} not supported by NaMaster")

            lensing_fields.append(field)
        
        return pixel_scheme, density_fields, lensing_fields, nbin_source, nbin_lens, f_sky


    def collect_results(self):
        if self.comm is None:
            return

        self.results = self.comm.gather(self.results, root=0)

        if self.rank == 0:
            # Concatenate results
            self.results = sum(self.results, [])

    def setup_results(self):
        self.results = []

    def choose_ell_bins(self, pixel_scheme, f_sky):
        import pymaster as nmt
        from .utils.nmt_utils import MyNmtBinFlat, MyNmtBin
        if pixel_scheme.name == 'healpix':
            # This is just approximate.  It will be very wrong
            # in cases with non-square patches.
            area = f_sky * 4 * np.pi
            width = np.sqrt(area) #radians
            nlb = self.config['bandwidth']

            # user can specify the bandwidth, or we can just use
            # the maximum sensible value of Delta ell.
            nlb = nlb if nlb>0 else max(1,int(2 * np.pi / width))

            # The subclass of NmtBin that we use here just adds some
            # helper methods compared to the default NaMaster one.
            # Can feed these back upstream if useful.
            ell_bins = MyNmtBin(int(pixel_scheme.nside), nlb=nlb)
        elif pixel_scheme.name == 'gnomonic':
            # For the flat case we have to specify the complete ell ranges
            # for each bin.  First convert the overall map widths into radians.
            lx = np.radians(pixel_scheme.nx * pixel_scheme.pixel_size_x)
            ly = np.radians(pixel_scheme.ny * pixel_scheme.pixel_size_y)

            # The overall min and max values for the entire range.
            # The min comes from the width of the whole map, and the max from
            # the pixel size.
            ell_min = max(2 * np.pi / lx, 2 * np.pi / ly)
            ell_max = min(pixel_scheme.nx * np.pi / lx, pixel_scheme.ny * np.pi / ly)

            # If the user provided a bandwidth then again, use that.  Otherwise
            # use 2*ell_min. 
            d_ell = self.config['bandwidth']
            d_ell = d_ell if d_ell>0 else 2 * ell_min
            n_ell = int((ell_max - ell_min) / d_ell) - 1
            l_bpw = np.zeros([2, n_ell])

            # Turn these into ranges per band
            band_mins = ell_min + np.arange(n_ell) * d_ell
            band_maxs = l_bpw[0, :] + d_ell

            # and make the NaMaster object.
            ell_bins = MyNmtBinFlat(band_mins, band_maxs)

        return ell_bins


    def select_calculations(self, nbins_source, nbins_lens):
        # Build up a big list of all the calculations we want to
        # perform.  We should probably expose this in the configuration
        # file so you can skip some.
        calcs = []

        # For shear-shear we omit pairs with j>i
        k = SHEAR_SHEAR
        for i in range(nbins_source):
            for j in range(i + 1):
                calcs.append((i, j, k))

        # For shear-position we use all pairs
        k = SHEAR_POS
        for i in range(nbins_source):
            for j in range(nbins_lens):
                calcs.append((i, j, k))

        # For position-position we omit pairs with j>i.
        # We do keep cross-pairs, since even though we may not want to
        # do parameter estimation with them they are useful diagnostics.
        k = POS_POS
        for i in range(nbins_lens):
            for j in range(i + 1):
                calcs.append((i, j, k))

        return calcs

    def compute_power_spectra(self, pixel_scheme, i, j, k, f_wl, f_d, ell_bins, tomo_info, cl_theory):
        # Compute power spectra
        # TODO: now all possible auto- and cross-correlation are computed.
        #      This should be tunable.
        # TODO: Fix window functions, and how to save them.

        # k refers to the type of measurement we are making
        import sacc
        import pymaster as nmt
        import healpy

        CEE=sacc.standard_types.galaxy_shear_cl_ee
        CBB=sacc.standard_types.galaxy_shear_cl_bb
        CdE=sacc.standard_types.galaxy_shearDensity_cl_e
        CdB=sacc.standard_types.galaxy_shearDensity_cl_b
        Cdd=sacc.standard_types.galaxy_density_cl

        type_name = NAMES[k]
        print(f"Process {self.rank} calculating {type_name} spectrum for bin pair {i},{j}")
        sys.stdout.flush()


        # The binning information - effective (mid) ell values and
        # the window information
        ls = ell_bins.get_effective_ells()
        win = [ell_bins.get_window(b) for b,l  in enumerate(ls)]

        # The healpix pixel windows.  C_ell estimates on healpix
        # maps (like all pixelized maps) are modulated by a pixel
        # window function, which we calculate here so we can remove
        # it below.  These are trivially fast to compute, so no point
        # caching.
        if pixel_scheme.name == 'healpix':
            # Get the raw pixel window functions
            pixwin_t = healpy.pixwin(pixel_scheme.nside, False)
            pixwin_e, pixwin_b = healpy.pixwin(pixel_scheme.nside, True)
            # Interpolate to our ell values
            pixwin_ell = np.arange(pixwin_t.size)
            pixwin_t = np.interp(ls, pixwin_ell, pixwin_t)
            pixwin_e = np.interp(ls, pixwin_ell, pixwin_e)
            pixwin_b = np.interp(ls, pixwin_ell, pixwin_b)
        else:
            # TODO figure out what these are supposed to be
            pixwin_t = 1.0
            pixwin_e = 1.0
            pixwin_b = 1.0


        # We need the theory spectrum for this pair
        #TODO: when we have templates to deproject, use this.
        theory = cl_theory[(i,j,k)]
        ell_guess = cl_theory['ell']
        cl_guess = None

        if k == SHEAR_SHEAR:
            field_i = f_wl[i]
            field_j = f_wl[j]
            results_to_use = [(0, CEE, pixwin_e**2), (3, CBB, pixwin_b**2)]

        elif k == POS_POS:
            field_i = f_d[i]
            field_j = f_d[j]
            results_to_use = [(0, Cdd, pixwin_t**2)]

        elif k == SHEAR_POS:
            field_i = f_wl[i]
            field_j = f_d[j]
            results_to_use = [(0, CdE, pixwin_t*pixwin_e), (1, CdB, pixwin_t*pixwin_b)]

        if pixel_scheme.name == 'healpix':
            workspace = nmt.NmtWorkspace()
        elif pixel_scheme.name == 'gnomonic':
            workspace = nmt.NmtWorkspaceFlat()
        else:
            raise ValueError(f"No NaMaster workspace for pixel scheme {pixel_scheme.name}")

        # Compute mode-coupling matrix
        workspace.compute_coupling_matrix(field_i, field_j, ell_bins)

        # Get the coupled noise C_ell values to give to the master algorithm
        cl_noise = self.compute_noise(i,j,k,ell_bins,workspace,tomo_info)

        # Run the master algorithm
        if pixel_scheme.name == 'healpix':
            c = nmt.compute_full_master(field_i, field_j, ell_bins,
                cl_noise=cl_noise, cl_guess=cl_guess, workspace=workspace)
        elif pixel_scheme.name == 'gnomonic':
            c = nmt.compute_full_master_flat(field_i, field_j, ell_bins,
                cl_noise=cl_noise, cl_guess=cl_guess, ells_guess=ell_guess,
                workspace=workspace)

        # Save all the results, skipping things we don't want like EB modes
        for index, name, pixwin in results_to_use:
            self.results.append(Measurement(name, ls, c[index] / pixwin, win, i, j))


    def compute_noise(self, i, j, k, ell_bins, w, tomo_info):
        # No noise contribution in cross-correlations
        if (i!=j) or (k==SHEAR_POS):
            return None

        # We loaded in sigma_e and the densities
        # earlier on, and put them in the tomo_info dictionary
        if k==SHEAR_SHEAR:
            noise_level = tomo_info['sigma_e'][i]**2 / tomo_info['n_eff_steradian'][i]
        else:
            noise_level = 1.0 / tomo_info['n_lens_steradian'][i]

        # Number of ell values in the uncoupled C_ell (banded in flat case)
        if ell_bins.is_flat():
            n = ell_bins.get_n_bands()
        else:
            n = w.wsp.lmax + 1

        # Uncoupled noise C_ell value
        N1 = np.ones(n) * noise_level

        if k==SHEAR_SHEAR:
            N2 = [N1, N1, N1, N1]
        else:
            N2 = [N1]

        # Couple to take coupled C_ell value
        if ell_bins.is_flat():
            ell = ell_bins.get_effective_ells()
            cl_noise = w.couple_cell(N2)
        else:
            cl_noise = w.couple_cell(N2)

        return cl_noise


    def load_tomographic_quantities(self, nbin_source, nbin_lens, f_sky):
        # Get lots of bits of metadata from the input file,
        # per tomographic bin.
        metadata = self.open_input('tracer_metadata')
        sigma_e = metadata['tracers/sigma_e'][:]
        mean_R = metadata['tracers/R_gamma_mean'][:]
        N_eff = metadata['tracers/N_eff'][:]
        lens_counts = metadata['tracers/lens_counts'][:]
        metadata.close()

        area = 4*np.pi*f_sky
        n_eff = N_eff / area
        n_lens = lens_counts / area

        tomo_info = {
            "area_steradians": area,
            "n_eff_steradian": n_eff,
            "n_lens_steradian": n_lens,
            "sigma_e": sigma_e,
            "f_sky": f_sky,
            "mean_R": mean_R,
        }

        warnings.warn("Using unweighted lens samples here")

        return tomo_info



    def load_tracers(self, nbin_source, nbin_lens):
        # Load the N(z) and convert to sacc tracers.
        # We need this both to put it into the output file,
        # but also potentially to compute the theory guess
        # for projecting out modes
        import sacc
        f = self.open_input('photoz_stack')

        tracers = {}

        for i in range(nbin_source):
            name = f"source_{i}"
            z = f['n_of_z/source/z'][:]
            Nz = f[f'n_of_z/source/bin_{i}'][:]
            T = sacc.BaseTracer.make("NZ", name, z, Nz)
            tracers[name] = T

        for i in range(nbin_lens):
            name = f"lens_{i}"
            z = f['n_of_z/lens/z'][:]
            Nz = f[f'n_of_z/lens/bin_{i}'][:]
            T = sacc.BaseTracer.make("NZ", name, z, Nz)
            tracers[name] = T

        return tracers




    def save_power_spectra(self, tracers, nbin_source, nbin_lens):
        import sacc
        from sacc.windows import TopHatWindow
        CEE=sacc.standard_types.galaxy_shear_cl_ee
        CBB=sacc.standard_types.galaxy_shear_cl_bb
        CdE=sacc.standard_types.galaxy_shearDensity_cl_e
        CdB=sacc.standard_types.galaxy_shearDensity_cl_b
        Cdd=sacc.standard_types.galaxy_density_cl

        S = sacc.Sacc()

        for tracer in tracers.values():
            S.add_tracer_object(tracer)

        # We have saved the results in a big list.  Each entry contains a single
        # bin pair and spectrum type, but many data points at different angles.
        # Here we pull them all out to add to sacc
        for d in self.results:
            tracer1 = f'source_{d.i}' if d.corr_type in [CEE, CBB, CdE, CdB] else f'lens_{d.i}'
            tracer2 = f'source_{d.j}' if d.corr_type in [CEE, CBB] else f'lens_{d.j}'

            n = len(d.l)
            for i in range(n):
                ell_vals = d.win[i][0]  # second term is weights
                win = TopHatWindow(ell_vals[0], ell_vals[-1])
                # We use optional tags i and j here to record the bin indices, as well
                # as in the tracer names, in case it helps to select on them later.
                S.add_data_point(d.corr_type, (tracer1, tracer2), d.value[i],
                    ell=d.l[i], window=win, i=d.i, j=d.j)

        # Save provenance information
        for key, value in self.gather_provenance().items():
            if isinstance(value, str) and '\n' in value:
                values = value.split("\n")
                for i,v in enumerate(values):
                    S.metadata[f'provenance/{key}_{i}'] = v
            else:
                S.metadata[f'provenance/{key}'] = value


        # And we're all done!
        output_filename = self.get_output("twopoint_data_fourier")
        S.save_fits(output_filename, overwrite=True)


if __name__ == '__main__':
    PipelineStage.main()


