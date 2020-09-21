from .base_stage import PipelineStage
from .data_types import TomographyCatalog, \
                        YamlFile, SACCFile, MapsFile, HDFFile, \
                        PhotozPDFFile, LensingNoiseMaps, ClusteringNoiseMaps, PNGFile
import numpy as np
import collections
from .utils import choose_pixelization, array_hash
from .utils.theory import theory_3x2pt
import sys
import warnings
import pathlib

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
        ('shear_photoz_stack', HDFFile),  # Photoz stack
        ('lens_photoz_stack', HDFFile),  # Photoz stack
        ('fiducial_cosmology', YamlFile),  # For the cosmological parameters
        ('tracer_metadata', TomographyCatalog),  # For density info
        ('source_maps', MapsFile),
        ('density_maps', MapsFile),
        ('aux_maps', MapsFile),
        ('mask', MapsFile),
        ('source_noise_maps', LensingNoiseMaps),
        ('lens_noise_maps', ClusteringNoiseMaps),
        ('shear_tomography_catalog', TomographyCatalog),  # For density info
        ('lens_tomography_catalog', TomographyCatalog),  # For density info
    ]
    outputs = [
        ('twopoint_data_fourier', SACCFile)
    ]

    config_options = {
        "mask_threshold": 0.0,
        "bandwidth": 0,
        "flip_g1": False,
        "flip_g2": False,
        "cache_dir": '',
        "deproject_syst_clustering": False,
        "systmaps_clustering_dir": '',
        "ell_min": 100,
        "ell_max": 1500,
        "n_ell": 20,
        "ell_spacing": 'log'
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
        pixel_scheme, maps, f_sky = self.load_maps()
        if self.rank==0:
            print("Loaded maps.")

        nbin_source = len(maps['g'])
        nbin_lens = len(maps['d'])

        # Get the complete list of calculations to be done,
        # for all the three spectra and all the bin pairs.
        # This will be used for parallelization.
        calcs = self.select_calculations(nbin_source, nbin_lens)


        # Load in the per-bin auto-correlation noise levels and 
        # mean response values
        # Note - this is currently unused, because we are using the noise
        # maps instead of an analytic form, but that could change later
        # so I will leave this here.
        tomo_info = self.load_tomographic_quantities(nbin_source, nbin_lens, f_sky)


        # Binning scheme, currently chosen from the geometry.
        # TODO: set ell binning from config
        ell_bins = self.choose_ell_bins(pixel_scheme, f_sky)

        workspaces = self.make_workspaces(maps, calcs, ell_bins)

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
        for i, j, k in calcs:
            self.compute_power_spectra(
                pixel_scheme, i, j, k, maps, workspaces, ell_bins, theory_cl)

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
        # Load the maps from their files.
        # First the mask
        with self.open_input('mask', wrapper=True) as f:
            info = f.read_map_info('mask')
            area = info['area']
            f_sky = info['f_sky']
            mask = f.read_map('mask')
            print("Loaded mask")
        # Then the shear maps and weights
        with self.open_input('source_maps', wrapper=True) as f:
            nbin_source = f.file['maps'].attrs['nbin_source']
            g1_maps = [f.read_map(f'g1_{b}') for b in range(nbin_source)]
            g2_maps = [f.read_map(f'g2_{b}') for b in range(nbin_source)]
            lensing_weights = [f.read_map(f'lensing_weight_{b}') for b in range(nbin_source)]
            print(f"Loaded 2 x {nbin_source} shear maps")
            print(f"Loaded {nbin_source} lensing weight maps")

        # And finally the density maps
        with self.open_input('density_maps', wrapper=True) as f:
            nbin_lens = f.file['maps'].attrs['nbin_lens']
            d_maps = [f.read_map(f'delta_{b}') for b in range(nbin_lens)]
            print(f"Loaded {nbin_lens} overdensity maps")


        # Choose pixelization and read mask and systematics maps
        pixel_scheme = choose_pixelization(**info)

        if self.rank == 0:
            print(f"Unmasked area = {area:.2f} deg^2, fsky = {f_sky:.2e}")

        if pixel_scheme.name != 'healpix':
            raise ValueError("TXTwoPointFourier can only run on healpix maps")


        # Using a flat mask as the clustering weight for now, since I need to know
        # how to turn the depth map into a weight
        clustering_weight = mask


        # Set any unseen pixels to zero weight.
        for d in d_maps:
            clustering_weight[clustering_weight==healpy.UNSEEN] = 0
            clustering_weight[d==healpy.UNSEEN] = 0

        # Mask any pixels which have the healpix bad value
        for (g1, g2, lw) in zip(g1_maps, g2_maps, lensing_weights):
            lw[g1 == healpy.UNSEEN] = 0
            lw[g2 == healpy.UNSEEN] = 0
            lw[lw == healpy.UNSEEN] = 0
            
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
                

        # Load HEALPix systematics maps
        deproject_syst_clustering = self.config['deproject_syst_clustering']
        if deproject_syst_clustering:  
            print('Deprojecting systematics maps for number counts')
            n_systmaps = 0
            s_maps = []
            systmaps_clustering_dir = self.config['systmaps_clustering_dir']
            systmaps_path = pathlib.Path(systmaps_clustering_dir)
            for systmap in systmaps_path.iterdir():
                try:
                    if systmap.is_file():
                        if pathlib.Path(systmap).suffix != ".fits":
                            print('Warning: Problem reading systematics map file', systmap, 'Not a HEALPix .fits file.')
                            warnings.warn("Systematics map file must be a HEALPix .fits file.")
                            print('Ignoring', systmap)
                        else:
                            systmap_file = str(systmap)
                            self.config[f'clustering_deproject_{n_systmaps}'] = systmap_file # for provenance
                            print('Reading clustering systematics map file:', systmap_file)
                            syst_map = healpy.read_map(systmap_file,verbose=False)

#                             # Find value at given ra,dec
#                             ra = 55.
#                             dec = -30.
#                             theta = 0.5 * np.pi - np.deg2rad(dec)
#                             phi = np.deg2rad(ra)
#                             nside = healpy.pixelfunc.get_nside(syst_map)
#                             ipix = healpy.ang2pix(nside, theta, phi)
#                             print('Syst map: value at ra,dec = 55,-30: ', syst_map[ipix])

                            # normalize map for Namaster
                            # set pixel values to value/mean - 1
                            syst_map_mask = syst_map != healpy.UNSEEN
                            mean = np.mean(syst_map[syst_map_mask]) # gives mean of all pixels with mask applied
                            if mean != 0:
                                print('Syst map: mean value = ', mean)
                                syst_map[~syst_map_mask] = 0 # sets unmasked pixels to zero 
                                syst_map = syst_map / mean - 1
#                                 print('Syst map', systmap_file, 'normalized value at ra,dec = 55,-30: ', syst_map[ipix])

                            s_maps.append(syst_map)
                            n_systmaps += 1
                except:
                    print('Warning: Problem with systematics map file',systmap)
                    print('Ignoring', systmap)
                    
            print('Number of systematics maps read: ', n_systmaps)
            if n_systmaps == 0:
                print('No systematics maps found. Skipping deprojection.')
                deproject_syst_clustering = False
            else:
                print("Using systematics maps for galaxy number counts.")
                # We assume all systematics maps have the same nside
                nside = healpy.pixelfunc.get_nside(syst_map)
                npix = healpy.nside2npix(nside)
                # needed for NaMaster:
                s_maps_nc = np.array(s_maps).reshape([n_systmaps, 1, npix])
            
        else:
            print("Not using systematics maps for deprojection in NaMaster")    

        if deproject_syst_clustering:
            density_fields = [(nmt.NmtField(clustering_weight, [d], templates=s_maps_nc, n_iter=0))
                              for d in d_maps]
        else:
            density_fields = [(nmt.NmtField(clustering_weight, [d], n_iter=0))
                              for d in d_maps]

        lensing_fields = [(nmt.NmtField(lw, [g1, g2], n_iter=0))
                          for (lw, g1, g2) in zip(lensing_weights, g1_maps, g2_maps)]

        # Collect together all the maps we will output
        maps = {
            'dw': clustering_weight,
            'lw': lensing_weights,
            'g': list(zip(g1_maps, g2_maps)),
            'd': d_maps,
            'lf': lensing_fields,
            'df': density_fields,

        }

        return pixel_scheme, maps, f_sky

    def load_workspace_cache(self):
        from .utils.nmt_utils import WorkspaceCache
        dirname = self.config['cache_dir']

        if not dirname:
            if self.rank == 0:
                print("Not using an on-disc cache.  Set cache_dir to use one")
            return {}

        cache = WorkspaceCache(dirname)
        return cache

    def save_workspace_cache(self, cache, spaces):
        if (cache is None) or (cache == {}):
            return

        for space in spaces.values():
            cache.put(space)

    def make_workspaces(self, maps, calcs, ell_bins):
        import pymaster as nmt

        # Make the field object
        if self.rank == 0:
            print("Preparing workspaces")

        # load the cache
        cache = self.load_workspace_cache()

        nbin_source = len(maps['g'])
        nbin_lens = len(maps['d'])

        # empty workspaces
        zero = np.zeros_like(maps['dw'])

        lensing_weights = maps['lw']
        density_weight  = maps['dw']
        lensing_fields = maps['lf']
        density_fields  = maps['df']

        ell_hash = array_hash(ell_bins.get_effective_ells())
        hashes = {id(x):array_hash(x) for x in lensing_weights}
        hashes[id(density_weight)] = array_hash(density_weight)


        # It's an oddity of NaMaster that we
        # need to supply a field object to the workspace even though the
        # coupling matrix only depends on the mask and ell binning.
        # So here we re-use the fields we've made for the actual
        # analysis, but to save the coupling matrices while being sure
        # we're using the right ones we derive a key based on the
        # mask and ell centers.

        # Each lensing field has its own mask
        lensing_fields = [(lw, lf)
                          for lw, lf in zip(lensing_weights, lensing_fields)]

        # The density fields share a mask, so just use the field
        # object for the first one.  
        density_field = (density_weight, density_fields[0])

        spaces = {}

        def get_workspace(f1, f2):
            w1, f1 = f1
            w2, f2 = f2

            # First we derive a hash which will change whenever either
            # the mask changes or the ell binning.
            # We combine the hashes of those three objects together
            # using xor (python  ^ operator), which seems to be standard.
            h1 = hashes[id(w1)]
            h2 = hashes[id(w2)]
            key = h1 ^ ell_hash
            # ... except that if we are using the same field twice then
            # x ^ x = 0, which causes problems (all the auto-bins would
            # have the same key).  So we only combine the hashes if the
            # fields are different.
            if h2 != h1:
                key ^= h2
            # Check on disc to see if we have one saved already.
            space = cache.get(key)
            # If not, compute it.  We will save it later
            if space is None:
                print(f'Rank {self.rank} computing coupling matrix '
                      f"{i}, {j}, {k}")
                space = nmt.NmtWorkspace()
                space.compute_coupling_matrix(f1, f2, ell_bins,is_teb=False, n_iter=1)
            else:
                print(f'Rank {self.rank} getting coupling matrix '
                       f'{i}, {j}, {k} from cache.')
            # This is a bit awkward - we attach the key to the
            # object to avoid more book-keeping.  This is used inside
            # the workspace cache
            space.txpipe_key = key
            return space

        for (i, j, k) in calcs:
            if k == SHEAR_SHEAR:
                f1 = lensing_fields[i]
                f2 = lensing_fields[j]
            elif k == SHEAR_POS:
                f1 = lensing_fields[i]
                f2 = density_field
            else:
                f1 = density_field
                f2 = density_field
            spaces[(i,j,k)] = get_workspace(f1, f2)

        self.save_workspace_cache(cache, spaces)
        return spaces

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
        from .utils.nmt_utils import MyNmtBin

        # commented code below is not needed anymore
        '''
        # This is just approximate.  It will be very wrong
        # in cases with non-square patches.
        area = f_sky * 4 * np.pi
        width = np.sqrt(area) #radians
        nlb = self.config['bandwidth']

        # user can specify the bandwidth, or we can just use
        # the maximum sensible value of Delta ell.
        nlb = nlb if nlb>0 else max(1,int(2 * np.pi / width))
        '''
        
        # The subclass of NmtBin that we use here just adds some
        # helper methods compared to the default NaMaster one.
        # Can feed these back upstream if useful.

        # Creating the ell binning from the edges using this Namaster constructor.
        edges = np.unique(np.geomspace(self.config['ell_min'], self.config['ell_max'], self.config['n_ell']).astype(int))
        ell_bins = MyNmtBin.from_edges(edges_int[:-1], edges_int[1:], is_Dell=False)
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

        calcs = [calc for calc in self.split_tasks_by_rank(calcs)]

        return calcs

    def compute_power_spectra(self, pixel_scheme, i, j, k, maps, workspaces, ell_bins, cl_theory):
        # Compute power spectra
        # TODO: now all possible auto- and cross-correlation are computed.
        #      This should be tunable.

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

        # We need the theory spectrum for this pair
        #TODO: when we have templates to deproject, use this.
        theory = cl_theory[(i,j,k)]
        ell_guess = cl_theory['ell']
        cl_guess = None

        if k == SHEAR_SHEAR:
            field_i = maps['lf'][i]
            field_j = maps['lf'][j]
            results_to_use = [(0, CEE, ), (3, CBB, )]

        elif k == POS_POS:
            field_i = maps['df'][i]
            field_j = maps['df'][j]
            results_to_use = [(0, Cdd)]

        elif k == SHEAR_POS:
            field_i = maps['lf'][i]
            field_j = maps['df'][j]
            results_to_use = [(0, CdE), (1, CdB)]

        workspace = workspaces[(i,j,k)]

        # Get the coupled noise C_ell values to give to the master algorithm
        cl_noise = self.compute_noise(i, j, k, ell_bins, maps, workspace)

        # Run the master algorithm
        c = nmt.compute_full_master(field_i, field_j, ell_bins,
            cl_noise=cl_noise, cl_guess=cl_guess, workspace=workspace, n_iter=1)

        # Save all the results, skipping things we don't want like EB modes
        for index, name in results_to_use:
            self.results.append(Measurement(name, ls, c[index], win, i, j))


    def compute_noise(self, i, j, k, ell_bins, maps, workspace):
        import pymaster as nmt
        import healpy

        # No noise contribution in cross-correlations
        if (i!=j) or (k==SHEAR_POS):
            return None


        if k == SHEAR_SHEAR:
            noise_maps = self.open_input('source_noise_maps', wrapper=True)
            weight = maps['lw'][i]

        else:
            noise_maps = self.open_input('lens_noise_maps', wrapper=True)
            weight = maps['dw']
        
        nreal = noise_maps.number_of_realizations()
        noise_c_ells = []

        for r in range(nreal):
            print(f"Analyzing noise map {r}")
            # Load the noise map - may be either (g1, g2)
            # or rho1 - rho2

            # Also in the event there are any UNSEENs in these,
            # downweight them
            w = weight.copy()
            if k == SHEAR_SHEAR:
                realization = noise_maps.read_rotation(r, i)
                w[realization[0] == healpy.UNSEEN] = 0
                w[realization[1] == healpy.UNSEEN] = 0
            else:
                rho1, rho2 = noise_maps.read_density_split(r, i)
                realization = [rho1 - rho2]
                w[realization[0] == healpy.UNSEEN] = 0

            # Analyze it with namaster
            field = nmt.NmtField(w, realization, n_iter=0)
            cl_coupled = nmt.compute_coupled_cell(field, field)

            # Accumulate
            noise_c_ells.append(cl_coupled)

        mean_noise = np.mean(noise_c_ells, axis=0)

        # Since this is the noise on the half maps the real
        # noise C_ell will be (0.5)**2 times the size
        if k == POS_POS:
            mean_noise /= 4

        return mean_noise



    def load_tomographic_quantities(self, nbin_source, nbin_lens, f_sky):
        # Get lots of bits of metadata from the input file,
        # per tomographic bin.
        metadata = self.open_input('tracer_metadata')
        sigma_e = metadata['tracers/sigma_e'][:]
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
        }

        warnings.warn("Using unweighted lens samples here")

        return tomo_info



    def load_tracers(self, nbin_source, nbin_lens):
        # Load the N(z) and convert to sacc tracers.
        # We need this both to put it into the output file,
        # but also potentially to compute the theory guess
        # for projecting out modes
        import sacc
        f_shear = self.open_input('shear_photoz_stack')
        f_lens = self.open_input('lens_photoz_stack')

        tracers = {}

        for i in range(nbin_source):
            name = f"source_{i}"
            z = f_shear['n_of_z/source/z'][:]
            Nz = f_shear[f'n_of_z/source/bin_{i}'][:]
            T = sacc.BaseTracer.make("NZ", name, z, Nz)
            tracers[name] = T

        for i in range(nbin_lens):
            name = f"lens_{i}"
            z = f_lens['n_of_z/lens/z'][:]
            Nz = f_lens[f'n_of_z/lens/bin_{i}'][:]
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
        provenance = self.gather_provenance()
        provenance.update(SACCFile.generate_provenance())
        for key, value in provenance.items():
            if isinstance(value, str) and '\n' in value:
                values = value.split("\n")
                for i,v in enumerate(values):
                    S.metadata[f'provenance/{key}_{i}'] = v
            else:
                S.metadata[f'provenance/{key}'] = value


        # And we're all done!
        output_filename = self.get_output("twopoint_data_fourier")
        S.save_fits(output_filename, overwrite=True)



class TXTwoPointPlotsFourier(PipelineStage):

    name='TXTwoPointPlotsFourier'
    inputs = [
        ('summary_statistics_fourier', SACCFile),
        ('fiducial_cosmology', YamlFile),  # For example lines
    ]
    outputs = [
        ('shear_cl_ee', PNGFile),
        ('shearDensity_cl', PNGFile),
        ('density_cl', PNGFile),
    ]

    config_options = {
        'wspace': 0.05,
        'hspace': 0.05,
    }

    def read_nbin(self,s):
        sources = []
        lenses = []
        for tn,t in s.tracers.items():
            if 'source' in tn:
                sources.append(tn)
            if 'lens' in tn:
                lenses.append(tn)
        return len(sources), len(lenses)

    def run(self):
        import sacc
        import matplotlib
        import pyccl
        from .plotting import full_3x2pt_plots
        matplotlib.use('agg')
        matplotlib.rcParams["xtick.direction"]='in'
        matplotlib.rcParams["ytick.direction"]='in'

        filename = self.get_input('summary_statistics_fourier')
        s = sacc.Sacc.load_fits(filename)
        nbin_source, nbin_lens = self.read_nbin(s)  
 
        cosmo = pyccl.Cosmology.read_yaml("./data/fiducial_cosmology.yml")

        outputs = {
            "galaxy_density_cl": self.open_output('density_cl',
                figsize=(3.5*nbin_lens, 3*nbin_lens), wrapper=True),

            "galaxy_shearDensity_cl_e": self.open_output('shearDensity_cl',
                figsize=(3.5*nbin_lens, 3*nbin_source), wrapper=True),

            "galaxy_shear_cl_ee": self.open_output('shear_cl_ee',
                figsize=(3.5*nbin_source, 3*nbin_source), wrapper=True),

        }

        figures = {key: val.file for key, val in outputs.items()}

        full_3x2pt_plots([filename], ['summary_statistics_fourier'], 
                         figures=figures, cosmo=cosmo, theory_labels=['Fiducial'], xi=False, xlogscale=True)

        for fig in outputs.values():
            fig.close()


        

if __name__ == '__main__':
    PipelineStage.main()


