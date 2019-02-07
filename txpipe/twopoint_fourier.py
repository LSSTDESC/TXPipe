from ceci import PipelineStage
from .data_types import MetacalCatalog, TomographyCatalog, RandomsCatalog, YamlFile, SACCFile, DiagnosticMaps, HDFFile, PhotozPDFFile
import numpy as np
import collections
from .utils import choose_pixelization, HealpixScheme, GnomonicPixelScheme, ParallelStatsCalculator
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
        ('tomography_catalog', TomographyCatalog),  # For density info
    ]
    outputs = [
        ('twopoint_data', SACCFile)
    ]

    config_options = {
        "mask_threshold": 0.0,
        "bandwidth": 0,
        "apodization_size": 3.0,
        "apodization_type": "C1",  #"C1", "C2", or "Smooth"
        "flip_g2": False,
    }

    def run(self):
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
        noise, mean_R = self.load_tomographic_quantities(nbin_source, nbin_lens, f_sky)

        # Load the n(z) values, which are both saved in the output
        # file alongside the spectra, and used to calcualate the
        # fiducial theory C_ell
        tracers = self.load_tracers(nbin_source, nbin_lens)

        # This is needed in the deprojection calculation
        theory_cl = self.fiducial_theory(tracers, f_d, f_wl, nbin_source, nbin_lens)

        # Binning scheme, currently chosen from the geometry.
        # TODO: set ell binning from config
        ell_bins = self.choose_ell_bins(pixel_scheme, f_sky)

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




        # Namaster uses workspaces, which we re-use between
        # bins
        w00, w02, w22 = self.setup_workspaces(pixel_scheme, f_d, f_wl, ell_bins)
        print(f"Rank {self.rank} set up workspaces")

        # Run the compute power spectra portion in parallel
        # This splits the calculations among the parallel bins
        # It's not the most optimal way of doing it
        # as it's not dynamic, just a round-robin assignment.
        for i, j, k in self.split_tasks_by_rank(calcs):
            self.compute_power_spectra(
                pixel_scheme, i, j, k, f_wl, f_d, w00, w02, w22, ell_bins, noise, theory_cl)

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
        mask_threshold = self.config['mask_threshold']
        mask[mask <= mask_threshold] = 0
        mask[np.isnan(mask)] = 0
        mask_sum = mask.sum()
        f_sky = area / 41253.
        if self.rank == 0:
            print(f"Unmasked area = {area}, fsky = {f_sky}")


        # Load all the maps in.
        # TODO: make it possible to just do a subset of these
        ngal_maps = [map_file.read_map(f'ngal_{b}') for b in range(nbin_lens)]
        g1_maps = [map_file.read_map(f'g1_{b}') for b in range(nbin_source)]
        g2_maps = [map_file.read_map(f'g2_{b}') for b in range(nbin_source)]

        if self.config['flip_g2']:
            for g2 in g2_maps:
                w = np.where(g2!=healpy.UNSEEN)
                g2[w]*=-1

        # TODO: load systematics maps here, once we are making them.
        syst_nc = None
        syst_wl = None

        map_file.close()

        # Cut out any pixels below the threshold,
        # zeroing out any pixels there
        cut = mask < mask_threshold
        mask[cut] = 0

        # We also apply this cut to the count maps,
        # since otherwise the pixels below threshold would contaminate
        # the mean calculation below.
        for ng in ngal_maps:
            ng[cut] = 0


        # Convert the number count maps to overdensity maps.
        # First compute the overall mean object count per bin.
        # Maybe we should do this in the mapping code itself?
        n_means = [ng.sum()/mask_sum for ng in ngal_maps]
        # and then use that to convert to overdensity
        d_maps = [(ng/mu)-1 for (ng,mu) in zip(ngal_maps, n_means)]

        d_fields = []
        wl_fields = []

        if pixel_scheme.name == 'gnomonic':
            lx = np.radians(pixel_scheme.size_x)
            ly = np.radians(pixel_scheme.size_y)

            # Apodize the mask
            if apod_size > 0:
                if self.rank==0:
                    print(f"Apodizing mask with size {apod_size} deg and method {apod_type}")
                mask = nmt.mask_apodization(mask, lx, ly, apod_size, apotype=apod_type)
            elif self.rank==0:
                print("Not apodizing mask.")

            for i,d in enumerate(d_maps):
                # Density for gnomonic maps
                if self.rank == 0:
                    print(f"Generating gnomonic density field {i}")
                field = nmt.NmtFieldFlat(lx, ly, mask, [d], templates=syst_nc) 
                d_fields.append(field)

            for i,(g1,g2) in enumerate(zip(g1_maps, g2_maps)):
                # Density for gnomonic maps
                if self.rank == 0:
                    print(f"Generating gnomonic lensing field {i}")
                field = nmt.NmtFieldFlat(lx, ly, mask, [g1,g2], templates=syst_wl) 
                wl_fields.append(field)
            

        elif pixel_scheme.name == 'healpix':
            # Apodize the mask
            if apod_size > 0:
                if self.rank==0:
                    print(f"Apodizing mask with size {apod_size} deg and method {apod_type}")
                mask = nmt.mask_apodization(mask, apod_size, apotype=apod_type)
            elif self.rank==0:
                print("Not apodizing mask.")


            for i,d in enumerate(d_maps):
                # Density for gnomonic maps
                print(f"Generating healpix density field {i}")
                field = nmt.NmtField(mask, [d], templates=syst_nc)
                d_fields.append(field)

            for i,(g1,g2) in enumerate(zip(g1_maps, g2_maps)):
                # Density for gnomonic maps
                print(f"Generating healpix lensing field {i}")
                field = nmt.NmtField(mask, [g1, g2], templates=syst_wl) 
                wl_fields.append(field)

        else:
            raise ValueError(f"Pixelization scheme {pixel_scheme.name} not supported by NaMaster")

        return pixel_scheme, d_fields, wl_fields, nbin_source, nbin_lens, f_sky


    def collect_results(self):
        if self.comm is None:
            return

        self.results = self.comm.gather(self.results, root=0)

        if self.rank == 0:
            # Concatenate results
            self.results = sum(self.results, [])

            # Order by type, then bin i, then bin j
            order = ['CEE', 'CBB', 'Cdd', 'CdE', 'CdB']
            key = lambda r: (order.index(r.corr_type), r.i, r.j)
            self.results = sorted(self.results, key=key)

    def setup_results(self):
        self.results = []

    def choose_ell_bins(self, pixel_scheme, f_sky):
        import pymaster as nmt
        from .utils.nmt_utils import MyNmtBinFlat, MyNmtBin
        if pixel_scheme.name == 'healpix':
            # This is just approximate
            area = f_sky * 4 * np.pi
            width = np.sqrt(area) #radians
            nlb = self.config['bandwidth']
            nlb = nlb if nlb>0 else max(1,int(2 * np.pi / width))
            ell_bins = MyNmtBin(int(pixel_scheme.nside), nlb=nlb)
        elif pixel_scheme.name == 'gnomonic':
            lx = np.radians(pixel_scheme.nx * pixel_scheme.pixel_size_x)
            ly = np.radians(pixel_scheme.ny * pixel_scheme.pixel_size_y)
            ell_min = max(2 * np.pi / lx, 2 * np.pi / ly)
            ell_max = min(pixel_scheme.nx * np.pi / lx, pixel_scheme.ny * np.pi / ly)
            d_ell = self.config['bandwidth']
            d_ell = d_ell if d_ell>0 else 2 * ell_min
            n_ell = int((ell_max - ell_min) / d_ell) - 1
            l_bpw = np.zeros([2, n_ell])
            l_bpw[0, :] = ell_min + np.arange(n_ell) * d_ell
            l_bpw[1, :] = l_bpw[0, :] + d_ell
            ell_bins = MyNmtBinFlat(l_bpw[0, :], l_bpw[1, :])
            ell_bins.ell_mins = l_bpw[0, :]
            ell_bins.ell_maxs = l_bpw[1, :]

        return ell_bins

    def setup_workspaces(self, pixel_scheme, f_d, f_wl, ell_bins):
        import pymaster as nmt
        # choose scheme class
        if pixel_scheme.name == 'healpix':
            workspace_class = nmt.NmtWorkspace
        elif pixel_scheme.name == 'gnomonic':
            workspace_class = nmt.NmtWorkspaceFlat
        else:
            raise ValueError(f"No NaMaster workspace for pixel scheme {pixel_scheme.name}")

        # Compute mode-coupling matrix
        # TODO: mode-coupling could be pre-computed and provided in config.
        w00 = workspace_class()
        w00.compute_coupling_matrix(f_d[0], f_d[0], ell_bins)
        if self.rank==0:
            print("Computed w00 coupling matrix")

        w02 = workspace_class()
        w02.compute_coupling_matrix(f_d[0], f_wl[0], ell_bins)
        if self.rank==0:
            print("Computed w02 coupling matrix")

        w22 = workspace_class()
        w22.compute_coupling_matrix(f_wl[0], f_wl[0], ell_bins)
        if self.rank==0:
            print("Computed w22 coupling matrix")

        return w00, w02, w22

    def select_calculations(self, nbins_source, nbins_lens):
        calcs = []

        # For shear-shear we omit pairs with j<i
        k = SHEAR_SHEAR
        for i in range(nbins_source):
            for j in range(i + 1):
                calcs.append((i, j, k))

        # For shear-position we use all pairs
        k = SHEAR_POS
        for i in range(nbins_source):
            for j in range(nbins_lens):
                calcs.append((i, j, k))

        # For position-position we omit pairs with j<i
        k = POS_POS
        for i in range(nbins_lens):
            for j in range(i + 1):
                calcs.append((i, j, k))

        return calcs

    def compute_power_spectra(self, pixel_scheme, i, j, k, f_wl, f_d, w00, w02, w22, ell_bins, noise, cl_theory):
        # Compute power spectra
        # TODO: now all possible auto- and cross-correlation are computed.
        #      This should be tunable.
        # TODO: Fix window functions, and how to save them.

        # k refers to the type of measurement we are making
        type_name = NAMES[k]
        print(f"Process {self.rank} calcluating {type_name} spectrum for bin pair {i},{i}")
        sys.stdout.flush()

        # We need the theory spectrum for this pair
        theory = cl_theory[(i,j,k)]



        if k == SHEAR_SHEAR:
            ls = ell_bins.get_effective_ells()
            # Top-hat window functions
            win = [ell_bins.get_window(b) for b,l  in enumerate(ls)]
            cl_noise = self.compute_noise(i,j,k,w22,noise)
            cl_guess = [theory, np.zeros_like(theory), np.zeros_like(theory), np.zeros_like(theory)]
            c = self.compute_one_spectrum(
                pixel_scheme, w22, f_wl[i], f_wl[j], ell_bins, cl_noise, cl_guess)
            c_EE = c[0]
            c_BB = c[3]
            self.results.append(Measurement('CEE', ls, c_EE, win, i, j))
            self.results.append(Measurement('CBB', ls, c_BB, win, i, j))

        if k == POS_POS:
            ls = ell_bins.get_effective_ells()
            win = [ell_bins.get_window(b) for b,l  in enumerate(ls)]
            cl_noise = self.compute_noise(i,j,k,w00,noise)
            cl_guess = [theory]
            c = self.compute_one_spectrum(
                pixel_scheme, w00, f_d[i], f_d[j], ell_bins, cl_noise, cl_guess)
            c_dd = c[0]
            self.results.append(Measurement('Cdd', ls, c_dd, win, i, j))

        if k == SHEAR_POS:
            ls = ell_bins.get_effective_ells()
            win = [ell_bins.get_window(b) for b,l  in enumerate(ls)]
            cl_noise = self.compute_noise(i,j,k,w02,noise)
            cl_guess = [theory, np.zeros_like(theory)]

            c = self.compute_one_spectrum(
                pixel_scheme, w02, f_wl[i], f_d[j], ell_bins, cl_noise, cl_guess)
            c_dE = c[0]
            c_dB = c[1]
            self.results.append(Measurement('CdE', ls, c_dE, win, i, j))
            self.results.append(Measurement('CdB', ls, c_dB, win, i, j))

    def compute_noise(self, i, j, k, w, noise):
        # No noise contribution from cross-correlations
        if (i!=j) or (k==SHEAR_POS):
            return None
        print("x")
        # We loaded in sigma_e and the densities
        # earlier on and put them in the noise dictionary
        noise_level = noise[(i,k)]

        # ell-by-ell noise level of the right size
        N1 = np.ones(w.wsp.lmax + 1) * noise[(i,k)]

        # return N1

        # # Need the same noise for EE, BB, EB, BE
        # # or PE, PB
        if k==SHEAR_SHEAR:
            N2 = [N1, N1, N1, N1]
        elif k==SHEAR_POS:
            N2 = [N1, N1]
        else:
            N2 = [N1]
        return N2

        # # Run the same decoupling process that we will use
        # # on the full spectra
        # N_b = w.decouple_cell(N2)[0]

        # return N_b


    def compute_one_spectrum(self, pixel_scheme, w, f1, f2, ell_bins, cl_noise, theory):
        import pymaster as nmt



        if pixel_scheme.name == 'healpix':
            # correlates two fields f1 and f2 and returns an array of coupled
            # power spectra
            coupled_c_ell = nmt.compute_coupled_cell(f1, f2)
            # Compute 
            cl_bias = nmt.deprojection_bias(f1, f2, theory)

        elif pixel_scheme.name == 'gnomonic':
            coupled_c_ell = nmt.compute_coupled_cell_flat(f1, f2, ell_bins)
            ell_eff = ell_bins.get_effective_ells()
            cl_bias = nmt.deprojection_bias_flat(f1, f2, ell_bins, ell_eff, cl_theory)

        c_ell = w.decouple_cell(coupled_c_ell, cl_noise=cl_noise, cl_bias=cl_bias)
        return c_ell

    def load_tomographic_quantities(self, nbin_source, nbin_lens, f_sky):
        tomo = self.open_input('tomography_catalog')
        sigma_e = tomo['tomography/sigma_e'][:]
        mean_R = tomo['multiplicative_bias/mean_R'][:]
        N_eff = tomo['tomography/N_eff'][:]
        lens_counts = tomo['tomography/lens_counts'][:]
        tomo.close()

        area = 4*np.pi*f_sky

        n_eff = N_eff / area
        n_lens = lens_counts / area

        noise = {}
        for i in range(nbin_source):
            noise[(i,SHEAR_SHEAR)] = sigma_e[i]**2 / n_eff[i]
        for i in range(nbin_lens):
            noise[(i,POS_POS)] = 1.0 / n_lens[i]

        warnings.warn("Using unweighted lens samples here")

        return noise, mean_R



    def load_tracers(self, nbin_source, nbin_lens):
        import sacc
        f = self.open_input('photoz_stack')

        tracers = []

        for i in range(nbin_source):
            z = f['n_of_z/source/z'].value
            Nz = f[f'n_of_z/source/bin_{i}'].value
            T = sacc.Tracer(f"LSST source_{i}", b"spin2", z, Nz, exp_sample=b"LSST-source")
            tracers.append(T)

        for i in range(nbin_lens):
            z = f['n_of_z/lens/z'].value
            Nz = f[f'n_of_z/lens/bin_{i}'].value
            T = sacc.Tracer(f"LSST lens_{i}", b"spin0", z, Nz, exp_sample=b"LSST-lens")
            tracers.append(T)

        return tracers

    def fiducial_theory(self, tracers, f_d, f_wl, nbin_source, nbin_lens):
        import pyccl as ccl

        filename = self.get_input('fiducial_cosmology')
        cosmo = ccl.Cosmology.read_yaml(filename)

        # We will need the theory C_ell in a continuum up until
        # the full ell_max, because we will need a weighted sum
        # over the values
        ell_max = f_d[0].fl.lmax
        ell = np.arange(ell_max+1, dtype=int)

        # Convert from SACC tracers (which just store N(z))
        # to CCL tracers (which also have cosmology info in them).
        CTracers = {}

        # Lensing tracers - need to think a little more about
        # the fiducial intrinsic alignment here
        for i in range(nbin_source):
            x = tracers[i]
            tag = ('S', i)
            CTracers[tag] = ccl.WeakLensingTracer(cosmo, dndz=(x.z, x.Nz))
        # Position tracers - even more important to think about fiducial biases
        # here - these will be very very wrong otherwise!
        # Important enough that I'll put in a warning.
        warnings.warn("Not using galaxy bias in fiducial theory density spectra")

        for i in range(nbin_lens):
            x = tracers[i + nbin_source]
            tag = ('P', i) 
            b = np.ones_like(x.z)
            CTracers[tag] = ccl.NumberCountsTracer(cosmo, dndz=(x.z, x.Nz),
                                        has_rsd=False, bias=(x.z,b))

        # Use CCL to actually calculate the C_ell values for the different cases
        theory_cl = {}
        k = SHEAR_SHEAR
        for i in range(nbin_source):
            for j in range(i+1):
                Ti = CTracers[('S',i)]
                Tj = CTracers[('S',j)]
                # The full theory C_ell over the range 0..ellmax
                theory_cl [(i,j,k)] = ccl.angular_cl(cosmo, Ti, Tj, ell)
                

        # The same for the galaxy galaxy-lensing cross-correlation
        k = SHEAR_POS
        for i in range(nbin_source):
            for j in range(nbin_lens):
                Ti = CTracers[('S',i)]
                Tj = CTracers[('P',j)]
                theory_cl [(i,j,k)] = ccl.angular_cl(cosmo, Ti, Tj, ell)

        # And finally for the density correlations
        k = POS_POS
        for i in range(nbin_lens):
            for j in range(i+1):
                Ti = CTracers[('P',i)]
                Tj = CTracers[('P',j)]
                theory_cl [(i,j,k)] = ccl.angular_cl(cosmo, Ti, Tj, ell)

        return theory_cl


    def save_power_spectra(self, tracers, nbin_source, nbin_lens):
        import sacc

        fields = ['corr_type', 'l', 'value', 'i', 'j', 'win']
        output = {f: list() for f in fields}

        q1 = []
        q2 = []
        type = []
        q1s = {
            'CEE': 'E', # Galaxy E-mode
            'CBB': 'B', # Galaxy B-mode
            'Cdd': 'P', # Galaxy position
            'CdE': 'P',
            'CdB': 'P',
            }
        q2s = {
            'CEE': 'E',
            'CBB': 'B',
            'Cdd': 'P',
            'CdE': 'E',
            'CdB': 'B',
            }
        for corr_type in ['CEE', 'CBB', 'Cdd', 'CdE', 'CdB']:
            data = [r for r in self.results if r.corr_type == corr_type]
            for bin_pair_data in data:
                n = len(bin_pair_data.l)
                type += ['Corr' for i in range(n)]
                q1 += [q1s[corr_type] for i in range(n)]
                q2 += [q2s[corr_type] for i in range(n)]

                for f in fields:
                    v = getattr(bin_pair_data, f)
                    if np.isscalar(v):
                        v = [v for i in range(n)]
                    elif f == 'win':
                        v = [sacc.Window(v[0], v[1])]
                    else:
                        v = v.tolist()
                    output[f] += v

        binning = sacc.Binning(type, output['l'], output[
                               'i'], q1, output['j'], q2)
        mean = sacc.MeanVec(output['value'])
        s = sacc.SACC(tracers, binning, mean)
        s.printInfo()
        output_filename = self.get_output("twopoint_data")
        s.saveToHDF(output_filename)


if __name__ == '__main__':
    PipelineStage.main()


