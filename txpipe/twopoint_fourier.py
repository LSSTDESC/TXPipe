from ceci import PipelineStage
from .data_types import MetacalCatalog, TomographyCatalog, RandomsCatalog, YamlFile, SACCFile, DiagnosticMaps, HDFFile, PhotozPDFFile
import numpy as np
import collections
from .utils import choose_pixelization, HealpixScheme, GnomonicPixelScheme, ParallelStatsCalculator
import sys

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
    ]
    outputs = [
        ('twopoint_data', SACCFile)
    ]

    config_options = {
        "mask_threshold": 0.0,
        "bandwidth": 0,
        "apodization_size": 3.0,
        "apodization_type": "C1",  #"C1", "C2", or "Smooth"
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

        # Binning scheme, currently chosen from the geometry.
        # TODO: set ell binning from config
        ell_bins = self.choose_ell_bins(pixel_scheme, f_sky)
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
                pixel_scheme, i, j, k, f_wl, f_d, w00, w02, w22, ell_bins)

        if self.rank==0:
            print(f"Collecting results together")
        # Pull all the results together to the master process.
        self.collect_results()

        # Write the collect results out to HDF5.
        if self.rank == 0:
            self.save_power_spectra(nbin_source, nbin_lens)
            print("Saved power spectra")




    def load_maps(self):
        import pymaster as nmt

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

    def compute_power_spectra(self, pixel_scheme, i, j, k, f_wl, f_d, w00, w02, w22, ell_bins):
        # Compute power spectra
        # TODO: now all possible auto- and cross-correlation are computed.
        #      This should be tunable.
        # TODO: Fix window functions, and how to save them.

        # k refers to the type of measurement we are making
        type_name = NAMES[k]
        print(f"Process {self.rank} calcluating {type_name} spectrum for bin pair {i},{i}")
        sys.stdout.flush()
        if k == SHEAR_SHEAR:
            ls = ell_bins.get_effective_ells()
            # Top-hat window functions
            win = [ell_bins.get_window(b) for b,l  in enumerate(ls)]
            c = self.compute_one_spectrum(
                pixel_scheme, w22, f_wl[i], f_wl[j], ell_bins)
            c_EE = c[0]
            c_BB = c[3]
            self.results.append(Measurement('CEE', ls, c_EE, win, i, j))
            self.results.append(Measurement('CBB', ls, c_BB, win, i, j))

        if k == POS_POS:
            ls = ell_bins.get_effective_ells()
            win = [ell_bins.get_window(b) for b,l  in enumerate(ls)]
            c = self.compute_one_spectrum(
                pixel_scheme, w00, f_d[i], f_d[j], ell_bins)
            c_dd = c[0]
            self.results.append(Measurement('Cdd', ls, c_dd, win, i, j))

        if k == SHEAR_POS:
            ls = ell_bins.get_effective_ells()
            win = [ell_bins.get_window(b) for b,l  in enumerate(ls)]
            c = self.compute_one_spectrum(
                pixel_scheme, w02, f_wl[i], f_d[j], ell_bins)
            c_dE = c[0]
            c_dB = c[1]
            self.results.append(Measurement('CdE', ls, c_dE, win, i, j))
            self.results.append(Measurement('CdB', ls, c_dB, win, i, j))


    def compute_one_spectrum(self, pixel_scheme, w, f1, f2, ell_bins):
        import pymaster as nmt
        if pixel_scheme.name == 'healpix':
            # correlates two fields f1 and f2 and returns an array of coupled
            # power spectra
            coupled_c_ell = nmt.compute_coupled_cell(f1, f2)
        elif pixel_scheme.name == 'gnomonic':
            coupled_c_ell = nmt.compute_coupled_cell_flat(f1, f2, ell_bins)

        c_ell = w.decouple_cell(coupled_c_ell)
        return c_ell

    def save_power_spectra(self, nbin_source, nbin_lens):
        import sacc
        f = self.open_input('photoz_stack')

        tracers = []

        for i in range(nbin_source):
            z = f['n_of_z/source/z'].value
            Nz = f[f'n_of_z/source/bin_{i}'].value
            T = sacc.Tracer(f"LSST source_{i}", b"spin0", z, Nz, exp_sample=b"LSST-source")
            tracers.append(T)

        for i in range(nbin_lens):
            z = f['n_of_z/lens/z'].value
            Nz = f[f'n_of_z/lens/bin_{i}'].value
            T = sacc.Tracer(f"LSST lens_{i}", b"spin0", z, Nz, exp_sample=b"LSST-lens")
            tracers.append(T)

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


