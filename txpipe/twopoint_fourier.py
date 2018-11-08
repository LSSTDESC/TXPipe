from ceci import PipelineStage
from .data_types import MetacalCatalog, TomographyCatalog, RandomsCatalog, YamlFile, SACCFile, DiagnosticMaps, HDFFile, PhotozPDFFile
import numpy as np
import pymaster as nmt
import sacc
import collections
from .utils import choose_pixelization, HealpixScheme, GnomonicPixelScheme, ParallelStatsCalculator

# Using the same convention as in twopoint.py
SHEAR_SHEAR = 0
SHEAR_POS = 1
POS_POS = 2

Measurement = collections.namedtuple(
    'Measurement',
    ['corr_type', 'l', 'value', 'win', 'd_ell', 'i', 'j'])


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
        ('shear_catalog', MetacalCatalog),  # Shear catalog
        ('tomography_catalog', TomographyCatalog),  # Tomography catalog
        ('photoz_stack', HDFFile),  # Photoz stack
        ('diagnostic_maps', DiagnosticMaps),
    ]
    outputs = [
        ('twopoint_data', SACCFile)
    ]

    config_options = {
        "chunk_rows": 10000,
        "mask_threshold": 0.0,
    }

    def run(self):
        config = self.config

        if self.comm:
            self.comm.Barrier()

        self.setup_results()

        # Get some metadata from the tomography file
        tomo_file = self.open_input('tomography_catalog', wrapper=True)
        nbin_source = tomo_file.read_nbin('source')
        nbin_lens = tomo_file.read_nbin('lens')
        tomo_file.close()

        # Generate iterators for shear and tomography catalogs
        cols_shear = ['ra', 'dec', 'mcal_g', 'mcal_flags', 'mcal_mag',
                      'mcal_s2n_r', 'mcal_T']

        # Get the complete list of calculations to be done,
        # for all the three spectra and all the bin pairs.
        # This will be used for parallelization.
        calcs = self.select_calculations(nbin_source, nbin_lens)

        # Generate namaster fields
        pixel_scheme, f_d, f_wl = self.load_maps(nbin_source, nbin_lens)
        print("Loaded maps and converted to NaMaster fields")

        # Binning scheme, currently chosen from the geometry.
        # TODO: set ell binning from config
        ell_bins, d_ell = self.choose_ell_bins(pixel_scheme)
        print("Chosen {} ell bins".format(ell_bins.get_n_bands()))

        # Namaster uses workspaces, which we re-use between
        # bins
        w00, w02, w22 = self.setup_workspaces(pixel_scheme, f_d, f_wl, ell_bins)
        print("Set up workspaces")

        # Run the compute power spectra portion in parallel
        # This splits the calculations among the parallel bins
        # It's not the most optimal way of doing it
        # as it's not dynamic, just a round-robin assignment.
        for i, j, k in self.split_tasks_by_rank(calcs):
            self.compute_power_spectra(
                pixel_scheme, i, j, k, f_wl, f_d, w00, w02, w22, ell_bins, d_ell)

        # Pull all the results together to the master process.
        self.collect_results()

        # Write the collect results out to HDF5.
        if self.rank == 0:
            self.save_power_spectra(nbin_source, nbin_lens)
            print("Saved power spectra")




    def load_maps(self, nbin_source, nbin_lens):
        # Load the various input maps and their metadata
        map_file = self.open_input('diagnostic_maps', wrapper=True)
        pix_info = map_file.read_map_info('mask')

        # Choose pixelization and read mask and systematics maps
        pixel_scheme = choose_pixelization(**pix_info)

        # Load the mask. It should automatically be the same shape as the
        # others, based on how it was originally generated.
        # We remove any pixels that are at or below our threshold (default=0)
        mask = map_file.read_map('mask')
        mask_threshold = self.config['mask_threshold']
        mask[mask <= mask_threshold] = 0      
        mask_sum = mask.sum()

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

        if pixel_scheme.name == 'gnomonic':
            lx = np.radians(pixel_scheme.size_x)
            ly = np.radians(pixel_scheme.size_y)

            # Density for gnomonic maps
            d_fields = [
                nmt.NmtFieldFlat(lx, ly, mask, [d], templates=syst_nc) 
                for d in d_maps]
            
            # Lensing for gnomonic maps
            wl_fields = [
                nmt.NmtFieldFlat(lx, ly, mask, [g1,g2], templates=syst_wl) 
                for g1,g2 in zip(g1_maps, g2_maps)
            ]

        elif pixel_scheme.name == 'healpix':
            # Density for healpix maps
            d_fields = [
                nmt.NmtField(mask, [d], templates=syst_nc) 
                for d in d_maps
            ]
            # Lensing for healpix maps
            wl_fields = [
                nmt.NmtField(mask, [g1, g2], templates=syst_wl) 
                for g1,g2 in zip(g1_maps, g2_maps)
            ]
        else:
            raise ValueError(f"Pixelization scheme {pixel_scheme.name} not supported by NaMaster")

        return pixel_scheme, d_fields, wl_fields


    def collect_results(self):
        if self.comm is None:
            return

        self.results = self.comm.gather(self.results, root=0)

        if self.rank == 0:
            # Concatenate results
            self.results = sum(self.results, [])

            # Order by type, then bin i, then bin j
            order = [b'Cll', b'Cdd', b'Cdl']
            key = lambda r: (order.index(r.corr_type), r.i, r.j)
            self.results = sorted(self.results, key=key)

    def setup_results(self):
        self.results = []

    def choose_ell_bins(self, pixel_scheme):
        if pixel_scheme.name == 'healpix':
            nlb = min(1, int(1. / np.mean(mask)))
            ell_bins = nmt.NmtBin(pixel_scheme.nside, nlb=nlb)
        elif pixel_scheme.name == 'gnomonic':
            lx = np.radians(pixel_scheme.nx * pixel_scheme.pixel_size_x)
            ly = np.radians(pixel_scheme.ny * pixel_scheme.pixel_size_y)
            ell_min = max(2 * np.pi / lx, 2 * np.pi / ly)
            ell_max = min(pixel_scheme.nx * np.pi / lx, pixel_scheme.ny * np.pi / ly)
            d_ell = 2 * ell_min
            n_ell = int((ell_max - ell_min) / d_ell) - 1
            print('n_ell', n_ell)
            print('ell_max',ell_max)
            print('ell_min',ell_min)
            print('d_ell',d_ell)
            l_bpw = np.zeros([2, n_ell])
            l_bpw[0, :] = ell_min + np.arange(n_ell) * d_ell
            l_bpw[1, :] = l_bpw[0, :] + d_ell
            ell_bins = nmt.NmtBinFlat(l_bpw[0, :], l_bpw[1, :])
            # for k,v in locals().items():
            #     print(f"{k}: {v}")

        return ell_bins, d_ell

    def setup_workspaces(self, pixel_scheme, f_d, f_wl, ell_bins):
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

        w02 = workspace_class()
        w02.compute_coupling_matrix(f_d[0], f_wl[0], ell_bins)

        w22 = workspace_class()
        w22.compute_coupling_matrix(f_wl[0], f_wl[0], ell_bins)

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

    def compute_power_spectra(self, pixel_scheme, i, j, k, f_wl, f_d, w00, w02, w22, ell_bins, d_ell):
        # Compute power spectra
        # TODO: now all possible auto- and cross-correlation are computed.
        #      This should be tunable.

        # k refers to the type of measurement we are making

        if k == SHEAR_SHEAR:
            print(i, j, k)
            ls = ell_bins.get_effective_ells()
            # Top-hat window functions
            win = [[np.arange(l - d_ell, l + d_ell),
                    np.ones(np.arange(l - d_ell, l + d_ell).size) / (2 * d_ell)] for l in ls]
            c = self.compute_one_spectrum(
                pixel_scheme, w22, f_wl[i], f_wl[j], ell_bins)
            c_ll = c[0]
            self.results.append(Measurement(
                b'Cll', ls, c_ll, win, d_ell, i, j))

        if k == POS_POS:
            print(i, j, k)
            ls = ell_bins.get_effective_ells()
            win = [[np.arange(l - d_ell, l + d_ell),
                    np.ones(np.arange(l - d_ell, l + d_ell).size) / (2 * d_ell)] for l in ls]
            c = self.compute_one_spectrum(
                pixel_scheme, w00, f_d[i], f_d[j], ell_bins)
            c_dd = c[0]
            self.results.append(Measurement(
                b'Cdd', ls, c_dd, win, d_ell, i, j))

        if k == SHEAR_POS:
            print(i, j, k)
            ls = ell_bins.get_effective_ells()
            win = [[np.arange(l - d_ell, l + d_ell),
                    np.ones(np.arange(l - d_ell, l + d_ell).size) / (2 * d_ell)] for l in ls]
            c = self.compute_one_spectrum(
                pixel_scheme, w02, f_wl[i], f_d[j], ell_bins)
            c_dl = c[0]
            self.results.append(Measurement(
                b'Cdl', ls, c_dl, win, d_ell, i, j))


    def compute_one_spectrum(self, pixel_scheme, w, f1, f2, ell_bins):
        if pixel_scheme.name == 'healpix':
            # correlates two fields f1 and f2 and returns an array of coupled
            # power spectra
            coupled_c_ell = nmt.compute_coupled_cell(f1, f2)
        elif pixel_scheme.name == 'gnomonic':
            coupled_c_ell = nmt.compute_coupled_cell_flat(f1, f2, ell_bins)

        c_ell = w.decouple_cell(coupled_c_ell)
        return c_ell

    def save_power_spectra(self, nbin_source, nbin_lens):
        # c_ll, c_Dl, c_dd, ell_bins, d_ell
        f = self.open_input('photoz_stack')

        tracers = []

        tomo_file = self.open_input('tomography_catalog', wrapper=True)
        nbin_source = tomo_file.read_nbin('source')
        nbin_lens = tomo_file.read_nbin('lens')

        for i in range(nbin_source):
            z = f['n_of_z/source/z'].value
            Nz = f[f'n_of_z/source/bin_{i}'].value
            T = sacc.Tracer(f"LSST source_{i}".encode('ascii'), b"spin0", z, Nz, exp_sample=b"LSST-source")
            tracers.append(T)

        for i in range(nbin_lens):
            z = f['n_of_z/lens/z'].value
            Nz = f[f'n_of_z/lens/bin_{i}'].value
            T = sacc.Tracer(f"LSST lens_{i}".encode('ascii'), b"spin0", z, Nz, exp_sample=b"LSST-lens")
            tracers.append(T)

        fields = ['corr_type', 'l', 'value', 'i', 'j', 'win']
        output = {f: list() for f in fields}

        q1 = []
        q2 = []
        type = []
        for corr_type in [b'Cll', b'Cdd', b'Cdl']:
            data = [r for r in self.results if r.corr_type == corr_type]
            for bin_pair_data in data:
                n = len(bin_pair_data.l)
                type += ['Corr' for i in range(n)]
                if corr_type == b'Cll':
                    q1 += ['S' for i in range(n)]
                    q2 += ['S' for i in range(n)]
                elif corr_type == b'Cdd':
                    q1 += ['P' for i in range(n)]
                    q2 += ['P' for i in range(n)]
                elif corr_type == b'Cdl':
                    q1 += ['P' for i in range(n)]
                    q2 += ['S' for i in range(n)]
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


