from .base_stage import PipelineStage
from .data_types import (
    TomographyCatalog,
    FiducialCosmology,
    SACCFile,
    MapsFile,
    HDFFile,
    PhotozPDFFile,
    LensingNoiseMaps,
    ClusteringNoiseMaps,
    PNGFile,
    QPNOfZFile,
)
from ceci.config import StageParameter
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


NAMES = {
    SHEAR_SHEAR: "shear-shear",
    SHEAR_POS: "shear-position",
    POS_POS: "position-position",
}

Measurement = collections.namedtuple(
    "Measurement",
    ["corr_type", "l", "value", "noise", "noise_coupled", "win", "i", "j"],
)


class TXTwoPointFourier(PipelineStage):
    """
    Make Fourier space 3x2pt measurements using NaMaster

    This Pipeline Stage computes all auto- and cross-correlations
    for a list of tomographic bins, including all galaxy-galaxy,
    galaxy-shear and shear-shear power spectra. Sources and lenses
    both come from the same shear_catalog and tomography_catalog objects.

    The power spectra are computed after deprojecting a number of
    systematic-dominated modes, represented as input maps.

    In the future we will want to include the following generalizations:

    - TODO: specify which cross-correlations in particular to include (e.g. which bin pairs and which source/lens combinations).
    - TODO: include flags for rejected objects. Is this included in the tomography_catalog?
    - TODO: ell-binning is currently static.
    """

    name = "TXTwoPointFourier"
    inputs = [
        ("shear_photoz_stack", QPNOfZFile),  # Photoz stack
        ("lens_photoz_stack", QPNOfZFile),  # Photoz stack
        ("fiducial_cosmology", FiducialCosmology),  # For the cosmological parameters
        ("tracer_metadata", HDFFile),  # For density info
        ("lens_maps", MapsFile),
        ("source_maps", MapsFile),
        ("density_maps", MapsFile),
        ("mask", MapsFile),
        ("source_noise_maps", LensingNoiseMaps),
        ("lens_noise_maps", ClusteringNoiseMaps),
    ]
    outputs = [("twopoint_data_fourier", SACCFile)]

    config_options = {
        "mask_threshold": StageParameter(float, 0.0, msg="Threshold for masking pixels"),
        "flip_g1": StageParameter(bool, False, msg="Whether to flip the sign of g1"),
        "flip_g2": StageParameter(bool, False, msg="Whether to flip the sign of g2"),
        "cache_dir": StageParameter(str, "./cache/twopoint_fourier", msg="Directory for caching intermediate results"),
        "low_mem": StageParameter(bool, False, msg="Whether to use low memory mode"),
        "deproject_syst_clustering": StageParameter(
            bool, False, msg="Whether to deproject systematic modes from clustering"
        ),
        "systmaps_clustering_dir": StageParameter(str, "", msg="Directory containing systematic maps for clustering"),
        "ell_min": StageParameter(int, 100, msg="Minimum ell value for power spectra"),
        "ell_max": StageParameter(int, 1500, msg="Maximum ell value for power spectra"),
        "n_ell": StageParameter(int, 20, msg="Number of ell bins"),
        "ell_spacing": StageParameter(str, "log", msg="Spacing of ell bins (log or linear)"),
        "true_shear": StageParameter(bool, False, msg="Whether to use true shear values"),
        "analytic_noise": StageParameter(bool, False, msg="Whether to use analytic noise estimates"),
        "gaussian_sims_factor": StageParameter(
            list,
            default=[1.0],
            msg="Factor by which to decrease lens density to account for increased density contrast.",
        ),
        "b0": StageParameter(float, 1.0, msg="Galaxy bias parameter"),
        "do_shear_shear": StageParameter(bool, True, msg="Whether to compute shear-shear power spectra"),
        "do_shear_pos": StageParameter(bool, True, msg="Whether to compute shear-position power spectra"),
        "do_pos_pos": StageParameter(bool, True, msg="Whether to compute position-position power spectra"),
        "compute_theory": StageParameter(bool, True, msg="Whether to compute theory predictions"),
    }

    def run(self):
        import pymaster
        import healpy
        import sacc
        import pyccl
        from .utils.nmt_utils import choose_ell_bins

        config = self.config
        if self.comm:
            self.comm.Barrier()

        self.setup_results()

        # Generate namaster fields
        pixel_scheme, maps, f_sky = self.load_maps()
        if self.rank == 0:
            print("Loaded maps.")

        nbin_source = maps["nbin_source"]
        nbin_lens = maps["nbin_lens"]

        # Do this at the beginning since its fast and sometimes crashes.
        # It is best to avoid losing running time later.
        # Load the n(z) values, which are both saved in the output
        # file alongside the spectra, and then used to calculate the
        # fiducial theory C_ell, which is used in the deprojection calculation
        tracer_sacc = self.load_tracers(nbin_source, nbin_lens)

        if self.config["compute_theory"]:
            with self.open_input("fiducial_cosmology", wrapper=True) as f:
                cosmo = f.to_ccl()

            theory_ell = np.unique(np.geomspace(self.config["ell_min"], self.config["ell_max"], 100).astype(int))
            theory_sacc = theory_3x2pt(cosmo,
                                    tracer_sacc,
                                    bias=self.config["b0"],
                                    smooth=True,
                                    ell_values=theory_ell
            )
        else:
            theory_sacc = sacc.Sacc()

        # Get the complete list of calculations to be done,
        # for all the three spectra and all the bin pairs.
        # This will be used for parallelization.
        calcs = self.select_calculations(nbin_source, nbin_lens)

        # Binning scheme, currently chosen from the geometry.
        ell_bins = choose_ell_bins(**config)

        self.hash_metadata = None  # Filled in make_workspaces
        workspace_cache = self.make_workspaces(maps, calcs, ell_bins)

        # If we are rank zero print out some info
        if self.rank == 0:
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
                pixel_scheme, i, j, k, maps, workspace_cache, ell_bins, theory_sacc, f_sky
            )

        if self.rank == 0:
            print(f"Collecting results together")
        # Pull all the results together to the master process.
        self.collect_results()

        # Write the collect results out to HDF5.
        if self.rank == 0:
            self.save_power_spectra(tracer_sacc, nbin_source, nbin_lens)
            print("Saved power spectra")

    def load_maps(self):
        import pymaster as nmt
        import healpy

        # Load the maps from their files.
        # First the mask
        with self.open_input("mask", wrapper=True) as f:
            info = f.read_map_info("mask")
            area = info["area"]
            f_sky = info["f_sky"]
            mask = f.read_mask(thresh=self.config["mask_threshold"])
            if self.rank == 0:
                print("Loaded mask")
        
        # Using a flat mask as the clustering weight for now, since I need to know
        # how to turn the depth map into a weight
        clustering_weight = mask

        if self.config["do_shear_shear"] or self.config["do_shear_pos"]:
            # Then the shear maps and weights
            with self.open_input("source_maps", wrapper=True) as f:
                nbin_source = f.file["maps"].attrs["nbin_source"]
                g1_maps = [f.read_map(f"g1_{b}") for b in range(nbin_source)]
                g2_maps = [f.read_map(f"g2_{b}") for b in range(nbin_source)]
                lensing_weights = [
                    f.read_map(f"lensing_weight_{b}") for b in range(nbin_source)
                ]
                if self.rank == 0:
                    print(f"Loaded 2 x {nbin_source} shear maps")
                    print(f"Loaded {nbin_source} lensing weight maps")
        else:
            g1_maps, g2_maps, lensing_weights = [], [], []
            nbin_source = 0


        if self.config["do_shear_pos"] or self.config["do_pos_pos"]:
            # And finally the density maps
            with self.open_input("density_maps", wrapper=True) as f:
                nbin_lens = f.file["maps"].attrs["nbin_lens"]
                d_maps = [f.read_map(f"delta_{b}") for b in range(nbin_lens)]
                print(f"Loaded {nbin_lens} overdensity maps")
        else:
            d_maps = []
            nbin_lens = 0


        # Choose pixelization and read mask and systematics maps
        pixel_scheme = choose_pixelization(**info)

        if pixel_scheme.name != "healpix":
            raise ValueError("TXTwoPointFourier can only run on healpix maps")

        if self.rank == 0:
            print(f"Unmasked area = {area:.2f} deg^2, fsky = {f_sky:.2e}")
            print(f"Nside = {pixel_scheme.nside}")

        # Mask any pixels which have the healpix bad value
        for g1, g2, lw in zip(g1_maps, g2_maps, lensing_weights):
            lw[g1 == healpy.UNSEEN] = 0
            lw[g2 == healpy.UNSEEN] = 0
            lw[lw == healpy.UNSEEN] = 0

        # When running on the CosmoDC2 mocks I've found I need to flip
        # both g1 and g2 in order to get both positive galaxy-galaxy lensing
        # and shear-shear spectra.
        if self.config["flip_g1"]:
            for g1 in g1_maps:
                w = np.where(g1 != healpy.UNSEEN)
                g1[w] *= -1

        if self.config["flip_g2"]:
            for g2 in g2_maps:
                w = np.where(g2 != healpy.UNSEEN)
                g2[w] *= -1

        # Load HEALPix systematics maps
        deproject_syst_clustering = self.config["deproject_syst_clustering"]
        if deproject_syst_clustering:
            print("Deprojecting systematics maps for number counts")
            n_systmaps = 0
            s_maps = []
            systmaps_clustering_dir = self.config["systmaps_clustering_dir"]
            systmaps_path = pathlib.Path(systmaps_clustering_dir)
            for systmap in systmaps_path.iterdir():
                try:
                    if systmap.is_file():
                        if pathlib.Path(systmap).suffix != ".fits":
                            print(
                                "Warning: Problem reading systematics map file",
                                systmap,
                                "Not a HEALPix .fits file.",
                            )
                            warnings.warn(
                                "Systematics map file must be a HEALPix .fits file."
                            )
                            print("Ignoring", systmap)
                        else:
                            systmap_file = str(systmap)
                            self.config[
                                f"clustering_deproject_{n_systmaps}"
                            ] = systmap_file  # for provenance
                            print(
                                "Reading clustering systematics map file:", systmap_file
                            )
                            syst_map = healpy.read_map(systmap_file, verbose=False)

                            # normalize map for Namaster
                            # calculate the mean, accounting for case where mask isn't binary
                            unmasked = clustering_weight > 0.0
                            mean = (syst_map[unmasked] * clustering_weight[unmasked]).sum() / clustering_weight[
                                unmasked
                            ].sum()
                            print("Syst map: mean value = ", mean)
                            # subtract the mean rather than normalise by it, as some systematics will have ~0 mean
                            # (other pixels can stay as hp.UNSEEN since they are masked anyway)
                            syst_map[unmasked] -= mean

                            s_maps.append(syst_map)
                            n_systmaps += 1
                except:
                    print("Warning: Problem with systematics map file", systmap)
                    print("Ignoring", systmap)

            print("Number of systematics maps read: ", n_systmaps)
            if n_systmaps == 0:
                print("No systematics maps found. Skipping deprojection.")
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

        # This seems to be an off-by-one bug or ambiguity in namaster v2.
        lmax1 = self.config["ell_max"] - 1

        if self.config["do_shear_pos"] or self.config["do_pos_pos"]:
            if deproject_syst_clustering:
                density_fields = [
                    (nmt.NmtField(clustering_weight, [d], templates=s_maps_nc, n_iter=0, lmax=lmax1)) for d in d_maps
                ]
            else:
                density_fields = [(nmt.NmtField(clustering_weight, [d], n_iter=0, lmax=lmax1)) for d in d_maps]
        else:
            density_fields = []

        if self.config["do_shear_pos"] or self.config["do_shear_shear"]:
            lensing_fields = [
                (nmt.NmtField(lw, [g1, g2], n_iter=0, lmax=lmax1))
                for (lw, g1, g2) in zip(lensing_weights, g1_maps, g2_maps)
            ]
        else:
            lensing_fields = []

        # Collect together all the maps we will output
        maps = {
            "dw": clustering_weight,
            "lw": lensing_weights,
            "g": list(zip(g1_maps, g2_maps)),
            "d": d_maps,
            "lf": lensing_fields,
            "df": density_fields,
            "nbin_source": nbin_source,
            "nbin_lens": nbin_lens,
        }

        return pixel_scheme, maps, f_sky

    def make_workspaces(self, maps, calcs, ell_bins):
        import pymaster as nmt
        from .utils.nmt_utils import WorkspaceCache

        # Make the field object
        if self.rank == 0:
            print("Preparing workspaces")

        # load the cache
        cache = WorkspaceCache(self.config["cache_dir"], low_mem=self.config["low_mem"])

        nbin_source = len(maps["g"])
        nbin_lens = len(maps["d"])

        # empty workspaces
        zero = np.zeros_like(maps["dw"])

        lensing_weights = maps["lw"]
        density_weight = maps["dw"]
        lensing_fields = maps["lf"]
        density_fields = maps["df"]

        ell_hash = array_hash(ell_bins.get_effective_ells())
        hashes = {id(x): array_hash(x) for x in lensing_weights}
        hashes[id(density_weight)] = array_hash(density_weight)

        self.hash_metadata = {
            "ell_hash": ell_hash,
            "mask_lens": hashes[id(density_weight)],
        }

        for i, w in enumerate(lensing_weights):
            self.hash_metadata[f"mask_source_{i}"] = hashes[id(w)]

        # It's an oddity of NaMaster that we
        # need to supply a field object to the workspace even though the
        # coupling matrix only depends on the mask and ell binning.
        # So here we re-use the fields we've made for the actual
        # analysis, but to save the coupling matrices while being sure
        # we're using the right ones we derive a key based on the
        # mask and ell centers.

        # Each lensing field has its own mask
        lensing_fields = [(lw, lf) for lw, lf in zip(lensing_weights, lensing_fields)]

        # The density fields share a mask, so just use the field
        # object for the first one.
        try:
            density_field = (density_weight, density_fields[0])
        # if no density_maps provided, density_fields is an empty list
        except IndexError:
            density_field = (density_weight, None)

        spaces = {}

        for i, j, k in calcs:
            if k == SHEAR_SHEAR:
                w1, f1 = lensing_fields[i]
                w2, f2 = lensing_fields[j]
            elif k == SHEAR_POS:
                w1, f1 = lensing_fields[i]
                w2, f2 = density_field
            else:
                w1, f1 = density_field
                w2, f2 = density_field

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
            space = cache.get(i, j, k, key=key)
            # If not, compute it.  We will save it later
            if space is None:
                print(f"Rank {self.rank} computing coupling matrix {i}, {j}, {k}")
                space = nmt.NmtWorkspace()
                space.compute_coupling_matrix(f1, f2, ell_bins, is_teb=False)
            else:
                print(f"Rank {self.rank} getting coupling matrix {i}, {j}, {k} from cache.")
            # This is a bit awkward - we attach the key to the
            # object to avoid more book-keeping.  This is used inside
            # the workspace cache
            space.txpipe_key = key

            # We now automatically cache everything, non-optionally,
            # but to save memory we don't keep them all loaded
            cache.put(i, j, k, space)

        return cache

    def collect_results(self):
        if self.comm is None:
            return

        self.results = self.comm.gather(self.results, root=0)

        if self.rank == 0:
            # Concatenate results
            self.results = sum(self.results, [])

    def setup_results(self):
        self.results = []

    def select_calculations(self, nbins_source, nbins_lens):
        # Build up a big list of all the calculations we want to
        # perform.  We should probably expose this in the configuration
        # file so you can skip some.
        calcs = []

        # For shear-shear we omit pairs with j>i
        if self.config["do_shear_shear"]:
            k = SHEAR_SHEAR
            for i in range(nbins_source):
                for j in range(i + 1):
                    calcs.append((i, j, k))

        # For shear-position we use all pairs
        if self.config["do_shear_pos"]:
            k = SHEAR_POS
            for i in range(nbins_source):
                for j in range(nbins_lens):
                    calcs.append((i, j, k))

        # For position-position we omit pairs with j>i.
        # We do keep cross-pairs, since even though we may not want to
        # do parameter estimation with them they are useful diagnostics.
        if self.config["do_pos_pos"]:
            k = POS_POS
            for i in range(nbins_lens):
                for j in range(i + 1):
                    calcs.append((i, j, k))

        calcs = [calc for calc in self.split_tasks_by_rank(calcs)]

        return calcs

    def compute_power_spectra(self, pixel_scheme, i, j, k, maps, workspace_cache, ell_bins, cl_theory, f_sky):
        # Compute power spectra
        # TODO: now all possible auto- and cross-correlation are computed.
        #      This should be tunable.

        # k refers to the type of measurement we are making
        import sacc
        import pymaster as nmt
        import healpy

        CEE = sacc.standard_types.galaxy_shear_cl_ee
        CEB = sacc.standard_types.galaxy_shear_cl_eb
        CBE = sacc.standard_types.galaxy_shear_cl_be
        CBB = sacc.standard_types.galaxy_shear_cl_bb
        CdE = sacc.standard_types.galaxy_shearDensity_cl_e
        CdB = sacc.standard_types.galaxy_shearDensity_cl_b
        Cdd = sacc.standard_types.galaxy_density_cl

        type_name = NAMES[k]
        print(f"Process {self.rank} calculating {type_name} spectrum for bin pair {i},{j}")
        sys.stdout.flush()

        if k == SHEAR_SHEAR:
            field_i = self.get_field(maps, i, "shear")
            field_j = self.get_field(maps, j, "shear")
            results_to_use = [(0, CEE), (1, CEB), (2, CBE), (3, CBB)]

        elif k == POS_POS:
            field_i = self.get_field(maps, i, "pos")
            field_j = self.get_field(maps, j, "pos")
            results_to_use = [(0, Cdd)]

        elif k == SHEAR_POS:
            field_i = self.get_field(maps, i, "shear")
            field_j = self.get_field(maps, j, "pos")
            results_to_use = [(0, CdE), (1, CdB)]

        workspace = workspace_cache.get(i, j, k)

        # Load mask for calculation of cl_guess and (optionally) analytic noise calculation.
        with self.open_input("mask", wrapper=True) as f:
            mask = f.read_mask(thresh=self.config["mask_threshold"])
            if self.rank == 0:
                print("Loaded mask")

        cl_guess = nmt.compute_coupled_cell(field_i, field_j) / np.mean(mask * mask)

        if self.config["analytic_noise"]:
            # we are going to subtract the noise afterwards
            c = nmt.compute_full_master(
                field_i,
                field_j,
                ell_bins,
                cl_guess=cl_guess,
                workspace=workspace,
            )
            pcl = nmt.compute_coupled_cell(field_i, field_j)
            c = workspace.decouple_cell(pcl)
            # noise to subtract (already decoupled)
            n_ell, n_ell_coupled = self.compute_noise_analytic(
                i, j, k, maps, f_sky, workspace, mask
            )
            if n_ell is not None:
                c = c - n_ell

            # Writing out the noise for later cross-checks
        else:
            # Get the coupled noise C_ell values to give to the master algorithm
            n_ell, n_ell_coupled = self.compute_noise(
                i, j, k, ell_bins, maps, workspace
            )
            c = nmt.compute_full_master(
                field_i,
                field_j,
                ell_bins,
                cl_noise=n_ell_coupled,
                cl_guess=cl_guess,
                workspace=workspace,
            )

        if n_ell is None:
            n_ell_coupled = np.zeros((c.shape[0], 3 * self.config["nside"]))
            n_ell = np.zeros_like(c)

        def window_pixel(ell, nside):
            r_theta = 1 / (np.sqrt(3.0) * nside)
            x = ell * r_theta
            f = 0.532 + 0.006 * (x - 0.5) ** 2
            y = f * x
            return np.exp(-(y**2) / 2)

        # The binning information - effective (mid) ell values and
        # the window information
        ls = ell_bins.get_effective_ells()

        # Current treatment of the beam is incorrect so commented out for now.
        # Issue has been opened on GitHub (#344)
        c_beam = c# / window_pixel(ls, pixel_scheme.nside) ** 2
        print("c_beam, k, i, j", c_beam, k, i, j)

        # this has shape n_cls, n_bpws, n_cls, lmax+1
        bandpowers = workspace.get_bandpower_windows()

        # Save all the results, skipping things we don't want like EB modes
        for index, name in results_to_use:
            win = bandpowers[index, :, index, :]
            self.results.append(
                Measurement(
                    name,
                    ls,
                    c_beam[index],
                    n_ell[index],
                    n_ell_coupled[index],
                    win,
                    i,
                    j,
                )
            )

    def get_field(self, maps, i, kind):
        # In this class this is very simple, just retrieving a map from
        # the dictionary. But we want to avoid loading the full catalogs
        # for all of the tomographic bins at once in the sub-class that
        # uses them, so we make this a method that it can override to load
        # them dynamically.
        if kind == "shear":
            return maps["lf"][i]
        else:
            return maps["df"][i]

    def compute_noise(self, i, j, k, ell_bins, maps, workspace):

        if self.config["true_shear"] and k == SHEAR_SHEAR:
            # if using true shear (i.e. no shape noise) we do not need to add any noise for the shear-shear case
            return None

        import pymaster as nmt
        import healpy

        # No noise contribution in cross-correlations
        if (i != j) or (k == SHEAR_POS):
            return None, None

        if k == SHEAR_SHEAR:
            noise_maps = self.open_input("source_noise_maps", wrapper=True)
            weight = maps["lw"][i]

        else:
            noise_maps = self.open_input("lens_noise_maps", wrapper=True)
            weight = maps["dw"]

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
            field = nmt.NmtField(w, realization, n_iter=0, lmax=self.config["ell_max"] - 1)
            cl_coupled = nmt.compute_coupled_cell(field, field)

            # Accumulate
            noise_c_ells.append(cl_coupled)

        mean_noise = np.mean(noise_c_ells, axis=0)

        # Since this is the noise on the half maps the real
        # noise C_ell will be (0.5)**2 times the size
        if k == POS_POS:
            mean_noise /= 4

        return workspace.decouple_cell(mean_noise), mean_noise

    def compute_noise_analytic(self, i, j, k, maps, f_sky, workspace, mask):
        # Copied from the HSC branch
        import pymaster
        import healpy as hp

        if (i != j) or (k == SHEAR_POS):
            return None, None

        # This bit only works with healpix maps but it's checked beforehand so that's fine
        if k == SHEAR_SHEAR:
            with self.open_input("source_maps", wrapper=True) as f:
                var_map = f.read_map(f"var_e_{i}")
            print("i, j", i, j)
            var_map[var_map == hp.UNSEEN] = 0.0
            nside = hp.get_nside(var_map)
            pxarea = hp.nside2pixarea(nside)
            n_ls = np.mean(var_map) * pxarea
            # pdb.set_trace()
            print("np.mean(var_map) * pxarea", i, j, k, n_ls)
            n_ell_coupled = np.zeros((4, 3 * nside))
            # Note that N_ell = 0 for ell = 1, 2 for shear
            n_ell_coupled[0, 2:] = n_ls
            n_ell_coupled[3, 2:] = n_ls

        if k == POS_POS:
            ### New method ###
            with self.open_input("lens_maps", wrapper=True) as f:
                lens_map = f.read_map(f"ngal_{i}")
            lens_map[lens_map == hp.UNSEEN] = 0.0
            nside = hp.get_nside(lens_map)
            pxarea = hp.nside2pixarea(nside)
            # compute the mean galaxy counts (in pixels above the mask threshold)
            above_thresh = mask > 0.0
            mu_N = np.sum(lens_map[above_thresh]) / np.sum(mask[above_thresh])
            # coupled N_ells:
            n_ls = pxarea * np.mean(mask) / mu_N
            n_ell_coupled = n_ls * np.ones((1, 3 * nside))

        n_ell = workspace.decouple_cell(n_ell_coupled)
        

        return n_ell, n_ell_coupled

    def load_tomographic_quantities(self, nbin_source, nbin_lens, f_sky):
        # Get lots of bits of metadata from the input file,
        # per tomographic bin.
        metadata = self.open_input("tracer_metadata")
        sigma_e = metadata["tracers/sigma_e"][:]
        N_eff = metadata["tracers/N_eff"][:]
        lens_counts = metadata["tracers/lens_counts"][:]
        metadata.close()

        area = 4 * np.pi * f_sky
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

        sacc_data = sacc.Sacc()

        if nbin_source > 0:
            with self.open_input("shear_photoz_stack", wrapper=True) as f:
                for i in range(nbin_source):
                    z, Nz = f.get_bin_n_of_z(i)
                    sacc_data.add_tracer("NZ", f"source_{i}", z, Nz)

        if nbin_lens > 0:
            with self.open_input("lens_photoz_stack", wrapper=True) as f:
                for i in range(nbin_lens):
                    z, Nz = f.get_bin_n_of_z(i)
                    sacc_data.add_tracer("NZ", f"lens_{i}", z, Nz)

        return sacc_data

    def save_power_spectra(self, tracer_sacc, nbin_source, nbin_lens):
        import sacc
        from sacc.windows import BandpowerWindow

        CEE = sacc.standard_types.galaxy_shear_cl_ee
        CEB = sacc.standard_types.galaxy_shear_cl_eb
        CBE = sacc.standard_types.galaxy_shear_cl_be
        CBB = sacc.standard_types.galaxy_shear_cl_bb
        CdE = sacc.standard_types.galaxy_shearDensity_cl_e
        CdB = sacc.standard_types.galaxy_shearDensity_cl_b
        Cdd = sacc.standard_types.galaxy_density_cl

        S = tracer_sacc.copy()

        # We have saved the results in a big list.  Each entry contains a single
        # bin pair and spectrum type, but many data points at different angles.
        # Here we pull them all out to add to sacc
        for d in self.results:
            tracer1 = (
                f"source_{d.i}"
                if d.corr_type in [CEE, CEB, CBE, CBB, CdE, CdB]
                else f"lens_{d.i}"
            )
            tracer2 = f"source_{d.j}" if d.corr_type in [CEE, CEB, CBE, CBB] else f"lens_{d.j}"


            n = len(d.l)
            # The full set of ell values (unbinned), used to store the
            # bandpowers
            ell_full = np.arange(d.win.shape[1])
            win = BandpowerWindow(ell_full, d.win.T)

            if self.config["gaussian_sims_factor"] != [1.0] and "lens" in tracer2:
                print(
                    "ATTENTION: We are multiplying the measurement and the noise saved in the sacc file by the gaussian sims factor."
                )

                value = d.value * self.config["gaussian_sims_factor"][int(tracer2[-1])]
                noise = d.noise * self.config["gaussian_sims_factor"][int(tracer2[-1])]

                if "lens" in tracer1:
                    value *= self.config["gaussian_sims_factor"][int(tracer1[-1])]
                    noise *= self.config["gaussian_sims_factor"][int(tracer1[-1])]
            else:
                value = d.value
                noise = d.noise

            for i in range(n):
                # We use optional tags i and j here to record the bin indices, as well
                # as in the tracer names, in case it helps to select on them later.
                S.add_data_point(
                    d.corr_type,
                    (tracer1, tracer2),
                    value[i],
                    ell=d.l[i],
                    window=win,
                    i=d.i,
                    j=d.j,
                    n_ell=noise[i],
                    window_ind=i,
                )

            # Add n_ell_coupled to tracer metadata. This will work as far as
            # the coupled noise is constant
            tr = S.tracers[tracer1]
            if (tracer1 == tracer2) and ("n_ell_coupled" not in tr.metadata):

                if self.config["analytic_noise"] is False:
                    # If computed through simulations, it might be better to
                    # take the mean since, for now, only a float can be passed
                    i = 0 if "lens" in tracer1 else 2
                    tr.metadata["n_ell_coupled"] = np.mean(d.noise_coupled)
                else:
                    # Save the last element because the first one is zero for
                    # shear
                    tr.metadata["n_ell_coupled"] = d.noise_coupled[-1]

                if self.config["gaussian_sims_factor"] != [1.0] and "lens" in tracer1:
                    print(tracer1)
                    print(
                        "ATTENTION: We are multiplying the coupled noise saved in the sacc file by the gaussian sims factor."
                    )
                    print("Original noise:", d.noise_coupled)
                    tr.metadata["n_ell_coupled"] *= self.config["gaussian_sims_factor"][int(tracer1[-1])] ** 2

        # Save provenance information
        provenance = self.gather_provenance()
        provenance.update(SACCFile.generate_provenance())
        for key, value in provenance.items():
            if isinstance(value, str) and "\n" in value:
                values = value.split("\n")
                for i, v in enumerate(values):
                    S.metadata[f"provenance/{key}_{i}"] = v
            else:
                S.metadata[f"provenance/{key}"] = value

        # Save hashes needed to recover workspace hash
        for k, v in self.hash_metadata.items():
            S.metadata[f"hash/{k}"] = v

        if self.config["cache_dir"]:
            S.metadata["cache_dir"] = self.config["cache_dir"]

        # Adding binning information to pass later to TJPCov. I shouldn't be
        # necessary, but it is at the moment.
        S.metadata["binning/ell_min"] = self.config["ell_min"]
        S.metadata["binning/ell_max"] = self.config["ell_max"]
        S.metadata["binning/ell_spacing"] = self.config["ell_spacing"]
        S.metadata["binning/n_ell"] = self.config["n_ell"]

        # And we're all done!
        output_filename = self.get_output("twopoint_data_fourier")
        S.save_fits(output_filename, overwrite=True)


class TXTwoPointFourierCatalog(TXTwoPointFourier):
    name = "TXTwoPointFourierCatalog"
    inputs = [
        ("binned_shear_catalog", HDFFile),
        ("binned_lens_catalog", HDFFile),
        ("binned_random_catalog", HDFFile),
        ("binned_random_catalog_sub", HDFFile),
        ("tracer_metadata", HDFFile),
        ("shear_photoz_stack", QPNOfZFile),  # Photoz stack
        ("lens_photoz_stack", QPNOfZFile),  # Photoz stack
        ("mask", MapsFile),
        ("fiducial_cosmology", FiducialCosmology),
        ("source_maps", MapsFile),
        ("lens_maps", MapsFile),
    ]

    config_options_cat = {  # Config specific to catalog-based C_ells
        "use_randoms_clustering": StageParameter(
            bool,
            True,
            msg="Whether to use randoms instead of a map-based mask for clustering"
        ),
        "calc_noise_dp_bias": StageParameter(
            bool,
            False,
            msg="Whether to analytically compute bias in the shot noise component "
                "induced by deprojection."
        ),
        "ell_max_deproj": StageParameter(
            int,
            1500,
            "Maximum multipole to use for mode deprojection."
        )
    }
    config_options = TXTwoPointFourier.config | config_options_cat


    def load_maps(self):
        import pymaster as nmt
        import healpy as hp

        with self.open_input("binned_lens_catalog", wrapper=True) as f:
            nbin_lens = f.file["lens"].attrs["nbin_lens"]
        with self.open_input("binned_shear_catalog", wrapper=True) as f:
            nbin_source = f.file["shear"].attrs["nbin_source"]

        # Load mask for galaxy clustering
        if self.config["use_mask_clustering"] or self.config["deproject_syst_clustering"]:
            with self.open_input("mask", wrapper=True) as f:
                info = f.read_map_info("mask")
                mask_gc = f.read_mask(thresh=self.config["mask_threshold"])
                if self.rank == 0:
                    print("Loaded mask")
        else:
            mask_gc = None

        # Load HEALPix systematics maps if required
        if deproject_syst_clustering:
            print("Deprojecting systematics maps for number counts")
            n_systmaps = 0
            s_maps = []
            systmaps_clustering_dir = self.config["systmaps_clustering_dir"]
            systmaps_path = pathlib.Path(systmaps_clustering_dir)
            for systmap in systmaps_path.iterdir():
                try:
                    if systmap.is_file():
                        if pathlib.Path(systmap).suffix != ".fits":
                            print(
                                "Warning: Problem reading systematics map file",
                                systmap,
                                "Not a HEALPix .fits file.",
                            )
                            warnings.warn("Systematics map file must be a HEALPix .fits file.")
                            print("Ignoring", systmap)
                        else:
                            systmap_file = str(systmap)
                            self.config[f"clustering_deproject_{n_systmaps}"] = systmap_file  # for provenance
                            print("Reading clustering systematics map file:", systmap_file)
                            syst_map = hp.read_map(systmap_file, verbose=False)

                            # normalize map for Namaster
                            # calculate the mean, accounting for case where mask isn't binary
                            unmasked = mask_gc > 0.0
                            mean = (syst_map[unmasked] * mask_gc[unmasked]).sum() / mask_gc[
                                unmasked
                            ].sum()
                            print("Syst map: mean value = ", mean)
                            # subtract the mean rather than normalise by it, as some systematics will have ~0 mean
                            # (other pixels can stay as hp.UNSEEN since they are masked anyway)
                            syst_map[unmasked] -= mean

                            s_maps.append(syst_map)
                            n_systmaps += 1
                except:
                    print("Warning: Problem with systematics map file", systmap)
                    print("Ignoring", systmap)

            print("Number of systematics maps read: ", n_systmaps)
            if n_systmaps == 0:
                print("No systematics maps found. Skipping deprojection.")
                deproject_syst_clustering = False
            else:
                print("Using systematics maps for galaxy number counts.")
                # We assume all systematics maps have the same nside
                nside = hp.pixelfunc.get_nside(syst_map)
                npix = hp.nside2npix(nside)
                # needed for NaMaster:
                s_maps_gc = np.array(s_maps).reshape([n_systmaps, 1, npix])

        else:
            print("Not using systematics maps for deprojection in NaMaster")
            s_maps_gc = None

        maps = {
            "mask": mask_gc,
            "nbin_source": nbin_source,
            "nbin_lens": nbin_lens,
            "systmaps": s_maps_gc,
            "nside": nside
        }

        print(maps)
        
        return mask_gc, s_maps_gc

    def get_field(self, maps, i, kind, get_shears=True):
        # In this subclass we load the catalog field objects dynamically,
        # because they are much larger than the map objects, or can be
        # at full LSST scale
        import pymaster as nmt
        import healpy as hp

        # If I don't do the -1 here I get an error in namaster
        lmax = self.config["ell_max"] - 1

        if kind == "shear":
            if self.rank == 0:
                print("Loading shear catalog for bin", i)
            with self.open_input("binned_shear_catalog") as f:
                group = f[f"shear/bin_{i}"]
                weight = group["weight"][:]
                positions = np.array(
                    [group["ra"][:], group["dec"][:]]
                )
                if get_shears:
                    g1 = group["g1"][:] * (-1) ** self.config["flip_g1"]
                    g2 = group["g2"][:] * (-1) ** self.config["flip_g2"]
                    shear = [g1, g2]
                else:
                    shear = None
            field = nmt.NmtFieldCatalog(positions,
                weight,
                shear,
                lmax=lmax,
                lmax_mask=lmax,
                spin=2,
                lonlat=True
            )
        else:
            if self.rank == 0:
                print("Loading lens catalog for bin", i)
            with self.open_input("binned_lens_catalog") as f:
                group = f[f"lens/bin_{i}"]
                positions = np.array(
                    [group["ra"][:], group["dec"][:]]
                )
                weight = group["weight"][:]
            # Load randoms for this bin if using them
            if self.config["use_randoms_clustering"]:
                with self.open_input("binned_random_catalog") as f:
                    group = f[f"randoms/bin_{i}"][:]
                    pos_rand = np.array([
                        group["ra"][:], group["dec"][:]
                    ])
                    weight_rand = group["weight"][:]
                # Get values of each template at position of each random
                ipix_rand = hp.ang2pix(maps["nside"], *pos_rand, lonlat=True)
                s_maps_nc = s_maps_nc[:, :, ipix_rand].squeeze()
                # No need for mask - set to None
                mask_gc = None
            else:
                pos_rand = None
                weight_rand = None
                # In this case we do need the mask - retrieve from maps
                mask_gc = maps["mask"]
            field = nmt.NmtFieldCatalogClustering(
                positions,
                weight,
                positions_rand=pos_rand,
                weights_rand=weight_rand,
                lmax=lmax,
                mask=mask_gc,
                lmax_mask=lmax,
                lmax_deproj=self.config["ell_max_deproj"],
                spin=0,
                lonlat=True,
                calculate_noise_dp_bias=self.config["calc_noise_dp_bias"]
            )
        
        return field

    def get_uuid(self, kind):
        if kind == "shear":
            with self.open_input("binned_shear_catalog") as f:
                return int(f['provenance'].attrs["uuid"], 16)
        else:
            with self.open_input("binned_lens_catalog") as f:
                return int(f['provenance'].attrs["uuid"], 16)


if __name__ == "__main__":
    PipelineStage.main()
