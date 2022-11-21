from .base_stage import PipelineStage
from .data_types import (
    ShearCatalog,
    HDFFile,
    FiducialCosmology,
    SACCFile,
    YamlFile,
    MapsFile,
)
from .twopoint_fourier import TXTwoPointFourier
import numpy as np
import warnings
import os
import pickle

# require TJPCov to be in PYTHONPATH
d2r = np.pi / 180
sq_deg_on_sky = 360**2 / np.pi

# Needed changes: 1) ell and theta spacing could be further optimized 2) coupling matrix


class TXFourierGaussianCovariance(PipelineStage):
    """
    Compute a Gaussian Fourier-space covariance with TJPCov using f_sky only

    It imports TJPCov to do so, and runs at a fiducial cosmology.

    This version does not account for mask geometry, only the total sky area
    measured.
    """
    name = "TXFourierGaussianCovariance"
    parallel = False
    do_xi = False

    inputs = [
        ("fiducial_cosmology", FiducialCosmology),  # For the cosmological parameters
        ("twopoint_data_fourier", SACCFile),  # For the binning information
        ("tracer_metadata", HDFFile),  # For metadata
    ]

    outputs = [
        ("summary_statistics_fourier", SACCFile),
    ]

    config_options = {
        "pickled_wigner_transform": "",
        "use_true_shear": False,
        "galaxy_bias": [0.0],
        "gaussian_sims_factor": [1.],
    }

    def run(self):
        import pyccl as ccl
        import sacc
        import tjpcov
        import threadpoolctl

        # read the fiducial cosmology
        cosmo = self.read_cosmology()

        # read binning
        two_point_data = self.read_sacc()

        # read the n(z) and f_sky from the source summary stats
        meta = self.read_number_statistics()

        # Binning choices. The ell binning is a linear piece with all the
        # integer values up to 500 -- these are from firecrown, might need
        # to change later
        meta["ell"] = np.concatenate(
            (
                np.linspace(2, 500 - 1, 500 - 2),
                np.logspace(np.log10(500), np.log10(6e4), 500),
            )
        )

        # Theta binning - log spaced between 1 .. 300 arcmin.
        meta["theta"] = np.logspace(np.log10(1 / 60), np.log10(300.0 / 60), 3000)

        # C_ell covariance
        cov = self.compute_covariance(cosmo, meta, two_point_data=two_point_data)

        self.save_outputs(two_point_data, cov)

    def save_outputs(self, two_point_data, cov):
        filename = self.get_output("summary_statistics_fourier")
        two_point_data.add_covariance(cov)
        two_point_data.save_fits(filename, overwrite=True)

    def read_cosmology(self):
        return self.open_input("fiducial_cosmology", wrapper=True).to_ccl()

    def read_sacc(self):
        import sacc

        f = self.get_input("twopoint_data_fourier")
        two_point_data = sacc.Sacc.load_fits(f)

        # Remove the data types that we won't use for inference
        mask = [
            two_point_data.indices(sacc.standard_types.galaxy_shear_cl_ee),
            two_point_data.indices(sacc.standard_types.galaxy_shearDensity_cl_e),
            two_point_data.indices(sacc.standard_types.galaxy_density_cl),
            # not doing b-modes, do we want to?
        ]
        print("Length before cuts = ", len(two_point_data))
        mask = np.concatenate(mask)
        two_point_data.keep_indices(mask)
        print("Length after cuts = ", len(two_point_data))
        two_point_data.to_canonical_order()

        return two_point_data

    def read_number_statistics(self):
        input_data = self.open_input("tracer_metadata")

        # per-bin quantities
        N_eff = input_data["tracers/N_eff"][:] # for sources
        N_lens = input_data["tracers/lens_counts"][:]
        # For the gaussian sims, lambda = nbar(1+b*delta),
        # instead of lambda = nbar(1+delta), where delta is the density contrast field.
        # Then, we need to scale up the shot noise term for the lenses
        # in the covariance for the same b factor.
        # Here we decrease the number density for this factor, since shot noise term is 1/nbar.
        print('N_lens:', N_lens)
        N_lens = N_lens/np.array(self.config["gaussian_sims_factor"])**2

        if self.config["gaussian_sims_factor"] != [1.]:
            print("ATTENTION: We are dividing N_lens by the gaussian sims factor squared:", np.array(self.config["gaussian_sims_factor"])**2)
            print("Scaled N_lens is:", N_lens)

        if self.config["use_true_shear"]:
            nbins = len(input_data["tracers/sigma_e"][:])
            sigma_e = np.array([0.0 for i in range(nbins)])
        else:
            sigma_e = input_data["tracers/sigma_e"][:]

        # area in sq deg
        area_deg2 = input_data["tracers"].attrs["area"]
        area_unit = input_data["tracers"].attrs["area_unit"]
        if area_unit != "deg^2":
            raise ValueError("Units of area have changed")

        input_data.close()

        # area in steradians and sky fraction
        area = area_deg2 * np.radians(1) ** 2
        area_arcmin2 = area_deg2 * 60**2
        full_sky = 4 * np.pi
        f_sky = area / full_sky

        # Density information from counts
        n_eff = N_eff / area
        n_lens = N_lens / area

        # for printing out only
        n_eff_arcmin = N_eff / area_arcmin2
        n_lens_arcmin = N_lens / area_arcmin2

        # Feedback
        print(f"area =  {area_deg2:.1f} deg^2")
        print(f"f_sky:  {f_sky}")
        print(f"N_eff:  {N_eff} (totals)")
        print(f"N_lens: {N_lens} (totals)")
        print(f"n_eff:  {n_eff} / steradian")
        print(f"     =  {np.around(n_eff_arcmin,2)} / sq arcmin")
        print(f"lens density: {n_lens} / steradian")
        print(f"            = {np.around(n_lens_arcmin,2)} / arcmin")

        # Pass all this back as a dictionary
        meta = {
            "f_sky": f_sky,
            "sigma_e": sigma_e,
            "n_eff": n_eff,
            "n_lens": n_lens,
        }

        return meta

    def get_tracer_info(self, cosmo, meta, two_point_data):
        # Generates CCL tracers from n(z) information in the data file
        import pyccl as ccl

        ccl_tracers = {}
        tracer_noise = {}

        for tracer in two_point_data.tracers:

            # Pull out the integer corresponding to the tracer index
            tracer_dat = two_point_data.get_tracer(tracer)
            nbin = int(two_point_data.tracers[tracer].name.split("_")[1])

            z = tracer_dat.z.copy().flatten()
            nz = tracer_dat.nz.copy().flatten()

            # Identify source tracers and gnerate WeakLensingTracer objects
            # based on them
            if "source" in tracer or "src" in tracer:
                sigma_e = meta["sigma_e"][nbin]
                n_eff = meta["n_eff"][nbin]
                ccl_tracers[tracer] = ccl.WeakLensingTracer(
                        cosmo, dndz=(z, nz)
                    )  # CCL automatically normalizes dNdz
                tracer_noise[tracer] = sigma_e**2 / n_eff

            # or if it is a lens bin then generaete the corresponding
            # CCL tracer class
            elif "lens" in tracer:
                # Get galaxy bias for this sample. Default value = 1.
                if self.config["galaxy_bias"] == [0.0]:
                    b0 = 1
                    print(
                        f"Using galaxy bias = 1 for {tracer} (since you didn't specify any biases)"
                    )
                else:
                    b0 = self.config["galaxy_bias"][nbin]
                    print(f"Using galaxy bias = {b0} for {tracer}")

                b = b0 * np.ones(len(z))
                n_gal = meta["n_lens"][nbin]
                tracer_noise[tracer] = 1 / n_gal
                ccl_tracers[tracer] = ccl.NumberCountsTracer(
                    cosmo, has_rsd=False, dndz=(z, nz), bias=(z, b)
                )

        return ccl_tracers, tracer_noise

    def get_spins(self, tracer_comb):
        # Get the Wigner Transform factors
        WT_factors = {}
        WT_factors["lens", "source"] = (0, 2)
        WT_factors["source", "lens"] = (2, 0)  # same as (0,2)
        WT_factors["source", "source"] = {"plus": (2, 2), "minus": (2, -2)}
        WT_factors["lens", "lens"] = (0, 0)

        tracers = []
        for i in tracer_comb:
            if "lens" in i:
                tracers += ["lens"]
            if "source" in i:
                tracers += ["source"]
        return WT_factors[tuple(tracers)]

    # compute a single covariance matrix for a given pair of C_ell or xi.
    def compute_covariance_block(
        self,
        cosmo,
        meta,
        ell_bins,
        tracer_comb1=None,
        tracer_comb2=None,
        ccl_tracers=None,
        tracer_Noise=None,
        two_point_data=None,
        xi_plus_minus1="plus",
        xi_plus_minus2="plus",
        cache=None,
        WT=None,
    ):
        import pyccl as ccl
        from tjpcov import bin_cov

        cl = {}

        # tracers 1,2,3,4 = tracer_comb1[0], tracer_comb1[1], tracer_comb2[0], tracer_comb2[1]
        # In the dicts below we use '13' to indicate C_ell_(1,3), etc.
        # This index maps to this usae
        reindex = {
            (0, 0): 13,
            (1, 1): 24,
            (0, 1): 14,
            (1, 0): 23,
        }

        ell = meta["ell"]

        # Getting all the C_ell that we need, saving the results in a cache
        # for later re-use
        for i in (0, 1):
            for j in (0, 1):
                local_key = reindex[(i, j)]
                # For symmetric pairs we may have saved the C_ell the other
                # way around, so try both keys
                cache_key1 = (tracer_comb1[i], tracer_comb2[j])
                cache_key2 = (tracer_comb2[j], tracer_comb1[i])
                if cache_key1 in cache:
                    cl[local_key] = cache[cache_key1]
                elif cache_key2 in cache:
                    cl[local_key] = cache[cache_key2]
                else:
                    # If not cached then we must compute
                    t1 = tracer_comb1[i]
                    t2 = tracer_comb2[j]
                    c = ccl.angular_cl(cosmo, ccl_tracers[t1], ccl_tracers[t2], ell)
                    print("Computed C_ell for ", cache_key1)
                    cache[cache_key1] = c
                    cl[local_key] = c

        # The shape noise C_ell values.
        # These are zero for cross bins and as computed earlier for auto bins
        SN = {}
        SN[13] = (
            tracer_Noise[tracer_comb1[0]] if tracer_comb1[0] == tracer_comb2[0] else 0
        )
        SN[24] = (
            tracer_Noise[tracer_comb1[1]] if tracer_comb1[1] == tracer_comb2[1] else 0
        )
        SN[14] = (
            tracer_Noise[tracer_comb1[0]] if tracer_comb1[0] == tracer_comb2[1] else 0
        )
        SN[23] = (
            tracer_Noise[tracer_comb1[1]] if tracer_comb1[1] == tracer_comb2[0] else 0
        )

        # The overall normalization factor at the front of the matrix
        if self.do_xi:
            norm = np.pi * 4 * meta["f_sky"]
        else:
            norm = (2 * ell + 1) * np.gradient(ell) * meta["f_sky"]

        # The coupling is an identity matrix at least when we neglect
        # the mask
        coupling_mat = {}
        coupling_mat[1324] = np.eye(len(ell))
        coupling_mat[1423] = np.eye(len(ell))

        # Initial covariance of C_ell components
        cov = {}
        cov[1324] = np.outer(cl[13] + SN[13], cl[24] + SN[24]) * coupling_mat[1324]
        cov[1423] = np.outer(cl[14] + SN[14], cl[23] + SN[23]) * coupling_mat[1423]

        # for shear-shear components we also add a B-mode contribution
        first_is_shear_shear = ("source" in tracer_comb1[0]) and (
            "source" in tracer_comb1[1]
        )
        second_is_shear_shear = ("source" in tracer_comb2[0]) and (
            "source" in tracer_comb2[1]
        )

        if self.do_xi and (first_is_shear_shear or second_is_shear_shear):
            # this adds the B-mode shape noise contribution.
            # We assume B-mode power (C_ell) is 0
            Bmode_F = 1
            if xi_plus_minus1 != xi_plus_minus2:
                # in the cross term, this contribution is subtracted.
                # eq. 29-31 of https://arxiv.org/pdf/0708.0387.pdf
                Bmode_F = -1
            # below the we multiply zero to maintain the shape of the Cl array, these are effectively
            # B-modes
            cov[1324] += (
                np.outer(cl[13] * 0 + SN[13], cl[24] * 0 + SN[24])
                * coupling_mat[1324]
                * Bmode_F
            )
            cov[1423] += (
                np.outer(cl[14] * 0 + SN[14], cl[23] * 0 + SN[23])
                * coupling_mat[1423]
                * Bmode_F
            )

        cov["final"] = cov[1423] + cov[1324]

        if self.do_xi:
            s1_s2_1 = self.get_spins(tracer_comb1)
            s1_s2_2 = self.get_spins(tracer_comb2)

            # For the shear-shear we have two sets of spins, plus and minus,
            # which are returned as a dict, so we need to pull out the one we need
            # Otherwise it's just specified as a tuple, e.g. (2,0)
            if isinstance(s1_s2_1, dict):
                s1_s2_1 = s1_s2_1[xi_plus_minus1]
            if isinstance(s1_s2_2, dict):
                s1_s2_2 = s1_s2_2[xi_plus_minus2]

            # Use these terms to project the covariance from C_ell to xi(theta)
            th, cov["final"] = WT.projected_covariance2(
                l_cl=ell, s1_s2=s1_s2_1, s1_s2_cross=s1_s2_2, cl_cov=cov["final"]
            )

        # Normalize
        cov["final"] /= norm

        # Put the covariance into bins.
        # This is optional in the case of a C_ell covariance (only if bins in ell are
        # supplied, otherwise the matrix is for each ell value individually).  It is
        # required for real-space covariances since these are always binned.
        if self.do_xi:
            thb, cov["final_b"] = bin_cov(r=th / d2r, r_bins=ell_bins, cov=cov["final"])
        else:
            if ell_bins is not None:
                lb, cov["final_b"] = bin_cov(r=ell, r_bins=ell_bins, cov=cov["final"])
        return cov

    def get_angular_bins(self, cl_sacc):
        # This function replicates `choose_ell_bins` in twopoint_fourier.py
        # TODO: Move this to txpipe/utils/nmt_utils.py
        from .utils.nmt_utils import MyNmtBin

        ell_min = cl_sacc.metadata["binning/ell_min"]
        ell_max = cl_sacc.metadata["binning/ell_max"]
        ell_spacing = cl_sacc.metadata["binning/ell_spacing"]
        n_ell = cl_sacc.metadata["binning/n_ell"]
        edges = np.unique(np.geomspace(ell_min, ell_max, n_ell).astype(int))
        return edges

    def make_wigner_transform(self, meta):
        import threadpoolctl
        from tjpcov import wigner_transform

        path = self.config["pickled_wigner_transform"]
        if path:
            if os.path.exists(path):
                print(f"Loading precomputed wigner transform from {path}")
                WT = pickle.load(open(path, "rb"))
                return WT
            else:
                print(f"Precomputed wigner transform {path} not found.")
                print("Will compute it and then save it.")

        # We don't want to use n processes with n threads each by accident,
        # where n is the number of CPUs we have
        # so for this bit of the code, which uses python's multiprocessing,
        # we limit the number of threads that numpy etc can use.
        # After this is finished this will switch back to allowing all the CPUs
        # to be used for threading instead.
        num_processes = int(os.environ.get("OMP_NUM_THREADS", 1))
        print("Generating Wigner Transform.")
        with threadpoolctl.threadpool_limits(1):
            WT = wigner_transform(
                l=meta["ell"],
                theta=meta["theta"] * d2r,
                s1_s2=[(2, 2), (2, -2), (0, 2), (2, 0), (0, 0)],
                ncpu=num_processes,
            )
            print("Computed Wigner Transform.")

        if path:
            try:
                pickle.dump(WT, open(path, "wb"))
            except OSError:
                sys.stderr.write(f"Could not save wigner transform to {path}")
        return WT

    # compute all the covariances and then combine them into one single giant matrix
    def compute_covariance(self, cosmo, meta, two_point_data):
        from tjpcov import bin_cov

        ccl_tracers, tracer_Noise = self.get_tracer_info(
            cosmo, meta, two_point_data=two_point_data
        )
        # we will loop over all these
        tracer_combs = two_point_data.get_tracer_combinations()
        N2pt = len(tracer_combs)

        WT = self.make_wigner_transform(meta)

        # the bit below is just counting the number of 2pt functions, and accounting
        # for the fact that xi needs to be double counted
        N2pt0 = 0
        if self.do_xi:
            N2pt0 = N2pt
            tracer_combs_temp = tracer_combs.copy()
            for combo in tracer_combs:
                if ("source" in combo[0]) and ("source" in combo[1]):
                    N2pt += 1
                    tracer_combs_temp += [combo]
            tracer_combs = tracer_combs_temp.copy()

        ell_bins = self.get_angular_bins(two_point_data)
        Nell_bins = len(ell_bins) - 1

        cov_full = np.zeros((Nell_bins * N2pt, Nell_bins * N2pt))
        count_xi_pm1 = 0
        count_xi_pm2 = 0
        cl_cache = {}
        xi_pm = [
            [("plus", "plus"), ("plus", "minus")],
            [("minus", "plus"), ("minus", "minus")],
        ]

        print("Total number of 2pt functions:", N2pt)
        print("Number of 2pt functions without xim:", N2pt0)

        # Look through the chunk of matrix, tracer pair by tracer pair
        # Order of the covariance needs to be the cannonical order of saac. For a 3x2pt matrix that is:
        # -galaxy_density_xi
        # -galaxy_shearDensity_xi_t
        # -galaxy_shear_xi_minus
        # -galaxy_shear_xi_plus

        xim_start = N2pt0 - (N2pt - N2pt0)
        xim_end = N2pt0

        for i in range(N2pt):
            tracer_comb1 = tracer_combs[i]

            count_xi_pm1 = 1 if i in range(xim_start, xim_end) else 0

            for j in range(i, N2pt):
                tracer_comb2 = tracer_combs[j]
                print(
                    f"Computing {tracer_comb1} x {tracer_comb2}: chunk ({i},{j}) of ({N2pt},{N2pt})"
                )

                count_xi_pm2 = 1 if j in range(xim_start, xim_end) else 0

                if (
                    self.do_xi
                    and ("source" in tracer_comb1[0] and "source" in tracer_comb1[1])
                    or ("source" in tracer_comb2[0] and "source" in tracer_comb2[1])
                ):
                    cov_ij = self.compute_covariance_block(
                        cosmo,
                        meta,
                        ell_bins,
                        tracer_comb1=tracer_comb1,
                        tracer_comb2=tracer_comb2,
                        ccl_tracers=ccl_tracers,
                        tracer_Noise=tracer_Noise,
                        two_point_data=two_point_data,
                        xi_plus_minus1=xi_pm[count_xi_pm1][count_xi_pm2][0],
                        xi_plus_minus2=xi_pm[count_xi_pm1][count_xi_pm2][1],
                        cache=cl_cache,
                        WT=WT,
                    )

                else:
                    cov_ij = self.compute_covariance_block(
                        cosmo,
                        meta,
                        ell_bins,
                        tracer_comb1=tracer_comb1,
                        tracer_comb2=tracer_comb2,
                        ccl_tracers=ccl_tracers,
                        tracer_Noise=tracer_Noise,
                        two_point_data=two_point_data,
                        cache=cl_cache,
                        WT=WT,
                    )

                # Fill in this chunk of the matrix
                cov_ij = cov_ij["final_b"]
                # Find the right location in the matrix
                start_i = i * Nell_bins
                start_j = j * Nell_bins
                end_i = start_i + Nell_bins
                end_j = start_j + Nell_bins
                # and fill it in, and the transpose component
                cov_full[start_i:end_i, start_j:end_j] = cov_ij
                cov_full[start_j:end_j, start_i:end_i] = cov_ij.T

        try:
            np.linalg.cholesky(cov_full)
        except:
            print(
                "liAnalg.LinAlgError: Covariance not positive definite! "
                "Most likely this is a problem in xim. "
                "We will continue for now but this needs to be fixed."
            )

        return cov_full


class TXRealGaussianCovariance(TXFourierGaussianCovariance):
    """
    Compute a Gaussian real-space covariance with TJPCov using f_sky only

    This version does not account for mask geometry, only the total sky area
    measured.

    It is implemented as a subclass of the Fourier-space version, so also uses
    TJPCov and a fiducial cosmology.
    """
    name = "TXRealGaussianCovariance"
    parallel = False
    do_xi = True

    inputs = [
        ("fiducial_cosmology", FiducialCosmology),  # For the cosmological parameters
        ("twopoint_data_real", SACCFile),  # For the binning information
        ("tracer_metadata", HDFFile),  # For metadata
    ]

    outputs = [
        ("summary_statistics_real", SACCFile),
    ]

    config_options = {
        "min_sep": 2.5,  # arcmin
        "max_sep": 250,
        "nbins": 20,
        "pickled_wigner_transform": "",
        "use_true_shear": False,
        "galaxy_bias": [0.0],
        "gaussian_sims_factor": [1.],
    }

    def run(self):
        super().run()

    def get_angular_bins(self, two_point_data):
        # this should be changed to read from sacc file
        th_arcmin = np.logspace(
            np.log10(self.config["min_sep"]),
            np.log10(self.config["max_sep"]),
            self.config["nbins"] + 1,
        )
        return th_arcmin / 60.0

    def read_sacc(self):
        import sacc

        f = self.get_input("twopoint_data_real")
        two_point_data = sacc.Sacc.load_fits(f)

        mask = [
            two_point_data.indices(sacc.standard_types.galaxy_density_xi),
            two_point_data.indices(sacc.standard_types.galaxy_shearDensity_xi_t),
            two_point_data.indices(sacc.standard_types.galaxy_shear_xi_plus),
            two_point_data.indices(sacc.standard_types.galaxy_shear_xi_minus),
        ]
        mask = np.concatenate(mask)
        two_point_data.keep_indices(mask)

        two_point_data.to_canonical_order()

        return two_point_data

    def save_outputs(self, two_point_data, cov):
        filename = self.get_output("summary_statistics_real")
        two_point_data.add_covariance(cov)
        two_point_data.save_fits(filename, overwrite=True)


class TXFourierTJPCovariance(PipelineStage):
    """
    Compute a Gaussian Fourier-space covariance with TJPCov using mask geometry

    This also calls out to TJPCov, using more recent additions to that package.

    This version, for speed, re-uses the workspace objects cached in the twopoint
    fourier measurement stage.
    """
    name = "TXFourierTJPCovariance"
    do_xi = False

    inputs = [
        ("fiducial_cosmology", FiducialCosmology),  # For the cosmological parameters
        ("twopoint_data_fourier", SACCFile),  # For the binning information
        ("tracer_metadata_yml", YamlFile),  # For metadata
        ("mask", MapsFile),  # For the lens mask
        ("density_maps", MapsFile),  # For the clustering mask
        ("source_maps", MapsFile),  # For the sources masks
    ]

    outputs = [
        ("summary_statistics_fourier", SACCFile),
    ]

    config_options = {"galaxy_bias": [0.0], "IA": 0.5, "cache_dir": "",
                      'cov_type': ["FourierGaussianNmt",
                                   "FourierSSCHaloModel"]}

    def run(self):
        from tjpcov.covariance_calculator import CovarianceCalculator
        import healpy
        # Read the metadata from earlier in the pipeline
        with self.open_input("tracer_metadata_yml", wrapper=True) as f:
            meta = f.content
        # check the units are what we are expecting
        assert meta["area_unit"] == "deg^2"
        assert meta["density_unit"] == "arcmin^{-2}"

        # get the number of bins from metadata
        nbin_lens = meta["nbin_lens"]
        nbin_source = meta["nbin_source"]

        # set up some config options for TJPCov
        tjp_config = {}
        tjp_config["cov_type"] = self.config['cov_type']
        cl_sacc = self.read_sacc()
        tjp_config["sacc_file"] = cl_sacc

        # Get the CCL cosmo object to pass to TJPCov
        with self.open_input("fiducial_cosmology", wrapper=True) as f:
            tjp_config["cosmo"] = f.to_ccl()

        # Choose linear bias values to pass to CCL.
        # Based on configuration option for user.
        if self.config["galaxy_bias"] == [0.0]:
            bias = [1.0 for i in range(nbin_lens)]
        else:
            bias = self.config["galaxy_bias"]
            if not len(bias) == nbin_lens:
                raise ValueError("Wrong number of bias values supplied")

        # Set more TJPCov config options
        for i in range(nbin_lens):
            tjp_config[f"bias_lens_{i}"] = bias[i]
            tjp_config[f"Ngal_lens_{i}"] = meta["lens_density"][i]

        # Load masks
        # For clustering, we follow twopoint_fourier.py:214-219
        with self.open_input("mask", wrapper=True) as f:
            mask = f.read_map("mask")
            mask[mask == healpy.UNSEEN] = 0.0
            if self.rank == 0:
                print("Loaded mask")

        # Set any unseen pixels to zero weight.
        # TODO: unify this code with the code in twopoint_fourier.py

        with self.open_input("density_maps", wrapper=True) as f:
            nbin_lens = f.file["maps"].attrs["nbin_lens"]
            d_maps = [f.read_map(f"delta_{b}") for b in range(nbin_lens)]
            print(f"Loaded {nbin_lens} overdensity maps")

        # twopoint_fourier.py:219
        for d in d_maps:
            mask[d == healpy.UNSEEN] = 0

        # twopoint_fourier.py:225
        with self.open_input("source_maps", wrapper=True) as f:
            lensing_weights = []
            for b in range(nbin_source):
                lw = f.read_map(f"lensing_weight_{b}")
                lw[lw == healpy.UNSEEN] = 0.0
                lensing_weights.append(lw)

            if self.rank == 0:
                print(f"Loaded {nbin_source} lensing weight maps")

        # Following twopoint_fourier.py:197 all clustering maps use this mask
        masks = {f"lens_{i}": mask for i in range(nbin_lens)}
        masks.update({f"source_{i}": lensing_weights[i] for i in range(nbin_source)})
        masks_names = {f"lens_{i}": "mask_lens" for i in range(nbin_lens)}
        masks_names.update(
            {f"source_{i}": f"mask_source_{i}" for i in range(nbin_source)}
        )

        tjp_config[f"mask_file"] = masks
        tjp_config[f"mask_names"] = masks_names

        # Set the TJPCov specific cache:
        if self.config["cache_dir"]:
            tjp_config["outdir"] = self.config["cache_dir"]
        else:
            tjp_config["outdir"] = cl_sacc.metadata.get("cache_dir", ".")

        # MPI
        if self.comm:
            self.comm.Barrier()
            tjp_config["use_mpi"] = True

        # Run TJPCov
        # I shouldn't need to pass the binnning (unless the cache is not set or
        # there is one of the workspaces missing). For generality, I will pass
        # it.
        tjp_config["binning_info"] = self.recover_NmtBin(cl_sacc)

        # For now, since they're only strings, pass the workspaces even if not
        # requested
        workspaces = self.get_workspaces_dict(cl_sacc, masks_names)
        tjp_config["workspaces"] =  workspaces

        # Compute the covariance and save it in the cache folder. This will
        # save also the independent terms.
        calculator = CovarianceCalculator({"tjpcov": tjp_config})
        calculator.create_sacc_cov("summary_statistics_fourier",
                                   save_terms=True)

        # Write the sacc file with the covariance in the TXPipe output folder
        if self.rank == 0:
            cov = calculator.get_covariance()
            cl_sacc.add_covariance(cov)
            output_filename = self.get_output("summary_statistics_fourier")
            cl_sacc.save_fits(output_filename, overwrite=True)
            print("Saved power spectra with its Gaussian covariance")

    def get_workspaces_dict(self, cl_sacc, masks_names):
        # Based on txpipe/twopoint_fourier.py
        # TODO: Move this to txpipe/utils/nmt_utils.py
        cache_dir = cl_sacc.metadata.get("cache_dir", None)
        cache = self.load_workspace_cache(cache_dir)
        if cache == {}:
            return {}

        hashes = {}
        masks_names_list = list(masks_names.values())
        for m in masks_names_list:
            if m not in hashes:
                hashes[m] = cl_sacc.metadata[f"hash/{m}"]
        ell_hash = cl_sacc.metadata["hash/ell_hash"]

        w = {'00': {}, '02': {}, '22': {}}
        # Get the number of data points per Cell
        dtype = cl_sacc.get_data_types()[0]
        trs = cl_sacc.get_tracer_combinations()[0]
        ell_eff, _ = cl_sacc.get_ell_cl(dtype, *trs)
        n_ell = ell_eff.size
        for tr1, tr2 in cl_sacc.get_tracer_combinations():
            # This assumes that the name of the tracers will include 'lens'or
            # 'source'
            s1 = 0 if 'lens' in tr1 else 2
            s2 = 0 if 'lens' in tr2 else 2
            sk = ''.join(sorted(f'{s1}{s2}'))
            m1 = masks_names[tr1]
            m2 = masks_names[tr2]
            key = (m1, m2)
            # checking if the combination m1,m2 or m2,m1 is already done.
            if (key in w[sk]) or (key[::-1] in w[sk]):
                continue
            # Build workspace hash (twopoint_fourier.py:387-395)
            h1 = hashes[m1]
            h2 = hashes[m2]
            cache_key = h1 ^ ell_hash
            if h2 != h1:
                cache_key ^= h2

            w[sk][key] = str(cache.get_path(cache_key))

        return w

    def load_workspace_cache(self, dirname):
        # Copied from twopoint_fourier.py
        from .utils.nmt_utils import WorkspaceCache

        if not dirname:
            if self.rank == 0:
                print("Not using an on-disc cache.  Set cache_dir to use one")
            return {}

        cache = WorkspaceCache(dirname)
        return cache

    def recover_NmtBin(self, cl_sacc):
        # This function replicates `choose_ell_bins` in twopoint_fourier.py
        # TODO: Move this to txpipe/utils/nmt_utils.py
        from .utils.nmt_utils import MyNmtBin

        ell_min = cl_sacc.metadata["binning/ell_min"]
        ell_max = cl_sacc.metadata["binning/ell_max"]
        ell_spacing = cl_sacc.metadata["binning/ell_spacing"]
        n_ell = cl_sacc.metadata["binning/n_ell"]

        ell_bins = MyNmtBin.from_binning_info(ell_min, ell_max, n_ell, ell_spacing)

        # Check that the binning is compatible with the one in the file
        dtype = cl_sacc.get_data_types()[0]
        trs = cl_sacc.get_tracer_combinations()[0]
        ell_eff, _ = cl_sacc.get_ell_cl(dtype, *trs)
        if not np.all(ell_bins.get_effective_ells() == ell_eff):
            print(ell_bins.get_effective_ells())
            print(ell_eff)
            raise ValueError(
                "The reconstructed NmtBin object is not "
                + "compatible with the ells in the sacc file"
            )
        return ell_bins

    def read_sacc(self):
        # Loads a sacc file.
        import sacc

        f = self.get_input("twopoint_data_fourier")
        two_point_data = sacc.Sacc.load_fits(f)

        # Since NaMaster computes all terms (B-modes included). Keep all of
        # them.
        return two_point_data
