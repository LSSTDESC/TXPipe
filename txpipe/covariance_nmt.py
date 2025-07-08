from .base_stage import PipelineStage
from .data_types import ShearCatalog, HDFFile, FiducialCosmology, SACCFile, MapsFile
import numpy as np
import warnings
import os
import pickle
import sys

# require TJPCov to be in PYTHONPATH
d2r = np.pi / 180

# Needed changes: 1) ell and theta spacing could be further optimized 2) coupling matrix


class TXFourierNamasterCovariance(PipelineStage):
    """
    Compute a Gaussian Fourier-space covariance with NaMaster

    This functionality duplicates that of TXFourierTJPCovariance, and we should
    rationalize.
    """
    name = "TXFourierNamasterCovariance"
    do_xi = False

    inputs = [
        ("fiducial_cosmology", FiducialCosmology),  # For the cosmological parameters
        ("twopoint_data_fourier", SACCFile),  # For the binning information
        ("tracer_metadata", HDFFile),  # For metadata
        ("mask", MapsFile),  # For the mask
    ]

    outputs = [
        ("summary_statistics_fourier", SACCFile),
    ]

    config_options = {
        "pickled_wigner_transform": "",
        "use_true_shear": False,
        "scratch_dir": "temp",
        "nside": 1024,
    }

    def run(self):
        import pymaster as nmt
        import h5py
        import healpy as hp
        import scipy
        import pyccl as ccl
        import sacc
        import tjpcov
        import threadpoolctl

        comm = self.comm
        size = self.size
        rank = self.rank

        self.scratch_dir = self.config["scratch_dir"]
        if rank == 0:
            if not os.path.exists(self.scratch_dir):
                os.makedirs(self.scratch_dir)

        # read the fiducial cosmology
        cosmo = self.read_cosmology()

        # read binning
        two_point_data, any_source, any_lens = self.read_sacc()

        # read the n(z) and f_sky from the source summary stats
        meta = self.read_number_statistics(any_source, any_lens)

        # read the mask
        with self.open_input("mask", wrapper=True) as f:
            m = f.read_map("mask")

        nside = self.config["nside"]
        if self.rank == 0:
            print("Read map. Up/downgrading to nside =", nside)
        
        m = hp.ud_grade(m, nside)
        msk = 1 * (m == 1)
        if self.rank == 0:
            print("Apodizing mask with 1 deg scale")
        msk = nmt.mask_apodization(msk, 1.0, apotype="Smooth")

        # get w-workspace
        if rank == 0:
            spinlist = self.get_w_spinlist()
        else:
            spinlist = None

        if comm is None:
            spinlist = spinlist[0]
        else:
            spinlist = comm.scatter(spinlist, root=0)

        self.get_w(msk, spinlist)

        if comm is not None:
            comm.Barrier()

        self.read_w()

        # get cw-workspace
        if rank == 0:
            spinlist = self.get_cw_spinlist()
        else:
            spinlist = None

        if comm is None:
            spinlist = spinlist[0]
        else:
            spinlist = comm.scatter(spinlist, root=0)
        self.get_cw(spinlist)

        if comm is not None:
            comm.Barrier()

        self.read_cw()

        # Binning choices. The ell binning is a linear piece with all the
        # integer values up to 500 -- these are from firecrown, might need
        # to change later
        meta["ell"] = np.concatenate(
            (
                np.linspace(2, 500 - 1, 500 - 2),
                np.logspace(np.log10(500), np.log10(6e4), 500),
            )
        )

        # creating ell arrays for the tjp-block and nmt-block
        tjp_ell_mask = meta["ell"] > nside * 3
        meta["ell_tjp"] = meta["ell"][tjp_ell_mask]
        meta["ell_nmt0"] = np.linspace(0, 3 * nside - 1, 3 * nside)
        meta["ell_nmt"] = meta["ell"][np.invert(tjp_ell_mask)]

        # Theta binning - log spaced between 1 .. 300 arcmin.
        meta["theta"] = np.logspace(np.log10(1 / 60), np.log10(300.0 / 60), 3000)

        if rank == 0:
            diclist, covsize = self.make_mpi_dict(
                cosmo, meta, two_point_data=two_point_data
            )
        else:
            diclist = None
            covsize = None

        if comm is None:
            diclist = diclist[0]
        else:
            diclist = comm.scatter(diclist, root=0)
            covsize = comm.bcast(covsize, root=0)

        self.compute_covariance(cosmo, meta, two_point_data, diclist)

        if self.comm is not None:
            comm.Barrier()

        cov = self.put_together(covsize)
        self.save_outputs(two_point_data, cov)

    def save_outputs(self, two_point_data, cov):
        filename = self.get_output("summary_statistics_fourier")
        two_point_data.add_covariance(cov, overwrite=True)
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

        any_source = any(d.data_type in [sacc.standard_types.galaxy_shear_cl_ee, sacc.standard_types.galaxy_shearDensity_cl_e] for d in two_point_data.data)
        any_lens = any(d.data_type in [sacc.standard_types.galaxy_shearDensity_cl_e, sacc.standard_types.galaxy_density_cl] for d in two_point_data.data)

        return two_point_data, any_source, any_lens

    def read_number_statistics(self, any_source, any_lens):

        # Read the bits of the metadata that we need
        with self.open_input("tracer_metadata") as input_data:
            # area in sq deg
            area_deg2 = input_data["tracers"].attrs["area"]
            area_unit = input_data["tracers"].attrs["area_unit"]

            if any_source:
                N_eff = input_data["tracers/N_eff"][:]
                if self.config["use_true_shear"]:
                    nbins = len(input_data["tracers/sigma_e"][:])
                    sigma_e = np.array([0.0 for i in range(nbins)])
                else:
                    sigma_e = input_data["tracers/sigma_e"][:]

            if any_lens:
                N_lens = input_data["tracers/lens_counts"][:]
                
        if area_unit != "deg^2":
            raise ValueError("Units of area have changed")


        # area in steradians and sky fraction
        area = area_deg2 * np.radians(1) ** 2
        area_arcmin2 = area_deg2 * 60**2
        full_sky = 4 * np.pi
        f_sky = area / full_sky

        # Output dict
        meta = {
            "f_sky": f_sky,
        }


        # General bits
        print(f"area =  {area_deg2:.1f} deg^2")
        print(f"f_sky:  {f_sky}")

        # Lens related bits
        if any_lens:
            n_lens = N_lens / area
            n_lens_arcmin = N_lens / area_arcmin2
            print(f"N_lens: {N_lens} (totals)")
            print(f"lens density: {n_lens} / steradian")
            print(f"            = {np.around(n_lens_arcmin,2)} / arcmin")
            meta["n_lens"] = n_lens

        # Source-related bits
        if any_source:
            n_eff = N_eff / area
            n_eff_arcmin = N_eff / area_arcmin2
            print(f"N_eff:  {N_eff} (totals)")
            print(f"n_eff:  {n_eff} / steradian")
            print(f"     =  {np.around(n_eff_arcmin,2)} / sq arcmin")
            meta["sigma_e"] = sigma_e
            meta["n_eff"] = n_eff



        return meta

    def get_w_spinlist(self):
        size = self.size
        allspins = [(0, 0), (2, 0), (2, 2)]
        spinlist = [[] for i in range(size)]
        for i, spins in enumerate(allspins):
            num = i % size
            spinlist[num].append(spins)
        print("Spin list:", spinlist)
        return spinlist

    def get_w(self, msk, spinlist):
        import pymaster as nmt
        print("Setting up workspaces")
        nside = self.config["nside"]

        self.f0 = nmt.NmtField(msk, [msk], n_iter=0)
        # Spin-2 field
        self.f2 = nmt.NmtField(msk, [msk, msk], n_iter=2)
        # Binning
        self.b = nmt.NmtBin.from_nside_linear(nside, 48)

        # Workspace
        for spins in spinlist:
            s1 = spins[0]
            s2 = spins[1]
            self.w = nmt.NmtWorkspace()
            print("Computing coupling matrix", s1, s2)
            self.w.compute_coupling_matrix(
                getattr(self, f"f{s1}"), getattr(self, f"f{s2}"), self.b
            )
            self.w.write_to(f"{self.scratch_dir}/w{s1}{s2}.fits")

    def read_w(self):
        import pymaster as nmt
        print("Rank", self.rank, "reading workspaces")
        # These are accessed via getattr in compute_covariance_block
        self.w00 = nmt.NmtWorkspace()
        self.w00.read_from(f"{self.scratch_dir}/w00.fits")
        self.w20 = nmt.NmtWorkspace()
        self.w20.read_from(f"{self.scratch_dir}/w20.fits")
        self.w22 = nmt.NmtWorkspace()
        self.w22.read_from(f"{self.scratch_dir}/w22.fits")

    def get_cw_spinlist(self):
        size = self.size
        allspins = [
            (0, 0, 0, 0),
            (0, 0, 2, 0),
            (0, 0, 2, 2),
            (2, 0, 2, 0),
            (2, 0, 2, 2),
            (2, 2, 2, 2),
        ]
        spinlist = [[] for i in range(size)]
        for i, spins in enumerate(allspins):
            num = i % size
            spinlist[num].append(spins)
        return spinlist

    def get_cw(self, spinlist):
        import pymaster as nmt

        for spins in spinlist:
            s1 = spins[0]
            s2 = spins[1]
            s3 = spins[2]
            s4 = spins[3]
            print("Rank", self.rank, "getting covariance workspace", s1, s2, s3, s4)
            cw = nmt.NmtCovarianceWorkspace()
            cw.compute_coupling_coefficients(
                getattr(self, f"f{s1}"),
                getattr(self, f"f{s2}"),
                getattr(self, f"f{s3}"),
                getattr(self, f"f{s4}"),
            )
            cw.write_to(f"{self.scratch_dir}/cw{s1}{s2}{s3}{s4}.fits")

    def read_cw(self):
        import pymaster as nmt

        # These are accessed via getattr in compute_covariance_block
        self.cw0000 = nmt.NmtCovarianceWorkspace()
        self.cw0000.read_from(f"{self.scratch_dir}/cw0000.fits")

        self.cw0020 = nmt.NmtCovarianceWorkspace()
        self.cw0020.read_from(f"{self.scratch_dir}/cw0020.fits")

        self.cw0022 = nmt.NmtCovarianceWorkspace()
        self.cw0022.read_from(f"{self.scratch_dir}/cw0022.fits")

        self.cw2020 = nmt.NmtCovarianceWorkspace()
        self.cw2020.read_from(f"{self.scratch_dir}/cw2020.fits")

        self.cw2022 = nmt.NmtCovarianceWorkspace()
        self.cw2022.read_from(f"{self.scratch_dir}/cw2022.fits")

        self.cw2222 = nmt.NmtCovarianceWorkspace()
        self.cw2222.read_from(f"{self.scratch_dir}/cw2222.fits")

    def get_tracer_info(self, cosmo, meta, two_point_data):
        # Generates CCL tracers from n(z) information in the data file
        import pyccl as ccl
        import scipy

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
                b = 1.0 * np.ones(len(z))  # place holder
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
        from tjpcov.wigner_transform import bin_cov
        import pymaster as nmt
        import scipy

        print("Computing cov block for ", tracer_comb1, tracer_comb2)

        cl = {}

        # tracers 1,2,3,4 = tracer_comb1[0], tracer_comb1[1], tracer_comb2[0], tracer_comb2[1]
        # In the dicts below we use '13' to indicate C_ell_(1,3), etc.
        # This index maps to this usage
        reindex = {
            (0, 0): 13,
            (1, 1): 24,
            (0, 1): 14,
            (1, 0): 23,
        }

        ell = meta["ell"]
        ell_tjp = meta["ell_tjp"]
        ell_nmt = meta["ell_nmt"]  # interpolate nmt covariance to this ell
        ell_nmt0 = meta["ell_nmt0"]  # ell for calculating nmt covariance

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

        # create cl's for tjp and nmt separately
        cl_tjp = {}
        for label in [13, 14, 23, 24]:
            cl_tjp[label] = np.interp(ell_tjp, ell, cl[label])

        cl_nmt = {}
        for label in [13, 14, 23, 24]:
            cl_nmt[label] = np.interp(ell_nmt0, ell, cl[label])

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

        # The tjp part of the covariance

        # The coupling is an identity matrix at least when we neglect
        # the mask
        coupling_mat = {}
        coupling_mat[1324] = np.eye(len(ell_tjp))
        coupling_mat[1423] = np.eye(len(ell_tjp))

        # Initial covariance of C_ell components
        cov_tjp = {}

        cov_tjp[1324] = (
            np.outer(cl_tjp[13] + SN[13], cl_tjp[24] + SN[24]) * coupling_mat[1324]
        )
        cov_tjp[1423] = (
            np.outer(cl_tjp[14] + SN[14], cl_tjp[23] + SN[23]) * coupling_mat[1423]
        )

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
            cov_tjp[1324] += (
                np.outer(cl_tjp[13] * 0 + SN[13], cl_tjp[24] * 0 + SN[24])
                * coupling_mat[1324]
                * Bmode_F
            )
            cov_tjp[1423] += (
                np.outer(cl_tjp[14] * 0 + SN[14], cl_tjp[23] * 0 + SN[23])
                * coupling_mat[1423]
                * Bmode_F
            )

        cov_tjp["final"] = cov_tjp[1423] + cov_tjp[1324]

        # The nmt part of the covariance:
        w1spin, w2spin, nmtspin = self.get_nmt_spin(tracer_comb1, tracer_comb2)
        w1 = getattr(self, "w" + w1spin)
        w2 = getattr(self, "w" + w2spin)
        cw = getattr(self, "cw" + nmtspin)

        shape = self.get_nmt_shape(tracer_comb1, tracer_comb2, meta)

        nmt_input = self.get_nmt_input(tracer_comb1, tracer_comb2, cl_nmt, SN)

        nmt_input_bmode = self.get_nmt_input_bmode(
            tracer_comb1, tracer_comb2, cl_nmt, SN
        )

        nmt_cov = nmt.gaussian_covariance(
            cw,
            int(nmtspin[0]),
            int(nmtspin[1]),
            int(nmtspin[2]),
            int(nmtspin[3]),
            nmt_input[13],
            nmt_input[14],
            nmt_input[23],
            nmt_input[24],
            wa=w1,
            wb=w2,
            coupled=True,
        ).reshape(shape)[:, 0, :, 0]

        # for shear-shear components we also add a B-mode contribution
        first_is_shear_shear = ("source" in tracer_comb1[0]) and (
            "source" in tracer_comb1[1]
        )
        second_is_shear_shear = ("source" in tracer_comb2[0]) and (
            "source" in tracer_comb2[1]
        )

        if self.do_xi and (first_is_shear_shear or second_is_shear_shear):
            Bmode_F = 1
            if xi_plus_minus1 != xi_plus_minus2:
                Bmode_F = -1

            nmt_cov += (
                nmt.gaussian_covariance(
                    cw,
                    int(nmtspin[0]),
                    int(nmtspin[1]),
                    int(nmtspin[2]),
                    int(nmtspin[3]),
                    nmt_input_bmode[13],
                    nmt_input_bmode[14],
                    nmt_input_bmode[23],
                    nmt_input_bmode[24],
                    wa=w1,
                    wb=w2,
                    coupled=True,
                ).reshape(shape)[:, 0, :, 0]
                * Bmode_F
            )

        # Transform nmt part covariance back to the un-normalized
        # tjp part to combine them together
        nmt_cov *= 1.0 / (meta["f_sky"] ** 3)
        norm_nmt = (2 * ell_nmt0 + 1) * np.gradient(ell_nmt0) * meta["f_sky"]
        nmt_cov *= norm_nmt

        # interpolate nmt-part covariance to the original ell
        f = scipy.interpolate.interp2d(ell_nmt0, ell_nmt0, nmt_cov, kind="cubic")
        nmt_cov = f(ell_nmt, ell_nmt)

        # Combining tjp and nmt parts together
        cov = {}
        cov["final"] = np.zeros((len(ell), len(ell)))
        cov["final"][: len(ell_nmt), : len(ell_nmt)] = nmt_cov
        cov["final"][len(ell_nmt) :, len(ell_nmt) :] = cov_tjp["final"]

        # Normalization and/or wigner transform
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
            print(
                "tracer combos:",
                tracer_comb1,
                tracer_comb2,
                " xi+/xi-:",
                xi_plus_minus1,
                xi_plus_minus2,
                " spins:",
                s1_s2_1,
                s1_s2_2,
            )
            th, cov["final"] = WT.projected_covariance(
                ell_cl=ell, s1_s2=s1_s2_1, s1_s2_cross=s1_s2_2, cl_cov=cov["final"]
            )

        # Normalize

        # The overall normalization factor at the front of the matrix
        if self.do_xi:
            norm = np.pi * 4 * meta["f_sky"]
        else:
            norm = (2 * ell + 1) * np.gradient(ell) * meta["f_sky"]

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
        return cov["final_b"]

    def get_nmt_spin(self, tracer_comb1, tracer_comb2):
        s1_s2_1 = self.get_spins(tracer_comb1)
        s1_s2_2 = self.get_spins(tracer_comb2)
        if isinstance(s1_s2_1, dict):
            s1_s2_1 = s1_s2_1["plus"]
        if isinstance(s1_s2_2, dict):
            s1_s2_2 = s1_s2_2["plus"]
        w1spin = str(abs(s1_s2_1[0])) + str(abs(s1_s2_1[1]))
        w2spin = str(abs(s1_s2_2[0])) + str(abs(s1_s2_2[1]))
        nmtspin = (
            str(abs(s1_s2_1[0]))
            + str(abs(s1_s2_1[1]))
            + str(abs(s1_s2_2[0]))
            + str(abs(s1_s2_2[1]))
        )
        return w1spin, w2spin, nmtspin

    def get_nmt_shape(self, tracer_comb1, tracer_comb2, meta):
        nell = len(meta["ell_nmt0"])
        s1_s2_1 = self.get_spins(tracer_comb1)
        s1_s2_2 = self.get_spins(tracer_comb2)
        if isinstance(s1_s2_1, dict):
            s1_s2_1 = s1_s2_1["plus"]
        if isinstance(s1_s2_2, dict):
            s1_s2_2 = s1_s2_2["plus"]
        s1 = (abs(s1_s2_1[0]), abs(s1_s2_1[1]))
        s2 = (abs(s1_s2_2[0]), abs(s1_s2_2[1]))
        dim2 = sum(s1) + 1 if sum(s1) == 0 else sum(s1)
        dim4 = sum(s2) + 1 if sum(s2) == 0 else sum(s2)
        return [nell, dim2, nell, dim4]

    def get_nmt_input(self, tracer_comb1, tracer_comb2, cl_nmt, SN):

        s1_s2_1 = self.get_spins(tracer_comb1)
        s1_s2_2 = self.get_spins(tracer_comb2)
        if isinstance(s1_s2_1, dict):
            s1_s2_1 = s1_s2_1["plus"]
        if isinstance(s1_s2_2, dict):
            s1_s2_2 = s1_s2_2["plus"]
        s1 = abs(s1_s2_1[0])
        s2 = abs(s1_s2_1[1])
        s3 = abs(s1_s2_2[0])
        s4 = abs(s1_s2_2[1])

        cl130 = 0 * cl_nmt[13]
        cl13 = [cl_nmt[13] + SN[13], cl130, cl130, cl130 + SN[13]]

        cl140 = 0 * cl_nmt[14]
        cl14 = [cl_nmt[14] + SN[14], cl140, cl140, cl140 + SN[14]]

        cl230 = 0 * cl_nmt[23]
        cl23 = [cl_nmt[23] + SN[23], cl230, cl230, cl230 + SN[23]]

        cl240 = 0 * cl_nmt[24]
        cl24 = [cl_nmt[24] + SN[24], cl240, cl240, cl240 + SN[24]]

        n13 = 1 if s1 + s3 == 0 else s1 + s3
        n14 = 1 if s1 + s4 == 0 else s1 + s4
        n23 = 1 if s2 + s3 == 0 else s2 + s3
        n24 = 1 if s2 + s4 == 0 else s2 + s4

        nmt_input = {}
        nmt_input[13] = cl13[:n13]
        nmt_input[14] = cl14[:n14]
        nmt_input[23] = cl23[:n23]
        nmt_input[24] = cl24[:n24]

        return nmt_input

    def get_nmt_input_bmode(self, tracer_comb1, tracer_comb2, cl_nmt, SN):

        s1_s2_1 = self.get_spins(tracer_comb1)
        s1_s2_2 = self.get_spins(tracer_comb2)
        if isinstance(s1_s2_1, dict):
            s1_s2_1 = s1_s2_1["plus"]
        if isinstance(s1_s2_2, dict):
            s1_s2_2 = s1_s2_2["plus"]
        s1 = abs(s1_s2_1[0])
        s2 = abs(s1_s2_1[1])
        s3 = abs(s1_s2_2[0])
        s4 = abs(s1_s2_2[1])

        cl130 = 0 * cl_nmt[13]
        cl13 = [cl130 + SN[13], cl130, cl130, cl130 + SN[13]]

        cl140 = 0 * cl_nmt[14]
        cl14 = [cl140 + SN[14], cl140, cl140, cl140 + SN[14]]

        cl230 = 0 * cl_nmt[23]
        cl23 = [cl230 + SN[23], cl230, cl230, cl230 + SN[23]]

        cl240 = 0 * cl_nmt[24]
        cl24 = [cl240 + SN[24], cl240, cl240, cl240 + SN[24]]

        n13 = 1 if s1 + s3 == 0 else s1 + s3
        n14 = 1 if s1 + s4 == 0 else s1 + s4
        n23 = 1 if s2 + s3 == 0 else s2 + s3
        n24 = 1 if s2 + s4 == 0 else s2 + s4

        nmt_input = {}
        nmt_input[13] = cl13[:n13]
        nmt_input[14] = cl14[:n14]
        nmt_input[23] = cl23[:n23]
        nmt_input[24] = cl24[:n24]

        return nmt_input

    def get_angular_bins(self, two_point_data):
        # Assume that the ell binning is the same for each of the bins.
        # This is true in the current pipeline.
        X = two_point_data.get_data_points("galaxy_shear_cl_ee", i=0, j=0)
        # Further assume that the ell ranges are contiguous, so that
        # the max value of one window is the min value of the next.
        # So we just need the lower edges of each bin and then the
        # final maximum value of the last bin
        ell_edges = [x["window"].min for x in X]
        ell_edges.append(X[-1]["window"].max)

        return np.array(ell_edges)

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
            WT = wigner_transform.WignerTransform(
                ell=meta["ell"],
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

    # make a list of list of dictionary, to be scattered to each node. i.e.
    # each node receives a list of dictionary, which contains info about
    # the row and column of the covariance block (i,j), as well as some other
    # info, e.g. tracer combo and xi_pm
    def make_mpi_dict(self, cosmo, meta, two_point_data):
        # we will loop over all these
        tracer_combs = two_point_data.get_tracer_combinations()
        N2pt = len(tracer_combs)

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

        covsize = {"Nell_bins": Nell_bins, "N2pt": N2pt}
        count_xi_pm1 = 0
        count_xi_pm2 = 0
        cl_cache = {}
        xi_pm = [
            [("plus", "plus"), ("plus", "minus")],
            [("minus", "plus"), ("minus", "minus")],
        ]

        xim_start = N2pt0 - (N2pt - N2pt0)
        xim_end = N2pt0

        # create a list of list of dictionaries, and scatter it using MPI
        alldic = []

        # Look through the chunk of matrix, tracer pair by tracer pair
        for i in range(0, N2pt):
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

                    dic = {
                        "ij": (i, j),
                        "tracer_comb1": tracer_comb1,
                        "tracer_comb2": tracer_comb2,
                        "xi_plus_minus1": xi_pm[count_xi_pm1][count_xi_pm2][0],
                        "xi_plus_minus2": xi_pm[count_xi_pm1][count_xi_pm2][1],
                    }
                    alldic.append(dic)

                else:
                    dic = {
                        "ij": (i, j),
                        "tracer_comb1": tracer_comb1,
                        "tracer_comb2": tracer_comb2,
                    }
                    alldic.append(dic)

        size = self.size
        diclist = [[] for i in range(size)]
        for i, dic in enumerate(alldic):
            num = i % size
            diclist[num].append(dic)

        return diclist, covsize

    # compute all the covariances and then combine them into one single giant matrix
    def compute_covariance(self, cosmo, meta, two_point_data, diclist):
        from tjpcov.wigner_transform import bin_cov

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

        # Look through the chunk of matrix, tracer pair by tracer pair
        for num, dic in enumerate(diclist):
            tracer_comb1 = dic["tracer_comb1"]
            tracer_comb2 = dic["tracer_comb2"]
            if (
                self.do_xi
                and ("source" in tracer_comb1[0] and "source" in tracer_comb1[1])
                or ("source" in tracer_comb2[0] and "source" in tracer_comb2[1])
            ):
                # print(tracer_comb1,tracer_comb2,dic['xi_plus_minus1'],dic['xi_plus_minus2'])
                cov_ij = self.compute_covariance_block(
                    cosmo,
                    meta,
                    ell_bins,
                    tracer_comb1=dic["tracer_comb1"],
                    tracer_comb2=dic["tracer_comb2"],
                    ccl_tracers=ccl_tracers,
                    tracer_Noise=tracer_Noise,
                    two_point_data=two_point_data,
                    xi_plus_minus1=dic["xi_plus_minus1"],
                    xi_plus_minus2=dic["xi_plus_minus2"],
                    cache=cl_cache,
                    WT=WT,
                )

            else:
                cov_ij = self.compute_covariance_block(
                    cosmo,
                    meta,
                    ell_bins,
                    tracer_comb1=dic["tracer_comb1"],
                    tracer_comb2=dic["tracer_comb2"],
                    ccl_tracers=ccl_tracers,
                    tracer_Noise=tracer_Noise,
                    two_point_data=two_point_data,
                    cache=cl_cache,
                    WT=WT,
                )
            i = dic["ij"][0]
            j = dic["ij"][1]
            np.savetxt(f"{self.scratch_dir}/cov_{i}_{j}.txt", cov_ij)

    def put_together(self, covsize):
        Nell_bins = covsize["Nell_bins"]
        N2pt = covsize["N2pt"]
        cov_full = np.zeros((Nell_bins * N2pt, Nell_bins * N2pt))

        for i in range(0, N2pt):
            for j in range(i, N2pt):
                # Fill in this chunk of the matrix
                cov_ij = np.loadtxt(f"{self.scratch_dir}/cov_{i}_{j}.txt")
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


class TXRealNamasterCovariance(TXFourierNamasterCovariance):
    """
    Compute a Gaussian real-space covariance with NaMaster

    We don't yet have another stage for this, but should rationalize
    when comparing to TJPCov.
    """
    name = "TXRealNamasterCovariance"
    do_xi = True

    inputs = [
        ("fiducial_cosmology", FiducialCosmology),  # For the cosmological parameters
        ("twopoint_data_real", SACCFile),  # For the binning information
        ("tracer_metadata", HDFFile),  # For metadata
        ("mask", MapsFile),
    ]

    outputs = [
        ("summary_statistics_real", SACCFile),
    ]

    config_options = {
        "pickled_wigner_transform": "",
        "use_true_shear": False,
        "galaxy_bias": [0.0],
        "scratch_dir": "temp",
    }

    def run(self):
        super().run()

    def get_angular_bins(self, two_point_data):
        min_sep = float(two_point_data.metadata["provenance/config/min_sep"])
        max_sep = float(two_point_data.metadata["provenance/config/max_sep"])
        nbins = int(two_point_data.metadata["provenance/config/nbins"])

        th_arcmin = np.logspace(
            np.log10(min_sep),
            np.log10(max_sep),
            nbins + 1,
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

        any_source = any(d.data_type in [sacc.standard_types.galaxy_shearDensity_xi_t, sacc.standard_types.galaxy_shear_xi_plus, sacc.standard_types.galaxy_shear_xi_minus] for d in two_point_data.data)
        any_lens = any(d.data_type in [sacc.standard_types.galaxy_density_xi, sacc.standard_types.galaxy_shearDensity_xi_t] for d in two_point_data.data)

        return two_point_data, any_source, any_lens

    def save_outputs(self, two_point_data, cov):
        filename = self.get_output("summary_statistics_real")
        two_point_data.add_covariance(cov, overwrite=True)
        two_point_data.save_fits(filename, overwrite=True)
