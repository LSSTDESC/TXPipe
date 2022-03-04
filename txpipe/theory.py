from .base_stage import PipelineStage
from .data_types import FiducialCosmology, SACCFile
import numpy as np


class TXTwoPointTheoryReal(PipelineStage):
    """
    Compute theory in CCL in real space and save to a sacc file.
    """

    name = "TXTwoPointTheoryReal"
    inputs = [
        ("twopoint_data_real", SACCFile),
        ("fiducial_cosmology", FiducialCosmology),  # For example lines
    ]
    outputs = [
        ("twopoint_theory_real", SACCFile),
    ]

    def run(self):
        import sacc

        filename = self.get_input("twopoint_data_real")
        s = sacc.Sacc.load_fits(filename)

        # TODO: when there is a better Cosmology serialization method
        # switch to that
        print("Manually specifying matter_power_spectrum and Neff")
        cosmo = self.open_input("fiducial_cosmology", wrapper=True).to_ccl(
            matter_power_spectrum="halofit", Neff=3.046
        )
        print(cosmo)

        s_theory = self.replace_with_theory_real(s, cosmo)

        # Remove covariance
        s_theory.covariance = None

        # save the output to Sacc file
        s_theory.save_fits(self.get_output("twopoint_theory_real"), overwrite=True)

    def read_nbin(self, s):
        import sacc

        xip = sacc.standard_types.galaxy_shear_xi_plus
        gammat = sacc.standard_types.galaxy_shearDensity_xi_t
        wtheta = sacc.standard_types.galaxy_density_xi

        source_tracers = set()
        for b1, b2 in s.get_tracer_combinations(xip):
            source_tracers.add(b1)
            source_tracers.add(b2)

        lens_tracers = set()
        if wtheta in s.get_data_types():
            for b1, b2 in s.get_tracer_combinations(wtheta):
                lens_tracers.add(b1)
                lens_tracers.add(b2)
        else:
            for sbin, lbin in s.get_tracer_combinations(gammat):
                if lbin not in lens_tracers:
                    lens_tracers.add(lbin)


        return len(source_tracers), len(lens_tracers)

    def get_ccl_tracers(self, s, cosmo, smooth=False):

        # ccl tracers object
        import pyccl

        tracers = {}

        nbin_source, nbin_lens = self.read_nbin(s)

        # Make the lensing tracers
        for i in range(nbin_source):
            name = f"source_{i}"
            Ti = s.get_tracer(name)
            nz = smooth_nz(Ti.nz) if smooth else Ti.nz
            print("smooth:", smooth)
            # Convert to CCL form
            try:
                tracers[name] = pyccl.WeakLensingTracer(cosmo, (Ti.z, nz))
            except pyccl.errors.CCLError:
                print(
                    "To avoid a CCL_ERROR_INTEG we reduce the number of points in the nz by half in source bin %d"
                    % i
                )
                tracers[name] = pyccl.WeakLensingTracer(cosmo, (Ti.z[::2], nz[::2]))

        # And the clustering tracers
        for i in range(nbin_lens):
            name = f"lens_{i}"
            Ti = s.get_tracer(name)
            nz = smooth_nz(Ti.nz) if smooth else Ti.nz

            # Convert to CCL form
            tracers[name] = pyccl.NumberCountsTracer(
                cosmo, has_rsd=False, dndz=(Ti.z, nz), bias=(Ti.z, np.ones_like(Ti.z))
            )

        return tracers

    def replace_with_theory_real(self, s, cosmo):

        import pyccl

        nbin_source, nbin_lens = self.read_nbin(s)
        ell = np.unique(np.logspace(np.log10(2), 5, 400).astype(int))
        tracers = self.get_ccl_tracers(s, cosmo)

        if "galaxy_shear_xi_plus" in s.get_data_types():
            for i in range(nbin_source):
                for j in range(i + 1):
                    print(f"Computing theory lensing-lensing ({i},{j})")

                    # compute theory
                    print(tracers[f"source_{i}"], tracers[f"source_{j}"])
                    cl = pyccl.angular_cl(
                        cosmo, tracers[f"source_{i}"], tracers[f"source_{j}"], ell
                    )
                    theta, *_ = s.get_theta_xi(
                        "galaxy_shear_xi_plus", f"source_{i}", f"source_{j}"
                    )
                    # CCL inputs theta in degrees. 
                    xip = pyccl.correlation(cosmo, ell, cl, theta / 60, corr_type="L+")
                    xim = pyccl.correlation(cosmo, ell, cl, theta / 60, corr_type="L-")

                    # replace data values in the sacc object for the theory ones
                    ind_xip = s.indices(
                        "galaxy_shear_xi_plus", (f"source_{i}", f"source_{j}")
                    )
                    ind_xim = s.indices(
                        "galaxy_shear_xi_minus", (f"source_{i}", f"source_{j}")
                    )
                    for p, q in enumerate(ind_xip):
                        s.data[q].value = xip[p]
                    for p, q in enumerate(ind_xim):
                        s.data[q].value = xim[p]

        
        if "galaxy_density_xi" in s.get_data_types():
            for i in range(nbin_lens):
                for j in range(i + 1):
                    print(f"Computing theory density-density ({i},{j})")

                    # compute theory
                    cl = pyccl.angular_cl(
                        cosmo, tracers[f"lens_{i}"], tracers[f"lens_{j}"], ell
                    )
                    theta, *_ = s.get_theta_xi(
                        "galaxy_density_xi", f"lens_{i}", f"lens_{j}"
                    )
                    wtheta = pyccl.correlation(cosmo, ell, cl, theta / 60, corr_type="GG")

                    # replace data values in the sacc object for the theory ones
                    ind = s.indices("galaxy_density_xi", (f"lens_{i}", f"lens_{j}"))
                    for p, q in enumerate(ind):
                        s.data[q].value = wtheta[p]

        print (s.get_data_types())
        if "galaxy_shearDensity_xi_t" in s.get_data_types():
            for i in range(nbin_source):
                for j in range(nbin_lens):
                    print(f"Computing theory lensing-density (S{i},L{j})")

                    # compute theory
                    cl = pyccl.angular_cl(
                        cosmo, tracers[f"source_{i}"], tracers[f"lens_{j}"], ell
                    )
                    theta, *_ = s.get_theta_xi(
                        "galaxy_shearDensity_xi_t", f"source_{i}", f"lens_{j}"
                    )
                    gt = pyccl.correlation(cosmo, ell, cl, theta / 60, corr_type="GL")

                    ind = s.indices(
                        "galaxy_shearDensity_xi_t", (f"source_{i}", f"lens_{j}")
                    )
                    for p, q in enumerate(ind):
                        s.data[q].value = gt[p]

            return s


class TXTwoPointTheoryFourier(TXTwoPointTheoryReal):
    """
    Compute theory from CCL in Fourier space and save to a sacc file.
    """

    name = "TXTwoPointTheoryFourier"
    inputs = [
        ("twopoint_data_fourier", SACCFile),
        ("fiducial_cosmology", FiducialCosmology),  # For example lines
    ]
    outputs = [
        ("twopoint_theory_fourier", SACCFile),
    ]

    def run(self):
        import sacc

        filename = self.get_input("twopoint_data_fourier")
        s = sacc.Sacc.load_fits(filename)

        # TODO: when there is a better Cosmology serialization method
        # switch to that
        print("Manually specifying matter_power_spectrum and Neff")
        cosmo = self.open_input("fiducial_cosmology", wrapper=True).to_ccl(
            matter_power_spectrum="halofit", Neff=3.046
        )
        print(cosmo)

        s_theory = self.replace_with_theory_fourier(s, cosmo)

        # Remove covariance
        s_theory.covariance = None

        # save the output to Sacc file
        s_theory.save_fits(self.get_output("twopoint_theory_fourier"), overwrite=True)

    def read_nbin(self, s):
        import sacc

        cl_ee = sacc.standard_types.galaxy_shear_cl_ee
        cl_density = sacc.standard_types.galaxy_density_cl

        source_tracers = set()
        for b1, b2 in s.get_tracer_combinations(cl_ee):
            source_tracers.add(b1)
            source_tracers.add(b2)

        lens_tracers = set()
        for b1, b2 in s.get_tracer_combinations(cl_density):
            lens_tracers.add(b1)
            lens_tracers.add(b2)

        return len(source_tracers), len(lens_tracers)

    def replace_with_theory_fourier(self, s, cosmo):

        import pyccl

        nbin_source, nbin_lens = self.read_nbin(s)
        tracers = self.get_ccl_tracers(s, cosmo)

        data_types = s.get_data_types()
        if "galaxy_shearDensity_cl_b" in data_types:
            # Remove galaxy_shearDensity_cl_b measurement values
            ind_b = s.indices("galaxy_shearDensity_cl_b")
            s.remove_indices(ind_b)
        if "galaxy_shear_cl_bb" in data_types:
            # Remove galaxy_shear_cl_bb  measurement values
            ind_bb = s.indices("galaxy_shear_cl_bb")
            s.remove_indices(ind_bb)

        for i in range(nbin_source):
            for j in range(i + 1):
                print(f"Computing theory lensing-lensing ({i},{j})")

                # compute theory
                print(tracers[f"source_{i}"], tracers[f"source_{j}"])
                ell, *_ = s.get_ell_cl(
                    "galaxy_shear_cl_ee", f"source_{i}", f"source_{j}"
                )
                cl = pyccl.angular_cl(
                    cosmo, tracers[f"source_{i}"], tracers[f"source_{j}"], ell
                )

                # replace data values in the sacc object for the theory ones
                ind = s.indices("galaxy_shear_cl_ee", (f"source_{i}", f"source_{j}"))
                for p, q in enumerate(ind):
                    s.data[q].value = cl[p]

        for i in range(nbin_lens):
            for j in range(i + 1):
                print(f"Computing theory density-density ({i},{j})")

                # compute theory
                ell, *_ = s.get_ell_cl("galaxy_density_cl", f"lens_{i}", f"lens_{j}")
                cl = pyccl.angular_cl(
                    cosmo, tracers[f"lens_{i}"], tracers[f"lens_{j}"], ell
                )

                # replace data values in the sacc object for the theory ones
                ind = s.indices("galaxy_density_cl", (f"lens_{i}", f"lens_{j}"))
                for p, q in enumerate(ind):
                    s.data[q].value = cl[p]

        for i in range(nbin_source):

            for j in range(nbin_lens):
                print(f"Computing theory lensing-density (S{i},L{j})")

                # compute theory
                ell, *_ = s.get_ell_cl(
                    "galaxy_shearDensity_cl_e", f"source_{i}", f"lens_{j}"
                )
                cl = pyccl.angular_cl(
                    cosmo, tracers[f"source_{i}"], tracers[f"lens_{j}"], ell
                )

                # replace data values in the sacc object for the theory ones
                ind = s.indices(
                    "galaxy_shearDensity_cl_e", (f"source_{i}", f"lens_{j}")
                )
                for p, q in enumerate(ind):
                    s.data[q].value = cl[p]

        return s
