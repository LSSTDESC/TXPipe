import numpy as np
import warnings


class Calibrator:
    """
    Base class for classes which calibrate shear measurements.

    These classes do not calculate the calibration factors - that is
    done by the Calculator classes in calibration_tools.py. Instead
    they apply the calibrations after they have been calculated.

    Subclasses do the actual work.  The base classis only useful for
    the load class method, which chooses the correct subclass
    depending on the file it is given.
    """

    def apply(self, g1, g2):
        raise NotImplementedError("Use a subclass of Calibrator not the base")

    @classmethod
    def load(cls, tomo_file, null=False):
        """
        Load a set of Calibrator objects from a tomography file.
        These will be instances of a subclass of Calibrator, depending on the file
        contents. Returns a list of calibrators, one for each the source bins, and
        one for the overall 2D bin.

        Parameters
        ----------
        tomo_file: str
            The name of the tomography file to load
        null:
            Whether to ignore the tomo file type and return null calibrators.
            Useful for "calibrating" true shears

        Returns
        -------
        cals: list
            A set of Calibrators, one per bin
        cal2D: Calibrator
            A single Calibrator for the 2D bin
        """
        import h5py

        # Check the catalog type
        with h5py.File(tomo_file, "r") as f:
            cat_type = f["tomography"].attrs["catalog_type"]

        # choose a subclass based on this
        if null:
            subcls = NullCalibrator
        elif cat_type == "metacal":
            subcls = MetaCalibrator
        elif cat_type == "metadetect":
            subcls = MetaDetectCalibrator
        elif cat_type == "lensfit":
            subcls = LensfitCalibrator
        elif cat_type == "hsc":
            subcls = HSCCalibrator
        else:
            raise ValueError(f"Unknown catalog type {cat_type} in tomo file")

        # load instances of the subclass instead
        return subcls.load(tomo_file)


class NullCalibrator:
    """
    This calibrator subclass does nothing - it's designed
    """

    def __init__(self, mu=None):
        if mu is None:
            self.mu1 = 0.0
            self.mu2 = 0.0
        else:
            self.mu1, self.mu2 = mu

    def apply(self, g1, g2, subtract_mean=False):
        """
        "Calibrate" a set of shears.

        As this is a null calibrator it just returns a copy of them

        Parameters
        ----------
        g1: array or float
            Shear 1 component

        g2: array or float
            Shear 2 component

        subtract_mean: bool
            this is ignored but is here for consistency with other
            classes
        """
        # In the scalar case we just return the same values
        if subtract_mean:
            return g1 - self.mu1, g2 - self.mu2
        elif np.isscalar(g1):
            return g1, g2
        else:
            # array case with no subtraction
            # for consistency with the other calibrators and cases
            # we return copies here
            return g1.copy(), g2.copy()

    @classmethod
    def load(cls, tomo_file):
        """
        Make a set of null calibrators.

        You can use the parent Calibrator.load to automatically
        load the correct subclass.

        Parameters
        ----------
        tomo_file: str
            A tomography file name. Used only to get nbin

        Returns
        -------
        cals: list
            A set of Calibrators, one per bin
        cal2D: NullCalibrator
            A single Calibrator for the 2D bin
        """
        import h5py

        with h5py.File(tomo_file, "r") as f:
            nbin = f["tomography"].attrs["nbin"]

            # Load the mean shear values
            mu1 = f["tomography/mean_e1"][:]
            mu2 = f["tomography/mean_e2"][:]
            mu1_2d = f["tomography/mean_e1_2d"][0]
            mu2_2d = f["tomography/mean_e2_2d"][0]

        return [NullCalibrator([mu1[i], mu2[i]]) for i in range(nbin)], NullCalibrator(
            [mu1_2d, mu2_2d]
        )

    def save(self, group, index):
        pass


class MetaCalibrator(Calibrator):
    def __init__(self, R, S, mu, mu_is_calibrated=True):
        self.R_gamma = R
        self.R_sel = S
        self.R = R + S
        self.Rinv = np.linalg.inv(R)
        if mu_is_calibrated:
            self.mu = np.array(mu)
        else:
            self.mu = self.Rinv @ mu

    def apply(self, g1, g2, subtract_mean=True):
        """
        Calibrate a set of shears using the response matrix and
        mean shear subtraction.

        Parameters
        ----------
        g1: array or float
            Shear 1 component

        g2: array or float
            Shear 2 component

        subtract_mean: bool
            whether to subtract mean shear (default True)
        """
        if not subtract_mean:
            g1, g2 = self.Rinv @ [g1, g2]
        elif np.isscalar(g1):
            g1, g2 = self.Rinv @ [g1, g2] - self.mu
        else:
            g1, g2 = self.Rinv @ [g1, g2] - self.mu[:, np.newaxis]
        return g1, g2

    @classmethod
    def load(cls, tomo_file):
        """
        Make a set of Metacal calibrators using the info in a tomography file.

        You can use the parent Calibrator.load to automatically
        load the correct subclass.

        Parameters
        ----------
        tomo_file: str
            A tomography file name the cal factors are read from

        Returns
        -------
        cals: list
            A set of MetaCalibrators, one per bin
        cal2D: MetaCalibrator
            A single MetaCalibrator for the 2D bin
        """
        import h5py

        with h5py.File(tomo_file, "r") as f:
            # Load the response values
            R = f["response/R_gamma_mean"][:]
            S = f["response/R_S"][:]

            R_2d = f["response/R_gamma_mean_2d"][:]
            S_2d = f["response/R_S_2d"][:]

            # Load the mean shear values
            mu1 = f["tomography/mean_e1"][:]
            mu2 = f["tomography/mean_e2"][:]
            mu1_2d = f["tomography/mean_e1_2d"][0]
            mu2_2d = f["tomography/mean_e2_2d"][0]

        # make the calibrator objects
        n = len(R)
        calibrators = [cls(R[i], S[i], [mu1[i], mu2[i]]) for i in range(n)]
        calibrator2d = cls(R_2d, S_2d, [mu1_2d, mu2_2d])
        return calibrators, calibrator2d

    def save(self, outfile, i):
        if i == "2d":
            outfile["response/R_gamma_mean_2d"][:] = self.R_gamma
            outfile["response/R_S_2d"][:] = self.R_sel
            outfile["tomography/mean_e1_2d"][0] = self.mu[0]
            outfile["tomography/mean_e2_2d"][0] = self.mu[1]
        else:
            outfile["response/R_gamma_mean"][i] = self.R_gamma
            outfile["response/R_S"][i] = self.R_sel
            outfile["tomography/mean_e1"][i] = self.mu[0]
            outfile["tomography/mean_e2"][i] = self.mu[1]


class MetaDetectCalibrator(MetaCalibrator):
    # This is the same as the metacal one except the names
    # we load from are different (because S is not separately calculated)
    def __init__(self, R, mu, mu_is_calibrated=True):
        S = np.zeros_like(R)
        super().__init__(R, S, mu, mu_is_calibrated)

    @classmethod
    def load(cls, tomo_file):
        """
        Make a set of MetaDetect calibrators using the info in a tomography file.

        You can use the parent Calibrator.load to automatically
        load the correct subclass.

        Parameters
        ----------
        tomo_file: str
            A tomography file name the cal factors are read from

        Returns
        -------
        cals: list
            A set of MetaDetectCalibrators, one per bin
        cal2D: MetaCalibrator
            A single MetaDetectCalibrator for the 2D bin
        """
        import h5py

        with h5py.File(tomo_file, "r") as f:
            # Load the response values
            R = f["response/R"][:]
            R_2d = f["response/R_2d"][:]
            n = len(R)

            # Load the mean shear values
            mu1 = f["tomography/mean_e1"][:]
            mu2 = f["tomography/mean_e2"][:]
            mu1_2d = f["tomography/mean_e1_2d"][0]
            mu2_2d = f["tomography/mean_e2_2d"][0]

        # make the calibrator objects
        calibrators = [cls(R[i], [mu1[i], mu2[i]]) for i in range(n)]
        calibrator2d = cls(R_2d, [mu1_2d, mu2_2d])
        return calibrators, calibrator2d

    def save(self, outfile, i):
        if i == "2d":
            outfile["response/R_2d"][:] = self.R
            outfile["tomography/mean_e1_2d"][0] = self.mu[0]
            outfile["tomography/mean_e2_2d"][0] = self.mu[1]
        else:
            outfile["response/R"][i] = self.R
            outfile["tomography/mean_e1"][i] = self.mu[0]
            outfile["tomography/mean_e2"][i] = self.mu[1]


class LensfitCalibrator(Calibrator):
    def __init__(self, K, c):
        self.K = K
        self.c = c

    @classmethod
    def load(cls, tomo_file):
        """
        Make a set of Lensfit calibrators using the info in a tomography file.

        You can use the parent Calibrator.load to automatically
        load the correct subclass.

        Parameters
        ----------
        tomo_file: str
            A tomography file name the cal factors are read from

        Returns
        -------
        cals: list
            A set of LensfitCalibrators, one per bin
        cal2D: LensfitCalibrator
            A single LensfitCalibrator for the 2D bin
        """
        import h5py

        with h5py.File(tomo_file, "r") as f:
            K = f["response/K"][:]
            K_2d = f["response/K_2d"][:]

            C = f["response/C"][:, :]
            C_2d = f["response/C_2d"][:]

        n = len(K)
        calibrators = [cls(K[i], C[i]) for i in range(n)]
        calibrator2d = cls(K_2d, C_2d)
        return calibrators, calibrator2d

    def save(self, outfile, i):
        if i == "2d":
            outfile["response/K_2d"][:] = self.K
            outfile["response/C_2d"][:] = self.c
            outfile["tomography/mean_e1_2d"][0] = self.c[0]
            outfile["tomography/mean_e2_2d"][0] = self.c[1]
        else:
            outfile["response/K"][i] = self.K
            outfile["response/C"][i] = self.c
            outfile["tomography/mean_e1"][i] = self.c[0]
            outfile["tomography/mean_e2"][i] = self.c[1]

    def apply(self, g1, g2, subtract_mean=True):
        """
        For KiDS (see Joachimi et al., 2020, arXiv:2007.01844):
        Appendix C, equation C.4 and C.5
        Correcting for multiplicative shear calibration.
        Additionally optionally correct for residual additive bias (true
        for KiDS-1000 and KV450.)


        The c term is only included if subtract_mean = True

        Parameters
        ----------
        g1: array or float
            Shear 1 component

        g2: array or float
            Shear 2 component

        subtract_mean: bool
            whether to subtract the constant c term (default True)
        """

        if subtract_mean:
            g1 = (g1 - self.c[0]) / (1 + self.K)
            g2 = (g2 - self.c[1]) / (1 + self.K)
        else:
            g1 = g1 / (1 + self.K)
            g2 = g2 / (1 + self.K)
        return g1, g2


class HSCCalibrator(Calibrator):
    def __init__(self, R, K):
        self.R = R
        self.K = K
        print('=============== caibrators/HSCCalibrator.__init__============================')

    @classmethod
    def load(cls, tomo_file):
        """
        Make a set of HSC calibrators using the info in a tomography file.

        You can use the parent Calibrator.load to automatically
        load the correct subclass.

        Parameters
        ----------
        tomo_file: str
            A tomography file name the cal factors are read from

        Returns
        -------
        cals: list
            A set of HSCCalibrators, one per bin
        cal2D: HSCCalibrator
            A single HSCalibrator for the 2D bin
        """
        import h5py
        print('=============== caibrators/HSCCalibrator.load============================')
        with h5py.File(tomo_file, "r") as f:
            K = f["response/K"][:]
            K_2d = f["response/K_2d"][0]

            R = f["response/R_mean"][:]
            R_2d = f["response/R_mean_2d"][0]

        n = len(K)
        calibrators = [cls(R[i], K[i]) for i in range(n)]
        calibrator2d = cls(R_2d, K_2d)
        return calibrators, calibrator2d

    def save(self, outfile, i):
        if i == "2d":
            outfile["response/R_mean_2d"][:] = self.R
            outfile["response/K_2d"][:] = self.K
        else:
            outfile["response/R_mean"][i] = self.R
            outfile["response/K"][i] = self.K
        print('=============== caibrators/HSCCalibrator.save============================')

    def apply(self, g1, g2, c1, c2):
        """
        For HSC (see Mandelbaum et al., 2018, arXiv:1705.06745):
        gi = 1/(1 + mhat)[ei/(2R) - ci] (Eq. (A6) in Mandelbaum et al., 2018)
        R = 1 - < e_rms^2 >w (Eq. (A1) in Mandelbaum et al., 2018)
        mhat = < m >w (Eq. (A2) in Mandelbaum et al., 2018) (we call this K)


        Parameters
        ----------
        g1: array or float
            Shear 1 component

        g2: array or float
            Shear 2 component

        c1: array or float
            Shear 1 additive bias component

        c2: array or float
            Shear 2 additive bias component
        """
        print('=============== caibrators/HSCCalibrator.apply============================')
        g1 = (g1 / (2 * self.R) - c1) / (1 + self.K)
        g2 = (g2 / (2 * self.R) - c2) / (1 + self.K)
        return g1, g2
