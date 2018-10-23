from ceci import PipelineStage
from .data_types import MetacalCatalog, HDFFile, YamlFile, SACCFile




class TXFourierGaussianCovariance(PipelineStage):
    name='TXFourierGaussianCovariance'

    inputs = [
        ('fiducial_cosmology', YamlFile),  # For the cosmological parameters
        ('photoz_stack', HDFFile),  # For the n(z)
        ('twopoint_data', SACCFile), # For the binning information,  Re
    ]

    outputs = [
        ('covariance', HDFFile),
    ]

    config_options = {
    }


    def run(self):
        # read the fiducial cosmology
        cosmo = self.read_cosmology()

        # read binning
        binning_info = self.read_binning()

        # read the n(z) and f_sky from the source summary stats
        nz, sigma_e, n_eff, fsky = self.read_number_statistics()

        # calculate the overall total C_ell values, including noise
        theory_c_ell = self.compute_theory_c_ell(cosmo, nz)
        noise_c_ell = self.compute_noise_c_ell(n_eff, sigma_e, fsky)

        # compute covariance
        C = self.compute_covariance(binning_info, theory_c_ell, noise_c_ell)

    def read_cosmology(self):
        import pyccl as ccl
        filename = self.get_input('fiducial_cosmology')
        params = ccl.Parameters.read_yaml(filename)
        cosmo = ccl.Cosmology(params)
        return cosmo

    def read_binning(self):
        import sacc
        f = self.get_input('twopoint_data')
        sacc_data = sacc.SACC.loadFromHDF(f)

        codes = {
            'xip': ('+','+'),
            'xim': ('-','-'),
            'wtheta': ('P','P'),
            'gammat': ('S','S'),
        }

        binning = {quant: {} for quant in codes}

        for quant, (code1,code2) in codes.items():
            bin_pairs = sacc_data.binning.get_bin_pairs(code1,code2)
            for (b1, b2) in bin_pairs:
                angle = sacc_data.binning.get_angle(code1, code2, b1, b2)
                binning[quant][(b1,b2)] = angle

        return binning

    def read_number_statistics(self, binning):
        input_data = self.open_input('photoz_stack')

        ...

        return nz, n_eff, sigma_e, fsky

    def compute_theory_c_ell(self, cosmo, nz):
        # Turn the nz into CCL Tracer objects
        # Use the cosmology object to calculate C_ell values
        ...
        return theory_c_ell

    def compute_noise_c_ell(self, binning, n_eff, sigma_e, fsky):
        ...
        return noise_c_ell


    def compute_covariance(self, theory_c_ell, noise_c_ell):
        ...
        pass
