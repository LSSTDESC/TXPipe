from .base_stage import PipelineStage
from .data_types import MetacalCatalog, HDFFile, YamlFile, SACCFile, TomographyCatalog, CSVFile
from .data_types import DiagnosticMaps
import numpy as np
import pandas as pd
import warnings

#Needed changes: 1) area is hard coded to 4sq.deg. as file is buggy. 2) code fixed to equal-spaced ell values in real space. 3)


class TXFourierGaussianCovariance(PipelineStage):
    name='TXFourierGaussianCovariance'

    inputs = [
        ('fiducial_cosmology', YamlFile),  # For the cosmological parameters
        ('photoz_stack', HDFFile),  # For the n(z)
        ('twopoint_data_fourier', SACCFile), # For the binning information,  Re
        ('diagnostic_maps', DiagnosticMaps),
        ('tomography_catalog', TomographyCatalog)
        #('photoz_pdfs', PhotozPDFFile)
    ]

    outputs = [
        ('summary_statistics', SACCFile),
    ]

    config_options = {
    }


    def run(self):
        import pyccl as ccl
        import sacc
        # read the fiducial cosmology
        cosmo = self.read_cosmology()

        # read binning
        sacc_data = self.read_sacc()

        # read the n(z) and f_sky from the source summary stats
        nz, n_eff, n_lens, sigma_e, fsky, N_tomo_bins = self.read_number_statistics()

        # calculate the overall total C_ell values, including noise
        theory_c_ell = self.compute_theory_c_ell(cosmo, nz, sacc_data)
        noise_c_ell = self.compute_noise_c_ell(n_eff, sigma_e, n_lens, sacc_data)

        # compute covariance
        cov = self.compute_covariance(sacc_data, theory_c_ell, noise_c_ell, fsky)
        self.save_outputs(sacc_data, cov)
        #self.save_outputs_firecrown(C,data_vector,ells,nz,binning_info)

    def read_cosmology(self):
        import pyccl as ccl
        filename = self.get_input('fiducial_cosmology')
        cosmo = ccl.Cosmology.read_yaml(filename)

        print("COSMOLOGY OBJECT:")
        print(cosmo)
        return cosmo

    def read_sacc(self):
        import sacc
        f = self.get_input('twopoint_data_fourier')
        sacc_data = sacc.Sacc.load_fits(f)

        # Remove the data types that we won't use for inference
        mask = [
            sacc_data.indices(sacc.standard_types.galaxy_shear_cl_ee),
            sacc_data.indices(sacc.standard_types.galaxy_shearDensity_cl_e),
            sacc_data.indices(sacc.standard_types.galaxy_density_cl),
        ]
        mask = np.concatenate(mask)
        sacc_data.keep_indices(mask)

        sacc_data.to_canonical_order()

        return sacc_data


    def read_number_statistics(self):
        input_data = self.open_input('photoz_stack')
        tomo_file = self.open_input('tomography_catalog')
        maps_file = self.open_input('diagnostic_maps')

        N_tomo_bins=len(tomo_file['tomography/sigma_e'][:])
        print("NBINS: ", N_tomo_bins)

        nz = {}
        nz['z'] = input_data[f'n_of_z/source/z'][:]
        for i in range(N_tomo_bins):
            nz['bin_'+ str(i)] = input_data[f'n_of_z/source/bin_{i}'][:]

        N_eff = tomo_file['tomography/N_eff'][:]
        N_lens = tomo_file['tomography/lens_counts'][:]

        area = maps_file['maps'].attrs['area']
        print(area)
        area = 4.0/((180./np.pi)**2) #read this in when not array of 0.04 (CONVERTED TO SR)
        print("area: ",area)
        print("NEFF : ", N_eff)
        n_eff=N_eff / area
        print((180./np.pi)**2)
        print("nEFF : ", n_eff)
        sigma_e = tomo_file['tomography/sigma_e'][:]

        n_lens = N_lens / area
        print("lens density : ", n_eff)

        fullsky=4*np.pi #*(180./np.pi)**2 #(FULL SKY IN STERADIANS)
        fsky=area/fullsky

        input_data.close()
        tomo_file.close()
        maps_file.close()

        print('n_eff, sigma_e, fsky: ')
        print( n_eff, sigma_e, fsky)
        return nz, n_eff, n_lens, sigma_e, fsky, N_tomo_bins

    def compute_theory_c_ell(self, cosmo, nz, sacc_data):
        # Turn the nz into CCL Tracer objects
        # Use the cosmology object to calculate C_ell values
        import pyccl as ccl
        import sacc

        CEE = sacc.standard_types.galaxy_shear_cl_ee
        CEd = sacc.standard_types.galaxy_shearDensity_cl_e
        Cdd = sacc.standard_types.galaxy_density_cl
        theory = {}

        for data_type in [CEE, CEd, Cdd]:
            for t1, t2 in sacc_data.get_tracer_combinations(data_type):
                ell = sacc_data.get_tag('ell', data_type, (t1, t2))
                tracer1 = sacc_data.get_tracer(t1)
                tracer2 = sacc_data.get_tracer(t2)
                dndz1 = (tracer1.z, tracer1.nz)
                dndz2 = (tracer2.z, tracer2.nz)

                if data_type in [CEE, CEd]:
                    cTracer1 = ccl.WeakLensingTracer(cosmo, dndz=dndz1)
                else:
                    bias = (tracer1.z, np.ones_like(tracer1.z))
                    cTracer1 = ccl.NumberCountsTracer(cosmo, has_rsd=False, bias=bias, dndz=dndz1)
                    warnings.warn("Not including bias in fiducial cosmology")

                if data_type == CEE:
                    cTracer2 = ccl.WeakLensingTracer(cosmo, dndz=dndz1)
                else:
                    bias = (tracer2.z, np.ones_like(tracer2.z))
                    cTracer2 = ccl.NumberCountsTracer(cosmo, has_rsd=False, bias=bias, dndz=dndz2)

                print(" - Calculating fiducial C_ell for ", data_type, t1, t2)
                theory[(data_type, t1, t2)] = ccl.angular_cl(cosmo, cTracer1, cTracer2, ell)

        return theory


    def compute_noise_c_ell(self, n_eff, sigma_e, n_lens, sacc_data):
        import sacc

        CEE=sacc.standard_types.galaxy_shear_cl_ee
        CEd=sacc.standard_types.galaxy_shearDensity_cl_e
        Cdd=sacc.standard_types.galaxy_density_cl

        noise = {}

        for data_type in sacc_data.get_data_types():
            for (tracer1, tracer2) in sacc_data.get_tracer_combinations(data_type):
                ell = sacc_data.get_tag('ell', data_type, (tracer1, tracer2))
                n = len(ell)
                b = int(tracer1.split("_")[1])
                if tracer1 != tracer2:
                    N = np.zeros_like(ell)
                elif data_type == CEE:
                    N = np.ones(n) * (sigma_e[b]**2 / n_eff[b])
                else:
                    assert data_type == Cdd
                    N = np.ones(n) / n_lens[b]
                noise[(data_type, tracer1, tracer2)] = N

        return noise


    def compute_covariance(self, sacc_data, theory_c_ell, noise_c_ell, fsky):
        import sacc
        n = len(sacc_data)
        cov = np.zeros((n, n))

        # It's useful to convert these names to two-character
        # mappings as we will need combinations of them

        names = {
            sacc.standard_types.galaxy_density_cl: 'DD',
            sacc.standard_types.galaxy_shear_cl_ee: 'EE',
            sacc.standard_types.galaxy_shearDensity_cl_e: 'ED',
        }

        # ell values must be the same for this all to work
        # TODO: Fix this for cases where we only want to work with a subset of the data
        ell = sacc_data.get_tag('ell', sacc.standard_types.galaxy_shear_cl_ee, ('source_0', 'source_0'))
        ell_indices = {ell:v for v,ell in enumerate(ell)}


        # Compute total C_ell for each bin (signal + noise)
        CL = {}
        for dt in sacc_data.get_data_types():
            for t1, t2 in sacc_data.get_tracer_combinations(dt):
                A, B = names[dt]
                # Save both the first combination and the flipped version, e.g. 
                # C^{ED}_{ij} = C^{DE}_{ji}
                # In many cases these will be the same, but this will
                # not be the slow part of the code
                CL[(A, B, t1, t2)] = theory_c_ell[(dt, t1, t2)] + noise_c_ell[(dt, t1, t2)]
                CL[(B, A, t2, t1)] = CL[(A, B, t1, t2)]


        # <C^{AB}_{ij} C^{CD}_{mn}> = C^{AC}_{im} C^{BD}_{jn} + C^{AD}_{in} C^{BC}_{jk}
        # First loop over rows
        for x in range(n):
            # The data point object
            d1 = sacc_data.data[x]
            # Binning information for this data point
            # This is the nominal ell value
            ell1 = d1['ell']
            ell_index = ell_indices[ell1]
            # And the window function to get the range
            win = d1['window']
            delta_ell = win.max - win.min

            # The prefactor depends only on ell and delta_ell
            f = 1.0/((2*ell1+1)*fsky*delta_ell)

            # This is e.g. "EE", "DE" or "ED"
            A,B = names[d1.data_type]

            # These tracer names are also part of the C_ell dictonary key
            i = d1.tracers[0]
            j = d1.tracers[1]

            # Now loop over columns
            for y in range(n):
                d2 = sacc_data.data[y]
                ell2 = d2['ell']

                # Dirac delta function
                if ell1 != ell2:
                    continue

                # Again, this is EE, ED, or DD
                C,D = names[d2.data_type]
                # Also tracer names
                k = d2.tracers[0]
                l = d2.tracers[1]

                cov[x,y] = f * (CL[(A, C, i, k)][ell_index] * CL[(B, D, j, l)][ell_index] + CL[(A, D, i, l)][ell_index] * CL[(B, C, j, k)][ell_index])

        return cov

    def save_outputs(self, sacc_data, cov):
        filename = self.get_output('summary_statistics')
        sacc_data.add_covariance(cov)
        sacc_data.save_fits(filename)
