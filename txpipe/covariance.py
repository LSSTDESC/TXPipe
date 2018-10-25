from ceci import PipelineStage
from .data_types import MetacalCatalog, HDFFile, YamlFile, SACCFile, TomographyCatalog
from .data_types import DiagnosticMaps
import numpy as np

class TXFourierGaussianCovariance(PipelineStage):
    name='TXFourierGaussianCovariance'

    inputs = [
        ('fiducial_cosmology', YamlFile),  # For the cosmological parameters
        ('photoz_stack', HDFFile),  # For the n(z)
        ('twopoint_data', SACCFile), # For the binning information,  Re
        ('diagnostic_maps', DiagnosticMaps), 
        ('tomography_catalog', TomographyCatalog)
        #('photoz_pdfs', PhotozPDFFile)
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
        nz, sigma_e, n_eff, fsky, N_tomo_bins = self.read_number_statistics()

        # calculate the overall total C_ell values, including noise
        theory_c_ell = self.compute_theory_c_ell(cosmo, nz, binning_info)
        noise_c_ell = self.compute_noise_c_ell(n_eff, sigma_e, binning_info)

        # compute covariance
        C = self.compute_covariance(binning_info, theory_c_ell, noise_c_ell, fsky)

    def read_cosmology(self):
        import pyccl as ccl
        filename = self.get_input('fiducial_cosmology')
        cosmo = ccl.Cosmology.read_yaml(filename)
        print("COSMOLOGY OBJECT")
        print(cosmo)
        return cosmo

    def read_binning(self):
        import sacc
        f = self.get_input('twopoint_data')
        sacc_data = sacc.SACC.loadFromHDF(f)
        ###
        codes = {
            'Cll' : ('S','S'),
            #'Cdd' : ('P','P'),
            #'Cdl' : ('P','S'),
         }
        binning = {quant: {} for quant in codes}

        for quant, (code1,code2) in codes.items():
            bin_pairs = sacc_data.binning.get_bin_pairs(code1,code2)
            for (b1, b2) in bin_pairs:
                angle = sacc_data.binning.get_angle(code1, code2, b1, b2)
                binning[quant][(b1,b2)] = angle
        print("BINNING:")
        print(binning['Cll'])

        #print(len(binning['Cll'][0,0]))
        return binning 

    
    def read_number_statistics(self):
        input_data = self.open_input('photoz_stack')
        tomo_file = self.open_input('tomography_catalog')
        maps_file = self.open_input('diagnostic_maps')

        #nbin_source=tomo_file.read_nbin('source', wrapper=True)
        
        N_tomo_bins=len(tomo_file['tomography/sigma_e'].value)
        print("NBINS: ", N_tomo_bins)
        nz = {}
        n_eff = {}
        nz['z'] = input_data[f'n_of_z/source/z'].value
        for i in range(N_tomo_bins):
            nz['bin_'+ str(i)] = input_data[f'n_of_z/source/bin_{i}'].value 

        N_eff = tomo_file['tomography/N_eff'].value
        #print(N_eff)
        #area = maps_file['maps'].attrs['area']
        #print('AREA')
        #print(area)
        area = 4.0 #read this in when not array of 0.04
        n_eff=N_eff/area
        #print(n_eff)

        sigma_e = tomo_file['tomography/sigma_e'].value

        fullsky=4*np.pi*(180./np.pi)**2
        fsky=area/fullsky
    
        input_data.close()
        tomo_file.close()
        maps_file.close()
    
        #print(nz, n_eff, sigma_e, fsky) 
        return nz, n_eff, sigma_e, fsky, N_tomo_bins

    def compute_theory_c_ell(self, cosmo, nz, binning):
        # Turn the nz into CCL Tracer objects
        # Use the cosmology object to calculate C_ell values
        import pyccl as ccl

        theory_c_ell = {}
        ell=binning['Cll'][0,0]  #change this
        z = nz.get('z')
        
        for key in binning['Cll']:
            nz_1 = nz.get('bin_'+ str(key[0]))
            nz_2 = nz.get('bin_' + str(key[1]))
            tracer1 = ccl.WeakLensingTracer(cosmo, dndz=(z, nz_1))
            tracer2 = ccl.WeakLensingTracer(cosmo, dndz=(z, nz_2))
            theory_c_ell[str(key[0]) + str(key[1])] = ccl.angular_cl(cosmo, tracer1, tracer2, ell)

        print('THEORY_Cl:')
        print(theory_c_ell)
        return theory_c_ell


    def compute_noise_c_ell(self, n_eff, sigma_e, binning):
        #avg number of galaxies in a zbin
        noise_c_ell = {}

        for key in binning['Cll']:
            if key[0]==key[1]:
                noise_c_ell[str(key[0]) + str(key[1])] = (sigma_e[key[0]]**2/n_eff[key[0]]) 
            else:
                noise_c_ell[str(key[0]) + str(key[1])] = 0.0
        
        print('NOISE_Cl:')
        print(noise_c_ell)
        return noise_c_ell



    def switch_keys(self, bin_1, bin_2, theory_c_ell):
        if str(bin_1) + str(bin_2) in theory_c_ell.keys():
            obs_c_ell = theory_c_ell.get(str(bin_1) + str(bin_2))
            return obs_c_ell                
        else:
            obs_c_ell = theory_c_ell.get(str(bin_2) + str(bin_1))
            return obs_c_ell

    def compute_covariance(self, binning, theory_c_ell, noise_c_ell, fsky):
        ell=binning['Cll'][0,0]
        delta_ell=ell[1]-ell[0] #not in general equal, this needs to be improved 



        def switch_keys(bin_1, bin_2, theory_c_ell):
            if str(bin_1) + str(bin_2) in theory_c_ell.keys():
                obs_c_ell = theory_c_ell.get(str(bin_1) + str(bin_2))
                return obs_c_ell                
            else:
                obs_c_ell = theory_c_ell.get(str(bin_2) + str(bin_1))
                return obs_c_ell



        cov_dict = {}


        for key_row in binning['Cll']:
            for key_col in binning['Cll']:
                i = key_row[0]
                j = key_row[1]
                m = key_col[0]
                n = key_col[1]
                
                #print('im:', str(i)+str(m))

                obs_c_ell_im = switch_keys(str(i), str(m), theory_c_ell)
                obs_c_ell_jn = switch_keys(str(j), str(n), theory_c_ell)
                obs_c_ell_in = switch_keys(str(i), str(n), theory_c_ell)
                obs_c_ell_jm = switch_keys(str(j), str(m), theory_c_ell)
             
                prefactor = 1./((2*ell+1)*delta_ell*fsky)
                cov_dict['(' + str(i) + str(j) + ',' + str(m) + str(n) + ')'] = prefactor*(obs_c_ell_im*obs_c_ell_jn+obs_c_ell_in*obs_c_ell_jm)

        for key in cov_dict:
            print(key)

        #cov = np.zeros(len(binning['Cll']))
        pass
