from ceci import PipelineStage
from .data_types import MetacalCatalog, HDFFile, YamlFile, SACCFile, TomographyCatalog, CSVFile
from .data_types import DiagnosticMaps
import numpy as np
import pandas as pd

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
        ('covariance', CSVFile),
    ]

    config_options = {
    }


    def run(self):
        # read the fiducial cosmology
        cosmo = self.read_cosmology()

        # read the n(z) and f_sky from the source summary stats
        nz, n_eff, sigma_e, fsky, N_tomo_bins = self.read_number_statistics()

        # read the lensing binning
        binning_info_lensing, data_vector_lensing, ells_lensing = self.read_sacc('Cll')

        # calculate the overall total C_ell values, including noise
        theory_c_ell_lensing, theory_c_ell_galaxy_galaxy, theory_c_ell_clustering = self.compute_theory_c_ell(cosmo, nz, binning_info)
        noise_c_ell_lensing, noise_c_ell_galaxy_galaxy, noise_c_ell_clustering = self.compute_noise_c_ell(n_eff, sigma_e, binning_info)

        # compute the lensing covariance
        C_lensing = self.compute_covariance_lensing(binning_info_lensing, theory_c_ell_lensing, noise_c_ell_lensing, fsky)
        self.save_outputs_firecrown(C_lensing,data_vector_lensing,ells_lensing,nz,binning_info_lensing)

        # read galaxy galaxy binning
        binning_info_galaxy_galaxy, data_vector_galaxy_galaxy, ells_galaxy_galaxy = self.read_sacc('Cdl')

        # compute galaxy galaxy covariance
        C_galaxy_galaxy = self.compute_covariance_clustering(binning_info_galaxy_galaxy, theory_c_ell_galaxy_galaxy, noise_c_ell_galaxy_galaxy, fsky)
        self.save_outputs_firecrown(C_galaxy_galaxy,data_vector_galaxy_galaxy,ells_galaxy_galaxy,nz,binning_info_galaxy_galaxy)

        # read binning
        binning_info_clustering, data_vector_clustering, ells_clustering = self.read_sacc('Cdd')

        # compute covariance
        C_clustering = self.compute_covariance_clustering(binning_info, theory_c_ell_clustering, noise_c_ell_clustering, fsky)
        self.save_outputs_firecrown(C_clustering,data_vector_clustering,ells_clustering,nz,binning_info_clustering)


    def read_cosmology(self):
        import pyccl as ccl
        filename = self.get_input('fiducial_cosmology')
        cosmo = ccl.Cosmology.read_yaml(filename)

        print("COSMOLOGY OBJECT:")
        print(cosmo)
        return cosmo

    def read_sacc(self,code):
        import sacc
        f = self.get_input('twopoint_data_fourier')
        sacc_data = sacc.SACC.loadFromHDF(f)

        codes = {
            'Cll' : ('E','E'),
            'Cdd' : ('P','P'),
            'Cdl' : ('P','S'),
            }

        binning = {quant: {} for quant in codes}

        for quant, (code1,code2) in [codes[code]]:
            bin_pairs = sacc_data.binning.get_bin_pairs(code1,code2)
            for (b1, b2) in bin_pairs:
                angle = sacc_data.binning.get_angle(code1, code2, b1, b2)
                binning[quant][(b1,b2)] = angle
        values = sacc_data.mean.vector
        ells = [row[1] for row in sacc_data.binning.binar]
        return binning, values, ells


    def read_number_statistics(self):
        input_data = self.open_input('photoz_stack')
        tomo_file = self.open_input('tomography_catalog')
        maps_file = self.open_input('diagnostic_maps')

        N_tomo_bins=len(tomo_file['tomography/sigma_e'].value)
        print("NBINS: ", N_tomo_bins)

        nz = {}
        nz['z'] = input_data[f'n_of_z/source/z'].value
        for i in range(N_tomo_bins):
            nz['bin_'+ str(i)] = input_data[f'n_of_z/source/bin_{i}'].value

        N_eff = tomo_file['tomography/N_eff'].value
        area = maps_file['maps'].attrs['area']
        print(area)
        area = 4.0/((180./np.pi)**2) #read this in when not array of 0.04 (CONVERTED TO SR)
        print("area: ",area)
        print("NEFF : ", N_eff)
        n_eff=N_eff/area
        print((180./np.pi)**2)
        print("nEFF : ", n_eff)
        sigma_e = tomo_file['tomography/sigma_e'].value

        fullsky=4*np.pi #*(180./np.pi)**2 #(FULL SKY IN STERADIANS)
        fsky=area/fullsky

        input_data.close()
        tomo_file.close()
        maps_file.close()

        print('n_eff, sigma_e, fsky: ')
        print( n_eff, sigma_e, fsky)
        return nz, n_eff, sigma_e, fsky, N_tomo_bins

    def compute_theory_c_ell(self, cosmo, nz, binning):
        # Turn the nz into CCL Tracer objects
        # Use the cosmology object to calculate C_ell values
        import pyccl as ccl

        theory_c_ell_lensing = {}
        ell=binning['Cll'][0,0]
        z = nz.get('z')

        for key in binning['Cll']:
            nz_1 = nz.get('bin_'+ str(key[0]))
            nz_2 = nz.get('bin_' + str(key[1]))
            tracer1 = ccl.WeakLensingTracer(cosmo, dndz=(z, nz_1))
            tracer2 = ccl.WeakLensingTracer(cosmo, dndz=(z, nz_2))
            theory_c_ell_lensing[str(key[0]) + str(key[1])] = ccl.angular_cl(cosmo, tracer1, tracer2, ell)

        theory_c_ell_clustering = {}
        ell=binning['Cdd'][0,0]
        z = nz.get('z')
        b = 1.5 # Need to update this to use a value that makes sense for the BOSS sample

        for key in binning['Cdd']:
            nz_1 = nz.get('bin_'+ str(key[0]))
            nz_2 = nz.get('bin_' + str(key[1]))
            tracer1 = ccl.NumberCountsTracer(cosmo, dndz=(z,dNdz), bias=(z,b))
            tracer2 = ccl.NumberCountsTracer(cosmo, dndz=(z,dNdz), bias=(z,b))
            theory_c_ell_clustering[str(key[0]) + str(key[1])] = ccl.angular_cl(cosmo, tracer1, tracer2, ell)


        theory_c_ell_galaxy_galaxy = {}
        ell=binning['Cdl'][0,0]
        z = nz.get('z')
        b = 1.5*np.ones(len(z)) #Setting bias to 1.5 for now

        for key in binning['Cdl']:
            nz_1 = nz.get('bin_'+ str(key[0]))
            nz_2 = nz.get('bin_' + str(key[1]))
            tracer1 = ccl.NumberCountsTracer(cosmo, dndz=(z,dNdz), bias=(z,b))
            tracer2 = ccl.WeakLensingTracer(cosmo, dndz=(z, nz_2))
            theory_c_ell_galaxy_galaxy[str(key[0]) + str(key[1])] = ccl.angular_cl(cosmo, tracer1, tracer2, ell)

        return theory_c_ell_lensing, theory_c_ell_galaxy_galaxy, theory_c_ell_clustering


    def compute_noise_c_ell(self, n_eff, sigma_e, binning):
        #avg number of galaxies in a zbin
        noise_c_ell_lensing = {}
        noise_c_ell_clustering = {}
        noise_c_ell_galaxy_galaxy = {}
        ell=binning['Cll'][0,0]

        for key in binning['Cll']:
            if key[0]==key[1]:
                noise_c_ell_lensing[str(key[0]) + str(key[1])] = np.ones(len(ell))*(sigma_e[key[0]]**2/n_eff[key[0]])
            else:
                noise_c_ell_lensing[str(key[0]) + str(key[1])] = np.zeros(len(ell))

        ell=binning['Cdl'][0,0]

        for key in binning['Cdl']:
            if key[0]==key[1]:
                noise_c_ell_galaxy_galaxy[str(key[0]) + str(key[1])] = np.ones(len(ell))*(sigma_e[key[0]]**2/n_eff[key[0]])
            else:
                noise_c_ell_galaxy_galaxy[str(key[0]) + str(key[1])] = np.zeros(len(ell))

        ell=binning['Cdd'][0,0]

        for key in binning['Cdd']:
            if key[0]==key[1]:
                noise_c_ell_clustering[str(key[0]) + str(key[1])] = np.ones(len(ell))
            else:
                noise_c_ell_clustering[str(key[0]) + str(key[1])] = np.zeros(len(ell))
        return noise_c_ell_lensing, noise_c_ell_galaxy_galaxy, noise_c_ell_clustering


    def compute_covariance_lensing(self, binning, theory_c_ell, noise_c_ell, fsky):
        ell=binning['Cll'][0,0]
        delta_ell=ell[1]-ell[0] #not in general equal, this needs to be improved

        obs_c_ell = {}
        for key in binning['Cll']:
            obs_c_ell[str(key[0]) + str(key[1])] = theory_c_ell[str(key[0]) + str(key[1])] + noise_c_ell[str(key[0]) + str(key[1])]

        def switch_keys(bin_1, bin_2, obs_c_ell):
            if str(bin_1) + str(bin_2) in theory_c_ell.keys():
                obs_c_ell_xy = obs_c_ell.get(str(bin_1) + str(bin_2))
                return obs_c_ell_xy
            else:
                obs_c_ell_yx = obs_c_ell.get(str(bin_2) + str(bin_1))
                return obs_c_ell_yx


        indexrow = 0
        indexcol = 0
        cov=np.zeros((len(binning['Cll'])*len(ell),len(binning['Cll'])*len(ell)))

        for key_row in binning['Cll']:
            for key_col in binning['Cll']:

                i = key_row[0]
                j = key_row[1]
                m = key_col[0]
                n = key_col[1]

                obs_c_ell_im = switch_keys(str(i), str(m), obs_c_ell)
                obs_c_ell_jn = switch_keys(str(j), str(n), obs_c_ell)
                obs_c_ell_in = switch_keys(str(i), str(n), obs_c_ell)
                obs_c_ell_jm = switch_keys(str(j), str(m), obs_c_ell)

                prefactor = 1./((2*ell+1)*delta_ell*fsky)

                mini_cov = np.zeros((len(ell),len(ell)))
                for a in range(len(ell)):
                    for b in range(len(ell)):
                        if a==b:
                            mini_cov[a][b] = obs_c_ell_im[a]*obs_c_ell_jn[b] + obs_c_ell_in[a]*obs_c_ell_jm[b]

                cov[indexrow*len(ell):indexrow*len(ell)+len(ell),indexcol*len(ell):indexcol*len(ell)+len(ell)] = prefactor*mini_cov
                print(prefactor)
                print(mini_cov)

                indexcol += 1
                if indexcol == 10:
                    indexrow += 1
                    indexcol = 0
        return cov

    def compute_covariance_galaxy_galaxy(self, binning, theory_c_ell, noise_c_ell, fsky):
        ell=binning['Cdl'][0,0]
        delta_ell=ell[1]-ell[0] #not in general equal, this needs to be improved

        obs_c_ell = {}
        for key in binning['Cdl']:
            obs_c_ell[str(key[0]) + str(key[1])] = theory_c_ell[str(key[0]) + str(key[1])] + noise_c_ell[str(key[0]) + str(key[1])]

        def switch_keys(bin_1, bin_2, obs_c_ell):
            if str(bin_1) + str(bin_2) in theory_c_ell.keys():
                obs_c_ell_xy = obs_c_ell.get(str(bin_1) + str(bin_2))
                return obs_c_ell_xy
            else:
                obs_c_ell_yx = obs_c_ell.get(str(bin_2) + str(bin_1))
                return obs_c_ell_yx


        indexrow = 0
        indexcol = 0
        cov=np.zeros((len(binning['Cdl'])*len(ell),len(binning['Cdl'])*len(ell)))

        for key_row in binning['Cdl']:
            for key_col in binning['Cdl']:

                i = key_row[0]
                j = key_row[1]
                m = key_col[0]
                n = key_col[1]

                obs_c_ell_im = switch_keys(str(i), str(m), obs_c_ell)
                obs_c_ell_jn = switch_keys(str(j), str(n), obs_c_ell)
                obs_c_ell_in = switch_keys(str(i), str(n), obs_c_ell)
                obs_c_ell_jm = switch_keys(str(j), str(m), obs_c_ell)

                prefactor = 1./((2*ell+1)*delta_ell*fsky)

                mini_cov = np.zeros((len(ell),len(ell)))
                for a in range(len(ell)):
                    for b in range(len(ell)):
                        if a==b:
                            mini_cov[a][b] = obs_c_ell_im[a]*obs_c_ell_jn[b] + obs_c_ell_in[a]*obs_c_ell_jm[b]

                cov[indexrow*len(ell):indexrow*len(ell)+len(ell),indexcol*len(ell):indexcol*len(ell)+len(ell)] = prefactor*mini_cov
                print(prefactor)
                print(mini_cov)

                indexcol += 1
                if indexcol == 10:
                    indexrow += 1
                    indexcol = 0
        return cov

    def compute_covariance_clustering(self, binning, theory_c_ell, noise_c_ell, fsky):
        ell=binning['Cdd'][0,0]
        delta_ell=ell[1]-ell[0] #not in general equal, this needs to be improved

        obs_c_ell = {}
        for key in binning['Cdd']:
            obs_c_ell[str(key[0]) + str(key[1])] = theory_c_ell[str(key[0]) + str(key[1])] + noise_c_ell[str(key[0]) + str(key[1])]

        def switch_keys(bin_1, bin_2, obs_c_ell):
            if str(bin_1) + str(bin_2) in theory_c_ell.keys():
                obs_c_ell_xy = obs_c_ell.get(str(bin_1) + str(bin_2))
                return obs_c_ell_xy
            else:
                obs_c_ell_yx = obs_c_ell.get(str(bin_2) + str(bin_1))
                return obs_c_ell_yx


        indexrow = 0
        indexcol = 0
        cov=np.zeros((len(binning['Cdd'])*len(ell),len(binning['Cdd'])*len(ell)))

        for key_row in binning['Cdd']:
            for key_col in binning['Cdd']:

                i = key_row[0]
                j = key_row[1]
                m = key_col[0]
                n = key_col[1]

                obs_c_ell_im = switch_keys(str(i), str(m), obs_c_ell)
                obs_c_ell_jn = switch_keys(str(j), str(n), obs_c_ell)
                obs_c_ell_in = switch_keys(str(i), str(n), obs_c_ell)
                obs_c_ell_jm = switch_keys(str(j), str(m), obs_c_ell)

                prefactor = 1./((2*ell+1)*delta_ell*fsky)

                mini_cov = np.zeros((len(ell),len(ell)))
                for a in range(len(ell)):
                    for b in range(len(ell)):
                        if a==b:
                            mini_cov[a][b] = obs_c_ell_im[a]*obs_c_ell_jn[b] + obs_c_ell_in[a]*obs_c_ell_jm[b]

                cov[indexrow*len(ell):indexrow*len(ell)+len(ell),indexcol*len(ell):indexcol*len(ell)+len(ell)] = prefactor*mini_cov
                print(prefactor)
                print(mini_cov)

                indexcol += 1
                if indexcol == 10:
                    indexrow += 1
                    indexcol = 0
        return cov

    def save_outputs_firecrown(self,cov,data_vector,ells,nz,binning):
        #Saving as a CSV file for now because FireCrown can import this but this
        # may eventually be saved in SACC.

        # cov formats
        # cov=np.zeros((len(binning['Cll'])*len(ell),len(binning['Cll'])*len(ell)))

        cov_output = CSVFile()
        cov_output_name = self.get_output('covariance')
        iis = np.range(cov.shape[0])
        jjs = np.range(cov.shape[1])
        #Note this won't be right if the matrix isn't square but we can't
        #save this as a pandas dataframe in the way Firecrown is expecting
        #in that case anyways.
        vals = [cov[iis[x]][jjs[x]] for x in range(len(iis))]
        cov_dic = {'i':iis,'j':jjs,'cov':vals}
        cov_df = pd.DataFrame(cov_dic)
        #cov_output.save_file(cov_output_name,cov_df)
        np.save('./outputs/cov_test.npy',cov)

        data_output = CSVFile()
        data_out_name = 'outputs/data_vec.csv'
        data_output_dic = {'ell':ells, 'measured_statistic':data_vector}
        data_output_df = pd.DataFrame(data_output_dic)
        #data_output_df.save_file(data_output_name,data_output_df)

        nz_output = CSVFile()
        nz_out_name = '/outputs/nz_vec.csv'
        zs = []
        dndzs = []
        #Change this to reference the binning
        for bin in range(self.config['nbins_lens']):
            zs+=nz['z']
            dndzs+= nz.get('bin_'+bin)
        for bin in range(self.config['nbins_source']):
            zs+=nz['z']
            dndzs+= nz.get('bin_'+bin)
        nz_output_dic = {'z':nz['z'], 'dndz':dndzs}
        nz_output_df = pd.DataFrame(nz_output_dic)
        #nz_output_df.save_file(nz_output_name,nz_output_df)
