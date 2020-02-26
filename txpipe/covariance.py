
from .base_stage import PipelineStage
from .data_types import MetacalCatalog, HDFFile, YamlFile, SACCFile, TomographyCatalog, CSVFile
from .data_types import DiagnosticMaps
import numpy as np
import pandas as pd
import warnings
import pyccl as ccl
import sacc

# need to figure out the right way to do this...
import sys
sys.path.append("../TJPCov")
from wigner_transform import *
from parser import *
d2r=np.pi/180

#Needed changes: 1) area is hard coded to 4sq.deg. as file is buggy. 2) code fixed to equal-spaced ell values in real space. 3)

class TXFourierGaussianCovariance(PipelineStage):
    name='TXFourierGaussianCovariance'

    inputs = [
        ('fiducial_cosmology', YamlFile),  # For the cosmological parameters
        ('photoz_stack', HDFFile),  # For the n(z)
        ('twopoint_data_fourier', SACCFile), # For the binning information,  Re
        ('diagnostic_maps', DiagnosticMaps),
        ('tomography_catalog', TomographyCatalog),
    ]

    outputs = [
        ('summary_statistics_fourier', SACCFile),
    ]

    config_options = {
            #bias
        'do_xi':False,
            #IA
    }

    def run(self):

        # read the fiducial cosmology
        cosmo = self.read_cosmology()

        # read binning
        if not self.config['do_xi']:
            two_point_data = self.read_sacc_fourier()
        else:
            two_point_data = self.read_sacc_real()

        # read the n(z) and f_sky from the source summary stats
        n_eff, n_lens, sigma_e, fsky = self.read_number_statistics()
        
        meta = {}
        meta['fsky'] = fsky
        meta['ell'] = np.arange(2,500) #5000
        meta['th'] = np.logspace(np.log10(0.1/60),np.log10(600./60),60) #200
        meta['sigma_e'] = sigma_e
        meta['n_eff'] = n_eff # per radian2
        meta['n_lens'] = n_lens # per radian2
        meta['IA'] = np.array([1.0, 1.0])
        meta['gal_bias'] = np.array([1.0, 1.0])

        #meta['ell_bins'] = np.arange(2,500,45)
        #two_point_data.metadata['fsky']=fsky
        #two_point_data.metadata['ell']=np.arange(2,500) #np.arange(2,500)
        #two_point_data.metadata['ell_bins']= 1.0*np.arange(2,500,45)
        #th_min=2.5/60 # in degrees
        #th_max=250./60
        #n_th_bins=20
        #two_point_data.metadata['th_bins']=np.logspace(np.log10(th_min),np.log10(th_max),n_th_bins+1)
        #meta['th_bins']=np.logspace(np.log10(th_min),np.log10(th_max),n_th_bins+1)

        #th=np.logspace(np.log10(th_min*0.98),np.log10(1),n_th_bins*30) #covariance is oversampled at th values and then binned.
        #th2=np.linspace(1,th_max*1.02,n_th_bins*30) #binned covariance can be sensitive to the th values. Make sue you check convergence for your application
        # th2=np.logspace(np.log10(1),np.log10(th_max),60*6)
        #two_point_data.metadata['th']=np.unique(np.sort(np.append(th,th2)))
        #meta['th']=np.unique(np.sort(np.append(th,th2)))
        #thb=0.5*(two_point_data.metadata['th_bins'][1:]+two_point_data.metadata['th_bins'][:-1])
        #thb=0.5*(meta['th_bins'][1:]+meta['th_bins'][:-1])

        #C_ell covariance
        cov = self.get_all_cov(cosmo, meta, two_point_data=two_point_data,do_xi=self.config['do_xi'])
        
        #xi covariance .... right now shear-shear is xi+ only. xi- needs to be added in the loops.
        #cov_xi = get_all_cov(two_point_data=twopoint_data,do_xi=True)

        # calculate the overall total C_ell values, including noise
        #theory_c_ell = self.compute_theory_c_ell(cosmo, nz, sacc_data)
        #noise_c_ell = self.compute_noise_c_ell(n_eff, sigma_e, n_lens, sacc_data)

        # compute covariance
        #cov = self.compute_covariance(sacc_data, theory_c_ell, noise_c_ell, fsky)
        if not self.config['do_xi']:
            self.save_outputs_fourier(two_point_data, cov)
        else:
            self.save_outputs_real(two_point_data, cov)

    def save_outputs_fourier(self, two_point_data, cov):
        filename = self.get_output('summary_statistics_fourier')
        two_point_data.add_covariance(cov)
        two_point_data.save_fits(filename, overwrite=True)

    def read_cosmology(self):
        filename = self.get_input('fiducial_cosmology')
        cosmo = ccl.Cosmology.read_yaml(filename)

        print("COSMOLOGY OBJECT:")
        print(cosmo)
        return cosmo

    def read_sacc_fourier(self):
        f = self.get_input('twopoint_data_fourier')
        two_point_data = sacc.Sacc.load_fits(f)

        # Remove the data types that we won't use for inference
        mask = [
            two_point_data.indices(sacc.standard_types.galaxy_shear_cl_ee),
            two_point_data.indices(sacc.standard_types.galaxy_shearDensity_cl_e),
            two_point_data.indices(sacc.standard_types.galaxy_density_cl),
        ]
        mask = np.concatenate(mask)
        two_point_data.keep_indices(mask)

        two_point_data.to_canonical_order()

        return two_point_data

    def save_outputs_real(self, two_point_data, cov):
        filename = self.get_output('summary_statistics_real')
        two_point_data.add_covariance(cov)
        two_point_data.save_fits(filename, overwrite=True)

    def read_sacc_real(self):
        f = self.get_input('twopoint_data_real')
        two_point_data = sacc.Sacc.load_fits(f)

        # Remove the data types that we won't use for inference
        mask = [
            two_point_data.indices(sacc.standard_types.galaxy_density_xi),
            two_point_data.indices(sacc.standard_types.galaxy_shearDensity_xi_t),
            two_point_data.indices(sacc.standard_types.galaxy_shear_xi_plus),
         #   two_point_data.indices(sacc.standard_types.galaxy_shear_xi_minus),
        ]
        mask = np.concatenate(mask)
        two_point_data.keep_indices(mask)

        two_point_data.to_canonical_order()

        return two_point_data

    def read_number_statistics(self):
        input_data = self.open_input('photoz_stack')
        tomo_file = self.open_input('tomography_catalog')
        maps_file = self.open_input('diagnostic_maps')

        N_eff = tomo_file['tomography/N_eff']
        N_lens = tomo_file['tomography/lens_counts']

        # area in sq deg
        area_deg2 = maps_file['maps'].attrs['area']
        area_unit = maps_file['maps'].attrs['area_unit']
        if area_unit != 'sq deg':
            raise ValueError("Units of area have changed")
        area = area_deg2 * np.radians(1)**2

        print(f"area = {area_deg2:.1f} deg^2")
        print("NEFF : ", N_eff)
        n_eff = N_eff / area
        print("nEFF : ", n_eff)
        sigma_e = tomo_file['tomography/sigma_e'][:]

        n_lens = N_lens / area
        print("lens density : ", n_lens)

        fullsky=4*np.pi #*(180./np.pi)**2 #(FULL SKY IN STERADIANS)
        fsky=area/fullsky

        input_data.close()
        tomo_file.close()
        maps_file.close()

        print('n_lens, n_eff, sigma_e, fsky: ')
        print(n_lens, n_eff, sigma_e, fsky)
        return n_eff, n_lens, sigma_e, fsky

    def get_tracer_info(self, cosmo, meta, two_point_data):
        
        ccl_tracers={}
        tracer_Noise={}

        for tracer in two_point_data.tracers:
            tracer_dat = two_point_data.get_tracer(tracer)
            nbin = int(two_point_data.tracers[tracer].name[-1]) #might be a better way of doing this?
            z = tracer_dat.z
            dNdz = tracer_dat.nz
            dNdz /= (dNdz*np.gradient(z)).sum()

            if 'source' in tracer or 'src' in tracer:
                sigma_e = meta['sigma_e'][nbin]
                AI = meta['IA'][nbin]*np.ones(len(z))
                Ngal = meta['n_eff'][nbin]
                #Ngal = Ngal*3600/d2r**2
                #dNdz *= Ngal
                
                ccl_tracers[tracer]=ccl.WeakLensingTracer(cosmo, dndz=(z, dNdz),ia_bias=(z,AI)) #CCL automatically normalizes dNdz
                tracer_Noise[tracer]=sigma_e**2/Ngal
            
            elif 'lens' in tracer:
                b = meta['gal_bias'][nbin]*np.ones(len(z))
                Ngal = meta['n_lens'][nbin]
                #Ngal = Ngal*3600/d2r**2
                #dNdz *= Ngal
                tracer_Noise[tracer]=1./Ngal
                ccl_tracers[tracer]=ccl.NumberCountsTracer(cosmo, has_rsd=False, dndz=(z,dNdz), bias=(z,b))
        
        return ccl_tracers,tracer_Noise

    def get_cov_WT_spin(self, tracer_comb=None):
    #     tracers=tuple(i.split('_')[0] for i in tracer_comb)

        WT_factors={}
        WT_factors['lens','source']=(0,2)
        WT_factors['source','lens']=(2,0) #same as (0,2)
        WT_factors['source','source']={'plus':(2,2),'minus':(2,-2)}
        WT_factors['lens','lens']=(0,0)

        tracers=[]
        for i in tracer_comb:
            if 'lens' in i:
                tracers+=['lens']
            if 'source' in i:
                tracers+=['source']
        return WT_factors[tuple(tracers)]


    #compute a single covariance matrix for a given pair of C_ell or xi.  
    def cl_gaussian_cov(self, cosmo, meta, ell_bins, tracer_comb1=None,tracer_comb2=None,ccl_tracers=None,tracer_Noise=None,two_point_data=None,do_xi=False,
                    xi_plus_minus1='plus',xi_plus_minus2='plus'):
        #fsky should be read from the sacc
        #tracers 1,2,3,4=tracer_comb1[0],tracer_comb1[1],tracer_comb2[0],tracer_comb2[1]
        ell=meta['ell']
        cl={}
        cl[13] = ccl.angular_cl(cosmo, ccl_tracers[tracer_comb1[0]], ccl_tracers[tracer_comb2[0]], ell)
        cl[24] = ccl.angular_cl(cosmo, ccl_tracers[tracer_comb1[1]], ccl_tracers[tracer_comb2[1]], ell)
        cl[14] = ccl.angular_cl(cosmo, ccl_tracers[tracer_comb1[0]], ccl_tracers[tracer_comb2[1]], ell)
        cl[23] = ccl.angular_cl(cosmo, ccl_tracers[tracer_comb1[1]], ccl_tracers[tracer_comb2[0]], ell)

        SN={}
        SN[13]=tracer_Noise[tracer_comb1[0]] if tracer_comb1[0]==tracer_comb2[0]  else 0
        SN[24]=tracer_Noise[tracer_comb1[1]] if tracer_comb1[1]==tracer_comb2[1]  else 0
        SN[14]=tracer_Noise[tracer_comb1[0]] if tracer_comb1[0]==tracer_comb2[1]  else 0
        SN[23]=tracer_Noise[tracer_comb1[1]] if tracer_comb1[1]==tracer_comb2[0]  else 0

        if do_xi:
            norm=np.pi*4*meta['fsky']
        else: 
            norm=(2*ell+1)*np.gradient(ell)*meta['fsky']

        coupling_mat={}
        coupling_mat[1324]=np.eye(len(ell)) #placeholder
        coupling_mat[1423]=np.eye(len(ell)) #placeholder

        cov={}
        cov[1324]=np.outer(cl[13]+SN[13],cl[24]+SN[24])*coupling_mat[1324]
        cov[1423]=np.outer(cl[14]+SN[14],cl[23]+SN[23])*coupling_mat[1423]

        cov['final']=cov[1423]+cov[1324]

        if do_xi:
            s1_s2_1 = self.get_cov_WT_spin(tracer_comb=tracer_comb1)
            s1_s2_2 = self.get_cov_WT_spin(tracer_comb=tracer_comb2)
            if isinstance(s1_s2_1,dict):
                s1_s2_1=s1_s2_1[xi_plus_minus1]
            if isinstance(s1_s2_2,dict):
                s1_s2_2=s1_s2_2[xi_plus_minus2]


            WT_kwargs = {'l': ell,'theta': meta['th']*d2r,'s1_s2':[(2,2),(2,-2),(0,2),(2,0),(0,0)]}
            WT = wigner_transform(**WT_kwargs)

            th,cov['final']=WT.projected_covariance2(l_cl=ell,s1_s2=s1_s2_1, s1_s2_cross=s1_s2_2,
                                                      cl_cov=cov['final'])

        cov['final']/=norm

        if do_xi:
            thb,cov['final_b'] = bin_cov(r=th/d2r,r_bins=ell_bins,cov=cov['final'])
        else:
            if ell_bins is not None:
                lb,cov['final_b'] = bin_cov(r=ell,r_bins=ell_bins,cov=cov['final'])

        print(cov['final_b'].shape, cov['final'].shape)
#     cov[1324]=None #if want to save memory
#     cov[1423]=None #if want to save memory

        return cov
    
    def get_ell_bins(self, two_point_data):
        X = two_point_data.get_data_points('galaxy_shear_cl_ee',i=0,j=0)
        ell_edges = []
        for i in range(len(X)):
            ell_edges.append(X[i]['window'].min[0])
        ell_edges.append(X[-1]['window'].min[-1])
        ell_edges = np.array(ell_edges)
        return ell_edges

    def get_th_bins(self, two_point_data):
        th_arcmin = np.logspace(np.log10(self.config['min_sep']), np.log10(self.config['max_sep']), self.config['nbins']+1)
        return th_arcmin/60.0

    #compute all the covariances and then combine them into one single giant matrix
    def get_all_cov(self, cosmo, meta, two_point_data={},do_xi=False):
        #FIXME: Only input needed should be two_point_data, which is the sacc data file. Other parameters should be included within sacc and read from there.
        ccl_tracers,tracer_Noise = self.get_tracer_info(cosmo, meta, two_point_data=two_point_data)
        tracer_combs = two_point_data.get_tracer_combinations() # we will loop over all these
        N2pt = len(tracer_combs)

        if not do_xi:
            ell_bins = self.get_ell_bins(two_point_data)
        else:
            ell_bins = self.get_th_bins(two_point_data)

        if ell_bins is not None:
            Nell_bins = len(ell_bins)-1
        else: 
            Nell_bins = len(ell_bins)

        cov_full=np.zeros((Nell_bins*N2pt,Nell_bins*N2pt))
        for i in np.arange(N2pt):
            tracer_comb1=tracer_combs[i]
            indx_i=i*Nell_bins
            for j in np.arange(i,N2pt):
                tracer_comb2=tracer_combs[j]
                indx_j=j*Nell_bins
                cov_ij = self.cl_gaussian_cov(cosmo, meta, ell_bins, tracer_comb1=tracer_comb1,tracer_comb2=tracer_comb2,ccl_tracers=ccl_tracers,
                                        tracer_Noise=tracer_Noise,do_xi=do_xi,two_point_data=two_point_data)
                #if do_xi or meta['th_bins'] is not None:
                #    cov_ij=cov_ij['final_b']
                #else:
                cov_ij=cov_ij['final_b']
                cov_full[indx_i:indx_i+Nell_bins,indx_j:indx_j+Nell_bins]=cov_ij
                cov_full[indx_j:indx_j+Nell_bins,indx_i:indx_i+Nell_bins]=cov_ij.T
        return cov_full



class TXRealGaussianCovariance(TXFourierGaussianCovariance):
    name='TXRealGaussianCovariance'

    inputs = [
        ('fiducial_cosmology', YamlFile),  # For the cosmological parameters
        ('photoz_stack', HDFFile),  # For the n(z)
        ('twopoint_data_real', SACCFile), # For the binning information,  Re
        ('diagnostic_maps', DiagnosticMaps),
        ('tomography_catalog', TomographyCatalog)
    ]

    outputs = [
        ('summary_statistics_real', SACCFile),
    ]

    config_options = {
        'do_xi':True,
        'min_sep':2.5,
        'max_sep':250,
        'nbins':20,
            }



    def run(self):
        super().run()


