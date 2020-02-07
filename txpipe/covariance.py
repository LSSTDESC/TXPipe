from .base_stage import PipelineStage
from .data_types import MetacalCatalog, HDFFile, YamlFile, SACCFile, TomographyCatalog, CSVFile
from .data_types import DiagnosticMaps
import numpy as np
import pandas as pd
import warnings
import pyccl as ccl
import sacc

import sys
sys.path.append("/global/homes/c/chihway/TJPCov")
from wigner_transform import *
from parser import *
d2r=np.pi/180

#Needed changes: 1) area is hard coded to 4sq.deg. as file is buggy. 2) code fixed to equal-spaced ell values in real space. 3)

# how do we want to have the real and Fourier stages? 


class TXGaussianCovariance(PipelineStage):
    name='TXGaussianCovariance'

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

        # read the fiducial cosmology
        cosmo = self.read_cosmology()

        # read binning
        two_point_data = self.read_sacc()

        # read the n(z) and f_sky from the source summary stats
        nz, n_eff, n_lens, sigma_e, fsky, N_tomo_bins = self.read_number_statistics()
        
        meta = {}
        meta['fsky'] = fsky
        meta['ell'] = np.arange(2,500)
        meta['ell_bins'] = np.arange(2,500,45)
        #two_point_data.metadata['fsky']=fsky
        #two_point_data.metadata['ell']=np.arange(2,500) #np.arange(2,500)
        #two_point_data.metadata['ell_bins']= 1.0*np.arange(2,500,45)
        th_min=2.5/60 # in degrees
        th_max=250./60
        n_th_bins=20
        #two_point_data.metadata['th_bins']=np.logspace(np.log10(th_min),np.log10(th_max),n_th_bins+1)
        meta['th_bins']=np.logspace(np.log10(th_min),np.log10(th_max),n_th_bins+1)

        th=np.logspace(np.log10(th_min*0.98),np.log10(1),n_th_bins*30) #covariance is oversampled at th values and then binned.
        th2=np.linspace(1,th_max*1.02,n_th_bins*30) #binned covariance can be sensitive to the th values. Make sue you check convergence for your application
        # th2=np.logspace(np.log10(1),np.log10(th_max),60*6)
        #two_point_data.metadata['th']=np.unique(np.sort(np.append(th,th2)))
        meta['th']=np.unique(np.sort(np.append(th,th2)))
        #thb=0.5*(two_point_data.metadata['th_bins'][1:]+two_point_data.metadata['th_bins'][:-1])
        thb=0.5*(meta['th_bins'][1:]+meta['th_bins'][:-1])


        #C_ell covariance
        cov_cl = get_all_cov(cosmo, meta, two_point_data=two_point_data,do_xi=False)
        #xi covariance .... right now shear-shear is xi+ only. xi- needs to be added in the loops.
        #cov_xi = get_all_cov(two_point_data=twopoint_data,do_xi=True)

        # calculate the overall total C_ell values, including noise
        #theory_c_ell = self.compute_theory_c_ell(cosmo, nz, sacc_data)
        #noise_c_ell = self.compute_noise_c_ell(n_eff, sigma_e, n_lens, sacc_data)

        # compute covariance
        #cov = self.compute_covariance(sacc_data, theory_c_ell, noise_c_ell, fsky)
        self.save_outputs(two_point_data, cov_cl)
        #self.save_outputs_firecrown(C,data_vector,ells,nz,binning_info)

    def save_outputs(self, two_point_data, cov):
        filename = self.get_output('summary_statistics')
        two_point_data.add_covariance(cov)
        two_point_data.save_fits(filename, overwrite=True)

    def read_cosmology(self):
        filename = self.get_input('fiducial_cosmology')
        cosmo = ccl.Cosmology.read_yaml(filename)

        print("COSMOLOGY OBJECT:")
        print(cosmo)
        return cosmo

    def read_sacc(self):
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

        # area in sq deg
        area_deg2 = maps_file['maps'].attrs['area']
        area_unit = maps_file['maps'].attrs['area_unit']
        if area_unit != 'sq deg':
            raise ValueError("Units of area have changed")
        area = area_deg2 * np.radians(1)**2

        print(f"area = {area_deg2:.1f} deg^2 = {area:.1f} sr")
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

"""
filename='./examples/des_y1_3x2pt/des_y1_3x2pt.yaml'
inp_config,inp_dat=parse(filename)

# cosmo = ccl.Cosmology(Omega_c = 0.27, Omega_b = 0.045, h = 0.67, sigma8 = 0.83, n_s = 0.96,transfer_function='boltzmann_class')
cosmo_param_names=['Omega_c', 'Omega_b', 'h', 'sigma8', 'n_s' ,'transfer_function']
cosmo_params={name:inp_config['parameters'][name] for name in cosmo_param_names}
cosmo = ccl.Cosmology(**cosmo_params)

twopoint_data = sacc.Sacc.load_fits(inp_config['two_point']['sacc_file']) #FROM FIRECROWN EXAMPLE.

#FIXME: f_sky, ell_bins and th_bins should be passed by sacc. ell0 and th can be decided based on binning or can also be passed by sacc.
twopoint_data.metadata['fsky']=0.3
twopoint_data.metadata['ell']=np.arange(2,500)
twopoint_data.metadata['ell_bins']=np.arange(2,500,20)
th_min=2.5/60 # in degrees
th_max=250./60
n_th_bins=20
twopoint_data.metadata['th_bins']=np.logspace(np.log10(th_min),np.log10(th_max),n_th_bins+1)

th=np.logspace(np.log10(th_min*0.98),np.log10(1),n_th_bins*30) #covariance is oversampled at th values and then binned.
th2=np.linspace(1,th_max*1.02,n_th_bins*30) #binned covariance can be sensitive to the th values. Make sue you check convergence for your application
# th2=np.logspace(np.log10(1),np.log10(th_max),60*6)
twopoint_data.metadata['th']=np.unique(np.sort(np.append(th,th2)))
thb=0.5*(twopoint_data.metadata['th_bins'][1:]+twopoint_data.metadata['th_bins'][:-1])

#The spin based factor to decide the wigner transform. Based on spin of tracers. Sometimes we may use s1_s2 to denote these factors
WT_factors={}
WT_factors['lens','source']=(0,2)
WT_factors['source','lens']=(2,0) #same as (0,2)
WT_factors['source','source']={'plus':(2,2),'minus':(2,-2)}
WT_factors['lens','lens']=(0,0)

"""
#this function will generate and return CCL_tracer objects and also compute the noise for all the tracers
def get_tracer_info(cosmo, two_point_data={}):
    ccl_tracers={}
    tracer_Noise={}
    for tracer in two_point_data.tracers:
        tracer_dat = two_point_data.get_tracer(tracer)
        z= tracer_dat.z
        #FIXME: Following should be read from sacc dataset.
        Ngal = 26. #arc_min^2
        sigma_e=.26
        b = 1.5*np.ones(len(z)) #Galaxy bias (constant with scale and z)
        AI = .5*np.ones(len(z)) #Galaxy bias (constant with scale and z)
        Ngal=Ngal*3600/d2r**2

        dNdz = tracer_dat.nz
        dNdz/=(dNdz*np.gradient(z)).sum()
        dNdz*=Ngal

        if 'source' in tracer or 'src' in tracer:
            ccl_tracers[tracer]=ccl.WeakLensingTracer(cosmo, dndz=(z, dNdz),ia_bias=(z,AI)) #CCL automatically normalizes dNdz
            tracer_Noise[tracer]=sigma_e**2/Ngal
        elif 'lens' in tracer:
            tracer_Noise[tracer]=1./Ngal
            ccl_tracers[tracer]=ccl.NumberCountsTracer(cosmo, has_rsd=False, dndz=(z,dNdz), bias=(z,b))
    return ccl_tracers,tracer_Noise

def get_cov_WT_spin(tracer_comb=None):
#     tracers=tuple(i.split('_')[0] for i in tracer_comb)
    tracers=[]
    for i in tracer_comb:
        if 'lens' in i:
            tracers+=['lens']
        if 'src' in i:
            tracers+=['source']
    return WT_factors[tuple(tracers)]




#compute a single covariance matrix for a given pair of C_ell or xi.  
def cl_gaussian_cov(cosmo, meta, tracer_comb1=None,tracer_comb2=None,ccl_tracers=None,tracer_Noise=None,two_point_data=None,do_xi=False,
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
    else: #do c_ell
        norm=(2*ell+1)*np.gradient(ell)*meta['fsky']

    coupling_mat={}
    coupling_mat[1324]=np.eye(len(ell)) #placeholder
    coupling_mat[1423]=np.eye(len(ell)) #placeholder

    cov={}
    cov[1324]=np.outer(cl[13]+SN[13],cl[24]+SN[24])*coupling_mat[1324]
    cov[1423]=np.outer(cl[14]+SN[14],cl[23]+SN[23])*coupling_mat[1423]

    cov['final']=cov[1423]+cov[1324]

    if do_xi:
        s1_s2_1=get_cov_WT_spin(tracer_comb=tracer_comb1)
        s1_s2_2=get_cov_WT_spin(tracer_comb=tracer_comb2)
        if isinstance(s1_s2_1,dict):
            s1_s2_1=s1_s2_1[xi_plus_minus1]
        if isinstance(s1_s2_2,dict):
            s1_s2_2=s1_s2_2[xi_plus_minus2]
        th,cov['final']=WT.projected_covariance2(l_cl=ell,s1_s2=s1_s2_1, s1_s2_cross=s1_s2_2,
                                                      cl_cov=cov['final'])

    cov['final']/=norm

    if do_xi:
        thb,cov['final_b']=bin_cov(r=th/d2r,r_bins=meta['th_bins'],cov=cov['final'])
    else:
        if meta['ell_bins'] is not None:
            lb,cov['final_b']=bin_cov(r=ell,r_bins=meta['ell_bins'],cov=cov['final'])

#     cov[1324]=None #if want to save memory
#     cov[1423]=None #if want to save memory
    return cov


#compute all the covariances and then combine them into one single giant matrix
def get_all_cov(cosmo, meta, two_point_data={},do_xi=False):
    #FIXME: Only input needed should be two_point_data, which is the sacc data file. Other parameters should be included within sacc and read from there.
    ccl_tracers,tracer_Noise=get_tracer_info(cosmo, two_point_data=two_point_data)
    tracer_combs=two_point_data.get_tracer_combinations()# we will loop over all these
    N2pt=len(tracer_combs)
    if meta['ell_bins'] is not None:
        Nell_bins=len(meta['ell_bins'])-1
    else:
        Nell_bins=len(meta['ell_bins'])
    if do_xi:
        Nell_bins=len(meta['th_bins'])-1
    cov_full=np.zeros((Nell_bins*N2pt,Nell_bins*N2pt))
    for i in np.arange(N2pt):
        tracer_comb1=tracer_combs[i]
        indx_i=i*Nell_bins
        for j in np.arange(i,N2pt):
            tracer_comb2=tracer_combs[j]
            indx_j=j*Nell_bins
            cov_ij=cl_gaussian_cov(cosmo, meta, tracer_comb1=tracer_comb1,tracer_comb2=tracer_comb2,ccl_tracers=ccl_tracers,
                                        tracer_Noise=tracer_Noise,do_xi=do_xi,two_point_data=two_point_data)
            if do_xi or meta['th_bins'] is not None:
                cov_ij=cov_ij['final_b']
            else:
                cov_ij=cov_ij['final']
            cov_full[indx_i:indx_i+Nell_bins,indx_j:indx_j+Nell_bins]=cov_ij
            cov_full[indx_j:indx_j+Nell_bins,indx_i:indx_i+Nell_bins]=cov_ij.T
    return cov_full


