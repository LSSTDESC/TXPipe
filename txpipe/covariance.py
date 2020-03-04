
from .base_stage import PipelineStage
from .data_types import MetacalCatalog, HDFFile, YamlFile, SACCFile, TomographyCatalog, CSVFile
from .data_types import DiagnosticMaps
import numpy as np
import warnings
import os

# require TJPCov to be in PYTHONPATH
d2r=np.pi/180



#Needed changes: 1) area is hard coded to 4sq.deg. as file is buggy. 2) code fixed to equal-spaced ell values in real space. 3)

class TXFourierGaussianCovariance(PipelineStage):
    name='TXFourierGaussianCovariance'
    do_xi=False,

    inputs = [
        ('fiducial_cosmology', YamlFile),    # For the cosmological parameters
        ('photoz_stack', HDFFile),           # For the n(z)
        ('twopoint_data_fourier', SACCFile), # For the binning information
        ('tracer_metadata', HDFFile),         # For metadata
    ]

    outputs = [
        ('summary_statistics_fourier', SACCFile),
    ]

    config_options = {
    }

    def run(self):
        import pyccl as ccl
        import sacc
        import wigner_transform
        # Okay, this is a slightly psychotic hack, and a better way
        # is to make a bunch of changes to ceci, and if we grow that code
        # we should do that.  But right now the only things ceci knows
        # about threads is the OMP_NUM_THREADS variable, so we want to use
        # that to decide how many processes to us in multiprocessing.
        # But we don't want to accidentally parallelize twice (e.g. in numpy),
        # so we're going to set our ncpu from OMP_NUM_THREADS and then override
        # that variable so OMP isn't actually used.
        # TODO: make this better
        num_processes = int(os.environ.get("OMP_NUM_THREADS", 1))
        os.environ['OMP_NUM_THREADS'] = '1'

        # read the fiducial cosmology
        cosmo = self.read_cosmology()

        # read binning
        two_point_data = self.read_sacc()

        # read the n(z) and f_sky from the source summary stats        
        n_eff, n_lens, sigma_e, fsky = self.read_number_statistics()
        
        # the following is a list of things that are somewhat awkwardly passed through the functions... think about how this can be done more elegantly.
        meta = {}
        meta['fsky'] = fsky
        meta['ell'] = np.arange(2,5000) # this needs to be tested
        meta['th'] = np.logspace(np.log10(0.1/60),np.log10(600./60),200) # this needs to be tested
        meta['sigma_e'] = sigma_e
        meta['n_eff'] = n_eff # per radian2
        meta['n_lens'] = n_lens # per radian2

        #C_ell covariance
        cov = self.get_all_cov(cosmo, meta, two_point_data=two_point_data, num_processes=num_processes)
        
        self.save_outputs(two_point_data, cov)

    def save_outputs(self, two_point_data, cov):
        filename = self.get_output('summary_statistics_fourier')
        two_point_data.add_covariance(cov)
        two_point_data.save_fits(filename, overwrite=True)

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
        two_point_data = sacc.Sacc.load_fits(f)

        # Remove the data types that we won't use for inference
        mask = [
            two_point_data.indices(sacc.standard_types.galaxy_shear_cl_ee),
            two_point_data.indices(sacc.standard_types.galaxy_shearDensity_cl_e),
            two_point_data.indices(sacc.standard_types.galaxy_density_cl),
            # not doing b-modes, do we want to?
        ]
        mask = np.concatenate(mask)
        two_point_data.keep_indices(mask)

        two_point_data.to_canonical_order()

        return two_point_data


    def read_number_statistics(self):
        input_data = self.open_input('tracer_metadata')

        N_eff = input_data['tracers/N_eff']
        N_lens = input_data['tracers/lens_counts']

        # area in sq deg
        area_deg2 = input_data['tracers'].attrs['area']
        area_unit = input_data['tracers'].attrs['area_unit']
        if area_unit != 'sq deg':
            raise ValueError("Units of area have changed")
        area = area_deg2 * np.radians(1)**2

        print(f"area = {area_deg2:.1f} deg^2")
        print("NEFF : ", N_eff)
        n_eff = N_eff / area
        print("nEFF : ", n_eff)
        sigma_e = input_data['tracers/sigma_e'][:]

        n_lens = N_lens / area
        print("lens density : ", n_lens)

        full_sky=4*np.pi #*(180./np.pi)**2 #(FULL SKY IN STERADIANS)
        fsky=area/full_sky

        input_data.close()

        print('n_lens, n_eff, sigma_e, fsky: ')
        print(n_lens, n_eff, sigma_e, fsky)
        return n_eff, n_lens, sigma_e, fsky

    def get_tracer_info(self, cosmo, meta, two_point_data):
        import pyccl as ccl
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
                IA = 1.0*np.ones(len(z)) # place holder
                Ngal = meta['n_eff'][nbin]
                # check that this normalization is correct
                #Ngal = Ngal*3600/d2r**2
                #dNdz *= Ngal
                
                ccl_tracers[tracer]=ccl.WeakLensingTracer(cosmo, dndz=(z, dNdz),ia_bias=(z,IA)) #CCL automatically normalizes dNdz
                tracer_Noise[tracer]=sigma_e**2/Ngal
            
            elif 'lens' in tracer:
                b = 1.0*np.ones(len(z))  # place holder
                Ngal = meta['n_lens'][nbin]
                #Ngal = Ngal*3600/d2r**2
                #dNdz *= Ngal
                tracer_Noise[tracer]=1./Ngal
                ccl_tracers[tracer]=ccl.NumberCountsTracer(cosmo, has_rsd=False, dndz=(z,dNdz), bias=(z,b))
        
        return ccl_tracers,tracer_Noise

    def get_cov_WT_spin(self, tracer_comb=None):

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
    def cl_gaussian_cov(self, cosmo, meta, ell_bins, 
        tracer_comb1=None, tracer_comb2=None, ccl_tracers=None, tracer_Noise=None,
        two_point_data=None,
        xi_plus_minus1='plus', xi_plus_minus2='plus', num_processes=1):
        import pyccl as ccl
        from wigner_transform import bin_cov, wigner_transform

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

        if self.do_xi:
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

        if self.do_xi:
            s1_s2_1 = self.get_cov_WT_spin(tracer_comb=tracer_comb1)
            s1_s2_2 = self.get_cov_WT_spin(tracer_comb=tracer_comb2)
            if isinstance(s1_s2_1,dict):
                s1_s2_1=s1_s2_1[xi_plus_minus1]
            if isinstance(s1_s2_2,dict):
                s1_s2_2=s1_s2_2[xi_plus_minus2]


            WT_kwargs = {
                'l': ell,
                'theta': meta['th']*d2r,
                's1_s2':[(2,2),(2,-2),(0,2),(2,0),(0,0)],
                'ncpu': num_processes,
            }
            WT = wigner_transform(**WT_kwargs)

            th, cov['final']=WT.projected_covariance2(l_cl=ell,s1_s2=s1_s2_1, s1_s2_cross=s1_s2_2,
                                                      cl_cov=cov['final'])

        cov['final']/=norm

        if self.do_xi:
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
        # this should be changed to read from sacc file
        th_arcmin = np.logspace(np.log10(self.config['min_sep']), np.log10(self.config['max_sep']), self.config['nbins']+1)
        return th_arcmin/60.0

    #compute all the covariances and then combine them into one single giant matrix
    def get_all_cov(self, cosmo, meta, two_point_data={}, num_processes=1):
        #FIXME: Only input needed should be two_point_data, which is the sacc data file. Other parameters should be included within sacc and read from there.
        ccl_tracers,tracer_Noise = self.get_tracer_info(cosmo, meta, two_point_data=two_point_data)
        tracer_combs = two_point_data.get_tracer_combinations() # we will loop over all these
        N2pt = len(tracer_combs)
        
        N2pt0 = -1
        if self.do_xi:
            N2pt0 = N2pt*1
            tracer_combs_temp = tracer_combs.copy()
            for combo in tracer_combs:
                if ('source' in combo[0]) and ('source' in combo[1]):
                    N2pt+=1
                    tracer_combs_temp+=[combo]
            tracer_combs = tracer_combs_temp.copy()

        if not self.do_xi:
            ell_bins = self.get_ell_bins(two_point_data)
        else:
            ell_bins = self.get_th_bins(two_point_data)

        if ell_bins is not None:
            Nell_bins = len(ell_bins)-1
        else: 
            Nell_bins = len(ell_bins)

        cov_full=np.zeros((Nell_bins*N2pt,Nell_bins*N2pt))
        count_xi_pm1 = 0
        count_xi_pm2 = 0
        xi_pm = [[('plus','plus'), ('plus', 'minus')], [('minus','plus'), ('minus', 'minus')]]
        for i in np.arange(N2pt):
            tracer_comb1=tracer_combs[i]
            indx_i=i*Nell_bins
            if i==N2pt0:
                count_xi_pm1 = 1
            for j in np.arange(i,N2pt):
                tracer_comb2=tracer_combs[j]
                indx_j=j*Nell_bins
                if j==N2pt0:
                    count_xi_pm2 = 1
                if self.do_xi and ('source' in tracer_comb1) and ('source' in tracer_comb2):
                    cov_ij = self.cl_gaussian_cov(
                        cosmo,
                        meta,
                        ell_bins, 
                        tracer_comb1=tracer_comb1,
                        tracer_comb2=tracer_comb2,
                        ccl_tracers=ccl_tracers,
                        tracer_Noise=tracer_Noise,
                        two_point_data=two_point_data,
                        xi_plus_minus1=xi_pm[count_xi_pm1,count_xi_pm2][0],
                        xi_plus_minus2=xi_pm[count_xi_pm1,count_xi_pm2][1],
                        num_processes=num_processes,
                    )

                else:
                    cov_ij = self.cl_gaussian_cov(
                        cosmo,
                        meta,
                        ell_bins,
                        tracer_comb1=tracer_comb1,
                        tracer_comb2=tracer_comb2,
                        ccl_tracers=ccl_tracers,
                        tracer_Noise=tracer_Noise,
                        two_point_data=two_point_data,
                        num_processes=num_processes,
                    )

                cov_ij=cov_ij['final_b']
                cov_full[indx_i:indx_i+Nell_bins,indx_j:indx_j+Nell_bins]=cov_ij
                cov_full[indx_j:indx_j+Nell_bins,indx_i:indx_i+Nell_bins]=cov_ij.T
        return cov_full



class TXRealGaussianCovariance(TXFourierGaussianCovariance):
    name='TXRealGaussianCovariance'
    do_xi = True

    inputs = [
        ('fiducial_cosmology', YamlFile),     # For the cosmological parameters
        ('photoz_stack', HDFFile),            # For the n(z)
        ('twopoint_data_real', SACCFile),     # For the binning information
        ('tracer_metadata', HDFFile),         # For metadata

    ]

    outputs = [
        ('summary_statistics_real', SACCFile),
    ]

    config_options = {
        'min_sep':2.5,  # arcmin
        'max_sep':250,
        'nbins':20,
            }

    def run(self):
        super().run()

    def read_sacc(self):
        import sacc
        f = self.get_input('twopoint_data_real')
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
        filename = self.get_output('summary_statistics_real')
        two_point_data.add_covariance(cov)
        two_point_data.save_fits(filename, overwrite=True)
