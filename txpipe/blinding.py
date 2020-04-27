from .base_stage import PipelineStage
from .data_types import SACCFile
import numpy as np
import warnings

class TXBlinding(PipelineStage):
    """
    Blinding the data vectors.

    """
    name='TXBlinding'
    inputs = [
        ('twopoint_data_real_raw', SACCFile),
    ]
    outputs = [
        ('twopoint_data_real', SACCFile),
    ]
    config_options = {
        'seed': 1972,  ## seed uniquely specifies the shift in parameters
        'Omega_b': [0.0485, 0.001], ## fiducial_model_value, shift_sigma
        'Omega_c': [0.2545, 0.01],
        'w0': [-1.0, 0.1],
        'h': [0.682, 0.02],
        'sigma8': [0.801, 0.01],
        'n_s': [0.971, 0.03],
        'b0': 0.95,  ### we assume bias to be of the form b0/growth
        'delete_unblinded': False 
    }


    def run(self):
        """
        Run the analysis for this stage.
        
         - Load two point SACC file
         - Blinding it 
         - Output blinded data
         - Optionally deletete unblinded data
        """
        import sacc, sys, os, shutil
        sys.stdout.flush()

        unblinded_fname = self.get_input('twopoint_data_real_raw')
        sacc = sacc.Sacc.load_fits(unblinded_fname)
        blinded_sacc = self.blind_muir(sacc)
        print ("Writing a very small sacc file [on NERSC: 1200 baud teetu-teetu-shhhhh]")
        blinded_sacc.save_fits(self.get_output('twopoint_data_real'), overwrite=True)
        if self.config['delete_unblinded']:
            print(f"Replacing {unblinded_fname} with empty...")
            open (unblinded_fname,'w').close()
                

    def blind_muir(self, sacc):
        import pyccl as ccl
        import firecrown
        import io
        import copy
        ## here we actually do blinding
        print(f"Blinding... ")
        np.random.seed(self.config["seed"])
        # blind signature -- this ensures seed is consistent across
        # numpy versions
        blind_sig = ''.join(format(x, '02x') for x in np.random.bytes(4))
        if self.rank==0:
            print(f"Blinding signature: %s"%(blind_sig))


        fid_params = {
            'Omega_b':  self.config['Omega_b'][0],
            'Omega_c':  self.config['Omega_c'][0],
            'h': self.config['h'][0],
            'w0': self.config['w0'][0],
            'sigma8': self.config['sigma8'][0],
            'n_s': self.config['n_s'][0],
        }
        ## now get biases
        bz={}
        fidCosmo=ccl.Cosmology(**fid_params)
        for key,tracer in sacc.tracers.items():
            if 'lens' in key:
                zeff = (tracer.z*tracer.nz).sum()/tracer.nz.sum()
                bz[key] = self.config['b0']/ccl.growth_factor(fidCosmo,1/(1+zeff)) 
        
        offset_params = fid_params.copy()
        for par in fid_params.keys():
            offset_params [par] += self.config[par][1]*np.random.normal(0.,1.)
        fc_config = {
            'parameters': {
                'Omega_k': 0.0,
                'wa': 0.0,
                'one': 1},
            'two_point': {
               'module' : 'firecrown.ccl.two_point',
               'sacc_data' : sacc,
               'systematics' : {
                   'dummy' : {
                       'kind' : 'PhotoZShiftBias',
                       'delta_z' : 'one' }},
                'sources' : {},
                'statistics' : {}
                }
        }


        for k,v in bz.items():
            fc_config ['parameters']['bias_%s'%k] = v
            
        srclist=[]
        lenslist=[]
        for key,tracer in sacc.tracers.items():
            ## This is a hack, need to think how to do this better
            if 'source' in key:
                fc_config['two_point']['sources'][key] = {
                                     'kind' : 'WLSource',
                                     'sacc_tracer' : key}
                srclist.append(key)
            if 'lens' in key:
                fc_config['two_point']['sources'][key] = {
                                     'kind' : 'NumberCountsSource',
                                     'bias' : 'bias_%s'%key,
                                     'sacc_tracer' : key}
                lenslist.append(key)

        ## list(dict.fromkeys()) gives unique elements while preserving order
        types=list(dict.fromkeys([(p.data_type, p.tracers,
                                   "{dtype}_{tracer1}_{tracer2}".format(dtype=p.data_type,
                                            tracer1=p.tracers[0],tracer2=p.tracers[1]))
                                   for p in sacc.data]))
        for dtype,(tracer1,tracer2),fcname in types:
            fc_config['two_point']['statistics'][fcname] = {
                                   'sources' : [tracer1, tracer2],
                                   'sacc_data_type' : dtype}

        ## now try to get predictions
        pred={}
        for name,pars in [('fid',fid_params),('ofs',offset_params)]:
            print ("Calling firecrown : %s"%(name))
            fc_config['parameters'].update(pars)
            config, data = firecrown.parse(fc_config)
            cosmo = firecrown.get_ccl_cosmology(config['parameters'])
            firecrown.compute_loglike(cosmo=cosmo, data=data)
            pred[name] = np.hstack([data['two_point']['data']['statistics'][n].predicted_statistic_ for
                                  _,_,n in types])
            ## if there is some sanity this should work
            assert(len(pred[name])==len(sacc))

        diffvec = pred['ofs']-pred['fid']
        ## now add offsets
        for p, delta in zip(sacc.data,diffvec):
            p.value+=delta
        print(f"Blinding done.")
            
        return sacc

            
