from ceci import PipelineStage
from .data_types import MetacalCatalog, TomographyCatalog, RandomsCatalog, YamlFile, SACCFile, DiagnosticMaps
import numpy as np
import pymaster as nmt
from .utils import choose_pixelization, HealpixScheme, GnomonicPixelScheme, ParallelStatsCalculator

def source_mapper_iterator(shear_it,bin_it,ms_it,pixel_scheme,bins) :
    """Joint loop through shear and tomography catalogs and bin
    them into tomographic bin, map type and map pixel

    This is later on passed to the stats calculator.
    
    Parameters
    ----------
    shear_it: iterable
        Iterator providing chunks of the shear catalog
    
    bin_it: iterable
        Iterator providing chunks of the tomographic bin column
    
    ms_it: iterable
        Iterator providing chunks of the multiplicative bias column
    
    pixel_scheme: HealpixScheme or GnomonicPixelScheme object
        Pixelization scheme of the output maps
    
    bins: array_like
        List of allowed tomographic bin indices
    
    Yields
    ------
    index: int
        Integer that places each objects in a given pixel of a
        given tomographic bin of a particular map type
        (e.g. number counts or shear).
    weights: array_like
        Quantity to be histogrammed by the stats calculator.
    """
       
    #Organize tasks, bins and pixels
    tasks=[0,1,2] #0-> density, 1-> gamma_1, 2-> gamma_2
    npix=pixel_scheme.npix
    nbins=len(bins)
    ntasks=len(tasks)
    #We will generate nbins*ntasks maps with npix pixels each

    #Loop through data
    for sh,bn,ms in zip(shear_it,bin_it,ms_it) :
        #Get pixel indices
        pix_nums=pixel_scheme.ang2pix(sh['ra'],sh['dec'])

        for p in np.unique(pix_nums) : #Loop through pixels
            if p<0 or p>=npix :
                continue
            mask_pix=(pix_nums==p)
            for i_t,task in enumerate(tasks) : #Loop through tasks (number counts, gamma_x)
                if i_t==0 : 
                    #Number counts.
                    #TODO: update this to take any weights into account if needed.
                    #TODO: ParallelStatsCalculator may need to be updated with a weights array.
                    w=np.ones_like(sh['ra'])
                else :
                    #Shear
                    #TODO: account for all shape-measurement weights.
                    w=sh['mcal_g'][:,i_t-1]
                for ib,b in enumerate(bins) : #Loop through tomographic bins
                    mask_bins=(bn['source_bin']==b)
                    index=p+npix*(i_t+ntasks*ib)
                    yield index,w[mask_pix & mask_bins]


def source_mapper(shear_it,bin_it,ms_it,pixel_scheme,bins,comm=None):
    """Produces number counts and shear maps from shear and tomography catalogs.
    
    Parameters
    ----------
    shear_it: iterable
        Iterator providing chunks of the shear catalog
    
    bin_it: iterable
        Iterator providing chunks of the tomographic bin column
    
    ms_it: iterable
        Iterator providing chunks of the multiplicative bias column
    
    pixel_scheme: HealpixScheme or GnomonicPixelScheme object
        Pixelization scheme of the output maps
    
    bins: array_like
        List of allowed tomographic bin indices
    
    comm: MPI communicator, optional
        An MPI comm for parallel processing.  If None, calculation is serial.

    Returns
    ------
    count: array_like
        3D array with dimensions [nbins,ntasks,npix], where npix is the total
        number of pixels in the map, ntasks=3 (number counts, gamma_1, gamma_2)
        and nbins is the number of tomographic bins considered.
        This array contains the number of objects used to estimate the map
        at each pixel.
    w_mean: array_like
        Array with the same dimensions as count. It contains the mean value of
        the quantity appropriate for each task, i.e. mean galaxy weight for
        number counts, mean shear_1 and shear_2 for gamma_1 and gamma_2.
    """
    #Organize maps structure
    ntasks=3 #Number counts, gamma_1 and gamma_2
    nbins=len(bins)
    npix=pixel_scheme.npix
    npix_all=npix*nbins*ntasks #Total number of pixels.

    #Initialize stats calculator
    stats=ParallelStatsCalculator(npix_all, sparse=False)

    #Initialize mapping iterator
    weights_iterator=source_mapper_iterator(shear_it,bin_it,ms_it,pixel_scheme,bins)

    #Calculate map
    count,w_mean,w_std=stats.calculate(weights_iterator,comm)

    # Reshape 1D histograms into maps.  Support both 1D healpix maps
    # and 2D flat maps
    shape = (nbins,ntasks,) + pixel_scheme.shape
    count=count.reshape(shape)
    w_mean=w_mean.reshape(shape)
    w_std=w_std.reshape(shape)

    return count,w_mean

def maps2fields(count,mean,mask,syst_nc,syst_wl,pixel_scheme,mask_thr=0.) :
    """Generates NaMaster's NmtField objects based on input maps.
    
    Parameters
    ----------
    count: array_like
        3D array with dimensions [nbins,ntasks,npix], where npix is the total
        number of pixels in the map, ntasks=3 (number counts, gamma_1, gamma_2)
        and nbins is the number of tomographic bins considered.
        This array contains the number of objects used to estimate the map
        at each pixel.
    
    mean: array_like
        Array with the same dimensions as count. It contains the mean value of
        the quantity appropriate for each task, i.e. mean galaxy weight for
        number counts, mean shear_1 and shear_2 for gamma_1 and gamma_2.
    
    mask: array_like
        Sky mask. For number counts, it is interpreted as a masked fraction
        (and corrected for).

    syst_nc, syst_wl: array_like
        List of systematics maps for number counts and weak lensing.
        The dimensions of these should be [n_syst,n_maps,n_pix], where 
        n_syst is the number of systematics to account for, n_maps=1 
        for number counts and n_maps=2 for lensing (different spins)
        and n_pix is the size of each map.

    field_class: NmtField or NmtFieldClass
        Type of NmtField class to use.

    mask_thr: float, optional
        Throw away any pixels where mask <= mask_thr

    Returns
    -------
    dfields: array_like
        Array of spin-0 NmtField objects for number counts. One per
        tomographic bin.
    wfields: array_like
        Array of spin-2 NmtField objects for weak lensing. One per
        tomographic bin.
    """
    #TODO: this whole function could be parallelized

    #Determine number of bins
    nbins = mean.shape[0]
    ntasks = mean.shape[1]

    if ntasks!=3 :
        #TODO: right now we need both lensing and clustering. This should be more flexible"
        raise ValueError("We need maps for both lensing and clustering")
    
    #Find unmasked pixels
    ipix_mask_good=np.where(mask>mask_thr)[0]
    #Neglect pixels below mask threshold
    mask[mask<=mask_thr]=0


    #Maps for number counts, one per tomography bin
    nmap=count[:,0,:]*mean[:,0,:]
    nmean=np.sum(nmap*mask[None,:])/np.sum(mask)
    dmap=np.zeros_like(nmap)
    for b in range(nbins):
        dmap[b, ipix_mask_good]=nmap[b,ipix_mask_good]/(nmean*mask[ipix_mask_good])-1 #delta_g

    if pixel_scheme.name == 'gnomonic':
        lx = np.radians(pixel_scheme.size_x)
        ly = np.radians(pixel_scheme.size_y)
        print(f"lx = {lx}")
        # Density
        dfields=[nmt.NmtFieldFlat(lx,ly,mask,[d],templates=syst_nc) for d in dmap]
        # Lensing
        wfields=[nmt.NmtFieldFlat(lx, ly, mask,[m[1],m[2]],templates=syst_wl) for m in mean]

    elif pixel_scheme.name == 'healpix':
        # Density
        dfields=[nmt.NmtField(mask,[d],syst_nc) for d in dmap]
        # Lensing
        wfields=[nmt.NmtField(mask,[m[1],m[2]],syst_wl) for m in mean]
    else:
        raise ValueError(f"Pixelization scheme {pixel_scheme.name} not supported by NaMaster")

    
    return dfields,wfields
        
class TXTwoPointFourier(PipelineStage):
    """This Pipeline Stage computes all auto- and cross-correlations
    for a list of tomographic bins, including all galaxy-galaxy,
    galaxy-shear and shear-shear power spectra. Sources and lenses
    both come from the same shear_catalog and tomography_catalog objects.

    The power spectra are computed after deprojecting a number of
    systematic-dominated modes, represented as input maps.

    In the future we will want to include the following generalizations:
     - TODO: specify different bins for shear and clustering.
     - TODO: different source and lens catalogs.
     - TODO: specify which cross-correlations in particular to include
             (e.g. which bin pairs and which source/lens combinations).
     - TODO: include flags for rejected objects. Is this included in 
             the tomography_catalog?
     - TODO: final interface with SACC is still missing.
     - TODO: make sure NaMaster works in python-3.6
     - TODO: ell-binning is currently static.
    """
    name='TXTwoPointFourier'
    inputs = [
        ('shear_catalog',MetacalCatalog), #Shear catalog
        ('tomography_catalog', TomographyCatalog), #Tomography catalog
        ('diagnostic_maps',DiagnosticMaps), #Mask (generated by TXSysMapMaker)
    ]
    outputs = [
        ('twopoint_data', SACCFile)
    ]

    config_options = {
        "chunk_rows": 10000
    }
    
    def run(self) :
        config = self.config

        diagnostic_map_file = self.open_input('diagnostic_maps', wrapper=True)

        mask_info = diagnostic_map_file.read_map_info('mask')
        pix_schm = choose_pixelization(**mask_info)

        #Choose pixelization and read mask and systematics maps
        pixelization=mask_info['pixelization'].lower()

        mask = diagnostic_map_file.read_map('mask')

        syst_nc = None
        syst_wl = None
        #Read mask and systematic maps
        # if config['syst_nc']!='none' :
        #     #Should we check for this or for whether it exists?
        #     #Is this actually in config?
        #     dum,syst_nc=mapreading(config['syst_nc'],i_map=None)
        #     nmaps=len(syst_nc)
        #     syst_nc=syst_nc.reshape([nmaps,1,pix_schm.npix])
        # else :
        #     syst_nc=None
        # if config['syst_wl']!='none' :
        #     #Should we check for this or for whether it exists?
        #     #Is this actually in config?
        #     dum,syst_wl=mapreading(config['syst_wl'],i_map=None)
        #     if len(syst_wl)%2!=0 :
        #         raise ValueError("There must be an even number of systematics maps for weak lensing")
        #     nmaps=len(syst_nc)/2
        #     syst_wl=syst_nc.reshape([nmaps,2,pix_schm.npix])
        # else :
        #     syst_wl=None

        #Generate iterators for shear and tomography catalogs
        cols_shear=['ra','dec','mcal_g','mcal_flags','mcal_mag',
                    'mcal_s2n_r','mcal_T']

        # Get some metadat from the tomography file
        tomo_file = self.open_input('tomography_catalog', wrapper=True)
        nbin_source = tomo_file.read_nbin('source')
        nbin_lens = tomo_file.read_nbin('lens')
        source_bins = list(range(nbin_source))
        lens_bins = list(range(nbin_lens))
        tomo_file.close()

        shear_iterator=(data for start,end,data in self.iterate_fits('shear_catalog',
                                                                     1,
                                                                     cols_shear,
                                                                     config['chunk_rows']))
        bin_iterator=(data for start,end,data in self.iterate_hdf('tomography_catalog',
                                                                  'tomography',
                                                                  ['source_bin'],
                                                                  config['chunk_rows']))
        ms_iterator=(data for start,end,data in self.iterate_hdf('tomography_catalog',
                                                                 'multiplicative_bias',
                                                                 ['R_gamma'],
                                                                 config['chunk_rows']))

        #Generate maps
        m_count,m_mean=source_mapper(shear_iterator,bin_iterator,ms_iterator,pix_schm,source_bins,self.comm)
        print("Made source maps")
        #Generate namaster fields
        f_d,f_w=maps2fields(m_count,m_mean,mask,syst_nc,syst_wl, pix_schm)
        print("Converted maps to NaMaster fields")

        ell_bins = self.choose_ell_bins(pix_schm)
        print("Chosen pix scheme")

        w00, w02, w22 = self.setup_workspaces(pix_schm, f_d, f_w, ell_bins)
        print("Computed workspaces")

        c_ll, c_dl, c_dd = self.compute_power_spectra(pix_schm, nbin_source, nbin_lens, f_w, f_d, w00, w02, w22, ell_bins)
        print("Computed all power spectra")

        #Save power spectra
        #TODO: SACC interface here
        self.save_power_spectra(c_ll, c_dl, c_dd)
        print("Saved power spectra")

        #Binning scheme
        #TODO: set ell binning from config

    def choose_ell_bins(self, pix_schm):
        if pix_schm.name == 'healpix':
            nlb=min(1,int(1./np.mean(mask)))
            ell_bins=nmt.NmtBin(pix_schm.nside,nlb=nlb)
        elif pix_schm.name == 'gnomonic':
            lx=np.radians(pix_schm.nx*pix_schm.pixel_size_x)
            ly=np.radians(pix_schm.ny*pix_schm.pixel_size_y)
            ell_min=max(2*np.pi/lx,2*np.pi/ly)
            ell_max=min(pix_schm.nx*np.pi/lx,pix_schm.ny*np.pi/ly)
            d_ell=2*ell_min
            n_ell=int((ell_max-ell_min)/d_ell)-1
            l_bpw=np.zeros([2,n_ell])
            l_bpw[0,:]=ell_min+np.arange(n_ell)*d_ell
            l_bpw[1,:]=l_bpw[0,:]+d_ell
            ell_bins=nmt.NmtBinFlat(l_bpw[0,:],l_bpw[1,:])
            # for k,v in locals().items():
            #     print(f"{k}: {v}")
        return ell_bins


    def setup_workspaces(self, pix_schm, f_d, f_w, ell_bins):
        # choose scheme class
        if pix_schm.name == 'healpix':
            workspace_class = nmt.NmtWorkspace
        elif pix_schm.name == 'gnomonic':
            workspace_class = nmt.NmtWorkspaceFlat
        else:
            raise ValueError(f"No NaMaster workspace for pixel scheme {pix_schm.name}")

        #Compute mode-coupling matrix
        #TODO: mode-coupling could be pre-computed and provided in config.
        w00=workspace_class()
        w00.compute_coupling_matrix(f_d[0],f_d[0],ell_bins);

        w02=workspace_class()
        w02.compute_coupling_matrix(f_d[0],f_w[0],ell_bins);

        w22=workspace_class()
        w22.compute_coupling_matrix(f_w[0],f_w[0],ell_bins);

        return w00, w02, w22

    def compute_power_spectra(self, pix_schm, nbin_source, nbin_lens, f_w, f_d, w00, w02, w22, ell_bins):
        #Compute power spectra
        #TODO: now all possible auto- and cross-correlation are computed.
        #      This should be tunable.
        c_dd={}
        c_dl={}
        c_ll={}

        # TODO: parallelize this bit
        for i in range(nbin_source):
            for j in range(i+1):
                print(f"Calculating shear-shear bin pair ({i},{j})")
                c = self.compute_one_spectrum(pix_schm, w22, f_w[i], f_w[j], ell_bins)
                #We neglect the EB and BE cross-correlations
                c_ll[(i,j)] = c[0],c[3] 
                
        for i in range(nbin_lens):
            for j in range(i+1):
                print(f"Calculating position-position bin pair ({i},{j})")
                c = self.compute_one_spectrum(pix_schm, w00, f_d[i], f_d[j], ell_bins)
                c_dd[(i,j)] = c[0]

        for i in range(nbin_source):
            for j in range(nbin_lens):
                print(f"Calculating position-shear bin pair ({i},{j})")
                c = self.compute_one_spectrum(pix_schm, w02, f_w[i], f_d[j], ell_bins)                
                c_dl[(i,j)] = c[0],c[1]

        return c_ll, c_dl, c_dd

    def compute_one_spectrum(self, pix_scheme, w, f1, f2, ell_bins):
        if pix_scheme.name == 'healpix':
            coupled_c_ell = nmt.compute_coupled_cell(f1, f2)
        elif pix_scheme.name == 'gnomonic':
            coupled_c_ell = nmt.compute_coupled_cell_flat(f1, f2, ell_bins)

        c_ell = w.decouple_cell(coupled_c_ell)
        return c_ell


    def save_power_spectra(self, c_ll, c_dl, c_dd):
        print("TODO: Save spectra")
        print(c_ll)
        print(c_dl)
        print(c_dd)



if __name__ == '__main__':
    PipelineStage.main()
