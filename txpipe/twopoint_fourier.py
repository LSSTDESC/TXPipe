from ceci import PipelineStage
from descformats.tx import MetacalCatalog, TomographyCatalog, RandomsCatalog, YamlFile, SACCFile
import numpy as np
import pymaster as nmt
from .utils import HealpixScheme, GnomonicPixelScheme, ParallelStatsCalculator

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
        pix_nums=pixel_scheme.ang2pix(s['ra'],s['dec'])

        for p in np.unique(pix_nums) : #Loop through pixels
            if p<0 or p>=npix :
                continue
            mask_pix=(pix_nums==p) :
            for i_t,task in enumerate(tasks) : #Loop through tasks (number counts, gamma_x)
                if i_t==0 : 
                    #Number counts.
                    #TODO: update this to take any weights into account if needed.
                    #TODO: ParallelStatsCalculator may need to be updated with a weights array.
                    w=np.ones_like(s['ra'])
                else :
                    #Shear
                    #TODO: account for all shape-measurement weights.
                    w=s['mcal_g'][:,i_t-1]
                for ib,b in enumerate(bins) : #Loop through tomographic bins
                    mask_bins=(b['bin']==b)
                    index=p+npix*(i_t+ntasks*ib)
                    yield index,weights[mask_pix & mask_bins]


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

    #Reshape 1d histograms into maps
    count=count.reshape([nbins,ntasks,npix])
    w_mean=w_mean.reshape([nbins,ntasks,npix])
    w_std=w_std.reshape([nbins,ntasks,npix])

    return count,w_mean

def maps2fields(count,mean,mask,syst_nc,syst_wl,field_class,mask_thr=0.) :
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
    nbins,ntasks,npix=mean.shape

    if ntasks!=3 :
        #TODO: right now we need both lensing and clustering. This should be more flexible"
        raise ValueError("We need maps for both lensing and clustering")
    
    #Find unmasked pixels
    ipix_mask_good=np.where(mask>mask_thr)[0]
    #Neglect pixels below mask threshold
    mask[mask<=mask_thr]=0

    #Maps for number counts
    nmap=count[:,0,:]*mean[:,0,:]
    nmean=np.sum(nmap*mask[None,:])/np.sum(mask)
    dmap=np.zeros_like(nmap);
    dmap[ipix_mask_good]=nmap[ipix_mask_good]/(nmean*mask[ipix_mask_good])-1 #delta_g
    dfields=[field_class(mask,[d],syst_nc) for d in dmap]

    #Maps for weak lensing
    wfields=[field_class(mask,[m[1],m[2]],syst_wl) for m in mean]
    
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
        ('mask',DiagnosticMap), #Mask (generated by TXSysMapMaker)
        ('syst_nc',DiagnosticMap), #Systematics maps for clustering (nc==number counts). 
                                   #Generated by TXSysMapMaker.
        ('syst_wl',DiagnosticMap), #Systematics maps for shear (wl==weak lensing).
                                   #Generated by TXSysMapMaker.
        ('config',YamlFile),
    ]
    outputs = [
        ('twopoint_data', SACCFile)
    ]

    #TODO: why do we need to specify this here?
    config_options = {
        'pixelization': 'healpix' #The pixelization scheme to use
    }
    
    def run(self) :
        config = self.read_config()

        #Choose pixelization and read mask and systematics maps
        pixelization=config['pixelization'].lower()
        if pixelization=='healpix' :
            pixtype=0
            mapreading=HealpixScheme.read_map
            FieldClass=nmt.NmtField
        elif pixelization=='gnomonic' or pixelization=='tan' or pixelization=='tangent':
            pixtype=1
            map_reading=GnomonicPixelScheme.read_map
            FieldClass=nmt.NmtFieldFlat
        else :
            raise ValueError("Pixelization scheme unknown")
     
        #Read mask and systematic maps
        pix_schm,mask=mapreading(config['mask'])
        if config['syst_nc']!='none' :
            #Should we check for this or for whether it exists?
            #Is this actually in config?
            dum,syst_nc=mapreading(config['syst_nc'],i_map=None)
            nmaps=len(syst_nc)
            syst_nc=syst_nc.reshape([nmaps,1,pix_schm.npix])
        else :
            syst_nc=None
        if config['syst_wl']!='none' :
            #Should we check for this or for whether it exists?
            #Is this actually in config?
            dum,syst_wl=mapreading(config['syst_wl'],i_map=None)
            if len(syst_wl)%2!=0 :
                raise ValueError("There must be an even number of systematics maps for weak lensing")
            nmaps=len(syst_nc)/2
            syst_wl=syst_nc.reshape([nmaps,2,pix_schm.npix])
        else :
            syst_wl=None

        #Generate iterators for shear and tomography catalogs
        cols_shear=['ra','dec','mcal_g','mcal_flags','mcal_mag',
                    'mcal_s2n_r','mcal_T','psfrec_T']
        shear_iterator=(data for start,end,data in self.iterate_fits('shear_catalog',
                                                                     1,
                                                                     cols_shear,
                                                                     config['chunk_rows']))
        bin_iterator=(data for start,end,data in self.iterate_hdf('tomography_catalog',
                                                                  'tomography',
                                                                  ['bin'],
                                                                  config['chunk_rows']))
        ms_iterator=(data for start,end,data in self.iterate_hdf('tomography_catalog',
                                                                 'tomography',
                                                                 ['R_gamma'],
                                                                 config['chunk_rows']))

        #Generate maps
        m_count,m_mean=source_mapper(shear_iterator,bin_iterator,ms_iterator,pix_schm,bins,self.comm)

        #Generate namaster fields
        f_d,f_w=maps2fields(m_count,m_mean,mask,syst_nc,syst_wl,field_class)

        #Binning scheme
        #TODO: set ell binning from config
        if pixtype==0 :
            nlb=min(1,int(1./np.mean(mask)))
            ell_bins=nmt.NmtBin(pix_schm.nside,nlb=nlb)
        elif pixtype==1 :
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


        #Compute mode-coupling matrix
        #TODO: mode-coupling could be pre-computed and provided in config.
        if len(f_d)>0 :
            w00=nmt.NmtWorkspace(); w00.compute_coupling_matrix(f_d[0],f_d[0],ell_bins);
            if len(f_w)>0 :
                w02=nmt.NmtWorkspace(); w02.compute_coupling_matrix(f_d[0],f_w[0],ell_bins);
        if len(f_w)>0 :
            w22=nmt.NmtWorkspace(); w22.compute_coupling_matrix(f_w[0],f_w[0],b);

        #Compute power spectra
        #TODO: now all possible auto- and cross-correlation are computed.
        #      This should be tunable.
        c_dd=[]; c_dl=[]; c_ll=[]; t_bins=[];
        for ib1,b1 in enumerate(bins) :
            for ib2,b2 in enumerate(bins) :
                if b2>=b1 : 
                    #This implies we don't care about lenses behind sources
                    #We also only care about the lower-triangular part of the matrix for the
                    #galaxy-galaxy and shear-shear components.
                    continue
                t_bins.append((b1,b2))
                if len(f_d)>0 :
                    c00=w00.decouple_cell(nmt.compute_coupled_cell(f_d[ib1],f_d[ib2]))
                    c_dd.append(c00[0])
                    if len(f_w)>0 :
                        c02=w02.decouple_cell(nmt.compute_coupled_cell(f_w[ib1],f_d[ib2]))
                        c_dl.append([c02[0],c02[1]])
                if len(f_w)>0 :
                    c22=w22.decouple_cell(nmt.compute_coupled_cell(f_w[ib1],f_w[ib2]))
                    c_ll.append([c22[0],c22[3]]) #We neglect the EB and BE cross-correlations

        #Save power spectra
        #TODO: SACC interface here

if __name__ == '__main__':
    PipelineStage.main()
