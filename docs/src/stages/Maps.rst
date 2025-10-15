Maps
====

These stages deal with making different kinds of maps for analysis and
plotting.

* :py:class:`~txpipe.maps.TXBaseMaps` - A base class for mapping stages

* :py:class:`~txpipe.maps.TXSourceMaps` - Generate source maps directly from binned, calibrated shear catalogs.

* :py:class:`~txpipe.maps.TXLensMaps` - Make tomographic lens number count maps

* :py:class:`~txpipe.maps.TXExternalLensMaps` - Make tomographic lens number count maps from external data

* :py:class:`~txpipe.maps.TXDensityMaps` - Convert galaxy count maps to overdensity delta maps

* :py:class:`~txpipe.noise_maps.TXSourceNoiseMaps` - Generate realizations of shear noise maps with random rotations

* :py:class:`~txpipe.noise_maps.TXLensNoiseMaps` - Generate lens density noise realizations using random splits

* :py:class:`~txpipe.noise_maps.TXExternalLensNoiseMaps` - Generate lens density noise realizations using random splits of an external catalog

* :py:class:`~txpipe.noise_maps.TXNoiseMapsJax` - Generate noise realisations of lens and source maps using JAX

* :py:class:`~txpipe.auxiliary_maps.TXAuxiliarySourceMaps` - Stage TXAuxiliarySourceMaps

* :py:class:`~txpipe.auxiliary_maps.TXAuxiliaryLensMaps` - Generate auxiliary maps from the lens catalog

* :py:class:`~txpipe.auxiliary_maps.TXUniformDepthMap` - Generate a uniform depth map from the mask

* :py:class:`~txpipe.masks.TXBaseMask` - Base class for generating binary survey masks using auxiliary input maps.

* :py:class:`~txpipe.masks.TXSimpleMask` - Generate a simple binary mask using cuts on depth and bright object maps.

* :py:class:`~txpipe.masks.TXSimpleMaskSource` - Generate a binary mask for source galaxies using positive lensing weights

* :py:class:`~txpipe.masks.TXSimpleMaskFrac` - Make a simple mask using a depth cut and bright object cut

* :py:class:`~txpipe.masks.TXCustomMask` - Make a mask from a custom list of cuts to aux maps (e.g depth cut and bright object cuts)

* :py:class:`~txpipe.convergence.TXConvergenceMaps` - Make a convergence map from a source map using Kaiser-Squires

* :py:class:`~txpipe.map_correlations.TXMapCorrelations` - Plot shear, density, and convergence correlations with survey property maps



.. autotxclass:: txpipe.maps.TXBaseMaps
    :members:
    :exclude-members: run

    Inputs: None

    Outputs: None
    
    Parallel: Yes - MPI


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            </UL>



.. autotxclass:: txpipe.maps.TXSourceMaps
    :members:
    :exclude-members: run

    Inputs: 

    - binned_shear_catalog: HDFFile

    Outputs: 

    - source_maps: MapsFile
    
    Parallel: Yes - Dask


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>block_size</strong>: (int) Default=0. Block size for dask processing (0 means auto)</LI>
            <LI><strong>chunk_rows</strong>: (int) Default=100000. The number of rows to read in each chunk of data at a time</LI>
            <LI><strong>pixelization</strong>: (str) Default=healpix. The pixelization scheme to use, currently just healpix</LI>
            <LI><strong>nside</strong>: (int) Default=0. The Healpix resolution parameter for the generated maps. Only required if using healpix</LI>
            <LI><strong>sparse</strong>: (bool) Default=True. Whether to generate sparse maps - faster and less memory for small sky areas</LI>
            <LI><strong>ra_cent</strong>: (float) Default=nan. Central RA for gnomonic projection (only required if pixelization==tan)</LI>
            <LI><strong>dec_cent</strong>: (float) Default=nan. Central Dec for gnomonic projection (only required if pixelization==tan)</LI>
            <LI><strong>npix_x</strong>: (int) Default=-1. Number of pixels in x direction for gnomonic projection</LI>
            <LI><strong>npix_y</strong>: (int) Default=-1. Number of pixels in y direction for gnomonic projection</LI>
            <LI><strong>pixel_size</strong>: (float) Default=nan. Pixel size of pixelization scheme</LI>
            </UL>



.. autotxclass:: txpipe.maps.TXLensMaps
    :members:
    :exclude-members: run

    Inputs: 

    - photometry_catalog: HDFFile
    - lens_tomography_catalog: TomographyCatalog

    Outputs: 

    - lens_maps: MapsFile
    
    Parallel: Yes - Dask


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>block_size</strong>: (int) Default=0. Block size for dask processing (0 means auto)</LI>
            <LI><strong>chunk_rows</strong>: (int) Default=100000. The number of rows to read in each chunk of data at a time</LI>
            <LI><strong>pixelization</strong>: (str) Default=healpix. The pixelization scheme to use, currently just healpix</LI>
            <LI><strong>nside</strong>: (int) Default=0. The Healpix resolution parameter for the generated maps. Only required if using healpix</LI>
            <LI><strong>sparse</strong>: (bool) Default=True. Whether to generate sparse maps - faster and less memory for small sky areas</LI>
            <LI><strong>ra_cent</strong>: (float) Default=nan. Central RA for gnomonic projection (only required if pixelization==tan)</LI>
            <LI><strong>dec_cent</strong>: (float) Default=nan. Central Dec for gnomonic projection (only required if pixelization==tan)</LI>
            <LI><strong>npix_x</strong>: (int) Default=-1. Number of pixels in x direction for gnomonic projection</LI>
            <LI><strong>npix_y</strong>: (int) Default=-1. Number of pixels in y direction for gnomonic projection</LI>
            <LI><strong>pixel_size</strong>: (float) Default=nan. Pixel size of pixelization scheme</LI>
            </UL>



.. autotxclass:: txpipe.maps.TXExternalLensMaps
    :members:
    :exclude-members: run

    Inputs: 

    - lens_catalog: HDFFile
    - lens_tomography_catalog: TomographyCatalog

    Outputs: 

    - lens_maps: MapsFile
    
    Parallel: Yes - Dask


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>block_size</strong>: (int) Default=0. Block size for dask processing (0 means auto)</LI>
            <LI><strong>chunk_rows</strong>: (int) Default=100000. The number of rows to read in each chunk of data at a time</LI>
            <LI><strong>pixelization</strong>: (str) Default=healpix. The pixelization scheme to use, currently just healpix</LI>
            <LI><strong>nside</strong>: (int) Default=0. The Healpix resolution parameter for the generated maps. Only required if using healpix</LI>
            <LI><strong>sparse</strong>: (bool) Default=True. Whether to generate sparse maps - faster and less memory for small sky areas</LI>
            <LI><strong>ra_cent</strong>: (float) Default=nan. Central RA for gnomonic projection (only required if pixelization==tan)</LI>
            <LI><strong>dec_cent</strong>: (float) Default=nan. Central Dec for gnomonic projection (only required if pixelization==tan)</LI>
            <LI><strong>npix_x</strong>: (int) Default=-1. Number of pixels in x direction for gnomonic projection</LI>
            <LI><strong>npix_y</strong>: (int) Default=-1. Number of pixels in y direction for gnomonic projection</LI>
            <LI><strong>pixel_size</strong>: (float) Default=nan. Pixel size of pixelization scheme</LI>
            </UL>



.. autotxclass:: txpipe.maps.TXDensityMaps
    :members:
    :exclude-members: run

    Inputs: 

    - lens_maps: MapsFile
    - mask: MapsFile

    Outputs: 

    - density_maps: MapsFile
    
    Parallel: No - Serial


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>mask_threshold</strong>: (float) Default=0.0. Threshold for masking pixels</LI>
            </UL>



.. autotxclass:: txpipe.noise_maps.TXSourceNoiseMaps
    :members:
    :exclude-members: run

    Inputs: 

    - shear_catalog: ShearCatalog
    - shear_tomography_catalog: TomographyCatalog
    - mask: MapsFile

    Outputs: 

    - source_noise_maps: LensingNoiseMaps
    
    Parallel: Yes - MPI


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>chunk_rows</strong>: (int) Default=100000. Number of rows to process in each chunk.</LI>
            <LI><strong>lensing_realizations</strong>: (int) Default=30. Number of lensing noise realizations to generate.</LI>
            <LI><strong>true_shear</strong>: (bool) Default=False. Whether to use true shear values for noise maps.</LI>
            </UL>



.. autotxclass:: txpipe.noise_maps.TXLensNoiseMaps
    :members:
    :exclude-members: run

    Inputs: 

    - lens_tomography_catalog: TomographyCatalog
    - photometry_catalog: HDFFile
    - mask: MapsFile

    Outputs: 

    - lens_noise_maps: ClusteringNoiseMaps
    
    Parallel: Yes - MPI


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>chunk_rows</strong>: (int) Default=100000. Number of rows to process in each chunk.</LI>
            <LI><strong>clustering_realizations</strong>: (int) Default=1. Number of clustering noise realizations to generate.</LI>
            <LI><strong>mask_in_weights</strong>: (bool) Default=False. Whether to include mask in weight calculations.</LI>
            </UL>



.. autotxclass:: txpipe.noise_maps.TXExternalLensNoiseMaps
    :members:
    :exclude-members: run

    Inputs: 

    - lens_tomography_catalog: TomographyCatalog
    - lens_catalog: HDFFile
    - mask: MapsFile

    Outputs: 

    - lens_noise_maps: ClusteringNoiseMaps
    
    Parallel: Yes - MPI


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>chunk_rows</strong>: (int) Default=100000. Number of rows to process in each chunk.</LI>
            <LI><strong>clustering_realizations</strong>: (int) Default=1. Number of clustering noise realizations to generate.</LI>
            <LI><strong>mask_in_weights</strong>: (bool) Default=False. Whether to include mask in weight calculations.</LI>
            </UL>



.. autotxclass:: txpipe.noise_maps.TXNoiseMapsJax
    :members:
    :exclude-members: run

    Inputs: 

    - shear_catalog: ShearCatalog
    - lens_tomography_catalog: TomographyCatalog
    - shear_tomography_catalog: TomographyCatalog
    - mask: MapsFile
    - lens_maps: MapsFile

    Outputs: 

    - source_noise_maps: LensingNoiseMaps
    - lens_noise_maps: ClusteringNoiseMaps
    
    Parallel: Yes - MPI


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>chunk_rows</strong>: (int) Default=4000000. </LI>
            <LI><strong>lensing_realizations</strong>: (int) Default=30. </LI>
            <LI><strong>clustering_realizations</strong>: (int) Default=1. </LI>
            <LI><strong>seed</strong>: (int) Default=0. </LI>
            </UL>



.. autotxclass:: txpipe.auxiliary_maps.TXAuxiliarySourceMaps
    :members:
    :exclude-members: run

    Inputs: 

    - shear_catalog: ShearCatalog
    - shear_tomography_catalog: HDFFile
    - source_maps: MapsFile

    Outputs: 

    - aux_source_maps: MapsFile
    
    Parallel: Yes - Dask


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>block_size</strong>: (int) Default=0. Block size for dask processing (0 means auto).</LI>
            <LI><strong>flag_exponent_max</strong>: (int) Default=8. Maximum exponent for flag bits (default 8).</LI>
            <LI><strong>psf_prefix</strong>: (str) Default=psf_. Prefix for PSF column names.</LI>
            <LI><strong>chunk_rows</strong>: (int) Default=100000. The number of rows to read in each chunk of data at a time</LI>
            <LI><strong>pixelization</strong>: (str) Default=healpix. The pixelization scheme to use, currently just healpix</LI>
            <LI><strong>nside</strong>: (int) Default=0. The Healpix resolution parameter for the generated maps. Only required if using healpix</LI>
            <LI><strong>sparse</strong>: (bool) Default=True. Whether to generate sparse maps - faster and less memory for small sky areas</LI>
            <LI><strong>ra_cent</strong>: (float) Default=nan. Central RA for gnomonic projection (only required if pixelization==tan)</LI>
            <LI><strong>dec_cent</strong>: (float) Default=nan. Central Dec for gnomonic projection (only required if pixelization==tan)</LI>
            <LI><strong>npix_x</strong>: (int) Default=-1. Number of pixels in x direction for gnomonic projection</LI>
            <LI><strong>npix_y</strong>: (int) Default=-1. Number of pixels in y direction for gnomonic projection</LI>
            <LI><strong>pixel_size</strong>: (float) Default=nan. Pixel size of pixelization scheme</LI>
            </UL>



.. autotxclass:: txpipe.auxiliary_maps.TXAuxiliaryLensMaps
    :members:
    :exclude-members: run

    Inputs: 

    - photometry_catalog: HDFFile

    Outputs: 

    - aux_lens_maps: MapsFile
    
    Parallel: Yes - Dask


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>block_size</strong>: (int) Default=0. Block size for dask processing (0 means auto).</LI>
            <LI><strong>bright_obj_threshold</strong>: (float) Default=22.0. Magnitude threshold for bright objects.</LI>
            <LI><strong>depth_band</strong>: (str) Default=i. Band for depth maps.</LI>
            <LI><strong>snr_threshold</strong>: (float) Default=10.0. S/N value for depth maps.</LI>
            <LI><strong>snr_delta</strong>: (float) Default=1.0. Delta for S/N thresholding.</LI>
            </UL>



.. autotxclass:: txpipe.auxiliary_maps.TXUniformDepthMap
    :members:
    :exclude-members: run

    Inputs: 

    - mask: MapsFile

    Outputs: 

    - aux_lens_maps: MapsFile
    
    Parallel: No - Serial


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>depth</strong>: (float) Default=25.0. Uniform depth value to assign everywhere.</LI>
            </UL>



.. autotxclass:: txpipe.masks.TXBaseMask
    :members:
    :exclude-members: run

    Inputs: None

    Outputs: 

    - mask: MapsFile
    
    Parallel: No - Serial


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            </UL>



.. autotxclass:: txpipe.masks.TXSimpleMask
    :members:
    :exclude-members: run

    Inputs: 

    - aux_lens_maps: MapsFile

    Outputs: 

    - mask: MapsFile
    
    Parallel: No - Serial


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>depth_cut</strong>: (float) Default=23.5. Depth cut for mask creation.</LI>
            <LI><strong>bright_object_max</strong>: (float) Default=10.0. Maximum allowed bright object count.</LI>
            </UL>



.. autotxclass:: txpipe.masks.TXSimpleMaskSource
    :members:
    :exclude-members: run

    Inputs: 

    - source_maps: MapsFile

    Outputs: 

    - mask: MapsFile
    
    Parallel: No - Serial


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            </UL>



.. autotxclass:: txpipe.masks.TXSimpleMaskFrac
    :members:
    :exclude-members: run

    Inputs: 

    - aux_lens_maps: MapsFile

    Outputs: 

    - mask: MapsFile
    
    Parallel: No - Serial


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>depth_cut</strong>: (float) Default=23.5. Depth cut for mask creation.</LI>
            <LI><strong>bright_object_max</strong>: (float) Default=10.0. Maximum allowed bright object count.</LI>
            <LI><strong>supreme_map_file</strong>: (str) Default=. Path to supreme map file for fracdet computation.</LI>
            </UL>



.. autotxclass:: txpipe.masks.TXCustomMask
    :members:
    :exclude-members: run

    Inputs: 

    - aux_lens_maps: MapsFile

    Outputs: 

    - mask: MapsFile
    
    Parallel: No - Serial


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>fracdet_name</strong>: (str) Default=footprint/fracdet_griz. Fracdet map name.</LI>
            <LI><strong>cuts</strong>: (list) Default=['footprint/fracdet_griz > 0']. List of mask cuts to apply.</LI>
            <LI><strong>degrade</strong>: (bool) Default=False. Degrade resolution if input map Nside differs from config nside.</LI>
            </UL>



.. autotxclass:: txpipe.convergence.TXConvergenceMaps
    :members:
    :exclude-members: run

    Inputs: 

    - source_maps: MapsFile

    Outputs: 

    - convergence_maps: MapsFile
    
    Parallel: No - Serial


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>lmax</strong>: (int) Default=0. Maximum multipole for convergence map (0 means 2*nside).</LI>
            <LI><strong>smoothing_sigma</strong>: (float) Default=10.0. Smoothing scale in arcmin.</LI>
            </UL>



.. autotxclass:: txpipe.map_correlations.TXMapCorrelations
    :members:
    :exclude-members: run

    Inputs: 

    - lens_maps: MapsFile
    - convergence_maps: MapsFile
    - source_maps: MapsFile
    - mask: MapsFile

    Outputs: 

    - map_systematic_correlations: FileCollection
    
    Parallel: No - Serial


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>supreme_path_root</strong>: (str) Default=/global/cscratch1/sd/erykoff/dc2_dr6/supreme/supreme_dc2_dr6d_v2. Root path for supreme files.</LI>
            <LI><strong>nbin</strong>: (int) Default=20. Number of tomographic bins.</LI>
            <LI><strong>outlier_fraction</strong>: (float) Default=0.05. Fraction of outliers to exclude.</LI>
            </UL>


