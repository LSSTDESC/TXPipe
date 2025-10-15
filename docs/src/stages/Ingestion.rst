Ingestion
=========

These stages import data into TXPipe input formats, or generate mock data from
simpler input catalogs.

* :py:class:`~txpipe.exposure_info.TXExposureInfo` - Ingest exposure information from an OpSim database

* :py:class:`~txpipe.ingest.dp02.TXIngestDataPreview02` - Ingest galaxy catalogs from DP0.2

* :py:class:`~txpipe.ingest.mocks.TXCosmoDC2Mock` - Simulate mock shear and photometry measurements from CosmoDC2 (or similar)

* :py:class:`~txpipe.ingest.mocks.TXBuzzardMock` - Simulate mock photometry from Buzzard.

* :py:class:`~txpipe.ingest.mocks.TXGaussianSimsMock` - Simulate mock photometry from gaussian simulations

* :py:class:`~txpipe.ingest.mocks.TXSimpleMock` - Load an ascii astropy table and put it in shear catalog format.

* :py:class:`~txpipe.ingest.mocks.TXMockTruthPZ` - Stage TXMockTruthPZ

* :py:class:`~txpipe.ingest.gcr.TXMetacalGCRInput` - Ingest metacal catalogs from GCRCatalogs

* :py:class:`~txpipe.ingest.gcr.TXIngestStars` - Ingest a star catalog from GCRCatalogs

* :py:class:`~txpipe.ingest.redmagic.TXIngestRedmagic` - Ingest a redmagic catalog

* :py:class:`~txpipe.ingest.base.TXIngestCatalogBase` - Base-Class for ingesting catalogs from external sources and saving to a format

* :py:class:`~txpipe.ingest.base.TXIngestCatalogFits` - Class for ingesting catalogs from FITS format and saving to a format

* :py:class:`~txpipe.ingest.base.TXIngestCatalogH5` - Class for ingesting catalogs from HDF5 files and saving to a format

* :py:class:`~txpipe.ingest.base.TXIngestMapsBase` - Base-Class for ingesting maps from external sources and saving to a format

* :py:class:`~txpipe.ingest.base.TXIngestMapsHsp` - Class for ingesting maps from external healsparse files and saving to a format

* :py:class:`~txpipe.ingest.dp1.TXIngestDataPreview1` - Ingest galaxy catalogs from DP1

* :py:class:`~txpipe.ingest.legacy.TXIngestDESY3Gold` - Ingest the DES Y3 Gold from hdf5 format

* :py:class:`~txpipe.ingest.legacy.TXIngestDESY3Footprint` - Ingest the DES Y3 Footprint maps (incl. badregions, foregrounds etc) from healsparse format

* :py:class:`~txpipe.ingest.legacy.TXIngestDESY3SpeczCat` - Ingest the spectroscopic catalog used for DES Y3 training of DNF

* :py:class:`~txpipe.simulation.TXLogNormalGlass` - Uses GLASS to generate a simulated catalog from lognormal fields



.. autotxclass:: txpipe.exposure_info.TXExposureInfo
    :members:
    :exclude-members: run

    Inputs: None

    Outputs: 

    - exposures: HDFFile
    
    Parallel: No - Serial


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>dc2_name</strong>: (str) Default=1.2p. Name of the DC2 run to use.</LI>
            <LI><strong>opsim_db</strong>: (str) Default=/global/projecta/projectdirs/lsst/groups/SSim/DC2/minion_1016_desc_dithered_v4.db. Path to the opsim database file.</LI>
            <LI><strong>propId</strong>: (int) Default=54. Proposal ID to filter visits.</LI>
            </UL>



.. autotxclass:: txpipe.ingest.dp02.TXIngestDataPreview02
    :members:
    :exclude-members: run

    Inputs: None

    Outputs: 

    - photometry_catalog: HDFFile
    - shear_catalog: ShearCatalog
    
    Parallel: No - Serial


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>pq_path</strong>: (str) Default=/global/cfs/cdirs/lsst/shared/rubin/DP0.2/objectTable/. Path to Parquet objectTable files.</LI>
            <LI><strong>tracts</strong>: (str) Default=. Comma-separated list of tracts to use (empty for all).</LI>
            </UL>



.. autotxclass:: txpipe.ingest.mocks.TXCosmoDC2Mock
    :members:
    :exclude-members: run

    Inputs: 

    - response_model: HDFFile

    Outputs: 

    - shear_catalog: ShearCatalog
    - photometry_catalog: HDFFile
    
    Parallel: No - Serial


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>cat_name</strong>: (str) Default=cosmoDC2. Name of the mock catalog to use.</LI>
            <LI><strong>visits_per_band</strong>: (int) Default=165. Number of visits per band for noise simulation.</LI>
            <LI><strong>snr_limit</strong>: (float) Default=4.0. S/N limit for object selection.</LI>
            <LI><strong>max_size</strong>: (int) Default=99999999999999. Maximum catalog size for testing.</LI>
            <LI><strong>extra_cols</strong>: (str) Default=. Extra columns to include (comma-separated).</LI>
            <LI><strong>max_npix</strong>: (int) Default=99999999999999. Maximum number of pixels.</LI>
            <LI><strong>unit_response</strong>: (bool) Default=False. Whether to use unit response in simulation.</LI>
            <LI><strong>cat_size</strong>: (int) Default=0. Catalog size (0 for all).</LI>
            <LI><strong>flip_g2</strong>: (bool) Default=True. Whether to flip g2 sign to match conventions.</LI>
            <LI><strong>apply_mag_cut</strong>: (bool) Default=False. Apply magnitude cut for descqa comparison.</LI>
            <LI><strong>Mag_r_limit</strong>: (float) Default=-19. Magnitude r limit for object selection.</LI>
            <LI><strong>metadetect</strong>: (bool) Default=True. Whether to mock a metacal catalog.</LI>
            <LI><strong>add_shape_noise</strong>: (bool) Default=True. Whether to add shape noise to simulation.</LI>
            <LI><strong>healpixels</strong>: (list) Default=[-1]. List of HEALPix pixels to use.</LI>
            </UL>



.. autotxclass:: txpipe.ingest.mocks.TXBuzzardMock
    :members:
    :exclude-members: run

    Inputs: 

    - response_model: HDFFile

    Outputs: 

    - shear_catalog: ShearCatalog
    - photometry_catalog: HDFFile
    
    Parallel: No - Serial


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>cat_name</strong>: (str) Default=buzzard. Name of the mock catalog to use.</LI>
            <LI><strong>visits_per_band</strong>: (int) Default=165. Number of visits per band for noise simulation.</LI>
            <LI><strong>snr_limit</strong>: (float) Default=4.0. S/N limit for object selection.</LI>
            <LI><strong>max_size</strong>: (int) Default=99999999999999. Maximum catalog size for testing.</LI>
            <LI><strong>extra_cols</strong>: (str) Default=. Extra columns to include (comma-separated).</LI>
            <LI><strong>max_npix</strong>: (int) Default=99999999999999. Maximum number of pixels.</LI>
            <LI><strong>unit_response</strong>: (bool) Default=False. Whether to use unit response in simulation.</LI>
            <LI><strong>flip_g2</strong>: (bool) Default=True. Whether to flip g2 sign to match conventions.</LI>
            </UL>



.. autotxclass:: txpipe.ingest.mocks.TXGaussianSimsMock
    :members:
    :exclude-members: run

    Inputs: 

    - response_model: HDFFile

    Outputs: 

    - shear_catalog: ShearCatalog
    - photometry_catalog: HDFFile
    
    Parallel: No - Serial


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>cat_name</strong>: (str) Default=GaussianSims. Name of the Gaussian simulation catalog.</LI>
            <LI><strong>visits_per_band</strong>: (int) Default=165. Number of visits per band for noise simulation.</LI>
            <LI><strong>snr_limit</strong>: (float) Default=0.0. S/N limit for object selection (0 for all).</LI>
            <LI><strong>max_size</strong>: (int) Default=99999999999999. Maximum catalog size for testing.</LI>
            <LI><strong>extra_cols</strong>: (str) Default=. Extra columns to include (comma-separated).</LI>
            <LI><strong>max_npix</strong>: (int) Default=99999999999999. Maximum number of pixels.</LI>
            <LI><strong>unit_response</strong>: (bool) Default=True. Whether to use unit response in simulation.</LI>
            <LI><strong>cat_size</strong>: (int) Default=0. Catalog size (0 for all).</LI>
            <LI><strong>flip_g2</strong>: (bool) Default=False. Whether to flip g2 sign to match conventions.</LI>
            <LI><strong>apply_mag_cut</strong>: (bool) Default=False. Apply magnitude cut for descqa comparison.</LI>
            <LI><strong>metadetect</strong>: (bool) Default=True. Whether to mock a metacal catalog.</LI>
            <LI><strong>add_shape_noise</strong>: (bool) Default=False. Whether to add shape noise to simulation.</LI>
            </UL>



.. autotxclass:: txpipe.ingest.mocks.TXSimpleMock
    :members:
    :exclude-members: run

    Inputs: 

    - mock_shear_catalog: TextFile

    Outputs: 

    - shear_catalog: ShearCatalog
    
    Parallel: No - Serial


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>mock_size_snr</strong>: (bool) Default=False. Whether to mock size S/N for simulation.</LI>
            </UL>



.. autotxclass:: txpipe.ingest.mocks.TXMockTruthPZ
    :members:
    :exclude-members: run

    Inputs: 

    - shear_catalog: ShearCatalog

    Outputs: 

    - photoz_pdfs: QPPDFFile
    
    Parallel: No - Serial


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>mock_sigma_z</strong>: (float) Default=0.001. Sigma_z for mock photo-z PDF generation.</LI>
            </UL>



.. autotxclass:: txpipe.ingest.gcr.TXMetacalGCRInput
    :members:
    :exclude-members: run

    Inputs: None

    Outputs: 

    - shear_catalog: ShearCatalog
    - photometry_catalog: HDFFile
    
    Parallel: No - Serial


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>cat_name</strong>: (str) Default=. Name of the GCR catalog to load.</LI>
            <LI><strong>single_tract</strong>: (str) Default=. Single tract to use (optional).</LI>
            <LI><strong>length</strong>: (int) Default=0. Number of rows to use (0 for all).</LI>
            <LI><strong>table_dir</strong>: (str) Default=. Directory for table files (optional).</LI>
            <LI><strong>data_release</strong>: (str) Default=. Data release identifier (optional).</LI>
            </UL>



.. autotxclass:: txpipe.ingest.gcr.TXIngestStars
    :members:
    :exclude-members: run

    Inputs: None

    Outputs: 

    - star_catalog: HDFFile
    
    Parallel: No - Serial


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>single_tract</strong>: (str) Default=. Single tract to use (optional).</LI>
            <LI><strong>cat_name</strong>: (str) Default=. Name of the GCR catalog to load.</LI>
            <LI><strong>length</strong>: (int) Default=0. Number of rows to use (0 for all).</LI>
            </UL>



.. autotxclass:: txpipe.ingest.redmagic.TXIngestRedmagic
    :members:
    :exclude-members: run

    Inputs: 

    - redmagic_catalog: FitsFile

    Outputs: 

    - lens_catalog: HDFFile
    - lens_tomography_catalog_unweighted: HDFFile
    - lens_photoz_stack: QPNOfZFile
    
    Parallel: No - Serial


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>lens_zbin_edges</strong>: (list) Default=[<class 'float'>]. Edges of lens redshift bins.</LI>
            <LI><strong>chunk_rows</strong>: (int) Default=100000. Number of rows to process in each chunk.</LI>
            <LI><strong>zmin</strong>: (float) Default=0.0. Minimum redshift for binning.</LI>
            <LI><strong>zmax</strong>: (float) Default=3.0. Maximum redshift for binning.</LI>
            <LI><strong>dz</strong>: (float) Default=0.01. Redshift bin width.</LI>
            <LI><strong>bands</strong>: (str) Default=grizy. Bands to use for redmagic selection.</LI>
            </UL>



.. autotxclass:: txpipe.ingest.base.TXIngestCatalogBase
    :members:
    :exclude-members: run

    Inputs: None

    Outputs: None
    
    Parallel: Yes - MPI


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            </UL>



.. autotxclass:: txpipe.ingest.base.TXIngestCatalogFits
    :members:
    :exclude-members: run

    Inputs: None

    Outputs: None
    
    Parallel: Yes - MPI


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            </UL>



.. autotxclass:: txpipe.ingest.base.TXIngestCatalogH5
    :members:
    :exclude-members: run

    Inputs: None

    Outputs: None
    
    Parallel: Yes - MPI


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            </UL>



.. autotxclass:: txpipe.ingest.base.TXIngestMapsBase
    :members:
    :exclude-members: run

    Inputs: None

    Outputs: None
    
    Parallel: Yes - MPI


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>input_nside</strong>: (int) Default=0. Input HEALPix nside value.</LI>
            <LI><strong>input_nest</strong>: (bool) Default=True. Whether input maps use NESTED ordering.</LI>
            </UL>



.. autotxclass:: txpipe.ingest.base.TXIngestMapsHsp
    :members:
    :exclude-members: run

    Inputs: None

    Outputs: None
    
    Parallel: Yes - MPI


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>input_nside</strong>: (int) Default=0. Input HEALPix nside value.</LI>
            <LI><strong>input_nest</strong>: (bool) Default=True. Whether input maps use NESTED ordering.</LI>
            </UL>



.. autotxclass:: txpipe.ingest.dp1.TXIngestDataPreview1
    :members:
    :exclude-members: run

    Inputs: None

    Outputs: 

    - photometry_catalog: PhotometryCatalog
    - shear_catalog: ShearCatalog
    - exposures: HDFFile
    - survey_property_maps: FileCollection
    
    Parallel: Yes - MPI


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>butler_config_file</strong>: (str) Default=/global/cfs/cdirs/lsst/production/gen3/rubin/DP1/repo/butler.yaml. Path to the LSST butler config file.</LI>
            <LI><strong>cosmology_tracts_only</strong>: (bool) Default=True. Use only cosmology tracts.</LI>
            <LI><strong>select_field</strong>: (str) Default=. Field to select (overrides cosmology_tracts_only).</LI>
            <LI><strong>collections</strong>: (str) Default=LSSTComCam/DP1. Butler collections to use.</LI>
            </UL>



.. autotxclass:: txpipe.ingest.legacy.TXIngestDESY3Gold
    :members:
    :exclude-members: run

    Inputs: 

    - des_photometry_catalog: HDFFile

    Outputs: 

    - photometry_catalog: HDFFile
    
    Parallel: No - Serial


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>input_group_name</strong>: (str) Default=catalog/gold. Input group name in the HDF5 file.</LI>
            <LI><strong>chunk_rows</strong>: (int) Default=100000. Number of rows to process in each chunk.</LI>
            </UL>



.. autotxclass:: txpipe.ingest.legacy.TXIngestDESY3Footprint
    :members:
    :exclude-members: run

    Inputs: None

    Outputs: 

    - aux_lens_maps: MapsFile
    
    Parallel: No - Serial


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>input_nside</strong>: (int) Default=0. Input HEALPix nside value.</LI>
            <LI><strong>input_nest</strong>: (bool) Default=True. Whether input maps use NESTED ordering.</LI>
            <LI><strong>input_filepaths</strong>: (list) Default=['']. List of input file paths.</LI>
            <LI><strong>input_labels</strong>: (list) Default=['']. List of input labels.</LI>
            <LI><strong>chunk_rows</strong>: (int) Default=100000. Number of rows to process in each chunk.</LI>
            </UL>



.. autotxclass:: txpipe.ingest.legacy.TXIngestDESY3SpeczCat
    :members:
    :exclude-members: run

    Inputs: 

    - des_specz_catalog: FitsFile

    Outputs: 

    - spectroscopic_catalog: HDFFile
    
    Parallel: No - Serial


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>chunk_rows</strong>: (int) Default=100000. Number of rows to process in each chunk.</LI>
            </UL>



.. autotxclass:: txpipe.simulation.TXLogNormalGlass
    :members:
    :exclude-members: run

    Inputs: 

    - mask: MapsFile
    - lens_photoz_stack: QPNOfZFile
    - fiducial_cosmology: FiducialCosmology
    - input_lss_weight_maps: MapsFile

    Outputs: 

    - photometry_catalog: HDFFile
    - lens_tomography_catalog_unweighted: TomographyCatalog
    - glass_cl_shells: HDFFile
    - glass_cl_binned: HDFFile
    - density_shells: HDFFile
    
    Parallel: No - Serial


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>num_dens</strong>: (float) Required. Number density of galaxies per square arcmin</LI>
            <LI><strong>zmin</strong>: (float) Default=0.0. Minimum redshift for the simulation</LI>
            <LI><strong>zmax</strong>: (float) Default=2.0. Maximum redshift for the simulation</LI>
            <LI><strong>dx</strong>: (int) Default=100. Comoving distance spacing for redshift shells in Mpc</LI>
            <LI><strong>bias0</strong>: (float) Default=2.0. Linear bias at zpivot</LI>
            <LI><strong>alpha_bz</strong>: (float) Default=0.0. Controls redshift evolution of bias</LI>
            <LI><strong>zpivot</strong>: (float) Default=0.6. Pivot redshift for bias evolution</LI>
            <LI><strong>shift</strong>: (float) Default=1.0. Lognormal shift parameter</LI>
            <LI><strong>contaminate</strong>: (bool) Default=False. Whether to apply contamination to the density field</LI>
            <LI><strong>random_seed</strong>: (int) Default=0. Random seed for reproducibility</LI>
            <LI><strong>cl_optional_file</strong>: (str) Default=none. Optional file for input C(l) values</LI>
            <LI><strong>ell_binned_min</strong>: (float) Default=0.1. Minimum ell for binned C(l) output</LI>
            <LI><strong>ell_binned_max</strong>: (float) Default=500000.0. Maximum ell for binned C(l) output</LI>
            <LI><strong>ell_binned_nbins</strong>: (int) Default=100. Number of ell bins for binned C(l) output</LI>
            <LI><strong>output_density_shell_maps</strong>: (bool) Default=False. Whether to output density maps for each shell</LI>
            </UL>


