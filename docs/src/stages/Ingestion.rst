Ingestion
=========

These stages import data into TXPipe input formats from general-purpose
or survey-independent sources.

* :py:class:`~txpipe.exposure_info.TXExposureInfo` - Ingest exposure information from an OpSim database

* :py:class:`~txpipe.ingest.dp02.TXIngestDataPreview02` - Ingest galaxy catalogs from DP0.2

* :py:class:`~txpipe.ingest.gcr.TXMetacalGCRInput` - Ingest metacal catalogs from GCRCatalogs

* :py:class:`~txpipe.ingest.gcr.TXIngestStars` - Ingest a star catalog from GCRCatalogs

* :py:class:`~txpipe.ingest.redmagic.TXIngestRedmagic` - Ingest a redmagic catalog

* :py:class:`~txpipe.ingest.desi.TXIngestDESIRandoms` - Ingest a DESI random catalog from FITS and write random_cats,

* :py:class:`~txpipe.ingest.desi.TXIngestDESI` - Ingest a DESI catalog (mock or real)

* :py:class:`~txpipe.ingest.base.TXIngestCatalogBase` - Base-Class for ingesting catalogs from external sources and saving to a format

* :py:class:`~txpipe.ingest.base.TXIngestCatalogFits` - Class for ingesting catalogs from FITS format and saving to a format

* :py:class:`~txpipe.ingest.base.TXIngestCatalogH5` - Class for ingesting catalogs from HDF5 files and saving to a format

* :py:class:`~txpipe.ingest.base.TXIngestMapsBase` - Base-Class for ingesting maps from external sources and saving to a format

* :py:class:`~txpipe.ingest.base.TXIngestMapsHsp` - Class for ingesting maps from external healsparse files and saving to a format

* :py:class:`~txpipe.ingest.dp1.TXIngestDataPreview1` - Ingest galaxy catalogs from DP1

* :py:class:`~txpipe.ingest.star_catalog.TXIngestStarCatalog` - Ingest the star catalog from the DP2 object table



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
            <LI><strong>propId</strong>: (int) Default=54. ID to filter visits.</LI>
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
            <LI><strong>length</strong>: (int) Default=0. Pre-known length, if the catalog has been checked at previously.</LI>
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
            <LI><strong>length</strong>: (int) Default=0. Pre-known length, if the catalog has been checked at previously.</LI>
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
            <LI><strong>lens_zbin_edges</strong>: (list) Default=[&lt;class &#x27;float&#x27;&gt;]. Edges of lens redshift bins.</LI>
            <LI><strong>chunk_rows</strong>: (int) Default=100000. Number of rows to process in each chunk.</LI>
            <LI><strong>zmin</strong>: (float) Default=0.0. Minimum redshift for binning.</LI>
            <LI><strong>zmax</strong>: (float) Default=3.0. Maximum redshift for binning.</LI>
            <LI><strong>dz</strong>: (float) Default=0.01. Redshift bin width.</LI>
            <LI><strong>bands</strong>: (list) Default=[&#x27;g&#x27;, &#x27;r&#x27;, &#x27;i&#x27;, &#x27;z&#x27;, &#x27;y&#x27;]. Bands to use for redmagic selection.</LI>
            </UL>



.. autotxclass:: txpipe.ingest.desi.TXIngestDESIRandoms
    :members:
    :exclude-members: run

    Inputs: 

    - desi_random_catalog: FitsFile
    - fiducial_cosmology: FiducialCosmology

    Outputs: 

    - random_cats: RandomsCatalog
    - binned_random_catalog: RandomsCatalog
    - binned_random_catalog_sub: RandomsCatalog
    
    Parallel: No - Serial


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>lens_zbin_edges</strong>: (list) Default=[&lt;class &#x27;float&#x27;&gt;]. Edges of lens redshift bins.</LI>
            <LI><strong>chunk_rows</strong>: (int) Default=100000. Number of rows to process in each chunk.</LI>
            <LI><strong>sample_rate</strong>: (float) Default=0.5. Fraction of randoms retained in the sub-catalog.</LI>
            <LI><strong>z_col</strong>: (str) Default=z. Redshift column name in the FITS random catalog.</LI>
            <LI><strong>ra_col</strong>: (str) Default=ra. Right ascension column name in the FITS random catalog.</LI>
            <LI><strong>dec_col</strong>: (str) Default=dec. Declination column name in the FITS random catalog.</LI>
            </UL>



.. autotxclass:: txpipe.ingest.desi.TXIngestDESI
    :members:
    :exclude-members: run

    Inputs: 

    - desi_catalog_selected: FitsFile

    Outputs: 

    - lens_catalog: HDFFile
    - binned_lens_catalog: BinnedCatalog
    - lens_tomography_catalog_unweighted: HDFFile
    - lens_tomography_catalog: HDFFile
    - lens_photoz_stack: QPNOfZFile
    
    Parallel: No - Serial


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>lens_zbin_edges</strong>: (list) Default=[&lt;class &#x27;float&#x27;&gt;]. Edges of lens redshift bins.</LI>
            <LI><strong>chunk_rows</strong>: (int) Default=100000. Number of rows to process in each chunk.</LI>
            <LI><strong>zmin</strong>: (float) Default=0.0. Minimum redshift for binning.</LI>
            <LI><strong>zmax</strong>: (float) Default=3.0. Maximum redshift for binning.</LI>
            <LI><strong>dz</strong>: (float) Default=0.01. Redshift bin width.</LI>
            <LI><strong>mock</strong>: (bool) Default=False. Whether to use a mock catalog.</LI>
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



.. autotxclass:: txpipe.ingest.star_catalog.TXIngestStarCatalog
    :members:
    :exclude-members: run

    Inputs: None

    Outputs: 

    - dp2_star_catalog: HDFFile
    
    Parallel: No - Serial


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>bands</strong>: (list) Default=[&#x27;r&#x27;, &#x27;i&#x27;, &#x27;z&#x27;]. Bands to get for catalog.</LI>
            <LI><strong>hats_path</strong>: (str) Required. </LI>
            <LI><strong>num_lsdb_daskwokers</strong>: (int) Required. </LI>
            </UL>


