Rubin DP2 Ingestion
===================

These stages ingest Rubin Data Preview 2 data, and the DESC-processed
versions of it, into TXPipe formats.

* :py:class:`~txpipe.ingest.metaDetect.TXDP2Ingestion` - Base class for things that ingest DP2 using a butler. Do not use directly.

* :py:class:`~txpipe.ingest.metaDetect.TXIngestRubinMetaDetect` - Initial ingestion of the Rubin MetaDetect catalog

* :py:class:`~txpipe.ingest.metaDetect.TXGenerateTractList` - Generate a list of Butler tracts based on a mask.

* :py:class:`~txpipe.ingest.metaDetect.TXIngestHealsparseMask` - Convert a healsparse mask file into a TXPipe maps file.

* :py:class:`~txpipe.ingest.metaDetect.TXIngestDP2Photometry` - Ingest DP2 photometry for the tracts in a tract list.

* :py:class:`~txpipe.ingest.metaDetect.TXIngestDESCMetaDetectV1_1` - Initial ingestion of the Rubin MetaDetect catalog



.. autotxclass:: txpipe.ingest.metaDetect.TXDP2Ingestion
    :members:
    :exclude-members: run

    Inputs: 

    - tract_list: TextFile

    Outputs: None
    
    Parallel: Yes - MPI


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>butler_repository_index</strong>: (str) Default=/global/cfs/cdirs/lsst/production/gen3/shared/data-repos.yaml. Path to repository index</LI>
            <LI><strong>butler_config_file</strong>: (str) Default=dp2. Path to the LSST butler config file, or code name for it in the repo index</LI>
            <LI><strong>collections</strong>: (str) Default=dp2. Butler collections to use.</LI>
            </UL>



.. autotxclass:: txpipe.ingest.metaDetect.TXIngestRubinMetaDetect
    :members:
    :exclude-members: run

    Inputs: 

    - tract_list: TextFile

    Outputs: 

    - shear_catalog: ShearCatalog
    
    Parallel: Yes - MPI


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>butler_repository_index</strong>: (str) Default=/global/cfs/cdirs/lsst/production/gen3/shared/data-repos.yaml. Path to repository index</LI>
            <LI><strong>butler_config_file</strong>: (str) Default=dp2. Path to the LSST butler config file, or code name for it in the repo index</LI>
            <LI><strong>collections</strong>: (str) Default=dp2. Butler collections to use.</LI>
            <LI><strong>exclusion_flag</strong>: (bool) Default=False. Decide if flags are used for exclusion or just flagged.</LI>
            <LI><strong>flag_list</strong>: (list) Default=[&#x27;is_primary&#x27;]. list of flags to use for combined.</LI>
            <LI><strong>all_columns</strong>: (bool) Default=False. do we want to save all columns or just the ones TXPipe needs.</LI>
            <LI><strong>pre_response_shape_noise</strong>: (float) Default=0.22. Estimate of shape noise before response for constructing weight.</LI>
            </UL>



.. autotxclass:: txpipe.ingest.metaDetect.TXGenerateTractList
    :members:
    :exclude-members: run

    Inputs: 

    - shear_mask: DataFile

    Outputs: 

    - tract_list: TextFile
    - tract_list_plot: PNGFile
    
    Parallel: Yes - MPI


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>butler_repository_index</strong>: (str) Default=/global/cfs/cdirs/lsst/production/gen3/shared/data-repos.yaml. Path to repository index</LI>
            <LI><strong>butler_config_file</strong>: (str) Default=dp2. Path to the LSST butler config file, or code name for it in the repo index</LI>
            <LI><strong>collections</strong>: (str) Default=dp2. Butler collections to use.</LI>
            <LI><strong>nside_low</strong>: (int) Default=512. The nside resolution for finding tracts from </LI>
            <LI><strong>dec_min</strong>: (float) Default=-40.0. Minimum declination to keep. Designed to cut out a little island from a deep field that snuck through</LI>
            </UL>



.. autotxclass:: txpipe.ingest.metaDetect.TXIngestHealsparseMask
    :members:
    :exclude-members: run

    Inputs: 

    - shear_mask: DataFile

    Outputs: 

    - mask: MapsFile
    
    Parallel: Yes - MPI


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            </UL>



.. autotxclass:: txpipe.ingest.metaDetect.TXIngestDP2Photometry
    :members:
    :exclude-members: run

    Inputs: 

    - tract_list: TextFile

    Outputs: 

    - photometry_catalog: PhotometryCatalog
    
    Parallel: Yes - MPI


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>butler_repository_index</strong>: (str) Default=/global/cfs/cdirs/lsst/production/gen3/shared/data-repos.yaml. Path to repository index</LI>
            <LI><strong>butler_config_file</strong>: (str) Default=dp2. Path to the LSST butler config file, or code name for it in the repo index</LI>
            <LI><strong>collections</strong>: (str) Default=dp2. Butler collections to use.</LI>
            </UL>



.. autotxclass:: txpipe.ingest.metaDetect.TXIngestDESCMetaDetectV1_1
    :members:
    :exclude-members: run

    Inputs: None

    Outputs: 

    - shear_catalog: ShearCatalog
    
    Parallel: Yes - MPI


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>base_dir</strong>: (str) Default=/pscratch/sd/e/esheldon/lsst-mdet-runs/run-dp2-v01.1. Top directory for catalog</LI>
            <LI><strong>exclusion_flag</strong>: (bool) Default=False. Decide if flags are used for exclusion or just flagged.</LI>
            <LI><strong>all_columns</strong>: (bool) Default=False. do we want to save all columns or just the ones TXPipe needs.</LI>
            <LI><strong>pre_response_shape_noise</strong>: (float) Default=0.22. Estimate of shape noise before response for constructing weight.</LI>
            </UL>


