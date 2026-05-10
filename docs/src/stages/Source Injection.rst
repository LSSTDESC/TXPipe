Source Injection
================

These stages ingest and use synthetic source injection information

* :py:class:`~txpipe.auxiliary_maps.TXAuxiliarySSIMaps` - Generate auxiliary maps from SSI catalogs

* :py:class:`~txpipe.map_plots.TXMapPlotsSSI` - Make plots of all the available maps that use SSI inputs

* :py:class:`~txpipe.ingest.ssi.TXIngestSSI` - Base-Class for ingesting SSI catalogs

* :py:class:`~txpipe.ingest.ssi.TXIngestSSIGCR` - Ingest SSI catalogs using GCR

* :py:class:`~txpipe.ingest.ssi.TXMatchSSI` - Match an SSI injection catalog and a photometry catalog

* :py:class:`~txpipe.ingest.ssi.TXIngestSSIDESBalrog` - Base-stage for ingesting a DES SSI catalog AKA "Balrog"

* :py:class:`~txpipe.ingest.ssi.TXIngestSSIMatchedDESBalrog` - Ingest a matched "SSI" catalog from DES (AKA Balrog)

* :py:class:`~txpipe.ingest.ssi.TXIngestSSIDetectionDESBalrog` - Ingest an "SSI" "detection" catalog from DES (AKA Balrog)

* :py:class:`~txpipe.magnification.TXSSIMagnification` - Compute the magnification coefficients using SSI outputs



.. autotxclass:: txpipe.auxiliary_maps.TXAuxiliarySSIMaps
    :members:
    :exclude-members: run

    Inputs: 

    - injection_catalog: HDFFile
    - matched_ssi_photometry_catalog: HDFFile

    Outputs: 

    - aux_ssi_maps: MapsFile
    
    Parallel: Yes - Dask


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>block_size</strong>: (int) Default=0. </LI>
            <LI><strong>depth_band</strong>: (str) Default=i. </LI>
            <LI><strong>snr_threshold</strong>: (float) Default=10.0. </LI>
            <LI><strong>snr_delta</strong>: (float) Default=1.0. </LI>
            </UL>



.. autotxclass:: txpipe.map_plots.TXMapPlotsSSI
    :members:
    :exclude-members: run

    Inputs: 

    - aux_ssi_maps: MapsFile

    Outputs: 

    - depth_ssi_meas_map: PNGFile
    - depth_ssi_true_map: PNGFile
    
    Parallel: No - Serial


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>projection</strong>: (str) Default=cart. </LI>
            <LI><strong>rot180</strong>: (bool) Default=False. </LI>
            <LI><strong>debug</strong>: (bool) Default=False. </LI>
            </UL>



.. autotxclass:: txpipe.ingest.ssi.TXIngestSSI
    :members:
    :exclude-members: run

    Inputs: None

    Outputs: None
    
    Parallel: Yes - MPI


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            </UL>



.. autotxclass:: txpipe.ingest.ssi.TXIngestSSIGCR
    :members:
    :exclude-members: run

    Inputs: None

    Outputs: 

    - injection_catalog: HDFFile
    - ssi_photometry_catalog: HDFFile
    - ssi_uninjected_photometry_catalog: HDFFile
    
    Parallel: No - Serial


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>injection_catalog_name</strong>: (str) Default=. </LI>
            <LI><strong>ssi_photometry_catalog_name</strong>: (str) Default=. </LI>
            <LI><strong>ssi_uninjected_photometry_catalog_name</strong>: (str) Default=. </LI>
            <LI><strong>GCRcatalog_path</strong>: (str) Default=. </LI>
            <LI><strong>flux_name</strong>: (str) Default=gaap3p0Flux. </LI>
            </UL>



.. autotxclass:: txpipe.ingest.ssi.TXMatchSSI
    :members:
    :exclude-members: run

    Inputs: 

    - injection_catalog: HDFFile
    - ssi_photometry_catalog: HDFFile

    Outputs: 

    - matched_ssi_photometry_catalog: HDFFile
    
    Parallel: Yes - MPI


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>chunk_rows</strong>: (int) Default=100000. </LI>
            <LI><strong>match_radius</strong>: (float) Default=0.5. </LI>
            <LI><strong>magnification</strong>: (int) Default=0. </LI>
            </UL>



.. autotxclass:: txpipe.ingest.ssi.TXIngestSSIDESBalrog
    :members:
    :exclude-members: run

    Inputs: None

    Outputs: None
    
    Parallel: Yes - MPI


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            </UL>



.. autotxclass:: txpipe.ingest.ssi.TXIngestSSIMatchedDESBalrog
    :members:
    :exclude-members: run

    Inputs: 

    - balrog_matched_catalog: FitsFile

    Outputs: 

    - matched_ssi_photometry_catalog: HDFFile
    
    Parallel: Yes - MPI


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            </UL>



.. autotxclass:: txpipe.ingest.ssi.TXIngestSSIDetectionDESBalrog
    :members:
    :exclude-members: run

    Inputs: 

    - balrog_detection_catalog: FitsFile

    Outputs: 

    - injection_catalog: HDFFile
    - ssi_detection_catalog: HDFFile
    
    Parallel: Yes - MPI


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            </UL>



.. autotxclass:: txpipe.magnification.TXSSIMagnification
    :members:
    :exclude-members: run

    Inputs: 

    - binned_lens_catalog_nomag: HDFFile
    - binned_lens_catalog_mag: HDFFile

    Outputs: 

    - magnification_coefficients: HDFFile
    - magnification_plot: PNGFile
    
    Parallel: No - Serial


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>chunk_rows</strong>: (int) Default=10000. </LI>
            <LI><strong>applied_magnification</strong>: (float) Default=1.02. </LI>
            <LI><strong>n_patches</strong>: (int) Default=20. </LI>
            <LI><strong>bootstrap_error</strong>: (bool) Default=True. </LI>
            </UL>


