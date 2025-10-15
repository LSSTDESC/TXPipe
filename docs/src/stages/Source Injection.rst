Source Injection
================

These stages ingest and use synthetic source injection information

* :py:class:`~txpipe.auxiliary_maps.TXAuxiliarySSIMaps` - Generate auxiliary maps from SSI catalogs

* :py:class:`~txpipe.map_plots.TXMapPlotsSSI` - Make plots of all the available maps that use SSI inputs

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

    - matched_ssi_photometry_catalog: HDFFile
    - injection_catalog: HDFFile
    - ssi_detection_catalog: HDFFile

    Outputs: 

    - aux_ssi_maps: MapsFile
    
    Parallel: Yes - Dask


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>block_size</strong>: (int) Default=0. Block size for dask processing (0 means auto).</LI>
            <LI><strong>depth_band</strong>: (str) Default=i. Band for depth maps.</LI>
            <LI><strong>snr_threshold</strong>: (float) Default=10.0. S/N value for depth maps.</LI>
            <LI><strong>snr_delta</strong>: (float) Default=1.0. Delta for S/N thresholding.</LI>
            <LI><strong>det_prob_threshold</strong>: (float) Default=0.8. Detection probability threshold for SSI depth.</LI>
            <LI><strong>mag_delta</strong>: (float) Default=0.01. Magnitude bin size for detection probability depth.</LI>
            <LI><strong>min_depth</strong>: (float) Default=18. Minimum magnitude for detection probability depth.</LI>
            <LI><strong>max_depth</strong>: (float) Default=26. Maximum magnitude for detection probability depth.</LI>
            <LI><strong>smooth_det_frac</strong>: (bool) Default=True. Apply smoothing to detection fraction vs magnitude.</LI>
            <LI><strong>smooth_window</strong>: (float) Default=0.5. Smoothing window size in magnitudes.</LI>
            </UL>



.. autotxclass:: txpipe.map_plots.TXMapPlotsSSI
    :members:
    :exclude-members: run

    Inputs: 

    - aux_ssi_maps: MapsFile

    Outputs: 

    - depth_ssi_meas_map: PNGFile
    - depth_ssi_true_map: PNGFile
    - depth_ssi_det_prob_map: PNGFile
    
    Parallel: No - Serial


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>projection</strong>: (str) Default=cart. Projection type for map plots (e.g., cart, moll)</LI>
            <LI><strong>rot180</strong>: (bool) Default=False. Whether to rotate the map by 180 degrees</LI>
            <LI><strong>debug</strong>: (bool) Default=False. Enable debug mode for plotting</LI>
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
            <LI><strong>injection_catalog_name</strong>: (str) Default=. Catalog of objects manually injected.</LI>
            <LI><strong>ssi_photometry_catalog_name</strong>: (str) Default=. Catalog of objects from real data with no injections.</LI>
            <LI><strong>ssi_uninjected_photometry_catalog_name</strong>: (str) Default=. Catalog of objects from real data with no injections.</LI>
            <LI><strong>GCRcatalog_path</strong>: (str) Default=. Path to GCRCatalogs for SSI runs.</LI>
            <LI><strong>flux_name</strong>: (str) Default=gaap3p0Flux. Flux column name to use.</LI>
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
            <LI><strong>chunk_rows</strong>: (int) Default=100000. Number of rows to process in each chunk.</LI>
            <LI><strong>match_radius</strong>: (float) Default=0.5. Matching radius in arcseconds.</LI>
            <LI><strong>magnification</strong>: (int) Default=0. Magnification label.</LI>
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
            <LI><strong>chunk_rows</strong>: (int) Default=10000. Number of rows to process in each chunk.</LI>
            <LI><strong>applied_magnification</strong>: (float) Default=1.02. Magnification applied to the 'magnified' SSI catalog.</LI>
            <LI><strong>n_patches</strong>: (int) Default=20. Number of patches for bootstrap error estimation.</LI>
            <LI><strong>bootstrap_error</strong>: (bool) Default=True. Whether to compute bootstrap errors.</LI>
            </UL>


