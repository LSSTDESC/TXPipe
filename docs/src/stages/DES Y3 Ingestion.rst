DES Y3 Ingestion
================

These stages ingest Dark Energy Survey Year 3 data products.

* :py:class:`~txpipe.ingest.stage3.desy3.TXIngestDESY3Gold` - Ingest the DES Y3 Gold photometry catalog from hdf5 format

* :py:class:`~txpipe.ingest.stage3.desy3.TXIngestDESY3Footprint` - Ingest the DES Y3 Footprint maps (incl. badregions, foregrounds etc) from healsparse format

* :py:class:`~txpipe.ingest.stage3.desy3.TXIngestDESY3SpeczCat` - Ingest the spectroscopic catalog used for DES Y3 training of DNF

* :py:class:`~txpipe.ingest.stage3.desy3.TXIngestDESY3Shear` - Convert the DES Y3 metacal shear catalog into TXPipe format.

* :py:class:`~txpipe.ingest.stage3.desy3.TXIngestDESY3SourceRedshift` - Extract the DES Y3 source n(z) from the published data vector file.



.. autotxclass:: txpipe.ingest.stage3.desy3.TXIngestDESY3Gold
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



.. autotxclass:: txpipe.ingest.stage3.desy3.TXIngestDESY3Footprint
    :members:
    :exclude-members: run

    Inputs: None

    Outputs: 

    - des_footprint_maps: MapsFile
    
    Parallel: No - Serial


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>input_nside</strong>: (int) Default=0. Input HEALPix nside value.</LI>
            <LI><strong>input_nest</strong>: (bool) Default=True. Whether input maps use NESTED ordering.</LI>
            <LI><strong>input_filepaths</strong>: (list) Default=[&#x27;&#x27;]. List of input file paths.</LI>
            <LI><strong>input_labels</strong>: (list) Default=[&#x27;&#x27;]. Labels to give the input maps.</LI>
            <LI><strong>chunk_rows</strong>: (int) Default=100000. Number of rows to process in each chunk.</LI>
            </UL>



.. autotxclass:: txpipe.ingest.stage3.desy3.TXIngestDESY3SpeczCat
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



.. autotxclass:: txpipe.ingest.stage3.desy3.TXIngestDESY3Shear
    :members:
    :exclude-members: run

    Inputs: 

    - des_shear_catalog: HDFFile

    Outputs: 

    - shear_catalog: ShearCatalog
    
    Parallel: No - Serial


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>input_group_name</strong>: (str) Default=catalog/. Input group name in the HDF5 file.</LI>
            <LI><strong>chunk_rows</strong>: (int) Default=100000. Number of rows to process in each chunk.</LI>
            </UL>



.. autotxclass:: txpipe.ingest.stage3.desy3.TXIngestDESY3SourceRedshift
    :members:
    :exclude-members: run

    Inputs: 

    - des_datavector_file: FitsFile

    Outputs: 

    - shear_photoz_stack: QPNOfZFile
    
    Parallel: No - Serial


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            </UL>


