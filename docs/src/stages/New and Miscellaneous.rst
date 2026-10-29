New and Miscellaneous
=====================

These stages either don't fit into a category above or have not yet been
assigned to one.

* :py:class:`~txpipe.metadata.TXTracerMetadata` - Collate metadata from various other files

* :py:class:`~txpipe.rail.conversions.TXParqetToHDF` - Generic stage to convert a Parquet File to HDF



.. autotxclass:: txpipe.metadata.TXTracerMetadata
    :members:
    :exclude-members: run

    Inputs: 

    - shear_catalog: ShearCatalog
    - shear_tomography_catalog: TomographyCatalog
    - lens_tomography_catalog: TomographyCatalog
    - mask: MapsFile

    Outputs: 

    - tracer_metadata: HDFFile
    - tracer_metadata_yml: YamlFile
    
    Parallel: No - Serial


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            </UL>



.. autotxclass:: txpipe.rail.conversions.TXParqetToHDF
    :members:
    :exclude-members: run

    Inputs: 

    - input: ParquetFile

    Outputs: 

    - output: HDFFile
    
    Parallel: No - Serial


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>hdf_group</strong>: (str) Default=/. HDF group path to write data to.</LI>
            </UL>


