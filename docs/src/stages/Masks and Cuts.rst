Masks and Cuts
==============

These stages make masks, and cut catalogs down to sub-samples.

* :py:class:`~txpipe.masks.TXBaseMask` - Base class for generating binary survey masks using auxiliary input maps.

* :py:class:`~txpipe.masks.TXSimpleMask` - Generate a simple binary mask using cuts on depth and bright object maps.

* :py:class:`~txpipe.masks.TXSimpleMaskSource` - Generate a binary mask for source galaxies using positive lensing weights

* :py:class:`~txpipe.masks.TXSimpleMaskFrac` - Make a simple mask using a depth cut and bright object cut

* :py:class:`~txpipe.masks.TXCustomMask` - Make a mask from a custom list of cuts to aux maps (e.g depth cut and bright object cuts)

* :py:class:`~txpipe.masks.TXJointMask` - Combine two binary masks into one using AND intersection.

* :py:class:`~txpipe.masks.TXDESIJointMask` - Combine the DESI lens mask and the shear catalog mask using AND intersection.

* :py:class:`~txpipe.masks.TXCutCatalog` - Cut a catalog to the footprint defined by an input mask, and write out the cut catalog.

* :py:class:`~txpipe.masks.TXCutShearCatalog` - Cut a shear catalog to the footprint defined by an input mask.



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

    - depth_map: MapsFile
    - bright_object_map: MapsFile

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
            <LI><strong>require_all_bins</strong>: (bool) Default=True. </LI>
            </UL>



.. autotxclass:: txpipe.masks.TXSimpleMaskFrac
    :members:
    :exclude-members: run

    Inputs: 

    - depth_map: MapsFile
    - bright_object_map: MapsFile

    Outputs: 

    - mask: MapsFile
    
    Parallel: No - Serial


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>depth_cut</strong>: (float) Default=23.5. Depth cut for mask creation.</LI>
            <LI><strong>bright_object_max</strong>: (float) Default=10.0. Maximum allowed bright object count.</LI>
            <LI><strong>supreme_map_file</strong>: (str) Default=none. Path to supreme map file for fracdet computation.</LI>
            <LI><strong>supreme_map_files</strong>: (list) Default=[]. List of supreme map files for fracdet computation.</LI>
            <LI><strong>frac_cut</strong>: (float) Default=0.0. Minimum fractional coverage to keep pixel in mask.</LI>
            </UL>



.. autotxclass:: txpipe.masks.TXCustomMask
    :members:
    :exclude-members: run

    Inputs: 

    - footprint_maps: MapsFile

    Outputs: 

    - mask: MapsFile
    
    Parallel: No - Serial


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>fracdet_name</strong>: (str) Default=footprint/fracdet_griz. Fracdet map name.</LI>
            <LI><strong>cuts</strong>: (list) Default=[&#x27;footprint/fracdet_griz &gt; 0&#x27;]. List of mask cuts to apply.</LI>
            <LI><strong>degrade</strong>: (bool) Default=False. Degrade resolution if input map Nside differs from config nside.</LI>
            </UL>



.. autotxclass:: txpipe.masks.TXJointMask
    :members:
    :exclude-members: run

    Inputs: 

    - mask1: MapsFile
    - mask2: MapsFile

    Outputs: 

    - mask: MapsFile
    
    Parallel: No - Serial


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            </UL>



.. autotxclass:: txpipe.masks.TXDESIJointMask
    :members:
    :exclude-members: run

    Inputs: 

    - desi_mask: MapsFile
    - shear_mask: MapsFile

    Outputs: 

    - mask: MapsFile
    
    Parallel: No - Serial


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            </UL>



.. autotxclass:: txpipe.masks.TXCutCatalog
    :members:
    :exclude-members: run

    Inputs: 

    - catalog: HDFFile
    - mask: MapsFile

    Outputs: 

    - cut_catalog: HDFFile
    
    Parallel: Yes - MPI


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>chunk_rows</strong>: (int) Default=100000. Number of rows to read per chunk.</LI>
            <LI><strong>catalog_groups</strong>: (list) Default=[&#x27;catalog&#x27;]. HDF5 group names in the input catalog.</LI>
            <LI><strong>ra_col</strong>: (str) Default=ra. RA column name.</LI>
            <LI><strong>dec_col</strong>: (str) Default=dec. Dec column name.</LI>
            </UL>



.. autotxclass:: txpipe.masks.TXCutShearCatalog
    :members:
    :exclude-members: run

    Inputs: 

    - shear_catalog: ShearCatalog
    - mask: MapsFile

    Outputs: 

    - cut_shear_catalog: HDFFile
    
    Parallel: Yes - MPI


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>chunk_rows</strong>: (int) Default=100000. Number of rows to read per chunk.</LI>
            <LI><strong>ra_col</strong>: (str) Default=ra. RA column name.</LI>
            <LI><strong>dec_col</strong>: (str) Default=dec. Dec column name.</LI>
            </UL>


