Tomography
==========

These stages assign objects in a shear catalog to tomographic bins,
independently of the source selection. They write a bin assignments file.
See :ref:`Source selection and tomography`.

* :py:class:`~txpipe.rail.summarize.PZCollateRailAssignmentsMetadetect` - Collect RAIL tomographic bin assignments for the metadetect variants into one file.

* :py:class:`~txpipe.rail.summarize.PZCollateRailAssignmentsScalarMetadetect` - Collect RAIL tomographic bin assignments for the scalar metadetect variants into one file.

* :py:class:`~txpipe.rail.summarize.PZGenerateSOMInformerSubsample` - Make a random subsample of the selected shear catalog objects, to train a SOM.

* :py:class:`~txpipe.tomography.TXTrainRandomForestTomography` - Train a random forest classifier to assign source galaxies to tomographic bins based on their ph ...

* :py:class:`~txpipe.tomography.TXComputeSourceTomography` - Base class for tomography.

* :py:class:`~txpipe.tomography.TXApplyRandomForestSourceTomography` - Assign source galaxies to tomographic bins using a trained random forest.

* :py:class:`~txpipe.tomography.TXSimpleRedshiftSourceTomography` - Assign source galaxies to tomographic bins by cutting on a redshift column.



.. autotxclass:: txpipe.rail.summarize.PZCollateRailAssignmentsMetadetect
    :members:
    :exclude-members: run

    Inputs: 

    - bhat_for_wide_data_ns: HDFFile
    - bhat_for_wide_data_1p: HDFFile
    - bhat_for_wide_data_1m: HDFFile
    - bhat_for_wide_data_2p: HDFFile
    - bhat_for_wide_data_2m: HDFFile

    Outputs: 

    - shear_bin_assignments: BinAssignments
    
    Parallel: Yes - MPI


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>source_zbin_edges</strong>: (list) Required. Redshift bin edges for source tomography</LI>
            </UL>



.. autotxclass:: txpipe.rail.summarize.PZCollateRailAssignmentsScalarMetadetect
    :members:
    :exclude-members: run

    Inputs: 

    - bhat_for_wide_data_ns: HDFFile
    - bhat_for_wide_data_1p: HDFFile
    - bhat_for_wide_data_1m: HDFFile

    Outputs: 

    - shear_bin_assignments: BinAssignments
    
    Parallel: Yes - MPI


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>source_zbin_edges</strong>: (list) Required. Redshift bin edges for source tomography</LI>
            </UL>



.. autotxclass:: txpipe.rail.summarize.PZGenerateSOMInformerSubsample
    :members:
    :exclude-members: run

    Inputs: 

    - shear_selection: CatalogSelection
    - shear_catalog: ShearCatalog

    Outputs: 

    - sub_selected_shear_data: HDFFile
    
    Parallel: Yes - MPI


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>size</strong>: (int) Default=5000000. Approximate of selected objects to include.</LI>
            <LI><strong>seed</strong>: (int) Default=7953. Random seed for sub-selection.</LI>
            <LI><strong>chunk_rows</strong>: (int) Default=100000. Number of rows to process at once</LI>
            </UL>



.. autotxclass:: txpipe.tomography.TXTrainRandomForestTomography
    :members:
    :exclude-members: run

    Inputs: 

    - spectroscopic_catalog: HDFFile

    Outputs: 

    - shear_tomography_classifier: PickleFile
    
    Parallel: Yes - MPI


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>bands</strong>: (list) Default=[&#x27;r&#x27;, &#x27;i&#x27;, &#x27;z&#x27;]. Bands from the catalog to use for selection</LI>
            <LI><strong>spec_mag_column_format</strong>: (str) Default=photometry/{band}. Format string for spectroscopic magnitude columns</LI>
            <LI><strong>spec_redshift_column</strong>: (str) Default=photometry/redshift. Column name for spectroscopic redshifts</LI>
            <LI><strong>source_zbin_edges</strong>: (list) Required. Redshift bin edges for source tomography</LI>
            <LI><strong>random_seed</strong>: (int) Default=42. Random seed for reproducibility</LI>
            </UL>



.. autotxclass:: txpipe.tomography.TXComputeSourceTomography
    :members:
    :exclude-members: run

    Inputs: 

    - shear_catalog: ShearCatalog

    Outputs: 

    - shear_bin_assignments: BinAssignments
    
    Parallel: Yes - MPI


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>chunk_rows</strong>: (int) Default=10000. Rows to read at once</LI>
            </UL>



.. autotxclass:: txpipe.tomography.TXApplyRandomForestSourceTomography
    :members:
    :exclude-members: run

    Inputs: 

    - shear_catalog: ShearCatalog
    - shear_tomography_classifier: PickleFile

    Outputs: 

    - shear_bin_assignments: BinAssignments
    
    Parallel: Yes - MPI


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            </UL>



.. autotxclass:: txpipe.tomography.TXSimpleRedshiftSourceTomography
    :members:
    :exclude-members: run

    Inputs: 

    - shear_catalog: ShearCatalog

    Outputs: 

    - shear_bin_assignments: BinAssignments
    
    Parallel: Yes - MPI


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>redshift_col</strong>: (str) Default=redshift_true. Column name to use for simple redshift cut</LI>
            <LI><strong>source_zbin_edges</strong>: (list) Required. Redshift bin edges for source tomography</LI>
            </UL>


