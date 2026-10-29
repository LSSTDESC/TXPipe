Calibration and Splitting
=========================

These stages compute the shear calibration of each tomographic bin from
the source selection and bin assignments, apply it, and split up catalogs
into sub-catalogs. See :ref:`Shear calibration guide`.

* :py:class:`~txpipe.lens_selector.TXLensCatalogSplitter` - Split a lens catalog file into a new file with separate bins

* :py:class:`~txpipe.lens_selector.TXTruthLensCatalogSplitter` - Split a lens catalog file into a new file with separate bins with true redshifts.

* :py:class:`~txpipe.lens_selector.TXExternalLensCatalogSplitter` - Split an external lens catalog into bins

* :py:class:`~txpipe.lens_selector.TXTruthLensCatalogSplitterWeighted` - Split a lens catalog file into a new file with separate bins with true redshifts.

* :py:class:`~txpipe.twopoint_null_tests.TXStarCatalogSplitter` - Split a star catalog into bright and dim stars

* :py:class:`~txpipe.calibrate.TXShearCalibration` - Split the shear catalog into calibrated bins

* :py:class:`~txpipe.shear_calibration.base.TXComputeShearCalibrationBase` - Base stage for source selection using S/N, size, and flag cuts and tomography

* :py:class:`~txpipe.shear_calibration.simple.TXComputeShearCalibrationSimple` - Source selection and tomography for mock catalogs that do not

* :py:class:`~txpipe.shear_calibration.hsc.TXComputeShearCalibrationHSC` - Source selection and tomography for HSC catalogs

* :py:class:`~txpipe.shear_calibration.lensfit.TXComputeShearCalibrationLensfit` - Source selection and tomography for lensfit catalogs

* :py:class:`~txpipe.shear_calibration.metacal.TXComputeShearCalibrationMetacal` - Source selection and tomography for metacal catalogs

* :py:class:`~txpipe.shear_calibration.metacal.TXComputeShearCalibrationDESY3` - Source selection and tomography for DESY3 catalogs

* :py:class:`~txpipe.shear_calibration.metadetect.TXComputeShearCalibrationMetadetect` - Source selection and tomography for metadetect catalogs

* :py:class:`~txpipe.shear_calibration.scalar_metadetect.TXComputeShearCalibrationScalarMetadetect` - Source selection and tomography for scalar metadetect catalogs.



.. autotxclass:: txpipe.lens_selector.TXLensCatalogSplitter
    :members:
    :exclude-members: run

    Inputs: 

    - lens_tomography_catalog_unweighted: TomographyCatalog
    - photometry_catalog: PhotometryCatalog
    - fiducial_cosmology: FiducialCosmology
    - lens_photoz_pdfs: HDFFile

    Outputs: 

    - binned_lens_catalog_unweighted: BinnedCatalog
    
    Parallel: Yes - MPI


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>initial_size</strong>: (int) Default=100000. Initial size of the output catalog.</LI>
            <LI><strong>chunk_rows</strong>: (int) Default=100000. Number of rows to process in each chunk.</LI>
            <LI><strong>extra_cols</strong>: (list) Default=[&#x27;&#x27;]. Extra columns to include in output.</LI>
            <LI><strong>redshift_column</strong>: (str) Default=zmean. Name of the redshift column to use.</LI>
            </UL>



.. autotxclass:: txpipe.lens_selector.TXTruthLensCatalogSplitter
    :members:
    :exclude-members: run

    Inputs: 

    - lens_tomography_catalog_unweighted: TomographyCatalog
    - photometry_catalog: PhotometryCatalog
    - fiducial_cosmology: FiducialCosmology

    Outputs: 

    - binned_lens_catalog_unweighted: BinnedCatalog
    
    Parallel: Yes - MPI


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>initial_size</strong>: (int) Default=100000. Initial size of the output catalog.</LI>
            <LI><strong>chunk_rows</strong>: (int) Default=100000. Number of rows to process in each chunk.</LI>
            <LI><strong>extra_cols</strong>: (list) Default=[&#x27;&#x27;]. Extra columns to include in output.</LI>
            <LI><strong>redshift_column</strong>: (str) Default=redshift_true. Column name for true redshift.</LI>
            </UL>



.. autotxclass:: txpipe.lens_selector.TXExternalLensCatalogSplitter
    :members:
    :exclude-members: run

    Inputs: 

    - lens_tomography_catalog_unweighted: TomographyCatalog
    - lens_catalog: HDFFile
    - fiducial_cosmology: FiducialCosmology

    Outputs: 

    - binned_lens_catalog_unweighted: BinnedCatalog
    
    Parallel: Yes - MPI


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>initial_size</strong>: (int) Default=100000. Initial size of the output catalog.</LI>
            <LI><strong>chunk_rows</strong>: (int) Default=100000. Number of rows to process in each chunk.</LI>
            <LI><strong>extra_cols</strong>: (list) Default=[&#x27;&#x27;]. Extra columns to include in output.</LI>
            <LI><strong>redshift_column</strong>: (str) Default=redshift. Column name for redshift.</LI>
            </UL>



.. autotxclass:: txpipe.lens_selector.TXTruthLensCatalogSplitterWeighted
    :members:
    :exclude-members: run

    Inputs: 

    - lens_tomography_catalog: TomographyCatalog
    - photometry_catalog: PhotometryCatalog
    - fiducial_cosmology: FiducialCosmology

    Outputs: 

    - binned_lens_catalog: BinnedCatalog
    
    Parallel: Yes - MPI


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>initial_size</strong>: (int) Default=100000. Initial size of the output catalog.</LI>
            <LI><strong>chunk_rows</strong>: (int) Default=100000. Number of rows to process in each chunk.</LI>
            <LI><strong>extra_cols</strong>: (list) Default=[&#x27;&#x27;]. Extra columns to include in output.</LI>
            <LI><strong>redshift_column</strong>: (str) Default=redshift_true. Column name for true redshift.</LI>
            </UL>



.. autotxclass:: txpipe.twopoint_null_tests.TXStarCatalogSplitter
    :members:
    :exclude-members: run

    Inputs: 

    - star_catalog: HDFFile

    Outputs: 

    - binned_star_catalog: HDFFile
    
    Parallel: No - Serial


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>chunk_rows</strong>: (int) Default=100000. Number of rows to process in each chunk.</LI>
            <LI><strong>initial_size</strong>: (int) Default=100000. Initial size for star catalog bins.</LI>
            </UL>



.. autotxclass:: txpipe.calibrate.TXShearCalibration
    :members:
    :exclude-members: run

    Inputs: 

    - shear_catalog: ShearCatalog
    - shear_tomography_catalog: TomographyCatalog
    - fiducial_cosmology: FiducialCosmology

    Outputs: 

    - binned_shear_catalog: ShearCatalog
    
    Parallel: Yes - MPI


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>use_true_shear</strong>: (bool) Default=False. Use true shear values instead of observed shear</LI>
            <LI><strong>chunk_rows</strong>: (int) Default=100000. Number of rows to process in each chunk</LI>
            <LI><strong>subtract_mean_shear</strong>: (bool) Default=True. Whether to subtract the mean shear from the calibrated shear</LI>
            <LI><strong>copy_redshift</strong>: (bool) Default=False. Whether to copy the redshift column from the input catalog</LI>
            <LI><strong>add_fiducial_distance</strong>: (bool) Default=False. Whether to add fiducial comoving distance to the output catalog</LI>
            <LI><strong>redshift_name</strong>: (str) Default=redshift_true. Name of the redshift column</LI>
            <LI><strong>extra_cols</strong>: (list) Default=[&#x27;&#x27;]. Additional columns to copy from the input catalog</LI>
            <LI><strong>shear_catalog_type</strong>: (str) Default=. Type of shear catalog (e.g., metadetect, metacal, lensfit, hsc)</LI>
            </UL>



.. autotxclass:: txpipe.shear_calibration.base.TXComputeShearCalibrationBase
    :members:
    :exclude-members: run

    Inputs: 

    - shear_catalog: ShearCatalog
    - shear_bin_assignments: BinAssignments
    - shear_selection: CatalogSelection

    Outputs: 

    - shear_tomography_catalog: TomographyCatalog
    
    Parallel: Yes - MPI


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>verbose</strong>: (bool) Default=False. Whether to print verbose output</LI>
            <LI><strong>chunk_rows</strong>: (int) Default=10000. Number of rows to process in each chunk</LI>
            </UL>



.. autotxclass:: txpipe.shear_calibration.simple.TXComputeShearCalibrationSimple
    :members:
    :exclude-members: run

    Inputs: 

    - shear_catalog: ShearCatalog
    - shear_bin_assignments: BinAssignments
    - shear_selection: CatalogSelection

    Outputs: 

    - shear_tomography_catalog: TomographyCatalog
    
    Parallel: Yes - MPI


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>verbose</strong>: (bool) Default=False. Whether to print verbose output</LI>
            <LI><strong>chunk_rows</strong>: (int) Default=10000. Number of rows to process in each chunk</LI>
            </UL>



.. autotxclass:: txpipe.shear_calibration.hsc.TXComputeShearCalibrationHSC
    :members:
    :exclude-members: run

    Inputs: 

    - shear_catalog: ShearCatalog
    - shear_bin_assignments: BinAssignments
    - shear_selection: CatalogSelection

    Outputs: 

    - shear_tomography_catalog: TomographyCatalog
    
    Parallel: Yes - MPI


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>verbose</strong>: (bool) Default=False. Whether to print verbose output</LI>
            <LI><strong>chunk_rows</strong>: (int) Default=10000. Number of rows to process in each chunk</LI>
            <LI><strong>max_shear_cut</strong>: (float) Default=0.0. Maximum shear value for object selection</LI>
            </UL>



.. autotxclass:: txpipe.shear_calibration.lensfit.TXComputeShearCalibrationLensfit
    :members:
    :exclude-members: run

    Inputs: 

    - shear_catalog: ShearCatalog
    - shear_bin_assignments: BinAssignments
    - shear_selection: CatalogSelection

    Outputs: 

    - shear_tomography_catalog: TomographyCatalog
    
    Parallel: Yes - MPI


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>verbose</strong>: (bool) Default=False. Whether to print verbose output</LI>
            <LI><strong>chunk_rows</strong>: (int) Default=10000. Number of rows to process in each chunk</LI>
            <LI><strong>input_m_is_weighted</strong>: (bool) Required. Whether the input m values are already weighted</LI>
            <LI><strong>dec_cut</strong>: (bool) Default=True. Whether to apply a declination cut</LI>
            </UL>



.. autotxclass:: txpipe.shear_calibration.metacal.TXComputeShearCalibrationMetacal
    :members:
    :exclude-members: run

    Inputs: 

    - shear_catalog: ShearCatalog
    - shear_bin_assignments: BinAssignments
    - shear_selection: CatalogSelection

    Outputs: 

    - shear_tomography_catalog: TomographyCatalog
    
    Parallel: Yes - MPI


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>verbose</strong>: (bool) Default=False. Whether to print verbose output</LI>
            <LI><strong>chunk_rows</strong>: (int) Default=10000. Number of rows to process in each chunk</LI>
            <LI><strong>delta_gamma</strong>: (float) Required. Delta gamma value for metacal response calculation</LI>
            <LI><strong>use_diagonal_response</strong>: (bool) Default=False. Whether to use only diagonal elements of the response matrix</LI>
            </UL>



.. autotxclass:: txpipe.shear_calibration.metacal.TXComputeShearCalibrationDESY3
    :members:
    :exclude-members: run

    Inputs: 

    - shear_catalog: ShearCatalog
    - shear_bin_assignments: BinAssignments
    - shear_selection: CatalogSelection
    - desy3_index_catalog: HDFFile

    Outputs: 

    - shear_tomography_catalog: TomographyCatalog
    
    Parallel: Yes - MPI


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>verbose</strong>: (bool) Default=False. Whether to print verbose output</LI>
            <LI><strong>chunk_rows</strong>: (int) Default=10000. Number of rows to process in each chunk</LI>
            <LI><strong>delta_gamma</strong>: (float) Required. Delta gamma value for metacal response calculation</LI>
            <LI><strong>use_diagonal_response</strong>: (bool) Default=False. Whether to use only diagonal elements of the response matrix</LI>
            </UL>



.. autotxclass:: txpipe.shear_calibration.metadetect.TXComputeShearCalibrationMetadetect
    :members:
    :exclude-members: run

    Inputs: 

    - shear_catalog: ShearCatalog
    - shear_bin_assignments: BinAssignments
    - shear_selection: CatalogSelection

    Outputs: 

    - shear_tomography_catalog: TomographyCatalog
    
    Parallel: Yes - MPI


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>verbose</strong>: (bool) Default=False. Whether to print verbose output</LI>
            <LI><strong>chunk_rows</strong>: (int) Default=10000. Number of rows to process in each chunk</LI>
            <LI><strong>delta_gamma</strong>: (float) Required. Delta gamma value for metadetect response calculation</LI>
            </UL>



.. autotxclass:: txpipe.shear_calibration.scalar_metadetect.TXComputeShearCalibrationScalarMetadetect
    :members:
    :exclude-members: run

    Inputs: 

    - shear_catalog: ShearCatalog
    - shear_bin_assignments: BinAssignments
    - shear_selection: CatalogSelection

    Outputs: 

    - shear_tomography_catalog: TomographyCatalog
    
    Parallel: Yes - MPI


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>verbose</strong>: (bool) Default=False. Whether to print verbose output</LI>
            <LI><strong>chunk_rows</strong>: (int) Default=10000. Number of rows to process in each chunk</LI>
            <LI><strong>delta_gamma</strong>: (float) Required. Delta gamma value for metadetect response calculation</LI>
            </UL>


