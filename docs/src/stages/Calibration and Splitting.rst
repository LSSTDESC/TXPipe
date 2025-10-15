Calibration and Splitting
=========================

These stages deal with calibrating shear and splitting up catalogs into
sub-catalogs.

* :py:class:`~txpipe.lens_selector.TXLensCatalogSplitter` - Split a lens catalog file into a new file with separate bins

* :py:class:`~txpipe.lens_selector.TXTruthLensCatalogSplitter` - Split a lens catalog file into a new file with separate bins with true redshifts.

* :py:class:`~txpipe.lens_selector.TXExternalLensCatalogSplitter` - Split an external lens catalog into bins

* :py:class:`~txpipe.lens_selector.TXTruthLensCatalogSplitterWeighted` - Split a lens catalog file into a new file with separate bins with true redshifts.

* :py:class:`~txpipe.twopoint_null_tests.TXStarCatalogSplitter` - Split a star catalog into bright and dim stars

* :py:class:`~txpipe.calibrate.TXShearCalibration` - Split the shear catalog into calibrated bins



.. autotxclass:: txpipe.lens_selector.TXLensCatalogSplitter
    :members:
    :exclude-members: run

    Inputs: 

    - lens_tomography_catalog_unweighted: TomographyCatalog
    - photometry_catalog: PhotometryCatalog
    - fiducial_cosmology: FiducialCosmology
    - lens_photoz_pdfs: HDFFile

    Outputs: 

    - binned_lens_catalog_unweighted: HDFFile
    
    Parallel: Yes - MPI


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>initial_size</strong>: (int) Default=100000. Initial size of the output catalog.</LI>
            <LI><strong>chunk_rows</strong>: (int) Default=100000. Number of rows to process in each chunk.</LI>
            <LI><strong>extra_cols</strong>: (list) Default=['']. Extra columns to include in output.</LI>
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

    - binned_lens_catalog_unweighted: HDFFile
    
    Parallel: Yes - MPI


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>initial_size</strong>: (int) Default=100000. Initial size of the output catalog.</LI>
            <LI><strong>chunk_rows</strong>: (int) Default=100000. Number of rows to process in each chunk.</LI>
            <LI><strong>extra_cols</strong>: (list) Default=['']. Extra columns to include in output.</LI>
            <LI><strong>redshift_column</strong>: (str) Default=redshift_true. </LI>
            </UL>



.. autotxclass:: txpipe.lens_selector.TXExternalLensCatalogSplitter
    :members:
    :exclude-members: run

    Inputs: 

    - lens_tomography_catalog_unweighted: TomographyCatalog
    - lens_catalog: HDFFile
    - fiducial_cosmology: FiducialCosmology

    Outputs: 

    - binned_lens_catalog_unweighted: HDFFile
    
    Parallel: Yes - MPI


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>initial_size</strong>: (int) Default=100000. Initial size of the output catalog.</LI>
            <LI><strong>chunk_rows</strong>: (int) Default=100000. Number of rows to process in each chunk.</LI>
            <LI><strong>extra_cols</strong>: (list) Default=['']. Extra columns to include in output.</LI>
            <LI><strong>redshift_column</strong>: (str) Default=redshift. </LI>
            </UL>



.. autotxclass:: txpipe.lens_selector.TXTruthLensCatalogSplitterWeighted
    :members:
    :exclude-members: run

    Inputs: 

    - lens_tomography_catalog: TomographyCatalog
    - photometry_catalog: PhotometryCatalog
    - fiducial_cosmology: FiducialCosmology

    Outputs: 

    - binned_lens_catalog: HDFFile
    
    Parallel: Yes - MPI


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>initial_size</strong>: (int) Default=100000. Initial size of the output catalog.</LI>
            <LI><strong>chunk_rows</strong>: (int) Default=100000. Number of rows to process in each chunk.</LI>
            <LI><strong>extra_cols</strong>: (list) Default=['']. Extra columns to include in output.</LI>
            <LI><strong>redshift_column</strong>: (str) Default=redshift_true. </LI>
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

    Outputs: 

    - binned_shear_catalog: ShearCatalog
    
    Parallel: Yes - MPI


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>use_true_shear</strong>: (bool) Default=False. Use true shear values instead of observed shear</LI>
            <LI><strong>chunk_rows</strong>: (int) Default=100000. Number of rows to process in each chunk</LI>
            <LI><strong>subtract_mean_shear</strong>: (bool) Default=True. Whether to subtract the mean shear from the calibrated shear</LI>
            <LI><strong>extra_cols</strong>: (list) Default=['']. Additional columns to copy from the input catalog</LI>
            <LI><strong>shear_catalog_type</strong>: (str) Default=. Type of shear catalog (e.g., metadetect, metacal, lensfit, hsc)</LI>
            <LI><strong>shear_prefix</strong>: (str) Default=. Prefix for shear-related columns in the input catalog</LI>
            </UL>


