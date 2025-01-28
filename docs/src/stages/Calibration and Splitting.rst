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

    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>initial_size</strong>: (int) Default=100000. </LI>
            <LI><strong>chunk_rows</strong>: (int) Default=100000. </LI>
            <LI><strong>extra_cols</strong>: (list) Default=['']. </LI>
            <LI><strong>redshift_column</strong>: (str) Default=zmean. </LI>
            </UL>



.. autotxclass:: txpipe.lens_selector.TXTruthLensCatalogSplitter
    :members:
    :exclude-members: run

    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>initial_size</strong>: (int) Default=100000. </LI>
            <LI><strong>chunk_rows</strong>: (int) Default=100000. </LI>
            <LI><strong>extra_cols</strong>: (list) Default=['']. </LI>
            <LI><strong>redshift_column</strong>: (str) Default=redshift_true. </LI>
            </UL>



.. autotxclass:: txpipe.lens_selector.TXExternalLensCatalogSplitter
    :members:
    :exclude-members: run

    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>initial_size</strong>: (int) Default=100000. </LI>
            <LI><strong>chunk_rows</strong>: (int) Default=100000. </LI>
            <LI><strong>extra_cols</strong>: (list) Default=['']. </LI>
            <LI><strong>redshift_column</strong>: (str) Default=redshift. </LI>
            </UL>



.. autotxclass:: txpipe.lens_selector.TXTruthLensCatalogSplitterWeighted
    :members:
    :exclude-members: run

    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>initial_size</strong>: (int) Default=100000. </LI>
            <LI><strong>chunk_rows</strong>: (int) Default=100000. </LI>
            <LI><strong>extra_cols</strong>: (list) Default=['']. </LI>
            <LI><strong>redshift_column</strong>: (str) Default=redshift_true. </LI>
            </UL>



.. autotxclass:: txpipe.twopoint_null_tests.TXStarCatalogSplitter
    :members:
    :exclude-members: run

    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>chunk_rows</strong>: (int) Default=100000. </LI>
            <LI><strong>initial_size</strong>: (int) Default=100000. </LI>
            </UL>



.. autotxclass:: txpipe.calibrate.TXShearCalibration
    :members:
    :exclude-members: run

    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>use_true_shear</strong>: (bool) Default=False. </LI>
            <LI><strong>chunk_rows</strong>: (int) Default=100000. </LI>
            <LI><strong>subtract_mean_shear</strong>: (bool) Default=True. </LI>
            <LI><strong>extra_cols</strong>: (list) Default=['']. </LI>
            <LI><strong>shear_catalog_type</strong>: (str) Default=. </LI>
            <LI><strong>shear_prefix</strong>: (str) Default=. </LI>
            </UL>


