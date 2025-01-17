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



.. autoclass:: txpipe.lens_selector.TXLensCatalogSplitter
   :members:


.. autoclass:: txpipe.lens_selector.TXTruthLensCatalogSplitter
   :members:


.. autoclass:: txpipe.lens_selector.TXExternalLensCatalogSplitter
   :members:


.. autoclass:: txpipe.lens_selector.TXTruthLensCatalogSplitterWeighted
   :members:


.. autoclass:: txpipe.twopoint_null_tests.TXStarCatalogSplitter
   :members:


.. autoclass:: txpipe.calibrate.TXShearCalibration
   :members:

