Two-Point
=========

These stages deal with measuring or predicting two-point statistics.

* :py:class:`~txpipe.random_cats.TXRandomCat` - Generate a catalog of randomly positioned points

* :py:class:`~txpipe.random_cats.TXSubsampleRandoms` - Randomly subsample the binned random catalog and save catalog

* :py:class:`~txpipe.twopoint_fourier.TXTwoPointFourier` - Make Fourier space 3x2pt measurements using NaMaster

* :py:class:`~txpipe.twopoint.TXTwoPoint` - Make 2pt measurements using TreeCorr

* :py:class:`~txpipe.twopoint.TXTwoPointPixel` - This subclass of the standard TXTwoPoint uses maps to compute

* :py:class:`~txpipe.twopoint.TXTwoPointPixelExtCross` - TXTwoPointPixel - External - Cross correlation

* :py:class:`~txpipe.theory.TXTwoPointTheoryReal` - Compute theory predictions for real-space 3x2pt measurements.

* :py:class:`~txpipe.theory.TXTwoPointTheoryFourier` - Compute theory predictions for Fourier-space 3x2pt measurements.

* :py:class:`~txpipe.jackknife.TXJackknifeCenters` - Generate jack-knife centers from random catalogs.

* :py:class:`~txpipe.jackknife.TXJackknifeCentersSource` - Stage TXJackknifeCentersSource

* :py:class:`~txpipe.extensions.clmm.rlens.TXTwoPointRLens` - Measure 2-pt shear-position using the Rlens metric



.. autoclass:: txpipe.random_cats.TXRandomCat
   :members:


.. autoclass:: txpipe.random_cats.TXSubsampleRandoms
   :members:


.. autoclass:: txpipe.twopoint_fourier.TXTwoPointFourier
   :members:


.. autoclass:: txpipe.twopoint.TXTwoPoint
   :members:


.. autoclass:: txpipe.twopoint.TXTwoPointPixel
   :members:


.. autoclass:: txpipe.twopoint.TXTwoPointPixelExtCross
   :members:


.. autoclass:: txpipe.theory.TXTwoPointTheoryReal
   :members:


.. autoclass:: txpipe.theory.TXTwoPointTheoryFourier
   :members:


.. autoclass:: txpipe.jackknife.TXJackknifeCenters
   :members:


.. autoclass:: txpipe.jackknife.TXJackknifeCentersSource
   :members:


.. autoclass:: txpipe.extensions.clmm.rlens.TXTwoPointRLens
   :members:

