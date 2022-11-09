Photo-z
=======

These stages deal with photo-z PDF training and estimation


* :py:class:`~txpipe.rail.train.PZRailTrainSource` - Train a photo-z model on the source sample using RAIL

* :py:class:`~txpipe.rail.train.PZRailTrainLens` - Train a photo-z model on the lens sample using RAIL

* :py:class:`~txpipe.rail.train.PZRailTrainLensFromSource` - Copy the RAIL source trained model to the lens file

* :py:class:`~txpipe.rail.train.PZRailTrainSourceFromLens` - Copy the RAIL lens trained model to the source file

* :py:class:`~txpipe.rail.estimate.PZRailEstimateSource` - Estimate source redshift PDFs and best-fits using RAIL

* :py:class:`~txpipe.rail.estimate.PZRailEstimateLens` - Estimate source redshift PDFs and best-fits using RAIL

* :py:class:`~txpipe.rail.estimate.PZRailEstimateSourceFromLens` - Make a source redshifts file by copying lens redshifts

* :py:class:`~txpipe.rail.estimate.PZRailEstimateLensFromSource` - Make a lens  redshifts file by copying source redshifts



.. autoclass:: txpipe.rail.train.PZRailTrainSource
   :members:


.. autoclass:: txpipe.rail.train.PZRailTrainLens
   :members:


.. autoclass:: txpipe.rail.train.PZRailTrainLensFromSource
   :members:


.. autoclass:: txpipe.rail.train.PZRailTrainSourceFromLens
   :members:


.. autoclass:: txpipe.rail.estimate.PZRailEstimateSource
   :members:


.. autoclass:: txpipe.rail.estimate.PZRailEstimateLens
   :members:


.. autoclass:: txpipe.rail.estimate.PZRailEstimateSourceFromLens
   :members:


.. autoclass:: txpipe.rail.estimate.PZRailEstimateLensFromSource
   :members:

