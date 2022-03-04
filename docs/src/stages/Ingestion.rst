Ingestion
=========

These stages import data into TXPipe input formats, or generate mock data from
simpler input catalogs.

* :py:class:`~txpipe.input_cats.TXCosmoDC2Mock` - Simulate mock shear and photometry measurements from CosmoDC2 (or similar)

* :py:class:`~txpipe.input_cats.TXBuzzardMock` - Simulate mock photometry from Buzzard.

* :py:class:`~txpipe.input_cats.TXGaussianSimsMock` - Simulate mock photometry from gaussian simulations

* :py:class:`~txpipe.metacal_gcr_input.TXMetacalGCRInput` - Ingest metacal catalogs from GCRCatalogs

* :py:class:`~txpipe.metacal_gcr_input.TXIngestStars` - Ingest a star catalog from GCRCatalogs

* :py:class:`~txpipe.exposure_info.TXExposureInfo` - Ingest exposure information from an OpSim database

* :py:class:`~txpipe.ingest_redmagic.TXIngestRedmagic` - Ingest a redmagic catalog



.. autoclass:: txpipe.input_cats.TXCosmoDC2Mock
   :members:


.. autoclass:: txpipe.input_cats.TXBuzzardMock
   :members:


.. autoclass:: txpipe.input_cats.TXGaussianSimsMock
   :members:


.. autoclass:: txpipe.metacal_gcr_input.TXMetacalGCRInput
   :members:


.. autoclass:: txpipe.metacal_gcr_input.TXIngestStars
   :members:


.. autoclass:: txpipe.exposure_info.TXExposureInfo
   :members:


.. autoclass:: txpipe.ingest_redmagic.TXIngestRedmagic
   :members:

