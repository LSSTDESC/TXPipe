Ingestion
=========

These stages import data into TXPipe input formats, or generate mock data from
simpler input catalogs.

* :py:class:`~txpipe.exposure_info.TXExposureInfo` - Ingest exposure information from an OpSim database

* :py:class:`~txpipe.ingest.dp02.TXIngestDataPreview02` - Ingest galaxy catalogs from DP0.2

* :py:class:`~txpipe.ingest.mocks.TXCosmoDC2Mock` - Simulate mock shear and photometry measurements from CosmoDC2 (or similar)

* :py:class:`~txpipe.ingest.mocks.TXBuzzardMock` - Simulate mock photometry from Buzzard.

* :py:class:`~txpipe.ingest.mocks.TXGaussianSimsMock` - Simulate mock photometry from gaussian simulations

* :py:class:`~txpipe.ingest.mocks.TXSimpleMock` - Load an ascii astropy table and put it in shear catalog format.

* :py:class:`~txpipe.ingest.mocks.TXMockTruthPZ` - Stage TXMockTruthPZ

* :py:class:`~txpipe.ingest.gcr.TXMetacalGCRInput` - Ingest metacal catalogs from GCRCatalogs

* :py:class:`~txpipe.ingest.gcr.TXIngestStars` - Ingest a star catalog from GCRCatalogs

* :py:class:`~txpipe.ingest.redmagic.TXIngestRedmagic` - Ingest a redmagic catalog

* :py:class:`~txpipe.simulation.TXLogNormalGlass` - Uses GLASS to generate a simulated catalog from lognormal fields



.. autoclass:: txpipe.exposure_info.TXExposureInfo
   :members:


.. autoclass:: txpipe.ingest.dp02.TXIngestDataPreview02
   :members:


.. autoclass:: txpipe.ingest.mocks.TXCosmoDC2Mock
   :members:


.. autoclass:: txpipe.ingest.mocks.TXBuzzardMock
   :members:


.. autoclass:: txpipe.ingest.mocks.TXGaussianSimsMock
   :members:


.. autoclass:: txpipe.ingest.mocks.TXSimpleMock
   :members:


.. autoclass:: txpipe.ingest.mocks.TXMockTruthPZ
   :members:


.. autoclass:: txpipe.ingest.gcr.TXMetacalGCRInput
   :members:


.. autoclass:: txpipe.ingest.gcr.TXIngestStars
   :members:


.. autoclass:: txpipe.ingest.redmagic.TXIngestRedmagic
   :members:


.. autoclass:: txpipe.simulation.TXLogNormalGlass
   :members:

