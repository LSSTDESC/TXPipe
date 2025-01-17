Source Injection
================

These stages ingest and use synthetic source injection information

* :py:class:`~txpipe.ingest.ssi.TXIngestSSIGCR` - Class for ingesting SSI catalogs using GCR

* :py:class:`~txpipe.ingest.ssi.TXMatchSSI` - Class for ingesting SSI injection and photometry catalogs

* :py:class:`~txpipe.ingest.ssi.TXIngestSSIMatched` - Base-stage for ingesting a matched SSI catalog

* :py:class:`~txpipe.ingest.ssi.TXIngestSSIMatchedDESBalrog` - Class for ingesting a matched "SSI" catalog from DES (AKA Balrog)

* :py:class:`~txpipe.magnification.TXSSIMagnification` - class for computing the magnification coefficients using SSI outputs



.. autoclass:: txpipe.ingest.ssi.TXIngestSSIGCR
   :members:


.. autoclass:: txpipe.ingest.ssi.TXMatchSSI
   :members:


.. autoclass:: txpipe.ingest.ssi.TXIngestSSIMatched
   :members:


.. autoclass:: txpipe.ingest.ssi.TXIngestSSIMatchedDESBalrog
   :members:


.. autoclass:: txpipe.magnification.TXSSIMagnification
   :members:

