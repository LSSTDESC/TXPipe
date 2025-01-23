Source Injection
================

These stages ingest and use synthetic source injection information

* :py:class:`~txpipe.auxiliary_maps.TXAuxiliarySSIMaps` - Generate auxiliary maps from SSI catalogs

* :py:class:`~txpipe.map_plots.TXMapPlotsSSI` - Make plots of all the available maps that use SSI inputs

* :py:class:`~txpipe.ingest.ssi.TXIngestSSI` - Base-Class for ingesting SSI catalogs

* :py:class:`~txpipe.ingest.ssi.TXIngestSSIGCR` - Ingest SSI catalogs using GCR

* :py:class:`~txpipe.ingest.ssi.TXMatchSSI` - Match an SSI injection catalog and a photometry catalog

* :py:class:`~txpipe.ingest.ssi.TXIngestSSIDESBalrog` - Base-stage for ingesting a DES SSI catalog AKA "Balrog"

* :py:class:`~txpipe.ingest.ssi.TXIngestSSIMatchedDESBalrog` - Ingest a matched "SSI" catalog from DES (AKA Balrog)

* :py:class:`~txpipe.ingest.ssi.TXIngestSSIDetectionDESBalrog` - Ingest an "SSI" "detection" catalog from DES (AKA Balrog)

* :py:class:`~txpipe.magnification.TXSSIMagnification` - Compute the magnification coefficients using SSI outputs



.. autoclass:: txpipe.auxiliary_maps.TXAuxiliarySSIMaps

    **parallel**: Yes - Dask

.. autoclass:: txpipe.map_plots.TXMapPlotsSSI

    **parallel**: No - Serial

.. autoclass:: txpipe.ingest.ssi.TXIngestSSI

    **parallel**: Yes - MPI

.. autoclass:: txpipe.ingest.ssi.TXIngestSSIGCR

    **parallel**: No - Serial

.. autoclass:: txpipe.ingest.ssi.TXMatchSSI

    **parallel**: Yes - MPI

.. autoclass:: txpipe.ingest.ssi.TXIngestSSIDESBalrog

    **parallel**: Yes - MPI

.. autoclass:: txpipe.ingest.ssi.TXIngestSSIMatchedDESBalrog

    **parallel**: Yes - MPI

.. autoclass:: txpipe.ingest.ssi.TXIngestSSIDetectionDESBalrog

    **parallel**: Yes - MPI

.. autoclass:: txpipe.magnification.TXSSIMagnification

    **parallel**: No - Serial
