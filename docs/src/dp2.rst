Rubin DP2 data
==============

TXPipe can analyse the metadetect shear catalogs from Rubin Data Preview 2 (DP2), both as released by Rubin and as re-processed by DESC. The example pipelines are in ``examples/dp2``. The workflow has two parts: ingestion, which reads the Rubin/DESC data products and makes TXPipe-format files, and analysis.

Ingestion
---------

Ingestion converts the DP2 products into the TXPipe formats described in :ref:`File Types`. It reads butler or parquet files, so it normally has to be run in an environment with the LSST science pipelines, such as the Rubin Science Platform (see :ref:`Stages requiring the LSST environment`). The stages are:

- :py:class:`~txpipe.ingest.metaDetect.TXGenerateTractList` makes a list of the tracts to use (``tract_list``), based on the mask, and plots them.
- :py:class:`~txpipe.ingest.metaDetect.TXIngestRubinMetaDetect` ingests the Rubin metadetect shear catalogs into a :ref:`metadetect catalog<Metadetect catalogs>`.
- :py:class:`~txpipe.ingest.metaDetect.TXIngestDESCMetaDetectV1_1` ingests the DESC-run metadetect catalogs (version 1.1) into a :ref:`scalar metadetect catalog<Scalar metadetect catalogs>`.
- :py:class:`~txpipe.ingest.metaDetect.TXIngestDP2Photometry` ingests the photometry catalog for the listed tracts.
- :py:class:`~txpipe.ingest.metaDetect.TXIngestHealsparseMask` converts a healsparse mask file to a TXPipe mask.
- :py:class:`~txpipe.masks.TXCutShearCatalog` can cut the ingested catalog down to a smaller region, to make test data sets.

The pipelines for these are ``examples/dp2/ingestion.yml`` (Rubin) and ``examples/dp2/ingestion1.1.yml`` (DESC version 1.1). They are marked as drafts in the files. The ingestion can be run in parallel with MPI, and the ``nprocess`` setting for each stage in the pipeline file controls this.

Analysis
--------

There are three analysis pipelines:

.. list-table::
    :header-rows: 1

    * - Pipeline
      - Catalog
      - Selection and tomography
    * - ``analysis_rubin_metadetect.yml``
      - Rubin metadetect
      - :py:class:`~txpipe.source_selection.TXDP2RubinMetadetectSelection`, with tomography from RAIL assignments collated by :py:class:`~txpipe.rail.summarize.PZCollateRailAssignmentsMetadetect`
    * - ``analysis_desc_metadetect_1.1.yml``
      - DESC scalar metadetect v1.1
      - :py:class:`~txpipe.source_selection.TXDP2DESCSelection`, with tomography from :py:class:`~txpipe.rail.summarize.PZCollateRailAssignmentsScalarMetadetect`
    * - ``analysis1.1-mini.yml``
      - A 1 square degree subset
      - As for the DESC version, for quick tests

Each has a configuration file with the options for all the stages (``config_rubin_metadetect.yml``, ``config_desc_metadetect_1.1.yml`` and ``config-mini.yml``). They use the Jinja2 templating in ceci to run the tomography stages once for each shear variant.

The tomography in these pipelines uses a self-organizing map (SOM) in RAIL. In outline:

1. The selection stage chooses the source sample.
2. :py:class:`~txpipe.rail.summarize.PZGenerateSOMInformerSubsample` takes a random subsample of the selected objects to train the SOM.
3. The RAIL SOM stages are trained, assign objects in each variant of the catalog to bins, and compute the tomographic n(z).
4. :py:class:`~txpipe.rail.summarize.PZCollateRailAssignmentsMetadetect` (or the scalar version) gathers the assignments for all the variants into a :ref:`bin assignments file<Tomographic bin assignments>`.
5. The calibration stage computes the response of each bin, as for any other metadetect catalog, and the rest of the pipeline (maps, two-point measurements, covariance, diagnostics) follows the usual structure.
6. :py:class:`~txpipe.rail.summarize.PZAddNonTomographicNZ` adds the non-tomographic n(z) to the photo-z stack.

The RAIL stages need the RAIL packages listed in ``bin/environment-local.yml``. See :ref:`Source selection and tomography` for the general structure.

Outputs from the example pipelines for the Rubin and DESC catalogs are shown in ``examples/dp2/dp2-shear.png`` and ``examples/dp2/dp2-shear-and-pz.png``.
