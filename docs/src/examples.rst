Example pipelines
=================

The ``examples`` directory contains pipeline files (``pipeline.yml`` or similar) and the matching configuration files (``config.yml``). Run any of them with ``ceci``:

.. code-block:: bash

    ceci examples/<name>/pipeline.yml

``ceci --dry-run`` prints the command for each stage instead of running anything, and is a quick way to check that a pipeline and its configuration are valid. See :ref:`Configuration Files` for the pipeline and configuration file formats.

The pipelines marked "CI" below are run end to end on every pull request, on the example data described in :ref:`Running an example pipeline`. The others need other data sets, which are mostly available to DESC members at NERSC.

.. list-table::
    :header-rows: 1

    * - Directory
      - Shear catalog
      - Description
    * - ``metadetect``
      - Metadetect
      - The main example (CI). A 1 square degree simulated metadetect catalog; random forest tomography. See :ref:`Running an example pipeline`.
    * - ``metadetect_source_only``
      - Metadetect
      - (CI) Only the source-sample part of the analysis, run with the dask debug mode.
    * - ``scalar_metadetect``
      - Scalar metadetect
      - (CI) As ``metadetect``, using a scalar metadetect catalog.
    * - ``metacal``
      - Metacal
      - (CI) The same simulated data, in the metacal format.
    * - ``redmagic``
      - Metacal
      - (CI) Uses a redMaGiC lens sample.
    * - ``lensfit``
      - Lensfit
      - (CI) A mock KiDS-style lensfit catalog.
    * - ``mock_shear``
      - Simple
      - (CI) Converts a text-file mock shear catalog and runs selection and binning on it.
    * - ``cosmodc2``
      - Metacal, metadetect
      - Larger CosmoDC2 simulations (1 and 20 square degrees), with variants for different computing sites, a redMaGiC lens sample, and cluster pipelines in ``Cluster_pipelines``.
    * - ``2.2i``, ``skysim``, ``buzzard``
      - Metacal (and other)
      - DC2 Run 2.2i, SkySim5000 and Buzzard simulations.
    * - ``gaussian_sims``, ``lognormal``
      - Metadetect (and other)
      - Gaussian and log-normal mock catalogs.
    * - ``desy1``, ``desy3``
      - Metacal, DES Y3
      - Reanalysis of DES Y1 and Y3 data.
    * - ``kids-1000``
      - Lensfit
      - Reanalysis of KiDS-1000.
    * - ``hscy1``, ``hscy3``
      - HSC
      - Reanalysis of HSC Y1 and Y3, one pipeline per field.
    * - ``dp0.2``, ``dp1``
      - HSC, simple
      - Rubin DC2-like data (DP0.2) and Data Preview 1. The DP1 pipeline is a template, run with ``-t field=ecdfs`` (or ``edfs``, ``lglf``).
    * - ``dp2``
      - Metadetect, scalar metadetect
      - Rubin Data Preview 2 and DESC-processed versions of it. See :ref:`Rubin DP2 data`.
    * - ``desi_mocks``, ``desi_dr1_stage3``
      - Simple
      - DESI mock catalogs and DR1 data.
    * - ``ext_cross_corr``, ``cmb``
      - Metadetect
      - Cross-correlation with external catalogs, and CMB lensing cross-correlations with Quaia.
    * - ``ssi``
      - 
      - Synthetic source injection analyses.
    * - ``randoms``, ``lssweights``, ``clustering_test``, ``twopoint_cluster``, ``star-challenge``
      - Various
      - Smaller pipelines for specific tests: random catalogs, large-scale-structure weights, clustering, cluster two-point functions and star catalogs.

.. note::

    The pipelines in ``examples`` that are not marked "CI" are not run automatically, and may need updating to work with the latest TXPipe. In particular the ``desy3`` pipeline uses ``TXComputeShearCalibrationDESY3``, which is currently disabled.

Pipelines that use the source selection stages choose a selection stage, a tomography stage and a calibration stage that match their catalog type; see :ref:`Source selection and tomography`.
