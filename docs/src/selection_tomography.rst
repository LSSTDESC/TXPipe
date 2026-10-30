Source selection and tomography
===============================

Preparing a shear catalog for cosmology happens in three steps, each done by a different kind of pipeline stage:

1. **Selection** decides which objects are good enough to be in the source sample at all (signal-to-noise, size, flag cuts, and so on). It is independent of redshift.
2. **Tomography** assigns each object to a tomographic redshift bin, using a classifier, a simple cut on true redshift, or an external photo-z tool such as a RAIL SOM.
3. **Calibration** combines the selection and the bin assignments, finds the objects that are selected *and* assigned to each bin, and computes the shear calibration for each bin.

Splitting these steps apart lets you change one without redoing the others: you can retrain the tomography without repeating the cuts, or try a new cut without retraining. It also means that tomographic binning can come from outside TXPipe: any tool that can produce a :ref:`bin assignments file<Tomographic bin assignments>` can be used.

.. code-block:: text

    shear_catalog ──► Selection ──► shear_selection ─────────────┐
         │                                                       ▼
         └──────────► Tomography ──► shear_bin_assignments ──► Calibration ──► shear_tomography_catalog
                                                                   ▲
                                          shear_catalog ───────────┘

The output, the :ref:`shear tomography catalog<Shear tomography catalogs>`, is what later stages such as ``TXShearCalibration`` (which makes the calibrated binned catalogs) and ``TXSourceMaps`` use.

.. note::

    These three stages replace the old ``TXSourceSelector*`` and ``TXSourceTomography`` stages, which did all of the work in one go. Pipelines that used them need updating; the table at the bottom of this page shows the equivalents.


Which stages do I need?
-----------------------

The selection and calibration stages you need depend on the type of shear catalog (see :ref:`File Types`). The tomography stage does not depend on the catalog type, but you do have to choose one of the methods below.

.. list-table::
    :header-rows: 1

    * - Catalog type
      - Selection stage
      - Calibration stage
    * - Metadetect
      - :py:class:`~txpipe.source_selection.TXMetaDetectSelection`
      - :py:class:`~txpipe.shear_calibration.metadetect.TXComputeShearCalibrationMetadetect`
    * - Scalar metadetect
      - :py:class:`~txpipe.source_selection.TXScalarMetaDetectSelection`
      - :py:class:`~txpipe.shear_calibration.scalar_metadetect.TXComputeShearCalibrationScalarMetadetect`
    * - Metacal
      - :py:class:`~txpipe.source_selection.TXMetacalSelection`
      - :py:class:`~txpipe.shear_calibration.metacal.TXComputeShearCalibrationMetacal`
    * - Lensfit
      - :py:class:`~txpipe.source_selection.TXLensfitSelection`
      - :py:class:`~txpipe.shear_calibration.lensfit.TXComputeShearCalibrationLensfit`
    * - HSC
      - :py:class:`~txpipe.source_selection.TXHSCSelection`
      - :py:class:`~txpipe.shear_calibration.hsc.TXComputeShearCalibrationHSC`
    * - Simple (no calibration needed)
      - :py:class:`~txpipe.source_selection.TXSimpleSelection`
      - :py:class:`~txpipe.shear_calibration.simple.TXComputeShearCalibrationSimple`

The base class ``TXComputeShearCalibration`` cannot be used directly; use one of the subclasses.


Step 1: selection
-----------------

:py:class:`~txpipe.source_selection.TXSimpleSelection` applies a flag cut, a minimum signal-to-noise cut ``s2n_min`` and a minimum size cut ``T_min`` on the ratio of the object size to the PSF size. The subclasses for each catalog type use the same cuts, applied to each variant of the catalog (see below).

The result is a :ref:`catalog selection file<Catalog selections>`, ``shear_selection``, with one boolean value per object, per variant.

For metacal and metadetect catalogs the selection is made on each sheared variant separately, because the effect of the selection on the calibration (the *selection response*) is found by comparing the results from different variants.

Survey-specific cuts
^^^^^^^^^^^^^^^^^^^^

Some surveys apply more cuts than the simple ones. These are implemented as subclasses that you choose in the pipeline instead of the generic selection stage:

- :py:class:`~txpipe.source_selection.TXDP2RubinMetadetectSelection` applies the cuts for the Rubin metadetect catalogs of DP2: primary-object and mask-fraction cuts, signal-to-noise and size cuts, magnitude cuts in g, r and i, and colour sanity cuts. See ``examples/dp2/analysis_rubin_metadetect.yml``.
- :py:class:`~txpipe.source_selection.TXDP2DESCSelection` applies a similar set for the DESC-run scalar metadetect catalogs of DP2, including a PSF-ellipticity cut. See ``examples/dp2/analysis_desc_metadetect_1.1.yml``.

All the cuts can be changed in the configuration file; see the stage documentation in :ref:`Selection` for the options and their defaults.

Writing your own selection
^^^^^^^^^^^^^^^^^^^^^^^^^^

To make a new selection, subclass one of the selection stages and override ``compute_selection``, which is given a chunk of data (a dictionary of arrays) and returns a boolean array. Use the helper class ``Cutter`` to keep a running record of how many objects each cut removes. Set the class attribute ``columns`` to list the catalog columns that the cuts need. :py:class:`~txpipe.source_selection.TXDP2DESCSelection` is a good example to follow.


Step 2: tomography
------------------

There are currently four ways to get tomographic bins.

**Random forest.** A random-forest classifier is trained to predict redshift bins from magnitudes and colours using a spectroscopic or other training catalog (:py:class:`~txpipe.tomography.TXTrainRandomForestTomography`), and applied to the shear catalog (:py:class:`~txpipe.tomography.TXApplyRandomForestSourceTomography`). Training is a separate stage because it can be slow with a large training sample. This is what the ``examples/metadetect``, ``examples/metacal`` and ``examples/cosmodc2`` pipelines do. The training stage needs the bin edges ``source_zbin_edges`` and the bands to use in its configuration.

**Simple redshift cuts.** If the catalog contains true (or other estimated) redshifts, :py:class:`~txpipe.tomography.TXSimpleRedshiftSourceTomography` bins on the column ``redshift_col``. This is used for tests and for mocks, for example in ``examples/desy1`` and ``examples/kids-1000``.

**RAIL** A RAIL tomography algorithm such as a SOM can be run on each variant of the catalog, and the results gathered into a single bin-assignments file by :py:class:`~txpipe.rail.summarize.PZCollateRailAssignmentsMetadetect` (or ``PZCollateRailAssignmentsScalarMetadetect`` for scalar metadetect catalogs). See ``examples/dp2/analysis_rubin_metadetect.yml`` for a complete pipeline, and :py:class:`~txpipe.rail.summarize.PZGenerateSOMInformerSubsample` for making a training sample from the selected objects.

**External** You could simply ingest an existing tomography file, for example from an external tool. There are no examples of this yet but it should be straightforward.

For metacal and metadetect catalogs, bins have to be assigned to every sheared variant so that the selection response can be computed. For a classifier that is done automatically; with an external tool you must run it on each variant.

Whatever the method, the result is a :ref:`bin assignments file<Tomographic bin assignments>`, ``shear_bin_assignments``, which includes the number of bins and the bin edges.


Step 3: calibration
-------------------

The ``TXComputeShearCalibration*`` stages read the shear catalog, the selection and the bin assignments, work out for each tomographic bin which objects are both selected and in that bin, and compute the calibration quantities (see :ref:`Shear calibration guide`). They write the :ref:`shear tomography catalog<Shear tomography catalogs>`. The configuration options differ between catalog types:

- Metacal, metadetect and scalar metadetect need the ``delta_gamma`` shear step used to make the variants. Metacal also has ``use_diagonal_response``.
- Lensfit needs ``input_m_is_weighted`` (whether the ``m`` values in the catalog are already weighted) and has an optional ``dec_cut``.

The calibration stages combine the cuts, so that the sample used for calibration is exactly the one used for science; if you change the selection or the tomography you must re-run the calibration stage.


Examples of the stage chain
---------------------------

A metadetect pipeline with random-forest tomography (``examples/metadetect/pipeline.yml``):

.. code-block:: yaml

    stages:
      - name: TXMetaDetectSelection
      - name: TXTrainRandomForestTomography
      - name: TXApplyRandomForestSourceTomography
      - name: TXComputeShearCalibrationMetadetect
      - name: TXShearCalibration




Converting old pipelines
------------------------

.. list-table::
    :header-rows: 1

    * - Old stage
      - Replace with
    * - ``TXSourceSelectorMetadetect``
      - ``TXMetaDetectSelection``, a tomography stage, and ``TXComputeShearCalibrationMetadetect``
    * - ``TXSourceSelectorMetacal``
      - ``TXMetacalSelection``, a tomography stage, and ``TXComputeShearCalibrationMetacal``
    * - ``TXSourceSelectorLensfit``
      - ``TXLensfitSelection``, a tomography stage, and ``TXComputeShearCalibrationLensfit``
    * - ``TXSourceSelectorHSC``
      - ``TXHSCSelection``, a tomography stage, and ``TXComputeShearCalibrationHSC``
    * - ``TXSourceSelectorSimple``
      - ``TXSimpleSelection``, a tomography stage, and ``TXComputeShearCalibrationSimple``
    * - ``TXSourceTomography``
      - ``TXTrainRandomForestTomography`` and ``TXApplyRandomForestSourceTomography``
    * - ``bin_ns`` in a metadetect tomography catalog
      - ``ns/bin`` (and likewise ``bin_1p`` becomes ``1p/bin``, and so on)

In the configuration file, the signal-to-noise and size cut options are now ``s2n_min`` and ``T_min`` and belong to the selection stage. Options that control the random forest (bands, bin edges) belong to the tomography stages, and ``delta_gamma`` belongs to the calibration stage.
