Selection
=========

These stages deal with selection objects and assigning them to tomographic
bins.

* :py:class:`~txpipe.source_selector.TXSourceSelectorBase` - Base stage for source selection using S/N, size, and flag cuts and tomography

* :py:class:`~txpipe.source_selector.TXSourceSelectorMetacal` - Source selection and tomography for metacal catalogs

* :py:class:`~txpipe.source_selector.TXSourceSelectorMetadetect` - Source selection and tomography for metadetect catalogs

* :py:class:`~txpipe.source_selector.TXSourceSelectorLensfit` - Source selection and tomography for lensfit catalogs

* :py:class:`~txpipe.source_selector.TXSourceSelectorSimple` - Source selection and tomography for mock catalogs that do not

* :py:class:`~txpipe.source_selector.TXSourceSelectorHSC` - Source selection and tomography for HSC catalogs

* :py:class:`~txpipe.lens_selector.TXBaseLensSelector` - Base class for lens object selection, using the BOSS Target Selection.

* :py:class:`~txpipe.lens_selector.TXTruthLensSelector` - Select lens objects based on true redshifts and BOSS criteria

* :py:class:`~txpipe.lens_selector.TXMeanLensSelector` - Select lens objects based on mean redshifts and BOSS criteria

* :py:class:`~txpipe.lens_selector.TXModeLensSelector` - Select lens objects based on best-fit redshifts and BOSS criteria

* :py:class:`~txpipe.lens_selector.TXRandomForestLensSelector` - Stage TXRandomForestLensSelector



.. autoclass:: txpipe.source_selector.TXSourceSelectorBase

    **parallel**: Yes - MPI

.. autoclass:: txpipe.source_selector.TXSourceSelectorMetacal

    **parallel**: Yes - MPI

.. autoclass:: txpipe.source_selector.TXSourceSelectorMetadetect

    **parallel**: Yes - MPI

.. autoclass:: txpipe.source_selector.TXSourceSelectorLensfit

    **parallel**: Yes - MPI

.. autoclass:: txpipe.source_selector.TXSourceSelectorSimple

    **parallel**: Yes - MPI

.. autoclass:: txpipe.source_selector.TXSourceSelectorHSC

    **parallel**: Yes - MPI

.. autoclass:: txpipe.lens_selector.TXBaseLensSelector

    **parallel**: Yes - MPI

.. autoclass:: txpipe.lens_selector.TXTruthLensSelector

    **parallel**: Yes - MPI

.. autoclass:: txpipe.lens_selector.TXMeanLensSelector

    **parallel**: Yes - MPI

.. autoclass:: txpipe.lens_selector.TXModeLensSelector

    **parallel**: Yes - MPI

.. autoclass:: txpipe.lens_selector.TXRandomForestLensSelector

    **parallel**: Yes - MPI
