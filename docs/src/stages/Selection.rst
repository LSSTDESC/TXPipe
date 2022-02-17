Selection
=========

These stages deal with selection objects and assigning them to tomographic
bins.

* :py:class:`~txpipe.source_selector.TXSourceSelectorBase` - Base stage for source selection using S/N, size, and flag cuts and tomography

* :py:class:`~txpipe.source_selector.TXSourceSelectorMetacal` - Source selection and tomography for metacal catalogs

* :py:class:`~txpipe.source_selector.TXSourceSelectorMetadetect` - Source selection and tomography for metadetect catalogs

* :py:class:`~txpipe.source_selector.TXSourceSelectorLensfit` - Source selection and tomography for lensfit catalogs

* :py:class:`~txpipe.source_selector.TXSourceSelectorHSC` - Source selection and tomography for HSC catalogs

* :py:class:`~txpipe.lens_selector.TXBaseLensSelector` - Base class for lens object selection, using the BOSS Target Selection.

* :py:class:`~txpipe.lens_selector.TXTruthLensSelector` - Select lens objects based on true redshifts and BOSS criteria

* :py:class:`~txpipe.lens_selector.TXMeanLensSelector` - Select lens objects based on mean redshifts and BOSS criteria

* :py:class:`~txpipe.lens_selector.TXModeLensSelector` - Select lens objects based on best-fit redshifts and BOSS criteria



.. autoclass:: txpipe.source_selector.TXSourceSelectorBase
   :members:


.. autoclass:: txpipe.source_selector.TXSourceSelectorMetacal
   :members:


.. autoclass:: txpipe.source_selector.TXSourceSelectorMetadetect
   :members:


.. autoclass:: txpipe.source_selector.TXSourceSelectorLensfit
   :members:


.. autoclass:: txpipe.source_selector.TXSourceSelectorHSC
   :members:


.. autoclass:: txpipe.lens_selector.TXBaseLensSelector
   :members:


.. autoclass:: txpipe.lens_selector.TXTruthLensSelector
   :members:


.. autoclass:: txpipe.lens_selector.TXMeanLensSelector
   :members:


.. autoclass:: txpipe.lens_selector.TXModeLensSelector
   :members:

