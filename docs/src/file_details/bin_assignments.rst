Tomographic bin assignments
===========================

A bin assignment file records the tomographic bin that an external method (a random forest classifier, a simple cut on true redshift, or a RAIL photo-z or SOM algorithm) assigns to each object of a shear catalog, *before* the source selection is applied. It is the output of the tomography stages (:py:class:`~txpipe.tomography.TXSimpleRedshiftSourceTomography`, :py:class:`~txpipe.tomography.TXApplyRandomForestSourceTomography`, and :py:class:`~txpipe.rail.summarize.PZCollateRailAssignmentsMetadetect` and its scalar version). In pipelines it is called ``shear_bin_assignments``, and it is read with the ``BinAssignments`` class. See :ref:`Source selection and tomography`.

The bin assignments are kept separate from the selection so that either can be changed without repeating the other, and so that tomography from different sources can be combined with any selection. The two are brought together by the ``TXComputeShearCalibration*`` stages, which write the :ref:`Shear tomography catalogs`.

The file has attributes ``nbin_source`` (the number of tomographic bins) and ``bin_edges`` (the redshift bin edges). The simple-redshift stage also records ``redshift_col``, and the random forest stage records ``bands``.

The datasets have one integer entry per object, with -1 meaning that the object was not assigned to a bin. Their names depend on the shear catalog type:

- **Simple, lensfit and HSC catalogs:** a single dataset ``bin``.
- **Metacal catalogs:** datasets ``bin``, ``bin_1p``, ``bin_1m``, ``bin_2p`` and ``bin_2m``, for the unsheared and sheared variants.
- **Metadetect catalogs:** a dataset ``bin`` in a group for each variant, so ``ns/bin``, ``1p/bin``, ``1m/bin``, ``2p/bin`` and ``2m/bin``.
- **Scalar metadetect catalogs:** the same, for the variants ``ns``, ``1p`` and ``1m`` only.

All are 16-bit integers.

The metacal and metadetect sheared-variant assignments are needed to compute the selection response. For metacal only the unsheared assignment ends up in the final tomography catalog.
