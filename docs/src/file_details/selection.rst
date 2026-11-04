Catalog selections
==================

A catalog selection file records which objects in a shear catalog pass the cuts that define the source sample, before any tomographic binning is applied. It is the output of the selection stages (:py:class:`~txpipe.source_selection.TXSimpleSelection` and its subclasses), is called ``shear_selection`` in pipelines, and is read with the ``CatalogSelection`` class. See :ref:`Source selection and tomography`.

The file contains boolean datasets with one entry per object in the corresponding group of the shear catalog. The layout follows that of the shear catalog:

- For simple, lensfit and HSC catalogs there is a group "shear" with one dataset, "selection".
- For metadetect and scalar metadetect catalogs there is one group for each variant, "shear/ns", "shear/1p", and so on, each with its own "selection" dataset, because the variants contain different objects and have different lengths.
- For metacal catalogs there is a group "shear" with a dataset "selection" for the unsheared catalog and "selection_1p", "selection_1m", "selection_2p", "selection_2m" for the sheared variants, because the selection must be repeated on each variant to compute the selection response.

Each group has an attribute ``n_selected``: the total number of objects with a selection value of True.

==========================  =============  ========  =========
Group                       Name           Kind      Meaning
==========================  =============  ========  =========
shear                       selection      1D bool   Simple, lensfit, HSC; metacal unsheared
shear                       selection_1p   1D bool   Metacal only; also 1m, 2p, 2m
shear/{variant}             selection      1D bool   Metadetect and scalar metadetect
==========================  =============  ========  =========
