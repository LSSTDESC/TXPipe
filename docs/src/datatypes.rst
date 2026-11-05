Data Type Classes
=================

TXPipe stage input and output files are all represented by a class which defines their type and various other pieces of information.

Generic data types, such as HDF and Text file types, are inherited from ceci. TXPipe defines more specific types, like :py:class:`~txpipe.data_types.types.ShearCatalog`.

Shear catalogs come in several flavours, each with its own subclass of ``ShearCatalog``: ``MetacalShearCatalog``, ``MetaDetectShearCatalog``, ``ScalarMetaDetectCatalog``, ``LensfitShearCatalog``, ``HSCShearCatalog`` and ``SimpleShearCatalog``. When you open a ``ShearCatalog`` it works out which type of file it is (from the ``catalog_type`` attribute, or from the columns it contains) and gives you an object of the right subclass. The subclasses know the names of the columns and groups that their format uses, so that stages can be written generically; for example ``get_tomography_bin_columns`` and ``get_extra_calibration_columns``. ``class_for_shear_catalog_type`` returns the class for a catalog type name such as ``"metadetect"``. The file formats are described in :ref:`File Types`.

Two further types connect the source-selection stages: ``CatalogSelection`` holds boolean cuts and ``BinAssignments`` holds tomographic bin assignments. See :ref:`Source selection and tomography`.

The available types are described below.



.. automodule:: txpipe.data_types
   :members:
   :undoc-members:
   :show-inheritance: