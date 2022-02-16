TXPipe File Tags and Types
==========================

This page is under construction.

Metadata
--------

Catalogs
--------

.. list-table:: Catalog files
   :widths: 10 10 80
   :header-rows: 1

   * - Tag
     - Class
     - Description
   * - ``shear_catalog``
     - ``ShearCatalog`` (hdf5)
     - Contains shear catalog information, in one of several layouts depending on the catalog calibration scheme. For real data it is generated outside the pipeline, but the `input_cats.py` classes can make mock catalogs. The core shear information is always in the ``shear`` HDF5 group in the file. 
   * - ``photometry_catalog``
     - ``HDFFile``
     - Contains photometry information for the general catalog used to select lens samples, among other things. 

Binning
-------

Maps & Masks
------------

Photo-Z
-------


