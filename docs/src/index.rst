.. TXPipe documentation master file, created by
   sphinx-quickstart on Wed Mar 21 16:45:49 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

TXPipe documentation
====================


TXPipe is the catalog-to-statistics pipeline for the LSST Dark Energy Science Collaboration.

It is designed to take catalogs of objects (as produced by the LSST Project), and generate measurements of two-point and other statistics from them, together with all the covariances, binning, and quality control needed to use these statistics in cosmological analyses.

.. toctree::
   :maxdepth: 1
   :caption: Getting started:

   installation
   example
   structure
   file_types
   files
   shear_calibration

.. toctree::
   :maxdepth: 1
   :caption: Using & Contributing:

   running
   adding
   parallel
   nersc
   lsst



.. toctree::
   :maxdepth: 1
   :caption: Reference:

   hdf5
   stages
   datatypes
   utils



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
