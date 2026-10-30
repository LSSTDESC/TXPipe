Data Sources
============

Before using TXPipe you need to have some input data files for it to process. These have to be a a specific format and structure. Typically this at least includes shear and photometry catalogs.

For standard data sets, DESC users can access these files at NERSC - see below.


Micro Testing Data Set
----------------------

As described in :ref:`Running an example pipeline`, you can access a simulated 1 square degree of data that is useful for checking that pipelines actually run. You can download and extract it like this:

.. code-block:: bash

    curl -O https://portal.nersc.gov/cfs/lsst/txpipe/data/example.tar.gz
    tar -zxvf example.tar.gz

Then the data files will be stored in `data/example/inputs`.

Mini Testing Data Set
---------------------

A 20 square degree data set for slightly more interesting tests is available on NERSC either to copy, or via the `DESC data registry <https://lsstdesc.org/dataregistry/index.html>`_.

You can access them directly in: `/global/cfs/cdirs/lsst/utilities/data-registry/lsst_desc_working/user/zuntz`.


Rubin Data Preview data
-----------------------

TXPipe has ingestion stages for Rubin Data Preview 1 (:py:class:`~txpipe.ingest.dp1.TXIngestDataPreview1`) and Data Preview 2, and the DESC-processed versions of DP2 metadetect catalogs. See :ref:`Rubin DP2 data`.

Other surveys
-------------

There are example pipelines, and ingestion stages for the input data, for DES Y1 and Y3 (``examples/desy1``, ``examples/desy3``), KiDS-1000 (``examples/kids-1000``), HSC Y1 and Y3 (``examples/hscy1``, ``examples/hscy3``), DESI (``examples/desi_dr1_stage3``, ``examples/desi_mocks``), and simulations. See :ref:`Example pipelines`.


Ingestion stages
----------------

There are various stages designed to convert data into TXPipe formats. See :ref:`Ingestion` for a listing.

