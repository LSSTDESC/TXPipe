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

Then the data files will be stored in `data/examples/inputs`.

Mini Testing Data Set
---------------------

A 20 square degree data set for slightly more interesting tests is available on NERSC either to copy, or via the `DESC data registry <https://lsstdesc.org/dataregistry/index.html>`_.

You can access them directly in: `/global/cfs/cdirs/lsst/utilities/data-registry/lsst_desc_working/user/zuntz` or see how to access them using the registry by looking in the file `examples/cosmodc2/pipeline-20deg2-nersc.yml`. In that file the registry is searched by name for a dataset using this syntax:


.. code-block:: yml
    shear_catalog:
        name: txpipe_cosmodc2_20deg2_shear_catalog.hdf5



Forthcoming Data Sets
---------------------

We will add files for data preview 1 and larger CosmoDC2 data to the registry and will update these docs as they become available.


Ingestion stages
----------------

There are various stages designed to convert data into TXPipe formats. See :ref:`Ingestion` for a listing.

