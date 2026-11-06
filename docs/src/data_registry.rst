Using the DESC data registry in pipelines
=========================================

The `DESC data registry <https://lsstdesc.org/dataregistry/index.html>`_ is a database that keeps track of the data sets that DESC members share at NERSC: where each one is, who owns it, and what version it is. A TXPipe pipeline can use it to find its input files, so that the pipeline file refers to a data set by its registry name or ID instead of by a path that may change.

The lookup is done by ceci, the workflow engine that runs TXPipe pipelines, not by TXPipe itself, so it works for any TXPipe stage. This page describes how to use it in a pipeline file.

.. important::

    The data registry only works on machines where it has been set up, which in practice means NERSC. It reads a database and a shared file system that are not available elsewhere, and the ``dataregistry`` package is not part of the standard local installation.

    On a laptop or another cluster, use explicit file paths in the pipeline file instead (see below), or copy the files you need from NERSC and point the pipeline at the copies.



Pipeline file inputs
--------------------

The ``inputs`` section of a :ref:`pipeline file<Pipeline files>` says where to find the files that are not made by an earlier stage in the pipeline. Each entry maps a file tag to a location. There are two kinds of location.

An **explicit path** is a string:

.. code-block:: yaml

    inputs:
        fiducial_cosmology: data/fiducial_cosmology.yml

A **registry lookup** is a dictionary with a single key, either ``id`` or ``name``:

.. code-block:: yaml

    inputs:
        # By dataset ID in the registry
        shear_catalog:
            id: 147
        # By dataset name
        photometry_catalog:
            name: txpipe_cosmodc2_20deg2_photometry_catalog.hdf5

You can mix the two kinds in the same pipeline, as the examples do. When ceci starts the pipeline it replaces each lookup with the absolute path of the data set at NERSC, and then runs exactly as if you had written the path yourself.

.. list-table::
    :header-rows: 1

    * - Key
      - Meaning
    * - ``id``
      - The ``dataset_id`` of the entry in the registry. This is a number that identifies exactly one data set, so it is the most precise choice.
    * - ``name``
      - The ``name`` of the data set. This must match exactly one data set; if there is more than one version of a data set with this name, the lookup stops with an error saying it found multiple data sets, and you should use the ``id`` instead.
    * - ``alias``
      - Not implemented yet; ceci raises an error if you use it.

Only data sets that are still active are found: ones that have been deleted or replaced by a newer version are ignored. Lookups search the default (working) part of the registry.

Some TXPipe pipelines list inputs that they do not use, with the value ``none``. These are just strings, not registry lookups; they are placeholders that let ceci start the pipeline.


Enabling the registry
---------------------

You must also add a ``registry`` section to the pipeline file, at the top level, to tell ceci to connect to the registry. Without it, a pipeline with registry lookups stops with the error "No data registry configured".

Usually the default configuration is all that is needed, and you can write an empty dictionary:

.. code-block:: yaml

    registry: {}

The registry section accepts these options, all of which are optional:

.. list-table::
    :header-rows: 1

    * - Option
      - Meaning
    * - ``config``
      - Path to the file with your credentials for the registry database. If you leave it out, the data registry uses the default location in your home directory.
    * - ``root_dir``
      - The directory that registry data sets are stored in. If you leave it out, the registry uses the default location at NERSC.
    * - ``owner_type``
      - One of ``production``, ``project``, ``group`` or ``user`` (the default). This is the default owner type that the registry connection uses.
    * - ``owner``
      - The default owner name for the registry connection.

The ``owner`` and ``owner_type`` options describe the connection to the registry. They are not used to filter lookups, which match only on the ``id`` or ``name``. The DESI example pipeline, for instance, sets ``owner_type: project`` and ``owner: wlss-dp2-desi-delta-sigma``.

If you do not use any registry lookups, leave the ``registry`` section out (or comment it out, as some examples do), so that ceci does not try to connect to the registry.


Finding data sets
-----------------

The ``dregs`` command-line tool, which comes with the ``dataregistry`` package, lists and finds data sets. At NERSC:

.. code-block:: bash

    dregs ls                      # data sets that you own
    dregs ls --owner none         # every data set in the registry
    dregs ls --keyword metadetect # search by keyword
    dregs path 147                # print the path of the dataset with this ID

Run ``dregs --help`` or see the `data registry documentation <https://lsstdesc.org/dataregistry/index.html>`_ for the full list of options, and for how to register your own data sets.

Registry ``id`` values are shared by everyone, so a pipeline with an ``id`` will run for any other DESC member. Check the ID with ``dregs path`` before you rely on it. The examples show a full set of inputs: for example ``examples/dp2/analysis_desc_metadetect_1.1.yml`` looks up its mask, shear catalog and spectroscopic data by ``id``, and ``examples/cosmodc2/pipeline-20deg2-nersc.yml`` looks up its catalogs by ``name``.


Dry runs and flow charts
------------------------

Because ceci resolves the inputs when it creates the pipeline, ``ceci --dry-run`` also connects to the registry, so it only works where the registry works.

Making a flow chart with ``ceci --flow-chart`` does not read any data. From ceci version 2.7 onwards it skips the registry and shows registry inputs as ``registry:`` followed by the lookup. Older versions, including the 2.6 that the TXPipe environment files currently install, connect to the registry for flow charts too, and so fail on machines that do not have it.


Troubleshooting
---------------

.. list-table::
    :header-rows: 1

    * - Error
      - Likely cause
    * - ``No module named 'dataregistry'``
      - The ``dataregistry`` package is not installed in your environment. This is the case on most machines other than NERSC.
    * - ``No data registry configured``
      - An input has an ``id`` or ``name`` lookup but the pipeline file has no ``registry`` section.
    * - ``Could not find any dataset matching ...``
      - No active data set has that name. Check the spelling, and that the data set has not been deleted or replaced.
    * - ``Found multiple datasets matching ...``
      - Several versions of the data set share that name. Use the ``id`` of the version you want instead.
    * - ``Must specify either id or name in registry lookup``
      - A lookup dictionary has neither key.

If the registry cannot connect to its database, check that you have the credentials file it expects at NERSC (or set the ``config`` option), and see the data registry documentation for setup instructions.
