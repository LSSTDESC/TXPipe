NERSC Shifter Installation
==========================

The easiest way to use TXPipe on NERSC is via the Shifter system, which creates a container with all the requirements pre-built.

To use this, first run::

    source /global/cfs/cdirs/lsst/groups/WL/users/zuntz/setup-txpipe

This will create a command ``tx``. You prefix any command with this to run it inside the container.

For example::

    tx ceci examples/laptop_pipeline.yml

Will run the pipeline inside the container.  You could run an individual stage instead by doing::

    tx python -m txpipe NameOfStage -h

If you need to update or add to the software dependencies in the image, please make a PR on `<github.com/joezuntz/txpipe-docker/>`_ modifying the file  ``txpipe-conda/Dockerfile``.

