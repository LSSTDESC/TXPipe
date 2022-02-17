Running TXPipe at NERSC
=======================

On NERSC, follow the setup command described in  :ref:`NERSC Shifter Installation` to set up the TXPipe dependencies.

NERSC Interactive Sessions
--------------------------

You should then only run small test commands on the login node, with only a small number of cores. To run larger test jobs you can start an interactive session. This example would give you one node for 1 hour:

.. code:: bash

    salloc --nodes 1 --qos interactive --time 01:00:00 --constraint haswell

Run the setup command again after starting your job. 

Running pipelines at NERSC
--------------------------

The ``tx`` command runs programs in a TXPipe environment. To run whole pipelines, you can do this:

.. code:: bash

    tx ceci examples/metacal/pipeline.yml


Running individual stages at NERSC
----------------------------------

You can also use the TX command to run individual stages, to test or develop them.

The easiest way to find the correct command to run is to run a pipeline with ceci (as above)
using the ``--dry-run`` flag to print out a list of commands.

They will normally have a form something like this at NERSC:

.. code:: bash

    srun -u --ntasks=4 --cpus-per-task=2 --nodes=1 tx python -m txpipe TXSourceSelector ...

