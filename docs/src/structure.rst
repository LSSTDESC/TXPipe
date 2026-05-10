Concept Overview
================

TXPipe is a collection of pipeline stages, which connect together and provide outputs to later stages.

You can launch these stages individually, specifying input and output files explicitly, by running on the command line. Or you can run a whole pipelines together.


Ceci
----

`Ceci <https://github.com/LSSTDESC/ceci>`_ is the framework that TXPipe uses to connect pipeline stages to workflow codes that know how to connect them together and run them.  Each TXPipe stage is implemented as a python class, and inherits from a ceci base class.

For information on adding new stages, see the section on :ref:`Adding new pipeline stages`.

You don't need to know anything about ceci to run TXPipe.


TXPipe Stages
-------------

Each TXPipe stage class has:

* A "name" string, which you can use on the command line or in configuration files to choose the stage.

* Lists ``inputs`` and ``outputs``. Each is a list of tuple pairs:

    #. The first item is a string "tag", which defines a particular input or output file.  This tag is used throughout TXPipe to show that the same file is e.g. an output from one stage or the input to some others. 

    #. The second item is a class which defines the type of the output file. There are general types (such as ``TextFile`` and ``HDFFile``), and more specific types, (like ``ShearCatalog`` and ``MapsFile``)


* A "config_options" dictionary of parameters. These can be types, like float or int, meaning that the parameter is required, or a specific value like ``1.0`` or ``urizy`` to indicate a default value.  Parameters are set by pipeline users when running.

* A "run" method doing the actual work of the class. This method can use the methods of the base class to load and save inputs and outputs.

You launch individual stages like this:

.. code-block:: bash

    python -m txpipe Name_of_stage <inputs, outputs, and options for stage>



TXPipe Pipelines
----------------

A run of a TXPipe pipeline is defined by two files in YML format:

* The pipeline file defines what pipeline stages are to be run, how they should be run, and where overall inputs and outputs should be.

* The configuration file chooses input parameters for the different stages.

See :ref:`Configuration Files` for more details.


