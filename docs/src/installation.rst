TXPipe Installation
===================

TXPipe requires python 3.6 or above.  Get the TXPipe code like this:

.. code-block:: bash

    git clone --recurse-submodules https://github.com/LSSTDESC/TXPipe
    cd TXPipe


As the pipeline has many components it has many dependencies. These are listed in the file ``requirements.txt``, but getting an environment which can run the code fully parallel requires care.


- :ref:`Conda Installation` is the easiest approach on a personal machine
- :ref:`NERSC Shifter Installation` is the easiest approach on NERSC
- :ref:`CC-IN2P3 Installation` for the French CC-IN2P3 machine

The last is probably easiest to adapt to other clusters or supercomputer.


