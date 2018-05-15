TXPipe Installation
==========

Requirements
------------

TXPipe requires python 3.6 or above.

The pipeline software ceci that TXPipe uses can be installed with:
``
    pip install ceci
``

Various stages of TXPipe also require:

``
    numpy
    scipy
    fitsio
    healpy
    astropy
``

which can all also be installed with pip.

To run the example you need some input data files.  You can obtain these from 
NERSC if you have access to the LSST space using:

`` 
    scp USERNAME@cori.nersc.gov:/global/projecta/projectdirs/lsst/groups/WL/users/zuntz/tract008766-shear-cat-0-10000.fits ./
``
