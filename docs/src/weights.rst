LSS Weights Guide
=======================

Galaxy clustering measurements can be biased by correlations between the observed galaxy density and survey properties such as seeing, depth, etc. or astrophysical foreground such as dust or stars. These observational systematics introduce spatial variations in the lens sample that are unrelated to the underlying cosmological density field.

TXPipe provides stages for identifying and correcting these effects by measuring the correlation between galaxy density and survey property maps, and generating multiplicative weights that can be applied to lens galaxies during clustering analyses.

Alternatively, systematics contamination can be removed at the power spectrum level using :ref:`mode-projection <modeprojection>`.

The weighting stages are implemented in ``txpipe/lssweights.py``. They operate on pixelized survey property maps and lens density maps.

Density Null Test Class
------------------------

The first stage you will likely want to use is ``TXLSSDensityNullTests``. This measures the relationship between the observed lens galaxy density and each survey property map, binned by the survey property map value, and quantifies the strength and statistical significance of each correlation.

For each tomographic lens bin, the stage:

- loads and normalizes the survey property maps
- computes the normalized galaxy density in bins of the survey property, for each survey property map
- estimates the covariance matrix of these measurements
- compares the measured trends to the null hypothesis of constant galaxy density
- produces summary plots and diagnostic statistics

The output consists of the measured density correlations together with covariance matrices (all bundled into the hdf5 output ``unweighted_density_correlation``), plus a collection of diagnostic plots for each survey property.

Covariance Estimation
^^^^^^^^^^^^^^^^^^^^^

The covariance of the binned density correlations is computed for the full density-correlation data vector, including both the covariance between survey property bins within a single survey property map and the cross-covariance between different survey property maps. For example, using five survey property maps with ten bins per map produces a single ``50 × 50`` covariance matrix.

The covariance can be computed at several levels of complexity.

By default, the covariance includes:

- diagonal shot-noise terms arising from the finite number of galaxies
- off-diagonal shot-noise terms describing correlations between survey property bins in different survey property maps.
- a sample variance contribution computed from a theoretical angular correlation function

The theoretical sample variance term is calculated using the fiducial cosmology and the lens redshift distribution. We use the following expression (cite Peebles):

.. math::
    \mathrm{Cov}(N_i, N_j)
    =
    \sum^{\theta^{\rm max}}_{\theta=\theta_{\rm min}} N^{\rm (pix \ pairs)}_{ij}(\theta) {\bar N}^2 w_{\rm true}(\theta)

where :math:`N^{\rm (pix \ pairs)}_{ij}`` is the number of pixel-pixel pairs between the two sp bins *i* and *j* in the given angular range measured with ``TreeCorr``, :math:`{\bar N}` is the average number of galaxies per pixel, and :math:`w_{\rm true}`` is the theoretical angular correlation function computed using ``pyccl``.

For faster testing, the configuration option ``simple_cov=True`` restricts the covariance to the diagonal shot-noise contribution only.

Weights Classes
------------------

The weights stages each produce a weight map per tomographic bin, the corresponding catalog weights, plus some summary plots. These classes usually take the binned density null tests as an input (but it is not strictly a requirement).

Currently, the primary LSS weights stages are:

- ``TXLSSWeightsLinBinned``: Fits a linear model to the binned trends computed in ``TXLSSDensityNullTests``. Map selection/regularization is performed by only correcting for the survey property maps with a *p*-value below a given threshold.
- ``TXLSSWeightsLinPix``: Fits a multivariate linear model directly to the pixelized data (galaxy density vs N survey property maps). This allows all selected systematics to be fit simultaneously while assuming a linear dependence of galaxy density on the survey properties. The covariance between pixels will not be considered in the fit. Any ``sklearn.linear_model`` class that uses the same arguments as ``sklearn.linear_model.LinearRegression`` can be used by this stage. Map selection/regularization can also be included using the *p*-values computed in ``TXLSSDensityNullTests``.
- ``TXLSSWeightsUnit``: Produces a trivial weight map with unit weight everywhere inside the valid footprint. This stage is useful when no systematic correction is required while maintaining compatibility with downstream pipeline stages.


Survey Property Maps
--------------------

The survey property maps are currently passed to all weight related stages in the config file, rather than as a formal TXPipe input.

All survey property maps in the specified directory will be used in the pipeline. If you want to test multiple different combinations of map inputs, we recommend creating multiple directories containing symlinks to the desired files.

Supporting Utilities
--------------------

Several utility functions used by the weighting stages are implemented in ``txpipe/lsstools/lsstools.py``.

.. _modeprojection:

Note on Mode-projection
-----------------------

LSS imaging systematics can also be mitigated using mode-projection directly on the measured power spectrum. See ``TXTwoPointFourier`` or ``TXTwoPointFourierCatalog`` for the TXPipe stages that implement this. If you are using mode projection method you may need to include the ``TXLSSWeightsUnit`` stage in the pipeline first. This will assign a weight of 1 to each galaxy.
