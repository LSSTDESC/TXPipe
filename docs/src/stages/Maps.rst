Maps
====

These stages deal with making different kinds of maps for analysis and
plotting.

* :py:class:`~txpipe.maps.TXBaseMaps` - A base class for mapping stages

* :py:class:`~txpipe.maps.TXSourceMaps` - Generate source maps directly from binned, calibrated shear catalogs.

* :py:class:`~txpipe.maps.TXLensMaps` - Make tomographic lens number count maps

* :py:class:`~txpipe.maps.TXExternalLensMaps` - Make tomographic lens number count maps from external data

* :py:class:`~txpipe.maps.TXDensityMaps` - Convert galaxy count maps to overdensity delta maps

* :py:class:`~txpipe.noise_maps.TXSourceNoiseMaps` - Generate realizations of shear noise maps with random rotations

* :py:class:`~txpipe.noise_maps.TXLensNoiseMaps` - Generate lens density noise realizations using random splits

* :py:class:`~txpipe.noise_maps.TXExternalLensNoiseMaps` - Generate lens density noise realizations using random splits of an external catalog

* :py:class:`~txpipe.noise_maps.TXNoiseMapsJax` - Generate noise realisations of lens and source maps using JAX

* :py:class:`~txpipe.auxiliary_maps.TXAuxiliarySourceMaps` - Stage TXAuxiliarySourceMaps

* :py:class:`~txpipe.auxiliary_maps.TXAuxiliaryLensMaps` - Generate auxiliary maps from the lens catalog

* :py:class:`~txpipe.auxiliary_maps.TXUniformDepthMap` - Generate a uniform depth map from the mask

* :py:class:`~txpipe.masks.TXSimpleMask` - Make a simple binary mask using a depth cut and bright object cut

* :py:class:`~txpipe.masks.TXSimpleMaskSource` - Stage TXSimpleMaskSource

* :py:class:`~txpipe.masks.TXSimpleMaskFrac` - Make a simple mask using a depth cut and bright object cut

* :py:class:`~txpipe.convergence.TXConvergenceMaps` - Make a convergence map from a source map using Kaiser-Squires

* :py:class:`~txpipe.map_correlations.TXMapCorrelations` - Plot shear, density, and convergence correlations with survey property maps



.. autoclass:: txpipe.maps.TXBaseMaps

    **parallel**: Yes - MPI

.. autoclass:: txpipe.maps.TXSourceMaps

    **parallel**: Yes - Dask

.. autoclass:: txpipe.maps.TXLensMaps

    **parallel**: Yes - Dask

.. autoclass:: txpipe.maps.TXExternalLensMaps

    **parallel**: Yes - Dask

.. autoclass:: txpipe.maps.TXDensityMaps

    **parallel**: No - Serial

.. autoclass:: txpipe.noise_maps.TXSourceNoiseMaps

    **parallel**: Yes - MPI

.. autoclass:: txpipe.noise_maps.TXLensNoiseMaps

    **parallel**: Yes - MPI

.. autoclass:: txpipe.noise_maps.TXExternalLensNoiseMaps

    **parallel**: Yes - MPI

.. autoclass:: txpipe.noise_maps.TXNoiseMapsJax

    **parallel**: Yes - MPI

.. autoclass:: txpipe.auxiliary_maps.TXAuxiliarySourceMaps

    **parallel**: Yes - Dask

.. autoclass:: txpipe.auxiliary_maps.TXAuxiliaryLensMaps

    **parallel**: Yes - Dask

.. autoclass:: txpipe.auxiliary_maps.TXUniformDepthMap

    **parallel**: No - Serial

.. autoclass:: txpipe.masks.TXSimpleMask

    **parallel**: No - Serial

.. autoclass:: txpipe.masks.TXSimpleMaskSource

    **parallel**: No - Serial

.. autoclass:: txpipe.masks.TXSimpleMaskFrac

    **parallel**: No - Serial

.. autoclass:: txpipe.convergence.TXConvergenceMaps

    **parallel**: No - Serial

.. autoclass:: txpipe.map_correlations.TXMapCorrelations

    **parallel**: No - Serial
