Maps
====

These stages deal with making different kinds of maps for analysis and
plotting.

* :py:class:`~txpipe.maps.TXBaseMaps` - A base class for mapping stages

* :py:class:`~txpipe.maps.TXSourceMaps` - Make tomographic shear maps

* :py:class:`~txpipe.maps.TXLensMaps` - Make tomographic lens number count maps

* :py:class:`~txpipe.maps.TXExternalLensMaps` - Make tomographic lens number count maps from external data

* :py:class:`~txpipe.maps.TXMainMaps` - Make both shear and number count maps

* :py:class:`~txpipe.maps.TXDensityMaps` - Convert galaxy count maps to overdensity delta maps

* :py:class:`~txpipe.noise_maps.TXSourceNoiseMaps` - Generate realizations of shear noise maps with random rotations

* :py:class:`~txpipe.noise_maps.TXLensNoiseMaps` - Generate lens density noise realizations using random splits

* :py:class:`~txpipe.noise_maps.TXExternalLensNoiseMaps` - Generate lens density noise realizations using random splits of an external catalog

* :py:class:`~txpipe.noise_maps.TXNoiseMapsJax` - Generate noise realisations of lens and source maps using JAX

* :py:class:`~txpipe.auxiliary_maps.TXAuxiliarySourceMaps` - Generate auxiliary maps from the source catalog

* :py:class:`~txpipe.auxiliary_maps.TXAuxiliaryLensMaps` - Generate auxiliary maps from the lens catalog

* :py:class:`~txpipe.auxiliary_maps.TXUniformDepthMap` - Generate a uniform depth map from the mask

* :py:class:`~txpipe.masks.TXSimpleMask` - Make a simple binary mask using a depth cut and bright object cut

* :py:class:`~txpipe.convergence.TXConvergenceMaps` - Make a convergence map from a source map using Kaiser-Squires

* :py:class:`~txpipe.map_correlations.TXMapCorrelations` - Plot shear, density, and convergence correlations with survey property maps



.. autoclass:: txpipe.maps.TXBaseMaps
   :members:


.. autoclass:: txpipe.maps.TXSourceMaps
   :members:


.. autoclass:: txpipe.maps.TXLensMaps
   :members:


.. autoclass:: txpipe.maps.TXExternalLensMaps
   :members:


.. autoclass:: txpipe.maps.TXMainMaps
   :members:


.. autoclass:: txpipe.maps.TXDensityMaps
   :members:


.. autoclass:: txpipe.noise_maps.TXSourceNoiseMaps
   :members:


.. autoclass:: txpipe.noise_maps.TXLensNoiseMaps
   :members:


.. autoclass:: txpipe.noise_maps.TXExternalLensNoiseMaps
   :members:


.. autoclass:: txpipe.noise_maps.TXNoiseMapsJax
   :members:


.. autoclass:: txpipe.auxiliary_maps.TXAuxiliarySourceMaps
   :members:


.. autoclass:: txpipe.auxiliary_maps.TXAuxiliaryLensMaps
   :members:


.. autoclass:: txpipe.auxiliary_maps.TXUniformDepthMap
   :members:


.. autoclass:: txpipe.masks.TXSimpleMask
   :members:


.. autoclass:: txpipe.convergence.TXConvergenceMaps
   :members:


.. autoclass:: txpipe.map_correlations.TXMapCorrelations
   :members:

