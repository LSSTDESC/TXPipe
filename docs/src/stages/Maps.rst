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



.. autotxclass:: txpipe.maps.TXBaseMaps
    :members:
    :exclude-members: run

    .. collapse:: Configuration

        .. raw:: html

            <UL>
            </UL>



.. autotxclass:: txpipe.maps.TXSourceMaps
    :members:
    :exclude-members: run

    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>block_size</strong>: (int) Default=0. </LI>
            <LI><strong>chunk_rows</strong>: (int) Default=100000. </LI>
            <LI><strong>pixelization</strong>: (str) Default=healpix. </LI>
            <LI><strong>nside</strong>: (int) Default=0. </LI>
            <LI><strong>sparse</strong>: (bool) Default=True. </LI>
            <LI><strong>ra_cent</strong>: (float) Default=nan. </LI>
            <LI><strong>dec_cent</strong>: (float) Default=nan. </LI>
            <LI><strong>npix_x</strong>: (int) Default=-1. </LI>
            <LI><strong>npix_y</strong>: (int) Default=-1. </LI>
            <LI><strong>pixel_size</strong>: (float) Default=nan. </LI>
            </UL>



.. autotxclass:: txpipe.maps.TXLensMaps
    :members:
    :exclude-members: run

    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>block_size</strong>: (int) Default=0. </LI>
            <LI><strong>chunk_rows</strong>: (int) Default=100000. </LI>
            <LI><strong>pixelization</strong>: (str) Default=healpix. </LI>
            <LI><strong>nside</strong>: (int) Default=0. </LI>
            <LI><strong>sparse</strong>: (bool) Default=True. </LI>
            <LI><strong>ra_cent</strong>: (float) Default=nan. </LI>
            <LI><strong>dec_cent</strong>: (float) Default=nan. </LI>
            <LI><strong>npix_x</strong>: (int) Default=-1. </LI>
            <LI><strong>npix_y</strong>: (int) Default=-1. </LI>
            <LI><strong>pixel_size</strong>: (float) Default=nan. </LI>
            </UL>



.. autotxclass:: txpipe.maps.TXExternalLensMaps
    :members:
    :exclude-members: run

    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>block_size</strong>: (int) Default=0. </LI>
            <LI><strong>chunk_rows</strong>: (int) Default=100000. </LI>
            <LI><strong>pixelization</strong>: (str) Default=healpix. </LI>
            <LI><strong>nside</strong>: (int) Default=0. </LI>
            <LI><strong>sparse</strong>: (bool) Default=True. </LI>
            <LI><strong>ra_cent</strong>: (float) Default=nan. </LI>
            <LI><strong>dec_cent</strong>: (float) Default=nan. </LI>
            <LI><strong>npix_x</strong>: (int) Default=-1. </LI>
            <LI><strong>npix_y</strong>: (int) Default=-1. </LI>
            <LI><strong>pixel_size</strong>: (float) Default=nan. </LI>
            </UL>



.. autotxclass:: txpipe.maps.TXDensityMaps
    :members:
    :exclude-members: run

    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>mask_threshold</strong>: (float) Default=0.0. </LI>
            </UL>



.. autotxclass:: txpipe.noise_maps.TXSourceNoiseMaps
    :members:
    :exclude-members: run

    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>chunk_rows</strong>: (int) Default=100000. </LI>
            <LI><strong>lensing_realizations</strong>: (int) Default=30. </LI>
            <LI><strong>true_shear</strong>: (bool) Default=False. </LI>
            </UL>



.. autotxclass:: txpipe.noise_maps.TXLensNoiseMaps
    :members:
    :exclude-members: run

    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>chunk_rows</strong>: (int) Default=100000. </LI>
            <LI><strong>clustering_realizations</strong>: (int) Default=1. </LI>
            <LI><strong>mask_in_weights</strong>: (bool) Default=False. </LI>
            </UL>



.. autotxclass:: txpipe.noise_maps.TXExternalLensNoiseMaps
    :members:
    :exclude-members: run

    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>chunk_rows</strong>: (int) Default=100000. </LI>
            <LI><strong>clustering_realizations</strong>: (int) Default=1. </LI>
            <LI><strong>mask_in_weights</strong>: (bool) Default=False. </LI>
            </UL>



.. autotxclass:: txpipe.noise_maps.TXNoiseMapsJax
    :members:
    :exclude-members: run

    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>chunk_rows</strong>: (int) Default=4000000. </LI>
            <LI><strong>lensing_realizations</strong>: (int) Default=30. </LI>
            <LI><strong>clustering_realizations</strong>: (int) Default=1. </LI>
            <LI><strong>seed</strong>: (int) Default=0. </LI>
            </UL>



.. autotxclass:: txpipe.auxiliary_maps.TXAuxiliarySourceMaps
    :members:
    :exclude-members: run

    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>block_size</strong>: (int) Default=0. </LI>
            <LI><strong>flag_exponent_max</strong>: (int) Default=8. </LI>
            <LI><strong>psf_prefix</strong>: (str) Default=psf_. </LI>
            <LI><strong>chunk_rows</strong>: (int) Default=100000. </LI>
            <LI><strong>pixelization</strong>: (str) Default=healpix. </LI>
            <LI><strong>nside</strong>: (int) Default=0. </LI>
            <LI><strong>sparse</strong>: (bool) Default=True. </LI>
            <LI><strong>ra_cent</strong>: (float) Default=nan. </LI>
            <LI><strong>dec_cent</strong>: (float) Default=nan. </LI>
            <LI><strong>npix_x</strong>: (int) Default=-1. </LI>
            <LI><strong>npix_y</strong>: (int) Default=-1. </LI>
            <LI><strong>pixel_size</strong>: (float) Default=nan. </LI>
            </UL>



.. autotxclass:: txpipe.auxiliary_maps.TXAuxiliaryLensMaps
    :members:
    :exclude-members: run

    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>block_size</strong>: (int) Default=0. </LI>
            <LI><strong>bright_obj_threshold</strong>: (float) Default=22.0. </LI>
            <LI><strong>depth_band</strong>: (str) Default=i. </LI>
            <LI><strong>snr_threshold</strong>: (float) Default=10.0. </LI>
            <LI><strong>snr_delta</strong>: (float) Default=1.0. </LI>
            </UL>



.. autotxclass:: txpipe.auxiliary_maps.TXUniformDepthMap
    :members:
    :exclude-members: run

    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>depth</strong>: (float) Default=25.0. </LI>
            </UL>



.. autotxclass:: txpipe.masks.TXSimpleMask
    :members:
    :exclude-members: run

    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>depth_cut</strong>: (float) Default=23.5. </LI>
            <LI><strong>bright_object_max</strong>: (float) Default=10.0. </LI>
            </UL>



.. autotxclass:: txpipe.masks.TXSimpleMaskSource
    :members:
    :exclude-members: run

    .. collapse:: Configuration

        .. raw:: html

            <UL>
            </UL>



.. autotxclass:: txpipe.masks.TXSimpleMaskFrac
    :members:
    :exclude-members: run

    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>depth_cut</strong>: (float) Default=23.5. </LI>
            <LI><strong>bright_object_max</strong>: (float) Default=10.0. </LI>
            <LI><strong>supreme_map_file</strong>: (str) Required. </LI>
            </UL>



.. autotxclass:: txpipe.convergence.TXConvergenceMaps
    :members:
    :exclude-members: run

    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>lmax</strong>: (int) Default=0. </LI>
            <LI><strong>smoothing_sigma</strong>: (float) Default=10.0. </LI>
            </UL>



.. autotxclass:: txpipe.map_correlations.TXMapCorrelations
    :members:
    :exclude-members: run

    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>supreme_path_root</strong>: (str) Default=/global/cscratch1/sd/erykoff/dc2_dr6/supreme/supreme_dc2_dr6d_v2. </LI>
            <LI><strong>nbin</strong>: (int) Default=20. </LI>
            <LI><strong>outlier_fraction</strong>: (float) Default=0.05. </LI>
            </UL>


