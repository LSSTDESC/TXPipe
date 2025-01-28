Diagnostics
===========

These stages compute and/or plot diagnostics of catalogs or other data

* :py:class:`~txpipe.diagnostics.TXDiagnosticQuantiles` - Measure quantiles of various values in the shear catalog.

* :py:class:`~txpipe.diagnostics.TXSourceDiagnosticPlots` - Make diagnostic plots of the shear catalog

* :py:class:`~txpipe.diagnostics.TXLensDiagnosticPlots` - Make diagnostic plots of the lens catalog

* :py:class:`~txpipe.psf_diagnostics.TXPSFDiagnostics` - Make histograms of PSF values

* :py:class:`~txpipe.psf_diagnostics.TXPSFMomentCorr` - Compute PSF Moments

* :py:class:`~txpipe.psf_diagnostics.TXTauStatistics` - Compute and plot PSF Tau statistics where the definition of Tau stats are eq. 20-22

* :py:class:`~txpipe.psf_diagnostics.TXRoweStatistics` - Compute and plot PSF Rowe statistics

* :py:class:`~txpipe.psf_diagnostics.TXGalaxyStarShear` - Compute and plot star x galaxy and star x star correlations.

* :py:class:`~txpipe.psf_diagnostics.TXGalaxyStarDensity` - Compute and plot star x galaxy and star x star density correlations

* :py:class:`~txpipe.psf_diagnostics.TXBrighterFatterPlot` - Compute and plot a diagnostic of the brighter-fatter effect

* :py:class:`~txpipe.twopoint_null_tests.TXGammaTFieldCenters` - Make diagnostic 2pt measurements of tangential shear around field centers

* :py:class:`~txpipe.twopoint_null_tests.TXGammaTStars` - Make diagnostic 2pt measurements of tangential shear around stars

* :py:class:`~txpipe.twopoint_null_tests.TXGammaTRandoms` - Make diagnostic 2pt measurements of tangential shear around randoms

* :py:class:`~txpipe.twopoint_null_tests.TXApertureMass` - Measure the aperture mass statistics with TreeCorr

* :py:class:`~txpipe.spatial_diagnostics.TXFocalPlanePlot` - Make diagnostic plot of  mean measured ellipticity and residual ellipticity



.. autotxclass:: txpipe.diagnostics.TXDiagnosticQuantiles
    :members:
    :exclude-members: run

    Parallel: Yes - Dask

    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>shear_prefix</strong>: (str) Default=mcal_. </LI>
            <LI><strong>psf_prefix</strong>: (str) Default=mcal_psf_. </LI>
            <LI><strong>nbins</strong>: (int) Default=20. </LI>
            <LI><strong>chunk_rows</strong>: (int) Default=0. </LI>
            <LI><strong>bands</strong>: (str) Default=riz. </LI>
            </UL>



.. autotxclass:: txpipe.diagnostics.TXSourceDiagnosticPlots
    :members:
    :exclude-members: run

    Parallel: Yes - MPI

    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>chunk_rows</strong>: (int) Default=100000. </LI>
            <LI><strong>delta_gamma</strong>: (float) Default=0.02. </LI>
            <LI><strong>shear_prefix</strong>: (str) Default=mcal_. </LI>
            <LI><strong>psf_prefix</strong>: (str) Default=mcal_psf_. </LI>
            <LI><strong>nbins</strong>: (int) Default=20. </LI>
            <LI><strong>g_min</strong>: (float) Default=-0.03. </LI>
            <LI><strong>g_max</strong>: (float) Default=0.05. </LI>
            <LI><strong>psfT_min</strong>: (float) Default=0.04. </LI>
            <LI><strong>psfT_max</strong>: (float) Default=0.36. </LI>
            <LI><strong>T_min</strong>: (float) Default=0.04. </LI>
            <LI><strong>T_max</strong>: (float) Default=4.0. </LI>
            <LI><strong>s2n_min</strong>: (int) Default=10. </LI>
            <LI><strong>s2n_max</strong>: (int) Default=300. </LI>
            <LI><strong>psf_unit_conv</strong>: (bool) Default=False. </LI>
            <LI><strong>bands</strong>: (str) Default=riz. </LI>
            </UL>



.. autotxclass:: txpipe.diagnostics.TXLensDiagnosticPlots
    :members:
    :exclude-members: run

    Parallel: Yes - Dask

    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>block_size</strong>: (int) Default=0. </LI>
            <LI><strong>delta_gamma</strong>: (float) Default=0.02. </LI>
            <LI><strong>mag_min</strong>: (int) Default=18. </LI>
            <LI><strong>mag_max</strong>: (int) Default=28. </LI>
            <LI><strong>snr_min</strong>: (int) Default=5. </LI>
            <LI><strong>snr_max</strong>: (int) Default=200. </LI>
            <LI><strong>bands</strong>: (str) Default=ugrizy. </LI>
            </UL>



.. autotxclass:: txpipe.psf_diagnostics.TXPSFDiagnostics
    :members:
    :exclude-members: run

    Parallel: Yes - MPI

    .. collapse:: Configuration

        .. raw:: html

            <UL>
            </UL>



.. autotxclass:: txpipe.psf_diagnostics.TXPSFMomentCorr
    :members:
    :exclude-members: run

    Parallel: No - Serial

    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>min_sep</strong>: (float) Default=0.5. </LI>
            <LI><strong>max_sep</strong>: (float) Default=250.0. </LI>
            <LI><strong>nbins</strong>: (int) Default=20. </LI>
            <LI><strong>bin_slop</strong>: (float) Default=0.01. </LI>
            <LI><strong>sep_units</strong>: (str) Default=arcmin. </LI>
            <LI><strong>subtract_mean</strong>: (bool) Default=False. </LI>
            </UL>



.. autotxclass:: txpipe.psf_diagnostics.TXTauStatistics
    :members:
    :exclude-members: run

    Parallel: No - Serial

    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>min_sep</strong>: (float) Default=0.5. </LI>
            <LI><strong>max_sep</strong>: (float) Default=250.0. </LI>
            <LI><strong>nbins</strong>: (int) Default=20. </LI>
            <LI><strong>bin_slop</strong>: (float) Default=0.01. </LI>
            <LI><strong>sep_units</strong>: (str) Default=arcmin. </LI>
            <LI><strong>npatch</strong>: (int) Default=150. </LI>
            <LI><strong>psf_size_units</strong>: (str) Default=sigma. </LI>
            <LI><strong>subtract_mean</strong>: (bool) Default=False. </LI>
            <LI><strong>dec_cut</strong>: (bool) Default=True. </LI>
            <LI><strong>star_type</strong>: (str) Default=PSF-reserved. </LI>
            <LI><strong>cov_method</strong>: (str) Default=bootstrap. </LI>
            <LI><strong>flip_g2</strong>: (bool) Default=False. </LI>
            <LI><strong>tomographic</strong>: (bool) Default=True. </LI>
            </UL>



.. autotxclass:: txpipe.psf_diagnostics.TXRoweStatistics
    :members:
    :exclude-members: run

    Parallel: No - Serial

    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>min_sep</strong>: (float) Default=0.5. </LI>
            <LI><strong>max_sep</strong>: (float) Default=250.0. </LI>
            <LI><strong>nbins</strong>: (int) Default=20. </LI>
            <LI><strong>bin_slop</strong>: (float) Default=0.01. </LI>
            <LI><strong>sep_units</strong>: (str) Default=arcmin. </LI>
            <LI><strong>psf_size_units</strong>: (str) Default=sigma. </LI>
            <LI><strong>definition</strong>: (str) Default=des-y1. </LI>
            <LI><strong>subtract_mean</strong>: (bool) Default=False. </LI>
            <LI><strong>star_type</strong>: (str) Default=PSF-reserved. </LI>
            <LI><strong>var_method</strong>: (str) Default=bootstrap. </LI>
            <LI><strong>flip_g2</strong>: (bool) Default=False. </LI>
            </UL>



.. autotxclass:: txpipe.psf_diagnostics.TXGalaxyStarShear
    :members:
    :exclude-members: run

    Parallel: No - Serial

    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>min_sep</strong>: (float) Default=0.5. </LI>
            <LI><strong>max_sep</strong>: (float) Default=250.0. </LI>
            <LI><strong>nbins</strong>: (int) Default=20. </LI>
            <LI><strong>bin_slop</strong>: (float) Default=0.1. </LI>
            <LI><strong>sep_units</strong>: (str) Default=arcmin. </LI>
            <LI><strong>psf_size_units</strong>: (str) Default=sigma. </LI>
            <LI><strong>shear_catalog_type</strong>: (str) Default=metacal. </LI>
            <LI><strong>star_type</strong>: (str) Default=PSF-reserved. </LI>
            <LI><strong>flip_g2</strong>: (bool) Default=False. </LI>
            </UL>



.. autotxclass:: txpipe.psf_diagnostics.TXGalaxyStarDensity
    :members:
    :exclude-members: run

    Parallel: No - Serial

    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>min_sep</strong>: (float) Default=0.5. </LI>
            <LI><strong>max_sep</strong>: (float) Default=250.0. </LI>
            <LI><strong>nbins</strong>: (int) Default=20. </LI>
            <LI><strong>bin_slop</strong>: (float) Default=0.1. </LI>
            <LI><strong>sep_units</strong>: (str) Default=arcmin. </LI>
            <LI><strong>psf_size_units</strong>: (str) Default=sigma. </LI>
            <LI><strong>star_type</strong>: (str) Default=PSF-reserved. </LI>
            <LI><strong>flip_g2</strong>: (bool) Default=False. </LI>
            </UL>



.. autotxclass:: txpipe.psf_diagnostics.TXBrighterFatterPlot
    :members:
    :exclude-members: run

    Parallel: No - Serial

    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>band</strong>: (str) Default=r. </LI>
            <LI><strong>nbin</strong>: (int) Default=20. </LI>
            <LI><strong>mmin</strong>: (float) Default=18.5. </LI>
            <LI><strong>mmax</strong>: (float) Default=23.5. </LI>
            </UL>



.. autotxclass:: txpipe.twopoint_null_tests.TXGammaTFieldCenters
    :members:
    :exclude-members: run

    Parallel: Yes - MPI

    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>calcs</strong>: (list) Default=[0, 1, 2]. </LI>
            <LI><strong>min_sep</strong>: (float) Default=2.5. </LI>
            <LI><strong>max_sep</strong>: (int) Default=250. </LI>
            <LI><strong>nbins</strong>: (int) Default=20. </LI>
            <LI><strong>bin_slop</strong>: (float) Default=0.1. </LI>
            <LI><strong>sep_units</strong>: (str) Default=arcmin. </LI>
            <LI><strong>flip_g1</strong>: (bool) Default=False. </LI>
            <LI><strong>flip_g2</strong>: (bool) Default=True. </LI>
            <LI><strong>verbose</strong>: (int) Default=1. </LI>
            <LI><strong>reduce_randoms_size</strong>: (float) Default=1.0. </LI>
            <LI><strong>var_method</strong>: (str) Default=shot. </LI>
            <LI><strong>npatch</strong>: (int) Default=5. </LI>
            <LI><strong>use_true_shear</strong>: (bool) Default=False. </LI>
            <LI><strong>subtract_mean_shear</strong>: (bool) Default=False. </LI>
            <LI><strong>use_randoms</strong>: (bool) Default=True. </LI>
            <LI><strong>patch_dir</strong>: (str) Default=./cache/patches. </LI>
            <LI><strong>low_mem</strong>: (bool) Default=False. </LI>
            <LI><strong>chunk_rows</strong>: (int) Default=100000. </LI>
            <LI><strong>share_patch_files</strong>: (bool) Default=False. </LI>
            <LI><strong>use_subsampled_randoms</strong>: (bool) Default=False. </LI>
            </UL>



.. autotxclass:: txpipe.twopoint_null_tests.TXGammaTStars
    :members:
    :exclude-members: run

    Parallel: Yes - MPI

    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>calcs</strong>: (list) Default=[0, 1, 2]. </LI>
            <LI><strong>min_sep</strong>: (float) Default=2.5. </LI>
            <LI><strong>max_sep</strong>: (int) Default=100. </LI>
            <LI><strong>nbins</strong>: (int) Default=20. </LI>
            <LI><strong>bin_slop</strong>: (int) Default=1. </LI>
            <LI><strong>sep_units</strong>: (str) Default=arcmin. </LI>
            <LI><strong>flip_g1</strong>: (bool) Default=False. </LI>
            <LI><strong>flip_g2</strong>: (bool) Default=True. </LI>
            <LI><strong>verbose</strong>: (int) Default=1. </LI>
            <LI><strong>reduce_randoms_size</strong>: (float) Default=1.0. </LI>
            <LI><strong>var_method</strong>: (str) Default=shot. </LI>
            <LI><strong>npatch</strong>: (int) Default=5. </LI>
            <LI><strong>use_true_shear</strong>: (bool) Default=False. </LI>
            <LI><strong>subtract_mean_shear</strong>: (bool) Default=False. </LI>
            <LI><strong>use_randoms</strong>: (bool) Default=True. </LI>
            <LI><strong>patch_dir</strong>: (str) Default=./cache/patches. </LI>
            <LI><strong>low_mem</strong>: (bool) Default=False. </LI>
            <LI><strong>chunk_rows</strong>: (int) Default=100000. </LI>
            <LI><strong>share_patch_files</strong>: (bool) Default=False. </LI>
            <LI><strong>use_subsampled_randoms</strong>: (bool) Default=False. </LI>
            </UL>



.. autotxclass:: txpipe.twopoint_null_tests.TXGammaTRandoms
    :members:
    :exclude-members: run

    Parallel: Yes - MPI

    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>calcs</strong>: (list) Default=[0, 1, 2]. </LI>
            <LI><strong>min_sep</strong>: (float) Default=2.5. </LI>
            <LI><strong>max_sep</strong>: (int) Default=100. </LI>
            <LI><strong>nbins</strong>: (int) Default=20. </LI>
            <LI><strong>bin_slop</strong>: (int) Default=1. </LI>
            <LI><strong>sep_units</strong>: (str) Default=arcmin. </LI>
            <LI><strong>flip_g1</strong>: (bool) Default=False. </LI>
            <LI><strong>flip_g2</strong>: (bool) Default=True. </LI>
            <LI><strong>verbose</strong>: (int) Default=1. </LI>
            <LI><strong>reduce_randoms_size</strong>: (float) Default=1.0. </LI>
            <LI><strong>var_method</strong>: (str) Default=shot. </LI>
            <LI><strong>npatch</strong>: (int) Default=5. </LI>
            <LI><strong>use_true_shear</strong>: (bool) Default=False. </LI>
            <LI><strong>subtract_mean_shear</strong>: (bool) Default=False. </LI>
            <LI><strong>use_randoms</strong>: (bool) Default=False. </LI>
            <LI><strong>patch_dir</strong>: (str) Default=./cache/patches. </LI>
            <LI><strong>low_mem</strong>: (bool) Default=False. </LI>
            <LI><strong>chunk_rows</strong>: (int) Default=100000. </LI>
            <LI><strong>share_patch_files</strong>: (bool) Default=False. </LI>
            <LI><strong>use_subsampled_randoms</strong>: (bool) Default=False. </LI>
            </UL>



.. autotxclass:: txpipe.twopoint_null_tests.TXApertureMass
    :members:
    :exclude-members: run

    Parallel: Yes - MPI

    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>calcs</strong>: (list) Default=[0, 1, 2]. </LI>
            <LI><strong>min_sep</strong>: (float) Default=0.5. </LI>
            <LI><strong>max_sep</strong>: (float) Default=300.0. </LI>
            <LI><strong>nbins</strong>: (int) Default=15. </LI>
            <LI><strong>bin_slop</strong>: (float) Default=0.02. </LI>
            <LI><strong>sep_units</strong>: (str) Default=arcmin. </LI>
            <LI><strong>flip_g1</strong>: (bool) Default=False. </LI>
            <LI><strong>flip_g2</strong>: (bool) Default=True. </LI>
            <LI><strong>verbose</strong>: (int) Default=1. </LI>
            <LI><strong>source_bins</strong>: (list) Default=[-1]. </LI>
            <LI><strong>lens_bins</strong>: (list) Default=[-1]. </LI>
            <LI><strong>reduce_randoms_size</strong>: (float) Default=1.0. </LI>
            <LI><strong>var_method</strong>: (str) Default=jackknife. </LI>
            <LI><strong>use_true_shear</strong>: (bool) Default=False. </LI>
            <LI><strong>subtract_mean_shear</strong>: (bool) Default=False. </LI>
            <LI><strong>use_randoms</strong>: (bool) Default=False. </LI>
            <LI><strong>low_mem</strong>: (bool) Default=False. </LI>
            <LI><strong>patch_dir</strong>: (str) Default=./cache/patches. </LI>
            <LI><strong>chunk_rows</strong>: (int) Default=100000. </LI>
            <LI><strong>share_patch_files</strong>: (bool) Default=False. </LI>
            </UL>



.. autotxclass:: txpipe.spatial_diagnostics.TXFocalPlanePlot
    :members:
    :exclude-members: run

    Parallel: No - Serial

    .. collapse:: Configuration

        .. raw:: html

            <UL>
            </UL>


