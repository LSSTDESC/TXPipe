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

    Inputs: 

    - shear_catalog: ShearCatalog
    - shear_tomography_catalog: TomographyCatalog

    Outputs: 

    - shear_catalog_quantiles: HDFFile
    
    Parallel: Yes - Dask


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>shear_prefix</strong>: (str) Default=mcal_. Prefix for shear columns in the catalog.</LI>
            <LI><strong>psf_prefix</strong>: (str) Default=mcal_psf_. Prefix for PSF columns in the catalog.</LI>
            <LI><strong>nbins</strong>: (int) Default=20. Number of quantile bins to compute.</LI>
            <LI><strong>chunk_rows</strong>: (int) Default=0. Number of rows to process in each chunk (0 means auto).</LI>
            <LI><strong>bands</strong>: (str) Default=riz. Bands to use for diagnostics.</LI>
            </UL>



.. autotxclass:: txpipe.diagnostics.TXSourceDiagnosticPlots
    :members:
    :exclude-members: run

    Inputs: 

    - shear_catalog: ShearCatalog
    - shear_tomography_catalog: TomographyCatalog
    - shear_catalog_quantiles: HDFFile

    Outputs: 

    - g_psf_T: PNGFile
    - g_psf_g: PNGFile
    - g1_hist: PNGFile
    - g2_hist: PNGFile
    - g_snr: PNGFile
    - g_T: PNGFile
    - g_colormag: PNGFile
    - source_snr_hist: PNGFile
    - source_mag_hist: PNGFile
    - response_hist: PNGFile
    - g_psf_T_out: TextFile
    - g_psf_g_out: TextFile
    - g_snr_out: TextFile
    - g_T_out: TextFile
    
    Parallel: Yes - MPI


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>chunk_rows</strong>: (int) Default=100000. Number of rows to process in each chunk.</LI>
            <LI><strong>delta_gamma</strong>: (float) Default=0.02. Delta gamma value for metacal response calculation.</LI>
            <LI><strong>shear_prefix</strong>: (str) Default=mcal_. Prefix for shear columns in the catalog.</LI>
            <LI><strong>psf_prefix</strong>: (str) Default=mcal_psf_. Prefix for PSF columns in the catalog.</LI>
            <LI><strong>nbins</strong>: (int) Default=20. Number of bins for histograms.</LI>
            <LI><strong>g_min</strong>: (float) Default=-0.03. Minimum g value for plots.</LI>
            <LI><strong>g_max</strong>: (float) Default=0.05. Maximum g value for plots.</LI>
            <LI><strong>psfT_min</strong>: (float) Default=0.04. Minimum PSF T value for plots.</LI>
            <LI><strong>psfT_max</strong>: (float) Default=0.36. Maximum PSF T value for plots.</LI>
            <LI><strong>T_min</strong>: (float) Default=0.04. Minimum T value for plots.</LI>
            <LI><strong>T_max</strong>: (float) Default=4.0. Maximum T value for plots.</LI>
            <LI><strong>s2n_min</strong>: (float) Default=10. Minimum S/N value for plots.</LI>
            <LI><strong>s2n_max</strong>: (float) Default=300. Maximum S/N value for plots.</LI>
            <LI><strong>psf_unit_conv</strong>: (bool) Default=False. Whether to convert PSF units.</LI>
            <LI><strong>bands</strong>: (str) Default=riz. Bands to use for diagnostics.</LI>
            </UL>



.. autotxclass:: txpipe.diagnostics.TXLensDiagnosticPlots
    :members:
    :exclude-members: run

    Inputs: 

    - photometry_catalog: HDFFile
    - lens_tomography_catalog: TomographyCatalog

    Outputs: 

    - lens_snr_hist: PNGFile
    - lens_mag_hist: PNGFile
    
    Parallel: Yes - Dask


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>block_size</strong>: (int) Default=0. Block size for dask processing (0 means auto).</LI>
            <LI><strong>delta_gamma</strong>: (float) Default=0.02. Delta gamma value for metacal response calculation.</LI>
            <LI><strong>mag_min</strong>: (float) Default=18. Minimum magnitude for plots.</LI>
            <LI><strong>mag_max</strong>: (float) Default=28. Maximum magnitude for plots.</LI>
            <LI><strong>snr_min</strong>: (float) Default=5. Minimum S/N for plots.</LI>
            <LI><strong>snr_max</strong>: (float) Default=200. Maximum S/N for plots.</LI>
            <LI><strong>bands</strong>: (str) Default=ugrizy. Bands to use for diagnostics.</LI>
            </UL>



.. autotxclass:: txpipe.psf_diagnostics.TXPSFDiagnostics
    :members:
    :exclude-members: run

    Inputs: 

    - star_catalog: HDFFile

    Outputs: 

    - e1_psf_residual_hist: PNGFile
    - e2_psf_residual_hist: PNGFile
    - T_frac_psf_residual_hist: PNGFile
    - star_psf_stats: YamlFile
    
    Parallel: Yes - MPI


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            </UL>



.. autotxclass:: txpipe.psf_diagnostics.TXPSFMomentCorr
    :members:
    :exclude-members: run

    Inputs: 

    - star_catalog: HDFFile

    Outputs: 

    - moments_stats: HDFFile
    
    Parallel: No - Serial


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>min_sep</strong>: (float) Default=0.5. Minimum separation in arcmin.</LI>
            <LI><strong>max_sep</strong>: (float) Default=250.0. Maximum separation in arcmin.</LI>
            <LI><strong>nbins</strong>: (int) Default=20. Number of bins.</LI>
            <LI><strong>bin_slop</strong>: (float) Default=0.01. Bin slop for treecorr.</LI>
            <LI><strong>sep_units</strong>: (str) Default=arcmin. Separation units.</LI>
            <LI><strong>subtract_mean</strong>: (bool) Default=False. Subtract mean from data.</LI>
            </UL>



.. autotxclass:: txpipe.psf_diagnostics.TXTauStatistics
    :members:
    :exclude-members: run

    Inputs: 

    - binned_shear_catalog: ShearCatalog
    - star_catalog: HDFFile
    - rowe_stats: HDFFile

    Outputs: 

    - tau0: PNGFile
    - tau2: PNGFile
    - tau5: PNGFile
    - tau_stats: HDFFile
    
    Parallel: No - Serial


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>min_sep</strong>: (float) Default=0.5. Minimum separation in arcmin.</LI>
            <LI><strong>max_sep</strong>: (float) Default=250.0. Maximum separation in arcmin.</LI>
            <LI><strong>nbins</strong>: (int) Default=20. Number of bins.</LI>
            <LI><strong>bin_slop</strong>: (float) Default=0.01. Bin slop for treecorr.</LI>
            <LI><strong>sep_units</strong>: (str) Default=arcmin. Separation units.</LI>
            <LI><strong>npatch</strong>: (int) Default=150. Number of patches for bootstrap.</LI>
            <LI><strong>psf_size_units</strong>: (str) Default=sigma. Units for PSF size.</LI>
            <LI><strong>subtract_mean</strong>: (bool) Default=False. Subtract mean from data.</LI>
            <LI><strong>dec_cut</strong>: (bool) Default=True. Affects KiDS-1000 only.</LI>
            <LI><strong>star_type</strong>: (str) Default=PSF-reserved. Star type to use.</LI>
            <LI><strong>cov_method</strong>: (str) Default=bootstrap. Covariance method.</LI>
            <LI><strong>flip_g2</strong>: (bool) Default=False. Flip g2 sign.</LI>
            <LI><strong>tomographic</strong>: (bool) Default=True. Use tomographic bins.</LI>
            </UL>



.. autotxclass:: txpipe.psf_diagnostics.TXRoweStatistics
    :members:
    :exclude-members: run

    Inputs: 

    - star_catalog: HDFFile
    - patch_centers: TextFile

    Outputs: 

    - rowe134: PNGFile
    - rowe25: PNGFile
    - rowe0: PNGFile
    - rowe_stats: HDFFile
    
    Parallel: No - Serial


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>min_sep</strong>: (float) Default=0.5. Minimum separation in arcmin.</LI>
            <LI><strong>max_sep</strong>: (float) Default=250.0. Maximum separation in arcmin.</LI>
            <LI><strong>nbins</strong>: (int) Default=20. Number of bins.</LI>
            <LI><strong>bin_slop</strong>: (float) Default=0.01. Bin slop for treecorr.</LI>
            <LI><strong>sep_units</strong>: (str) Default=arcmin. Separation units.</LI>
            <LI><strong>psf_size_units</strong>: (str) Default=sigma. Units for PSF size.</LI>
            <LI><strong>definition</strong>: (str) Default=des-y1. Definition for Rowe statistics.</LI>
            <LI><strong>subtract_mean</strong>: (bool) Default=False. Subtract mean from data.</LI>
            <LI><strong>star_type</strong>: (str) Default=PSF-reserved. Star type to use.</LI>
            <LI><strong>var_method</strong>: (str) Default=bootstrap. Variance method.</LI>
            <LI><strong>flip_g2</strong>: (bool) Default=False. Flip g2 sign.</LI>
            </UL>



.. autotxclass:: txpipe.psf_diagnostics.TXGalaxyStarShear
    :members:
    :exclude-members: run

    Inputs: 

    - shear_catalog: ShearCatalog
    - star_catalog: HDFFile
    - shear_tomography_catalog: TomographyCatalog

    Outputs: 

    - star_shear_test: PNGFile
    - star_star_test: PNGFile
    - star_shear_stats: HDFFile
    
    Parallel: No - Serial


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>min_sep</strong>: (float) Default=0.5. Minimum separation in arcmin.</LI>
            <LI><strong>max_sep</strong>: (float) Default=250.0. Maximum separation in arcmin.</LI>
            <LI><strong>nbins</strong>: (int) Default=20. Number of bins.</LI>
            <LI><strong>bin_slop</strong>: (float) Default=0.1. Bin slop for treecorr.</LI>
            <LI><strong>sep_units</strong>: (str) Default=arcmin. Separation units.</LI>
            <LI><strong>psf_size_units</strong>: (str) Default=sigma. Units for PSF size.</LI>
            <LI><strong>shear_catalog_type</strong>: (str) Default=metacal. Shear catalog type.</LI>
            <LI><strong>star_type</strong>: (str) Default=PSF-reserved. Star type to use.</LI>
            <LI><strong>flip_g2</strong>: (bool) Default=False. Flip g2 sign.</LI>
            </UL>



.. autotxclass:: txpipe.psf_diagnostics.TXGalaxyStarDensity
    :members:
    :exclude-members: run

    Inputs: 

    - shear_catalog: ShearCatalog
    - star_catalog: HDFFile
    - shear_tomography_catalog: TomographyCatalog
    - random_cats: RandomsCatalog

    Outputs: 

    - star_density_test: PNGFile
    - star_density_stats: HDFFile
    
    Parallel: No - Serial


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>min_sep</strong>: (float) Default=0.5. Minimum separation in arcmin.</LI>
            <LI><strong>max_sep</strong>: (float) Default=250.0. Maximum separation in arcmin.</LI>
            <LI><strong>nbins</strong>: (int) Default=20. Number of bins.</LI>
            <LI><strong>bin_slop</strong>: (float) Default=0.1. Bin slop for treecorr.</LI>
            <LI><strong>sep_units</strong>: (str) Default=arcmin. Separation units.</LI>
            <LI><strong>psf_size_units</strong>: (str) Default=sigma. Units for PSF size.</LI>
            <LI><strong>star_type</strong>: (str) Default=PSF-reserved. Star type to use.</LI>
            <LI><strong>flip_g2</strong>: (bool) Default=False. Flip g2 sign.</LI>
            </UL>



.. autotxclass:: txpipe.psf_diagnostics.TXBrighterFatterPlot
    :members:
    :exclude-members: run

    Inputs: 

    - star_catalog: HDFFile

    Outputs: 

    - brighter_fatter_plot: PNGFile
    - brighter_fatter_data: HDFFile
    
    Parallel: No - Serial


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>band</strong>: (str) Default=r. Band to use for magnitude.</LI>
            <LI><strong>nbin</strong>: (int) Default=20. Number of magnitude bins.</LI>
            <LI><strong>mmin</strong>: (float) Default=18.5. Minimum magnitude.</LI>
            <LI><strong>mmax</strong>: (float) Default=23.5. Maximum magnitude.</LI>
            </UL>



.. autotxclass:: txpipe.twopoint_null_tests.TXGammaTFieldCenters
    :members:
    :exclude-members: run

    Inputs: 

    - binned_shear_catalog: ShearCatalog
    - shear_photoz_stack: QPNOfZFile
    - lens_photoz_stack: QPNOfZFile
    - random_cats: RandomsCatalog
    - exposures: HDFFile
    - patch_centers: TextFile
    - tracer_metadata: HDFFile

    Outputs: 

    - gammat_field_center: SACCFile
    - gammat_field_center_plot: PNGFile
    
    Parallel: Yes - MPI


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>calcs</strong>: (list) Default=[0, 1, 2]. Which calculations to perform: 0=shear-shear, 1=shear-position, 2=position-position</LI>
            <LI><strong>min_sep</strong>: (float) Default=2.5. Minimum separation for correlation measurements.</LI>
            <LI><strong>max_sep</strong>: (float) Default=250. Maximum separation for correlation measurements.</LI>
            <LI><strong>nbins</strong>: (int) Default=20. Number of separation bins.</LI>
            <LI><strong>bin_slop</strong>: (float) Default=0.1. Tolerance for bin sloppiness in TreeCorr.</LI>
            <LI><strong>sep_units</strong>: (str) Default=arcmin. Units for separation (arcmin, degrees, etc.).</LI>
            <LI><strong>flip_g1</strong>: (bool) Default=False. Whether to flip the sign of g1.</LI>
            <LI><strong>flip_g2</strong>: (bool) Default=True. Whether to flip the sign of g2.</LI>
            <LI><strong>verbose</strong>: (int) Default=1. Verbosity level for TreeCorr output.</LI>
            <LI><strong>reduce_randoms_size</strong>: (float) Default=1.0. Factor to reduce the size of random catalogs.</LI>
            <LI><strong>var_method</strong>: (str) Default=shot. Method for computing variance (shot, jackknife, etc.).</LI>
            <LI><strong>npatch</strong>: (int) Default=5. Number of patches for null tests.</LI>
            <LI><strong>use_true_shear</strong>: (bool) Default=False. Whether to use true shear values.</LI>
            <LI><strong>subtract_mean_shear</strong>: (bool) Default=False. Whether to subtract mean shear.</LI>
            <LI><strong>use_randoms</strong>: (bool) Default=True. Whether to use random catalogs.</LI>
            <LI><strong>patch_dir</strong>: (str) Default=./cache/patches. Directory for storing patch files.</LI>
            <LI><strong>low_mem</strong>: (bool) Default=False. Whether to use low memory mode.</LI>
            <LI><strong>chunk_rows</strong>: (int) Default=100000. Number of rows to process in each chunk.</LI>
            <LI><strong>share_patch_files</strong>: (bool) Default=False. Whether to share patch files across processes.</LI>
            <LI><strong>use_subsampled_randoms</strong>: (bool) Default=False. Use subsampled randoms file for RR calculation.</LI>
            </UL>



.. autotxclass:: txpipe.twopoint_null_tests.TXGammaTStars
    :members:
    :exclude-members: run

    Inputs: 

    - binned_shear_catalog: ShearCatalog
    - shear_tomography_catalog: TomographyCatalog
    - shear_photoz_stack: QPNOfZFile
    - lens_photoz_stack: QPNOfZFile
    - random_cats: RandomsCatalog
    - binned_star_catalog: HDFFile
    - patch_centers: TextFile
    - tracer_metadata: HDFFile
    - binned_random_catalog: HDFFile

    Outputs: 

    - gammat_bright_stars: SACCFile
    - gammat_bright_stars_plot: PNGFile
    - gammat_dim_stars: SACCFile
    - gammat_dim_stars_plot: PNGFile
    
    Parallel: Yes - MPI


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>calcs</strong>: (list) Default=[0, 1, 2]. Which calculations to perform: 0=shear-shear, 1=shear-position, 2=position-position</LI>
            <LI><strong>min_sep</strong>: (float) Default=2.5. Minimum separation for correlation measurements.</LI>
            <LI><strong>max_sep</strong>: (float) Default=100. Maximum separation for correlation measurements.</LI>
            <LI><strong>nbins</strong>: (int) Default=20. Number of separation bins.</LI>
            <LI><strong>bin_slop</strong>: (float) Default=1. Tolerance for bin sloppiness in TreeCorr.</LI>
            <LI><strong>sep_units</strong>: (str) Default=arcmin. Units for separation (arcmin, degrees, etc.).</LI>
            <LI><strong>flip_g1</strong>: (bool) Default=False. Whether to flip the sign of g1.</LI>
            <LI><strong>flip_g2</strong>: (bool) Default=True. Whether to flip the sign of g2.</LI>
            <LI><strong>verbose</strong>: (int) Default=1. Verbosity level for TreeCorr output.</LI>
            <LI><strong>reduce_randoms_size</strong>: (float) Default=1.0. Factor to reduce the size of random catalogs.</LI>
            <LI><strong>var_method</strong>: (str) Default=shot. Method for computing variance (shot, jackknife, etc.).</LI>
            <LI><strong>npatch</strong>: (int) Default=5. Number of patches for null tests.</LI>
            <LI><strong>use_true_shear</strong>: (bool) Default=False. Whether to use true shear values.</LI>
            <LI><strong>subtract_mean_shear</strong>: (bool) Default=False. Whether to subtract mean shear.</LI>
            <LI><strong>use_randoms</strong>: (bool) Default=True. Whether to use random catalogs.</LI>
            <LI><strong>patch_dir</strong>: (str) Default=./cache/patches. Directory for storing patch files.</LI>
            <LI><strong>low_mem</strong>: (bool) Default=False. Whether to use low memory mode.</LI>
            <LI><strong>chunk_rows</strong>: (int) Default=100000. Number of rows to process in each chunk.</LI>
            <LI><strong>share_patch_files</strong>: (bool) Default=False. Whether to share patch files across processes.</LI>
            <LI><strong>use_subsampled_randoms</strong>: (bool) Default=False. Use subsampled randoms file for RR calculation.</LI>
            </UL>



.. autotxclass:: txpipe.twopoint_null_tests.TXGammaTRandoms
    :members:
    :exclude-members: run

    Inputs: 

    - binned_shear_catalog: ShearCatalog
    - shear_photoz_stack: QPNOfZFile
    - random_cats: RandomsCatalog
    - patch_centers: TextFile
    - tracer_metadata: HDFFile

    Outputs: 

    - gammat_randoms: SACCFile
    - gammat_randoms_plot: PNGFile
    
    Parallel: Yes - MPI


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>calcs</strong>: (list) Default=[0, 1, 2]. Which calculations to perform: 0=shear-shear, 1=shear-position, 2=position-position</LI>
            <LI><strong>min_sep</strong>: (float) Default=2.5. Minimum separation for correlation measurements.</LI>
            <LI><strong>max_sep</strong>: (float) Default=100. Maximum separation for correlation measurements.</LI>
            <LI><strong>nbins</strong>: (int) Default=20. Number of separation bins.</LI>
            <LI><strong>bin_slop</strong>: (float) Default=1. Tolerance for bin sloppiness in TreeCorr.</LI>
            <LI><strong>sep_units</strong>: (str) Default=arcmin. Units for separation (arcmin, degrees, etc.).</LI>
            <LI><strong>flip_g1</strong>: (bool) Default=False. Whether to flip the sign of g1.</LI>
            <LI><strong>flip_g2</strong>: (bool) Default=True. Whether to flip the sign of g2.</LI>
            <LI><strong>verbose</strong>: (int) Default=1. Verbosity level for TreeCorr output.</LI>
            <LI><strong>reduce_randoms_size</strong>: (float) Default=1.0. Factor to reduce the size of random catalogs.</LI>
            <LI><strong>var_method</strong>: (str) Default=shot. Method for computing variance (shot, jackknife, etc.).</LI>
            <LI><strong>npatch</strong>: (int) Default=5. Number of patches for null tests.</LI>
            <LI><strong>use_true_shear</strong>: (bool) Default=False. Whether to use true shear values.</LI>
            <LI><strong>subtract_mean_shear</strong>: (bool) Default=False. Whether to subtract mean shear.</LI>
            <LI><strong>use_randoms</strong>: (bool) Default=False. Whether to use random catalogs.</LI>
            <LI><strong>patch_dir</strong>: (str) Default=./cache/patches. Directory for storing patch files.</LI>
            <LI><strong>low_mem</strong>: (bool) Default=False. Whether to use low memory mode.</LI>
            <LI><strong>chunk_rows</strong>: (int) Default=100000. Number of rows to process in each chunk.</LI>
            <LI><strong>share_patch_files</strong>: (bool) Default=False. Whether to share patch files across processes.</LI>
            <LI><strong>use_subsampled_randoms</strong>: (bool) Default=False. Use subsampled randoms file for RR calculation.</LI>
            </UL>



.. autotxclass:: txpipe.twopoint_null_tests.TXApertureMass
    :members:
    :exclude-members: run

    Inputs: 

    - binned_shear_catalog: ShearCatalog
    - shear_photoz_stack: QPNOfZFile
    - patch_centers: TextFile
    - tracer_metadata: HDFFile

    Outputs: 

    - aperture_mass_data: SACCFile
    
    Parallel: Yes - MPI


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>calcs</strong>: (list) Default=[0, 1, 2]. Which calculations to perform: 0=shear-shear, 1=shear-position, 2=position-position</LI>
            <LI><strong>min_sep</strong>: (float) Default=0.5. Minimum separation for correlation measurements.</LI>
            <LI><strong>max_sep</strong>: (float) Default=300.0. Maximum separation for correlation measurements.</LI>
            <LI><strong>nbins</strong>: (int) Default=15. Number of separation bins.</LI>
            <LI><strong>bin_slop</strong>: (float) Default=0.02. Tolerance for bin sloppiness in TreeCorr.</LI>
            <LI><strong>sep_units</strong>: (str) Default=arcmin. Units for separation (arcmin, degrees, etc.).</LI>
            <LI><strong>flip_g1</strong>: (bool) Default=False. Whether to flip the sign of g1.</LI>
            <LI><strong>flip_g2</strong>: (bool) Default=True. Whether to flip the sign of g2.</LI>
            <LI><strong>verbose</strong>: (int) Default=1. Verbosity level for TreeCorr output.</LI>
            <LI><strong>source_bins</strong>: (list) Default=[-1]. List of source bins to use (-1 means all).</LI>
            <LI><strong>lens_bins</strong>: (list) Default=[-1]. List of lens bins to use (-1 means all).</LI>
            <LI><strong>reduce_randoms_size</strong>: (float) Default=1.0. Factor to reduce the size of random catalogs.</LI>
            <LI><strong>var_method</strong>: (str) Default=jackknife. Method for computing variance (jackknife, sample, etc.).</LI>
            <LI><strong>use_true_shear</strong>: (bool) Default=False. Whether to use true shear values.</LI>
            <LI><strong>subtract_mean_shear</strong>: (bool) Default=False. Whether to subtract mean shear.</LI>
            <LI><strong>use_randoms</strong>: (bool) Default=False. Whether to use random catalogs.</LI>
            <LI><strong>low_mem</strong>: (bool) Default=False. Whether to use low memory mode.</LI>
            <LI><strong>patch_dir</strong>: (str) Default=./cache/patches. Directory for storing patch files.</LI>
            <LI><strong>chunk_rows</strong>: (int) Default=100000. Number of rows to process in each chunk.</LI>
            <LI><strong>share_patch_files</strong>: (bool) Default=False. Whether to share patch files across processes.</LI>
            </UL>



.. autotxclass:: txpipe.spatial_diagnostics.TXFocalPlanePlot
    :members:
    :exclude-members: run

    Inputs: 

    - star_catalog: HDFFile

    Outputs: 

    - focalplane_g: PNGFile
    
    Parallel: No - Serial


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            </UL>


