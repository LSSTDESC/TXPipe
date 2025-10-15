Two-Point
=========

These stages deal with measuring or predicting two-point statistics.

* :py:class:`~txpipe.random_cats.TXRandomCat` - Generate a catalog of randomly positioned points

* :py:class:`~txpipe.random_cats.TXSubsampleRandoms` - Randomly subsample the binned random catalog and save catalog

* :py:class:`~txpipe.twopoint_fourier.TXTwoPointFourier` - Make Fourier space 3x2pt measurements using NaMaster

* :py:class:`~txpipe.twopoint.TXTwoPoint` - Make 2pt measurements using TreeCorr

* :py:class:`~txpipe.twopoint.TXTwoPointPixel` - This subclass of the standard TXTwoPoint uses maps to compute

* :py:class:`~txpipe.twopoint.TXTwoPointPixelExtCross` - TXTwoPointPixel - External - Cross correlation

* :py:class:`~txpipe.theory.TXTwoPointTheoryReal` - Compute theory predictions for real-space 3x2pt measurements.

* :py:class:`~txpipe.theory.TXTwoPointTheoryFourier` - Compute theory predictions for Fourier-space 3x2pt measurements.

* :py:class:`~txpipe.jackknife.TXJackknifeCenters` - Generate jack-knife centers from random catalogs.

* :py:class:`~txpipe.jackknife.TXJackknifeCentersSource` - Generate jack-knife centers from a shear catalog.



.. autotxclass:: txpipe.random_cats.TXRandomCat
    :members:
    :exclude-members: run

    Inputs: 

    - aux_lens_maps: MapsFile
    - mask: MapsFile
    - lens_photoz_stack: QPNOfZFile
    - fiducial_cosmology: FiducialCosmology

    Outputs: 

    - random_cats: RandomsCatalog
    - binned_random_catalog: RandomsCatalog
    - binned_random_catalog_sub: RandomsCatalog
    
    Parallel: Yes - MPI


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>density</strong>: (float) Default=100.0. Number per square arcmin at median depth.</LI>
            <LI><strong>Mstar</strong>: (float) Default=23.0. Schechter distribution Mstar parameter.</LI>
            <LI><strong>alpha</strong>: (float) Default=-1.25. Schechter distribution alpha parameter.</LI>
            <LI><strong>chunk_rows</strong>: (int) Default=100000. Number of rows to process in each chunk.</LI>
            <LI><strong>method</strong>: (str) Default=quadrilateral. Method for random generation: 'quadrilateral' or 'spherical_projection'.</LI>
            <LI><strong>sample_rate</strong>: (float) Default=0.5. Fraction of random catalog to be retained in the sub-sampled catalog.</LI>
            </UL>



.. autotxclass:: txpipe.random_cats.TXSubsampleRandoms
    :members:
    :exclude-members: run

    Inputs: 

    - binned_random_catalog: HDFFile

    Outputs: 

    - binned_random_catalog_sub: RandomsCatalog
    
    Parallel: Yes - MPI


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>chunk_rows</strong>: (int) Default=100000. Number of rows to process in each chunk.</LI>
            <LI><strong>sample_rate</strong>: (float) Default=0.5. Fraction of random catalog that should be retained in the subsampled catalog.</LI>
            </UL>



.. autotxclass:: txpipe.twopoint_fourier.TXTwoPointFourier
    :members:
    :exclude-members: run

    Inputs: 

    - shear_photoz_stack: QPNOfZFile
    - lens_photoz_stack: QPNOfZFile
    - fiducial_cosmology: FiducialCosmology
    - tracer_metadata: HDFFile
    - lens_maps: MapsFile
    - source_maps: MapsFile
    - density_maps: MapsFile
    - mask: MapsFile
    - source_noise_maps: LensingNoiseMaps
    - lens_noise_maps: ClusteringNoiseMaps

    Outputs: 

    - twopoint_data_fourier: SACCFile
    
    Parallel: Yes - MPI


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>mask_threshold</strong>: (float) Default=0.0. Threshold for masking pixels</LI>
            <LI><strong>flip_g1</strong>: (bool) Default=False. Whether to flip the sign of g1</LI>
            <LI><strong>flip_g2</strong>: (bool) Default=False. Whether to flip the sign of g2</LI>
            <LI><strong>cache_dir</strong>: (str) Default=./cache/twopoint_fourier. Directory for caching intermediate results</LI>
            <LI><strong>low_mem</strong>: (bool) Default=False. Whether to use low memory mode</LI>
            <LI><strong>deproject_syst_clustering</strong>: (bool) Default=False. Whether to deproject systematic modes from clustering</LI>
            <LI><strong>systmaps_clustering_dir</strong>: (str) Default=. Directory containing systematic maps for clustering</LI>
            <LI><strong>ell_min</strong>: (int) Default=100. Minimum ell value for power spectra</LI>
            <LI><strong>ell_max</strong>: (int) Default=1500. Maximum ell value for power spectra</LI>
            <LI><strong>n_ell</strong>: (int) Default=20. Number of ell bins</LI>
            <LI><strong>ell_spacing</strong>: (str) Default=log. Spacing of ell bins (log or linear)</LI>
            <LI><strong>true_shear</strong>: (bool) Default=False. Whether to use true shear values</LI>
            <LI><strong>analytic_noise</strong>: (bool) Default=False. Whether to use analytic noise estimates</LI>
            <LI><strong>gaussian_sims_factor</strong>: (list) Default=[1.0]. Factor for Gaussian simulations</LI>
            <LI><strong>b0</strong>: (float) Default=1.0. Galaxy bias parameter</LI>
            <LI><strong>do_shear_shear</strong>: (bool) Default=True. Whether to compute shear-shear power spectra</LI>
            <LI><strong>do_shear_pos</strong>: (bool) Default=True. Whether to compute shear-position power spectra</LI>
            <LI><strong>do_pos_pos</strong>: (bool) Default=True. Whether to compute position-position power spectra</LI>
            <LI><strong>compute_theory</strong>: (bool) Default=True. Whether to compute theory predictions</LI>
            </UL>



.. autotxclass:: txpipe.twopoint.TXTwoPoint
    :members:
    :exclude-members: run

    Inputs: 

    - binned_shear_catalog: ShearCatalog
    - binned_lens_catalog: HDFFile
    - binned_random_catalog: HDFFile
    - binned_random_catalog_sub: HDFFile
    - shear_photoz_stack: QPNOfZFile
    - lens_photoz_stack: QPNOfZFile
    - patch_centers: TextFile
    - tracer_metadata: HDFFile

    Outputs: 

    - twopoint_data_real_raw: SACCFile
    - twopoint_gamma_x: SACCFile
    
    Parallel: Yes - MPI


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>calcs</strong>: (list) Default=[0, 1, 2]. Which calculations to perform: 0=shear-shear, 1=shear-position, 2=position-position</LI>
            <LI><strong>min_sep</strong>: (float) Default=0.5. Minimum separation for correlation measurements</LI>
            <LI><strong>max_sep</strong>: (float) Default=300.0. Maximum separation for correlation measurements</LI>
            <LI><strong>nbins</strong>: (int) Default=9. Number of separation bins</LI>
            <LI><strong>bin_slop</strong>: (float) Default=0.0. Tolerance for bin sloppiness in TreeCorr</LI>
            <LI><strong>sep_units</strong>: (str) Default=arcmin. Units for separation (arcmin, degrees, etc.)</LI>
            <LI><strong>flip_g1</strong>: (bool) Default=False. Whether to flip the sign of g1</LI>
            <LI><strong>flip_g2</strong>: (bool) Default=True. Whether to flip the sign of g2</LI>
            <LI><strong>cores_per_task</strong>: (int) Default=20. Number of CPU cores to use per task</LI>
            <LI><strong>verbose</strong>: (int) Default=1. Verbosity level for TreeCorr output</LI>
            <LI><strong>source_bins</strong>: (list) Default=[-1]. List of source bins to use (-1 means all)</LI>
            <LI><strong>lens_bins</strong>: (list) Default=[-1]. List of lens bins to use (-1 means all)</LI>
            <LI><strong>reduce_randoms_size</strong>: (float) Default=1.0. Factor to reduce the size of random catalogs</LI>
            <LI><strong>do_shear_shear</strong>: (bool) Default=True. Whether to compute shear-shear correlations</LI>
            <LI><strong>do_shear_pos</strong>: (bool) Default=True. Whether to compute shear-position correlations</LI>
            <LI><strong>do_pos_pos</strong>: (bool) Default=True. Whether to compute position-position correlations</LI>
            <LI><strong>auto_only</strong>: (bool) Default=False. Whether to compute only auto-correlations</LI>
            <LI><strong>var_method</strong>: (str) Default=jackknife. Method for computing variance (jackknife, sample, etc.)</LI>
            <LI><strong>use_randoms</strong>: (bool) Default=True. Whether to use random catalogs</LI>
            <LI><strong>low_mem</strong>: (bool) Default=False. Whether to use low memory mode</LI>
            <LI><strong>patch_dir</strong>: (str) Default=./cache/patches. Directory for storing patch files</LI>
            <LI><strong>chunk_rows</strong>: (int) Default=100000. Number of rows to process in each chunk</LI>
            <LI><strong>share_patch_files</strong>: (bool) Default=False. Whether to share patch files across processes</LI>
            <LI><strong>metric</strong>: (str) Default=Euclidean. Distance metric to use (Euclidean, Arc, etc.)</LI>
            <LI><strong>gaussian_sims_factor</strong>: (list) Default=[1.0]. Factor for Gaussian simulations</LI>
            <LI><strong>use_subsampled_randoms</strong>: (bool) Default=True. Use subsampled randoms file for RR calculation</LI>
            </UL>



.. autotxclass:: txpipe.twopoint.TXTwoPointPixel
    :members:
    :exclude-members: run

    Inputs: 

    - density_maps: MapsFile
    - source_maps: MapsFile
    - binned_shear_catalog: ShearCatalog
    - binned_lens_catalog: HDFFile
    - binned_random_catalog: HDFFile
    - shear_photoz_stack: QPNOfZFile
    - lens_photoz_stack: QPNOfZFile
    - patch_centers: TextFile
    - tracer_metadata: HDFFile
    - mask: MapsFile

    Outputs: 

    - twopoint_data_real_raw: SACCFile
    - twopoint_gamma_x: SACCFile
    
    Parallel: Yes - MPI


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>calcs</strong>: (list) Default=[0, 1, 2]. Which calculations to perform: 0=shear-shear, 1=shear-position, 2=position-position</LI>
            <LI><strong>min_sep</strong>: (float) Default=0.5. Minimum separation for correlation measurements</LI>
            <LI><strong>max_sep</strong>: (float) Default=300.0. Maximum separation for correlation measurements</LI>
            <LI><strong>nbins</strong>: (int) Default=9. Number of separation bins</LI>
            <LI><strong>bin_slop</strong>: (float) Default=0.0. Tolerance for bin sloppiness in TreeCorr</LI>
            <LI><strong>sep_units</strong>: (str) Default=arcmin. Units for separation (arcmin, degrees, etc.)</LI>
            <LI><strong>flip_g1</strong>: (bool) Default=False. Whether to flip the sign of g1</LI>
            <LI><strong>flip_g2</strong>: (bool) Default=True. Whether to flip the sign of g2</LI>
            <LI><strong>verbose</strong>: (int) Default=1. Verbosity level for TreeCorr output</LI>
            <LI><strong>source_bins</strong>: (list) Default=[-1]. List of source bins to use (-1 means all)</LI>
            <LI><strong>lens_bins</strong>: (list) Default=[-1]. List of lens bins to use (-1 means all)</LI>
            <LI><strong>reduce_randoms_size</strong>: (float) Default=1.0. Factor to reduce the size of random catalogs</LI>
            <LI><strong>do_shear_shear</strong>: (bool) Default=True. Whether to compute shear-shear correlations</LI>
            <LI><strong>do_shear_pos</strong>: (bool) Default=True. Whether to compute shear-position correlations</LI>
            <LI><strong>do_pos_pos</strong>: (bool) Default=True. Whether to compute position-position correlations</LI>
            <LI><strong>var_method</strong>: (str) Default=jackknife. Method for computing variance (jackknife, sample, etc.)</LI>
            <LI><strong>low_mem</strong>: (bool) Default=False. Whether to use low memory mode</LI>
            <LI><strong>patch_dir</strong>: (str) Default=./cache/patches. Directory for storing patch files</LI>
            <LI><strong>chunk_rows</strong>: (int) Default=100000. Number of rows to process in each chunk</LI>
            <LI><strong>share_patch_files</strong>: (bool) Default=False. Whether to share patch files across processes</LI>
            <LI><strong>metric</strong>: (str) Default=Euclidean. Distance metric to use (Euclidean, Arc, etc.)</LI>
            <LI><strong>use_randoms</strong>: (bool) Default=True. Whether to use random catalogs</LI>
            <LI><strong>auto_only</strong>: (bool) Default=False. Whether to compute only auto-correlations</LI>
            <LI><strong>gaussian_sims_factor</strong>: (list) Default=[1.0]. Factor for Gaussian simulations</LI>
            <LI><strong>use_subsampled_randoms</strong>: (bool) Default=False. Use subsampled randoms file for RR calculation (not used for pixel estimator)</LI>
            </UL>



.. autotxclass:: txpipe.twopoint.TXTwoPointPixelExtCross
    :members:
    :exclude-members: run

    Inputs: 

    - density_maps: MapsFile
    - source_maps: MapsFile
    - binned_shear_catalog: ShearCatalog
    - binned_lens_catalog: HDFFile
    - binned_random_catalog: HDFFile
    - shear_photoz_stack: QPNOfZFile
    - lens_photoz_stack: QPNOfZFile
    - patch_centers: TextFile
    - tracer_metadata: HDFFile
    - mask: MapsFile

    Outputs: 

    - twopoint_data_ext_cross_raw: SACCFile
    
    Parallel: Yes - MPI


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>supreme_path_root</strong>: (str) Default=. Root path for supreme files</LI>
            <LI><strong>do_pos_ext</strong>: (bool) Default=True. Whether to compute position-ext correlations</LI>
            <LI><strong>do_shear_ext</strong>: (bool) Default=True. Whether to compute shear-ext correlations</LI>
            <LI><strong>calcs</strong>: (list) Default=[0, 1, 2]. Which calculations to perform: 0=shear-shear, 1=shear-position, 2=position-position</LI>
            <LI><strong>min_sep</strong>: (float) Default=0.5. Minimum separation for correlation measurements</LI>
            <LI><strong>max_sep</strong>: (float) Default=300.0. Maximum separation for correlation measurements</LI>
            <LI><strong>nbins</strong>: (int) Default=9. Number of separation bins</LI>
            <LI><strong>bin_slop</strong>: (float) Default=0.0. Tolerance for bin sloppiness in TreeCorr</LI>
            <LI><strong>sep_units</strong>: (str) Default=arcmin. Units for separation (arcmin, degrees, etc.)</LI>
            <LI><strong>flip_g1</strong>: (bool) Default=False. Whether to flip the sign of g1</LI>
            <LI><strong>flip_g2</strong>: (bool) Default=True. Whether to flip the sign of g2</LI>
            <LI><strong>verbose</strong>: (int) Default=1. Verbosity level for TreeCorr output</LI>
            <LI><strong>source_bins</strong>: (list) Default=[-1]. List of source bins to use (-1 means all)</LI>
            <LI><strong>lens_bins</strong>: (list) Default=[-1]. List of lens bins to use (-1 means all)</LI>
            <LI><strong>reduce_randoms_size</strong>: (float) Default=1.0. Factor to reduce the size of random catalogs</LI>
            <LI><strong>do_shear_shear</strong>: (bool) Default=False. Whether to compute shear-shear correlations</LI>
            <LI><strong>do_shear_pos</strong>: (bool) Default=False. Whether to compute shear-position correlations</LI>
            <LI><strong>do_pos_pos</strong>: (bool) Default=False. Whether to compute position-position correlations</LI>
            <LI><strong>var_method</strong>: (str) Default=jackknife. Method for computing variance (jackknife, sample, etc.)</LI>
            <LI><strong>low_mem</strong>: (bool) Default=False. Whether to use low memory mode</LI>
            <LI><strong>patch_dir</strong>: (str) Default=./cache/patches. Directory for storing patch files</LI>
            <LI><strong>chunk_rows</strong>: (int) Default=100000. Number of rows to process in each chunk</LI>
            <LI><strong>share_patch_files</strong>: (bool) Default=False. Whether to share patch files across processes</LI>
            <LI><strong>metric</strong>: (str) Default=Euclidean. Distance metric to use (Euclidean, Arc, etc.)</LI>
            <LI><strong>use_randoms</strong>: (bool) Default=True. Whether to use random catalogs</LI>
            <LI><strong>auto_only</strong>: (bool) Default=False. Whether to compute only auto-correlations</LI>
            <LI><strong>gaussian_sims_factor</strong>: (list) Default=[1.0]. Factor for Gaussian simulations</LI>
            <LI><strong>use_subsampled_randoms</strong>: (bool) Default=False. Use subsampled randoms file for RR calculation (not used for pixel estimator)</LI>
            </UL>



.. autotxclass:: txpipe.theory.TXTwoPointTheoryReal
    :members:
    :exclude-members: run

    Inputs: 

    - twopoint_data_real: SACCFile
    - fiducial_cosmology: FiducialCosmology

    Outputs: 

    - twopoint_theory_real: SACCFile
    
    Parallel: No - Serial


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>galaxy_bias</strong>: (list) Default=[0.0]. Galaxy bias values per bin, [0.0] for unit bias, or single negative value for global bias parameter</LI>
            <LI><strong>smooth</strong>: (bool) Default=False. Whether to smooth the theory predictions</LI>
            </UL>



.. autotxclass:: txpipe.theory.TXTwoPointTheoryFourier
    :members:
    :exclude-members: run

    Inputs: 

    - twopoint_data_fourier: SACCFile
    - fiducial_cosmology: FiducialCosmology

    Outputs: 

    - twopoint_theory_fourier: SACCFile
    
    Parallel: No - Serial


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>galaxy_bias</strong>: (list) Default=[0.0]. Galaxy bias values per bin, [0.0] for unit bias, or single negative value for global bias parameter</LI>
            <LI><strong>smooth</strong>: (bool) Default=False. Whether to smooth the theory predictions</LI>
            </UL>



.. autotxclass:: txpipe.jackknife.TXJackknifeCenters
    :members:
    :exclude-members: run

    Inputs: 

    - random_cats: RandomsCatalog

    Outputs: 

    - patch_centers: TextFile
    - jk: PNGFile
    
    Parallel: No - Serial


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>npatch</strong>: (int) Default=10. Number of jackknife patches to create</LI>
            <LI><strong>every_nth</strong>: (int) Default=100. Use every nth point from the random catalog to reduce data size</LI>
            </UL>



.. autotxclass:: txpipe.jackknife.TXJackknifeCentersSource
    :members:
    :exclude-members: run

    Inputs: 

    - shear_catalog: ShearCatalog

    Outputs: 

    - patch_centers: TextFile
    - jk: PNGFile
    
    Parallel: No - Serial


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>npatch</strong>: (int) Default=10. Number of jackknife patches to create</LI>
            <LI><strong>every_nth</strong>: (int) Default=100. Use every nth point from the random catalog to reduce data size</LI>
            </UL>


