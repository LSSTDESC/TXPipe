Extensions
==========

These stages are written for TXPipe extension projects. They are not imported
by default; see :ref:`TXPipe Stages`.

* :py:class:`~txpipe.extensions.cluster_counts.bin_cluster.CLClusterBinningRedshiftRichness` - Divide a cluster catalog into bins of redshift and richness.

* :py:class:`~txpipe.extensions.cluster_counts.sources_select_compute.CLClusterShearCatalogs` - Parameters

* :py:class:`~txpipe.extensions.cluster_counts.make_ensemble_profile.CLClusterEnsembleProfiles` - Compute stacked radial shear profiles of ensembles of clusters.

* :py:class:`~txpipe.extensions.cluster_counts.convert_to_sacc.CLClusterSACC` - Convert cluster ensemble profiles into a SACC file.

* :py:class:`~txpipe.extensions.twopoint_cluster.TXTwoPointCluster` - TXPipe task for measuring two-point correlation functions

* :py:class:`~txpipe.extensions.twopoint_scia.TXTwoPointSelfCalibrationIA` - This is the class to calculate the 2pt measurements needed for doing

* :py:class:`~txpipe.extensions.twopoint_scia.TXTwoPointSourcePixels` - This is utilizing the interface for source only implemented above to

* :py:class:`~txpipe.extensions.twopoint_scia.TXTwoPointSCIAArc` - This is an experimental class, for calculating the self-calibration terms but using

* :py:class:`~txpipe.extensions.hos.base.HOSStage` - A stage intended as a base for higher-order statistics (HOS) measurements.

* :py:class:`~txpipe.extensions.hos.fsb.HOSFSB` - Measure the filtered squared bispectrum (FSB) from over-density maps.

* :py:class:`~txpipe.extensions.cmb_lensing.ingest.TXIngestPlanckLensingMaps` - Ingest Planck CMB lensing maps.

* :py:class:`~txpipe.extensions.cmb_lensing.ingest.TXIngestQuaia` - Ingest Quaia lensing maps as TXPipe overdensity (delta) maps.

* :py:class:`~txpipe.extensions.cmb_lensing.twopoint_fourier_cross.TXCMBLensingCrossMonteCarloCorrection` - Compute equation 23 in https://arxiv.org/pdf/2407.04607

* :py:class:`~txpipe.extensions.cmb_lensing.twopoint_fourier_cross.TXTwoPointFourierCMBLensingCrossDensity` - Compute the cross-correlation maps between CMB lensing and galaxy density.



.. autotxclass:: txpipe.extensions.cluster_counts.bin_cluster.CLClusterBinningRedshiftRichness
    :members:
    :exclude-members: run

    Inputs: 

    - cluster_catalog: HDFFile

    Outputs: 

    - cluster_catalog_tomography: HDFFile
    
    Parallel: No - Serial


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>zedge</strong>: (list) Default=[0.2, 0.4, 0.6, 0.8, 1.0]. </LI>
            <LI><strong>richedge</strong>: (list) Default=[5.0, 10.0, 20.0]. </LI>
            <LI><strong>initial_size</strong>: (int) Default=100000. </LI>
            <LI><strong>chunk_rows</strong>: (int) Default=100000. </LI>
            </UL>



.. autotxclass:: txpipe.extensions.cluster_counts.sources_select_compute.CLClusterShearCatalogs
    :members:
    :exclude-members: run

    Inputs: 

    - cluster_catalog: HDFFile
    - shear_catalog: ShearCatalog
    - fiducial_cosmology: FiducialCosmology
    - shear_tomography_catalog: TomographyCatalog
    - source_photoz_pdfs: PhotozPDFFile

    Outputs: 

    - cluster_shear_catalogs: HDFFile
    
    Parallel: Yes - MPI


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>chunk_rows</strong>: (int) Default=100000. </LI>
            <LI><strong>max_radius</strong>: (float) Default=10.0. </LI>
            <LI><strong>delta_z</strong>: (float) Default=0.1. </LI>
            <LI><strong>redshift_cut_criterion</strong>: (str) Default=zmode. </LI>
            <LI><strong>redshift_weight_criterion</strong>: (str) Default=zmode. </LI>
            <LI><strong>redshift_cut_criterion_pdf_fraction</strong>: (float) Default=0.9. </LI>
            <LI><strong>subtract_mean_shear</strong>: (bool) Default=False. </LI>
            <LI><strong>coordinate_system</strong>: (str) Default=celestial. </LI>
            <LI><strong>use_true_shear</strong>: (bool) Default=False. </LI>
            <LI><strong>delta_sigma</strong>: (bool) Default=True. </LI>
            <LI><strong>use_shape_noise</strong>: (bool) Default=False. </LI>
            <LI><strong>use_radius</strong>: (bool) Default=True. </LI>
            <LI><strong>max_angle</strong>: (int) Default=30. </LI>
            </UL>



.. autotxclass:: txpipe.extensions.cluster_counts.make_ensemble_profile.CLClusterEnsembleProfiles
    :members:
    :exclude-members: run

    Inputs: 

    - cluster_catalog_tomography: HDFFile
    - fiducial_cosmology: FiducialCosmology
    - cluster_shear_catalogs: HDFFile

    Outputs: 

    - cluster_profiles: PickleFile
    
    Parallel: Yes - MPI


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>r_min</strong>: (float) Default=0.2. </LI>
            <LI><strong>r_max</strong>: (float) Default=3.0. </LI>
            <LI><strong>nbins</strong>: (int) Default=5. </LI>
            <LI><strong>magnification_profile</strong>: (bool) Default=False. </LI>
            <LI><strong>bin_units</strong>: (str) Default=mpc. </LI>
            <LI><strong>cov_type</strong>: (str) Default=jackknife_covariance. </LI>
            <LI><strong>jackknife_nside</strong>: (int) Default=32. </LI>
            <LI><strong>bootstrap_nboot</strong>: (int) Default=100. </LI>
            </UL>



.. autotxclass:: txpipe.extensions.cluster_counts.convert_to_sacc.CLClusterSACC
    :members:
    :exclude-members: run

    Inputs: 

    - cluster_profiles: PickleFile

    Outputs: 

    - cluster_sacc_catalog: SACCFile
    
    Parallel: Yes - MPI


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>cov_component</strong>: (str) Default=tan_jk. </LI>
            <LI><strong>survey_name</strong>: (str) Default=test_survey. </LI>
            <LI><strong>area</strong>: (float) Default=20.0. </LI>
            </UL>



.. autotxclass:: txpipe.extensions.twopoint_cluster.TXTwoPointCluster
    :members:
    :exclude-members: run

    Inputs: 

    - cluster_data_catalog: HDFFile
    - cluster_random_catalog: HDFFile

    Outputs: 

    - cluster_twopoint_real: SACCFile
    
    Parallel: Yes - MPI


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>redshift_bin_edges</strong>: (list) Default=[0.4, 0.8, 1.2]. </LI>
            <LI><strong>richness_bin_edges</strong>: (list) Default=[20, 30, 200]. </LI>
            <LI><strong>nbins</strong>: (int) Default=20. </LI>
            <LI><strong>min_sep</strong>: (float) Default=0.1. </LI>
            <LI><strong>max_sep</strong>: (float) Default=250.0. </LI>
            <LI><strong>units</strong>: (str) Default=arcmin. </LI>
            <LI><strong>binning_scale</strong>: (str) Default=Log. </LI>
            </UL>



.. autotxclass:: txpipe.extensions.twopoint_scia.TXTwoPointSelfCalibrationIA
    :members:
    :exclude-members: run

    Inputs: 

    - binned_shear_catalog: BinnedCatalog
    - binned_random_catalog_source: BinnedCatalog
    - shear_photoz_stack: QPNOfZFile
    - patch_centers: TextFile
    - fiducial_cosmology: FiducialCosmology
    - tracer_metadata: HDFFile

    Outputs: 

    - twopoint_data_SCIA: SACCFile
    - twopoint_gamma_x_SCIA: SACCFile
    
    Parallel: Yes - MPI


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>calcs</strong>: (list) Default=[5, 6, 7]. </LI>
            <LI><strong>min_sep</strong>: (float) Default=2.5. </LI>
            <LI><strong>max_sep</strong>: (float) Default=250.0. </LI>
            <LI><strong>nbins</strong>: (int) Default=20. </LI>
            <LI><strong>bin_slop</strong>: (float) Default=0.0. </LI>
            <LI><strong>flip_g1</strong>: (bool) Default=False. </LI>
            <LI><strong>flip_g2</strong>: (bool) Default=True. </LI>
            <LI><strong>cores_per_task</strong>: (int) Default=20. </LI>
            <LI><strong>verbose</strong>: (int) Default=1. </LI>
            <LI><strong>source_bins</strong>: (list) Default=[-1]. </LI>
            <LI><strong>lens_bins</strong>: (list) Default=[-1]. </LI>
            <LI><strong>reduce_randoms_size</strong>: (float) Default=1.0. </LI>
            <LI><strong>do_shear_source</strong>: (bool) Default=True. </LI>
            <LI><strong>do_shear_source_select</strong>: (bool) Default=True. </LI>
            <LI><strong>do_source_source</strong>: (bool) Default=False. </LI>
            <LI><strong>var_method</strong>: (str) Default=jackknife. </LI>
            <LI><strong>use_randoms</strong>: (bool) Default=False. </LI>
            <LI><strong>low_mem</strong>: (bool) Default=False. </LI>
            <LI><strong>patch_dir</strong>: (str) Default=./cache/pathces. </LI>
            <LI><strong>chunk_row</strong>: (int) Default=100000. </LI>
            <LI><strong>share_patch_files</strong>: (bool) Default=False. </LI>
            <LI><strong>metric</strong>: (str) Default=Rperp. </LI>
            <LI><strong>add_fiducial_distance</strong>: (bool) Default=True. </LI>
            <LI><strong>gaussian_sims_factor</strong>: (list) Default=[1.0]. </LI>
            <LI><strong>use_subsampled_randoms</strong>: (bool) Default=True. </LI>
            </UL>



.. autotxclass:: txpipe.extensions.twopoint_scia.TXTwoPointSourcePixels
    :members:
    :exclude-members: run

    Inputs: 

    - source_maps: MapsFile
    - binned_shear_catalog: BinnedCatalog
    - binned_random_catalog: BinnedCatalog
    - shear_photoz_stack: QPNOfZFile
    - patch_centers: TextFile
    - tracer_metadata: HDFFile
    - mask: MapsFile

    Outputs: 

    - twopoint_data_source_real_raw: SACCFile
    - twopoint_source_gamma_x: SACCFile
    
    Parallel: Yes - MPI


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>calcs</strong>: (list) Default=[0, 1, 2]. </LI>
            <LI><strong>min_sep</strong>: (float) Default=0.5. </LI>
            <LI><strong>max_sep</strong>: (float) Default=300.0. </LI>
            <LI><strong>nbins</strong>: (int) Default=9. </LI>
            <LI><strong>bin_slop</strong>: (float) Default=0.0. </LI>
            <LI><strong>sep_units</strong>: (str) Default=arcmin. </LI>
            <LI><strong>flip_g1</strong>: (bool) Default=False. </LI>
            <LI><strong>flip_g2</strong>: (bool) Default=True. </LI>
            <LI><strong>cores_per_task</strong>: (int) Default=20. </LI>
            <LI><strong>verbose</strong>: (int) Default=1. </LI>
            <LI><strong>source_bins</strong>: (list) Default=[-1]. </LI>
            <LI><strong>lens_bins</strong>: (list) Default=[-1]. </LI>
            <LI><strong>reduce_randoms_size</strong>: (float) Default=1.0. </LI>
            <LI><strong>do_shear_shear</strong>: (bool) Default=True. </LI>
            <LI><strong>do_shear_pos</strong>: (bool) Default=True. </LI>
            <LI><strong>do_pos_pos</strong>: (bool) Default=True. </LI>
            <LI><strong>var_method</strong>: (str) Default=jackknife. </LI>
            <LI><strong>low_mem</strong>: (bool) Default=False. </LI>
            <LI><strong>patch_dir</strong>: (str) Default=./cache/patches. </LI>
            <LI><strong>chunk_rows</strong>: (int) Default=100000. </LI>
            <LI><strong>share_patch_files</strong>: (bool) Default=False. </LI>
            <LI><strong>metric</strong>: (str) Default=Euclidean. </LI>
            <LI><strong>use_randoms</strong>: (bool) Default=True. </LI>
            <LI><strong>auto_only</strong>: (bool) Default=False. </LI>
            <LI><strong>gaussian_sims_factor</strong>: (list) Default=[1.0]. </LI>
            <LI><strong>use_subsampled_randoms</strong>: (bool) Default=False. </LI>
            </UL>



.. autotxclass:: txpipe.extensions.twopoint_scia.TXTwoPointSCIAArc
    :members:
    :exclude-members: run

    Inputs: 

    - binned_shear_catalog: BinnedCatalog
    - binned_random_catalog_source: HDFFile
    - shear_photoz_stack: QPNOfZFile
    - patch_centers: TextFile
    - fiducial_cosmology: FiducialCosmology
    - tracer_metadata: HDFFile

    Outputs: 

    - twopoint_data_SCIA: SACCFile
    - twopoint_gamma_x_SCIA: SACCFile
    
    Parallel: Yes - MPI


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>calcs</strong>: (list) Default=[5, 6, 7]. </LI>
            <LI><strong>min_sep</strong>: (float) Default=2.5. </LI>
            <LI><strong>max_sep</strong>: (float) Default=250.0. </LI>
            <LI><strong>nbins</strong>: (int) Default=20. </LI>
            <LI><strong>bin_slop</strong>: (float) Default=0.0. </LI>
            <LI><strong>flip_g1</strong>: (bool) Default=False. </LI>
            <LI><strong>flip_g2</strong>: (bool) Default=True. </LI>
            <LI><strong>cores_per_task</strong>: (int) Default=20. </LI>
            <LI><strong>verbose</strong>: (int) Default=1. </LI>
            <LI><strong>source_bins</strong>: (list) Default=[-1]. </LI>
            <LI><strong>lens_bins</strong>: (list) Default=[-1]. </LI>
            <LI><strong>reduce_randoms_size</strong>: (float) Default=1.0. </LI>
            <LI><strong>do_shear_source</strong>: (bool) Default=True. </LI>
            <LI><strong>do_shear_source_select</strong>: (bool) Default=True. </LI>
            <LI><strong>do_source_source</strong>: (bool) Default=False. </LI>
            <LI><strong>var_method</strong>: (str) Default=jackknife. </LI>
            <LI><strong>use_randoms</strong>: (bool) Default=False. </LI>
            <LI><strong>low_mem</strong>: (bool) Default=False. </LI>
            <LI><strong>patch_dir</strong>: (str) Default=./cache/pathces. </LI>
            <LI><strong>chunk_row</strong>: (int) Default=100000. </LI>
            <LI><strong>share_patch_files</strong>: (bool) Default=False. </LI>
            <LI><strong>metric</strong>: (str) Default=Arc. </LI>
            <LI><strong>sep_units</strong>: (str) Default=arcmin. </LI>
            <LI><strong>add_fiducial_distance</strong>: (bool) Default=True. </LI>
            <LI><strong>gaussian_sims_factor</strong>: (list) Default=[1.0]. </LI>
            <LI><strong>use_subsampled_randoms</strong>: (bool) Default=True. </LI>
            <LI><strong>use_redshift</strong>: (bool) Default=False. </LI>
            </UL>



.. autotxclass:: txpipe.extensions.hos.base.HOSStage
    :members:
    :exclude-members: run

    Inputs: None

    Outputs: None
    
    Parallel: Yes - MPI


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            </UL>



.. autotxclass:: txpipe.extensions.hos.fsb.HOSFSB
    :members:
    :exclude-members: run

    Inputs: 

    - density_maps: MapsFile
    - mask: MapsFile
    - lens_photoz_stack: QPNOfZFile
    - tracer_metadata: HDFFile

    Outputs: 

    - filtered_squared_bispectrum: SACCFile
    
    Parallel: Yes - MPI


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>ells_per_bin</strong>: (int) Default=10. </LI>
            <LI><strong>nfilters</strong>: (int) Default=5. </LI>
            <LI><strong>include_n32</strong>: (bool) Default=False. </LI>
            </UL>



.. autotxclass:: txpipe.extensions.cmb_lensing.ingest.TXIngestPlanckLensingMaps
    :members:
    :exclude-members: run

    Inputs: None

    Outputs: 

    - cmb_lensing_map: MapsFile
    
    Parallel: Yes - MPI


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>alm_file</strong>: (str) Required. </LI>
            <LI><strong>mask_file</strong>: (str) Required. </LI>
            <LI><strong>nside</strong>: (int) Default=512. </LI>
            </UL>



.. autotxclass:: txpipe.extensions.cmb_lensing.ingest.TXIngestQuaia
    :members:
    :exclude-members: run

    Inputs: None

    Outputs: 

    - density_maps: MapsFile
    - density_masks: MapsFile
    - lens_photoz_stack: QPNOfZFile
    
    Parallel: Yes - MPI


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>quaia_file</strong>: (str) Required. </LI>
            <LI><strong>selection_function_template</strong>: (str) Required. </LI>
            <LI><strong>nside</strong>: (int) Default=512. </LI>
            <LI><strong>sel_threshold</strong>: (float) Default=0.5. </LI>
            <LI><strong>num_z_bins</strong>: (int) Default=500. </LI>
            <LI><strong>zname</strong>: (str) Default=redshift_quaia. </LI>
            </UL>



.. autotxclass:: txpipe.extensions.cmb_lensing.twopoint_fourier_cross.TXCMBLensingCrossMonteCarloCorrection
    :members:
    :exclude-members: run

    Inputs: 

    - cmb_lensing_map: MapsFile
    - mask: MapsFile
    - fiducial_cosmology: FiducialCosmology

    Outputs: 

    - cmb_cross_montecarlo_correction: TextFile
    
    Parallel: Yes - MPI


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>cmb_redshift</strong>: (float) Default=1100.0. </LI>
            <LI><strong>nside</strong>: (int) Default=512. </LI>
            <LI><strong>nsim</strong>: (int) Default=1000. </LI>
            <LI><strong>mask_threshold</strong>: (float) Default=0.0. </LI>
            </UL>



.. autotxclass:: txpipe.extensions.cmb_lensing.twopoint_fourier_cross.TXTwoPointFourierCMBLensingCrossDensity
    :members:
    :exclude-members: run

    Inputs: 

    - cmb_lensing_map: MapsFile
    - cmb_lensing_beam: TextFile
    - density_maps: MapsFile
    - density_masks: MapsFile
    - cmb_cross_montecarlo_correction: TextFile
    - lens_photoz_stack: QPNOfZFile

    Outputs: 

    - twopoint_data_fourier_cmb_cross_density: SACCFile
    - twopoint_data_fourier_cmb_cross_density_plot: PNGFile
    
    Parallel: Yes - MPI


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>mask_threshold</strong>: (float) Default=0.0. </LI>
            <LI><strong>bandpower_width</strong>: (int) Default=30. </LI>
            <LI><strong>nside</strong>: (int) Default=512. </LI>
            </UL>


