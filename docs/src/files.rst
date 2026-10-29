TXPipe File Tags
================

TXPipe saves everything it does in files saved in the `output_dir` directory specified in the pipeline YAML file. This page is useful if you want to learn what one of those files means and find a link to a page that describes the format of that file.

TXPipe files are organized by `tags`, which are usually just the stem of the file name (i.e. without the directory or suffix). The tags input and output by each staged are defined in the stage class. Each file has a specific :ref:`type<File Types>`

This page briefly describes all the file tags that TXPipe currently generates and uses, and links to pages describing the format of each type of file.


Aliases
-------

Any stage can be "aliased" to a different tag when the pipeline is defined in the pipeline YAML file. This means that the name used for the stage internally can be different from the name used in the file names. This is useful when there are pipeline stages that can be run multiple times on different data sets. 

For example, the `TXPhotozPlot` class can be run twice, once on the source sample and once on the lens sample, to generate n(z) plots of each.


HDF5 Files
----------

HDF files are a general-purpose file format for storing tabular data. They are the primary storage format for larger data sets in TXPipe, primarily because they are easy to read and write in parallel.  Different groups of HDF5 files are described below.


Catalog HDF5 Files
------------------

TXPipe catalog files are almost all stored in HDF5 format. They are described in full on the :ref:`File Types` page.

.. list-table:: HDF5 Files
    :header-rows: 1

    * - File Tag
      - Kind
      - Description
    * - binned_lens_catalog
      - :ref:`Binned lens catalogs`
      - The foreground lens (clustering) sample, split into tomographic bins, with weights
    * - binned_lens_catalog_unweighted
      - :ref:`Binned lens catalogs`
      - The foreground lens (clustering) sample, split into tomographic bins, with only unit weights
    * - binned_random_catalog
      - :ref:`Binned random catalogs`
      - The random catalog, split into tomographic bins
    * - binned_random_catalog_sub
      - :ref:`Binned random catalogs`
      - A smaller sub-sample of the random catalog, split into tomographic bins. Used when the full large catalog is not needed and too slow.
    * - binned_shear_catalog
      - :ref:`Binned shear catalogs`
      - The shear/source/lensing sample, split into tomographic bins and calibrated
    * - binned_star_catalog
      - :ref:`Binned star catalogs`
      - The star catalog split into sub-classes, currently by brightness
    * - cluster_catalog
      - :ref:`Cluster catalogs`
      - Locations, redshifts, and richness of clusters
    * - cluster_shear_catalogs
      - :ref:`Cluster Shear Catalogs`
      - An catalog of shear values around clusters.
    * - exposures
      - :ref:`Exposure catalogs`
      - Catalogs centers of exposurs for use in systematics tests
    * - lens_catalog
      - :ref:`HDF File<Reading HDF5 Files>`
      - A catalog of objects to be used as lenses (when something external is used instead of the photometry catalog
    * - lens_tomography_catalog
      - :ref:`Lens tomography catalogs`
      - Tomographic selection information for the lens sample
    * - lens_tomography_catalog_unweighted
      - :ref:`Lens tomography catalogs`
      - Tomographic selection information for the lens sample, without weights
    * - photometry_catalog
      - :ref:`Photometry Catalogs`
      - Photometric measurements from which the lens sample is chosen
    * - random_cats
      - :ref:`Random Catalogs`
      - A catalog of random objects following the same tomographic and location selection as the lens sample but with no underlying structure
    * - shear_catalog
      - :ref:`Shear Catalogs<Ingested Catalog Files>`
      - Shear catalogs of various different types
    * - shear_catalog_quantiles
      - :ref:`HDF File<Reading HDF5 Files>`
      - Measurements of quantiles of shear catalog columns such as SNR, size, etc.
    * - shear_tomography_catalog
      - :ref:`Shear tomography catalogs`
      - Tomographic selection and shear calibration information for the source sample
    * - spectroscopic_catalog
      - :ref:`HDF File<Reading HDF5 Files>`
      - Training set for the photo-z model
    * - star_catalog
      - :ref:`Star catalogs`
      - Catalog of star locations and magnitudes for systematics tests
    * - bhat_for_wide_data_VARIANT
      - :ref:`HDF File<Reading HDF5 Files>`
      - RAIL SOM bin assignments, for different values of VARIANT: ns, 1p, 1m, 2p, 2m
    * - binned_catalog
      - :ref:`Binned catalogs`
      - Binned catalog input to PZRailSummarize, normally aliased to a specific sample
    * - binned_lens_catalog_mag
      - :ref:`Binned lens catalogs`
      - Binned lens catalog, input to TXSSIMagnification
    * - binned_lens_catalog_nomag
      - :ref:`Binned lens catalogs`
      - Binned lens catalog, alternative version
    * - binned_random_catalog_source
      - :ref:`Binned random catalogs`
      - Random catalog with the footprint of the source sample, split into tomographic bins
    * - cluster_catalog_tomography
      - :ref:`Cluster catalogs`
      - Cluster redshift/richness binning, made by CLClusterBinningRedshiftRichness
    * - cluster_data_catalog
      - :ref:`Cluster catalogs`
      - Cluster data catalog, input to TXTwoPointCluster
    * - cluster_random_catalog
      - :ref:`Cluster catalogs`
      - Cluster random catalog, input to TXTwoPointCluster
    * - cut_catalog
      - :ref:`HDF File<Reading HDF5 Files>`
      - A subset of a catalog made by TXCutCatalog
    * - cut_shear_catalog
      - :ref:`HDF File<Reading HDF5 Files>`
      - A spatial or other subset of a shear catalog made by TXCutShearCatalog
    * - des_photometry_catalog
      - :ref:`Photometry Catalogs`
      - DES Y3 photometry, as ingested by TXIngestDESY3Gold
    * - des_shear_catalog
      - :ref:`Metacal catalogs`
      - DES Y3 metacal shear catalog, as ingested by TXIngestDESY3Shear
    * - desy3_index_catalog
      - :ref:`HDF File<Reading HDF5 Files>`
      - DES Y3 tomographic index, input to TXComputeShearCalibrationDESY3 (currently disabled)
    * - dp2_star_catalog
      - :ref:`Star catalogs`
      - Star catalog ingested from DP2
    * - injection_catalog
      - :ref:`HDF File<Reading HDF5 Files>`
      - Catalog of injected synthetic sources, made by the SSI ingestion stages
    * - matched_ssi_photometry_catalog
      - :ref:`Photometry Catalogs`
      - Photometry matched between injected and detected objects, for synthetic source injection
    * - shear_bin_assignments
      - :ref:`Tomographic bin assignments`
      - Tomographic bin assigned to each source catalog object by the tomography stage, before cuts
    * - shear_selection
      - :ref:`Catalog selections`
      - Boolean cuts defining the source sample, before tomographic binning
    * - ssi_detection_catalog
      - :ref:`HDF File<Reading HDF5 Files>`
      - Detection information for injected sources, made by TXIngestSSIDetectionDESBalrog
    * - ssi_photometry_catalog
      - :ref:`Photometry Catalogs`
      - Photometry of the injected field, made by TXIngestSSIGCR
    * - ssi_uninjected_photometry_catalog
      - :ref:`Photometry Catalogs`
      - Photometry of the same field without injected sources, made by TXIngestSSIGCR
    * - sub_selected_shear_data
      - :ref:`HDF File<Reading HDF5 Files>`
      - Random subset of the selected shear catalog objects, used to train the SOM photo-z method
    * - weights_catalog
      - :ref:`HDF File<Reading HDF5 Files>`
      - Lens weights, input to TXPhotozStack and TXTruePhotozStack


Map HDF Files
-------------

TXPipe map files are described in full under :ref:`Maps Files`.

.. list-table:: HDF5 Files
    :header-rows: 1

    * - File Tag
      - Kind
      - Description
    * - aux_lens_maps
      - :ref:`Auxiliary Lens Maps`
      - Maps related to the lens sample, most notably the depth
    * - aux_source_maps
      - :ref:`Auxiliary Source Maps`
      - Maps related to the source sample, such as PSF and weight
    * - convergence_maps
      - :ref:`Convergence Maps`
      - Reconstructed convergence maps, typically starting from the shear maps
    * - density_maps
      - :ref:`Density maps`
      - Over-density maps generated from lens number count maps
    * - input_lss_weight_maps
      - :ref:`Maps Files`
      - Weight maps used in GLASS simulations
    * - lens_maps
      - :ref:`Lens maps`
      - Weighted and raw number density maps of the source sample
    * - lens_noise_maps
      - :ref:`Lens Noise Maps`
      - Density and number count maps for random halves of the lens and density maps
    * - lss_weight_maps
      - :ref:`LSS Weight Maps`
      - Maps of weights for the lens sample
    * - mask
      - :ref:`Mask`
      - Binary or fractional pixel coverage masks
    * - source_maps
      - :ref:`Source maps`
      - Tomographic cosmic shear maps
    * - source_noise_maps
      - :ref:`Source Noise Maps`
      - Tomographic cosmic shear map realizations with all object shears radomly rotated
    * - aux_ssi_maps
      - :ref:`Maps Files`
      - Auxiliary maps for synthetic source injection
    * - cmb_lensing_map
      - :ref:`Maps Files`
      - CMB lensing convergence map
    * - density_masks
      - :ref:`Maps Files`
      - Density masks made by TXIngestQuaia
    * - des_footprint_maps
      - :ref:`Maps Files`
      - DES Y3 footprint maps
    * - desi_mask
      - :ref:`Maps Files`
      - Mask for the DESI sample
    * - flag_maps
      - :ref:`Maps Files`
      - Maps of object-flag fractions
    * - footprint_maps
      - :ref:`Maps Files`
      - Footprint maps, input to TXCustomMask
    * - mask1
      - :ref:`Maps Files`
      - First input mask to be combined by TXJointMask
    * - mask2
      - :ref:`Maps Files`
      - Second input mask to be combined by TXJointMask
    * - original_flagship_mask
      - :ref:`Maps Files`
      - Original mask of the Flagship simulation before conversion
    * - psf_maps
      - :ref:`Maps Files`
      - Maps of PSF properties in each tomographic bin, made by TXPSFMaps
    * - sel_func_pred_maps
      - :ref:`Maps Files`
      - Maps of the predicted density selection function
    * - shear_mask
      - :ref:`Maps Files`
      - Shear-sample mask, input to TXGenerateTractList and TXDESIJointMask
    * - survey_property_maps
      - :ref:`Maps Files`
      - Directory of survey property maps (e.g. DP1 observing conditions)


Photo-z HDF Files
----------------- 

The :ref:`Photo-z Files` page describes the types and formats of photometric redshift files in more detail.

.. list-table:: HDF5 Files
    :header-rows: 1

    * - File Tag
      - Kind
      - Description
    * - lens_photoz_pdfs
      - :ref:`Photo-z PDF Files`
      - Per-object PDFs for the lens sample
    * - lens_photoz_realizations
      - :ref:`Photo-z n(z) Files`
      - Per-tomographic bin photo-z realizations for the lens sample
    * - lens_photoz_stack
      - :ref:`Photo-z n(z) Files`
      - Mean tomographic bin photo-z for the lens sample
    * - shear_photoz_stack
      - :ref:`Photo-z n(z) Files`
      - Mean tomographic bin photo-z for the source sample
    * - source_photoz_pdfs
      - :ref:`Photo-z PDF Files`
      - Per-object PDFs for the source sample
    * - source_photoz_realizations
      - :ref:`Photo-z n(z) Files`
      - Per-tomographic bin photo-z realizations for the source sample



Miscellaneous HDF Files
-----------------------

Various miscellaneous HDF5 files with no common structure are also generated in the pipeline. See :ref:`the generic HDF5 page <Reading HDF5 Files>` for information on reading them.


.. list-table:: HDF5 Files
    :header-rows: 1

    * - File Tag
      - Description
    * - brighter_fatter_data
      - Measurements of PSF size and ellipticity mismatch as a function of magnitude
    * - density_shells
      - Simulation density shell maps when simulating log-normal maps with GLASS
    * - glass_cl_binned
      - Tomographic log-normal C_ell realizations from GLASS
    * - glass_cl_shells
      - Shell log-normal C_ell realizations from GLASS
    * - response_model
      - A model used for generating mock shear catalog calibration distributions.
    * - rowe_stats
      - Tabulation of the Rowe PSF statistics
    * - star_density_stats
      - Cross-correlation measurements between star and galaxy positions
    * - star_shear_stats
      - Cross-correlation measurements between star PSFs and galaxy shears
    * - tau_stats
      - Measurements of PSF Tau statistics
    * - tracer_metadata
      - Collected lens and source sample metadata.  See :ref:`Metadata`.
    * - input_photoz_stack
      - Photo-z n(z) ensemble input to TXPipe stages that modify it
    * - magnification_coefficients
      - Magnification bias coefficients, made by TXSSIMagnification
    * - moments_stats
      - PSF moment correlation statistics, made by TXPSFMomentCorr
    * - output_photoz_stack
      - Photo-z n(z) ensemble including the added non-tomographic n(z)
    * - response_in_bins
      - Shear response in bins of SNR and T/Tpsf, made by TXResponseInBins
    * - sel_func_model_info
      - Selection function model information, made by TXPredictDensitySelectionFunction
    * - unweighted_density_correlation
      - Density correlations with systematics maps, before applying weights
    * - weighted_density_correlation
      - Density correlations with systematics maps, after applying weights


SACC Files
-----------

SACC is a DESC library for storing cosmological measurements and all the metadata associated with them needed to perform parameter estimation. The TXPipe intefface class for SACC files is  `txpipe.data_types.SaccFile`, but you can also just use the SACC library directly.

.. list-table:: SACC Files
  :header-rows: 1

  * - File Tag
    - Description
  * - aperture_mass_data
    - Mass aperture statistic measurements
  * - gammat_bright_stars
    - Diagnostic measurements of the tangential shear around bright stars
  * - gammat_dim_stars
    - Diagnostic measurements of the tangential shear around bright stars
  * - gammat_field_center
    - Diagnostic measurements of the tangential shear around field centers
  * - gammat_randoms
    - Diagnostic measurements of the tangential shear around random positions
  * - summary_statistics_fourier
    - 3x2pt C_ell spectrum measurements with covariance, suitable for parameter estimation (may be blinded)
  * - summary_statistics_real
    - 3x2pt correlation function measurements with covariance, suitable for parameter estimation (may be blinded)
  * - twopoint_data_fourier
    - 3x2pt C_ell spectrum measurements with shot noise only  (may be blinded)
  * - twopoint_data_real
    - 3x2pt correlation function measurements with shot noise only  (may be blinded)
  * - twopoint_data_real_raw
    - Unblind 3x2pt correlation function measurements
  * - twopoint_gamma_x
    - Diagnostic measurement of cross-shear around lens sample
  * - twopoint_theory_fourier
    - Theory prediction for the 3x2pt C_ell spectrum based on fiducial cosmology
  * - twopoint_theory_real
    - Theory prediction for the 3x2pt correlation function based on fiducial cosmology
  * - cluster_sacc_catalog
    - Cluster counts data in SACC format
  * - cluster_twopoint_real
    - Cluster two-point measurements
  * - delta_sigma
    - Excess surface density Delta Sigma measurement
  * - delta_sigma_theory
    - Theory prediction for Delta Sigma
  * - filtered_squared_bispectrum
    - Filtered squared bispectrum, made by HOSFSB
  * - twopoint_data_SCIA
    - Two-point data made by the self-calibration IA extension
  * - twopoint_data_ext_cross_raw
    - Cross-correlation with external data, before blinding
  * - twopoint_data_fourier_cat
    - Catalog-based Fourier-space two-point data, made by TXTwoPointFourierCatalog
  * - twopoint_data_fourier_cmb_cross_density
    - CMB lensing - density cross-correlation
  * - twopoint_data_source_real_raw
    - Source real-space data made by TXTwoPointSourcePixel, before blinding
  * - twopoint_gamma_x_SCIA
    - Gamma_x measurement made by the self-calibration IA extension
  * - twopoint_source_gamma_x
    - Gamma_x measurement made by TXTwoPointSourcePixel


PNG Images
----------

Images are used for both quality diagnostics and plots of measured summary statistics. These would mostly require tweaking to be publication-ready.

.. list-table:: PNG Images
  :header-rows: 1

  * - File Tag
    - Description
  * - T_frac_psf_residual_hist
    - Histogram of fractional error in PSF size
  * - bright_object_map
    - Map of distribution of bright object counts
  * - brighter_fatter_plot
    - Brighter-fatter effect diagnostic plots
  * - convergence_map
    - Map plot of the reconstructed convergence
  * - density_cl
    - C_ell power spectrum of the density field
  * - density_xi
    - C_ell power spectrum of lensing
  * - density_xi_ratio
    - Ratio of C_ell lensing power spectrum to fiducial theory prediction
  * - depth_map
    - Map of estimated n (default 5) sigma depth in a selected band
  * - e1_psf_residual_hist
    - Histogram of PSF residuals in e1
  * - e2_psf_residual_hist
    - Histogram of PSF residuals in e2
  * - flag_map
    - Map of counts of flagged objects
  * - g1_hist
    - Histogram of shear g1
  * - g2_hist
    - Histogram of shear g2
  * - g_T
    - Trend of shear as a function of quadratic size T
  * - g_colormag
    - Trend of shear as a function of colors and magnitudes 
  * - g_psf_T
    - Trend of shear as a function of psf size
  * - g_psf_g
    - Trend of shear as a function of psf ellipticity 
  * - g_snr
    - Trend of shear as a function of galaxy signal-to-noise
  * - gammat_bright_stars_plot
    - Tangential shear around bright stars
  * - gammat_dim_stars_plot
    - Tangential shear around dim stars
  * - gammat_field_center_plot
    - Tangential shear around field centers
  * - gammat_randoms_plot
    - Tangential shear around random positions
  * - jk
    - Jack-knife regions used in shot noise estimation and 2pt measurement
  * - lens_mag_hist
    - Histogram of lens magnitudes
  * - lens_map
    - Map of lens sample counts
  * - lens_nz
    - Lens sample ensemble photometric redshift n(z)
  * - lens_photoz_realizations_plot
    - Realizations of lens sample photo-z
  * - lens_snr_hist
    - Histogram of lens sample signal-to-noise
  * - mask_map
    - Map image of mask values
  * - nz_lens
    - Lens sample ensemble photometric redshift n(z)
  * - nz_source
    - Source sample ensemble photometric redshift n(z)
  * - psf_map
    - Map of the mean PSF ellipticity
  * - response_hist
    - Histogram of the metadetection response values
  * - rowe0
    - First Rowe statistic zero measurement plot
  * - rowe134
    - Second set of Rowe stat measurement plots
  * - rowe25
    - Third set of Rowe stat measurements plots
  * - shearDensity_cl
    - Galaxy Galaxy-Lensing measured spectrum plot
  * - shearDensity_xi
    - Galaxy Galaxy-Lensing measured correlation plot
  * - shearDensity_xi_ratio
    - Galaxy Galaxy-Lensing measured correlation ratio to theory plot
  * - shearDensity_xi_x
    - Galaxy Galaxy-Lensing cross-shear diagnostic plot
  * - shear_cl_ee
    - Measured power spectrum E-mode plot
  * - shear_cl_ee_ratio
    - Measured power spectrum E-mode ratio to theory plot
  * - shear_map
    - g1 and g2 maps
  * - shear_xi_minus
    - Shear correlation function plot
  * - shear_xi_minus_ratio
    - Shear correlation function ratio plot
  * - shear_xi_plus
    - Shear correlation function plot
  * - shear_xi_plus_ratio
    - Shear correlation function ratio plot
  * - source_mag_hist
    - Histogram of magnitudes of source sample
  * - source_nz
    - Plot of source sample ensemble photometric redshift n(z)
  * - source_photoz_realizations_plot
    - Plot of realizations of the source sample ensemble photometric redshift n(z)
  * - source_snr_hist
    - Histogram of source sample signal-to-noise
  * - star_density_test
    - Galaxy-star density correlation function plot
  * - star_shear_test
    - Galaxy-star shear correlation function plot
  * - star_star_test
    - Star-star shear correlation function plot
  * - tau0
    - First PSF Tau statistic plot
  * - tau2
    - Second PSF Tau statistic plot
  * - tau5
    - Third PSF Tau statistic plot
  * - bright_object_map_plot
    - Map of bright objects
  * - delta_sigma_plot
    - Delta Sigma measurement plot
  * - depth_map_plot
    - Depth map
  * - depth_ssi_det_prob_map_plot
    - SSI detection-probability depth map
  * - depth_ssi_meas_map_plot
    - SSI measured depth map
  * - depth_ssi_true_map_plot
    - SSI true depth map
  * - focalplane_g
    - Mean shear across the focal plane
  * - lens_map_plot
    - Lens density maps
  * - magnification_plot
    - Magnification coefficient plot
  * - mask_map_plot
    - Mask map
  * - nz_plot
    - Tomographic n(z) plot
  * - psf_map_plot
    - PSF property maps
  * - response_in_bins_plot
    - Shear response in SNR/size bins, made by TXResponseInBins
  * - shear_map_plot
    - Shear maps
  * - tract_list_plot
    - Footprint of the DP2 tracts selected by TXGenerateTractList
  * - twopoint_data_fourier_cmb_cross_density_plot
    - CMB lensing - density cross-correlation plot


Text Files
----------

Smaller data items in TXPipe are sometimes stored in simple text formats.

.. list-table:: PNG Images
  :header-rows: 1

  * - File Tag
    - Description
  * - calibration_table
    - A secondary training set for tomographic binning
  * - g_T_out
    - Measurements of shear as a function of quadratic size T
  * - g_psf_T_out
    - Measurements of shear as a function of PSF size
  * - g_psf_g_out
    - Measurements of shear as a function of PSF shear
  * - g_snr_out
    - Measurements of shear as a function of signal-to-noise
  * - mock_shear_catalog
    - A mock shear catalog for testing
  * - patch_centers
    - Locations of jackknife patch centres
  * - rlens_measurement
    - Tangential shear in comoving coordinates
  * - cmb_cross_montecarlo_correction
    - Monte Carlo correction for the CMB cross-correlation
  * - cmb_lensing_beam
    - Beam of the CMB lensing map
  * - tract_list
    - List of DP2 tracts to ingest, made by TXGenerateTractList


YAML Files
----------

The YAML format is used for somewhat more structured key/value type data instead of plain text files.

.. list-table:: PNG Images
  :header-rows: 1

  * - File Tag
    - Description
  * - fiducial_cosmology
    - A representation of the fiducial cosmology loaded by CCL. Has its own `FiducialCosmology` class.
  * - star_psf_stats
    - PSF diagnostics for stars
  * - tracer_metadata_yml
    - Collected metadata for the lens and source samples


Pickle Files
------------

In a few miscellaneous cases, TXPipe uses the Python pickle format for storing data. We are generally trying to phase this out as it is not very interpretable, but RAIL makes extensive use of this for storing trained PZ models, because they are complex and varied.

.. list-table:: PNG Images
  :header-rows: 1

  * - File Tag
    - Description
  * - flow
    - Example model for simulating mock spectroscopic population
  * - lens_direct_calibration_model
    - Model for the NZDIR photo-z n(z) estimation for the lens sample
  * - lens_photoz_model
    - Model for BPZ photo-z estimation PDF p(z) estimation
  * - source_direct_calibration_model
    - Model for the NZDIR photo-z n(z) estimation for the source sample
  * - cluster_profiles
    - Cluster ensemble radial profiles
  * - shear_tomography_classifier
    - Trained random forest used to assign source galaxies to tomographic bins, made by TXTrainRandomForestTomography


Other Files and Directories
---------------------------

- lss_weight_summary - Directory
- map_systematic_correlations - Directory
- redmagic_catalog - FITS file
- ideal_specz_catalog - parquet file
- specz_catalog_pq - parquet file
- balrog_detection_catalog - FITS: DES Balrog detection catalog
- balrog_matched_catalog - FITS: DES Balrog matched catalog
- des_datavector_file - FITS: DES Y3 data vector
- des_specz_catalog - FITS: DES spectroscopic catalog
- desi_catalog - FITS: DESI catalog
- desi_catalog_selected - FITS: DESI catalog after selection
- desi_random_catalog - FITS: DESI random catalog
- lss_density_plots - Directory of LSS density plots
