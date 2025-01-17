TXPipe File Tags
================

TXPipe saves everything it does in files saved in the `output_dir` directory specified in the pipeline YAML file.

This page describes all the files that TXPipe currently generates and uses.

TXPipe files are organized by `tags`, which are usually just the stem of the file name (i.e. without the directory or suffix). The tags input and output by each staged are defined in the stage class. Each file has a specific :ref:`type<File Types>`

Aliases
-------

Any stage can be "aliased" to a different tag when the pipeline is defined in the pipeline YAML file. This means that the name used for the stage internally can be different from the name used in the file names. This is useful when there are pipeline stages that can be run multiple times on different data sets. 

For example, the `TXPhotozPlot` class can be run twice, once on the source sample and once on the lens sample, to generate n(z) plots of each.

Catalog HDF Files
-----------------

Map HDF Files
-------------

HDF5 Files
----------
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
    * - brighter_fatter_data
      - :ref:`HDF File<Reading HDF5 Files>`
      - Measurements of PSF size and ellipticity mismatch as a function of magnitude
    * - cluster_catalog
      - :ref:`Cluster catalogs`
      - Locations, redshifts, and richness of clusters
    * - cluster_shear_catalogs
      - :ref:`Cluster Shear Catalogs`
      - An catalog of shear values around clusters.
    * - convergence_maps
      - :ref:`Convergence Maps`
      - Reconstructed convergence maps, typically starting from the shear maps
    * - density_maps
      - :ref:`Density maps`
      - Over-density maps generated from lens number count maps
    * - density_shells
      - :ref:`HDF File<Reading HDF5 Files>`
      - Simulation density shell maps when simulating log-normal maps with GLASS
    * - exposures
      - :ref:`Exposure catalogs`
      - Catalogs centers of exposurs for use in systematics tests
    * - glass_cl_binned
      - :ref:`HDF File<Reading HDF5 Files>`
      - Tomographic log-normal C_ell realizations from GLASS
    * - glass_cl_shells
      - :ref:`HDF File<Reading HDF5 Files>`
      - Shell log-normal C_ell realizations from GLASS
    * - input_lss_weight_maps
      - :ref:`Maps Files`
      - Weight maps used in GLASS simulations
    * - lens_catalog
      - :ref:`HDF File<Reading HDF5 Files>`
      - A catalog of objects to be used as lenses (when something external is used instead of the photometry catalog
    * - lens_maps
      - :ref:`Lens maps`
      - Weighted and raw number density maps of the source sample
    * - lens_noise_maps
      - :ref:`Lens Noise Maps`
      - Density and number count maps for random halves of the lens and density maps
    * - lens_photoz_pdfs
      - :ref:`Photo-z PDF Files`
      - Per-object PDFs for the lens sample
    * - lens_photoz_realizations
      - :ref:`Photo-z n(z) Files`
      - Per-tomographic bin photo-z realizations for the lens sample
    * - lens_photoz_stack
      - :ref:`Photo-z n(z) Files`
      - Mean tomographic bin photo-z for the lens sample
    * - lens_tomography_catalog
      - :ref:`Lens tomography catalogs`
      - Tomographic selection information for the lens sample
    * - lens_tomography_catalog_unweighted
      - :ref:`Lens tomography catalogs`
      - Tomographic selection information for the lens sample, without weights
    * - lss_weight_maps
      - :ref:`LSS Weight Maps`
      - Maps of weights for the lens sample
    * - mask
      - :ref:`Mask`
      - Binary or fractional pixel coverage masks
    * - photometry_catalog
      - :ref:`Photometry Catalogs`
      - Photometric measurements from which the lens sample is chosen
    * - random_cats
      - :ref:`Random Catalogs`
      - A catalog of random objects following the same tomographic and location selection as the lens sample but with no underlying structure
    * - response_model
      - :ref:`HDF File<Reading HDF5 Files>`
      - A model used for generating mock shear catalog calibration distributions.
    * - rowe_stats
      - :ref:`HDF File<Reading HDF5 Files>`
      - Tabulation of the Rowe PSF statistics
    * - shear_catalog
      - :ref:`Shear Catalogs<Overall Pipeline inputs>`
      - Shear catalogs of various different types
    * - shear_catalog_quantiles
      - :ref:`HDF File<Reading HDF5 Files>`
      - Measurements of quantiles of shear catalog columns such as SNR, size, etc.
    * - shear_photoz_stack
      - :ref:`Photo-z n(z) Files`
      - Mean tomographic bin photo-z for the source sample
    * - shear_tomography_catalog
      - :ref:`Shear tomography catalogs`
      - Tomographic selection and shear calibration information for the source sample
    * - source_maps
      - :ref:`Source maps`
      - Tomographic cosmic shear maps
    * - source_noise_maps
      - :ref:`Source Noise Maps`
      - Tomographic cosmic shear map realizations with all object shears radomly rotated
    * - source_photoz_pdfs
      - :ref:`Photo-z PDF Files`
      - Per-object PDFs for the source sample
    * - source_photoz_realizations
      - 
      - 
    * - spectroscopic_catalog
      - 
      - 
    * - star_catalog
      - 
      - 
    * - star_density_stats
      - 
      - 
    * - star_shear_stats
      - 
      - 
    * - tau_stats
      - 
      - 
    * - tracer_metadata
      - 
      - 

SACC Files
-----------
.. list-table:: SACC Files
  :header-rows: 1

  * - File Tag
    - Description
  * - aperture_mass_data
    - 
  * - gammat_bright_stars
    - 
  * - gammat_dim_stars
    - 
  * - gammat_field_center
    - 
  * - gammat_randoms
    - 
  * - summary_statistics_fourier
    - 
  * - summary_statistics_real
    - 
  * - twopoint_data_fourier
    - 
  * - twopoint_data_real
    - 
  * - twopoint_data_real_raw
    - 
  * - twopoint_gamma_x
    - 
  * - twopoint_theory_fourier
    - 
  * - twopoint_theory_real
    - 


PNG Images
----------

.. list-table:: PNG Images
  :header-rows: 1

  * - File Tag
    - Description
  * - T_frac_psf_residual_hist
    - 
  * - bright_object_map
    - 
  * - brighter_fatter_plot
    - 
  * - convergence_map
    - 
  * - density_cl
    - 
  * - density_xi
    - 
  * - density_xi_ratio
    - 
  * - depth_map
    - 
  * - e1_psf_residual_hist
    - 
  * - e2_psf_residual_hist
    - 
  * - flag_map
    - 
  * - g1_hist
    - 
  * - g2_hist
    - 
  * - g_T
    - 
  * - g_colormag
    - 
  * - g_psf_T
    - 
  * - g_psf_g
    - 
  * - g_snr
    - 
  * - gammat_bright_stars_plot
    - 
  * - gammat_dim_stars_plot
    - 
  * - gammat_field_center_plot
    - 
  * - gammat_randoms_plot
    - 
  * - jk
    - 
  * - lens_mag_hist
    - 
  * - lens_map
    - 
  * - lens_nz
    - 
  * - lens_photoz_realizations_plot
    - 
  * - lens_snr_hist
    - 
  * - mask_map
    - 
  * - nz_lens
    - 
  * - nz_source
    - 
  * - psf_map
    - 
  * - response_hist
    - 
  * - rowe0
    - 
  * - rowe134
    - 
  * - rowe25
    - 
  * - shearDensity_cl
    - 
  * - shearDensity_xi
    - 
  * - shearDensity_xi_ratio
    - 
  * - shearDensity_xi_x
    - 
  * - shear_cl_ee
    - 
  * - shear_cl_ee_ratio
    - 
  * - shear_map
    - 
  * - shear_xi_minus
    - 
  * - shear_xi_minus_ratio
    - 
  * - shear_xi_plus
    - 
  * - shear_xi_plus_ratio
    - 
  * - source_mag_hist
    - 
  * - source_nz
    - 
  * - source_photoz_realizations_plot
    - 
  * - source_snr_hist
    - 
  * - star_density_test
    - 
  * - star_shear_test
    - 
  * - star_star_test
    - 
  * - tau0
    - 
  * - tau2
    - 
  * - tau5
    - 


Text Files
----------

.. list-table:: PNG Images
  :header-rows: 1

  * - File Tag
    - Description
  * - calibration_table
    - 
  * - g_T_out
    - 
  * - g_psf_T_out
    - 
  * - g_psf_g_out
    - 
  * - g_snr_out
    - 
  * - mock_shear_catalog
    - 
  * - patch_centers
    - 
  * - rlens_measurement
    - 

YAML Files
----------

.. list-table:: PNG Images
  :header-rows: 1

  * - fiducial_cosmology
    -
  * - star_psf_stats
    -
  * - tracer_metadata_yml
    -


Pickle Files
------------

.. list-table:: PNG Images
  :header-rows: 1

  * - flow
    - 
  * - lens_direct_calibration_model
    - 
  * - lens_photoz_model
    - 
  * - source_direct_calibration_model
    - 


Other Files and Directories
---------------------------

lss_weight_summary - Directory
map_systematic_correlations - Directory
redmagic_catalog - FITS file
ideal_specz_catalog - parquet file
specz_catalog_pq - parquet file
