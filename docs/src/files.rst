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


Other Files and Directories
---------------------------

- lss_weight_summary - Directory
- map_systematic_correlations - Directory
- redmagic_catalog - FITS file
- ideal_specz_catalog - parquet file
- specz_catalog_pq - parquet file
