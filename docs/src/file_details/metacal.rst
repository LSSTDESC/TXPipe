Metacal catalogs
================

Metacalibration catalogs are used in TXPipe for Dark Energy Survey Y1 and Y3 data and for DC2-based simulations. TXPipe reads them with the ``MetacalShearCatalog`` class (see :ref:`Data Type Classes`).

Each quantity is measured under 5 different conditions: the nominal measurement, and four variants that are used to calculate calibration factors that account for shear measurement and selection biases. The variants are labelled "1m", "1p", "2m", "2p", and are given as a suffix on the column name (for example ``g1_1p``). Quantities that are not re-measured under shear, such as ``ra``, ``dec`` and the PSF values, have only the one version.

Everything is stored in a single group called "shear". The group has an attribute ``bands`` listing the photometric bands, and (optionally) ``catalog_type = "metacal"``. If the attribute is missing TXPipe identifies the catalog type from the presence of a ``g1_1p`` column.

The source selection stage is :py:class:`~txpipe.source_selection.TXMetacalSelection`, and the calibration stage is :py:class:`~txpipe.shear_calibration.metacal.TXComputeShearCalibrationMetacal`. See :ref:`Shear calibration guide`.

The columns, for an example three-band (z, i, r) catalog, are:

=====  =============  ==========  ==================================================================================
Group  Name           Kind        Meaning
=====  =============  ==========  ==================================================================================
shear  T              1D float64  Measured object size, T = Ixx + Iyy (arcsec^2)
shear  T_1m           1D float64  Measured object size, T = Ixx + Iyy (arcsec^2), measured in the 1m sheared variant
shear  T_1p           1D float64  Measured object size, T = Ixx + Iyy (arcsec^2), measured in the 1p sheared variant
shear  T_2m           1D float64  Measured object size, T = Ixx + Iyy (arcsec^2), measured in the 2m sheared variant
shear  T_2p           1D float64  Measured object size, T = Ixx + Iyy (arcsec^2), measured in the 2p sheared variant
shear  T_err          1D float64  Uncertainty on T
shear  T_err_1m       1D float64  Uncertainty on T, measured in the 1m sheared variant
shear  T_err_1p       1D float64  Uncertainty on T, measured in the 1p sheared variant
shear  T_err_2m       1D float64  Uncertainty on T, measured in the 2m sheared variant
shear  T_err_2p       1D float64  Uncertainty on T, measured in the 2p sheared variant
shear  dec            1D float64  Declination (degrees)
shear  flags          1D int64    Measurement flags; zero means no problems
shear  flags_1m       1D int64    Measurement flags; zero means no problems, measured in the 1m sheared variant
shear  flags_1p       1D int64    Measurement flags; zero means no problems, measured in the 1p sheared variant
shear  flags_2m       1D int64    Measurement flags; zero means no problems, measured in the 2m sheared variant
shear  flags_2p       1D int64    Measurement flags; zero means no problems, measured in the 2p sheared variant
shear  g1             1D float64  First ellipticity component (uncalibrated)
shear  g1_1m          1D float64  First ellipticity component (uncalibrated), measured in the 1m sheared variant
shear  g1_1p          1D float64  First ellipticity component (uncalibrated), measured in the 1p sheared variant
shear  g1_2m          1D float64  First ellipticity component (uncalibrated), measured in the 2m sheared variant
shear  g1_2p          1D float64  First ellipticity component (uncalibrated), measured in the 2p sheared variant
shear  g2             1D float64  Second ellipticity component (uncalibrated)
shear  g2_1m          1D float64  Second ellipticity component (uncalibrated), measured in the 1m sheared variant
shear  g2_1p          1D float64  Second ellipticity component (uncalibrated), measured in the 1p sheared variant
shear  g2_2m          1D float64  Second ellipticity component (uncalibrated), measured in the 2m sheared variant
shear  g2_2p          1D float64  Second ellipticity component (uncalibrated), measured in the 2p sheared variant
shear  id             1D int64    Unique object ID
shear  mag_err_i      1D float64  Uncertainty on i-band magnitude
shear  mag_err_i_1m   1D float64  Uncertainty on i-band magnitude, measured in the 1m sheared variant
shear  mag_err_i_1p   1D float64  Uncertainty on i-band magnitude, measured in the 1p sheared variant
shear  mag_err_i_2m   1D float64  Uncertainty on i-band magnitude, measured in the 2m sheared variant
shear  mag_err_i_2p   1D float64  Uncertainty on i-band magnitude, measured in the 2p sheared variant
shear  mag_err_r      1D float64  Uncertainty on r-band magnitude
shear  mag_err_r_1m   1D float64  Uncertainty on r-band magnitude, measured in the 1m sheared variant
shear  mag_err_r_1p   1D float64  Uncertainty on r-band magnitude, measured in the 1p sheared variant
shear  mag_err_r_2m   1D float64  Uncertainty on r-band magnitude, measured in the 2m sheared variant
shear  mag_err_r_2p   1D float64  Uncertainty on r-band magnitude, measured in the 2p sheared variant
shear  mag_err_z      1D float64  Uncertainty on z-band magnitude
shear  mag_err_z_1m   1D float64  Uncertainty on z-band magnitude, measured in the 1m sheared variant
shear  mag_err_z_1p   1D float64  Uncertainty on z-band magnitude, measured in the 1p sheared variant
shear  mag_err_z_2m   1D float64  Uncertainty on z-band magnitude, measured in the 2m sheared variant
shear  mag_err_z_2p   1D float64  Uncertainty on z-band magnitude, measured in the 2p sheared variant
shear  mag_i          1D float64  i-band magnitude
shear  mag_i_1m       1D float64  i-band magnitude, measured in the 1m sheared variant
shear  mag_i_1p       1D float64  i-band magnitude, measured in the 1p sheared variant
shear  mag_i_2m       1D float64  i-band magnitude, measured in the 2m sheared variant
shear  mag_i_2p       1D float64  i-band magnitude, measured in the 2p sheared variant
shear  mag_r          1D float64  r-band magnitude
shear  mag_r_1m       1D float64  r-band magnitude, measured in the 1m sheared variant
shear  mag_r_1p       1D float64  r-band magnitude, measured in the 1p sheared variant
shear  mag_r_2m       1D float64  r-band magnitude, measured in the 2m sheared variant
shear  mag_r_2p       1D float64  r-band magnitude, measured in the 2p sheared variant
shear  mag_z          1D float64  z-band magnitude
shear  mag_z_1m       1D float64  z-band magnitude, measured in the 1m sheared variant
shear  mag_z_1p       1D float64  z-band magnitude, measured in the 1p sheared variant
shear  mag_z_2m       1D float64  z-band magnitude, measured in the 2m sheared variant
shear  mag_z_2p       1D float64  z-band magnitude, measured in the 2p sheared variant
shear  psf_T_mean     1D float64  PSF size at the object position
shear  psf_g1         1D float64  PSF ellipticity component 1 at the object position
shear  psf_g2         1D float64  PSF ellipticity component 2 at the object position
shear  ra             1D float64  Right ascension (degrees)
shear  redshift_true  1D float64  True redshift (simulations only)
shear  s2n            1D float64  Measured signal-to-noise ratio
shear  s2n_1m         1D float64  Measured signal-to-noise ratio, measured in the 1m sheared variant
shear  s2n_1p         1D float64  Measured signal-to-noise ratio, measured in the 1p sheared variant
shear  s2n_2m         1D float64  Measured signal-to-noise ratio, measured in the 2m sheared variant
shear  s2n_2p         1D float64  Measured signal-to-noise ratio, measured in the 2p sheared variant
shear  true_g1        1D float64  True input shear 1 (simulations only)
shear  true_g2        1D float64  True input shear 2 (simulations only)
shear  weight         1D float64  Shape measurement weight
shear  weight_1m      1D float64  Shape measurement weight, measured in the 1m sheared variant
shear  weight_1p      1D float64  Shape measurement weight, measured in the 1p sheared variant
shear  weight_2m      1D float64  Shape measurement weight, measured in the 2m sheared variant
shear  weight_2p      1D float64  Shape measurement weight, measured in the 2p sheared variant
=====  =============  ==========  ==================================================================================
