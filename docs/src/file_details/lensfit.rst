Lensfit catalogs
================

Lensfit is the shear measurement method used in KiDS and CFHTLenS. In TXPipe it is used in re-analysis of KiDS data. TXPipe reads these catalogs with the LensfitShearCatalog class.

Lensfit has a simpler calibration scheme than metacal or metadetect and does not involve variant catalogs: calibration uses the per-object multiplicative bias m, and additive terms c1 and c2, which are averaged over each tomographic bin. The additive terms are different in the north and south of the survey, so the declination is also needed (this is what :py:meth:`~txpipe.data_types.ShearCatalog.get_extra_calibration_columns` returns).

The selection stage is :py:class:`~txpipe.source_selection.TXLensfitSelection` and the calibration stage is :py:class:`~txpipe.shear_calibration.lensfit.TXComputeShearCalibrationLensfit`.

=====  ============  ==========  ==================================================
Group  Name          Kind        Meaning
=====  ============  ==========  ==================================================
shear  T             1D float32  Measured object size, T = Ixx + Iyy (arcsec^2)
shear  c1            1D int64    Additive bias term, component 1
shear  c2            1D int64    Additive bias term, component 2
shear  dec           1D float64  Declination (degrees)
shear  extendedness  1D int64    Star/galaxy classifier flag
shear  flags         1D int16    Measurement flags; zero means no problems
shear  g1            1D float32  First ellipticity component (uncalibrated)
shear  g2            1D float32  Second ellipticity component (uncalibrated)
shear  m             1D int64    Multiplicative bias for the object
shear  mag_err_i     1D float32  Uncertainty on i-band magnitude
shear  mag_err_r     1D float32  Uncertainty on r-band magnitude
shear  mag_i         1D float32  i-band magnitude
shear  mag_r         1D float32  r-band magnitude
shear  mean_z        1D float64  Mean photometric redshift
shear  objectId      1D int32    Unique object ID
shear  psf_T_mean    1D float32  PSF size at the object position
shear  psf_g1        1D float32  PSF ellipticity component 1 at the object position
shear  psf_g2        1D float32  PSF ellipticity component 2 at the object position
shear  ra            1D float64  Right ascension (degrees)
shear  region        1D int32    Survey region index
shear  s2n           1D float32  Measured signal-to-noise ratio
shear  sigma_e       1D int64    Per-object ellipticity uncertainty
shear  snr_i         1D float32  i-band signal-to-noise
shear  snr_r         1D float32  r-band signal-to-noise
shear  tilename      1D int32    Survey tile index
shear  weight        1D float32  Shape measurement weight
=====  ============  ==========  ==================================================
