Scalar metadetect catalogs
==========================

Scalar metadetect catalogs are a simplified version of :ref:`Metadetect catalogs` used for the DESC DP2 metadetection catalogs, in which the response to shear is assumed to be a scalar multiple of the identity matrix, so that it is only necessary to apply shear in one direction. They are read with the ``ScalarMetaDetectCatalog`` class.

Only three variants are stored, because the shear response is the same in both directions: "ns" (no shear), "1p" and "1m". As for the full metadetect catalogs, each variant is a separate sub-group of the "shear" group containing a complete catalog with its own number of objects. The group has the attribute ``catalog_type = "scalar_metadetect"``; if it is missing the type is identified by the presence of a "1p" sub-group without a "2p" one.

The selection stages are :py:class:`~txpipe.source_selection.TXScalarMetaDetectSelection` and :py:class:`~txpipe.source_selection.TXDP2DESCSelection`, and the calibration stage is :py:class:`~txpipe.shear_calibration.scalar_metadetect.TXComputeShearCalibrationScalarMetadetect`.

The columns in each variant of the example catalog are:

===============  =============  ==========  ==================================================
Group            Name           Kind        Meaning
===============  =============  ==========  ==================================================
shear/{variant}  T              1D float64  Measured object size, T = Ixx + Iyy (arcsec^2)
shear/{variant}  T_err          1D float64  Uncertainty on T
shear/{variant}  dec            1D float64  Declination (degrees)
shear/{variant}  flags          1D int64    Measurement flags; zero means no problems
shear/{variant}  g1             1D float64  First ellipticity component (uncalibrated)
shear/{variant}  g2             1D float64  Second ellipticity component (uncalibrated)
shear/{variant}  id             1D int64    Unique object ID
shear/{variant}  mag_err_i      1D float64  Uncertainty on i-band magnitude
shear/{variant}  mag_err_r      1D float64  Uncertainty on r-band magnitude
shear/{variant}  mag_err_z      1D float64  Uncertainty on z-band magnitude
shear/{variant}  mag_i          1D float64  i-band magnitude
shear/{variant}  mag_r          1D float64  r-band magnitude
shear/{variant}  mag_z          1D float64  z-band magnitude
shear/{variant}  psf_T_mean     1D float64  PSF size at the object position
shear/{variant}  psf_g1         1D float64  PSF ellipticity component 1 at the object position
shear/{variant}  psf_g2         1D float64  PSF ellipticity component 2 at the object position
shear/{variant}  ra             1D float64  Right ascension (degrees)
shear/{variant}  redshift_true  1D float64  True redshift (simulations only)
shear/{variant}  s2n            1D float64  Measured signal-to-noise ratio
shear/{variant}  true_g1        1D float64  True input shear 1 (simulations only)
shear/{variant}  true_g2        1D float64  True input shear 2 (simulations only)
shear/{variant}  weight         1D float64  Shape measurement weight
===============  =============  ==========  ==================================================
