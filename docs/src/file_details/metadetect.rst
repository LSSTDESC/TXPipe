Metadetect catalogs
===================

Metadetect catalogs are used for Rubin/LSST data (including DP1 and DP2 products) and for simulations of them. TXPipe reads them with the ``MetaDetectShearCatalog`` class.

Metadetect runs the whole detection and measurement process on the image five times: once on the original image (the "ns", for "no shear", version) and once for each of four versions of the image to which a small artificial shear has been applied: "1p", "1m", "2p" and "2m". Because detection is repeated, **each variant is a complete catalog in its own right, and the variants generally have different numbers of objects**. The sheared variants are used to calculate calibration factors that account for shear measurement and *selection* biases.

Each variant is stored in its own sub-group of the "shear" group, named for the variant. The "shear" group has a ``bands`` attribute listing the photometric bands, and optionally ``catalog_type = "metadetect"``. If the attribute is missing TXPipe identifies the catalog type from the presence of a "2p" sub-group.

Because the variants have different lengths, every stage that works with metadetect catalogs reads one variant at a time. The source selection, tomographic bin assignments, and the "bin" columns of the shear tomography catalog all follow the same layout, with one dataset per variant: see :ref:`Catalog selections`, :ref:`Tomographic bin assignments` and :ref:`Shear tomography catalogs`.

The selection stages for metadetect catalogs are :py:class:`~txpipe.source_selection.TXMetaDetectSelection` and the Rubin/DESC-specific versions :py:class:`~txpipe.source_selection.TXDP2RubinMetadetectSelection` and :py:class:`~txpipe.source_selection.TXDP2DESCSelection`. The calibration stage is :py:class:`~txpipe.shear_calibration.metadetect.TXComputeShearCalibrationMetadetect`.

The columns in each variant of the example metadetect catalog are shown below. Real Rubin catalogs have additional columns such as ``mfrac`` (the masked fraction of the object), ``is_primary``, and per-fit flags; the ingestion stages in :ref:`Rubin DP2 Ingestion` list what they write.

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
