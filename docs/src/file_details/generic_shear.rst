Generic shear catalogs
======================

Generic shear catalogs are a type of shear catalog that require no calibration. They are used for testing and development purposes, for mock catalogs, and for catalogs such as the DP1 ingestion output where calibration is not applied. They are read with the ``SimpleShearCatalog`` class.

Everything is stored in a single "shear" group, which should have the attribute ``catalog_type = "simple"``; TXPipe cannot identify this type of catalog from its columns alone, so the attribute is needed.

The selection stage is :py:class:`~txpipe.source_selection.TXSimpleSelection` and the calibration stage is :py:class:`~txpipe.shear_calibration.simple.TXComputeShearCalibrationSimple`; the latter just counts objects and does not compute any response.

The columns in a generic shear catalog are:

=======  =================  ==========  =========
Group    Name               Kind        Meaning
=======  =================  ==========  =========
shear                       Group
shear    T                  1D float64
shear    T_err              1D float64
shear    dec                1D float64
shear    flags              1D float64
shear    g1                 1D float64
shear    g2                 1D float64
shear    galaxy_id          1D int64
shear    psf_T_mean         1D float64
shear    psf_g1             1D float64
shear    psf_g2             1D float64
shear    ra                 1D float64
shear    redshift_true      1D float64
shear    s2n                1D float64
shear    weight             1D float64
=======  =================  ==========  =========

