HSC catalogs
============

HSC catalogs follow the conventions of the Hyper Suprime-Cam survey shear catalogs, and are read with the ``HSCShearCatalog`` class. The calibration scheme is similar to the one used for :ref:`Lensfit catalogs`: there are no variant catalogs, and instead each object has additive terms and, optionally, a per-object multiplicative correction that is averaged over each tomographic bin.

All the data is in a single group called "shear" which should have the attribute ``catalog_type = "hsc"``. As well as the standard columns used for selection (``ra``, ``dec``, ``g1``, ``g2``, ``weight``, ``T``, ``s2n``, ``psf_T_mean``, ``flags``, and magnitudes ``mag_{band}``) it has the columns:

=======  =========  ==========  ============================================================
Group    Name       Kind        Meaning
=======  =========  ==========  ============================================================
shear    c1         1D float    Additive bias term, component 1
shear    c2         1D float    Additive bias term, component 2
shear    aselepsf1  1D float    Optional. PSF-related selection-bias term, component 1
shear    aselepsf2  1D float    Optional. PSF-related selection-bias term, component 2
shear    msel       1D float    Optional. Selection bias correction
=======  =========  ==========  ============================================================

The three optional columns are used if they are present. They are listed by :py:meth:`~txpipe.data_types.ShearCatalog.get_extra_calibration_columns`.

The selection stage is :py:class:`~txpipe.source_selection.TXHSCSelection`, which adds an optional maximum-shear cut to the standard cuts, and the calibration stage is :py:class:`~txpipe.shear_calibration.hsc.TXComputeShearCalibrationHSC`.

.. warning::

    The HSC calibration is currently marked as incomplete in the code (see the docstring of ``TXComputeShearCalibrationHSC``) and may fail when computing the final output statistics.
