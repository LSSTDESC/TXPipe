Diagnostics
===========

These stages compute and/or plot diagnostics of catalogs or other data

* :py:class:`~txpipe.diagnostics.TXDiagnosticQuantiles` - Measure quantiles of various values in the shear catalog.

* :py:class:`~txpipe.diagnostics.TXSourceDiagnosticPlots` - Make diagnostic plots of the shear catalog

* :py:class:`~txpipe.diagnostics.TXLensDiagnosticPlots` - Make diagnostic plots of the lens catalog

* :py:class:`~txpipe.psf_diagnostics.TXPSFDiagnostics` - Make histograms of PSF values

* :py:class:`~txpipe.psf_diagnostics.TXPSFMomentCorr` - Compute PSF Moments

* :py:class:`~txpipe.psf_diagnostics.TXTauStatistics` - Compute and plot PSF Tau statistics where the definition of Tau stats are eq.20-22

* :py:class:`~txpipe.psf_diagnostics.TXRoweStatistics` - Compute and plot PSF Rowe statistics

* :py:class:`~txpipe.psf_diagnostics.TXGalaxyStarShear` - Compute and plot star x galaxy and star x star correlations.

* :py:class:`~txpipe.psf_diagnostics.TXGalaxyStarDensity` - Compute and plot star x galaxy and star x star density correlations

* :py:class:`~txpipe.psf_diagnostics.TXBrighterFatterPlot` - Compute and plot a diagnostic of the brighter-fatter effect

* :py:class:`~txpipe.twopoint_null_tests.TXGammaTFieldCenters` - Make diagnostic 2pt measurements of tangential shear around field centers

* :py:class:`~txpipe.twopoint_null_tests.TXGammaTStars` - Make diagnostic 2pt measurements of tangential shear around stars

* :py:class:`~txpipe.twopoint_null_tests.TXGammaTRandoms` - Make diagnostic 2pt measurements of tangential shear around randoms

* :py:class:`~txpipe.twopoint_null_tests.TXApertureMass` - Measure the aperture mass statistics with TreeCorr

* :py:class:`~txpipe.spatial_diagnostics.TXFocalPlanePlot` - Make diagnostic plot of  mean measured ellipticity and residual ellipticity



.. autoclass:: txpipe.diagnostics.TXDiagnosticQuantiles

    **parallel**: Yes - Dask

.. autoclass:: txpipe.diagnostics.TXSourceDiagnosticPlots

    **parallel**: Yes - MPI

.. autoclass:: txpipe.diagnostics.TXLensDiagnosticPlots

    **parallel**: Yes - Dask

.. autoclass:: txpipe.psf_diagnostics.TXPSFDiagnostics

    **parallel**: Yes - MPI

.. autoclass:: txpipe.psf_diagnostics.TXPSFMomentCorr

    **parallel**: No - Serial

.. autoclass:: txpipe.psf_diagnostics.TXTauStatistics

    **parallel**: No - Serial

.. autoclass:: txpipe.psf_diagnostics.TXRoweStatistics

    **parallel**: No - Serial

.. autoclass:: txpipe.psf_diagnostics.TXGalaxyStarShear

    **parallel**: No - Serial

.. autoclass:: txpipe.psf_diagnostics.TXGalaxyStarDensity

    **parallel**: No - Serial

.. autoclass:: txpipe.psf_diagnostics.TXBrighterFatterPlot

    **parallel**: No - Serial

.. autoclass:: txpipe.twopoint_null_tests.TXGammaTFieldCenters

    **parallel**: Yes - MPI

.. autoclass:: txpipe.twopoint_null_tests.TXGammaTStars

    **parallel**: Yes - MPI

.. autoclass:: txpipe.twopoint_null_tests.TXGammaTRandoms

    **parallel**: Yes - MPI

.. autoclass:: txpipe.twopoint_null_tests.TXApertureMass

    **parallel**: Yes - MPI

.. autoclass:: txpipe.spatial_diagnostics.TXFocalPlanePlot

    **parallel**: No - Serial
