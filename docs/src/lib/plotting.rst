Plotting Utilities
==================

These are plotting tools used in several places in TXPipe.

The main function is ``full_3x2pt_plots``, which makes the standard set of correlation function and power spectrum plots from one or more SACC files, optionally with a theory prediction or as ratios to it. Real-space statistics are shown as theta * xi on a linear scale, Fourier-space statistics on log-log axes, and the ratio plots compare data to theory. The plotting for each is done by ``make_real_space_plot``, ``make_fourier_space_plot`` and ``make_ratio_plot``, which choose the axis limits and which panels to show from tables at the top of the module.


.. automodule:: txpipe.plotting.correlations
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: txpipe.plotting.histogram
   :members:
   :undoc-members:
   :show-inheritance: