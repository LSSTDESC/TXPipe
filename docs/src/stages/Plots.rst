Plots
=====

These stages make plots out TXPipe output data

* :py:class:`~txpipe.photoz_stack.TXPhotozPlot` - Make n(z) plots of source and lens galaxies

* :py:class:`~txpipe.map_plots.TXMapPlots` - Make plots of all the available maps

* :py:class:`~txpipe.convergence.TXConvergenceMapPlots` - Make plots convergence maps

* :py:class:`~txpipe.rail.summarize.PZRealizationsPlot` - Stage PZRealizationsPlot

* :py:class:`~txpipe.twopoint_plots.TXTwoPointPlots` - Make plots of the correlation functions and their ratios to theory

* :py:class:`~txpipe.twopoint_plots.TXTwoPointPlotsFourier` - Make plots of the C_ell and their ratios to theory

* :py:class:`~txpipe.twopoint_plots.TXTwoPointPlotsTheory` - Stage TXTwoPointPlotsTheory



.. autotxclass:: txpipe.photoz_stack.TXPhotozPlot
    :members:
    :exclude-members: run

    Inputs: 

    - photoz_stack: QPNOfZFile

    Outputs: 

    - nz_plot: PNGFile
    
    Parallel: No - Serial


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>label</strong>: (str) Default=. Label for the n(z) plot.</LI>
            <LI><strong>zmax</strong>: (float) Default=3.0. Maximum redshift for plotting.</LI>
            </UL>



.. autotxclass:: txpipe.map_plots.TXMapPlots
    :members:
    :exclude-members: run

    Inputs: 

    - source_maps: MapsFile
    - lens_maps: MapsFile
    - density_maps: MapsFile
    - mask: MapsFile
    - aux_source_maps: MapsFile
    - aux_lens_maps: MapsFile

    Outputs: 

    - depth_map: PNGFile
    - lens_map: PNGFile
    - shear_map: PNGFile
    - flag_map: PNGFile
    - psf_map: PNGFile
    - mask_map: PNGFile
    - bright_object_map: PNGFile
    
    Parallel: No - Serial


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>projection</strong>: (str) Default=cart. Projection type for map plots (e.g., cart, moll)</LI>
            <LI><strong>rot180</strong>: (bool) Default=False. Whether to rotate the map by 180 degrees</LI>
            <LI><strong>debug</strong>: (bool) Default=False. Enable debug mode for plotting</LI>
            </UL>



.. autotxclass:: txpipe.convergence.TXConvergenceMapPlots
    :members:
    :exclude-members: run

    Inputs: 

    - convergence_maps: MapsFile

    Outputs: 

    - convergence_map: PNGFile
    
    Parallel: No - Serial


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>projection</strong>: (str) Default=cart. Projection type for convergence map plots (e.g., cart, moll, orth).</LI>
            </UL>



.. autotxclass:: txpipe.rail.summarize.PZRealizationsPlot
    :members:
    :exclude-members: run

    Inputs: 

    - photoz_realizations: QPMultiFile

    Outputs: 

    - photoz_realizations_plot: PNGFile
    
    Parallel: No - Serial


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>zmax</strong>: (float) Default=3.0. Maximum redshift for plotting.</LI>
            <LI><strong>nz</strong>: (int) Default=301. Number of redshift bins for plotting.</LI>
            </UL>



.. autotxclass:: txpipe.twopoint_plots.TXTwoPointPlots
    :members:
    :exclude-members: run

    Inputs: 

    - twopoint_data_real: SACCFile
    - twopoint_gamma_x: SACCFile

    Outputs: 

    - shear_xi_plus: PNGFile
    - shear_xi_minus: PNGFile
    - shearDensity_xi: PNGFile
    - density_xi: PNGFile
    - shearDensity_xi_x: PNGFile
    
    Parallel: No - Serial


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>wspace</strong>: (float) Default=0.05. Width space between subplots.</LI>
            <LI><strong>hspace</strong>: (float) Default=0.05. Height space between subplots.</LI>
            </UL>



.. autotxclass:: txpipe.twopoint_plots.TXTwoPointPlotsFourier
    :members:
    :exclude-members: run

    Inputs: 

    - summary_statistics_fourier: SACCFile
    - twopoint_theory_fourier: SACCFile

    Outputs: 

    - shear_cl_ee: PNGFile
    - shearDensity_cl: PNGFile
    - density_cl: PNGFile
    - shear_cl_ee_ratio: PNGFile
    
    Parallel: No - Serial


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>wspace</strong>: (float) Default=0.05. Width space between subplots.</LI>
            <LI><strong>hspace</strong>: (float) Default=0.05. Height space between subplots.</LI>
            </UL>



.. autotxclass:: txpipe.twopoint_plots.TXTwoPointPlotsTheory
    :members:
    :exclude-members: run

    Inputs: 

    - twopoint_data_real: SACCFile
    - twopoint_gamma_x: SACCFile
    - twopoint_theory_real: SACCFile

    Outputs: 

    - shear_xi_plus: PNGFile
    - shear_xi_minus: PNGFile
    - shearDensity_xi: PNGFile
    - density_xi: PNGFile
    - shear_xi_plus_ratio: PNGFile
    - shear_xi_minus_ratio: PNGFile
    - shearDensity_xi_ratio: PNGFile
    - density_xi_ratio: PNGFile
    - shearDensity_xi_x: PNGFile
    
    Parallel: No - Serial


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>wspace</strong>: (float) Default=0.05. Width space between subplots.</LI>
            <LI><strong>hspace</strong>: (float) Default=0.05. Height space between subplots.</LI>
            </UL>


