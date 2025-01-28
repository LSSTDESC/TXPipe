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

    Parallel: No - Serial

    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>label</strong>: (str) Default=. </LI>
            <LI><strong>zmax</strong>: (float) Default=3.0. </LI>
            </UL>



.. autotxclass:: txpipe.map_plots.TXMapPlots
    :members:
    :exclude-members: run

    Parallel: No - Serial

    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>projection</strong>: (str) Default=cart. </LI>
            <LI><strong>rot180</strong>: (bool) Default=False. </LI>
            <LI><strong>debug</strong>: (bool) Default=False. </LI>
            </UL>



.. autotxclass:: txpipe.convergence.TXConvergenceMapPlots
    :members:
    :exclude-members: run

    Parallel: No - Serial

    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>projection</strong>: (str) Default=cart. </LI>
            </UL>



.. autotxclass:: txpipe.rail.summarize.PZRealizationsPlot
    :members:
    :exclude-members: run

    Parallel: No - Serial

    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>zmax</strong>: (float) Default=3.0. </LI>
            <LI><strong>nz</strong>: (int) Default=301. </LI>
            </UL>



.. autotxclass:: txpipe.twopoint_plots.TXTwoPointPlots
    :members:
    :exclude-members: run

    Parallel: No - Serial

    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>wspace</strong>: (float) Default=0.05. </LI>
            <LI><strong>hspace</strong>: (float) Default=0.05. </LI>
            </UL>



.. autotxclass:: txpipe.twopoint_plots.TXTwoPointPlotsFourier
    :members:
    :exclude-members: run

    Parallel: No - Serial

    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>wspace</strong>: (float) Default=0.05. </LI>
            <LI><strong>hspace</strong>: (float) Default=0.05. </LI>
            </UL>



.. autotxclass:: txpipe.twopoint_plots.TXTwoPointPlotsTheory
    :members:
    :exclude-members: run

    Parallel: No - Serial

    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>wspace</strong>: (float) Default=0.05. </LI>
            <LI><strong>hspace</strong>: (float) Default=0.05. </LI>
            </UL>


