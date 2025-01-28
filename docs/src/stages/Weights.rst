Weights
=======

These stages deal with weighting the lens sample

* :py:class:`~txpipe.lssweights.TXLSSWeights` - Base class for LSS systematic weights

* :py:class:`~txpipe.lssweights.TXLSSWeightsLinBinned` - Compute LSS systematic weights using simultanious linear regression on the binned

* :py:class:`~txpipe.lssweights.TXLSSWeightsLinPix` - Compute LSS systematic weights using simultanious linear regression at the

* :py:class:`~txpipe.lssweights.TXLSSWeightsUnit` - Assign weight=1 to all lens objects



.. autotxclass:: txpipe.lssweights.TXLSSWeights
    :members:
    :exclude-members: run

    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>supreme_path_root</strong>: (str) Default=. </LI>
            <LI><strong>nbin</strong>: (int) Default=20. </LI>
            <LI><strong>outlier_fraction</strong>: (float) Default=0.01. </LI>
            <LI><strong>allow_weighted_input</strong>: (bool) Default=False. </LI>
            <LI><strong>nside_coverage</strong>: (int) Default=32. </LI>
            </UL>



.. autotxclass:: txpipe.lssweights.TXLSSWeightsLinBinned
    :members:
    :exclude-members: run

    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>supreme_path_root</strong>: (str) Default=. </LI>
            <LI><strong>nbin</strong>: (int) Default=20. </LI>
            <LI><strong>outlier_fraction</strong>: (float) Default=0.05. </LI>
            <LI><strong>pvalue_threshold</strong>: (float) Default=0.05. </LI>
            <LI><strong>equal_area_bins</strong>: (bool) Default=True. </LI>
            <LI><strong>simple_cov</strong>: (bool) Default=False. </LI>
            <LI><strong>diag_blocks_only</strong>: (bool) Default=True. </LI>
            <LI><strong>b0</strong>: (list) Default=[1.0]. </LI>
            <LI><strong>allow_weighted_input</strong>: (bool) Default=False. </LI>
            <LI><strong>nside_coverage</strong>: (int) Default=32. </LI>
            </UL>



.. autotxclass:: txpipe.lssweights.TXLSSWeightsLinPix
    :members:
    :exclude-members: run

    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>supreme_path_root</strong>: (str) Default=. </LI>
            <LI><strong>nbin</strong>: (int) Default=20. </LI>
            <LI><strong>outlier_fraction</strong>: (float) Default=0.05. </LI>
            <LI><strong>pvalue_threshold</strong>: (float) Default=0.05. </LI>
            <LI><strong>equal_area_bins</strong>: (bool) Default=True. </LI>
            <LI><strong>simple_cov</strong>: (bool) Default=False. </LI>
            <LI><strong>diag_blocks_only</strong>: (bool) Default=True. </LI>
            <LI><strong>b0</strong>: (list) Default=[1.0]. </LI>
            <LI><strong>regression_class</strong>: (str) Default=LinearRegression. </LI>
            <LI><strong>allow_weighted_input</strong>: (bool) Default=False. </LI>
            <LI><strong>nside_coverage</strong>: (int) Default=32. </LI>
            </UL>



.. autotxclass:: txpipe.lssweights.TXLSSWeightsUnit
    :members:
    :exclude-members: run

    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>nside_coverage</strong>: (int) Default=32. </LI>
            </UL>


