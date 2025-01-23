Weights
=======

These stages deal with weighting the lens sample

* :py:class:`~txpipe.lssweights.TXLSSWeights` - BaseClass to compute LSS systematic weights

* :py:class:`~txpipe.lssweights.TXLSSWeightsLinBinned` - Class compute LSS systematic weights using simultanious linear regression on the binned

* :py:class:`~txpipe.lssweights.TXLSSWeightsLinPix` - Class compute LSS systematic weights using simultanious linear regression at the

* :py:class:`~txpipe.lssweights.TXLSSWeightsUnit` - Class assigns weight=1 to all lens objects



.. autoclass:: txpipe.lssweights.TXLSSWeights

    **parallel**: No - Serial

.. autoclass:: txpipe.lssweights.TXLSSWeightsLinBinned

    **parallel**: No - Serial

.. autoclass:: txpipe.lssweights.TXLSSWeightsLinPix

    **parallel**: No - Serial

.. autoclass:: txpipe.lssweights.TXLSSWeightsUnit

    **parallel**: No - Serial
