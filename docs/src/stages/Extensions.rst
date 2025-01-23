Extensions
==========

These stages are written for TXPipe extension projects.

* :py:class:`~txpipe.extensions.clmm.bin_cluster.CLClusterBinningRedshiftRichness` - Stage CLClusterBinningRedshiftRichness

* :py:class:`~txpipe.extensions.clmm.sources_select_compute.CLClusterShearCatalogs` - Configuration Parameters:

* :py:class:`~txpipe.extensions.clmm.make_ensemble_profile.CLClusterEnsembleProfiles` - Stage CLClusterEnsembleProfiles

* :py:class:`~txpipe.extensions.twopoint_scia.TXSelfCalibrationIA` - This is the subclass of the Twopoint class that will handle calculating the



.. autoclass:: txpipe.extensions.clmm.bin_cluster.CLClusterBinningRedshiftRichness

    **parallel**: No - Serial

.. autoclass:: txpipe.extensions.clmm.sources_select_compute.CLClusterShearCatalogs

    **parallel**: Yes - MPI

.. autoclass:: txpipe.extensions.clmm.make_ensemble_profile.CLClusterEnsembleProfiles

    **parallel**: Yes - MPI

.. autoclass:: txpipe.extensions.twopoint_scia.TXSelfCalibrationIA

    **parallel**: Yes - MPI
