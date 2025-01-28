Extensions
==========

These stages are written for TXPipe extension projects.

* :py:class:`~txpipe.extensions.clmm.bin_cluster.CLClusterBinningRedshiftRichness` - Stage CLClusterBinningRedshiftRichness

* :py:class:`~txpipe.extensions.clmm.sources_select_compute.CLClusterShearCatalogs` - Configuration Parameters:

* :py:class:`~txpipe.extensions.clmm.make_ensemble_profile.CLClusterEnsembleProfiles` - Stage CLClusterEnsembleProfiles

* :py:class:`~txpipe.extensions.twopoint_scia.TXSelfCalibrationIA` - This is the subclass of the Twopoint class that will handle calculating the



.. autotxclass:: txpipe.extensions.clmm.bin_cluster.CLClusterBinningRedshiftRichness
    :members:
    :exclude-members: run

    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>zedge</strong>: (list) Default=[0.2, 0.4, 0.6, 0.8, 1.0]. </LI>
            <LI><strong>richedge</strong>: (list) Default=[5.0, 10.0, 20.0]. </LI>
            <LI><strong>initial_size</strong>: (int) Default=100000. </LI>
            <LI><strong>chunk_rows</strong>: (int) Default=100000. </LI>
            </UL>



.. autotxclass:: txpipe.extensions.clmm.sources_select_compute.CLClusterShearCatalogs
    :members:
    :exclude-members: run

    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>chunk_rows</strong>: (int) Default=100000. </LI>
            <LI><strong>max_radius</strong>: (float) Default=10.0. </LI>
            <LI><strong>delta_z</strong>: (float) Default=0.1. </LI>
            <LI><strong>redshift_cut_criterion</strong>: (str) Default=zmean. </LI>
            <LI><strong>redshift_weight_criterion</strong>: (str) Default=zmean. </LI>
            <LI><strong>redshift_cut_criterion_pdf_fraction</strong>: (float) Default=0.9. </LI>
            <LI><strong>subtract_mean_shear</strong>: (bool) Default=True. </LI>
            <LI><strong>coordinate_system</strong>: (str) Default=celestial. </LI>
            </UL>



.. autotxclass:: txpipe.extensions.clmm.make_ensemble_profile.CLClusterEnsembleProfiles
    :members:
    :exclude-members: run

    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>r_min</strong>: (float) Default=0.2. </LI>
            <LI><strong>r_max</strong>: (float) Default=3.0. </LI>
            <LI><strong>nbins</strong>: (int) Default=5. </LI>
            <LI><strong>delta_sigma_profile</strong>: (bool) Default=True. </LI>
            <LI><strong>shear_profile</strong>: (bool) Default=False. </LI>
            <LI><strong>magnification_profile</strong>: (bool) Default=False. </LI>
            </UL>



.. autotxclass:: txpipe.extensions.twopoint_scia.TXSelfCalibrationIA
    :members:
    :exclude-members: run

    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>calcs</strong>: (list) Default=[0, 1, 2]. </LI>
            <LI><strong>min_sep</strong>: (float) Default=2.5. </LI>
            <LI><strong>max_sep</strong>: (float) Default=250.0. </LI>
            <LI><strong>nbins</strong>: (int) Default=20. </LI>
            <LI><strong>bin_slop</strong>: (float) Default=0.1. </LI>
            <LI><strong>flip_g2</strong>: (bool) Default=True. </LI>
            <LI><strong>cores_per_task</strong>: (int) Default=20. </LI>
            <LI><strong>verbose</strong>: (int) Default=1. </LI>
            <LI><strong>source_bins</strong>: (list) Default=[-1]. </LI>
            <LI><strong>lens_bins</strong>: (list) Default=[-1]. </LI>
            <LI><strong>reduce_randoms_size</strong>: (float) Default=1.0. </LI>
            <LI><strong>do_shear_pos</strong>: (bool) Default=True. </LI>
            <LI><strong>do_pos_pos</strong>: (bool) Default=False. </LI>
            <LI><strong>do_shear_shear</strong>: (bool) Default=False. </LI>
            <LI><strong>var_method</strong>: (str) Default=jackknife. </LI>
            <LI><strong>3Dcoords</strong>: (bool) Default=True. </LI>
            <LI><strong>metric</strong>: (str) Default=Rperp. </LI>
            <LI><strong>use_true_shear</strong>: (bool) Default=False. </LI>
            <LI><strong>subtract_mean_shear</strong>: (bool) Default=False. </LI>
            <LI><strong>redshift_shearcatalog</strong>: (bool) Default=False. </LI>
            <LI><strong>chunk_rows</strong>: (int) Default=100000. </LI>
            <LI><strong>use_subsampled_randoms</strong>: (bool) Default=False. </LI>
            </UL>


