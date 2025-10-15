Higher Order Statistic Extensions
=================================

These stages are written for TXPipe extension projects on higher order or alternative statistics.

* :py:class:`~txpipe.extensions.twopoint_scia.TXSelfCalibrationIA` - This is the subclass of the Twopoint class that will handle calculating the

* :py:class:`~txpipe.extensions.hos.base.HOSStage` - A stage intended as a base for higher-order statistics (HOS) measurements.

* :py:class:`~txpipe.extensions.hos.fsb.HOSFSB` - Measure the filtered squared bispectrum (FSB) from over-density maps.



.. autotxclass:: txpipe.extensions.twopoint_scia.TXSelfCalibrationIA
    :members:
    :exclude-members: run

    Inputs: 

    - shear_catalog: ShearCatalog
    - shear_tomography_catalog: TomographyCatalog
    - shear_photoz_stack: QPNOfZFile
    - random_cats_source: RandomsCatalog
    - lens_tomography_catalog: TomographyCatalog
    - patch_centers: TextFile
    - photoz_pdfs: PhotozPDFFile
    - fiducial_cosmology: FiducialCosmology
    - tracer_metadata: HDFFile

    Outputs: 

    - twopoint_data_SCIA: SACCFile
    - gammaX_scia: SACCFile
    
    Parallel: Yes - MPI


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



.. autotxclass:: txpipe.extensions.hos.base.HOSStage
    :members:
    :exclude-members: run

    Inputs: None

    Outputs: None
    
    Parallel: Yes - MPI


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            </UL>



.. autotxclass:: txpipe.extensions.hos.fsb.HOSFSB
    :members:
    :exclude-members: run

    Inputs: 

    - density_maps: MapsFile
    - mask: MapsFile
    - lens_photoz_stack: QPNOfZFile
    - tracer_metadata: HDFFile

    Outputs: 

    - filtered_squared_bispectrum: SACCFile
    
    Parallel: Yes - MPI


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>ells_per_bin</strong>: (int) Default=10. </LI>
            <LI><strong>nfilters</strong>: (int) Default=5. </LI>
            <LI><strong>include_n32</strong>: (bool) Default=False. </LI>
            </UL>


