CMB Lensing Extensions
======================

These stages are written for TXPipe extension projects cross-correlating with CMB lensing.

* :py:class:`~txpipe.extensions.cmb_lensing.ingest.TXIngestPlanckLensingMaps` - Ingest Planck CMB lensing maps.

* :py:class:`~txpipe.extensions.cmb_lensing.ingest.TXIngestQuaia` - Ingest Quaia lensing maps as TXPipe overdensity (delta) maps.

* :py:class:`~txpipe.extensions.cmb_lensing.twopoint_fourier_cross.TXCMBLensingCrossMonteCarloCorrection` - Compute equation 23 in https://arxiv.org/pdf/2407.04607

* :py:class:`~txpipe.extensions.cmb_lensing.twopoint_fourier_cross.TXTwoPointFourierCMBLensingCrossDensity` - Compute the cross-correlation maps between CMB lensing and galaxy density.



.. autotxclass:: txpipe.extensions.cmb_lensing.ingest.TXIngestPlanckLensingMaps
    :members:
    :exclude-members: run

    Inputs: None

    Outputs: 

    - cmb_lensing_map: MapsFile
    
    Parallel: Yes - MPI


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>alm_file</strong>: (str) Required. </LI>
            <LI><strong>mask_file</strong>: (str) Required. </LI>
            <LI><strong>nside</strong>: (int) Default=512. </LI>
            </UL>



.. autotxclass:: txpipe.extensions.cmb_lensing.ingest.TXIngestQuaia
    :members:
    :exclude-members: run

    Inputs: None

    Outputs: 

    - density_maps: MapsFile
    - density_masks: MapsFile
    - lens_photoz_stack: QPNOfZFile
    
    Parallel: Yes - MPI


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>quaia_file</strong>: (str) Required. </LI>
            <LI><strong>selection_function_template</strong>: (str) Required. </LI>
            <LI><strong>nside</strong>: (int) Default=512. </LI>
            <LI><strong>sel_threshold</strong>: (float) Default=0.5. </LI>
            <LI><strong>num_z_bins</strong>: (int) Default=500. </LI>
            <LI><strong>zname</strong>: (str) Default=redshift_quaia. </LI>
            </UL>



.. autotxclass:: txpipe.extensions.cmb_lensing.twopoint_fourier_cross.TXCMBLensingCrossMonteCarloCorrection
    :members:
    :exclude-members: run

    Inputs: 

    - cmb_lensing_map: MapsFile
    - mask: MapsFile
    - fiducial_cosmology: FiducialCosmology

    Outputs: 

    - cmb_cross_montecarlo_correction: TextFile
    
    Parallel: Yes - MPI


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>cmb_redshift</strong>: (float) Default=1100.0. </LI>
            <LI><strong>nside</strong>: (int) Default=512. </LI>
            <LI><strong>nsim</strong>: (int) Default=1000. </LI>
            <LI><strong>mask_threshold</strong>: (float) Default=0.0. </LI>
            </UL>



.. autotxclass:: txpipe.extensions.cmb_lensing.twopoint_fourier_cross.TXTwoPointFourierCMBLensingCrossDensity
    :members:
    :exclude-members: run

    Inputs: 

    - cmb_lensing_map: MapsFile
    - cmb_lensing_beam: TextFile
    - density_maps: MapsFile
    - density_masks: MapsFile
    - cmb_cross_montecarlo_correction: TextFile
    - lens_photoz_stack: QPNOfZFile

    Outputs: 

    - twopoint_data_fourier_cmb_cross_density: SACCFile
    - twopoint_data_fourier_cmb_cross_density_plot: PNGFile
    
    Parallel: Yes - MPI


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>mask_threshold</strong>: (float) Default=0.0. </LI>
            <LI><strong>bandpower_width</strong>: (int) Default=30. </LI>
            </UL>


