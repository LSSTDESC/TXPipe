Delta Sigma
===========

These stages measure and predict the excess surface density

* :py:class:`~txpipe.delta_sigma.TXDeltaSigma` - Compute Delta-Sigma, the excess surface density around lenses.

* :py:class:`~txpipe.delta_sigma.TXDeltaSigmaTheory` - Compute theory predictions for the excess surface density Delta Sigma.

* :py:class:`~txpipe.delta_sigma.TXDeltaSigmaPlots` - Make plots of Delta Sigma results.



.. autotxclass:: txpipe.delta_sigma.TXDeltaSigma
    :members:
    :exclude-members: run

    Inputs: 

    - binned_shear_catalog: BinnedCatalog
    - binned_lens_catalog: BinnedCatalog
    - binned_random_catalog: BinnedCatalog
    - random_cats: HDFFile
    - shear_photoz_stack: QPNOfZFile
    - lens_photoz_stack: QPNOfZFile
    - fiducial_cosmology: FiducialCosmology

    Outputs: 

    - delta_sigma: SACCFile
    
    Parallel: Yes - MPI


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>source_bins</strong>: (list) Default=[-1]. List of source bins to use (-1 means all)</LI>
            <LI><strong>lens_bins</strong>: (list) Default=[-1]. List of lens bins to use (-1 means all)</LI>
            <LI><strong>r_min</strong>: (float) Default=0.1. Minimum radius to use in Mpc</LI>
            <LI><strong>r_max</strong>: (float) Default=10.0. Maximum radius to use in Mpc</LI>
            <LI><strong>nbins</strong>: (int) Default=10. Number of radial bins</LI>
            <LI><strong>photoz</strong>: (bool) Default=True. If True, the sources have photometric redshifts so we should not load or use per-objects z values. If False, use per-object values.</LI>
            <LI><strong>source_cat_w_col</strong>: (str) Default=weight. Source catalog weight column name.</LI>
            <LI><strong>lens_cat_w_col</strong>: (str) Default=weight. Lens catalog weight column name.</LI>
            <LI><strong>n_jobs</strong>: (int) Default=1. Number of multiprocessing processes to use</LI>
            <LI><strong>n_jackknife</strong>: (int) Default=0. Number of jackknife regions to use for error estimation, 0 means no jackknife</LI>
            <LI><strong>subsample</strong>: (float) Default=0.0. If set to non-zero, randomly subsample this fraction of the all the data sets for testing.</LI>
            </UL>



.. autotxclass:: txpipe.delta_sigma.TXDeltaSigmaTheory
    :members:
    :exclude-members: run

    Inputs: 

    - fiducial_cosmology: FiducialCosmology
    - lens_photoz_stack: QPNOfZFile

    Outputs: 

    - delta_sigma_theory: SACCFile
    
    Parallel: Yes - MPI


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>r_min</strong>: (float) Default=0.1. Minimum radius to use in Mpc</LI>
            <LI><strong>r_max</strong>: (float) Default=60.0. Maximum radius to use in Mpc</LI>
            <LI><strong>n_r</strong>: (int) Default=200. Number of radial bins</LI>
            <LI><strong>k_min</strong>: (float) Default=0.001. Min wavenumber to use in theory calculation</LI>
            <LI><strong>k_max</strong>: (float) Default=100.0. Max wavenumber to use in theory calculation</LI>
            <LI><strong>nk</strong>: (int) Default=128. Number of wavenumber values to use in theory calculation</LI>
            <LI><strong>a_min</strong>: (float) Default=0.2. Min scale factor to use in theory calculation</LI>
            <LI><strong>a_max</strong>: (float) Default=1.0. Max scale factor to use in theory calculation</LI>
            <LI><strong>na</strong>: (int) Default=128. Number of scale factor values to use in theory calculation</LI>
            </UL>



.. autotxclass:: txpipe.delta_sigma.TXDeltaSigmaPlots
    :members:
    :exclude-members: run

    Inputs: 

    - delta_sigma: SACCFile
    - delta_sigma_theory: SACCFile
    - fiducial_cosmology: FiducialCosmology

    Outputs: 

    - delta_sigma_plot: PNGFile
    
    Parallel: Yes - MPI


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>show_theory</strong>: (bool) Default=True. Whether to show theory line</LI>
            </UL>


