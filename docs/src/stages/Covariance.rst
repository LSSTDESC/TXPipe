Covariance
==========

These stages compute covariances of measurements

* :py:class:`~txpipe.covariance.TXFourierGaussianCovariance` - Compute a Gaussian Fourier-space covariance with TJPCov using f_sky only

* :py:class:`~txpipe.covariance.TXRealGaussianCovariance` - Compute a Gaussian real-space covariance with TJPCov using f_sky only

* :py:class:`~txpipe.covariance.TXFourierTJPCovariance` - Compute a Gaussian Fourier-space covariance with TJPCov using mask geometry

* :py:class:`~txpipe.covariance_nmt.TXFourierNamasterCovariance` - Compute a Gaussian Fourier-space covariance with NaMaster

* :py:class:`~txpipe.covariance_nmt.TXRealNamasterCovariance` - Compute a Gaussian real-space covariance with NaMaster



.. autotxclass:: txpipe.covariance.TXFourierGaussianCovariance
    :members:
    :exclude-members: run

    Inputs: 

    - fiducial_cosmology: FiducialCosmology
    - twopoint_data_fourier: SACCFile
    - tracer_metadata: HDFFile

    Outputs: 

    - summary_statistics_fourier: SACCFile
    
    Parallel: No - Serial


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>pickled_wigner_transform</strong>: (str) Default=. </LI>
            <LI><strong>use_true_shear</strong>: (bool) Default=False. </LI>
            <LI><strong>galaxy_bias</strong>: (list) Default=[0.0]. </LI>
            <LI><strong>gaussian_sims_factor</strong>: (list) Default=[1.0]. </LI>
            </UL>



.. autotxclass:: txpipe.covariance.TXRealGaussianCovariance
    :members:
    :exclude-members: run

    Inputs: 

    - fiducial_cosmology: FiducialCosmology
    - twopoint_data_real: SACCFile
    - tracer_metadata: HDFFile

    Outputs: 

    - summary_statistics_real: SACCFile
    
    Parallel: No - Serial


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>min_sep</strong>: (float) Default=2.5. </LI>
            <LI><strong>max_sep</strong>: (int) Default=250. </LI>
            <LI><strong>nbins</strong>: (int) Default=20. </LI>
            <LI><strong>pickled_wigner_transform</strong>: (str) Default=. </LI>
            <LI><strong>use_true_shear</strong>: (bool) Default=False. </LI>
            <LI><strong>galaxy_bias</strong>: (list) Default=[0.0]. </LI>
            <LI><strong>gaussian_sims_factor</strong>: (list) Default=[1.0]. </LI>
            </UL>



.. autotxclass:: txpipe.covariance.TXFourierTJPCovariance
    :members:
    :exclude-members: run

    Inputs: 

    - fiducial_cosmology: FiducialCosmology
    - twopoint_data_fourier: SACCFile
    - tracer_metadata_yml: YamlFile
    - mask: MapsFile
    - density_maps: MapsFile
    - source_maps: MapsFile

    Outputs: 

    - summary_statistics_fourier: SACCFile
    
    Parallel: Yes - MPI


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>galaxy_bias</strong>: (list) Default=[0.0]. </LI>
            <LI><strong>IA</strong>: (float) Default=0.5. </LI>
            <LI><strong>cache_dir</strong>: (str) Default=. </LI>
            <LI><strong>cov_type</strong>: (list) Default=['FourierGaussianNmt', 'FourierSSCHaloModel']. </LI>
            </UL>



.. autotxclass:: txpipe.covariance_nmt.TXFourierNamasterCovariance
    :members:
    :exclude-members: run

    Inputs: 

    - fiducial_cosmology: FiducialCosmology
    - twopoint_data_fourier: SACCFile
    - tracer_metadata: HDFFile
    - mask: MapsFile

    Outputs: 

    - summary_statistics_fourier: SACCFile
    
    Parallel: Yes - MPI


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>pickled_wigner_transform</strong>: (str) Default=. </LI>
            <LI><strong>use_true_shear</strong>: (bool) Default=False. </LI>
            <LI><strong>scratch_dir</strong>: (str) Default=temp. </LI>
            <LI><strong>nside</strong>: (int) Default=1024. </LI>
            </UL>



.. autotxclass:: txpipe.covariance_nmt.TXRealNamasterCovariance
    :members:
    :exclude-members: run

    Inputs: 

    - fiducial_cosmology: FiducialCosmology
    - twopoint_data_real: SACCFile
    - tracer_metadata: HDFFile
    - mask: MapsFile

    Outputs: 

    - summary_statistics_real: SACCFile
    
    Parallel: Yes - MPI


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>min_sep</strong>: (float) Default=2.5. </LI>
            <LI><strong>max_sep</strong>: (int) Default=250. </LI>
            <LI><strong>nbins</strong>: (int) Default=20. </LI>
            <LI><strong>pickled_wigner_transform</strong>: (str) Default=. </LI>
            <LI><strong>use_true_shear</strong>: (bool) Default=False. </LI>
            <LI><strong>galaxy_bias</strong>: (list) Default=[0.0]. </LI>
            </UL>


