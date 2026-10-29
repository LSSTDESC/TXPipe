Mock Catalogs
=============

These stages generate mock data from simpler input catalogs or from
simulations.

* :py:class:`~txpipe.ingest.mocks.TXCosmoDC2Mock` - Simulate mock shear and photometry measurements from CosmoDC2 (or similar)

* :py:class:`~txpipe.ingest.mocks.TXBuzzardMock` - Simulate mock photometry from Buzzard.

* :py:class:`~txpipe.ingest.mocks.TXGaussianSimsMock` - Simulate mock photometry from gaussian simulations

* :py:class:`~txpipe.ingest.mocks.TXSimpleMock` - Load an ascii astropy table and put it in shear catalog format.

* :py:class:`~txpipe.ingest.mocks.TXMockTruthPZ` - Make mock photo-z PDFs from the true redshifts in a shear catalog.

* :py:class:`~txpipe.ingest.flagship_mocks.TXIngestFlagshipMocks` - Ingest Flagship simulation mock catalogs into a TXPipe shear catalog.

* :py:class:`~txpipe.ingest.flagship_mocks.TXIngestFlagshipMasks` - Convert the Flagship simulation mask into a TXPipe maps file.

* :py:class:`~txpipe.simulation.TXLogNormalGlass` - Uses GLASS to generate a simulated catalog from lognormal fields



.. autotxclass:: txpipe.ingest.mocks.TXCosmoDC2Mock
    :members:
    :exclude-members: run

    Inputs: 

    - response_model: HDFFile

    Outputs: 

    - shear_catalog: ShearCatalog
    - photometry_catalog: HDFFile
    
    Parallel: No - Serial


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>cat_name</strong>: (str) Default=cosmoDC2. Name of the mock catalog to use.</LI>
            <LI><strong>years</strong>: (float) Default=1.0. The number of years of data to simulate for noise</LI>
            <LI><strong>snr_limit</strong>: (float) Default=4.0. S/N limit for object selection.</LI>
            <LI><strong>max_size</strong>: (int) Default=99999999999999. Maximum catalog size for testing.</LI>
            <LI><strong>extra_cols</strong>: (str) Default=. Extra columns to include (space-separated).</LI>
            <LI><strong>max_npix</strong>: (int) Default=99999999999999. Maximum number of pixels.</LI>
            <LI><strong>unit_response</strong>: (bool) Default=False. Whether to use unit response in simulation.</LI>
            <LI><strong>cat_size</strong>: (int) Default=0. Pre-computed catalog size, if known.</LI>
            <LI><strong>flip_g2</strong>: (bool) Default=True. Whether to flip g2 sign to match conventions. TreeCorr and NaMaster use flip_g2=True</LI>
            <LI><strong>apply_mag_cut</strong>: (bool) Default=False. Apply magnitude cut for descqa comparison.</LI>
            <LI><strong>Mag_r_limit</strong>: (float) Default=-19. Magnitude r limit for object selection.</LI>
            <LI><strong>Mag_i_limit</strong>: (float) Default=-19. Magnitude i limit for object selection.</LI>
            <LI><strong>metadetect</strong>: (bool) Default=True. Whether to make a metadetect-style catalog (True) or metacal (False).</LI>
            <LI><strong>add_shape_noise</strong>: (bool) Default=True. Whether to add shape noise to simulation.</LI>
            <LI><strong>healpixels</strong>: (list) Default=[-1]. List of HEALPix pixels to use.</LI>
            </UL>



.. autotxclass:: txpipe.ingest.mocks.TXBuzzardMock
    :members:
    :exclude-members: run

    Inputs: 

    - response_model: HDFFile

    Outputs: 

    - shear_catalog: ShearCatalog
    - photometry_catalog: HDFFile
    
    Parallel: No - Serial


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>cat_name</strong>: (str) Default=buzzard. Name of the mock catalog to use.</LI>
            <LI><strong>years</strong>: (float) Default=1.0. Number of years of data for noise simulation.</LI>
            <LI><strong>snr_limit</strong>: (float) Default=4.0. S/N limit for object selection.</LI>
            <LI><strong>max_size</strong>: (int) Default=99999999999999. Maximum catalog size for testing.</LI>
            <LI><strong>extra_cols</strong>: (str) Default=. Extra columns to include (space-separated).</LI>
            <LI><strong>max_npix</strong>: (int) Default=99999999999999. Maximum number of pixels.</LI>
            <LI><strong>unit_response</strong>: (bool) Default=False. Whether to use unit response in simulation.</LI>
            <LI><strong>flip_g2</strong>: (bool) Default=True. Whether to flip g2 sign to match conventions.</LI>
            </UL>



.. autotxclass:: txpipe.ingest.mocks.TXGaussianSimsMock
    :members:
    :exclude-members: run

    Inputs: 

    - response_model: HDFFile

    Outputs: 

    - shear_catalog: ShearCatalog
    - photometry_catalog: HDFFile
    
    Parallel: No - Serial


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>cat_name</strong>: (str) Default=GaussianSims. Name of the Gaussian simulation catalog.</LI>
            <LI><strong>years</strong>: (float) Default=1.0. Number of years of data for noise simulation.</LI>
            <LI><strong>snr_limit</strong>: (float) Default=0.0. S/N limit for object selection (0 for all).</LI>
            <LI><strong>max_size</strong>: (int) Default=99999999999999. Maximum catalog size for testing.</LI>
            <LI><strong>extra_cols</strong>: (str) Default=. Extra columns to include (space-separated).</LI>
            <LI><strong>max_npix</strong>: (int) Default=99999999999999. Maximum number of pixels.</LI>
            <LI><strong>unit_response</strong>: (bool) Default=True. Whether to use unit response in simulation.</LI>
            <LI><strong>cat_size</strong>: (int) Default=0. Catalog size (0 for all).</LI>
            <LI><strong>flip_g2</strong>: (bool) Default=False. Whether to flip g2 sign to match conventions.</LI>
            <LI><strong>apply_mag_cut</strong>: (bool) Default=False. Apply magnitude cut for descqa comparison.</LI>
            <LI><strong>apply_i_mag_cut</strong>: (bool) Default=False. Apply observed i-band magnitude cut.</LI>
            <LI><strong>Mag_i_limit</strong>: (float) Default=24.25. Observed i-band magnitude limit for object selection.</LI>
            <LI><strong>metadetect</strong>: (bool) Default=True. Whether to make a metadetect-style catalog (True) or metacal (False).</LI>
            <LI><strong>add_shape_noise</strong>: (bool) Default=False. Whether to add shape noise to simulation.</LI>
            </UL>



.. autotxclass:: txpipe.ingest.mocks.TXSimpleMock
    :members:
    :exclude-members: run

    Inputs: 

    - mock_shear_catalog: TextFile

    Outputs: 

    - shear_catalog: ShearCatalog
    
    Parallel: No - Serial


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            </UL>



.. autotxclass:: txpipe.ingest.mocks.TXMockTruthPZ
    :members:
    :exclude-members: run

    Inputs: 

    - shear_catalog: ShearCatalog

    Outputs: 

    - photoz_pdfs: QPPDFFile
    
    Parallel: No - Serial


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>mock_sigma_z</strong>: (float) Default=0.001. Sigma_z for mock photo-z PDF generation.</LI>
            </UL>



.. autotxclass:: txpipe.ingest.flagship_mocks.TXIngestFlagshipMocks
    :members:
    :exclude-members: run

    Inputs: None

    Outputs: 

    - shear_catalog: ShearCatalog
    
    Parallel: Yes - MPI


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>input_file_pattern</strong>: (str) Default=/global/cfs/cdirs/lsst/groups/PZ/users/qhang/flagship_dp2/flagship_mock/Flagship/dp2_mock_run_flagship_gold_test/248*/output_deredden_lsst_obs_cond_dp2.pq. Glob pattern for input files</LI>
            <LI><strong>chunk_rows</strong>: (int) Default=1000000. Number of rows to process at once</LI>
            </UL>



.. autotxclass:: txpipe.ingest.flagship_mocks.TXIngestFlagshipMasks
    :members:
    :exclude-members: run

    Inputs: 

    - original_flagship_mask: MapsFile

    Outputs: 

    - desi_mask: MapsFile
    
    Parallel: Yes - MPI


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            </UL>



.. autotxclass:: txpipe.simulation.TXLogNormalGlass
    :members:
    :exclude-members: run

    Inputs: 

    - mask: MapsFile
    - lens_photoz_stack: QPNOfZFile
    - fiducial_cosmology: FiducialCosmology
    - input_lss_weight_maps: MapsFile

    Outputs: 

    - photometry_catalog: HDFFile
    - lens_tomography_catalog_unweighted: TomographyCatalog
    - glass_cl_shells: HDFFile
    - glass_cl_binned: HDFFile
    - density_shells: HDFFile
    
    Parallel: No - Serial


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>num_dens</strong>: (list) Required. Number density of galaxies per square arcmin</LI>
            <LI><strong>zmin</strong>: (float) Default=0.0. Minimum redshift for the simulation</LI>
            <LI><strong>zmax</strong>: (float) Default=2.0. Maximum redshift for the simulation</LI>
            <LI><strong>dx</strong>: (int) Default=100. Comoving distance spacing for redshift shells in Mpc</LI>
            <LI><strong>bias0</strong>: (float) Default=2.0. Linear bias at zpivot</LI>
            <LI><strong>alpha_bz</strong>: (float) Default=0.0. Controls redshift evolution of bias</LI>
            <LI><strong>zpivot</strong>: (float) Default=0.6. Pivot redshift for bias evolution</LI>
            <LI><strong>shift</strong>: (float) Default=1.0. Lognormal shift parameter</LI>
            <LI><strong>contaminate</strong>: (bool) Default=False. Whether to apply contamination to the density field</LI>
            <LI><strong>random_seed</strong>: (int) Default=0. Random seed for reproducibility</LI>
            <LI><strong>cl_optional_file</strong>: (str) Default=none. Optional file for input C(l) values. Otherwise they are computed (slow)</LI>
            <LI><strong>ell_binned_min</strong>: (float) Default=0.1. Minimum ell for binned C(l) output</LI>
            <LI><strong>ell_binned_max</strong>: (float) Default=500000.0. Maximum ell for binned C(l) output</LI>
            <LI><strong>ell_binned_nbins</strong>: (int) Default=100. Number of ell bins for binned C(l) output</LI>
            <LI><strong>output_density_shell_maps</strong>: (bool) Default=False. Whether to output density maps for each shell</LI>
            <LI><strong>density_shell_nest</strong>: (bool) Default=False. Whether output density maps be in nest ordering</LI>
            </UL>


