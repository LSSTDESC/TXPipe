Ingestion
=========

These stages import data into TXPipe input formats, or generate mock data from
simpler input catalogs.

* :py:class:`~txpipe.exposure_info.TXExposureInfo` - Ingest exposure information from an OpSim database

* :py:class:`~txpipe.ingest.dp02.TXIngestDataPreview02` - Ingest galaxy catalogs from DP0.2

* :py:class:`~txpipe.ingest.mocks.TXCosmoDC2Mock` - Simulate mock shear and photometry measurements from CosmoDC2 (or similar)

* :py:class:`~txpipe.ingest.mocks.TXBuzzardMock` - Simulate mock photometry from Buzzard.

* :py:class:`~txpipe.ingest.mocks.TXGaussianSimsMock` - Simulate mock photometry from gaussian simulations

* :py:class:`~txpipe.ingest.mocks.TXSimpleMock` - Load an ascii astropy table and put it in shear catalog format.

* :py:class:`~txpipe.ingest.mocks.TXMockTruthPZ` - Stage TXMockTruthPZ

* :py:class:`~txpipe.ingest.gcr.TXMetacalGCRInput` - Ingest metacal catalogs from GCRCatalogs

* :py:class:`~txpipe.ingest.gcr.TXIngestStars` - Ingest a star catalog from GCRCatalogs

* :py:class:`~txpipe.ingest.redmagic.TXIngestRedmagic` - Ingest a redmagic catalog

* :py:class:`~txpipe.simulation.TXLogNormalGlass` - Uses GLASS to generate a simulated catalog from lognormal fields



.. autotxclass:: txpipe.exposure_info.TXExposureInfo
    :members:
    :exclude-members: run

    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>dc2_name</strong>: (str) Default=1.2p. </LI>
            <LI><strong>opsim_db</strong>: (str) Default=/global/projecta/projectdirs/lsst/groups/SSim/DC2/minion_1016_desc_dithered_v4.db. </LI>
            <LI><strong>propId</strong>: (int) Default=54. </LI>
            </UL>



.. autotxclass:: txpipe.ingest.dp02.TXIngestDataPreview02
    :members:
    :exclude-members: run

    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>pq_path</strong>: (str) Default=/global/cfs/cdirs/lsst/shared/rubin/DP0.2/objectTable/. </LI>
            <LI><strong>tracts</strong>: (str) Default=. </LI>
            </UL>



.. autotxclass:: txpipe.ingest.mocks.TXCosmoDC2Mock
    :members:
    :exclude-members: run

    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>cat_name</strong>: (str) Default=cosmoDC2. </LI>
            <LI><strong>visits_per_band</strong>: (int) Default=165. </LI>
            <LI><strong>snr_limit</strong>: (float) Default=4.0. </LI>
            <LI><strong>max_size</strong>: (int) Default=99999999999999. </LI>
            <LI><strong>extra_cols</strong>: (str) Default=. </LI>
            <LI><strong>max_npix</strong>: (int) Default=99999999999999. </LI>
            <LI><strong>unit_response</strong>: (bool) Default=False. </LI>
            <LI><strong>cat_size</strong>: (int) Default=0. </LI>
            <LI><strong>flip_g2</strong>: (bool) Default=True. </LI>
            <LI><strong>apply_mag_cut</strong>: (bool) Default=False. </LI>
            <LI><strong>Mag_r_limit</strong>: (int) Default=-19. </LI>
            <LI><strong>metadetect</strong>: (bool) Default=True. </LI>
            <LI><strong>add_shape_noise</strong>: (bool) Default=True. </LI>
            <LI><strong>healpixels</strong>: (list) Default=[-1]. </LI>
            </UL>



.. autotxclass:: txpipe.ingest.mocks.TXBuzzardMock
    :members:
    :exclude-members: run

    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>cat_name</strong>: (str) Default=buzzard. </LI>
            <LI><strong>visits_per_band</strong>: (int) Default=165. </LI>
            <LI><strong>snr_limit</strong>: (float) Default=4.0. </LI>
            <LI><strong>max_size</strong>: (int) Default=99999999999999. </LI>
            <LI><strong>extra_cols</strong>: (str) Default=. </LI>
            <LI><strong>max_npix</strong>: (int) Default=99999999999999. </LI>
            <LI><strong>unit_response</strong>: (bool) Default=False. </LI>
            <LI><strong>flip_g2</strong>: (bool) Default=True. </LI>
            </UL>



.. autotxclass:: txpipe.ingest.mocks.TXGaussianSimsMock
    :members:
    :exclude-members: run

    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>cat_name</strong>: (str) Default=GaussianSims. </LI>
            <LI><strong>visits_per_band</strong>: (int) Default=165. </LI>
            <LI><strong>snr_limit</strong>: (float) Default=0.0. </LI>
            <LI><strong>max_size</strong>: (int) Default=99999999999999. </LI>
            <LI><strong>extra_cols</strong>: (str) Default=. </LI>
            <LI><strong>max_npix</strong>: (int) Default=99999999999999. </LI>
            <LI><strong>unit_response</strong>: (bool) Default=True. </LI>
            <LI><strong>cat_size</strong>: (int) Default=0. </LI>
            <LI><strong>flip_g2</strong>: (bool) Default=False. </LI>
            <LI><strong>apply_mag_cut</strong>: (bool) Default=False. </LI>
            <LI><strong>metadetect</strong>: (bool) Default=True. </LI>
            <LI><strong>add_shape_noise</strong>: (bool) Default=False. </LI>
            </UL>



.. autotxclass:: txpipe.ingest.mocks.TXSimpleMock
    :members:
    :exclude-members: run

    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>mock_size_snr</strong>: (bool) Default=False. </LI>
            </UL>



.. autotxclass:: txpipe.ingest.mocks.TXMockTruthPZ
    :members:
    :exclude-members: run

    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>mock_sigma_z</strong>: (float) Default=0.001. </LI>
            </UL>



.. autotxclass:: txpipe.ingest.gcr.TXMetacalGCRInput
    :members:
    :exclude-members: run

    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>cat_name</strong>: (str) Required. </LI>
            <LI><strong>single_tract</strong>: (str) Default=. </LI>
            <LI><strong>length</strong>: (int) Default=0. </LI>
            <LI><strong>table_dir</strong>: (str) Default=. </LI>
            <LI><strong>data_release</strong>: (str) Default=. </LI>
            </UL>



.. autotxclass:: txpipe.ingest.gcr.TXIngestStars
    :members:
    :exclude-members: run

    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>single_tract</strong>: (str) Default=. </LI>
            <LI><strong>cat_name</strong>: (str) Required. </LI>
            <LI><strong>length</strong>: (int) Default=0. </LI>
            </UL>



.. autotxclass:: txpipe.ingest.redmagic.TXIngestRedmagic
    :members:
    :exclude-members: run

    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>lens_zbin_edges</strong>: (list) Default=[<class 'float'>]. </LI>
            <LI><strong>chunk_rows</strong>: (int) Default=100000. </LI>
            <LI><strong>zmin</strong>: (float) Default=0.0. </LI>
            <LI><strong>zmax</strong>: (float) Default=3.0. </LI>
            <LI><strong>dz</strong>: (float) Default=0.01. </LI>
            <LI><strong>bands</strong>: (str) Default=grizy. </LI>
            </UL>



.. autotxclass:: txpipe.simulation.TXLogNormalGlass
    :members:
    :exclude-members: run

    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>num_dens</strong>: (NoneType) Default=None. </LI>
            <LI><strong>zmin</strong>: (float) Default=0.0. </LI>
            <LI><strong>zmax</strong>: (float) Default=2.0. </LI>
            <LI><strong>dx</strong>: (int) Default=100. </LI>
            <LI><strong>bias0</strong>: (float) Default=2.0. </LI>
            <LI><strong>alpha_bz</strong>: (float) Default=0.0. </LI>
            <LI><strong>zpivot</strong>: (float) Default=0.6. </LI>
            <LI><strong>shift</strong>: (float) Default=1.0. </LI>
            <LI><strong>contaminate</strong>: (bool) Default=False. </LI>
            <LI><strong>random_seed</strong>: (int) Default=0. </LI>
            <LI><strong>cl_optional_file</strong>: (str) Default=none. </LI>
            <LI><strong>ell_binned_min</strong>: (float) Default=0.1. </LI>
            <LI><strong>ell_binned_max</strong>: (float) Default=500000.0. </LI>
            <LI><strong>ell_binned_nbins</strong>: (int) Default=100. </LI>
            <LI><strong>output_density_shell_maps</strong>: (bool) Default=False. </LI>
            </UL>


