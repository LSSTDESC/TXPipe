Selection
=========

These stages decide which objects in a shear catalog are good enough to
be in the source sample, before they are assigned to tomographic bins. They
write a catalog selection file. See :ref:`Source selection and tomography`.

* :py:class:`~txpipe.source_selection.TXSimpleSelection` - Compute non-redshift related shear catalog cuts.

* :py:class:`~txpipe.source_selection.TXMetaDetectSelection` - Compute source-sample cuts for metadetect catalogs.

* :py:class:`~txpipe.source_selection.TXScalarMetaDetectSelection` - Compute source-sample cuts for scalar metadetect catalogs.

* :py:class:`~txpipe.source_selection.TXMetacalSelection` - Compute source-sample cuts for metacal catalogs.

* :py:class:`~txpipe.source_selection.TXLensfitSelection` - Compute source-sample cuts for lensfit catalogs.

* :py:class:`~txpipe.source_selection.TXHSCSelection` - Compute source-sample cuts for HSC catalogs.

* :py:class:`~txpipe.source_selection.TXDP2RubinMetadetectSelection` - Compute source-sample cuts for the Rubin DP2 metadetect catalogs.

* :py:class:`~txpipe.source_selection.TXDP2DESCSelection` - Compute source-sample cuts for the DESC-processed DP2 scalar metadetect catalogs.



.. autotxclass:: txpipe.source_selection.TXSimpleSelection
    :members:
    :exclude-members: run

    Inputs: 

    - shear_catalog: ShearCatalog

    Outputs: 

    - shear_selection: CatalogSelection
    
    Parallel: Yes - MPI


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>chunk_rows</strong>: (int) Default=10000. Number of rows to process at once</LI>
            <LI><strong>s2n_min</strong>: (float) Default=10.0. Minimum s/n for selection</LI>
            <LI><strong>T_min</strong>: (float) Default=0.5. Minimum T/Tpsf for selection</LI>
            <LI><strong>verbose</strong>: (bool) Default=False. Whether to print out lots of information</LI>
            </UL>



.. autotxclass:: txpipe.source_selection.TXMetaDetectSelection
    :members:
    :exclude-members: run

    Inputs: 

    - shear_catalog: ShearCatalog

    Outputs: 

    - shear_selection: CatalogSelection
    
    Parallel: Yes - MPI


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>chunk_rows</strong>: (int) Default=10000. Number of rows to process at once</LI>
            <LI><strong>s2n_min</strong>: (float) Default=10.0. Minimum s/n for selection</LI>
            <LI><strong>T_min</strong>: (float) Default=0.5. Minimum T/Tpsf for selection</LI>
            <LI><strong>verbose</strong>: (bool) Default=False. Whether to print out lots of information</LI>
            </UL>



.. autotxclass:: txpipe.source_selection.TXScalarMetaDetectSelection
    :members:
    :exclude-members: run

    Inputs: 

    - shear_catalog: ShearCatalog

    Outputs: 

    - shear_selection: CatalogSelection
    
    Parallel: Yes - MPI


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>chunk_rows</strong>: (int) Default=10000. Number of rows to process at once</LI>
            <LI><strong>s2n_min</strong>: (float) Default=10.0. Minimum s/n for selection</LI>
            <LI><strong>T_min</strong>: (float) Default=0.5. Minimum T/Tpsf for selection</LI>
            <LI><strong>verbose</strong>: (bool) Default=False. Whether to print out lots of information</LI>
            </UL>



.. autotxclass:: txpipe.source_selection.TXMetacalSelection
    :members:
    :exclude-members: run

    Inputs: 

    - shear_catalog: ShearCatalog

    Outputs: 

    - shear_selection: CatalogSelection
    
    Parallel: Yes - MPI


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>chunk_rows</strong>: (int) Default=10000. Number of rows to process at once</LI>
            <LI><strong>s2n_min</strong>: (float) Default=10.0. Minimum s/n for selection</LI>
            <LI><strong>T_min</strong>: (float) Default=0.5. Minimum T/Tpsf for selection</LI>
            <LI><strong>verbose</strong>: (bool) Default=False. Whether to print out lots of information</LI>
            </UL>



.. autotxclass:: txpipe.source_selection.TXLensfitSelection
    :members:
    :exclude-members: run

    Inputs: 

    - shear_catalog: ShearCatalog

    Outputs: 

    - shear_selection: CatalogSelection
    
    Parallel: Yes - MPI


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>chunk_rows</strong>: (int) Default=10000. Number of rows to process at once</LI>
            <LI><strong>s2n_min</strong>: (float) Default=10.0. Minimum s/n for selection</LI>
            <LI><strong>T_min</strong>: (float) Default=0.5. Minimum T/Tpsf for selection</LI>
            <LI><strong>verbose</strong>: (bool) Default=False. Whether to print out lots of information</LI>
            </UL>



.. autotxclass:: txpipe.source_selection.TXHSCSelection
    :members:
    :exclude-members: run

    Inputs: 

    - shear_catalog: ShearCatalog

    Outputs: 

    - shear_selection: CatalogSelection
    
    Parallel: Yes - MPI


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>chunk_rows</strong>: (int) Default=10000. Number of rows to process at once</LI>
            <LI><strong>s2n_min</strong>: (float) Default=10.0. Minimum s/n for selection</LI>
            <LI><strong>T_min</strong>: (float) Default=0.5. Minimum T/Tpsf for selection</LI>
            <LI><strong>verbose</strong>: (bool) Default=False. Whether to print out lots of information</LI>
            <LI><strong>max_shear_cut</strong>: (float) Default=0.0. Maximum shear value for object selection</LI>
            </UL>



.. autotxclass:: txpipe.source_selection.TXDP2RubinMetadetectSelection
    :members:
    :exclude-members: run

    Inputs: 

    - shear_catalog: ShearCatalog

    Outputs: 

    - shear_selection: CatalogSelection
    
    Parallel: Yes - MPI


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>chunk_rows</strong>: (int) Default=10000. Number of rows to process at once</LI>
            <LI><strong>s2n_min</strong>: (float) Default=10.0. Minimum gauss_snr for object selection</LI>
            <LI><strong>T_min</strong>: (float) Default=0.5. Minimum T/Tpsf for selection</LI>
            <LI><strong>verbose</strong>: (bool) Default=False. Whether to print out lots of information</LI>
            <LI><strong>mfrac_max</strong>: (float) Default=0.1. Maximum mask fraction for object selection</LI>
            <LI><strong>T_max</strong>: (float) Default=20.0. Maximum gauss_T for object selection</LI>
            <LI><strong>Tratio_min</strong>: (float) Default=0.5. Minimum gauss_T / gauss_psfReconvolved_T for object selection</LI>
            <LI><strong>i_min</strong>: (float) Default=18.2. Minimum i-band magnitude</LI>
            <LI><strong>i_max</strong>: (float) Default=23.5. Maximum i-band magnitude</LI>
            <LI><strong>r_min</strong>: (float) Default=15.2. Minimum r-band magnitude</LI>
            <LI><strong>g_min</strong>: (float) Default=15.2. Minimum g-band magnitude</LI>
            <LI><strong>rmi_min</strong>: (float) Default=-1.5. Minimum r-i color for object selection</LI>
            <LI><strong>rmi_max</strong>: (float) Default=4.0. Maximum r-i color for object selection</LI>
            <LI><strong>imz_min</strong>: (float) Default=-1.5. Minimum i-z color for object selection</LI>
            <LI><strong>imz_max</strong>: (float) Default=4.0. Maximum i-z color for object selection</LI>
            </UL>



.. autotxclass:: txpipe.source_selection.TXDP2DESCSelection
    :members:
    :exclude-members: run

    Inputs: 

    - shear_catalog: ShearCatalog

    Outputs: 

    - shear_selection: CatalogSelection
    
    Parallel: Yes - MPI


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>chunk_rows</strong>: (int) Default=10000. Number of rows to process at once</LI>
            <LI><strong>s2n_min</strong>: (float) Default=10.0. Minimum signal-to-noise ratio for object selection</LI>
            <LI><strong>T_min</strong>: (float) Default=0.5. Minimum T/Tpsf for selection</LI>
            <LI><strong>verbose</strong>: (bool) Default=False. Whether to print out lots of information</LI>
            <LI><strong>max_psf_g</strong>: (float) Default=0.05. Maximum PSF ellipticity for object selection</LI>
            <LI><strong>rmi_min</strong>: (float) Default=-2.0. Minimum r-i color for object selection</LI>
            <LI><strong>rmi_max</strong>: (float) Default=3.0. Maximum r-i color for object selection</LI>
            <LI><strong>imz_min</strong>: (float) Default=-2.0. Minimum i-z color for object selection</LI>
            <LI><strong>imz_max</strong>: (float) Default=3.0. Maximum i-z color for object selection</LI>
            <LI><strong>T_max</strong>: (float) Default=20.0. T threshold for object selection</LI>
            <LI><strong>Tratio_min</strong>: (float) Default=0.5. Minimum T ratio for object selection</LI>
            <LI><strong>Tratio_max</strong>: (float) Default=20.0. Maximum T ratio for object selection</LI>
            <LI><strong>s2n_max</strong>: (float) Default=100.0. Maximum signal-to-noise ratio for object selection</LI>
            <LI><strong>mfrac_max</strong>: (float) Default=0.1. Maximum mask fraction for object selection</LI>
            <LI><strong>i_min</strong>: (float) Default=18.2. Minimum i-band magnitude</LI>
            <LI><strong>i_max</strong>: (float) Default=23.5. Maximum i-band magnitude</LI>
            <LI><strong>r_min</strong>: (float) Default=15.2. Minimum g-band magnitude</LI>
            <LI><strong>r_max</strong>: (float) Default=30.0. Maximum g-band magnitude</LI>
            </UL>


