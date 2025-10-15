Clusters Extensions
===================

These stages are written for TXPipe extension projects by the clusters working group.

* :py:class:`~txpipe.extensions.cluster_counts.bin_cluster.CLClusterBinningRedshiftRichness` - Stage CLClusterBinningRedshiftRichness

* :py:class:`~txpipe.extensions.cluster_counts.sources_select_compute.CLClusterShearCatalogs` - Parameters

* :py:class:`~txpipe.extensions.cluster_counts.make_ensemble_profile.CLClusterEnsembleProfiles` - Stage CLClusterEnsembleProfiles

* :py:class:`~txpipe.extensions.cluster_counts.convert_to_sacc.CLClusterSACC` - Stage CLClusterSACC

* :py:class:`~txpipe.extensions.twopoint_cluster.TXTwoPointCluster` - TXPipe task for measuring two-point correlation functions



.. autotxclass:: txpipe.extensions.cluster_counts.bin_cluster.CLClusterBinningRedshiftRichness
    :members:
    :exclude-members: run

    Inputs: 

    - cluster_catalog: HDFFile

    Outputs: 

    - cluster_catalog_tomography: HDFFile
    
    Parallel: No - Serial


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>zedge</strong>: (list) Default=[0.2, 0.4, 0.6, 0.8, 1.0]. </LI>
            <LI><strong>richedge</strong>: (list) Default=[5.0, 10.0, 20.0]. </LI>
            <LI><strong>initial_size</strong>: (int) Default=100000. </LI>
            <LI><strong>chunk_rows</strong>: (int) Default=100000. </LI>
            </UL>



.. autotxclass:: txpipe.extensions.cluster_counts.sources_select_compute.CLClusterShearCatalogs
    :members:
    :exclude-members: run

    Inputs: 

    - cluster_catalog: HDFFile
    - shear_catalog: ShearCatalog
    - fiducial_cosmology: FiducialCosmology
    - shear_tomography_catalog: TomographyCatalog
    - source_photoz_pdfs: PhotozPDFFile

    Outputs: 

    - cluster_shear_catalogs: HDFFile
    
    Parallel: Yes - MPI


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>chunk_rows</strong>: (int) Default=100000. </LI>
            <LI><strong>max_radius</strong>: (float) Default=10.0. </LI>
            <LI><strong>delta_z</strong>: (float) Default=0.1. </LI>
            <LI><strong>redshift_cut_criterion</strong>: (str) Default=zmode. </LI>
            <LI><strong>redshift_weight_criterion</strong>: (str) Default=zmode. </LI>
            <LI><strong>redshift_cut_criterion_pdf_fraction</strong>: (float) Default=0.9. </LI>
            <LI><strong>subtract_mean_shear</strong>: (bool) Default=False. </LI>
            <LI><strong>coordinate_system</strong>: (str) Default=celestial. </LI>
            <LI><strong>use_true_shear</strong>: (bool) Default=False. </LI>
            </UL>



.. autotxclass:: txpipe.extensions.cluster_counts.make_ensemble_profile.CLClusterEnsembleProfiles
    :members:
    :exclude-members: run

    Inputs: 

    - cluster_catalog_tomography: HDFFile
    - fiducial_cosmology: FiducialCosmology
    - cluster_shear_catalogs: HDFFile

    Outputs: 

    - cluster_profiles: PickleFile
    
    Parallel: Yes - MPI


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



.. autotxclass:: txpipe.extensions.cluster_counts.convert_to_sacc.CLClusterSACC
    :members:
    :exclude-members: run

    Inputs: 

    - cluster_profiles: PickleFile

    Outputs: 

    - cluster_sacc_catalog: SACCFile
    
    Parallel: Yes - MPI


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>r_min</strong>: (float) Default=0.2. </LI>
            <LI><strong>r_max</strong>: (float) Default=5.0. </LI>
            </UL>



.. autotxclass:: txpipe.extensions.twopoint_cluster.TXTwoPointCluster
    :members:
    :exclude-members: run

    Inputs: 

    - cluster_data_catalog: HDFFile
    - cluster_random_catalog: HDFFile

    Outputs: 

    - cluster_twopoint_real: SACCFile
    
    Parallel: Yes - MPI


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>redshift_bin_edges</strong>: (list) Default=[0.4, 0.8, 1.2]. </LI>
            <LI><strong>richness_bin_edges</strong>: (list) Default=[20, 30, 200]. </LI>
            <LI><strong>nbins</strong>: (int) Default=20. </LI>
            <LI><strong>min_sep</strong>: (float) Default=0.1. </LI>
            <LI><strong>max_sep</strong>: (float) Default=250.0. </LI>
            <LI><strong>units</strong>: (str) Default=arcmin. </LI>
            <LI><strong>binning_scale</strong>: (str) Default=Log. </LI>
            </UL>


