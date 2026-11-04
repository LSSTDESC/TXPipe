Weights
=======

These stages deal with weighting the lens sample

* :py:class:`~txpipe.lssweights.TXLSSDensityBase` - Base class for LSS systematics

* :py:class:`~txpipe.lssweights.TXLSSDensityNullTests` - Compute density correlations between lens sample density and survey property maps

* :py:class:`~txpipe.lssweights.TXLSSWeights` - Base class for LSS systematic weights

* :py:class:`~txpipe.lssweights.TXLSSWeightsLinBinned` - Compute LSS systematic weights using simultanious linear regression on the binned

* :py:class:`~txpipe.lssweights.TXLSSWeightsLinPix` - Compute LSS systematic weights using simultanious linear regression at the

* :py:class:`~txpipe.lssweights.TXLSSWeightsUnit` - Assign weight=1 to all lens objects



.. autotxclass:: txpipe.lssweights.TXLSSDensityBase
    :members:
    :exclude-members: run

    Inputs: 

    - lens_maps: MapsFile
    - convergence_maps: MapsFile
    - source_maps: MapsFile
    - mask: MapsFile

    Outputs: 

    - map_systematic_correlations: FileCollection
    
    Parallel: No - Serial


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>supreme_path_root</strong>: (str) Default=. Root path for supreme files.</LI>
            <LI><strong>nbin</strong>: (int) Default=20. Number of tomographic bins.</LI>
            <LI><strong>outlier_fraction</strong>: (float) Default=0.01. Fraction of outliers to exclude.</LI>
            <LI><strong>allow_weighted_input</strong>: (bool) Default=False. Allow weighted input catalogs.</LI>
            <LI><strong>nside_coverage</strong>: (int) Default=32. HEALPix nside for coverage maps.</LI>
            <LI><strong>fill_missing_pix</strong>: (bool) Default=False. If True will fill any pixels that are in the mask but dont have SP maps with teh mean SP map value (use with care)</LI>
            <LI><strong>equal_area_bins</strong>: (bool) Default=True. Use equal area bins in 1d correlations, instead of equal spacing</LI>
            </UL>



.. autotxclass:: txpipe.lssweights.TXLSSDensityNullTests
    :members:
    :exclude-members: run

    Inputs: 

    - binned_lens_catalog_unweighted: TomographyCatalog
    - mask: MapsFile
    - lens_photoz_stack: QPNOfZFile
    - fiducial_cosmology: FiducialCosmology

    Outputs: 

    - lss_density_plots: FileCollection
    - unweighted_density_correlation: HDFFile
    
    Parallel: Yes - MPI


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>supreme_path_root</strong>: (str) Default=. Root path for supreme files.</LI>
            <LI><strong>nbin</strong>: (int) Default=20. Number of tomographic bins.</LI>
            <LI><strong>outlier_fraction</strong>: (float) Default=0.01. Fraction of outliers to exclude.</LI>
            <LI><strong>allow_weighted_input</strong>: (bool) Default=False. Allow weighted input catalogs.</LI>
            <LI><strong>nside_coverage</strong>: (int) Default=32. HEALPix nside for coverage maps.</LI>
            <LI><strong>fill_missing_pix</strong>: (bool) Default=False. If True will fill any pixels that are in the mask but dont have SP maps with teh mean SP map value (use with care)</LI>
            <LI><strong>equal_area_bins</strong>: (bool) Default=True. Use equal area bins in 1d correlations, instead of equal spacing</LI>
            <LI><strong>simple_cov</strong>: (bool) Default=False. if True will use a diagonal shot noise only covariance for the 1d relations</LI>
            <LI><strong>diag_blocks_only</strong>: (bool) Default=True. If True, will compute only the diagonal blocks of the 1D covariance matrix (no correlation between SP maps)</LI>
            <LI><strong>b0</strong>: (list) Default=[1.0]. Galaxy bias values.</LI>
            </UL>



.. autotxclass:: txpipe.lssweights.TXLSSWeights
    :members:
    :exclude-members: run

    Inputs: 

    - binned_lens_catalog_unweighted: BinnedCatalog
    - lens_tomography_catalog_unweighted: TomographyCatalog
    - mask: MapsFile
    - unweighted_density_correlation: HDFFile

    Outputs: 

    - lss_weight_summary: FileCollection
    - weighted_density_correlation: HDFFile
    - lss_weight_maps: MapsFile
    - binned_lens_catalog: BinnedCatalog
    - lens_tomography_catalog: HDFFile
    
    Parallel: No - Serial


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>supreme_path_root</strong>: (str) Default=. Root path for supreme files.</LI>
            <LI><strong>nbin</strong>: (int) Default=20. Number of tomographic bins.</LI>
            <LI><strong>outlier_fraction</strong>: (float) Default=0.01. Fraction of outliers to exclude.</LI>
            <LI><strong>allow_weighted_input</strong>: (bool) Default=False. Allow weighted input catalogs.</LI>
            <LI><strong>nside_coverage</strong>: (int) Default=32. HEALPix nside for coverage maps.</LI>
            <LI><strong>fill_missing_pix</strong>: (bool) Default=False. If True will fill any pixels that are in the mask but dont have SP maps with teh mean SP map value (use with care)</LI>
            <LI><strong>equal_area_bins</strong>: (bool) Default=True. Use equal area bins in 1d correlations, instead of equal spacing</LI>
            </UL>



.. autotxclass:: txpipe.lssweights.TXLSSWeightsLinBinned
    :members:
    :exclude-members: run

    Inputs: 

    - binned_lens_catalog_unweighted: BinnedCatalog
    - lens_tomography_catalog_unweighted: TomographyCatalog
    - mask: MapsFile
    - unweighted_density_correlation: HDFFile

    Outputs: 

    - lss_weight_summary: FileCollection
    - weighted_density_correlation: HDFFile
    - lss_weight_maps: MapsFile
    - binned_lens_catalog: BinnedCatalog
    - lens_tomography_catalog: HDFFile
    
    Parallel: No - Serial


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>supreme_path_root</strong>: (str) Default=. Root path for supreme files.</LI>
            <LI><strong>nbin</strong>: (int) Default=20. Number of tomographic bins.</LI>
            <LI><strong>outlier_fraction</strong>: (float) Default=0.01. Fraction of outliers to exclude.</LI>
            <LI><strong>allow_weighted_input</strong>: (bool) Default=False. Allow weighted input catalogs.</LI>
            <LI><strong>nside_coverage</strong>: (int) Default=32. HEALPix nside for coverage maps.</LI>
            <LI><strong>fill_missing_pix</strong>: (bool) Default=False. If True will fill any pixels that are in the mask but dont have SP maps with teh mean SP map value (use with care)</LI>
            <LI><strong>equal_area_bins</strong>: (bool) Default=True. Use equal area bins in 1d correlations, instead of equal spacing</LI>
            <LI><strong>pvalue_threshold</strong>: (float) Default=0.05. max p-value for maps to be included in the corrected (a very simple form of regularization)</LI>
            </UL>



.. autotxclass:: txpipe.lssweights.TXLSSWeightsLinPix
    :members:
    :exclude-members: run

    Inputs: 

    - binned_lens_catalog_unweighted: BinnedCatalog
    - lens_tomography_catalog_unweighted: TomographyCatalog
    - mask: MapsFile
    - unweighted_density_correlation: HDFFile

    Outputs: 

    - lss_weight_summary: FileCollection
    - weighted_density_correlation: HDFFile
    - lss_weight_maps: MapsFile
    - binned_lens_catalog: BinnedCatalog
    - lens_tomography_catalog: HDFFile
    
    Parallel: No - Serial


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>supreme_path_root</strong>: (str) Default=. Root path for supreme files.</LI>
            <LI><strong>nbin</strong>: (int) Default=20. Number of tomographic bins.</LI>
            <LI><strong>outlier_fraction</strong>: (float) Default=0.01. Fraction of outliers to exclude.</LI>
            <LI><strong>allow_weighted_input</strong>: (bool) Default=False. Allow weighted input catalogs.</LI>
            <LI><strong>nside_coverage</strong>: (int) Default=32. HEALPix nside for coverage maps.</LI>
            <LI><strong>fill_missing_pix</strong>: (bool) Default=False. If True will fill any pixels that are in the mask but dont have SP maps with teh mean SP map value (use with care)</LI>
            <LI><strong>equal_area_bins</strong>: (bool) Default=True. Use equal area bins in 1d correlations, instead of equal spacing</LI>
            <LI><strong>pvalue_threshold</strong>: (float) Default=0.05. max p-value for maps to be corrected</LI>
            <LI><strong>regression_class</strong>: (str) Default=LinearRegression. sklearn.linear_model class to use in regression</LI>
            </UL>



.. autotxclass:: txpipe.lssweights.TXLSSWeightsUnit
    :members:
    :exclude-members: run

    Inputs: 

    - binned_lens_catalog_unweighted: BinnedCatalog
    - lens_tomography_catalog_unweighted: TomographyCatalog
    - mask: MapsFile

    Outputs: 

    - lss_weight_summary: FileCollection
    - weighted_density_correlation: HDFFile
    - lss_weight_maps: MapsFile
    - binned_lens_catalog: BinnedCatalog
    - lens_tomography_catalog: BinnedCatalog
    
    Parallel: No - Serial


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>nside_coverage</strong>: (int) Default=32. HEALPix nside for coverage maps.</LI>
            </UL>


