Lens Selection
==============

These stages deal with selecting lens objects and assigning them to
tomographic bins.

* :py:class:`~txpipe.lens_selector.TXBaseLensSelector` - Base class for lens object selection, using the BOSS Target Selection.

* :py:class:`~txpipe.lens_selector.TXTruthLensSelector` - Select lens objects based on true redshifts and BOSS criteria

* :py:class:`~txpipe.lens_selector.TXMeanLensSelector` - Select lens objects based on mean redshifts and BOSS criteria

* :py:class:`~txpipe.lens_selector.TXModeLensSelector` - Select lens objects based on best-fit redshifts and BOSS criteria

* :py:class:`~txpipe.lens_selector.TXCustomLensSelector` - Select lens objects based on best-fit redshifts and BOSS criteria

* :py:class:`~txpipe.lens_selector.TXRandomForestLensSelector` - Select lens objects and assign them to bins using a random forest classifier.

* :py:class:`~txpipe.lens_selector.TXDESISelector` - Select lens galaxies from a DESI spectroscopic FITS catalog.

* :py:class:`~txpipe.lens_selector.TXDESIMockSelector` - Select lens galaxies from a DESI mock Parquet catalog.

* :py:class:`~txpipe.lens_selector.TXDESIMultiMockSelector` - Select lens galaxies from a list of DESI mock Parquet catalogs.



.. autotxclass:: txpipe.lens_selector.TXBaseLensSelector
    :members:
    :exclude-members: run

    Inputs: None

    Outputs: 

    - lens_tomography_catalog_unweighted: TomographyCatalog
    
    Parallel: Yes - MPI


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>verbose</strong>: (bool) Default=False. Enable verbose output for lens selection.</LI>
            <LI><strong>chunk_rows</strong>: (int) Default=10000. Number of rows to process in each chunk.</LI>
            <LI><strong>lens_zbin_edges</strong>: (list) Default=[&lt;class &#x27;float&#x27;&gt;]. Edges of lens redshift bins.</LI>
            <LI><strong>cperp_cut</strong>: (float) Default=0.2. cperp cut for BOSS selection.</LI>
            <LI><strong>r_cpar_cut</strong>: (float) Default=13.5. r_cpar cut for BOSS selection.</LI>
            <LI><strong>r_lo_cut</strong>: (float) Default=16.0. Lower r-band magnitude cut.</LI>
            <LI><strong>r_hi_cut</strong>: (float) Default=19.6. Upper r-band magnitude cut.</LI>
            <LI><strong>i_lo_cut</strong>: (float) Default=17.5. Lower i-band magnitude cut.</LI>
            <LI><strong>i_hi_cut</strong>: (float) Default=19.9. Upper i-band magnitude cut.</LI>
            <LI><strong>r_i_cut</strong>: (float) Default=2.0. r-i color cut.</LI>
            <LI><strong>random_seed</strong>: (int) Default=42. Random seed for reproducibility.</LI>
            <LI><strong>selection_type</strong>: (str) Default=boss. Type of lens selection (e.g., boss).</LI>
            <LI><strong>maglim_band</strong>: (str) Default=i. Band for magnitude limit.</LI>
            <LI><strong>maglim_limit</strong>: (float) Default=24.1. Magnitude limit value.</LI>
            <LI><strong>extra_cols</strong>: (list) Default=[&#x27;&#x27;]. Extra columns to include in output.</LI>
            <LI><strong>apply_mask</strong>: (bool) Default=False. Whether to apply a mask to the selection.</LI>
            </UL>



.. autotxclass:: txpipe.lens_selector.TXTruthLensSelector
    :members:
    :exclude-members: run

    Inputs: 

    - photometry_catalog: PhotometryCatalog

    Outputs: 

    - lens_tomography_catalog_unweighted: TomographyCatalog
    
    Parallel: Yes - MPI


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>verbose</strong>: (bool) Default=False. Enable verbose output for lens selection.</LI>
            <LI><strong>chunk_rows</strong>: (int) Default=10000. Number of rows to process in each chunk.</LI>
            <LI><strong>lens_zbin_edges</strong>: (list) Default=[&lt;class &#x27;float&#x27;&gt;]. Edges of lens redshift bins.</LI>
            <LI><strong>cperp_cut</strong>: (float) Default=0.2. cperp cut for BOSS selection.</LI>
            <LI><strong>r_cpar_cut</strong>: (float) Default=13.5. r_cpar cut for BOSS selection.</LI>
            <LI><strong>r_lo_cut</strong>: (float) Default=16.0. Lower r-band magnitude cut.</LI>
            <LI><strong>r_hi_cut</strong>: (float) Default=19.6. Upper r-band magnitude cut.</LI>
            <LI><strong>i_lo_cut</strong>: (float) Default=17.5. Lower i-band magnitude cut.</LI>
            <LI><strong>i_hi_cut</strong>: (float) Default=19.9. Upper i-band magnitude cut.</LI>
            <LI><strong>r_i_cut</strong>: (float) Default=2.0. r-i color cut.</LI>
            <LI><strong>random_seed</strong>: (int) Default=42. Random seed for reproducibility.</LI>
            <LI><strong>selection_type</strong>: (str) Default=boss. Type of lens selection (e.g., boss).</LI>
            <LI><strong>maglim_band</strong>: (str) Default=i. Band for magnitude limit.</LI>
            <LI><strong>maglim_limit</strong>: (float) Default=24.1. Magnitude limit value.</LI>
            <LI><strong>extra_cols</strong>: (list) Default=[&#x27;&#x27;]. Extra columns to include in output.</LI>
            <LI><strong>apply_mask</strong>: (bool) Default=False. Whether to apply a mask to the selection.</LI>
            </UL>



.. autotxclass:: txpipe.lens_selector.TXMeanLensSelector
    :members:
    :exclude-members: run

    Inputs: 

    - photometry_catalog: PhotometryCatalog
    - lens_photoz_pdfs: HDFFile

    Outputs: 

    - lens_tomography_catalog_unweighted: TomographyCatalog
    
    Parallel: Yes - MPI


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>verbose</strong>: (bool) Default=False. Enable verbose output for lens selection.</LI>
            <LI><strong>chunk_rows</strong>: (int) Default=10000. Number of rows to process in each chunk.</LI>
            <LI><strong>lens_zbin_edges</strong>: (list) Default=[&lt;class &#x27;float&#x27;&gt;]. Edges of lens redshift bins.</LI>
            <LI><strong>cperp_cut</strong>: (float) Default=0.2. cperp cut for BOSS selection.</LI>
            <LI><strong>r_cpar_cut</strong>: (float) Default=13.5. r_cpar cut for BOSS selection.</LI>
            <LI><strong>r_lo_cut</strong>: (float) Default=16.0. Lower r-band magnitude cut.</LI>
            <LI><strong>r_hi_cut</strong>: (float) Default=19.6. Upper r-band magnitude cut.</LI>
            <LI><strong>i_lo_cut</strong>: (float) Default=17.5. Lower i-band magnitude cut.</LI>
            <LI><strong>i_hi_cut</strong>: (float) Default=19.9. Upper i-band magnitude cut.</LI>
            <LI><strong>r_i_cut</strong>: (float) Default=2.0. r-i color cut.</LI>
            <LI><strong>random_seed</strong>: (int) Default=42. Random seed for reproducibility.</LI>
            <LI><strong>selection_type</strong>: (str) Default=boss. Type of lens selection (e.g., boss).</LI>
            <LI><strong>maglim_band</strong>: (str) Default=i. Band for magnitude limit.</LI>
            <LI><strong>maglim_limit</strong>: (float) Default=24.1. Magnitude limit value.</LI>
            <LI><strong>extra_cols</strong>: (list) Default=[&#x27;&#x27;]. Extra columns to include in output.</LI>
            <LI><strong>apply_mask</strong>: (bool) Default=False. Whether to apply a mask to the selection.</LI>
            </UL>



.. autotxclass:: txpipe.lens_selector.TXModeLensSelector
    :members:
    :exclude-members: run

    Inputs: 

    - photometry_catalog: PhotometryCatalog
    - lens_photoz_pdfs: HDFFile

    Outputs: 

    - lens_tomography_catalog_unweighted: TomographyCatalog
    
    Parallel: Yes - MPI


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>verbose</strong>: (bool) Default=False. Enable verbose output for lens selection.</LI>
            <LI><strong>chunk_rows</strong>: (int) Default=10000. Number of rows to process in each chunk.</LI>
            <LI><strong>lens_zbin_edges</strong>: (list) Default=[&lt;class &#x27;float&#x27;&gt;]. Edges of lens redshift bins.</LI>
            <LI><strong>cperp_cut</strong>: (float) Default=0.2. cperp cut for BOSS selection.</LI>
            <LI><strong>r_cpar_cut</strong>: (float) Default=13.5. r_cpar cut for BOSS selection.</LI>
            <LI><strong>r_lo_cut</strong>: (float) Default=16.0. Lower r-band magnitude cut.</LI>
            <LI><strong>r_hi_cut</strong>: (float) Default=19.6. Upper r-band magnitude cut.</LI>
            <LI><strong>i_lo_cut</strong>: (float) Default=17.5. Lower i-band magnitude cut.</LI>
            <LI><strong>i_hi_cut</strong>: (float) Default=19.9. Upper i-band magnitude cut.</LI>
            <LI><strong>r_i_cut</strong>: (float) Default=2.0. r-i color cut.</LI>
            <LI><strong>random_seed</strong>: (int) Default=42. Random seed for reproducibility.</LI>
            <LI><strong>selection_type</strong>: (str) Default=boss. Type of lens selection (e.g., boss).</LI>
            <LI><strong>maglim_band</strong>: (str) Default=i. Band for magnitude limit.</LI>
            <LI><strong>maglim_limit</strong>: (float) Default=24.1. Magnitude limit value.</LI>
            <LI><strong>extra_cols</strong>: (list) Default=[&#x27;&#x27;]. Extra columns to include in output.</LI>
            <LI><strong>apply_mask</strong>: (bool) Default=False. Whether to apply a mask to the selection.</LI>
            </UL>



.. autotxclass:: txpipe.lens_selector.TXCustomLensSelector
    :members:
    :exclude-members: run

    Inputs: 

    - photometry_catalog: PhotometryCatalog
    - lens_photoz_pdfs: HDFFile
    - mask: MapsFile

    Outputs: 

    - lens_tomography_catalog_unweighted: TomographyCatalog
    
    Parallel: Yes - MPI


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>verbose</strong>: (bool) Default=False. Enable verbose output for lens selection.</LI>
            <LI><strong>chunk_rows</strong>: (int) Default=10000. Number of rows to process in each chunk.</LI>
            <LI><strong>lens_zbin_edges</strong>: (list) Default=[&lt;class &#x27;float&#x27;&gt;]. Edges of lens redshift bins.</LI>
            <LI><strong>cperp_cut</strong>: (float) Default=0.2. cperp cut for BOSS selection.</LI>
            <LI><strong>r_cpar_cut</strong>: (float) Default=13.5. r_cpar cut for BOSS selection.</LI>
            <LI><strong>r_lo_cut</strong>: (float) Default=16.0. Lower r-band magnitude cut.</LI>
            <LI><strong>r_hi_cut</strong>: (float) Default=19.6. Upper r-band magnitude cut.</LI>
            <LI><strong>i_lo_cut</strong>: (float) Default=17.5. Lower i-band magnitude cut.</LI>
            <LI><strong>i_hi_cut</strong>: (float) Default=19.9. Upper i-band magnitude cut.</LI>
            <LI><strong>r_i_cut</strong>: (float) Default=2.0. r-i color cut.</LI>
            <LI><strong>random_seed</strong>: (int) Default=42. Random seed for reproducibility.</LI>
            <LI><strong>selection_type</strong>: (str) Default=boss. Type of lens selection (e.g., boss).</LI>
            <LI><strong>maglim_band</strong>: (str) Default=i. Band for magnitude limit.</LI>
            <LI><strong>maglim_limit</strong>: (float) Default=24.1. Magnitude limit value.</LI>
            <LI><strong>extra_cols</strong>: (list) Default=[&#x27;&#x27;]. Extra columns to include in output.</LI>
            <LI><strong>apply_mask</strong>: (bool) Default=False. Whether to apply a mask to the selection.</LI>
            </UL>



.. autotxclass:: txpipe.lens_selector.TXRandomForestLensSelector
    :members:
    :exclude-members: run

    Inputs: 

    - photometry_catalog: PhotometryCatalog
    - spectroscopic_catalog: HDFFile

    Outputs: 

    - lens_tomography_catalog_unweighted: TomographyCatalog
    
    Parallel: Yes - MPI


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>verbose</strong>: (bool) Default=False. Enable verbose output for lens selection.</LI>
            <LI><strong>chunk_rows</strong>: (int) Default=10000. Number of rows to process in each chunk.</LI>
            <LI><strong>lens_zbin_edges</strong>: (list) Required. Edges of lens redshift bins.</LI>
            <LI><strong>cperp_cut</strong>: (float) Default=0.2. cperp cut for BOSS selection.</LI>
            <LI><strong>r_cpar_cut</strong>: (float) Default=13.5. r_cpar cut for BOSS selection.</LI>
            <LI><strong>r_lo_cut</strong>: (float) Default=16.0. Lower r-band magnitude cut.</LI>
            <LI><strong>r_hi_cut</strong>: (float) Default=19.6. Upper r-band magnitude cut.</LI>
            <LI><strong>i_lo_cut</strong>: (float) Default=17.5. Lower i-band magnitude cut.</LI>
            <LI><strong>i_hi_cut</strong>: (float) Default=19.9. Upper i-band magnitude cut.</LI>
            <LI><strong>r_i_cut</strong>: (float) Default=2.0. r-i color cut.</LI>
            <LI><strong>random_seed</strong>: (int) Default=42. Random seed for reproducibility.</LI>
            <LI><strong>selection_type</strong>: (str) Default=boss. Type of lens selection (e.g., boss).</LI>
            <LI><strong>maglim_band</strong>: (str) Default=i. Band for magnitude limit.</LI>
            <LI><strong>maglim_limit</strong>: (float) Default=24.1. Magnitude limit value.</LI>
            <LI><strong>extra_cols</strong>: (list) Default=[&#x27;&#x27;]. Extra columns to include in output.</LI>
            <LI><strong>apply_mask</strong>: (bool) Default=False. Whether to apply a mask to the selection.</LI>
            <LI><strong>bands</strong>: (list) Default=[&#x27;u&#x27;, &#x27;g&#x27;, &#x27;r&#x27;, &#x27;i&#x27;, &#x27;z&#x27;, &#x27;y&#x27;]. Photometric bands to use for classification.</LI>
            <LI><strong>spec_mag_column_format</strong>: (str) Default=photometry/{band}. Format string for spectroscopic magnitude columns.</LI>
            <LI><strong>spec_redshift_column</strong>: (str) Default=photometry/redshift. Column name for spectroscopic redshift.</LI>
            </UL>



.. autotxclass:: txpipe.lens_selector.TXDESISelector
    :members:
    :exclude-members: run

    Inputs: 

    - desi_catalog: FitsFile
    - mask: MapsFile

    Outputs: 

    - desi_catalog_selected: FitsFile
    
    Parallel: No - Serial


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>chunk_rows</strong>: (int) Default=100000. Rows per chunk.</LI>
            <LI><strong>zmin</strong>: (float) Default=0.0. Minimum spectroscopic redshift.</LI>
            <LI><strong>zmax</strong>: (float) Default=3.0. Maximum spectroscopic redshift.</LI>
            <LI><strong>mag_band</strong>: (str) Default=r. Band to apply magnitude limits in.</LI>
            <LI><strong>apply_mask</strong>: (bool) Default=False. Apply survey mask from the mask input.</LI>
            <LI><strong>ra_col</strong>: (str) Default=ra. RA column name in the DESI catalog.</LI>
            <LI><strong>dec_col</strong>: (str) Default=dec. Dec column name in the DESI catalog.</LI>
            <LI><strong>z_col</strong>: (str) Default=redshift. Spectroscopic redshift column name.</LI>
            <LI><strong>w_col</strong>: (str) Default=weight. Weight column name.</LI>
            <LI><strong>bands</strong>: (str) Default=. Band labels for output magnitudes (one char each).</LI>
            </UL>



.. autotxclass:: txpipe.lens_selector.TXDESIMockSelector
    :members:
    :exclude-members: run

    Inputs: 

    - desi_catalog: ParquetFile
    - mask: MapsFile

    Outputs: 

    - desi_catalog_selected: FitsFile
    
    Parallel: No - Serial


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>chunk_rows</strong>: (int) Default=100000. Rows per chunk.</LI>
            <LI><strong>zmin</strong>: (float) Default=0.0. Minimum spectroscopic redshift.</LI>
            <LI><strong>zmax</strong>: (float) Default=3.0. Maximum spectroscopic redshift.</LI>
            <LI><strong>mag_band</strong>: (str) Default=r. Band to apply magnitude limits in.</LI>
            <LI><strong>apply_mask</strong>: (bool) Default=False. Apply survey mask from the mask input.</LI>
            <LI><strong>ra_col</strong>: (str) Default=ra. RA column name in the DESI catalog.</LI>
            <LI><strong>dec_col</strong>: (str) Default=dec. Dec column name in the DESI catalog.</LI>
            <LI><strong>z_col</strong>: (str) Default=redshift. Spectroscopic redshift column name.</LI>
            <LI><strong>w_col</strong>: (str) Default=weight. Weight column name.</LI>
            <LI><strong>bands</strong>: (str) Default=. Band labels for output magnitudes (one char each).</LI>
            </UL>



.. autotxclass:: txpipe.lens_selector.TXDESIMultiMockSelector
    :members:
    :exclude-members: run

    Inputs: 

    - mask: MapsFile

    Outputs: 

    - desi_catalog_selected: FitsFile
    
    Parallel: No - Serial


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>mock_path_pattern</strong>: (str) Required. Glob pattern for input mock files</LI>
            <LI><strong>chunk_rows</strong>: (int) Default=100000. Rows per chunk.</LI>
            <LI><strong>zmin</strong>: (float) Default=0.0. Minimum spectroscopic redshift.</LI>
            <LI><strong>zmax</strong>: (float) Default=3.0. Maximum spectroscopic redshift.</LI>
            <LI><strong>mag_band</strong>: (str) Default=r. Band to apply magnitude limits in.</LI>
            <LI><strong>apply_mask</strong>: (bool) Default=False. Apply survey mask from the mask input.</LI>
            <LI><strong>ra_col</strong>: (str) Default=ra. RA column name in the DESI catalog.</LI>
            <LI><strong>dec_col</strong>: (str) Default=dec. Dec column name in the DESI catalog.</LI>
            <LI><strong>z_col</strong>: (str) Default=redshift. Spectroscopic redshift column name.</LI>
            <LI><strong>w_col</strong>: (str) Default=weight. Weight column name.</LI>
            <LI><strong>bands</strong>: (str) Default=. Band labels for output magnitudes (one char each).</LI>
            </UL>


