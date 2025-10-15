Selection
=========

These stages deal with selection objects and assigning them to tomographic
bins.

* :py:class:`~txpipe.source_selector.TXSourceSelectorBase` - Base stage for source selection using S/N, size, and flag cuts and tomography

* :py:class:`~txpipe.source_selector.TXSourceSelectorMetacal` - Source selection and tomography for metacal catalogs

* :py:class:`~txpipe.source_selector.TXSourceSelectorMetadetect` - Source selection and tomography for metadetect catalogs

* :py:class:`~txpipe.source_selector.TXSourceSelectorLensfit` - Source selection and tomography for lensfit catalogs

* :py:class:`~txpipe.source_selector.TXSourceSelectorSimple` - Source selection and tomography for mock catalogs that do not

* :py:class:`~txpipe.source_selector.TXSourceSelectorHSC` - Source selection and tomography for HSC catalogs

* :py:class:`~txpipe.lens_selector.TXBaseLensSelector` - Base class for lens object selection, using the BOSS Target Selection.

* :py:class:`~txpipe.lens_selector.TXTruthLensSelector` - Select lens objects based on true redshifts and BOSS criteria

* :py:class:`~txpipe.lens_selector.TXMeanLensSelector` - Select lens objects based on mean redshifts and BOSS criteria

* :py:class:`~txpipe.lens_selector.TXModeLensSelector` - Select lens objects based on best-fit redshifts and BOSS criteria

* :py:class:`~txpipe.lens_selector.TXCustomLensSelector` - Select lens objects based on best-fit redshifts and BOSS criteria

* :py:class:`~txpipe.lens_selector.TXRandomForestLensSelector` - Stage TXRandomForestLensSelector



.. autotxclass:: txpipe.source_selector.TXSourceSelectorBase
    :members:
    :exclude-members: run

    Inputs: 

    - shear_catalog: ShearCatalog
    - spectroscopic_catalog: HDFFile

    Outputs: 

    - shear_tomography_catalog: TomographyCatalog
    
    Parallel: Yes - MPI


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>input_pz</strong>: (bool) Default=False. Whether to use input photo-z posteriors</LI>
            <LI><strong>true_z</strong>: (bool) Default=False. Whether to use true redshifts instead of photo-z</LI>
            <LI><strong>bands</strong>: (str) Default=riz. Bands from the catalog to use for selection</LI>
            <LI><strong>verbose</strong>: (bool) Default=False. Whether to print verbose output</LI>
            <LI><strong>T_cut</strong>: (float) Required. Size cut threshold for object selection</LI>
            <LI><strong>s2n_cut</strong>: (float) Required. Signal-to-noise cut threshold for object selection</LI>
            <LI><strong>chunk_rows</strong>: (int) Default=10000. Number of rows to process in each chunk</LI>
            <LI><strong>source_zbin_edges</strong>: (list) Required. Redshift bin edges for source tomography</LI>
            <LI><strong>random_seed</strong>: (int) Default=42. Random seed for reproducibility</LI>
            <LI><strong>spec_mag_column_format</strong>: (str) Default=photometry/{band}. Format string for spectroscopic magnitude columns</LI>
            <LI><strong>spec_redshift_column</strong>: (str) Default=photometry/redshift. Column name for spectroscopic redshifts</LI>
            </UL>



.. autotxclass:: txpipe.source_selector.TXSourceSelectorMetacal
    :members:
    :exclude-members: run

    Inputs: 

    - shear_catalog: ShearCatalog
    - spectroscopic_catalog: HDFFile

    Outputs: 

    - shear_tomography_catalog: TomographyCatalog
    
    Parallel: Yes - MPI


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>input_pz</strong>: (bool) Default=False. Whether to use input photo-z posteriors</LI>
            <LI><strong>true_z</strong>: (bool) Default=False. Whether to use true redshifts instead of photo-z</LI>
            <LI><strong>bands</strong>: (str) Default=riz. Bands from the catalog to use for selection</LI>
            <LI><strong>verbose</strong>: (bool) Default=False. Whether to print verbose output</LI>
            <LI><strong>T_cut</strong>: (float) Required. Size cut threshold for object selection</LI>
            <LI><strong>s2n_cut</strong>: (float) Required. Signal-to-noise cut threshold for object selection</LI>
            <LI><strong>chunk_rows</strong>: (int) Default=10000. Number of rows to process in each chunk</LI>
            <LI><strong>source_zbin_edges</strong>: (list) Required. Redshift bin edges for source tomography</LI>
            <LI><strong>random_seed</strong>: (int) Default=42. Random seed for reproducibility</LI>
            <LI><strong>spec_mag_column_format</strong>: (str) Default=photometry/{band}. Format string for spectroscopic magnitude columns</LI>
            <LI><strong>spec_redshift_column</strong>: (str) Default=photometry/redshift. Column name for spectroscopic redshifts</LI>
            <LI><strong>delta_gamma</strong>: (float) Required. Delta gamma value for metacal response calculation</LI>
            <LI><strong>use_diagonal_response</strong>: (bool) Default=False. Whether to use only diagonal elements of the response matrix</LI>
            </UL>



.. autotxclass:: txpipe.source_selector.TXSourceSelectorMetadetect
    :members:
    :exclude-members: run

    Inputs: 

    - shear_catalog: ShearCatalog
    - spectroscopic_catalog: HDFFile

    Outputs: 

    - shear_tomography_catalog: TomographyCatalog
    
    Parallel: Yes - MPI


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>input_pz</strong>: (bool) Default=False. Whether to use input photo-z posteriors</LI>
            <LI><strong>true_z</strong>: (bool) Default=False. Whether to use true redshifts instead of photo-z</LI>
            <LI><strong>bands</strong>: (str) Default=riz. Bands from the catalog to use for selection</LI>
            <LI><strong>verbose</strong>: (bool) Default=False. Whether to print verbose output</LI>
            <LI><strong>T_cut</strong>: (float) Required. Size cut threshold for object selection</LI>
            <LI><strong>s2n_cut</strong>: (float) Required. Signal-to-noise cut threshold for object selection</LI>
            <LI><strong>chunk_rows</strong>: (int) Default=10000. Number of rows to process in each chunk</LI>
            <LI><strong>source_zbin_edges</strong>: (list) Required. Redshift bin edges for source tomography</LI>
            <LI><strong>random_seed</strong>: (int) Default=42. Random seed for reproducibility</LI>
            <LI><strong>spec_mag_column_format</strong>: (str) Default=photometry/{band}. Format string for spectroscopic magnitude columns</LI>
            <LI><strong>spec_redshift_column</strong>: (str) Default=photometry/redshift. Column name for spectroscopic redshifts</LI>
            <LI><strong>delta_gamma</strong>: (float) Required. Delta gamma value for metadetect response calculation</LI>
            </UL>



.. autotxclass:: txpipe.source_selector.TXSourceSelectorLensfit
    :members:
    :exclude-members: run

    Inputs: 

    - shear_catalog: ShearCatalog
    - spectroscopic_catalog: HDFFile

    Outputs: 

    - shear_tomography_catalog: TomographyCatalog
    
    Parallel: Yes - MPI


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>input_pz</strong>: (bool) Default=False. Whether to use input photo-z posteriors</LI>
            <LI><strong>true_z</strong>: (bool) Default=False. Whether to use true redshifts instead of photo-z</LI>
            <LI><strong>bands</strong>: (str) Default=riz. Bands from the catalog to use for selection</LI>
            <LI><strong>verbose</strong>: (bool) Default=False. Whether to print verbose output</LI>
            <LI><strong>T_cut</strong>: (float) Required. Size cut threshold for object selection</LI>
            <LI><strong>s2n_cut</strong>: (float) Required. Signal-to-noise cut threshold for object selection</LI>
            <LI><strong>chunk_rows</strong>: (int) Default=10000. Number of rows to process in each chunk</LI>
            <LI><strong>source_zbin_edges</strong>: (list) Required. Redshift bin edges for source tomography</LI>
            <LI><strong>random_seed</strong>: (int) Default=42. Random seed for reproducibility</LI>
            <LI><strong>spec_mag_column_format</strong>: (str) Default=photometry/{band}. Format string for spectroscopic magnitude columns</LI>
            <LI><strong>spec_redshift_column</strong>: (str) Default=photometry/redshift. Column name for spectroscopic redshifts</LI>
            <LI><strong>input_m_is_weighted</strong>: (bool) Required. Whether the input m values are already weighted</LI>
            <LI><strong>dec_cut</strong>: (bool) Default=True. Whether to apply a declination cut</LI>
            </UL>



.. autotxclass:: txpipe.source_selector.TXSourceSelectorSimple
    :members:
    :exclude-members: run

    Inputs: 

    - shear_catalog: ShearCatalog
    - spectroscopic_catalog: HDFFile

    Outputs: 

    - shear_tomography_catalog: TomographyCatalog
    
    Parallel: Yes - MPI


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>input_pz</strong>: (bool) Default=False. Whether to use input photo-z posteriors</LI>
            <LI><strong>true_z</strong>: (bool) Default=False. Whether to use true redshifts instead of photo-z</LI>
            <LI><strong>bands</strong>: (str) Default=riz. Bands from the catalog to use for selection</LI>
            <LI><strong>verbose</strong>: (bool) Default=False. Whether to print verbose output</LI>
            <LI><strong>T_cut</strong>: (float) Required. Size cut threshold for object selection</LI>
            <LI><strong>s2n_cut</strong>: (float) Required. Signal-to-noise cut threshold for object selection</LI>
            <LI><strong>chunk_rows</strong>: (int) Default=10000. Number of rows to process in each chunk</LI>
            <LI><strong>source_zbin_edges</strong>: (list) Required. Redshift bin edges for source tomography</LI>
            <LI><strong>random_seed</strong>: (int) Default=42. Random seed for reproducibility</LI>
            <LI><strong>spec_mag_column_format</strong>: (str) Default=photometry/{band}. Format string for spectroscopic magnitude columns</LI>
            <LI><strong>spec_redshift_column</strong>: (str) Default=photometry/redshift. Column name for spectroscopic redshifts</LI>
            </UL>



.. autotxclass:: txpipe.source_selector.TXSourceSelectorHSC
    :members:
    :exclude-members: run

    Inputs: 

    - shear_catalog: ShearCatalog
    - spectroscopic_catalog: HDFFile

    Outputs: 

    - shear_tomography_catalog: TomographyCatalog
    
    Parallel: Yes - MPI


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>input_pz</strong>: (bool) Default=False. Whether to use input photo-z posteriors</LI>
            <LI><strong>true_z</strong>: (bool) Default=False. Whether to use true redshifts instead of photo-z</LI>
            <LI><strong>bands</strong>: (str) Default=riz. Bands from the catalog to use for selection</LI>
            <LI><strong>verbose</strong>: (bool) Default=False. Whether to print verbose output</LI>
            <LI><strong>T_cut</strong>: (float) Required. Size cut threshold for object selection</LI>
            <LI><strong>s2n_cut</strong>: (float) Required. Signal-to-noise cut threshold for object selection</LI>
            <LI><strong>chunk_rows</strong>: (int) Default=10000. Number of rows to process in each chunk</LI>
            <LI><strong>source_zbin_edges</strong>: (list) Required. Redshift bin edges for source tomography</LI>
            <LI><strong>random_seed</strong>: (int) Default=42. Random seed for reproducibility</LI>
            <LI><strong>spec_mag_column_format</strong>: (str) Default=photometry/{band}. Format string for spectroscopic magnitude columns</LI>
            <LI><strong>spec_redshift_column</strong>: (str) Default=photometry/redshift. Column name for spectroscopic redshifts</LI>
            <LI><strong>max_shear_cut</strong>: (float) Default=0.0. Maximum shear value for object selection</LI>
            </UL>



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
            <LI><strong>lens_zbin_edges</strong>: (list) Default=[<class 'float'>]. Edges of lens redshift bins.</LI>
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
            <LI><strong>extra_cols</strong>: (list) Default=['']. Extra columns to include in output.</LI>
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
            <LI><strong>lens_zbin_edges</strong>: (list) Default=[<class 'float'>]. Edges of lens redshift bins.</LI>
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
            <LI><strong>extra_cols</strong>: (list) Default=['']. Extra columns to include in output.</LI>
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
            <LI><strong>lens_zbin_edges</strong>: (list) Default=[<class 'float'>]. Edges of lens redshift bins.</LI>
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
            <LI><strong>extra_cols</strong>: (list) Default=['']. Extra columns to include in output.</LI>
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
            <LI><strong>lens_zbin_edges</strong>: (list) Default=[<class 'float'>]. Edges of lens redshift bins.</LI>
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
            <LI><strong>extra_cols</strong>: (list) Default=['']. Extra columns to include in output.</LI>
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
            <LI><strong>lens_zbin_edges</strong>: (list) Default=[<class 'float'>]. Edges of lens redshift bins.</LI>
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
            <LI><strong>extra_cols</strong>: (list) Default=['']. Extra columns to include in output.</LI>
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
            <LI><strong>verbose</strong>: (bool) Default=False. </LI>
            <LI><strong>chunk_rows</strong>: (int) Default=10000. </LI>
            <LI><strong>lens_zbin_edges</strong>: (list) Default=[<class 'float'>]. </LI>
            <LI><strong>cperp_cut</strong>: (float) Default=0.2. cperp cut for BOSS selection.</LI>
            <LI><strong>r_cpar_cut</strong>: (float) Default=13.5. r_cpar cut for BOSS selection.</LI>
            <LI><strong>r_lo_cut</strong>: (float) Default=16.0. Lower r-band magnitude cut.</LI>
            <LI><strong>r_hi_cut</strong>: (float) Default=19.6. Upper r-band magnitude cut.</LI>
            <LI><strong>i_lo_cut</strong>: (float) Default=17.5. Lower i-band magnitude cut.</LI>
            <LI><strong>i_hi_cut</strong>: (float) Default=19.9. Upper i-band magnitude cut.</LI>
            <LI><strong>r_i_cut</strong>: (float) Default=2.0. r-i color cut.</LI>
            <LI><strong>random_seed</strong>: (int) Default=42. </LI>
            <LI><strong>selection_type</strong>: (str) Default=boss. Type of lens selection (e.g., boss).</LI>
            <LI><strong>maglim_band</strong>: (str) Default=i. Band for magnitude limit.</LI>
            <LI><strong>maglim_limit</strong>: (float) Default=24.1. Magnitude limit value.</LI>
            <LI><strong>extra_cols</strong>: (list) Default=['']. Extra columns to include in output.</LI>
            <LI><strong>apply_mask</strong>: (bool) Default=False. Whether to apply a mask to the selection.</LI>
            <LI><strong>bands</strong>: (str) Default=ugrizy. </LI>
            <LI><strong>spec_mag_column_format</strong>: (str) Default=photometry/{band}. </LI>
            <LI><strong>spec_redshift_column</strong>: (str) Default=photometry/redshift. </LI>
            </UL>


