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

* :py:class:`~txpipe.lens_selector.TXRandomForestLensSelector` - Stage TXRandomForestLensSelector



.. autotxclass:: txpipe.source_selector.TXSourceSelectorBase
    :members:
    :exclude-members: run

    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>input_pz</strong>: (bool) Default=False. </LI>
            <LI><strong>true_z</strong>: (bool) Default=False. </LI>
            <LI><strong>bands</strong>: (str) Default=riz. </LI>
            <LI><strong>verbose</strong>: (bool) Default=False. </LI>
            <LI><strong>T_cut</strong>: (float) Required. </LI>
            <LI><strong>s2n_cut</strong>: (float) Required. </LI>
            <LI><strong>chunk_rows</strong>: (int) Default=10000. </LI>
            <LI><strong>source_zbin_edges</strong>: (list) Default=[<class 'float'>]. </LI>
            <LI><strong>random_seed</strong>: (int) Default=42. </LI>
            </UL>



.. autotxclass:: txpipe.source_selector.TXSourceSelectorMetacal
    :members:
    :exclude-members: run

    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>input_pz</strong>: (bool) Default=False. </LI>
            <LI><strong>true_z</strong>: (bool) Default=False. </LI>
            <LI><strong>bands</strong>: (str) Default=riz. </LI>
            <LI><strong>verbose</strong>: (bool) Default=False. </LI>
            <LI><strong>T_cut</strong>: (float) Required. </LI>
            <LI><strong>s2n_cut</strong>: (float) Required. </LI>
            <LI><strong>chunk_rows</strong>: (int) Default=10000. </LI>
            <LI><strong>source_zbin_edges</strong>: (list) Default=[<class 'float'>]. </LI>
            <LI><strong>random_seed</strong>: (int) Default=42. </LI>
            <LI><strong>delta_gamma</strong>: (float) Required. </LI>
            <LI><strong>use_diagonal_response</strong>: (bool) Default=False. </LI>
            </UL>



.. autotxclass:: txpipe.source_selector.TXSourceSelectorMetadetect
    :members:
    :exclude-members: run

    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>input_pz</strong>: (bool) Default=False. </LI>
            <LI><strong>true_z</strong>: (bool) Default=False. </LI>
            <LI><strong>bands</strong>: (str) Default=riz. </LI>
            <LI><strong>verbose</strong>: (bool) Default=False. </LI>
            <LI><strong>T_cut</strong>: (float) Required. </LI>
            <LI><strong>s2n_cut</strong>: (float) Required. </LI>
            <LI><strong>chunk_rows</strong>: (int) Default=10000. </LI>
            <LI><strong>source_zbin_edges</strong>: (list) Default=[<class 'float'>]. </LI>
            <LI><strong>random_seed</strong>: (int) Default=42. </LI>
            <LI><strong>delta_gamma</strong>: (float) Required. </LI>
            </UL>



.. autotxclass:: txpipe.source_selector.TXSourceSelectorLensfit
    :members:
    :exclude-members: run

    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>input_pz</strong>: (bool) Default=False. </LI>
            <LI><strong>true_z</strong>: (bool) Default=False. </LI>
            <LI><strong>bands</strong>: (str) Default=riz. </LI>
            <LI><strong>verbose</strong>: (bool) Default=False. </LI>
            <LI><strong>T_cut</strong>: (float) Required. </LI>
            <LI><strong>s2n_cut</strong>: (float) Required. </LI>
            <LI><strong>chunk_rows</strong>: (int) Default=10000. </LI>
            <LI><strong>source_zbin_edges</strong>: (list) Default=[<class 'float'>]. </LI>
            <LI><strong>random_seed</strong>: (int) Default=42. </LI>
            <LI><strong>input_m_is_weighted</strong>: (bool) Required. </LI>
            <LI><strong>dec_cut</strong>: (bool) Default=True. </LI>
            </UL>



.. autotxclass:: txpipe.source_selector.TXSourceSelectorSimple
    :members:
    :exclude-members: run

    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>input_pz</strong>: (bool) Default=False. </LI>
            <LI><strong>true_z</strong>: (bool) Default=False. </LI>
            <LI><strong>bands</strong>: (str) Default=riz. </LI>
            <LI><strong>verbose</strong>: (bool) Default=False. </LI>
            <LI><strong>T_cut</strong>: (float) Required. </LI>
            <LI><strong>s2n_cut</strong>: (float) Required. </LI>
            <LI><strong>chunk_rows</strong>: (int) Default=10000. </LI>
            <LI><strong>source_zbin_edges</strong>: (list) Default=[<class 'float'>]. </LI>
            <LI><strong>random_seed</strong>: (int) Default=42. </LI>
            </UL>



.. autotxclass:: txpipe.source_selector.TXSourceSelectorHSC
    :members:
    :exclude-members: run

    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>input_pz</strong>: (bool) Default=False. </LI>
            <LI><strong>true_z</strong>: (bool) Default=False. </LI>
            <LI><strong>bands</strong>: (str) Default=riz. </LI>
            <LI><strong>verbose</strong>: (bool) Default=False. </LI>
            <LI><strong>T_cut</strong>: (float) Required. </LI>
            <LI><strong>s2n_cut</strong>: (float) Required. </LI>
            <LI><strong>chunk_rows</strong>: (int) Default=10000. </LI>
            <LI><strong>source_zbin_edges</strong>: (list) Default=[<class 'float'>]. </LI>
            <LI><strong>random_seed</strong>: (int) Default=42. </LI>
            <LI><strong>max_shear_cut</strong>: (float) Default=0.0. </LI>
            </UL>



.. autotxclass:: txpipe.lens_selector.TXBaseLensSelector
    :members:
    :exclude-members: run

    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>verbose</strong>: (bool) Default=False. </LI>
            <LI><strong>chunk_rows</strong>: (int) Default=10000. </LI>
            <LI><strong>lens_zbin_edges</strong>: (list) Default=[<class 'float'>]. </LI>
            <LI><strong>cperp_cut</strong>: (float) Default=0.2. </LI>
            <LI><strong>r_cpar_cut</strong>: (float) Default=13.5. </LI>
            <LI><strong>r_lo_cut</strong>: (float) Default=16.0. </LI>
            <LI><strong>r_hi_cut</strong>: (float) Default=19.6. </LI>
            <LI><strong>i_lo_cut</strong>: (float) Default=17.5. </LI>
            <LI><strong>i_hi_cut</strong>: (float) Default=19.9. </LI>
            <LI><strong>r_i_cut</strong>: (float) Default=2.0. </LI>
            <LI><strong>random_seed</strong>: (int) Default=42. </LI>
            <LI><strong>selection_type</strong>: (str) Default=boss. </LI>
            <LI><strong>maglim_band</strong>: (str) Default=i. </LI>
            <LI><strong>maglim_limit</strong>: (float) Default=24.1. </LI>
            <LI><strong>extra_cols</strong>: (list) Default=['']. </LI>
            </UL>



.. autotxclass:: txpipe.lens_selector.TXTruthLensSelector
    :members:
    :exclude-members: run

    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>verbose</strong>: (bool) Default=False. </LI>
            <LI><strong>chunk_rows</strong>: (int) Default=10000. </LI>
            <LI><strong>lens_zbin_edges</strong>: (list) Default=[<class 'float'>]. </LI>
            <LI><strong>cperp_cut</strong>: (float) Default=0.2. </LI>
            <LI><strong>r_cpar_cut</strong>: (float) Default=13.5. </LI>
            <LI><strong>r_lo_cut</strong>: (float) Default=16.0. </LI>
            <LI><strong>r_hi_cut</strong>: (float) Default=19.6. </LI>
            <LI><strong>i_lo_cut</strong>: (float) Default=17.5. </LI>
            <LI><strong>i_hi_cut</strong>: (float) Default=19.9. </LI>
            <LI><strong>r_i_cut</strong>: (float) Default=2.0. </LI>
            <LI><strong>random_seed</strong>: (int) Default=42. </LI>
            <LI><strong>selection_type</strong>: (str) Default=boss. </LI>
            <LI><strong>maglim_band</strong>: (str) Default=i. </LI>
            <LI><strong>maglim_limit</strong>: (float) Default=24.1. </LI>
            <LI><strong>extra_cols</strong>: (list) Default=['']. </LI>
            </UL>



.. autotxclass:: txpipe.lens_selector.TXMeanLensSelector
    :members:
    :exclude-members: run

    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>verbose</strong>: (bool) Default=False. </LI>
            <LI><strong>chunk_rows</strong>: (int) Default=10000. </LI>
            <LI><strong>lens_zbin_edges</strong>: (list) Default=[<class 'float'>]. </LI>
            <LI><strong>cperp_cut</strong>: (float) Default=0.2. </LI>
            <LI><strong>r_cpar_cut</strong>: (float) Default=13.5. </LI>
            <LI><strong>r_lo_cut</strong>: (float) Default=16.0. </LI>
            <LI><strong>r_hi_cut</strong>: (float) Default=19.6. </LI>
            <LI><strong>i_lo_cut</strong>: (float) Default=17.5. </LI>
            <LI><strong>i_hi_cut</strong>: (float) Default=19.9. </LI>
            <LI><strong>r_i_cut</strong>: (float) Default=2.0. </LI>
            <LI><strong>random_seed</strong>: (int) Default=42. </LI>
            <LI><strong>selection_type</strong>: (str) Default=boss. </LI>
            <LI><strong>maglim_band</strong>: (str) Default=i. </LI>
            <LI><strong>maglim_limit</strong>: (float) Default=24.1. </LI>
            <LI><strong>extra_cols</strong>: (list) Default=['']. </LI>
            </UL>



.. autotxclass:: txpipe.lens_selector.TXModeLensSelector
    :members:
    :exclude-members: run

    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>verbose</strong>: (bool) Default=False. </LI>
            <LI><strong>chunk_rows</strong>: (int) Default=10000. </LI>
            <LI><strong>lens_zbin_edges</strong>: (list) Default=[<class 'float'>]. </LI>
            <LI><strong>cperp_cut</strong>: (float) Default=0.2. </LI>
            <LI><strong>r_cpar_cut</strong>: (float) Default=13.5. </LI>
            <LI><strong>r_lo_cut</strong>: (float) Default=16.0. </LI>
            <LI><strong>r_hi_cut</strong>: (float) Default=19.6. </LI>
            <LI><strong>i_lo_cut</strong>: (float) Default=17.5. </LI>
            <LI><strong>i_hi_cut</strong>: (float) Default=19.9. </LI>
            <LI><strong>r_i_cut</strong>: (float) Default=2.0. </LI>
            <LI><strong>random_seed</strong>: (int) Default=42. </LI>
            <LI><strong>selection_type</strong>: (str) Default=boss. </LI>
            <LI><strong>maglim_band</strong>: (str) Default=i. </LI>
            <LI><strong>maglim_limit</strong>: (float) Default=24.1. </LI>
            <LI><strong>extra_cols</strong>: (list) Default=['']. </LI>
            </UL>



.. autotxclass:: txpipe.lens_selector.TXRandomForestLensSelector
    :members:
    :exclude-members: run

    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>verbose</strong>: (bool) Default=False. </LI>
            <LI><strong>chunk_rows</strong>: (int) Default=10000. </LI>
            <LI><strong>lens_zbin_edges</strong>: (list) Default=[<class 'float'>]. </LI>
            <LI><strong>cperp_cut</strong>: (float) Default=0.2. </LI>
            <LI><strong>r_cpar_cut</strong>: (float) Default=13.5. </LI>
            <LI><strong>r_lo_cut</strong>: (float) Default=16.0. </LI>
            <LI><strong>r_hi_cut</strong>: (float) Default=19.6. </LI>
            <LI><strong>i_lo_cut</strong>: (float) Default=17.5. </LI>
            <LI><strong>i_hi_cut</strong>: (float) Default=19.9. </LI>
            <LI><strong>r_i_cut</strong>: (float) Default=2.0. </LI>
            <LI><strong>random_seed</strong>: (int) Default=42. </LI>
            <LI><strong>selection_type</strong>: (str) Default=boss. </LI>
            <LI><strong>maglim_band</strong>: (str) Default=i. </LI>
            <LI><strong>maglim_limit</strong>: (float) Default=24.1. </LI>
            <LI><strong>extra_cols</strong>: (list) Default=['']. </LI>
            <LI><strong>bands</strong>: (str) Default=ugrizy. </LI>
            </UL>


