Ensemble Photo-z
================

These stages compute ensemble redshift histograms n(z) for bins

* :py:class:`~txpipe.photoz_stack.TXPhotozStack` - Naive stacker using QP.

* :py:class:`~txpipe.photoz_stack.TXTruePhotozStack` - Make an ideal true source n(z) using true redshifts

* :py:class:`~txpipe.photoz_stack.TXMockGaussianPhotozStack` - Makes a mock Gaussian n(z) as a placeholder. Do not use for science!

* :py:class:`~txpipe.rail.summarize.PZRailSummarizeBase` - Base class to build the n(z) from some tomographic bins and

* :py:class:`~txpipe.rail.summarize.PZRailSummarize` - Runs a specified RAIL "CatSummarizer" on each tomographic bin and

* :py:class:`~txpipe.rail.summarize.PZRailPZSummarize` - Runs a specified RAIL *masked* "PZSummarizer" on each tomographic bin and

* :py:class:`~txpipe.rail.summarize.PZAddNonTomographicNZ` - Sum up tomographic n(z)s in a file and append the non-tomographic one.



.. autotxclass:: txpipe.photoz_stack.TXPhotozStack
    :members:
    :exclude-members: run

    Inputs: 

    - photoz_pdfs: QPPDFFile
    - tomography_catalog: TomographyCatalog
    - weights_catalog: HDFFile

    Outputs: 

    - photoz_stack: QPNOfZFile
    
    Parallel: Yes - MPI


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>chunk_rows</strong>: (int) Default=5000. Number of rows to process in each chunk.</LI>
            <LI><strong>tomo_name</strong>: (str) Default=source. Name of the tomographic binning.</LI>
            <LI><strong>weight_col</strong>: (str) Default=shear/ns/weight. Column name for weights in the input catalog.</LI>
            <LI><strong>zmax</strong>: (float) Default=0.0. Maximum redshift to use if not specified in input PDFs.</LI>
            <LI><strong>nz</strong>: (int) Default=0. Number of redshift histogram bins.</LI>
            </UL>



.. autotxclass:: txpipe.photoz_stack.TXTruePhotozStack
    :members:
    :exclude-members: run

    Inputs: 

    - tomography_catalog: TomographyCatalog
    - catalog: HDFFile
    - weights_catalog: HDFFile

    Outputs: 

    - photoz_stack: QPNOfZFile
    
    Parallel: Yes - MPI


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>chunk_rows</strong>: (int) Default=5000. Number of rows to read at once.</LI>
            <LI><strong>zmax</strong>: (float) Default=0.0. Maximum redshift for stacking.</LI>
            <LI><strong>nz</strong>: (int) Default=0. Number of redshift histogram bins.</LI>
            <LI><strong>weight_col</strong>: (str) Default=weight. Column name for weights in the input catalog.</LI>
            <LI><strong>redshift_group</strong>: (str) Default=. Group name for redshift column in input file.</LI>
            <LI><strong>redshift_col</strong>: (str) Default=redshift_true. Column name for true redshift in input file.</LI>
            </UL>



.. autotxclass:: txpipe.photoz_stack.TXMockGaussianPhotozStack
    :members:
    :exclude-members: run

    Inputs: None

    Outputs: 

    - photoz_stack: QPNOfZFile
    
    Parallel: No - Serial


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>means</strong>: (list) Required. Mean value of mock bins.</LI>
            <LI><strong>sigmas</strong>: (list) Required. Std dev values of mock bins.</LI>
            <LI><strong>zmax</strong>: (float) Default=2.0. Maximum redshift to use if not specified in input PDFs.</LI>
            <LI><strong>nz</strong>: (int) Default=201. Number of redshift histogram sample points.</LI>
            </UL>



.. autotxclass:: txpipe.rail.summarize.PZRailSummarizeBase
    :members:
    :exclude-members: run

    Inputs: None

    Outputs: 

    - photoz_stack: QPNOfZFile
    - photoz_realizations: QPMultiFile
    
    Parallel: Yes - MPI


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            </UL>



.. autotxclass:: txpipe.rail.summarize.PZRailSummarize
    :members:
    :exclude-members: run

    Inputs: 

    - binned_catalog: BinnedCatalog
    - model: PickleFile

    Outputs: 

    - photoz_stack: QPNOfZFile
    - photoz_realizations: QPMultiFile
    
    Parallel: Yes - MPI


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>catalog_group</strong>: (str) Default=. Group name in the catalog file for tomographic bins.</LI>
            <LI><strong>mag_prefix</strong>: (str) Default=photometry/mag_. Prefix for magnitude columns in the catalog.</LI>
            <LI><strong>tomography_name</strong>: (str) Default=. Name of the tomography scheme.</LI>
            <LI><strong>bands</strong>: (list) Default=[&#x27;u&#x27;, &#x27;g&#x27;, &#x27;r&#x27;, &#x27;i&#x27;, &#x27;z&#x27;, &#x27;y&#x27;]. Bands to use for summarization.</LI>
            <LI><strong>summarizer</strong>: (str) Default=NZDirSummarizer. Name of the RAIL summarizer class to use.</LI>
            <LI><strong>module</strong>: (str) Default=rail.estimation.algos.nz_dir. Python module path for the summarizer class.</LI>
            </UL>



.. autotxclass:: txpipe.rail.summarize.PZRailPZSummarize
    :members:
    :exclude-members: run

    Inputs: 

    - photoz_pdfs: HDFFile
    - tomography_catalog: TomographyCatalog
    - model: PickleFile

    Outputs: 

    - photoz_stack: QPNOfZFile
    - photoz_realizations: QPMultiFile
    
    Parallel: Yes - MPI


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>catalog_group</strong>: (str) Default=. Group name in the catalog file for tomographic bins.</LI>
            <LI><strong>mag_prefix</strong>: (str) Default=photometry/mag_. Prefix for magnitude columns in the catalog.</LI>
            <LI><strong>tomography_name</strong>: (str) Default=. Name of the tomography scheme.</LI>
            <LI><strong>bands</strong>: (list) Default=[&#x27;u&#x27;, &#x27;g&#x27;, &#x27;r&#x27;, &#x27;i&#x27;, &#x27;z&#x27;, &#x27;y&#x27;]. Bands to use for summarization.</LI>
            <LI><strong>summarizer</strong>: (str) Default=PointEstHistMaskedSummarizer. Name of the RAIL summarizer class to use.</LI>
            <LI><strong>module</strong>: (str) Default=rail.estimation.algos.point_est_hist. Python module path for the summarizer class.</LI>
            </UL>



.. autotxclass:: txpipe.rail.summarize.PZAddNonTomographicNZ
    :members:
    :exclude-members: run

    Inputs: 

    - input_photoz_stack: QPNOfZFile

    Outputs: 

    - output_photoz_stack: QPNOfZFile
    
    Parallel: Yes - MPI


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            </UL>


