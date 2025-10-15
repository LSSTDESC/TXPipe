Photo-z
=======

These stages deal with photo-z PDF training and estimation

* :py:class:`~txpipe.photoz_stack.TXPhotozStack` - Naive stacker using QP.

* :py:class:`~txpipe.photoz_stack.TXTruePhotozStack` - Make an ideal true source n(z) using true redshifts

* :py:class:`~txpipe.rail.summarize.PZRailSummarizeBase` - Base class to build the n(z) from some tomographic bins and

* :py:class:`~txpipe.rail.summarize.PZRailSummarize` - Runs a specified RAIL "CatSummarizer" on each tomographic bin and



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
            <LI><strong>weight_col</strong>: (str) Default=shear/00/weight. Column name for weights in the input catalog.</LI>
            <LI><strong>zmax</strong>: (float) Default=0.0. Maximum redshift to use if not specified in input PDFs.</LI>
            <LI><strong>nz</strong>: (int) Default=0. Number of redshift bins to use if not specified in input PDFs.</LI>
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
            <LI><strong>nz</strong>: (int) Default=0. Number of redshift bins for stacking.</LI>
            <LI><strong>weight_col</strong>: (str) Default=weight. Column name for weights in the input catalog.</LI>
            <LI><strong>redshift_group</strong>: (str) Default=. Group name for redshift column in input file.</LI>
            <LI><strong>redshift_col</strong>: (str) Default=redshift_true. Column name for true redshift in input file.</LI>
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
            <LI><strong>bands</strong>: (str) Default=ugrizy. Bands to use for summarization.</LI>
            <LI><strong>summarizer</strong>: (str) Default=NZDirSummarizer. Name of the RAIL summarizer class to use.</LI>
            <LI><strong>module</strong>: (str) Default=rail.estimation.algos.nz_dir. Python module path for the summarizer class.</LI>
            </UL>


