Ensemble Photo-z
================

These stages compute ensemble redshift histograms n(z) for bins

* :py:class:`~txpipe.rail.summarize.PZRailPZSummarize` - Runs a specified RAIL *masked* "PZSummarizer" on each tomographic bin and



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
            <LI><strong>bands</strong>: (str) Default=ugrizy. Bands to use for summarization.</LI>
            <LI><strong>summarizer</strong>: (str) Default=PointEstHistMaskedSummarizer. Name of the RAIL summarizer class to use.</LI>
            <LI><strong>module</strong>: (str) Default=rail.estimation.algos.point_est_hist. Python module path for the summarizer class.</LI>
            </UL>


