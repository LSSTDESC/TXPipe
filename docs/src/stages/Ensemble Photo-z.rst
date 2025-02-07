Ensemble Photo-z
================

These stages compute ensemble redshift histograms n(z) for bins

* :py:class:`~txpipe.rail.summarize.PZRailSummarize` - Stage PZRailSummarize



.. autotxclass:: txpipe.rail.summarize.PZRailSummarize
    :members:
    :exclude-members: run

    Parallel: Yes - MPI

    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>catalog_group</strong>: (str) Required. </LI>
            <LI><strong>mag_prefix</strong>: (str) Default=photometry/mag_. </LI>
            <LI><strong>tomography_name</strong>: (str) Required. </LI>
            <LI><strong>bands</strong>: (str) Default=ugrizy. </LI>
            </UL>


