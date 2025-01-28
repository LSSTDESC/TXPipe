Photo-z
=======

These stages deal with photo-z PDF training and estimation

* :py:class:`~txpipe.photoz_stack.TXPhotozStack` - Naive stacker using QP.

* :py:class:`~txpipe.photoz_stack.TXTruePhotozStack` - Make an ideal true source n(z) using true redshifts



.. autotxclass:: txpipe.photoz_stack.TXPhotozStack
    :members:
    :exclude-members: run

    Parallel: Yes - MPI

    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>chunk_rows</strong>: (int) Default=5000. </LI>
            <LI><strong>tomo_name</strong>: (str) Default=source. </LI>
            <LI><strong>weight_col</strong>: (str) Default=shear/00/weight. </LI>
            <LI><strong>zmax</strong>: (float) Default=0.0. </LI>
            <LI><strong>nz</strong>: (int) Default=0. </LI>
            </UL>



.. autotxclass:: txpipe.photoz_stack.TXTruePhotozStack
    :members:
    :exclude-members: run

    Parallel: Yes - MPI

    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>chunk_rows</strong>: (int) Default=5000. </LI>
            <LI><strong>zmax</strong>: (float) Required. </LI>
            <LI><strong>nz</strong>: (int) Required. </LI>
            <LI><strong>weight_col</strong>: (str) Default=weight. </LI>
            <LI><strong>redshift_group</strong>: (str) Required. </LI>
            <LI><strong>redshift_col</strong>: (str) Default=redshift_true. </LI>
            </UL>


