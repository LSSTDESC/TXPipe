Blinding
========

These stages deal with blinding measurements

* :py:class:`~txpipe.blinding.TXBlinding` - Blinds real-space measurements following Muir et al

* :py:class:`~txpipe.blinding.TXNullBlinding` - Pretend to blind but actually do nothing.



.. autotxclass:: txpipe.blinding.TXBlinding
    :members:
    :exclude-members: run

    Inputs: 

    - twopoint_data_real_raw: SACCFile

    Outputs: 

    - twopoint_data_real: SACCFile
    
    Parallel: No - Serial


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>seed</strong>: (int) Default=1972. </LI>
            <LI><strong>Omega_b</strong>: (list) Default=[0.0485, 0.001]. </LI>
            <LI><strong>Omega_c</strong>: (list) Default=[0.2545, 0.01]. </LI>
            <LI><strong>w0</strong>: (list) Default=[-1.0, 0.1]. </LI>
            <LI><strong>h</strong>: (list) Default=[0.682, 0.02]. </LI>
            <LI><strong>sigma8</strong>: (list) Default=[0.801, 0.01]. </LI>
            <LI><strong>n_s</strong>: (list) Default=[0.971, 0.03]. </LI>
            <LI><strong>b0</strong>: (float) Default=0.95. </LI>
            <LI><strong>delete_unblinded</strong>: (bool) Default=False. </LI>
            </UL>



.. autotxclass:: txpipe.blinding.TXNullBlinding
    :members:
    :exclude-members: run

    Inputs: 

    - twopoint_data_real_raw: SACCFile

    Outputs: 

    - twopoint_data_real: SACCFile
    
    Parallel: No - Serial


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            </UL>


