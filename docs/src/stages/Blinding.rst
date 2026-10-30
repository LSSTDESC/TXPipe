Blinding
========

These stages deal with blinding measurements. TXBlinding is the original
TXPipe method, and TXSmokescreenBlinding uses the DESC Smokescreen library
with any firecrown likelihood. See :doc:`/blinding` for how to use them.

* :py:class:`~txpipe.blinding.TXBlinding` - Blinds real-space measurements following Muir et al

* :py:class:`~txpipe.blinding.TXNullBlinding` - Pretend to blind but actually do nothing.

* :py:class:`~txpipe.smokescreen_blinding.TXSmokescreenBlinding` - Blinds two-point measurements using Smokescreen



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
            <LI><strong>seed</strong>: (int) Default=1972. Seed uniquely specifies the shift in parameters.</LI>
            <LI><strong>Omega_b</strong>: (list) Default=[0.0485, 0.001]. Fiducial value and shift sigma for Omega_b.</LI>
            <LI><strong>Omega_c</strong>: (list) Default=[0.2545, 0.01]. Fiducial value and shift sigma for Omega_c.</LI>
            <LI><strong>w0</strong>: (list) Default=[-1.0, 0.1]. Fiducial value and shift sigma for w0.</LI>
            <LI><strong>h</strong>: (list) Default=[0.682, 0.02]. Fiducial value and shift sigma for h.</LI>
            <LI><strong>sigma8</strong>: (list) Default=[0.801, 0.01]. Fiducial value and shift sigma for sigma8.</LI>
            <LI><strong>n_s</strong>: (list) Default=[0.971, 0.03]. Fiducial value and shift sigma for n_s.</LI>
            <LI><strong>b0</strong>: (float) Default=0.95. Bias parameter (b0/growth).</LI>
            <LI><strong>delete_unblinded</strong>: (bool) Default=False. Whether to delete unblinded data after blinding.</LI>
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



.. autotxclass:: txpipe.smokescreen_blinding.TXSmokescreenBlinding
    :members:
    :exclude-members: run

    Inputs: 

    - twopoint_data_raw: SACCFile

    Outputs: 

    - twopoint_data_blinded: SACCFile
    
    Parallel: No - Serial


    .. collapse:: Configuration

        .. raw:: html

            <UL>
            <LI><strong>likelihood_path</strong>: (str) Default=. Path to a firecrown likelihood file. If empty, the TXPipe 3x2pt theory model is used.</LI>
            <LI><strong>reference_cosmology</strong>: (dict) Default={}. Reference cosmology parameters (CCL names). Unset parameters take CCL vanilla LCDM values.</LI>
            <LI><strong>shifts_dict</strong>: (dict) Default={}. Parameter shifts: {param: value} for a fixed value, or {param: [a, b]} to draw it at random.</LI>
            <LI><strong>shift_distribution</strong>: (str) Default=flat. Distribution of random shifts: 'flat' (U(a, b)) or 'gaussian' (N(a, b)).</LI>
            <LI><strong>shift_type</strong>: (str) Default=add. How the shift is applied: 'add' or 'mult'.</LI>
            <LI><strong>systematics</strong>: (dict) Default={}. Values for the likelihood's systematics parameters. Empty uses their defaults.</LI>
            <LI><strong>b0</strong>: (float) Default=0.0. TXPipe theory model only, if systematics is empty: lens bias b0/growth. Zero means unset.</LI>
            <LI><strong>seed</strong>: (str) Default=2112. Seed that uniquely specifies the shifts.</LI>
            <LI><strong>encrypt_unblinded</strong>: (bool) Default=False. Whether to save an encrypted copy of the unblinded data, and its key.</LI>
            <LI><strong>encryption_dir</strong>: (str) Default=. Directory for the encrypted data and key. Empty means next to the unblinded data.</LI>
            <LI><strong>delete_unblinded</strong>: (bool) Default=False. Whether to delete unblinded data after blinding.</LI>
            </UL>


