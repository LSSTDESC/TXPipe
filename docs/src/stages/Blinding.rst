Blinding
========

These stages deal with blinding measurements. TXSmokescreenBlinding blinds
data with the DESC Smokescreen library, and TXNullBlinding copies data
without blinding it. See :doc:`/blinding` for how to use them.

* :py:class:`~txpipe.blinding.blinding.TXNullBlinding` - Pretend to blind but actually do nothing.

* :py:class:`~txpipe.blinding.smokescreen_blinding.TXSmokescreenBlinding` - Blinds two-point measurements using Smokescreen



.. autotxclass:: txpipe.blinding.blinding.TXNullBlinding
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



.. autotxclass:: txpipe.blinding.smokescreen_blinding.TXSmokescreenBlinding
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
            <LI><strong>shift_distribution</strong>: (str) Default=flat. Distribution of random shifts: &#x27;flat&#x27; (U(a, b)) or &#x27;gaussian&#x27; (N(a, b)).</LI>
            <LI><strong>shift_type</strong>: (str) Default=add. How the shift is applied: &#x27;add&#x27; or &#x27;mult&#x27;.</LI>
            <LI><strong>systematics</strong>: (dict) Default={}. Values for the likelihood&#x27;s systematics parameters. Empty uses their defaults.</LI>
            <LI><strong>b0</strong>: (float) Default=0.0. TXPipe theory model only, if systematics is empty: lens bias b0/growth. Zero means unset.</LI>
            <LI><strong>seed</strong>: (str) Default=Boitata was here. Seed that uniquely specifies the shifts.</LI>
            <LI><strong>encrypt_unblinded</strong>: (bool) Default=False. Whether to save an encrypted copy of the unblinded data, and its key.</LI>
            <LI><strong>encryption_dir</strong>: (str) Default=. Directory for the encrypted data and key. Empty means next to the unblinded data.</LI>
            <LI><strong>delete_unblinded</strong>: (bool) Default=False. Whether to delete unblinded data after blinding.</LI>
            </UL>


