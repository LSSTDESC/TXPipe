Blinding Guide
==============

Blinding hides the true cosmological result of an analysis until it is finalised. TXPipe blinds two-point measurements at the data-vector level, following `Muir et al. 2020 <https://arxiv.org/abs/1911.05929>`_: the measured data vector :math:`d` is replaced by

.. math::

    d_{\rm blind} = d + t(\theta_{\rm shifted}) - t(\theta_{\rm ref}),

where :math:`t` is a theory prediction, :math:`\theta_{\rm ref}` is a reference cosmology, and :math:`\theta_{\rm shifted}` is a cosmology drawn at random from a seed. Analysing the blinded data gives a result shifted by an unknown amount.

TXPipe has two blinding stages:

- ``TXBlinding``: the original TXPipe method. It blinds real-space data with the TXPipe 3x2pt theory model, drawing Gaussian shifts in a fixed set of parameters.
- ``TXSmokescreenBlinding``: uses the DESC `Smokescreen <https://github.com/LSSTDESC/Smokescreen>`_ library. It works on real or harmonic-space data, with the TXPipe theory model or any `firecrown <https://github.com/LSSTDESC/firecrown>`_ likelihood, and supports all the Smokescreen options.

The rest of this page describes ``TXSmokescreenBlinding``. The configuration options of both stages are listed in :doc:`stages/Blinding`.


Adding the stage to a pipeline
------------------------------

The stage has a generic input, ``twopoint_data_raw``, and output, ``twopoint_data_blinded``. Use ``aliases`` to connect them to the files you want to blind, and ``classname`` to run it more than once. For example, to blind both the real and harmonic-space measurements:

.. code-block:: yaml

    stages:
      - name: TXTwoPoint
      - name: TXSmokescreenBlindingReal
        classname: TXSmokescreenBlinding
        aliases:
            twopoint_data_raw: twopoint_data_real_raw
            twopoint_data_blinded: twopoint_data_real
      - name: TXTwoPointFourier
        aliases:
            twopoint_data_fourier: twopoint_data_fourier_raw
      - name: TXSmokescreenBlindingFourier
        classname: TXSmokescreenBlinding
        aliases:
            twopoint_data_raw: twopoint_data_fourier_raw
            twopoint_data_blinded: twopoint_data_fourier

The full example is ``examples/metadetect/pipeline_smokescreen.yml``.


Configuration
-------------

A configuration section for one of the stages above looks like this:

.. code-block:: yaml

    TXSmokescreenBlindingReal:
        reference_cosmology:
            Omega_c: 0.2545
            Omega_b: 0.0485
            h: 0.682
            sigma8: 0.801
            n_s: 0.971
        shifts_dict:
            Omega_c: [0.22, 0.29]
            sigma8: [0.75, 0.85]
            w0: [-1.2, -0.8]
        shift_distribution: flat
        shift_type: add
        b0: 0.95
        seed: "Boitata was here"
        delete_unblinded: true

The main options are:

- ``reference_cosmology``: the reference cosmology, using CCL parameter names. Parameters you do not set take the CCL vanilla ΛCDM values.
- ``shifts_dict``: the parameters to shift. A single value fixes the shifted parameter to it; a pair ``[a, b]`` draws it at random, from U(a, b) if ``shift_distribution`` is ``flat`` or from N(a, b) if it is ``gaussian``. The parameters must be CCL ones, such as ``sigma8``, ``Omega_c`` or ``w0``; derived parameters like S8 are not supported.
- ``shift_type``: ``add`` (the formula above) or ``mult``, which multiplies the data by :math:`t(\theta_{\rm shifted}) / t(\theta_{\rm ref})`.
- ``seed``: a string that fixes the random shifts. Choose it without looking at the result, and keep it private.
- ``likelihood_path``: a firecrown likelihood file. If empty, the TXPipe 3x2pt theory model is used (linear galaxy bias, no other systematics).
- ``systematics``: values for every systematics parameter of the likelihood. If it is empty, the likelihood defaults are used, or, for the TXPipe model, the lens bias ``b0`` / D(z) if ``b0`` is set.

Data points that the likelihood does not model, such as B-modes, are copied to the output unchanged, and the stage prints a warning listing them. Everything else in the file (tracers, covariance, tags) is kept, and the metadata records that the data is blinded and with which likelihood.


Using your own likelihood
-------------------------

Any firecrown likelihood file with a ``build_likelihood`` function can be used. TXPipe calls it with a ``NamedParameters`` object whose ``sacc_data`` entry is the ``sacc.Sacc`` object being blinded. The likelihood must:

- use the TXPipe tracer names (``source_0``, ``lens_0``, ...), or whatever names are in your file;
- *select* the points it models, rather than removing points from the sacc data, for example for scale cuts. The stage needs the likelihood's sacc indices to refer to the input file, and stops with an error if they do not;
- be given a value for every one of its systematics parameters in ``systematics``. Smokescreen checks that none are missing. Do not rely on the firecrown defaults: some are not physical, for example a multiplicative shear bias default of 1.

With the TXPipe theory model, each lens bin ``lens_i`` needs ``lens_i_bias``, ``lens_i_alphag``, ``lens_i_alphaz`` and ``lens_i_z_piv``; setting ``alphag`` to zero gives a constant bias in each bin. The simpler option is to set ``b0`` instead.

``examples/desy3_smokescreen`` has a complete example: a firecrown likelihood following the DES Y3 MagLim 3x2pt model, a script that converts the public DES Y3 data to sacc, and a pipeline that blinds it in σ8 and w0. Its ``README.md`` explains each step.


Keeping and recovering the unblinded data
-----------------------------------------

Two options control what happens to the unblinded file:

- ``encrypt_unblinded``: save an encrypted copy of the unblinded file (``<name>.encrpt``) and its key (``<name>.key``) in ``encryption_dir``, or next to the unblinded file if that is empty. Move the key somewhere safe, away from the data.
- ``delete_unblinded``: replace the unblinded file with an empty one after blinding, so that the analysis cannot use it by mistake. Running the stage again on the empty file gives an error.

When the analysis is ready to be unblinded, recover the data with the Smokescreen command-line tool, which is installed with Smokescreen:

.. code-block:: bash

    smokescreen decrypt --path_to_sacc twopoint_data_real_raw.encrpt --path_to_key twopoint_data_real_raw.key

or from Python:

.. code-block:: python

    from smokescreen.encryption import decrypt_file

    decrypt_file("twopoint_data_real_raw.encrpt", "twopoint_data_real_raw.key", save_file=True)

Both write the decrypted file next to the encrypted one, named without an extension (here ``twopoint_data_real_raw``); rename it to ``twopoint_data_real_raw.sacc`` to use it. The Python function also returns the file contents as bytes, so with ``save_file=False`` you can decrypt without writing anything to disk.
