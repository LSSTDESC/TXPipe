# DES Y3 3x2pt blinding with Smokescreen

This example blinds the public DES Y3 MagLim 3x2pt real-space data vector with
the `TXSmokescreenBlinding` stage. It uses a firecrown likelihood for DES Y3 that
lives in this directory. The DES Y3 data has been unblinded and published
already, so this is only a demonstration of the method.

Run all commands from the TXPipe root directory.

## 1. Prepare the data

```bash
python examples/desy3_smokescreen/prepare_data.py
```

This downloads the DES Y3 two-point file
`2pt_NG_final_2ptunblind_02_26_21_wnz_maglim_covupdate.fits` (28 MB, from the
[CosmoSIS standard library](https://github.com/joezuntz/cosmosis-standard-library/tree/main/likelihood/des-y3))
to `data/desy3/inputs/`. It then converts it to the sacc file
`data/desy3/inputs/des_y3_3x2pt_maglim.sacc`:

- the n(z) become tracers `source_0..3` and `lens_0..3`, following TXPipe naming;
- it keeps the first four MagLim lens bins and the w(θ) auto-correlations;
- it applies the DES Y3 MagLim 3x2pt scale cuts.

The result has 462 data points: 166 ξ+, 61 ξ−, 192 γt and 43 w(θ). It also
includes the DES Y3 covariance.

## 2. Blind

```bash
ceci examples/desy3_smokescreen/pipeline.yml
```

This writes the blinded data to
`data/desy3/outputs_smokescreen/twopoint_data_real.sacc`. Next to the input it
also writes an encrypted copy of the unblinded data (`.encrpt`) and its key
(`.key`).

The settings are in `config.yml`:

- **Shifts:** the data is blinded in σ8 and w0. Smokescreen draws σ8 from
  U(0.70, 0.90) and w0 from U(−1.4, −0.6) using the seed. It then shifts the
  data vector by the theory difference between that cosmology and the reference
  one (the DES Y3 fiducial cosmology, Ω_m = 0.3, σ8 = 0.8, w0 = −1).
- **Nuisance parameters:** these are fixed at the centres of the DES Y3 MagLim
  priors. Set every one of them: some firecrown defaults are not suitable here,
  such as a multiplicative shear bias of 1.
- **Unblinded data:** the example keeps the unblinded file so it can be re-run.
  In a real analysis, set `delete_unblinded: true` and store the key away from
  the data.

## 3. Plot the shift (optional)

```bash
python examples/desy3_smokescreen/plot_blinding.py
```

This saves `data/desy3/outputs_smokescreen/blinding_shift.png`. The plot shows
the raw and blinded data for one bin pair of each statistic, and the shift of
every point in units of its error bar. It needs the unblinded data, so it is
for checking the example only.

## The likelihood

`des_y3_3x2pt_likelihood.py` follows the DES Y3 3x2pt model as closely as
firecrown 1.15 allows:

- **Weak lensing:** NLA intrinsic alignments with redshift dependence (A1,
  α1 and z_piv = 0.62), multiplicative shear bias and photo-z shifts.
- **Number counts:** linear galaxy bias, photo-z shift and stretch, and
  constant magnification bias. The bias is s = 2α/5 from the DES Y3 α
  values.
- **Power spectrum:** the non-linear matter power spectrum from CCL, with
  the DES Y3 covariance.

It differs from the DES Y3 analysis in a few ways:

- It uses NLA instead of the fiducial TATT intrinsic-alignment model. With
  TATT, the shear C_ℓ of the low and high redshift bins can change sign at
  ℓ ≳ 10⁴. CCL's Hankel transform then fails for some cosmologies, so random
  blinding draws would sometimes crash.
- It leaves out the shear-ratio likelihood and the γt point-mass
  marginalisation.

You can use any other firecrown likelihood that models these data instead,
with the `likelihood_path` option.
