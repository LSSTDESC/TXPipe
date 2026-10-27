# Firecrown 1.7.5 → 1.15.2 theory validation

Checks that `theory_3x2pt` gives the same predictions after porting
`txpipe/utils/theory_model.py` to firecrown 1.15.

1. In an environment with firecrown 1.7.5 and a checkout of TXPipe before the port:

       python compute_theory.py <old_txpipe_checkout> out_old

2. In an environment with firecrown 1.15.2 and this branch:

       python compute_theory.py <this_txpipe_checkout> out_new

3. Compare:

       python plot_comparison.py out_old out_new "firecrown 1.7.5" "firecrown 1.15.2" plots

The test uses a synthetic sacc file (4 source and 3 lens bins, Gaussian n(z)) in real and
Fourier space, with every bias input form accepted by `make_bias_parameters`, plus a check that
unsupported B-mode data types are set to zero. The results in `plots/` are identical
(max relative difference exactly 0).
