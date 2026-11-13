Metadetect catalogs
===================

The "variant" groups in metadetect catalogs take the values "00", "1p", "1m", "2m", and contain catalogs 

The variants are used to calculate calibration factors that account for shear measurement and selection biases.

The columns in a metadetect catalog are:

=======  =========  ===============  ==========  =========
Group               Name             Kind        Meaning
=======  =========  ===============  ==========  =========
shear    {variant}  T                1D float64
shear    {variant}  T_err            1D float64
shear    {variant}  dec              1D float64
shear    {variant}  flags            1D int64
shear    {variant}  g1               1D float64
shear    {variant}  g2               1D float64
shear    {variant}  id               1D int64
shear    {variant}  mag_err_i        1D float64
shear    {variant}  mag_err_r        1D float64
shear    {variant}  mag_err_z        1D float64
shear    {variant}  mag_i            1D float64
shear    {variant}  mag_r            1D float64
shear    {variant}  mag_z            1D float64
shear    {variant}  mcal_psf_T_mean  1D float64
shear    {variant}  mcal_psf_g1      1D float64
shear    {variant}  mcal_psf_g2      1D float64
shear    {variant}  psf_g1           1D float64
shear    {variant}  psf_g2           1D float64
shear    {variant}  ra               1D float64
shear    {variant}  redshift_true    1D float64
shear    {variant}  s2n              1D float64
shear    {variant}  true_g1          1D float64
shear    {variant}  true_g2          1D float64
shear    {variant}  weight           1D float64 
=======  =========  ===============  ==========  =========


