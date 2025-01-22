Binned catalogs
===============

Binned catalogs are split with each tomographic bin having its own HDF5 group.

They can also include calibrations, applied such that they can be used directly in 2pt calculations or other science measurements.


Binned shear catalogs
---------------------

The tomographically binned shear catalogs also include the shear calibration to correct for measurement and selection biases. 

The bin index b runs from 0 to nbin-1, or "all" tag for a combined non-tomographic sample. The "all" bin is not simply the same as all the other bins combined, because the selection bias calibration is different.

Like the original source shear catalogs, the binned shear catalogs may include only a subset of magnitude values depending on what was measured during the pipeline. An example using riz bands is shown below.

=======  =======  =========  ==========  =========
Group             Name       Kind        Meaning
=======  =======  =========  ==========  =========
shear    bin_{b}  ra         1D float64
shear    bin_{b}  dec        1D float64
shear    bin_{b}  g1         1D float64
shear    bin_{b}  g2         1D float64
shear    bin_{b}  mag_r      1D float64
shear    bin_{b}  mag_i      1D float64
shear    bin_{b}  mag_z      1D float64
shear    bin_{b}  mag_err_r  1D float64
shear    bin_{b}  mag_err_i  1D float64
shear    bin_{b}  mag_err_z  1D float64
shear    bin_{b}  weight     1D float64
=======  =======  =========  ==========  =========



Binned lens catalogs
--------------------

There are two classes of binned lens catalog, weighted and unweighted. The latter includes a varying weight to account for survery property density correlations. In the former the weight is unity.  

The final weighted catalog should be used for all science measurements.

The bin index b runs from 0 to nbin-1, or "all" tag for a combined non-tomographic sample.

=======  =======  =================  ==========  =========
Group             Name               Kind        Meaning
=======  =======  =================  ==========  =========
lens     bin_{b}  comoving_distance  1D float64
lens     bin_{b}  dec                1D float64
lens     bin_{b}  mag_err_g          1D float64
lens     bin_{b}  mag_err_i          1D float64
lens     bin_{b}  mag_err_r          1D float64
lens     bin_{b}  mag_err_u          1D float64
lens     bin_{b}  mag_err_y          1D float64
lens     bin_{b}  mag_err_z          1D float64
lens     bin_{b}  mag_g              1D float64
lens     bin_{b}  mag_i              1D float64
lens     bin_{b}  mag_r              1D float64
lens     bin_{b}  mag_u              1D float64
lens     bin_{b}  mag_y              1D float64
lens     bin_{b}  mag_z              1D float64
lens     bin_{b}  ra                 1D float64
lens     bin_{b}  weight             1D float64
=======  =======  =================  ==========  =========


Binned random catalogs
----------------------

The binned random catalogs match the tomography of the lens sample. Two random binned catalogs are generated, one with a much smaller sample size, suitable for use when measuring the density-density correlation, and a larger one suitable for measuring the lensing-density cross-signal.  The smaller catalog has the suffix "_sub".

=======  =====  =================  ==========  =========
Group           Name               Kind        Meaning
=======  =====  =================  ==========  =========
randoms  bin_0  ra                 1D float32
randoms  bin_0  dec                1D float32
randoms  bin_0  comoving_distance  1D float32
randoms  bin_0  z                  1D float32
=======  =====  =================  ==========  =========


Binned star catalogs
--------------------

Binned star catalogs are not tomographic, but instead split into classes used differently in systematics tests. Currently that means two bins, "bright" and "dim".

=======  ==========  ======  ==========  =========
Group                Name    Kind        Meaning
=======  ==========  ======  ==========  =========
stars    bin_bright  dec     1D float64
stars    bin_bright  ra      1D float64
stars    bin_dim     dec     1D float64
stars    bin_dim     ra      1D float64
=======  ==========  ======  ==========  =========

