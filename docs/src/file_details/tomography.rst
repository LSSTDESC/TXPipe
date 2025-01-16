Tomography
==========


Tomography catalogs primarily list the tomographic bin selected for each object. The "bin" column is an integer that indexes the tomographic bin, one value per object in the main (shear or photometry) catalog. The first bin is indexed as zero. A value of -1 indicates that the object was not selected for any tomographic bin (e.g. because its SNR was too low).

The "counts" column is the number of objects in each bin. The "counts_2d" column is the number of objects selected non-tomographically (i.e. the number of objects with any bin value other than -1).

==========  ==========  ==========  =========
Group       Name        Kind        Meaning
==========  ==========  ==========  =========
tomography  bin         1D int32
tomography  counts      1D int32
tomography  counts_2d   1D int32
==========  ==========  ==========  =========


Shear tomography catalogs
-------------------------

Shear tomography catalogs additionally store:
- calibration information, depending on the type of shear catalog
- mean shear values
- effective number counts and ellipticity dispersion sigma_e

==========  ==========  ==========  =========
Group       Name        Kind        Meaning
==========  ==========  ==========  =========
response    R           3D float32
response    R_2d        2D float32
tomography  N_eff       1D float32
tomography  N_eff_2d    1D float32
tomography  bin         1D int32
tomography  counts      1D int32
tomography  counts_2d   1D int32
tomography  mean_e1     1D float32
tomography  mean_e1_2d  1D float32
tomography  mean_e2     1D float32
tomography  mean_e2_2d  1D float32
tomography  sigma_e     1D float32
tomography  sigma_e_2d  1D float32
==========  ==========  ==========  =========

Lens tomography catalogs
------------------------

Lens tomography catalogs additionally store a weight column.  Lens tomography files with the "_unweighted" suffix have unit values in that column. The version without the suffix has the weight values that account for systematic effects.

==========  ===========  ==========  =========
Group       Name         Kind        Meaning
==========  ===========  ==========  =========
tomography  bin          1D int32
tomography  counts       1D int32
tomography  counts_2d    1D int32
tomography  lens_weight  1D float32
==========  ===========  ==========  =========
