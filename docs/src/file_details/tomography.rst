Tomography catalogs
===================

Tomography catalogs list the tomographic bin selected for each object. The "bin" column is an integer that indexes the tomographic bin, one value per object in the main (shear or photometry) catalog. The first bin is indexed as zero. A value of -1 indicates that the object was not selected for any tomographic bin (e.g. because its SNR was too low).

The "counts" column is the number of objects in each bin. The "counts_2d" column is the number of objects selected non-tomographically (i.e. the number of objects with any bin value other than -1).

The tomography group has attributes ``nbin`` (the number of tomographic bins) and ``zmin_{i}``, ``zmax_{i}`` for each bin ``i``, giving the redshift edges used to define it, and the shear version also has ``catalog_type``.

==========  ==========  ==========  =========
Group       Name        Kind        Meaning
==========  ==========  ==========  =========
tomography  bin         1D int32
counts      counts      1D int32
counts      counts_2d   1D int32
==========  ==========  ==========  =========


Shear tomography catalogs
-------------------------

Shear tomography catalogs are the output of the ``TXComputeShearCalibration*`` stages (see :ref:`Shear calibration guide`). They combine the selection (:ref:`Catalog selections`) with the tomographic bin assignments (:ref:`Tomographic bin assignments`) and add:

- calibration information in the "response" group, whose contents depend on the type of shear catalog;
- mean shear values;
- effective number counts and ellipticity dispersion sigma_e.

The "bin" value of an object is its bin assignment if it passes the selection, or -1 if it does not.

The "counts" group contains one value per tomographic bin, and a "_2d" version for the whole non-tomographic sample:

==========  ==========  ============  =========
Group       Name        Kind          Meaning
==========  ==========  ============  =========
tomography  bin         1D int32      Selected bin for each object, or -1
tomography  class_id    1D int32      Hard link to ``bin``; needed by RAIL summarization stages
counts      counts      1D int32      Number of selected objects in each bin
counts      counts_2d   1D int32      Number of selected objects in the whole sample
counts      N_eff       1D float32    Effective number of objects in each bin
counts      N_eff_2d    1D float32
counts      mean_e1     1D float32    Mean of e1 in each bin
counts      mean_e1_2d  1D float32
counts      mean_e2     1D float32    Mean of e2 in each bin
counts      mean_e2_2d  1D float32
counts      sigma_e     1D float32    Ellipticity dispersion in each bin
counts      sigma_e_2d  1D float32
==========  ==========  ============  =========

The contents of the "response" group, and the way the "bin" values are stored, depend on the shear catalog type.

**Metacal catalogs** (one "bin" value per object, in the "tomography" group, and a per-object response):

=========  ================  ==================  =========
Group      Name              Kind                Meaning
=========  ================  ==================  =========
response   R_gamma           3D float32 (n,2,2)  Per-object shear response matrix
response   R_gamma_mean      3D float32          Mean shear response in each bin
response   R_S               3D float32          Selection response in each bin
response   R_gamma_mean_2d   2D float32          Mean shear response, whole sample
response   R_S_2d            2D float32          Selection response, whole sample
=========  ================  ==================  =========

**Metadetect catalogs**. The variants have different lengths, so the selected bin of each variant is stored separately, as "tomography/{variant}/bin" where variant is one of ns, 1p, 1m, 2p, 2m. The "ns" one is the same dataset as "tomography/bin". There are no per-object response values; the response is only computed for each bin:

=========  =====  ================  =========
Group      Name   Kind              Meaning
=========  =====  ================  =========
response   R      3D float32        Total (shear + selection) response matrix in each bin
response   R_2d   2D float32        Total response matrix, whole sample
=========  =====  ================  =========

**Scalar metadetect catalogs** have the same layout as metadetect but only the variants ns, 1p and 1m, and the response is a scalar: "response/R" has shape (nbin,) and "response/R_2d" has shape (1,).

**Lensfit catalogs**. The additive terms are different in the north and south of the survey:

=========  =======  ================  =========
Group      Name     Kind              Meaning
=========  =======  ================  =========
response   K        1D float32        Multiplicative correction in each bin
response   K_2d     1D float32        Multiplicative correction, whole sample
response   C_N      2D float32        Additive terms, north, in each bin
response   C_S      2D float32        Additive terms, south, in each bin
response   C_2d_N   1D float32        Additive terms, north, whole sample
response   C_2d_S   1D float32        Additive terms, south, whole sample
=========  =======  ================  =========

**HSC catalogs** store per-object ``R`` and per-bin ``K``, ``C`` and ``R_mean`` values.

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
