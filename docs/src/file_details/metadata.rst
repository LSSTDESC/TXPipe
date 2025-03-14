Metadata
========

The metadata files collect brief statistics for the source and lens samples as a whole, such as number densities. There are two versions of the metadata file, one in HDF5 format and a more human-readable one in YAML form. We should replace the HDF5 format with the YAML format entirely since they are redundant.

=======  =================  ==========  =========
Group    Name               Kind        Meaning
=======  =================  ==========  =========
tracers  N_eff              1D float32
tracers  N_eff_2d           1D float32
tracers  R                  3D float32
tracers  R_2d               2D float32
tracers  lens_counts        1D int32
tracers  lens_counts_2d     1D int32
tracers  lens_density       1D float64
tracers  lens_density_2d    1D float64
tracers  mean_e1            1D float32
tracers  mean_e1_2d         1D float32
tracers  mean_e2            1D float32
tracers  mean_e2_2d         1D float32
tracers  n_eff              1D float32
tracers  sigma_e            1D float32
tracers  sigma_e_2d         1D float32
tracers  source_counts      1D int32
tracers  source_counts_2d   1D int32
tracers  source_density     1D float64
tracers  source_density_2d  1D float64
=======  =================  ==========  =========


