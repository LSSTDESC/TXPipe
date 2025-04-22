Exposure catalogs
=================

Exposure catalogs are used for diagnostics where tangential shear around exposure centers (and later, chip centers) is measured. Currently this file is generated from an OpSim catalog, but we need to figure out how to ingest it from the LSST stack. It may change form at that point.

The columns in an exposure catalog are:


=========  ==============  ==========  =========
Group      Name            Kind        Meaning
=========  ==============  ==========  =========
exposures  ap_corr_map_id  1D int64
exposures  bgmean          1D float64
exposures  bgvar           1D float64
exposures  colorterm1      1D float64
exposures  colorterm2      1D float64
exposures  colorterm3      1D float64
exposures  darktime        1D float64
exposures  date-avg        1D str
exposures  dectel          1D float64
exposures  exptime         1D float64
exposures  filter          1D str
exposures  fluxmag0        1D float64
exposures  fluxmag0err     1D float64
exposures  imgtype         1D str
exposures  magzero_nobj    1D int64
exposures  magzero_rms     1D float64
exposures  mjd-obs         1D float64
exposures  obsid           1D int64
exposures  psf_id          1D int64
exposures  ratel           1D float64
exposures  rotangle        1D float64
exposures  rottype         1D str
exposures  runnum          1D int64
exposures  skywcs_id       1D int64
exposures  testtype        1D str
exposures  timesys         1D str
=========  ==============  ==========  =========

