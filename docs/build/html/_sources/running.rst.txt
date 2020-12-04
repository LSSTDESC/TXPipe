Running a pipeline:
===================

TXPipe is based on ceci on software. To run the pipeline it needs 2 files a *configuration file* and a *Pipeline file*

Examples of both can be found in the `examples folder <https://github.com/LSSTDESC/TXPipe/tree/Documentation/examples>`_. To running the actual pipeline will be as simple as::

    ceci examples/laptop_pipeline.yml

So what goes into this sort of file? We will use *laptop_pipeline.yml* as an example and go through it.
For details on Ceci please see `Ceci Documentation <https://ceci.readthedocs.io/en/latest/index.html>`_.


The pipeline file:
------------------

First part of the file is the stages::

    stages:
    - name: TXSourceSelector     # select and split objects into source bins
    - name: TXTruthLensSelector  # select objects for lens bins
    - name: PZPDFMLZ             # compute p(z) per galaxy using MLZ
    - name: TXPhotozStack        # stack p(z) into n(z)
    - name: TXMainMaps           # make source g1, g2 and lens n_gal maps
    - name: TXAuxiliaryMaps      # make PSF, depth, flag, and other maps
    - name: TXSimpleMask         # combine maps to make a simple mask
    - name: TXDensityMaps        # turn mask and ngal maps into overdensity maps
    - name: TXMapPlots           # make pictures of all the maps
    - name: TXTracerMetadata     # collate metadata
    - name: TXRandomCat          # generate lens bin random catalogs
    - name: TXJackknifeCenters   # Split the area into jackknife regions
    - name: TXTwoPoint           # Compute real-space 2-point correlations
      threads_per_process: 2
    - name: TXBlinding           # Blind the data following Muir et al
      threads_per_process: 2
    - name: TXTwoPointPlots      # Make plots of 2pt correlations
    - name: TXDiagnosticPlots    # Make a suite of diagnostic plots
    - name: TXGammaTFieldCenters # Compute and plot gamma_t around center points
      threads_per_process: 2
    - name: TXGammaTBrightStars  # Compute and plot gamma_t around bright stars
      threads_per_process: 2
    - name: TXGammaTRandoms      # Compute and plot gamma_t around randoms
      threads_per_process: 2
    - name: TXGammaTDimStars     # Compute and plot gamma_t around dim stars
      threads_per_process: 2
    - name: TXRoweStatistics     # Compute and plot Rowe statistics
      threads_per_process: 2
    - name: TXGalaxyStarDensity
    - name: TXGalaxyStarShear
    - name: TXPSFDiagnostics     # Compute and plots other PSF diagnostics
    - name: TXBrighterFatterPlot # Make plots tracking the brighter-fatter effect
    - name: TXPhotozPlots        # Plot the bin n(z)
    - name: TXConvergenceMaps    # Make convergence kappa maps from g1, g2 maps
    - name: TXConvergenceMapPlots # Plot the convergence map
    - name: TXMapCorrelations    # plot the correlations between systematics and data

Each line indicates a pipeline stage that needs to be run. Each stage of course points to one of the stages implemented in :doc:`TXPipe <stages>`.
Note a few lines have have specifically ``threads_per_process: 2`` which is just a way to indicate that these stages should be run with more threads. 
Another option available is ``nprocess: 32`` which would run the stage on 32 processes. 

Next follows modules, which simply is which modules and packages the pipeline stages are defined in.::

    modules: txpipe 

    python_paths:
    - submodules/WLMassMap/python/desc/
    - submodules/TJPCov
    - submodules/FlexZPipe

The ``python_paths`` is modules we need that are not in the TXPipe repo and where to find them. 

Then comes::

    output_dir: data/example/outputs

Which is where all outputs from the stages will be saved.

Launcher, determines how ceci schedules the stages: currently there are 3 options available:
* mini
* parsl
* cwl

See Ceci's documentation on `Launchers <https://ceci.readthedocs.io/en/latest/launchers.html>`_.
In the example we use::

    launcher:
        name: mini
        interval: 1.0

Interval is how often ceci checks if stages have completed. 

Next follows site, again this is a ceci configuration `details <https://ceci.readthedocs.io/en/latest/sites.html>`_::

    site:
        name: local
        max_threads: 2

It tells us where the code is to be run, and the ``max_threads`` that the code will be run on.

Then we have ``config`` which points to the configuration file mentioned as the other needed file.::

    config: examples/config/laptop_config.yml

Then we have the *inputs*::

    inputs:
        # See README for paths to download these files
        shear_catalog: data/example/inputs/shear_catalog.hdf5
        photometry_catalog: data/example/inputs/photometry_catalog.hdf5
        photoz_trained_model: data/example/inputs/cosmoDC2_trees_i25.3.npy
        calibration_table: data/example/inputs/sample_cosmodc2_w10year_errors.dat
        exposures: data/example/inputs/exposures.hdf5
        star_catalog: data/example/inputs/star_catalog.hdf5
        # This file comes with the code
        fiducial_cosmology: data/fiducial_cosmology.yml

This is the location of all the inputs that the pipeline will be run on. In the code the inputs will be refered to by the names i.e. ``shear_catalog`` etc. but here is where which shear catalog it is is specified. 

Finally a few more ceci details::

    resume: True

    log_dir: data/example/logs
    
    pipeline_log: data/example/log.txt

The first here is simply if possible should a restart of the pipeline resume from where it ended or start over.
Secondly for each stage there will be a log file detailing what has been done, where is this saved. 
While ``pipeline_log`` is where the overall parsl pipeline log is saved. 


Config file:
------------

Let us take a look at the how the *configuration file* will look like. 
First we have ``global`` which is configuration options that are shared across all stages::

  global:
    # This is read by many stages that read complete
    # catalog data, and tells them how many rows to read
    # at once
    chunk_rows: 100000
    # These mapping options are also read by a range of stages
    pixelization: healpix
    nside: 512
    sparse: True  # Generate sparse maps - faster if using small areas

Next follows the options for each stages. Options listed here will overwrite the options given at the beginning of the corresponding stage. As an example we can look at ``TXTwoPoint``::

  TXTwoPoint:
    binslop: 0.1
    delta_gamma: 0.02
    do_pos_pos: True
    do_shear_shear: True
    do_shear_pos: True
    flip_g2: True  # use true when using metacal shears
    min_sep: 2.5
    max_sep: 60.0
    nbins: 10
    verbose: 0
    subtract_mean_shear: True

Each line here overwrite the standard configuration given for the ``TXTwoPoint`` stage :doc:`TXTwoPoint <twopoint>`::

  config_options = {
        'calcs':[0,1,2],
        'min_sep':0.5,
        'max_sep':300.,
        'nbins':9,
        'bin_slop':0.1,
        'sep_units':'arcmin',
        'flip_g2':True,
        'cores_per_task':20,
        'verbose':1,
        'source_bins':[-1],
        'lens_bins':[-1],
        'reduce_randoms_size':1.0,
        'do_shear_shear': True,
        'do_shear_pos': True,
        'do_pos_pos': True,
        'var_methods': 'jackknife',
        'use_true_shear': False,
        'subtract_mean_shear':False

.. note::

  we don't need to replace all options, the options we don't replace will just use the options from the file.

