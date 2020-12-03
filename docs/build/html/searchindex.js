Search.setIndex({docnames:["FlexZPipe","TJPCov","WLMassMap","auxiliary_maps","base_stage","blinding","convergence","covariance","data_types","diagnostics","exposure_info","index","ingest_redmagic","input_cats","installation","lens_selector","map_correlations","map_plots","mapping","maps","masks","metacal_gcr_input","metadata","noise_maps","photoz","photoz_mlz","photoz_stack","plotting","psf_diagnostics","random_cats","randoms","source_selector","stages","submodules","twopoint","twopoint_fourier","ui","utils"],envversion:{"sphinx.domains.c":2,"sphinx.domains.changeset":1,"sphinx.domains.citation":1,"sphinx.domains.cpp":2,"sphinx.domains.index":1,"sphinx.domains.javascript":2,"sphinx.domains.math":2,"sphinx.domains.python":2,"sphinx.domains.rst":2,"sphinx.domains.std":1,"sphinx.ext.todo":2,"sphinx.ext.viewcode":1,sphinx:56},filenames:["FlexZPipe.rst","TJPCov.rst","WLMassMap.rst","auxiliary_maps.rst","base_stage.rst","blinding.rst","convergence.rst","covariance.rst","data_types.rst","diagnostics.rst","exposure_info.rst","index.rst","ingest_redmagic.rst","input_cats.rst","installation.rst","lens_selector.rst","map_correlations.rst","map_plots.rst","mapping.rst","maps.rst","masks.rst","metacal_gcr_input.rst","metadata.rst","noise_maps.rst","photoz.rst","photoz_mlz.rst","photoz_stack.rst","plotting.rst","psf_diagnostics.rst","random_cats.rst","randoms.rst","source_selector.rst","stages.rst","submodules.rst","twopoint.rst","twopoint_fourier.rst","ui.rst","utils.rst"],objects:{"txpipe.TXAuxiliaryMaps":{accumulate_maps:[3,2,1,""],choose_pixel_scheme:[3,2,1,""],data_iterator:[3,2,1,""],finalize_mappers:[3,2,1,""],name:[3,3,1,""],prepare_mappers:[3,2,1,""]},"txpipe.base_stage":{PipelineStage:[4,1,1,""]},"txpipe.base_stage.PipelineStage":{__dict__:[4,3,1,""],__init__:[4,2,1,""],__init_subclass__:[4,2,1,""],__module__:[4,3,1,""],__weakref__:[4,3,1,""],combined_iterators:[4,2,1,""],comm:[4,2,1,""],config:[4,2,1,""],config_options:[4,3,1,""],data_ranges_by_rank:[4,2,1,""],doc:[4,3,1,""],execute:[4,2,1,""],finalize:[4,2,1,""],gather_provenance:[4,2,1,""],generate_command:[4,2,1,""],generate_cwl:[4,2,1,""],get_input:[4,2,1,""],get_input_type:[4,2,1,""],get_module:[4,2,1,""],get_output:[4,2,1,""],get_output_type:[4,2,1,""],get_stage:[4,2,1,""],input_tags:[4,2,1,""],inputs:[4,3,1,""],is_mpi:[4,2,1,""],is_parallel:[4,2,1,""],iterate_fits:[4,2,1,""],iterate_hdf:[4,2,1,""],main:[4,2,1,""],name:[4,3,1,""],open_input:[4,2,1,""],open_output:[4,2,1,""],output_tags:[4,2,1,""],outputs:[4,3,1,""],parallel:[4,3,1,""],pipeline_stages:[4,3,1,""],rank:[4,2,1,""],read_config:[4,2,1,""],run:[4,2,1,""],size:[4,2,1,""],split_tasks_by_rank:[4,2,1,""],usage:[4,2,1,""]},"txpipe.blinding":{TXBlinding:[5,1,1,""],TXNullBlinding:[5,1,1,""]},"txpipe.blinding.TXBlinding":{run:[5,2,1,""]},"txpipe.blinding.TXNullBlinding":{run:[5,2,1,""]},"txpipe.convergence":{TXConvergenceMapPlots:[6,1,1,""],TXConvergenceMaps:[6,1,1,""]},"txpipe.covariance":{TXFourierGaussianCovariance:[7,1,1,""],TXRealGaussianCovariance:[7,1,1,""]},"txpipe.data_types":{base:[8,0,0,"-"],types:[8,0,0,"-"]},"txpipe.data_types.base":{DataFile:[8,1,1,""],Directory:[8,1,1,""],FileCollection:[8,1,1,""],FileValidationError:[8,4,1,""],FitsFile:[8,1,1,""],HDFFile:[8,1,1,""],PNGFile:[8,1,1,""],TextFile:[8,1,1,""],YamlFile:[8,1,1,""]},"txpipe.data_types.base.DataFile":{add_provenance:[8,2,1,""],close:[8,2,1,""],generate_provenance:[8,2,1,""],make_name:[8,2,1,""],open:[8,2,1,""],read_provenance:[8,2,1,""],suffix:[8,3,1,""],supports_parallel_write:[8,3,1,""],validate:[8,2,1,""],write_provenance:[8,2,1,""]},"txpipe.data_types.base.Directory":{open:[8,2,1,""],read_provenance:[8,2,1,""],suffix:[8,3,1,""],write_provenance:[8,2,1,""]},"txpipe.data_types.base.FileCollection":{path_for_file:[8,2,1,""],read_listing:[8,2,1,""],suffix:[8,3,1,""],write_listing:[8,2,1,""]},"txpipe.data_types.base.FitsFile":{close:[8,2,1,""],missing_columns:[8,2,1,""],open:[8,2,1,""],read_provenance:[8,2,1,""],required_columns:[8,3,1,""],suffix:[8,3,1,""],validate:[8,2,1,""],write_provenance:[8,2,1,""]},"txpipe.data_types.base.HDFFile":{close:[8,2,1,""],open:[8,2,1,""],read_provenance:[8,2,1,""],required_datasets:[8,3,1,""],suffix:[8,3,1,""],supports_parallel_write:[8,3,1,""],validate:[8,2,1,""],write_provenance:[8,2,1,""]},"txpipe.data_types.base.PNGFile":{close:[8,2,1,""],open:[8,2,1,""],read_provenance:[8,2,1,""],suffix:[8,3,1,""],write_provenance:[8,2,1,""]},"txpipe.data_types.base.TextFile":{suffix:[8,3,1,""]},"txpipe.data_types.base.YamlFile":{read:[8,2,1,""],read_provenance:[8,2,1,""],suffix:[8,3,1,""],write:[8,2,1,""],write_provenance:[8,2,1,""]},"txpipe.data_types.types":{CSVFile:[8,1,1,""],ClusteringNoiseMaps:[8,1,1,""],FiducialCosmology:[8,1,1,""],LensingNoiseMaps:[8,1,1,""],MapsFile:[8,1,1,""],NOfZFile:[8,1,1,""],PhotozPDFFile:[8,1,1,""],RandomsCatalog:[8,1,1,""],SACCFile:[8,1,1,""],ShearCatalog:[8,1,1,""],TomographyCatalog:[8,1,1,""],metacalibration_names:[8,5,1,""]},"txpipe.data_types.types.CSVFile":{save_file:[8,2,1,""],suffix:[8,3,1,""]},"txpipe.data_types.types.ClusteringNoiseMaps":{number_of_realizations:[8,2,1,""],read_density_split:[8,2,1,""]},"txpipe.data_types.types.FiducialCosmology":{to_ccl:[8,2,1,""]},"txpipe.data_types.types.LensingNoiseMaps":{number_of_realizations:[8,2,1,""],read_rotation:[8,2,1,""],required_datasets:[8,3,1,""]},"txpipe.data_types.types.MapsFile":{list_maps:[8,2,1,""],plot:[8,2,1,""],plot_gnomonic:[8,2,1,""],plot_healpix:[8,2,1,""],read_gnomonic:[8,2,1,""],read_healpix:[8,2,1,""],read_map:[8,2,1,""],read_map_info:[8,2,1,""],read_mask:[8,2,1,""],required_datasets:[8,3,1,""],write_map:[8,2,1,""]},"txpipe.data_types.types.NOfZFile":{get_n_of_z:[8,2,1,""],get_n_of_z_spline:[8,2,1,""],get_nbin:[8,2,1,""],plot:[8,2,1,""],required_datasets:[8,3,1,""],save_plot:[8,2,1,""]},"txpipe.data_types.types.PhotozPDFFile":{required_datasets:[8,3,1,""]},"txpipe.data_types.types.RandomsCatalog":{required_datasets:[8,3,1,""]},"txpipe.data_types.types.SACCFile":{close:[8,2,1,""],open:[8,2,1,""],read_provenance:[8,2,1,""],suffix:[8,3,1,""]},"txpipe.data_types.types.ShearCatalog":{catalog_type:[8,2,1,""],read_catalog_info:[8,2,1,""]},"txpipe.data_types.types.TomographyCatalog":{read_nbin:[8,2,1,""],read_zbins:[8,2,1,""],required_datasets:[8,3,1,""]},"txpipe.diagnostics":{TXDiagnosticPlots:[9,1,1,""]},"txpipe.exposure_info":{TXExposureInfo:[10,1,1,""]},"txpipe.ingest_redmagic":{TXIngestRedmagic:[12,1,1,""]},"txpipe.input_cats":{TXBuzzardMock:[13,1,1,""],TXCosmoDC2Mock:[13,1,1,""],make_mock_photometry:[13,5,1,""]},"txpipe.input_cats.TXCosmoDC2Mock":{load_metacal_response_model:[13,2,1,""],make_mock_metacal:[13,2,1,""],remove_undetected:[13,2,1,""],write_output:[13,2,1,""]},"txpipe.lens_selector":{TXBaseLensSelector:[15,1,1,""],TXMeanLensSelector:[15,1,1,""],TXTruthLensSelector:[15,1,1,""]},"txpipe.lens_selector.TXBaseLensSelector":{run:[15,2,1,""],select_lens:[15,2,1,""],setup_output:[15,2,1,""],write_global_values:[15,2,1,""],write_tomography:[15,2,1,""]},"txpipe.map_correlations":{TXMapCorrelations:[16,1,1,""]},"txpipe.map_plots":{TXMapPlots:[17,1,1,""]},"txpipe.mapping":{basic_maps:[18,0,0,"-"],dr1:[18,0,0,"-"]},"txpipe.mapping.basic_maps":{FlagMapper:[18,1,1,""],Mapper:[18,1,1,""]},"txpipe.mapping.basic_maps.FlagMapper":{add_data:[18,2,1,""],finalize:[18,2,1,""]},"txpipe.mapping.basic_maps.Mapper":{add_data:[18,2,1,""],finalize:[18,2,1,""]},"txpipe.mapping.dr1":{BrightObjectMapper:[18,1,1,""],DepthMapperDR1:[18,1,1,""]},"txpipe.mapping.dr1.BrightObjectMapper":{add_data:[18,2,1,""],finalize:[18,2,1,""]},"txpipe.mapping.dr1.DepthMapperDR1":{add_data:[18,2,1,""],finalize:[18,2,1,""]},"txpipe.maps":{TXBaseMaps:[19,1,1,""],TXDensityMaps:[19,1,1,""],TXExternalLensMaps:[19,1,1,""],TXLensMaps:[19,1,1,""],TXMainMaps:[19,1,1,""],TXSourceMaps:[19,1,1,""]},"txpipe.maps.TXBaseMaps":{accumulate_maps:[19,2,1,""],choose_pixel_scheme:[19,2,1,""],data_iterator:[19,2,1,""],finalize_mappers:[19,2,1,""],prepare_mappers:[19,2,1,""],save_maps:[19,2,1,""]},"txpipe.maps.TXExternalLensMaps":{data_iterator:[19,2,1,""]},"txpipe.maps.TXLensMaps":{accumulate_maps:[19,2,1,""],data_iterator:[19,2,1,""],finalize_mappers:[19,2,1,""],prepare_mappers:[19,2,1,""]},"txpipe.maps.TXMainMaps":{data_iterator:[19,2,1,""],finalize_mappers:[19,2,1,""],prepare_mappers:[19,2,1,""]},"txpipe.maps.TXSourceMaps":{accumulate_maps:[19,2,1,""],data_iterator:[19,2,1,""],finalize_mappers:[19,2,1,""],prepare_mappers:[19,2,1,""]},"txpipe.masks":{TXSimpleMask:[20,1,1,""]},"txpipe.metacal_gcr_input":{TXIngestStars:[21,1,1,""],TXMetacalGCRInput:[21,1,1,""]},"txpipe.metadata":{TXTracerMetadata:[22,1,1,""]},"txpipe.noise_maps":{TXExternalLensNoiseMaps:[23,1,1,""],TXNoiseMaps:[23,1,1,""],TXSourceNoiseMaps:[23,1,1,""]},"txpipe.noise_maps.TXExternalLensNoiseMaps":{accumulate_maps:[23,2,1,""],choose_pixel_scheme:[23,2,1,""],data_iterator:[23,2,1,""],finalize_mappers:[23,2,1,""],prepare_mappers:[23,2,1,""]},"txpipe.noise_maps.TXSourceNoiseMaps":{accumulate_maps:[23,2,1,""],choose_pixel_scheme:[23,2,1,""],data_iterator:[23,2,1,""],finalize_mappers:[23,2,1,""],prepare_mappers:[23,2,1,""]},"txpipe.photoz":{TXRandomPhotozPDF:[24,1,1,""]},"txpipe.photoz.TXRandomPhotozPDF":{calculate_photozs:[24,2,1,""],prepare_output:[24,2,1,""],run:[24,2,1,""],write_output:[24,2,1,""]},"txpipe.photoz_mlz":{PZPDFMLZ:[25,1,1,""]},"txpipe.photoz_mlz.PZPDFMLZ":{calculate_photozs:[25,2,1,""],prepare_output:[25,2,1,""],run:[25,2,1,""],write_output:[25,2,1,""]},"txpipe.photoz_stack":{TXPhotozPlots:[26,1,1,""],TXPhotozSourceStack:[26,1,1,""],TXPhotozStack:[26,1,1,""],TXSourceTrueNumberDensity:[26,1,1,""],TXTrueNumberDensity:[26,1,1,""]},"txpipe.photoz_stack.TXPhotozSourceStack":{get_metadata:[26,2,1,""],run:[26,2,1,""]},"txpipe.photoz_stack.TXPhotozStack":{get_metadata:[26,2,1,""]},"txpipe.photoz_stack.TXSourceTrueNumberDensity":{get_metadata:[26,2,1,""]},"txpipe.photoz_stack.TXTrueNumberDensity":{get_metadata:[26,2,1,""]},"txpipe.plotting":{correlations:[27,0,0,"-"],histogram:[27,0,0,"-"]},"txpipe.plotting.correlations":{apply_galaxy_bias_ggl:[27,5,1,""],axis_setup:[27,5,1,""],extract_observables_plot_data:[27,5,1,""],full_3x2pt_plots:[27,5,1,""],make_axis:[27,5,1,""],make_plot:[27,5,1,""],make_theory_plot_data:[27,5,1,""],smooth_nz:[27,5,1,""]},"txpipe.plotting.histogram":{manual_step_histogram:[27,5,1,""]},"txpipe.psf_diagnostics":{TXBrighterFatterPlot:[28,1,1,""],TXPSFDiagnostics:[28,1,1,""],TXRoweStatistics:[28,1,1,""],TXStarDensityTests:[28,1,1,""],TXStarShearTests:[28,1,1,""]},"txpipe.random_cats":{TXRandomCat:[29,1,1,""]},"txpipe.randoms":{randoms:[30,0,0,"-"]},"txpipe.randoms.randoms":{random_points_in_quadrilateral:[30,5,1,""],random_points_in_triangle:[30,5,1,""]},"txpipe.source_selector":{TXSourceSelector:[31,1,1,""]},"txpipe.source_selector.TXSourceSelector":{apply_classifier:[31,2,1,""],calculate_tomography:[31,2,1,""],run:[31,2,1,""],setup_output:[31,2,1,""],write_global_values:[31,2,1,""],write_tomography:[31,2,1,""]},"txpipe.twopoint":{Measurement:[34,1,1,""],TXGammaTBrightStars:[34,1,1,""],TXGammaTDimStars:[34,1,1,""],TXGammaTFieldCenters:[34,1,1,""],TXGammaTRandoms:[34,1,1,""],TXJackknifeCenters:[34,1,1,""],TXTwoPoint:[34,1,1,""],TXTwoPointLensCat:[34,1,1,""],TXTwoPointPlots:[34,1,1,""]},"txpipe.twopoint.Measurement":{__getnewargs__:[34,2,1,""],__new__:[34,2,1,""],__repr__:[34,2,1,""],corr_type:[34,3,1,""],i:[34,3,1,""],j:[34,3,1,""],object:[34,3,1,""]},"txpipe.twopoint.TXGammaTBrightStars":{read_nbin:[34,2,1,""],run:[34,2,1,""]},"txpipe.twopoint.TXGammaTDimStars":{read_nbin:[34,2,1,""],run:[34,2,1,""]},"txpipe.twopoint.TXGammaTFieldCenters":{read_nbin:[34,2,1,""],run:[34,2,1,""]},"txpipe.twopoint.TXGammaTRandoms":{read_nbin:[34,2,1,""],run:[34,2,1,""]},"txpipe.twopoint.TXJackknifeCenters":{plot:[34,2,1,""]},"txpipe.twopoint.TXTwoPoint":{calculate_pos_pos:[34,2,1,""],calculate_shear_pos:[34,2,1,""],calculate_shear_shear:[34,2,1,""],call_treecorr:[34,2,1,""],get_calibrated_catalog_bin:[34,2,1,""],read_nbin:[34,2,1,""],run:[34,2,1,""]},"txpipe.twopoint.TXTwoPointPlots":{get_theta_xi_err:[34,2,1,""],get_theta_xi_err_jk:[34,2,1,""]},"txpipe.twopoint_fourier":{Measurement:[35,1,1,""],TXTwoPointFourier:[35,1,1,""],TXTwoPointPlotsFourier:[35,1,1,""]},"txpipe.twopoint_fourier.Measurement":{__getnewargs__:[35,2,1,""],__new__:[35,2,1,""],__repr__:[35,2,1,""],corr_type:[35,3,1,""],i:[35,3,1,""],j:[35,3,1,""],l:[35,3,1,""],value:[35,3,1,""],win:[35,3,1,""]},"txpipe.utils":{calibration_tools:[37,0,0,"-"],fitting:[37,0,0,"-"],hdf_tools:[37,0,0,"-"],healpix:[37,0,0,"-"],misc:[37,0,0,"-"],mpi_utils:[37,0,0,"-"],provenance:[37,0,0,"-"],theory:[37,0,0,"-"],timer:[37,0,0,"-"]},"txpipe.utils.calibration_tools":{ParallelCalibratorMetacal:[37,1,1,""],ParallelCalibratorNonMetacal:[37,1,1,""],read_shear_catalog_type:[37,5,1,""]},"txpipe.utils.calibration_tools.ParallelCalibratorMetacal":{add_data:[37,2,1,""],collect:[37,2,1,""]},"txpipe.utils.calibration_tools.ParallelCalibratorNonMetacal":{add_data:[37,2,1,""],collect:[37,2,1,""]},"txpipe.utils.fitting":{fit_straight_line:[37,5,1,""]},"txpipe.utils.hdf_tools":{create_dataset_early_allocated:[37,5,1,""],repack:[37,5,1,""]},"txpipe.utils.healpix":{dilated_healpix_map:[37,5,1,""]},"txpipe.utils.misc":{unique_list:[37,5,1,""]},"txpipe.utils.mpi_utils":{mpi_reduce_large:[37,5,1,""]},"txpipe.utils.provenance":{find_module_versions:[37,5,1,""],get_caller_directory:[37,5,1,""],git_current_revision:[37,5,1,""],git_diff:[37,5,1,""]},"txpipe.utils.theory":{theory_3x2pt:[37,5,1,""]},submodules:{FlexZPipe:[0,0,0,"-"],TJPCov:[1,0,0,"-"],WLMassMap:[2,0,0,"-"]},txpipe:{TXAuxiliaryMaps:[3,1,1,""],base_stage:[4,0,0,"-"],blinding:[5,0,0,"-"],convergence:[6,0,0,"-"],covariance:[7,0,0,"-"],diagnostics:[9,0,0,"-"],exposure_info:[10,0,0,"-"],ingest_redmagic:[12,0,0,"-"],input_cats:[13,0,0,"-"],lens_selector:[15,0,0,"-"],map_correlations:[16,0,0,"-"],map_plots:[17,0,0,"-"],maps:[19,0,0,"-"],masks:[20,0,0,"-"],metacal_gcr_input:[21,0,0,"-"],metadata:[22,0,0,"-"],noise_maps:[23,0,0,"-"],photoz:[24,0,0,"-"],photoz_mlz:[25,0,0,"-"],photoz_stack:[26,0,0,"-"],psf_diagnostics:[28,0,0,"-"],random_cats:[29,0,0,"-"],source_selector:[31,0,0,"-"],twopoint:[34,0,0,"-"],twopoint_fourier:[35,0,0,"-"]}},objnames:{"0":["py","module","Python module"],"1":["py","class","Python class"],"2":["py","method","Python method"],"3":["py","attribute","Python attribute"],"4":["py","exception","Python exception"],"5":["py","function","Python function"]},objtypes:{"0":"py:module","1":"py:class","2":"py:method","3":"py:attribute","4":"py:exception","5":"py:function"},terms:{"2x2":37,"3x2pt":37,"abstract":19,"case":[4,8,34,37],"class":[3,4,5,6,7,8,9,10,12,13,15,16,17,18,19,20,21,22,23,24,25,26,28,29,31,34,35,37],"default":[4,37],"export":4,"final":[3,4,18,19,23,37],"float":37,"function":[4,8,13,24,26,32,37],"import":37,"int":[4,15,24,25,30,31,37],"long":[13,30],"new":[4,8,34,35,37],"null":[5,34],"return":[3,4,8,19,23,24,25,26,30,34,35,37],"static":[8,34,35],"true":[4,8,18,26,27,37],"var":19,"while":37,For:[4,34],Not:28,One:37,The:[4,8,11,14,15,24,25,30,31,35,37],Then:4,These:[8,32,33],Use:[13,37],Used:[34,35],Uses:26,Using:8,_1m:8,_1p:8,_2m:8,_2p:8,__dict__:4,__doc__:4,__getnewargs__:[34,35],__init__:4,__init_subclass__:4,__main__:4,__module__:4,__new__:[34,35],__repr__:[34,35],__version__:37,__weakref__:4,_cl:[34,35],about:[8,22],abov:[14,19],access:14,accord:31,accumul:[15,26,31],accumulate_map:[3,19,23],actual:[22,24,25,30],add:[4,31,37],add_data:[18,37],add_proven:8,added:[4,13],addit:4,advanc:[8,13,21],after:[28,34,35],aggreg:4,algorithm:15,alia:[34,35],all:[4,6,8,14,17,24,25,35,37],alloc:[4,26,37],allow:37,along:8,also:[4,8,13,14,21,31,37],alwai:8,analysi:[5,15,24,25,26,31,34],ani:[3,4,13,19,23,24,25,30,37],anti:30,anyth:[8,22],app:4,appli:[24,31],apply_classifi:31,apply_galaxy_bias_ggl:27,arg:[3,4,5,6,7,8,9,10,12,13,15,16,17,19,20,21,22,23,24,25,26,28,29,31,34,35,37],argpars:4,argument:[4,37],around:22,arrai:[8,13,15,24,25,26,30,31,37],assum:[13,24,25],astr511:13,astropi:14,attribut:[4,8,37],auto:35,autocorrel:28,automat:[4,37],auxiliary_map:4,auxilliari:[11,32],avail:[4,13,17,21,37],averag:[15,31],axes:27,axi:[24,25],axis_setup:27,band:13,bar:4,barnei:28,base:[3,5,6,7,8,9,10,11,12,13,15,16,17,18,19,20,21,22,23,24,25,26,28,29,31,34,35],base_stag:[4,5,6,7,9,10,12,13,15,16,17,19,20,21,22,23,24,25,26,28,29,31,34,35],basic:19,basic_map:18,baz:4,becaus:[19,28],been:4,being:[4,13,21],between:34,bia:[15,31,37],bias:[15,31],bin:[8,13,15,24,25,26,31,34,35,37],bin_index:8,bin_typ:8,blind:[4,11,32],block:34,bonus:4,bool:37,boss:15,boss_galaxy_t:15,both:[26,35,37],bright:3,brightobjectmapp:18,build:37,bunch:32,bundl:8,c_ell:37,calcul:[22,24,34],calculate_photoz:[24,25],calculate_pos_po:34,calculate_shear_po:34,calculate_shear_shear:34,calculate_tomographi:31,calibr:[15,31,37],calibration_tool:37,call:[4,8,28,34,37],call_treecorr:34,calld:4,caller:37,can:[3,4,8,14,19,23,24,30,37],cart:8,catalog:[8,19,21,22,24,25,31,37],catalog_typ:8,ceci:[4,14],center:34,chang:4,check:8,choic:31,choose_pixel_schem:[3,19,23],chosen:8,chunk:[3,4,15,19,23,24,25,31,37],chunk_row:4,classifi:31,classmethod:[4,8],clockwis:30,close:[8,15,24,31],cls:4,cluster:15,clusteringnoisemap:8,code:[4,14,37],col:4,collat:22,collect:[8,15,31,37],column:[4,8,15,19,26,31,37],com:37,combin:[4,19,35,37],combined_iter:4,come:35,comm:[3,4,5,6,7,9,10,12,13,15,16,17,18,19,20,21,22,23,24,25,26,28,29,31,34,35,37],comm_world:4,command:4,commun:[4,37],communict:37,communu:37,complain:8,complic:13,compon:34,comput:[24,25,35,37],concret:8,config:4,config_opt:4,configur:[4,13,31,34,37],consist:[30,31],constant:37,construct:[4,31],constructor:4,contain:[4,8,13,24,25,31,37],content:11,contigu:37,converg:[4,6],convergencemap:[11,32],convert:19,coordin:37,copi:[5,34,35],cori:14,corr:27,corr_typ:[34,35],correct:34,correl:[11,27,32,35],correspond:4,cosmo:[13,27],cosmolog:[13,21,37],cosmology_fil:37,could:4,count:[19,26,27],covari:[4,11,32,34,37],creat:[3,4,15,19,23,31,34,35,37],create_dataset_early_alloc:37,cross:[28,35],csv:8,csvfile:8,cubic:8,current:[4,35],cut:[15,31],cwl:4,data:[3,4,5,8,13,14,15,18,19,21,23,24,25,26,27,31,34,37],data_iter:[3,19,23],data_ranges_by_rank:4,data_typ:[11,32],datafil:8,datafram:8,datapoint:34,dataset:37,datayp:34,dc2:[13,21],debug:[4,37],dec:[8,34],decid:13,defin:[4,8],definit:8,deletet:5,delta:[19,26],delta_gamma:37,densiti:[19,28],depend:14,deproject:35,depth:3,depthmapperdr1:18,describ:19,design:[24,25],detect:[13,37],determin:[34,37],dev:37,deviat:13,diagnost:[4,11],dict:[4,8,13,24,25,31,37],dictionari:[3,4,8,13,19,23,37],diff:37,differ:[8,24,26,37],dilat:37,dilated_healpix_map:37,dimens:30,directli:19,directori:[8,37],disc:13,distribut:30,divid:[26,37],do_g:18,do_len:18,doc:4,doe:[8,24,25,26,37],doesn:22,domin:35,don:[8,22],done:[24,25],dr1:18,dr9:15,dtype:37,due:37,duplic:37,each:[4,8,15,19,24,25,26,31,37],edg:[8,27,37],edu:13,egg:4,either:[8,30,34,37],element:[4,37],ell:35,elsewher:[4,19],end:[4,8,15,24,25,31,37],entri:4,environ:14,err:34,error:[4,37],errorbar:34,esk:4,estim:[24,25,37],etc:37,evalu:[24,25],even:37,everi:37,exact:8,exampl:[14,37],except:[4,8,19],execut:4,exist:[3,8,13,19,23],expect:8,exposur:34,exposure_info:[4,11,32],extend:4,extern:[19,34],extra:11,extra_proven:8,extract_observables_plot_data:27,facil:4,factor:[15,31,34,37],faculti:13,fail:37,fake:26,fals:[4,8,18,27,37],far:[13,32],faster:[19,37],featur:[4,25,31],fiduci:37,fiducialcosmolog:8,field:[34,35],fig:27,fig_kw:8,figur:27,file:[4,5,8,13,14,15,24,25,26,31,34,37],filecollect:8,filenam:[8,37],filevalidationerror:8,fill:37,final_nam:4,finalize_mapp:[3,19,23],financ:28,find:[4,37],find_module_vers:37,first:[4,28,30],fit:[4,8,37],fit_bia:27,fit_straight_lin:37,fitsfil:8,fitsio:[8,14],five:[24,25],flag:[3,31,35],flag_exponent_max:18,flagmapp:18,flexzpip:[11,33],follow:[34,35],foo:4,form:4,format:[8,24,25,34,35],former:14,fourier:37,fourth:30,from:[3,4,8,13,14,19,21,23,24,25,31,34,35,37],full:[8,24,25,37],full_3x2pt_plot:27,futur:[4,35],galaxi:[15,19,23,28,35],gamma_x:34,gather_proven:4,gener:[3,4,8,13,19,23,24,25,30,31,34,35,37],generate_command:4,generate_cwl:4,generate_proven:8,get:[8,26,34,37],get_calibrated_catalog_bin:34,get_caller_directori:37,get_input:4,get_input_typ:4,get_metadata:26,get_modul:4,get_n_of_z:8,get_n_of_z_splin:8,get_nbin:8,get_output:4,get_output_typ:4,get_stag:4,get_theta_xi_err:34,get_theta_xi_err_jk:34,git:37,git_current_revis:37,git_diff:37,github:4,give:31,given:[4,34,37],global:14,gov:14,gradient:37,grandpar:37,greater:37,group:[8,14,15,31,37],group_nam:4,guess:37,h5group:8,h5py:[8,15,24,25,31,37],handi:[13,21],handl:4,has:[8,19],have:[4,13,14,24],hdf5:[4,8,24,25,37],hdf:13,hdf_tool:37,hdffile:8,hdu:8,hdunum:4,healpi:14,healpix:37,here:[4,13,24],high:28,histogram:[26,27],hold:4,home:4,how:37,howev:[24,25],http:[13,15,37],imag:[13,21],implement:[4,8,11],incid:30,includ:[4,35],incorrig:28,increment:37,index:[11,15,24,25,31],indic:8,individu:23,info:4,inform:[4,8,31,37],infrastructur:[13,21],ingest:11,ingest_redmag:[4,12],inherit:19,init:[3,19,23],input:[3,4,8,14,19,21,23,24,25,26,31,34,35,37],input_cat:[4,13],input_tag:4,insid:8,instal:[8,11],instanc:[4,8,34,35,37],instanti:[4,8],instead:[3,4,8,19,23,24,25,34],interact:34,intercept:37,intermediari:8,is_mpi:4,is_parallel:4,item:[4,37],iter:[3,4,15,19,23,26,31,37],iterate_fit:4,iterate_hdf:4,its:[4,13,26,34,37],itself:8,ivez:13,jackknif:34,job:14,jone:13,just:[4,8,13,22,24,25],kappa_:6,kappa_b:6,keep:4,kei:[8,37],kept:34,keyword:37,kind:[8,13,21],know:8,kwarg:[4,8,27,37],label:27,larg:37,later:[4,19,24,25,37],latter:14,leav:37,len:[11,19,26,32,34,35,37],length:4,lens:[19,34,35],lens_0:37,lens_1:37,lens_bin:[15,18],lens_photoz_stack:34,lens_selector:[4,15],lens_tomography_catalog:34,lensfit:[31,37],lensingnoisemap:8,letter:28,level:8,librari:8,like:[4,8],limit:13,line:[4,37],list:[4,8,15,31,35,37],list_map:8,live:37,load:[3,5,13,19,23,24,26,34],load_metacal_response_model:13,load_mod:8,locat:4,log10:13,log:24,log_dir:4,login:14,look:[4,37],loop:[3,4,15,19,23,24,26,31],lost:28,lse:13,lsst:14,lsst_snrdoc:13,lupton:13,lynn:13,machin:14,machineri:8,made:[8,19,31],mag_threshold:18,magnitud:[24,25,31],mai:4,main:4,mainli:[13,21],maintain:37,make:[6,17,19,26,31,34,37],make_axi:27,make_mock_metac:13,make_mock_photometri:13,make_nam:8,make_plot:27,make_theory_plot_data:27,mandatori:4,manual_step_histogram:27,map:[4,6,8,11,17,23,35,37],map_correl:[4,11,32],map_nam:[3,8,19,23],map_plot:[4,11,32],mapper:[3,18,19,23],mappingproxi:4,mapsfil:8,mask:[4,11,19,34],match:37,matric:[31,37],matrix:37,max:37,max_chunk_count:37,mean:[13,24,25,31],measur:[13,21,23,31,34,35],memori:37,messag:4,meta:34,metac:[13,21,24,31,34,37],metacal_col:13,metacal_data:13,metacal_fil:13,metacal_gcr_input:[4,11,32],metacalibr:[8,13,21,24,25,31,37],metacalibration_nam:8,metadata:[4,8,11,26,32],method:[4,8,24,25,26,34,37],might:[13,21,24],minim:37,misc:37,miss:4,missing_column:8,mnra:28,mock:[13,24,25],mode:[4,8,35],model:13,modifi:37,modul:[4,11,15,31,37],moment:24,more:[4,8,37],mostli:13,move:[4,24],mpi:[4,37],mpi_reduce_larg:37,mpi_util:37,much:[22,31],multipl:[15,31,37],must:[3,4,8,19,23,24,37],n_chunk:[24,25],n_gal:19,n_row:4,n_visit:13,n_z:[24,25],name:[3,4,8,27,28,37],namespac:4,nan:34,nan_error:37,nbin:[26,31],nbin_len:37,nbin_sourc:37,ndim:30,need:[4,14,22,24,31,34,37],neighbour:37,nersc:14,next:[3,19,23],ngal:19,nice:[34,35],nobj:[24,25],node:14,nofzfil:8,nois:[13,23],noise_map:[4,11,32],nompi:14,none:[3,4,5,6,7,8,9,10,12,13,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,31,34,35,37],nonmetacalibr:37,normal:24,note:[4,24,25],now:[4,19,28],nrow:[15,31],number:[4,8,19,24,25,26,30,34,35,37],number_density_stat:[15,31],number_of_r:8,numpi:14,obj:37,object:[3,4,8,13,15,18,19,22,23,24,25,26,31,34,35,37],obs:[13,21,27],obs_data:27,observ:8,obtain:14,okai:30,onc:[19,24,25,31,37],one:[4,13,24,37],ones:[4,8],onli:[4,13,19,34],open:[4,8,24,25],open_input:4,open_output:4,oper:37,option:[4,5,8,31,37],order:[4,37],org:15,origina:8,other:[4,8,19,37],otherwis:[8,37],our:[8,22],out:[13,15,24,25,31,37],outfil:[15,31],output:[4,5,8,15,24,25,26,31,37],output_fil:[24,25],output_tag:[3,4,19,23],over:[3,19,23],overal:[15,31],overdens:19,overflow:37,overrid:[3,8,19,23],overridden:[8,37],overwrit:37,own:37,packag:[4,8],page:11,pair:[8,35,37],parallel:[4,24,25],parallelcalibratormetac:37,parallelcalibratornonmetac:37,param:37,paramet:[4,8,13,15,24,25,31,37],parent:[4,24,26,37],part:[24,25],particular:[34,35],pass:[8,22,37],patch:34,patch_cent:34,path:[4,8],path_for_fil:8,pdf:[13,24,25,26,31],peopl:28,phot_data:15,photo:[13,24,25,26,31],photo_col:13,photo_data:13,photo_fil:13,photometr:[13,19,34],photometri:[13,15,19,21,24,25],photoz:[4,11],photoz_mlz:[4,11,32],photoz_stack:[4,11,32],photozpdffil:8,php:15,pickl:[34,35],pip:14,pipe:11,pipelin:[4,8,14,15,24,31,34,35],pipeline_stag:4,pipelinestag:[4,5,6,7,8,9,10,12,13,15,16,17,19,20,21,22,23,24,25,26,28,29,31,34,35],pixel:[3,8,19,23,37],pixel_schem:[3,18,19,23],place:37,placehold:24,plain:[8,34,35],plot:[6,8,11,17,26,34],plot_gnomon:8,plot_healpix:8,png:8,pngfile:8,point:[4,5,11,24,25,30,34],point_estim:[24,25],pos_po:37,posit:[15,31,34,37],posixpath:4,possibl:8,power:35,pre:37,prefer:4,prepar:[19,24,25],prepare_mapp:[3,19,23],prepare_output:[24,25],present:[8,28],preserv:37,presum:8,print:[4,37],process:[4,24,25,37],processor:37,prod:37,produc:4,profil:4,projecta:14,projectdir:14,properti:[4,8],proven:[8,37],provid:4,psf:3,psf_diagnost:[4,11,32],purer:[13,21],put:[4,8,15,31],python:[4,8,14],pz_data:31,pzpdfmlz:[4,25],quadrilater:30,quantiti:31,queri:4,question:37,r11:13,r12:13,r21:13,r22:13,r_std:13,rais:4,random:[8,11,23,24,25,32],random_cat:[4,11,32,34],random_points_in_quadrilater:30,random_points_in_triangl:30,randomli:[23,24],randomscatalog:8,rang:[4,24,25],rank:[4,37],raw:5,read:[4,8,13,15,19,31],read_catalog_info:8,read_config:4,read_density_split:8,read_gnomon:8,read_healpix:8,read_list:8,read_map:8,read_map_info:8,read_mask:8,read_nbin:[8,34],read_proven:8,read_rot:8,read_shear_catalog_typ:37,read_zbin:8,real:[5,34],realization_index:8,receiv:37,recogn:4,record:8,redmag:[11,32],redshift:[8,24,25,26],reduc:37,reduct:37,refer:4,region:34,reject:35,remov:37,remove_undetect:13,repack:37,repres:[8,35],represent:[34,35],requir:[4,8,11,19,34],required_column:8,required_dataset:8,resolv:4,respons:[4,13,15,31,37],result:37,retriev:13,return_al:8,rho:28,right:[4,19,24,25],robert:13,robin:4,rogu:28,root:37,rotat:23,round:4,row:[4,24,28],run:[4,5,14,15,19,24,25,26,31,34,37],sacc:[5,8,37],sacc_fil:27,saccfil:8,sadli:28,same:[4,19,26,35,37],sampl:[15,31,34],save:[4,8,13,19,24,25,37],save_fil:8,save_map:19,save_plot:8,scalar:37,scipi:[14,37],scp:14,sdss3:15,search:[4,11],second:30,see:4,select:[4,11,15,19,31,37],select_len:15,selector:[11,32,37],self:[4,8,34,35],sent:37,separ:[4,19,37],seper:34,seq:37,sequenc:37,set:[13,14,15,26,31,37],setup:14,setup_output:[15,31],shape:[15,24,25,31,37],share:32,shear:[8,15,19,24,31,34,35,37],shear_catalog:35,shear_catalog_typ:37,shear_data:31,shear_photoz_stack:34,shear_po:37,shear_shear:37,shear_tomography_catalog:[31,34],shearcatalog:8,should:[4,8],similar:[24,25],simpler:31,simpli:4,simul:[5,13,21],sinc:34,singl:[4,13,37],size:[4,13,22,31,37],skip_nan:37,smooth:27,smooth_nz:27,snr:13,snr_delta:18,snr_limit:13,snr_threshold:18,softwar:[4,14],some:[14,19,24,25,31],sometim:[28,34],somewher:4,sourc:[3,4,5,6,7,8,9,10,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,34,35,37],source_0:37,source_1:37,source_bin:[18,31],source_selector:[4,11,32],space:[14,26,34,37],spam:4,spars:18,special:4,specif:[4,8,37],specifi:[4,30,35],spectra:35,spit:[24,25],split:[4,26,31],split_tasks_by_rank:4,src1:34,src2:34,stack:26,stackoverflow:37,stage:[5,8,11,13,14,15,21,22,24,26,31,34,35,37],standard:[4,8,13,34],star:[28,34],start:[13,15,21,24,25,31,37],stat:28,statist:[24,25,28],std:37,stderr:37,stdout:37,still:30,store:[8,34],str:[4,8,37],straight:37,string:[4,8,34,35,37],strip:13,structur:19,sub:4,subclass:[3,4,8,19,23,26,34],subgroup:8,submodul:11,subsequ:8,suffix:8,suit:23,sum:37,suppli:[3,8,19,23,37],supports_parallel_writ:8,systemat:[34,35],tabl:[13,31],tag:[4,8],take:34,taken:4,talli:37,target:15,target_s:13,task:4,teach:13,temporari:4,test:[4,13,21,34],text:8,textfil:[4,8],than:37,thei:[4,28],them:[4,15,28,30,31,37],theori:[27,37],theory_3x2pt:37,theory_cl:37,theory_data:27,theory_label:27,theory_sacc_fil:27,theta:34,thi:[3,4,5,8,13,14,15,19,21,22,24,25,26,31,34,35,37],thing:[24,25],think:28,third:30,those:[15,31,37],through:[4,15,19,24,26,31],throughout:13,time:[24,37],tjpcov:[11,33],to_ccl:8,todo:[4,31,35],togeth:[4,22,37],tomo_bin:[15,31],tomograph:[31,34,35],tomographi:[11,15,19,26,31],tomography_catalog:[15,35],tomographycatalog:8,tool:4,top:8,total:[4,37],tracer:37,tracer_metadata:34,track:4,tree:25,treecorr:34,triangl:30,trivial:5,truth:26,tupl:[34,35],turn:8,two:[5,11,13,28,30,31,34],twopoint:[4,11,32],twopoint_data_r:5,twopoint_data_real_raw:[5,34],twopoint_fouri:[4,11,32],twopoint_gamma_x:34,txauxiliarymap:[3,4],txbaselensselector:[4,15],txbasemap:[3,4,19,23],txblind:[4,5],txbrighterfatterplot:[4,28],txbuzzardmock:[4,13],txconvergencemap:[4,6],txconvergencemapplot:[4,6],txcosmodc2mock:[4,13],txdensitymap:[4,19],txdiagnosticplot:[4,9],txexposureinfo:[4,10],txexternallensmap:[4,19],txexternallensnoisemap:[4,23],txfouriergaussiancovari:[4,7],txgalaxystardens:4,txgalaxystarshear:4,txgammatbrightstar:[4,34],txgammatdimstar:[4,34],txgammatfieldcent:[4,34],txgammatrandom:[4,34],txingestredmag:[4,12],txingeststar:[4,21],txjackknifecent:[4,34],txlensmap:[4,19],txmainmap:[4,19],txmapcorrel:[4,16],txmapplot:[4,17],txmeanlensselector:[4,15],txmetacalgcrinput:[4,21],txnoisemap:[4,23],txnullblind:[4,5],txphotozplot:[4,26],txphotozsourcestack:[4,26],txphotozstack:[4,26],txpipe:[3,4,5,6,7,8,9,10,12,13,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,34,35,37],txpsfdiagnost:[4,28],txrandomcat:[4,29],txrandomphotozpdf:[4,24],txrealgaussiancovari:[4,7],txrowestatist:[4,28],txsimplemask:[4,20],txsourcemap:[4,19],txsourcenoisemap:[4,23],txsourceselector:[4,31],txsourcetruenumberdens:[4,26],txstandaloneauxiliarymap:4,txstardensitytest:[4,28],txstarsheartest:[4,28],txt:[4,8],txtracermetadata:[4,22],txtruenumberdens:[4,26],txtruthlensselector:[4,15],txtwopoint:[4,34],txtwopointfouri:[4,35],txtwopointlenscat:[4,34],txtwopointplot:[4,34],txtwopointplotsfouri:[4,35],type:[4,8,24,25,30,37],unblind:5,under:4,undetect:13,unfortun:13,uniformli:30,uniqu:37,unique_list:37,unit_respons:13,unknown:8,unseen:37,unspecifi:37,updat:19,usag:[4,37],use:[4,5,8,14,19,28,31,34],used:[4,8,13,15,21,24,25,31,37],useful:[8,13,21],user:[14,31],usernam:14,uses:[14,26,34],using:[4,8,14,34,37],usual:[4,28],util:11,valid:8,valu:[3,4,8,15,19,23,24,25,26,31,35,37],variabl:4,variant:[8,24,25,31,37],variou:14,vector:[5,30],veri:37,version:[4,37],vertex:30,vertic:30,view:8,wai:4,want:[24,34,35],washington:13,weak:4,weight:19,well:37,what:8,when:14,whenev:[4,37],where:[4,8,13,37],whether:37,which:[4,8,14,19,24,25,31,34,35,37],whilst:37,who:28,whole:[24,25],win:35,within:4,without:5,wlmassmap:[11,33],wonderfulli:28,work:30,would:31,wrap:37,wrapper:[4,34],write:[4,8,15,24,25,31,37],write_global_valu:[15,31],write_list:8,write_map:8,write_output:[13,24,25],write_proven:8,write_tomographi:[15,31],written:33,www:15,x_err:37,xlogscal:27,y_err:37,yaml:[4,8,37],yamlfil:8,yield:4,ymax:27,ymin:27,yml:[4,8],you:[8,14,30,37],zeljko:13,zero:37,zuntz:14},titles:["FlexZPipe","TJPCov","WLMassMap","Auxilliary maps","Base stage","Blinding","Convergencemaps","covariance stage","data_types","Diagnostics plots","exposure_info","TXPipe documentation","Ingest redmagic","Input Catalogs","TXPipe Installation","Lens selector","map_correlations","map_plots","mapping","maps","masks","metacal_gcr_input","metadata","noise_maps","photoz","photoz_mlz","photoz_stack","plotting","psf_diagnostics","random_cats","randoms","source_selector","Stages implemented in TX-Pipe:","Submodules for TXPipe","Twopoint correlations","twopoint_fourier","Ui","Utilities"],titleterms:{The:32,auxilliari:3,base:[4,32],blind:5,catalog:13,convergencemap:6,correl:34,covari:7,data_typ:8,diagnost:[9,32],document:11,exposure_info:10,extra:32,flexzpip:0,implement:32,indic:11,ingest:[12,32],input:13,instal:14,len:15,map:[3,18,19,32],map_correl:16,map_plot:17,mask:[20,32],metacal_gcr_input:21,metadata:22,noise_map:23,photoz:[24,32],photoz_mlz:25,photoz_stack:26,pipe:32,plot:[9,27,32],point:32,psf_diagnost:28,random:30,random_cat:29,redmag:12,requir:14,select:32,selector:15,source_selector:31,stage:[4,7,32],submodul:33,tabl:11,tjpcov:1,tomographi:32,two:32,twopoint:34,twopoint_fouri:35,txpipe:[11,14,33],util:[32,37],wlmassmap:2}})