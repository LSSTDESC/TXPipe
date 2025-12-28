The Base Stage Class
====================

All TXPipe stages inherit from this base class. It adds only a little to the parent ceci PipelineStage class.



.. autoclass:: txpipe.base_stage.PipelineStage

   Runnig and Configuration
   ---------------------------
   .. automethod:: run
   .. autoattribute:: config


   Reading and Writing Inputs and Outputs
   ---------------------------------------

   .. automethod:: open_input
   .. automethod:: open_output
   .. automethod:: iterate_fits
   .. automethod:: iterate_hdf
   .. automethod:: combined_iterators

   Finding Input and Output File Names and Types
   ---------------------------------------------
   
   You can often just use the input and output methods above rather than finding
   file paths directly, but if for any reason you need to find the actual file paths
   you can use these methods.

   .. automethod:: get_input
   .. automethod:: get_output
   .. automethod:: get_input_type
   .. automethod:: get_output_type
   .. automethod:: output_tags
   .. automethod:: input_tags
   .. automethod:: get_config_dict

   Parallelization Methods
   -----------------------

   .. automethod:: is_mpi
   .. automethod:: is_dask
   .. automethod:: split_tasks_by_rank
   .. automethod:: map_tasks_by_rank
   .. automethod:: data_ranges_by_rank

   Diagnostic Methods
   ------------------

   .. automethod:: time_stamp
   .. automethod:: memory_report

   Introspection Methods
   ---------------------

   These methods are sometimes useful for finding out about the pipeline stages that are
   available in a pipeline, or when subclassing a method.

   .. automethod:: get_stage
   .. automethod:: get_module
   .. automethod:: describe_configuration
   



