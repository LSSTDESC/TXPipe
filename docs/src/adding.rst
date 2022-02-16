Adding new pipeline stages
==========================

New stage inputs
-----------------
- look at flow chart
- think what inputs your new work needs. Do they already exist in the pipeline
- If they need to be generated first, consider adding stages 
- th


New stage outputs
-----------------

Think what format to put things in.
Use an existing format type

Parallelization
---------------





Pipeline Stage Methods
----------------------

The pipeline stages can use these methods to interact with the pipeline:

Basic tools to find the file path:

- `self.get_input(tag)`
- `self.get_output(tag)`

Get base class to find and open the file for you:

- `self.open_input(tag, **kwargs)`
- `self.open_output(tag, parallel=True, **kwargs)`


Look for a section in a yaml input file tagged "config"
and read it.  If the config_options class variable exists in the class
then it checks those options are set or uses any supplied defaults:

- `self.config['name_of_option']`

Parallelization tools - MPI attributes:

- `self.rank`
- `self.size`
- `self.comm`

(Parallel) IO tools - reading data in chunks, splitting up 
according to MPI rank:

- `self.iterate_fits(tag, hdunum, cols, chunk_rows)`
- `self.iterate_hdf(tag, group_name, cols, chunk_rows)`

