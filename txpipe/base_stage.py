from ceci import PipelineStage as PipelineStageBase
from .data_types import HDFFile
from textwrap import dedent


class PipelineStage(PipelineStageBase):
    name = "Error"
    inputs = {}
    outputs = {}

    def gather_provenance(self):
        provenance = {}

        for key, value in self.config.items():
            provenance[f"config/{key}"] = str(value)

        for name, tag_cls in self.inputs:
            f = self.open_input(name, wrapper=True)
            input_id = f.provenance['uuid']
            provenance[f"input/{name}"] = input_id
            f.close()

        return provenance

    def open_output(self, tag, wrapper=False, **kwargs):
        """
        Find and open an output file with the given tag, in write mode.

        For general files this will simply return a standard
        python file object.

        For specialized file types like FITS or HDF5 it will return
        a more specific object - see the types.py file for more info.

        This is an extended version of the parent class method which
        also saves configuration information.  Putting this here right
        now for testing.

        """
        path = self.get_output(tag)
        output_class = self.get_output_type(tag)

        # HDF files can be opened for parallel writing
        # under MPI.  This checks if:
        # - we have been told to open in parallel
        # - we are actually running under MPI
        # and adds the flags required if all these are true
        run_parallel = kwargs.pop('parallel', False) and self.is_mpi()
        if run_parallel:
            if not output_class.supports_parallel_write:
                raise ValueError(f"Tried to open file for parallel output, but not"
                    f" supported for type {output_class}.  Tag was {tag} and"
                    f" path was {path}"
                    )
            kwargs['driver'] = 'mpio'
            kwargs['comm'] = self.comm

            # XXX: This is also not a dependency, but it should be.
            #      Or even better would be to make it a dependency of descformats where it
            #      is actually used.
            import h5py
            if not h5py.get_config().mpi:
                print(dedent("""\
                Your h5py installation is not MPI-enabled.
                Options include:
                  1) Set nprocess to 1 for all stages
                  2) Upgrade h5py to use mpi.  See instructions here:
                     http://docs.h5py.org/en/latest/build.html#custom-installation
                Note: If using conda, the most straightforward way is to enable it is
                    conda install -c spectraldns h5py-parallel
                """))
                raise RuntimeError("h5py module is not MPI-enabled.")

        # Return an opened object representing the file
        obj = output_class(path, 'w', **kwargs)


        for key, value in self.gather_provenance().items():
            obj.add_provenance(key, value)



        if wrapper:
            return obj
        else:
            return obj.file
