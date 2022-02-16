from ceci import PipelineStage as PipelineStageBase
from .data_types import HDFFile
from textwrap import dedent
from .utils.provenance import find_module_versions, git_diff, git_current_revision
import sys
import datetime
import socket


class PipelineStage(PipelineStageBase):
    name = "Error"
    inputs = []
    outputs = []
    config_options = {}

    def run(self):
        print("Please do not execute this stage again.")

    def memory_report(self, tag=None):
        """
        Print a report about memory currently available
        on the node the process is running on.

        Parameters
        ----------
        tag: str
            Additional info to print in the output line. Default is empty.
        """
        import psutil

        t = datetime.datetime.now()

        # The different types of memory are really fiddly and don't
        # correspond to how you usually imagine. The simplest thing
        # to report here is just how much memory is left on the machine.
        mem = psutil.virtual_memory()
        avail = mem.available / 1024**3
        total = mem.total / 1024**3

        if tag is None:
            tag = ""
        else:
            tag = f" {tag}:"

        # This gives you the name of the host.  At NERSC that is the node name
        host = socket.gethostname()

        # Print messsage
        print(
            f"{t}: Process {self.rank}:{tag} Remaining memory on {host} {avail:.1f} GB / {total:.1f} GB"
        )

    def combined_iterators(self, rows, *inputs, parallel=True):
        if not len(inputs) % 3 == 0:
            raise ValueError(
                "Arguments to combined_iterators should be in threes: "
                "tag, group, value"
            )
        n = len(inputs) // 3

        iterators = []
        for i in range(n):
            tag = inputs[3 * i]
            section = inputs[3 * i + 1]
            cols = inputs[3 * i + 2]
            iterators.append(
                self.iterate_hdf(tag, section, cols, rows, parallel=parallel)
            )

        for it in zip(*iterators):
            data = {}
            for (s, e, d) in it:
                data.update(d)
            yield s, e, data

    def gather_provenance(self):
        provenance = {}

        for key, value in self.config.items():
            provenance[f"config/{key}"] = str(value)

        for name, tag_cls in self.inputs:
            try:
                f = self.open_input(name, wrapper=True)
                input_id = f.provenance["uuid"]
                f.close()
            except (OSError, IOError, KeyError):
                input_id = "UNKNOWN"

            provenance[f"input/{name}"] = input_id

        provenance["gitdiff"] = git_diff()
        provenance["githead"] = git_current_revision()

        for module, version in find_module_versions().items():
            provenance[f"versions/{module}"] = version

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
        run_parallel = kwargs.pop("parallel", False) and self.is_mpi()
        if run_parallel:
            if not output_class.supports_parallel_write:
                raise ValueError(
                    f"Tried to open file for parallel output, but not"
                    f" supported for type {output_class}.  Tag was {tag} and"
                    f" path was {path}"
                )
            kwargs["driver"] = "mpio"
            kwargs["comm"] = self.comm

            # XXX: This is also not a dependency, but it should be.
            #      Or even better would be to make it a dependency of descformats where it
            #      is actually used.
            import h5py

            if not h5py.get_config().mpi:
                print(
                    dedent(
                        """\
                Your h5py installation is not MPI-enabled.
                Options include:
                  1) Set nprocess to 1 for all stages
                  2) Upgrade h5py to use mpi.  See instructions here:
                     http://docs.h5py.org/en/latest/build.html#custom-installation
                Note: If using conda, the most straightforward way is to enable it is
                    conda install -c spectraldns h5py-parallel
                """
                    )
                )
                raise RuntimeError("h5py module is not MPI-enabled.")

        extra_provenance = self.gather_provenance()

        # Return an opened object representing the file
        obj = output_class(path, "w", extra_provenance=extra_provenance, **kwargs)

        if wrapper:
            return obj
        else:
            return obj.file
