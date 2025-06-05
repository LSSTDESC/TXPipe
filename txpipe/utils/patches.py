import numpy as np
import os
import pathlib


class PatchMaker:
    """
    Split a TreeCorr catalog into patches, hopefully faster than the native version.

    The native TreeCorr catalog patch splitter can end up using large amounts of memory
    and/or being very slow. This code re-creates some of its behaviour.

    If it proves faster, some of the ideas in this approach could be upstreamed to
    TreeCorr, though it currently only works for HDF5 not FITS.

    The main entry point for this class is the run class method.
    """

    def __init__(
        self,
        patch_filenames,
        patch_centers,
        columns,
        initial_size,
        max_size,
        my_patches=None,
    ):
        """
        Set up the patch maker object.

        Parameters
        ----------
        patch_filenames: List[str]
            A list of all the patch file names. Can include ones not made by this
            process
        patch_centers: array
            xyz coordinates of patch centers shape (npatch, 3)
        columns: Dict[str: str]
            Old names of columns mapped to new names
        initial_size: int
            Guess at initial number of objects in each patch
        max_size: int
            Maximum possible number of objects in each patch
        my_patches: List[int] or None
            Sequence of patch indices for this process
        """
        import sklearn.neighbors

        # Support parallelization through this mechanism - each
        # process is given some patches to work on.
        if my_patches is None:
            my_patches = np.arange(len(patch_centers))

        # Different
        self.columns = columns

        if initial_size <= 0:
            raise ValueError("Error in patch creation - patch initial size must be > 0")

        # This is used to work out the nearest patch center to each galaxy
        self.ball = sklearn.neighbors.BallTree(patch_centers)

        # Open and set up the output patch files.
        self.files = {
            i: self.setup_file(patch_filenames[i], initial_size, max_size)
            for i in my_patches
        }

        # The current and maximum size of each patch
        self.max_size = max_size
        self.index = {i: 0 for i in my_patches}

    def setup_file(self, filename, initial_size, max_size):
        import h5py

        f = h5py.File(filename, "w")
        for col in self.columns:
            f.create_dataset(col, (initial_size,), maxshape=(max_size,))
        return f

    def find_patch(self, data):
        # convert to spherical coordinates and find sin/cos
        lon = np.radians(data[self.columns["ra"]])
        lat = np.radians(data[self.columns["dec"]])
        sin_lat = np.sin(lat)
        sin_lon = np.sin(lon)
        cos_lat = np.cos(lat)
        cos_lon = np.cos(lon)

        # Convert to Euclidean coordinates.
        # I established the convention here just by
        # trying each combination until plotting
        # the center points matched up
        n = len(lat)
        xyz = np.empty((n, 3))
        xyz[:, 0] = cos_lon * cos_lat
        xyz[:, 1] = sin_lon * cos_lat
        xyz[:, 2] = sin_lat

        # Use the BallTree to find the nearest patch center
        # to each object
        nearest = self.ball.query(xyz, return_distance=False)

        return nearest

    def resize(self, f, new_size):
        for col in self.columns:
            f[col].resize((new_size,))

    def add_data(self, data):
        nearest = self.find_patch(data)
        n = len(nearest)

        # For each patch we are keeping track of
        for i, f in self.files.items():
            # Find objects in this chunk of data in that patch
            sel = np.where(nearest == i)[0]

            if sel.size == 0:
                continue

            # Work out the start of this new data chunk in the
            # output file, and the end
            ni = len(sel)
            s = self.index[i]
            e = s + ni

            # If this is bigger than what we said was the max size then
            # something has gone wrong
            if e > self.max_size:
                raise ValueError("Failed to make patches (max_size). Open an issue")

            # Check if we need to re-size our columns
            # because more data than we start with is in there
            col = f["ra"]
            while col.size < e:
                new_size = min(int(col.size * 1.5), self.max_size)
                if new_size == 1:
                    new_size = 2
                self.resize(f, new_size)

            # At lat we can write out this chunk of data
            # Need to convert from the name in the patch file, which
            # is always the plain ra, dec, g1, g2, w, to whatever it's
            # called in the input file
            for col, name in self.columns.items():
                d = data[name][sel]
                if (col == "ra") or (col == "dec"):
                    d = np.radians(d)
                f[col][s:e] = d

            # Update this output index
            self.index[i] = e

    def finish(self):
        empty = []
        nonempty = []
        for i, f in self.files.items():
            e = self.index[i]
            if e == 0:
                empty.append(f.filename)
            else:
                nonempty.append(f.filename)
                self.resize(f, e)
            f.close()
        # Now we have to delete any files
        # that don't have any objects in.
        for f in empty:
            os.remove(f)

        # Then we have to rename the patches
        # to remove the gaps in the numbering.
        # But we have to do that in the run class method,
        # because different processes are running different
        # patches
        return nonempty

    @staticmethod
    def patches_already_made(cat):
        # Check if the patches files have already been made, and if so
        # whether they were made with the same configuration. We do this
        # by dumping the catalog config at the end of the patch making and
        # trying to load it at the start, comparing its contents to the
        # curent cat config

        import yaml

        fn = pathlib.Path(cat.save_patch_dir, "done.yml")

        if not fn.exists():
            return False

        with fn.open() as f:
            info = yaml.safe_load(f)

        # If the file isn't properly written something
        # must have gone wrong
        if info is None:
            return False

        # Check that the full configuration is the same.
        # This might be unnecessary, but hopefully remaking
        # the patches is not too slow.
        return info == cat.config

    @staticmethod
    def patches_missing(cat):
        # Check if the number of patch files matches that in cat.npatch

        saved_patch_files = [f for f in os.listdir(cat.config['save_patch_dir']) if '.hdf5' in f]
        nsaved = len(saved_patch_files)

        return cat.npatch != nsaved

    @staticmethod
    def write_sentinel_file(cat):
        # Write the catalog config to a file to indicate that
        # the patches are fully written
        import yaml

        fn = pathlib.Path(cat.save_patch_dir, "done.yml")
        with fn.open("w") as f:
            yaml.dump(cat.config, f)

    @classmethod
    def run(cls, cat, chunk_rows, comm=None):
        """
        Create a patchmaker for a catalog and run it.

        Parameters
        ----------
        cat: Catalog
            The treecorr.Catalog object to split up. Should not have already loaded
            the catalog data, since that defeats the point
        chunk_rows: int
            Number of rows of data to read at once on each process
        comm: communicator or None
            MPI communicator for parallel runs

        Returns
        -------
        npatch: int
            The number of patches created for this file, or 0 if there are none
        """
        import h5py

        is_root = comm is None or comm.rank == 0

        # Get the columns to be used here. Do all the ones
        # that are needed in this case
        cols = {}
        for col in ["ra", "dec", "g1", "g2", "w", "r"]:
            # Check if the name of the column is set
            name = cat.config[f"{col}_col"]
            if name != "0":
                cols[col] = name

        if cat.save_patch_dir is None:
            if is_root:
                print(
                    f"Catalog {cat.file_name} does not have a patch directory set, so not making patches."
                )
            return 0, False

        patch_filenames = cat.get_patch_file_names(cat.save_patch_dir)

        npatch = len(cat.patch_centers)

        # Check for existing patches.
        if cls.patches_already_made(cat):
            contains_empty = cls.patches_missing(cat)
            if comm is None or comm.rank == 0:
                print(f"Patches already done for {cat.save_patch_dir}")
            return npatch, contains_empty

        # find the catalog full length, which we use as a maximum possible size
        with h5py.File(cat.file_name, "r") as f:
            g = f[cat.config["ext"]]
            ra_col = cat.config["ra_col"]
            max_size = g[ra_col].size


        # Do the parallelization - split up the patches in chunks
        if comm is None:
            my_patches = None
        else:
            my_patches = np.array_split(np.arange(npatch), comm.size)[comm.rank]

        initial_size = max_size // npatch
        patch_centers = cat.patch_centers

        if initial_size == 0:
            initial_size = 2

        # make the patchmaker object
        patchmaker = cls(
            patch_filenames,
            patch_centers,
            cols,
            initial_size,
            max_size,
            my_patches=my_patches,
        )


        # Read the data in the input catalog in chunks
        with h5py.File(cat.file_name, "r") as f:
            # Get the group within the file
            g = f[cat.config["ext"]]
            nchunk = int(np.ceil(max_size / chunk_rows))

            # Loop through reading chunks of data and adding them
            # to the patches
            for i in range(nchunk):
                s = i * chunk_rows
                e = s + chunk_rows
                data = {}
                for (simple_name, cat_name) in cols.items():
                    d = g[cat_name][s:e]
                    if simple_name == "g1" and cat.config["flip_g1"]:
                        d = -d
                    elif simple_name == "g2" and cat.config["flip_g2"]:
                        d = -d
                    data[cat_name] = d
                patchmaker.add_data(data)

        nonempty = patchmaker.finish()

        contains_empty = True if len(nonempty) != npatch else False

        # Collect the list of all non-empty patch files
        # that were made
        if comm is not None:
            # collect and flatten the list of non-empty patches
            nonempty = comm.gather(nonempty)
            if comm.rank == 0:
                nonempty = [f for fs in nonempty for f in fs]

        # Some patches are omitted if they are empty, but that
        # can mean we have gaps in the numbering. So this bit renames
        # the patch files so they are contiguous again, which seems
        # to be what TreeCorr does
        if (comm is None) or (comm.rank == 0):
            # build up this dict of renamings
            renames = {}
            # q is the index of the new file (renamed), and
            # it only increments when a file is actually found
            q = 0
            for i, fn in enumerate(patch_filenames[:]):
                if fn in nonempty:
                    if i != q:
                        renames[fn] = patch_filenames[q]
                    q += 1
            # Since the order is guaranteed this should never end up overwriting
            # an existing patch before it is moved.
            for old_name, new_name in renames.items():
                print(
                    f"Renaming patch {old_name} - {new_name} since some patches empty"
                )
                os.rename(old_name, new_name)

            # Touch a sentinel file to indicate that this completed
            cls.write_sentinel_file(cat)

        # make the rest of the processes wait until root has
        # finished renaming
        if comm is not None:
            comm.Barrier()

        return npatch, contains_empty
