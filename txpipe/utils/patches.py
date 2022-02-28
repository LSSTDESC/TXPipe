import numpy as np
import os


class PatchMaker:
    def __init__(
        self,
        patch_filenames,
        patch_centers,
        columns,
        initial_size,
        max_size,
        my_patches=None,
    ):
        import sklearn.neighbors

        if my_patches is None:
            my_patches = np.arange(len(patch_centers))

        self.columns = columns
        self.ball = sklearn.neighbors.BallTree(patch_centers)
        self.files = {
            i: self.setup_file(patch_filenames[i], initial_size, max_size)
            for i in my_patches
        }
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
                self.resize(f, new_size)

            # At lat we can write out this chunk of data
            # Need to convert from the name in the patch file, which
            # is always the plain ra, dec, g1, g2, w, to whatever it's
            # called in the input file
            for col, name in self.columns.items():
                f[col][s:e] = data[name][sel]

            # Update this output index
            self.index[i] = e

    def finish(self):
        for i, f in self.files.items():
            e = self.index[i]
            self.resize(f, e)
            f.close()

    @classmethod
    def run(cls, cat, chunk_rows, comm=None):
        import h5py

        is_root = comm is None or comm.rank == 0

        # Get the columns
        cols = {}
        for col in ["ra", "dec", "g1", "g2", "w"]:
            name = cat.config[f"{col}_col"]
            if name != "0":
                cols[col] = name


        if cat.save_patch_dir is None:
            if is_root:
                print(f"Catalog {cat} does not have a patch directory set.")
                print("Not making patches")
            return

        patch_filenames = cat.get_patch_file_names(cat.save_patch_dir)


        if all(os.path.exists(p) for p in patch_filenames):
            if comm is None or comm.rank == 0:
                print(f"Patches already exist for {cat.save_patch_dir}")
            return

        # find the catalog full length, which we use as a maximum possible size
        with h5py.File(cat.file_name, "r") as f:
            g = f[cat.config["ext"]]
            ra_col = cat.config["ra_col"]
            max_size = g[ra_col].size

        npatch = len(cat.patch_centers)

        # Do the parallelization
        if comm is None:
            my_patches = None
        else:
            my_patches = np.array_split(np.arange(npatch), comm.size)[comm.rank]

        initial_size = max_size // npatch
        patch_centers = cat.patch_centers
        patchmaker = cls(
            patch_filenames,
            patch_centers,
            cols,
            initial_size,
            max_size,
            my_patches=my_patches,
        )

        with h5py.File(cat.file_name, "r") as f:
            g = f[cat.config["ext"]]
            nchunk = int(np.ceil(max_size / chunk_rows))
            for i in range(nchunk):
                s = i * chunk_rows
                e = s + chunk_rows
                data = {col: g[col][s:e] for col in cols.values()}
                patchmaker.add_data(data)

        patchmaker.finish()
