import os
import collections
import timeit
import numpy as np
from ...base_stage import PipelineStage
from ...data_types import ShearCatalog, HDFFile, PhotozPDFFile, FiducialCosmology, TomographyCatalog, ShearCatalog

class ExtendingArrays:
    def __init__(self, n, size_step, dtypes):
        self.dtypes = dtypes
        self.arrays = collections.defaultdict(list)
        self.pos = np.zeros(n, dtype=int)
        self.size_step = size_step
        self.narr = len(dtypes)
        self.counts = np.zeros(n, dtype=int)

    def nbytes(self):
        n = 0
        for l in self.arrays.values():
            for group in l:
                for arr in group:
                    n += arr.nbytes
        return n

    def collect(self, index):
        if index not in self.arrays:
            return [np.array([],dtype=dt) for dt in self.dtypes]
        arrays = self.arrays[index]
        last_count = self.pos[index]
        n = len(self.arrays[index])
        output = []
        for i in range(self.narr):
            arrs = [a[i] for a in arrays]
            # chop the end off the last array
            arrs[-1] = arrs[-1][:last_count]
            output.append(np.concatenate(arrs))
        return output
                    
    def total_counts(self):
        return self.counts.sum()

    def extend(self, index):
        l = self.arrays[index]
        l.append([np.zeros(self.size_step, dtype=dt) for dt in self.dtypes])

    def append(self, index, values):
        l = self.arrays[index]
        if not l:
            self.extend(index)
        c = self.pos[index]
        if c == self.size_step:
            self.extend(index)
            c = 0
        arrs = l[-1]
        for arr, value in zip(arrs, values):
            arr[c] = value
        self.pos[index] = c + 1
        self.counts[index] += 1


class CLClusterShearCatalogs(PipelineStage):
    name = "CLClusterShearCatalogs"
    inputs = [
        ("cluster_catalog", HDFFile),
        ("shear_catalog", ShearCatalog),
        ("fiducial_cosmology", FiducialCosmology),
        ("shear_tomography_catalog", TomographyCatalog),
        ("source_photoz_pdfs", PhotozPDFFile),
    ]

    outputs = [
        ("cluster_shear_catalogs", HDFFile),
    ]

    config_options = {
        "chunk_rows": 100_000,  # rows to read at once from source cat
        "max_radius": 10.0,  # Mpc
        "delta_z": 0.1,
        "redshift_criterion": "mean",  # might also need PDF
    }

    def run(self):
        import sklearn.neighbors
        import astropy
        import h5py

        # load cluster catalog as an astropy table
        clusters = self.load_cluster_catalog()
        ncluster = len(clusters)

        # We parallelize by cluster. Each process is responsible for a different
        # chunk of the clusters
        # my_clusters = self.choose_my_clusters(clusters)
        # turn the physical scale max_radius to an angular scale at the redshift of each cluster
        cluster_theta_max = self.compute_theta_max(clusters["redshift"])
        max_theta_max = cluster_theta_max.max()
        max_theta_max_arcmin = np.degrees(max_theta_max) * 60
        if self.rank == 0:
            print(f"Max theta_max = {max_theta_max} radians = {max_theta_max_arcmin} arcmin")

        # make a Ball Tree that we can use to find out which objects
        # are nearby any clusters
        pos = np.radians([clusters["dec"], clusters["ra"]]).T
        tree = sklearn.neighbors.BallTree(pos, metric="haversine")

        per_cluster_data = ExtendingArrays(ncluster, 10_000, [int, float, float])

        
        delta_z = self.config["delta_z"]
        for s, e, data in self.iterate_source_catalog():
            print(f"Process {self.rank} processing chunk {s:,} - {e:,}")
                
            # Get the location of the galaxies in this chunk of data,
            # in the form that the tree requires
            X = np.radians([data["dec"], data["ra"]]).T

            # This is a bit fiddly.  First we will get all the objects that
            # are near each cluster using the maximum search radius for any cluster.
            # Then in a moment we will cut down to the ones that are
            # within the radius for each specific cluster.
            # This is a bit roundabout but sklearn doesn't let us search with
            # a radius per cluster, only per galaxy.
            t0 = timeit.default_timer()
            nearby_clusters, cluster_distances = tree.query_radius(X, max_theta_max, return_distance=True)
            t1 = timeit.default_timer()
            #print("Search took", t1 - t0)


            for (index, distance, zgal, gal_index) in zip(nearby_clusters, cluster_distances, data["redshift"], data["original_index"]):
                # max distance allowed to each cluster
                dist_good = distance < cluster_theta_max[index]
                
                index = index[dist_good]
                distance = distance[dist_good]

                cluster_z = clusters["redshift"][index]
                z_good = zgal > cluster_z + delta_z

                index = index[z_good]
                distance = distance[z_good]

                weights = np.ones_like(distance)
                #self.compute_weights(data, index, my_clusters["redshift"][index])

                for i, w, d in zip(index, weights, distance):
                    per_cluster_data.append(i, [gal_index, w, d])

            t2 = timeit.default_timer()
            import gc
            gc.collect()

        print(f"Process {self.rank} done reading")

        # The overall number of indices for every pair
        overall_count = self.comm.reduce(per_cluster_data.total_counts())

        if self.rank == 0:
            outfile = self.open_output("cluster_shear_catalogs")
            catalog_group = outfile.create_group("catalog")
            catalog_group.create_dataset("cluster_sample_start", shape=(ncluster,), dtype=np.int32)
            catalog_group.create_dataset("cluster_sample_count", shape=(ncluster,), dtype=np.int32)
            catalog_group.create_dataset("cluster_id", shape=(ncluster,), dtype=np.int64)
            catalog_group.create_dataset("cluster_theta_max_arcmin", shape=(ncluster,), dtype=np.float64)
            index_group = outfile.create_group("index")
            index_group.create_dataset("cluster_index", shape=(overall_count), dtype=np.int64)
            index_group.create_dataset("source_index", shape=(overall_count), dtype=np.int64)
            index_group.create_dataset("weight", shape=(overall_count), dtype=np.float64)
            index_group.create_dataset("distance_radians", shape=(overall_count), dtype=np.float64)


        start = 0
        for i, c in enumerate(clusters):
            if (self.rank == 0) and (i%100 == 0):
                print(f"Collecting data for cluster {i}")

            indices, weights, distances = per_cluster_data.collect(i)

            if self.comm is not None:
                indices, weights, distances = self.collect(indices, weights, distances)

            if self.rank != 0:
                continue

            t1 = timeit.default_timer()

            n = indices.size
            catalog_group["cluster_sample_start"][i] = start
            catalog_group["cluster_sample_count"][i] = n
            catalog_group["cluster_id"][i] = c["id"]
            catalog_group["cluster_theta_max_arcmin"][i] = max_theta_max_arcmin[i]

            index_group["cluster_index"][start:start + n] = i
            index_group["source_index"][start:start + n] = indices
            index_group["weight"][start:start + n] = weights
            index_group["distance_radians"][start:start + n] = distances

            t2 = timeit.default_timer()
            print("Time for write = ", t2 - t1)

    def collect(self, indices, weights, distances):
        # total number of background objects for t
        t1 = timeit.default_timer()
        counts = np.array(self.comm.allgather(indices.size))
        # This collects together all the results from different processes for this cluster
        if self.rank == 0:
            total = counts.sum()
            all_indices = np.empty(total, dtype=indices.dtype)
            all_weights = np.empty(total, dtype=weights.dtype)
            all_distances = np.empty(total, dtype=distances.dtype)
            self.comm.Gatherv(sendbuf=distances, recvbuf=(all_distances, counts))
            self.comm.Gatherv(sendbuf=weights, recvbuf=(all_weights, counts))
            self.comm.Gatherv(sendbuf=indices, recvbuf=(all_indices, counts))
            indices = all_indices
            weights = all_weights
            distances = all_distances
        else:
            self.comm.Gatherv(sendbuf=distances, recvbuf=(None, counts))
            self.comm.Gatherv(sendbuf=weights, recvbuf=(None, counts))
            self.comm.Gatherv(sendbuf=indices, recvbuf=(None, counts))
        t2 = timeit.default_timer()
        if self.rank == 0:
            print("Time for gather = ", t2 - t1)
        return indices, weights, distances


    def compute_theta_max(self, z):
        import pyccl
        with self.open_input("fiducial_cosmology", wrapper=True) as f:
            cosmo = f.to_ccl()
        a = 1.0 / (1 + z)
        d_a = cosmo.angular_diameter_distance(a)
        max_r = self.config["max_radius"]

        # This is in radians, which is what is expected by sklearn
        theta_max = max_r / d_a

        if self.rank == 0:
            theta_max_arcmin = np.degrees(theta_max) * 60
            print("Min search angle = ", theta_max_arcmin.min(), "arcmin")
            print("Mean search angle = ", theta_max_arcmin.mean(), "arcmin")
            print("Max search angle = ", theta_max_arcmin.max(), "arcmin")

        return theta_max


    def compute_weights(self, data, index, z_cluster):
        return np.ones_like(index)
        # import clmm
        # z_gals = data["redshift"][index]
        # do some calculation with z_gals and z_cluster from clmm

    def choose_my_clusters(self, clusters):
        from astropy.table import Table
        n = len(clusters['ra'])
        my_indices = np.array_split(np.arange(n), self.size)[self.rank]
        my_clusters = {
            name: col[my_indices]
            for name, col in clusters.items()
        }
        my_clusters["index"] = my_indices
        return Table(my_clusters)


    def load_cluster_catalog(self):
        from astropy.table import Table
        with self.open_input("cluster_catalog") as f:
            g = f["clusters/"]
            ra = g["ra"][:]
            dec = g["dec"][:]
            redshift = g["redshift"][:]
            rich = g["richness"][:]
            ids = g["cluster_id"][:]

        return Table({"ra": ra, "dec": dec, "redshift": redshift, "richness": rich, "id": ids})

    def iterate_source_catalog(self):
        rows = self.config["chunk_rows"]

        # where and what to read from the shear catalog
        with self.open_input("shear_catalog", wrapper=True) as f:
            shear_group = f.get_primary_catalog_group()
        shear_cols = ["ra", "dec"]

        # where and what to read rom the PZ catalog. This is in a QP
        # format where the mode and mean are stored in a file called
        # "ancil". The columns are called zmode and zmean.
        # TODO: Support "pdf" option here and read from /data/yvals
        pz_group = "ancil"
        pz_col = "z" + self.config["redshift_criterion"]
        pz_cols = [pz_col]

        # where and what to read from the tomography catalog.
        # We just want the values from the source bin. We will use
        # any selected object, so we just ask for bin >= 0.
        # (bin = -1 means non-selected)
        tomo_group = "tomography"
        tomo_cols = ["source_bin"]

        for s, e, data in self.combined_iterators(
                rows,
                "shear_catalog",
                shear_group,
                shear_cols,
                "source_photoz_pdfs",
                pz_group,
                pz_cols,
                "shear_tomography_catalog",
                tomo_group,
                tomo_cols,
                parallel=True
        ):

            data["original_index"] = np.arange(s, e, dtype=int)
            # cut down to objects in the WL sample
            wl_sample = data["source_bin"] >= 0
            data = {name: col[wl_sample] for name, col in data.items()}
            # rename zmean or zmode to redshift so it is simpler above
            data["redshift"] = data.pop(pz_col)
            yield s, e, data


