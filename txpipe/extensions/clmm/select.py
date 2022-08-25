import os
import collections
import timeit
import numpy as np
from ...base_stage import PipelineStage
from ...data_types import ShearCatalog, HDFFile, PhotozPDFFile, FiducialCosmology, TomographyCatalog, ShearCatalog


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

        # load cluster catalog
        clusters = self.load_cluster_catalog()
        ncluster = len(clusters["ra"])

        # We parallelize by cluster. Each process is responsible for a different
        # chunk of the clusters
        my_clusters = self.choose_my_clusters(clusters)
        # turn the physical scale max_radius to an angular scale at the redshift of each cluster
        my_clusters["theta_max"] = self.compute_theta_max(my_clusters["redshift"])
        max_theta_max = my_clusters["theta_max"].max()
        my_ncluster = len(my_clusters)
        print(f"Process {self.rank} dealing with {my_ncluster:,} clusters")

        # make a Ball Tree that we can use to find out which objects
        # are nearby any clusters
        pos = np.radians([my_clusters["dec"], my_clusters["ra"]]).T
        tree = sklearn.neighbors.BallTree(pos, metric="haversine")

        indices_per_cluster = [list() for i in range(my_ncluster)]
        weights_per_cluster = [list() for i in range(my_ncluster)]

        delta_z = self.config["delta_z"]
        for s, e, data in self.iterate_source_catalog(my_clusters):
            if self.rank == 0:
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
            if self.rank == 0:
                print("Search took", t1 - t0)

            # nearby_clusters is the list of the clusters near to each galaxy
            # We want to flip this to get the galaxies near to each cluster
            # Or we could just iterate what we have
            counts = []
            for g, (index, distance, zgal) in enumerate(zip(nearby_clusters, cluster_distances, data["redshift"])):
                counts.append(index.size)
                # max distance allowed to each cluster
                cluster_theta_max = my_clusters["theta_max"][index]
                dist_good = distance < cluster_theta_max
                index = index[dist_good]
                distance = distance[dist_good]

                cluster_z = my_clusters["redshift"][index]
                z_good = zgal > cluster_z + delta_z

                index = index[z_good]
                distance = distance[z_good]

                weights = np.ones_like(distance)
                #self.compute_weights(data, index, my_clusters["redshift"][index])

                for i, w in zip(index, weights):
                    indices_per_cluster[i].append(g + s)
                    weights_per_cluster[i].append(w)
            print(np.mean(counts), np.max(counts))

            t2 = timeit.default_timer()
            if self.rank == 0:
                print("Invert took", t2 - t1)


#        breakpoint()
        
        indices_per_cluster = [np.array(index)  for index in indices_per_cluster]
        weights_per_cluster = [np.array(index)  for index in weights_per_cluster]

        filename = self.get_output("cluster_shear_catalogs") + f".{self.rank}"
        with h5py.File(filename, "w") as f:
            print(f"Process {self.rank} writing file {filename}")
            # store metadata
            f.attrs["delta_z_cut"] = 0.1  # or get from configuration

            for i, c in enumerate(my_clusters):
                original_index = c["index"]
                g = f.create_group(f"cluster_{original_index}")
                g.create_dataset("source_sample_index", data=indices_per_cluster[i])
                g.create_dataset("source_weight", data=weights_per_cluster[i])
                g.attrs["source_count"] = weights_per_cluster[i].size
                g.attrs["redshift"] = c["redshift"]
                g.attrs["richness"] = c["richness"]
                g.attrs["id"] = c["id"]
                g.attrs["ra"] = c["ra"]
                g.attrs["dec"] = c["dec"]

        if self.comm is not None:
            self.comm.Barrier()

        # open a master file containing everything
        if self.rank != 0:
            return

        with self.open_output("cluster_shear_catalogs") as collated_file:
            g = collated_file.create_group("catalogs")
            for i in range(self.size):
                filename = self.get_output("cluster_shear_catalogs") + f".{i}"
                print(f"Process 0 reading file {filename} and collating to main file")
                with h5py.File(filename, "r") as subfile:
                    if i == 0:
                        for k, v in subfile.attrs.items():
                            collated_file.attrs[k] = v

                    for group in subfile.keys():
                        subfile.copy(group, g)
#                        g[group] = subfile[group]

        # Now that we are done, remove the per-rank files
        for i in range(self.size):
            filename = self.get_output("cluster_shear_catalogs") + f".{i}"
#            os.remove(filename)


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
        with self.open_input("cluster_catalog") as f:
            g = f["clusters/"]
            ra = g["ra"][:]
            dec = g["dec"][:]
            redshift = g["redshift"][:]
            rich = g["richness"][:]
            ids = g["cluster_id"][:]

        return {"ra": ra, "dec": dec, "redshift": redshift, "richness": rich, "id": ids}

    def iterate_source_catalog(self, my_clusters):
        # for now my_clusters is unused, but we could use it later
        # to make the search more efficient.
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

        # We set parallel=False here because all the processes
        # are reading all the catalog, beacuse the parallelization
        # is done by cluster, not by object. Though we could look
        # again at that.
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
                parallel=False
        ):

            # cut down to objects in the WL sample
            wl_sample = data["source_bin"] >= 0
            data = {name: col[wl_sample] for name, col in data.items()}
            # rename zmean or zmode to redshift so it is simpler above
            data["redshift"] = data.pop(pz_col)
            yield s, e, data


