import numpy as np
from ...base_stage import PipelineStage
from ...data_types import ShearCatalog, HDFFile, PhotozPDFFile, FiducialCosmology, TomographyCatalog


class CLClusterShearCatalogs(PipelineStage):
    name = "CLClusterShearCatalogs"
    inputs = [
        ("cluster_catalog", HDFFile),
        ("shear_catalog", HDFFile),
        ("fiducial_cosmology", FiducialCosmology),
        ("shear_tomography_catalog", TomographyCatalog),
        ("source_photoz_pdfs", PhotozPDFFile),
    ]

    outputs = [
        ("cluster_shear_catalogs", HDFFile),
    ]

    config_options = {
        "max_radius": 10.0,  # Mpc
        "delta_z": 0.1,
        "redshift_criterion": "mean",  # might also need PDF
    }

    def run(self):
        import sklearn.neighbors
        import astropy


        # load cluster catalog
        clusters = self.load_cluster_catalog()
        ncluster = len(clusters["ra"])

        # We parallelize by cluster. Each process is responsible for a different
        # chunk of the clusters
        my_clusters = self.choose_my_clusters(clusters)
        # turn the physical scale max_radius to an angular scale at the redshift of each cluster
        my_clusters["theta_max"] = self.compute_theta_max(my_clusters["z"])
        max_theta_max = my_clusters["theta_max"].max()
        my_ncluster = len(my_clusters)

        # make a Ball Tree that we can use to find out which objects
        # are nearby any clusters
        pos = np.radians([my_clusters["dec"], my_clusters["ra"]]).T
        tree = sklearn.neighbors.BallTree(pos, metric="haversine")

        indices_per_cluster = [list() for i in range(my_ncluster)]
        weights_per_cluster = [list() for i in range(my_ncluster)]

        max_radius = np.radians(self.config["max_radius"])
        delta_z = self.config["delta_z"]
        for s, e, data in self.iterate_source_catalog(my_clusters):
            # Get the location of the galaxies in this chunk of data,
            # in the form that the tree requires
            X = np.radians([data["dec"], data["ra"]]).T

            # This is a bit fiddly.  First we will get all the objects that
            # are near each cluster using the maximum search radius for any cluster.
            # Then in a moment we will cut down to the ones that are
            # within the radius for each specific cluster.
            # This is a bit roundabout but sklearn doesn't let us search with
            # a radius per cluster, only per galaxy.
            indices, distances = tree.query_radius(X, max_theta_max, return_distance=True)

            for i, cluster in enumerate(my_clusters):
                # use tree to find all the source galaxies near enought this cluster
                # by cutting down from the full list (which includes objects that are)
                # a bit too far away, see above.
                index = indices[i]
                dist_good = distances[i] < cluster["max_theta"]
                index = index[dist_good]

                # cut down the index to only include source galaxies
                # behind the cluster, with some buffer
                z_good = data["redshift"][index] > (cluster["redshift"] + delta_z)
                index = index[z_good]

                # this will in future call CLMM
                weights = self.compute_weights(data, index, cluster["redshift"])

                indices_per_cluster[i].append(index + s)
                weights_per_cluster[i].append(weights)

        # indices_per_cluster is a list of arrays, one array per cluster
        # each array is the index in the shear catalog of all the galaxies
        # near that cluster

        indices_per_cluster = [np.concatenate(index) for index in indices_per_cluster]
        weights_per_cluster = [np.concatenate(index) for index in weights_per_cluster]

        filename = self.get_output("cluster_shear_catalogs") + f".{self.rank}"
        with h5py.File(filename, "w") as f:
            # store metadata
            f.attrs["delta_z_cut"] = 0.1  # or get from configuration

            for i, c in enumerate(my_clusters):
                original_index = c["index"]
                g = f.create_group(f"cluster_{index}")
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
                if i == 0:
                    for k, v in subfile.attrs.items():
                        collated_file.attrs[k] = v

                filename = self.get_output("cluster_shear_catalogs") + f".{i}"
                with h5py.File(filename, "r") as subfile:
                    for group in subfile.keys():
                        g[group] = subfile[group]

        # Now that we are done, remove the per-rank files
        for i in range(self.size):
            filename = self.get_output("cluster_shear_catalogs") + f".{i}"
            os.remove(filename)


    def compute_theta_max(self, z):
        import pyccl
        with self.open_input("fiducial_cosmology", wrapper=True) as f:
            cosmo = f.to_ccl()
        a = 1.0 / (1 + z)
        d_a = cosmo.angular_diameter_distance(a)
        max_r = self.config["max_radius"]

        # This is in radians, which is what is expected by sklearn
        max_theta = max_r / d_a

        if self.rank == 0:
            max_theta_arcmin = np.degrees(max_theta) * 60
            print("Min search angle = ", max_theta_arcmin.min())
            print("Mean search angle = ", max_theta_arcmin.mean())
            print("Max search angle = ", max_theta_arcmin.max())

        return max_theta


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
            z = g["redshift"][:]
            rich = g["richness"][:]
            ids = g["cluster_id"][:]

        return {"ra": ra, "dec": dec, "z": z, "richness": rich, "id": ids}

    def iterate_source_catalog(self, my_clusters):
        # get response, shear, location, redshifts, weights
        return chunk
