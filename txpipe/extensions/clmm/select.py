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
        "redshift_criterion": "mean",  # might also need PDF
    }

    def run(self):
        import sklearn.neighbors

        # load cluster catalog
        clusters = self.load_cluster_catalog()
        ncluster = len(clusters["ra"])

        # We parallelize by cluster. Each process is responsible for a different
        # chunk of the clusters
        my_clusters = self.choose_my_clusters(clusters)
        z_clusters = clusters["z"]
        # turn the physical scale max_radius to an angular scale at the redshift of each cluster
        theta_clusters = self.compute_theta_max(z_clusters)
        

        # make a Ball Tree that we can use to find out which objects
        # are nearby any clusters
        pos = np.radians([my_clusters["dec"], my_clusters["ra"]]).T
        tree = sklearn.neighbors.BallTree(pos, metric="haversine")

        indices_per_clusters = [list() for c in my_clusters]
        weights_per_clusters = [list() for c in my_clusters]

        max_radius = np.radians(self.config["max_radius"])
        for s, e, data in self.iterate_source_catalog(clusters, my_clusters):
            # Get the location of the galaxies in this chunk of data,
            # in the form that the tree requires
            X = np.radians([data["dec"], data["ra"]]).T

            # use tree to find points near each cluster
            indices = tree.query_radius(X, theta_clusters)


            for i, (c, index) in enumerate(zip(my_clusters, indices)):
                # c is a cluster index
                # index is all the source galaxies in this source chunk near on the sky
                index = self.redshift_cut(data, index, z_clusters[c])

                # this may call CLMM
                weights = self.compute_weights(index, data, z_clusters[c]) # ???

                sources_per_cluster[i].append(index + s)
                weights_per_cluster[i].append(index + s)

        # indices_per_cluster is a list of arrays, one array per cluster
        # each array is the index in the shear catalog of all the galaxies
        # near that cluster

        indices_per_cluster = [np.concatenate(index) for index in indices_per_cluster]
        weights_per_cluster = [np.concatenate(index) for index in weights_per_cluster]
        counts_per_cluster = [index.size for index in index_per_cluster]

        filename = self.get_output("cluster_shear_catalogs") + f".{self.rank}"
        with h5py.File(filename, "w") as f:
            # store metadata
            f.attrs["delta_z_cut"] = 0.1  # or get from configuration

            for i, c in enumerate(my_clusters):
                g = f.create_group(f"cluster_{c}")
                g.create_dataset("index", data=indices_per_cluster[i])
                g.create_dataset("weight", data=weights_per_cluster[i])
                g.attrs["count"] = count
                g.attrs["z_cluster"] = z_clusters[c]
                g.attrs["richness"] = clusters["richness"][c]
                g.attrs["id"] = clusters["id"][c]
                g.attrs["ra"] = clusters["ra"][c]
                g.attrs["dec"] = clusters["dec"][c]

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
                with h5py.File(filename, "w") as subfile:
                    for group in subfile.keys():
                        g[group] = subfile[group]


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






    def compute_weights(self):
        import clmm
        ...

    def choose_my_clusters(self, clusters):
        n = len(clusters['ra'])
        my_ = np.array_split(np.arange(n), self.size)[self.rank]
        m
        return my_clusters


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
