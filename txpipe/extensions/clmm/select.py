import os
import gc
import numpy as np
from ...base_stage import PipelineStage
from ...data_types import ShearCatalog, HDFFile, PhotozPDFFile, FiducialCosmology, TomographyCatalog, ShearCatalog
from .utils import ExtendingArrays
from ...utils.calibrators import Calibrator

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
        "subtract_mean_shear": True,
    }

    def run(self):
        import sklearn.neighbors
        import astropy
        import h5py
        import clmm

        # load cluster catalog as an astropy table
        clusters = self.load_cluster_catalog()
        ncluster = len(clusters)

        # turn the physical scale max_radius to an angular scale at the redshift of each cluster
        cluster_theta_max = self.compute_theta_max(clusters["redshift"])

        # For the neighbour search, sklearn doesn't let us have
        # a different distance per cluster. So we use the maximum
        # distance over all the clusters and then filter down later.
        max_theta_max = cluster_theta_max.max()
        max_theta_max_arcmin = np.degrees(max_theta_max) * 60
        if self.rank == 0:
            print(f"Max theta_max = {max_theta_max} radians = {max_theta_max_arcmin} arcmin")

        # make a Ball Tree that we can use to find out which objects
        # are nearby any clusters
        pos = np.radians([clusters["dec"], clusters["ra"]]).T
        tree = sklearn.neighbors.BallTree(pos, metric="haversine")

        # We use this object (see utils.py) to store the neighbours for
        # each cluster in an extensible but memory-efficient way
        per_cluster_data = ExtendingArrays(ncluster, 10_000, [int, float, float])

        with self.open_input("fiducial_cosmology", wrapper=True) as f:
            ccl_cosmo = f.to_ccl()
            clmm_cosmo = clmm.Cosmology()._init_from_cosmo(ccl_cosmo)

        # Buffer in redshift behind each object        
        delta_z = self.config["delta_z"]

        # Loop through the source catalog.
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
            nearby_clusters, cluster_distances = tree.query_radius(X, max_theta_max, return_distance=True)

            # Now we loop through each galaxy, and take all the clusters nearby it.
            # We have to int
            for (index, distance, zgal, gal_index) in zip(nearby_clusters, cluster_distances, data["redshift"], data["original_index"]):
                # Cut down to clusters close enough to this galaxy
                dist_good = distance < cluster_theta_max[index]
                index = index[dist_good]
                distance = distance[dist_good]

                # Cut down to clusters that are in front of this galaxy
                z_good = zgal > clusters["redshift"][index] + delta_z
                index = index[z_good]
                distance = distance[z_good]

                # Placeholder: we should replace this with a call to CLMM
                weights = self.compute_weights(clmm_cosmo, data, index, my_clusters["redshift"][index])

                # Now loop through all the nearby clusters and save the fact that this
                # galaxy is behind them
                for i, w, d in zip(index, weights, distance):
                    per_cluster_data.append(i, [gal_index, w, d])

            gc.collect()

        print(f"Process {self.rank} done reading")

        # The overall number of cluster-galaxy pairs
        overall_count = int(self.comm.reduce(per_cluster_data.total_counts()))
        
        # The root process saves all the data. First it setps up the output
        # file here.
        if self.rank == 0:
            print("Overall pair count = ", overall_count)
            outfile = self.open_output("cluster_shear_catalogs")
            # Create space for the catalog
            catalog_group = outfile.create_group("catalog")
            catalog_group.create_dataset("cluster_sample_start", shape=(ncluster,), dtype=np.int32)
            catalog_group.create_dataset("cluster_sample_count", shape=(ncluster,), dtype=np.int32)
            catalog_group.create_dataset("cluster_id", shape=(ncluster,), dtype=np.int64)
            catalog_group.create_dataset("cluster_theta_max_arcmin", shape=(ncluster,), dtype=np.float64)
            # and for the index into that catalog
            index_group = outfile.create_group("index")
            index_group.create_dataset("cluster_index", shape=(overall_count,), dtype=np.int64)
            index_group.create_dataset("source_index", shape=(overall_count,), dtype=np.int64)
            index_group.create_dataset("weight", shape=(overall_count,), dtype=np.float64)
            index_group.create_dataset("distance_arcmin", shape=(overall_count,), dtype=np.float64)


        # Now we loop through each cluster and collect all the galaxies
        # behind it from all the different processes.
        start = 0
        for i, c in enumerate(clusters):
            if (self.rank == 0) and (i%100 == 0):
                print(f"Collecting data for cluster {i}")

            # Each process collects all the galaxies for this cluster
            indices, weights, distances = per_cluster_data.collect(i)

            # If we are running in parallel then collect together the values from
            # all the processes
            if self.comm is not None:
                indices, weights, distances = self.collect(indices, weights, distances)

            # Only the root process does the writing, so the others just
            # go to the next set of clusters.
            if self.rank != 0:
                continue

            # Sort so the indices are montonic increasing
            srt = indices.argsort()
            indices = indices[srt]
            weights = weights[srt]
            distances = distances[srt]
            
            # And finally write out all the data from the root process.
            n = indices.size
            catalog_group["cluster_sample_start"][i] = start
            catalog_group["cluster_sample_count"][i] = n
            catalog_group["cluster_id"][i] = c["id"]
            catalog_group["cluster_theta_max_arcmin"][i] = np.degrees(cluster_theta_max[i]) * 60

            index_group["cluster_index"][start:start + n] = i
            index_group["source_index"][start:start + n] = indices
            index_group["weight"][start:start + n] = weights
            index_group["distance_arcmin"][start:start + n] = np.degrees(distances) * 60

            start += n

        if self.rank == 0:
            outfile.close()

    def collect(self, indices, weights, distances):
        # total number of background objects for t

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

        return indices, weights, distances


    def compute_theta_max(self, z):
        """
        Convert the maximum radius into a maximum angle at the redshift
        of each cluster.
        """
        import pyccl

        # Load a fiducial cosmology and the the angular diameter distance
        # to each redshift in it
        with self.open_input("fiducial_cosmology", wrapper=True) as f:
            cosmo = f.to_ccl()
        a = 1.0 / (1 + z)
        d_a = cosmo.angular_diameter_distance(a)


        # Use this to convert the max radius in megaparsec to an angle.
        # The d_a should also be in Mpc.
        # This is in radians, which is what is expected by sklearn
        theta_max = self.config["max_radius"] / d_a

        if self.rank == 0:
            theta_max_arcmin = np.degrees(theta_max) * 60
            print("Min search angle = ", theta_max_arcmin.min(), "arcmin")
            print("Mean search angle = ", theta_max_arcmin.mean(), "arcmin")
            print("Max search angle = ", theta_max_arcmin.max(), "arcmin")

        return theta_max


    def compute_weights(self, clmm_cosmo, data, index, z_cluster):
        import clmm

        # Depending on whether we are using the PDF or not, choose
        # some keywords to give to compute_galaxy_weights
        if self.config["redshift_criterion"] == "pdf":
            # We need the z and PDF(z) arrays in this case
            pdf_z = data["pdf_z"]
            pdf_pz = data["pdf_pz"]
            redshift_keywords = {
                "pzpdf":pdf_pz,
                "pzbins":pdf_z,
                "use_pdz":True
            }
        else:
            # point-estimated redshift
            z_source = data["redshift"][index]
            redshift_keywords = {
                "z_source":z_source,
                "use_pdz":False
            }

        weight = clmm.dataops.compute_galaxy_weights(
            z_cluster,
            clmm_cosmo,
            is_deltasigma=True,
            use_shape_noise=True,
            use_shape_error=False,
            validate_input=True,
            shape_component1=data["g1"],
            shape_component2=data["g2"],
            **redshift_keywords
        )

        return weight



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
        """
        Iterate through the shear catalog, loading the locations
        of them and whether they are assigned to any tomographic bin.
        """
        rows = self.config["chunk_rows"]

        # where and what to read from the shear catalog
        with self.open_input("shear_catalog", wrapper=True) as f:
            shear_group = "shear"
            shear_cols, rename = f.get_primary_catalog_names()

        # load the shear calibration information
        # for the moment, load the average overall shear calibrator,
        # that applies to the collective 2D bin. This won't be high
        # accuracy, but I'm not clear right now exactly what the right
        # model is yet. Here this calibrator is just used for the
        # weight information calculation, which needs g1 and g2 estimates.
        # The same calibration should be used later for the actual shears.
        if self.rank == 0:
            print("Using single 2D shear calibration!")
        _, shear_cal = Calibrator.load(self.get_input("shear_tomography_catalog"))
        subtract_mean = self.config["subtract_mean_shear"]


        # where and what to read rom the PZ catalog. This is in a QP
        # format where the mode and mean are stored in a file called
        # "ancil". The columns are called zmode and zmean.
        # TODO: Support "pdf" option here and read from /data/yvals
        redshift_criterion = self.config["redshift_criterion"]

        if redshift_criterion == "pdf":
            # This is not actually a single column but an array
            pz_group = "data"
            pz_cols = ["yvals"]

            # we will also need the z axis values in this case
            with self.open_input("source_photoz_pdfs") as f:
                # this data seems to be 1D in my QP file.
                # but that's the kind of thing they might change.
                pdf_z = np.squeeze(f["/meta/xvals"][0])
        else:
            pz_group = "ancil"
            pz_col = "z" + self.config["redshift_criterion"]
            pz_cols = [pz_col]
            rename[pz_col] = "redshift"


        # where and what to read from the tomography catalog.
        # We just want the values from the source bin. We will use
        # any selected object, so we just ask for bin >= 0.
        # (bin = -1 means non-selected)
        tomo_group = "tomography"
        tomo_cols = ["source_bin"]

        # Loop through all these input files simultaneously.
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
            # For each chunk of data we also want to store the original
            # index.  This comes in useful later because we will cut
            # down here to just objects in the WL sample.
            data["original_index"] = np.arange(s, e, dtype=int)

            # cut down to objects in the WL sample
            wl_sample = data["source_bin"] >= 0
            data = {name: col[wl_sample] for name, col in data.items()}

            # Apply the shear calibration to this sample.
            # Optionally subtract the mean (of the whole WL sample,
            # not the local mean)
            data["g1"], data["g2"] = shear_cal.apply(data["g1"],
                                                     data["g2"],
                                                     subtract_mean=subtract_mean
            )

            # give the shear columns a unified name, whether
            # they are metacal, metadetect, etc., also rename
            # zmean or zmode to "redshift"
            for old, new in renames.items():
                data[new] = data.pop(old)

            # If we are in PDF mode then we need this extra info
            if redshift_criterion == "pdf":
                data["pdf_z"] = pdf_z
                # also rename this for clarity
                data["pdf_pz"] = data.pop("yvals")

            # Give this chunk of data to the main run function
            yield s, e, data
