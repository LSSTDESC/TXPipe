import gc
import numpy as np
from ...base_stage import PipelineStage
from ...data_types import ShearCatalog, HDFFile, PhotozPDFFile, FiducialCosmology, TomographyCatalog, ShearCatalog, CatalogSelection, BinAssignments
from .utils import iterate_source_catalog
import yaml
import warnings

class CLClusterSourceSelect(PipelineStage):
    """Find the background sources of each cluster.

    The sources are the objects of the shear catalog that pass the cuts in shear_selection
    and, with require_tomo_bin, are in a tomographic bin of shear_tomography_catalog, the
    sample its shear calibration is measured on. For each cluster the stage keeps those
    within max_radius in Mpc (or max_angle in arcmin) that are behind the cluster according
    to redshift_cut_criterion, and writes the row of each source in the shear catalog and
    its distance to the cluster. source_photoz_pdfs can be "none" if the redshift cut does
    not use photo-z.

    With variant_pairs, it also finds the pairs of each sheared version of the catalog
    (1p, 1m, 2p, 2m, or 1p, 1m for scalar metadetect), with the positions, cuts, tomographic bin (from shear_bin_assignments)
    and redshift of that version, so that CLClusterShearCalibrator has every galaxy of each
    version for the selection response. The profiles only use the pairs of the main catalog.
    shear_bin_assignments can be "none" unless variant_pairs needs the bins of the
    sheared versions (require_tomo_bin or tomo_bins).
    """
    name = "CLClusterSourceSelect"
    inputs = [
        ("cluster_catalog", HDFFile),
        ("shear_catalog", ShearCatalog),
        ("fiducial_cosmology", FiducialCosmology),
        ("shear_tomography_catalog", TomographyCatalog),
        ("shear_selection", CatalogSelection),
        ("source_photoz_pdfs", PhotozPDFFile),
        ("shear_bin_assignments", BinAssignments),
    ]

    outputs = [
        ("cluster_shear_selection", HDFFile),
    ]

    config_options = {
        "chunk_rows": 100_000,  # rows to read at once from source cat
        "max_radius": 10.0,  # Mpc
        "delta_z": 0.1,
        "redshift_cut_criterion": "zmode",  # pdf / mean / true / median / tomo_bins / catalog_redshift
        "redshift_cut_criterion_pdf_fraction": 0.9,  # pdf / mean / true / median
        "use_radius": True,
        "max_angle": 30, # arcmin
        "use_true_shear": False,
        "require_tomo_bin": True,  # sources must be in a tomographic bin (bin >= 0)
        "catalog_redshift_column": "mean_z",  # redshift column of the shear catalog, for the catalog_redshift criterion
        "variant_pairs": False,  # also find the pairs of each sheared version, for CLClusterShearCalibrator
    }

    def run(self):
        # The pairs of the main catalog and, with variant_pairs, of each sheared version
        versions = [None]
        if self.config["variant_pairs"]:
            versions += self.sheared_versions()
        outfile = self.open_output("cluster_shear_selection") if self.rank == 0 else None
        for version in versions:
            self.find_pairs(outfile, version)
        if self.rank == 0:
            outfile.close()

    def sheared_versions(self):
        """
        The sheared versions of the shear catalog, named by its classifier variants: the
        prefix and suffix of their columns and their bin column in shear_bin_assignments,
        e.g. 1p/ra, 1p/selection and 1p/bin for metadetect, or ra, selection_1p and bin_1p
        for AnaCal, where the versions are columns of the same rows.
        """
        with self.open_input("shear_catalog", wrapper=True) as f:
            names = list(zip(f.get_classifier_prefixes(), f.get_classifier_suffixes(), f.get_classifier_output_columns()))
            main_prefix = f.get_primary_catalog_prefix()
            catalog_type = f.catalog_type
        versions = [{"name": (p + s).strip("/_"), "prefix": p, "suffix": s, "bin": b} for p, s, b in names[1:]]
        if not versions:
            warnings.warn(f"A {catalog_type} catalog has no sheared versions: only the pairs of the main catalog are found")
            return versions

        criterion = self.config["redshift_cut_criterion"]
        # The versions of metadetect are separate catalogs, with other rows than the photo-z file
        if any(v["prefix"] != main_prefix for v in versions) and criterion != "tomo_bins":
            raise ValueError(f"The sheared versions of a {catalog_type} catalog are separate catalogs: "
                             f"with variant_pairs their redshift cut must be tomo_bins, not {criterion}")
        if (self.config["require_tomo_bin"] or criterion == "tomo_bins") and self.get_input("shear_bin_assignments").lower() == "none":
            raise ValueError("variant_pairs needs the shear_bin_assignments input, for the tomographic bins of the sheared versions")
        if criterion in ["zmode", "zmean", "zmedian", "pdf"]:
            warnings.warn(f"The sheared versions use the {criterion} photo-z of the main catalog, so the response "
                          "of the redshift cut is not in the selection response")
        return versions

    def find_pairs(self, outfile, version=None):
        """
        Find the pairs of the main catalog (version None) or of a sheared version, and write
        them in the catalog and index groups (catalog_<version> and index_<version>)
        of outfile, which is open in the root process.
        """
        import sklearn.neighbors
        import astropy
        import h5py

        # load cluster catalog as an astropy table
        clusters = self.load_cluster_catalog()
        ncluster = len(clusters)
        if self.config["use_radius"]:
        # turn the physical scale max_radius to an angular scale at the redshift of each cluster
            cluster_theta_max = self.compute_theta_max(clusters["redshift"])

        # For the neighbour search, sklearn doesn't let us have
        # a different distance per cluster. So we use the maximum
        # distance over all the clusters and then filter down later.
            max_theta_max = cluster_theta_max.max()
            max_theta_max_arcmin = np.degrees(max_theta_max) * 60
        else:
            max_theta_max_arcmin = self.config["max_angle"]
            max_theta_max = np.deg2rad(max_theta_max_arcmin / 60)
            cluster_theta_max = np.repeat(max_theta_max, ncluster)
        if self.rank == 0:
            print(f"Max theta_max = {max_theta_max} radians = {max_theta_max_arcmin} arcmin")

        # make a Ball Tree that we can use to find out which objects
        # are nearby any clusters
        pos = np.radians([clusters["dec"], clusters["ra"]]).T
        tree = sklearn.neighbors.BallTree(pos, metric="haversine")

        # Store the neighbours for each cluster
        per_cluster_data = [list() for i in range(ncluster)]

        # Buffer in redshift behind each object        
        delta_z = self.config["delta_z"]

        # Loop through the source catalog.
        require_tomo_bin = self.config["require_tomo_bin"]
        suffix = "" if version is None else f"_{version['name']}"
        if self.rank == 0 and version is not None:
            print(f"Pairs of the {version['name']} version of the catalog")
        for s, e, data in iterate_source_catalog(self, self.config["redshift_cut_criterion"], read_selection=True, read_tomo_bin=require_tomo_bin, version=version):
            print(f"Process {self.rank} processing chunk {s:,} - {e:,}")

            # The sources are the objects that pass the selection cuts
            # and, with require_tomo_bin, are in a tomographic bin
            wl_sample = data["selection"]
            if require_tomo_bin:
                wl_sample = wl_sample & (data["bin"] >= 0)
            wl_index = np.flatnonzero(wl_sample)
            if wl_index.size == 0:
                continue

            nearby_gals, nearby_gal_dists = self.find_galaxies_near_each_cluster(
                data, clusters, tree, max_theta_max, wl_index, cluster_theta_max if self.config["use_radius"] else None
            )


            # Now we have all the galaxies near each cluster,
            # within the radius of that cluster
            for cluster_index, gal_index in nearby_gals.items():
                # gal_index is an index into this chunk of galaxy data
                # pointing to all the galaxies near this cluster
                gal_index = np.array(gal_index)
                distance = np.array(nearby_gal_dists[cluster_index])

                if gal_index.size == 0:
                    continue

                cluster_z = clusters[cluster_index]["redshift"]
                # # Cut down to clusters that are in front of this galaxy
                z_good = self.select_on_redshift(data, gal_index, cluster_z, delta_z)
                
                gal_index = gal_index[z_good]
                distance = distance[z_good]

                if gal_index.size == 0:
                    continue

                per_cluster_data[cluster_index].append((data["original_index"][gal_index], distance))

            gc.collect()

        print(f"Process {self.rank} done reading")

        # The overall number of cluster-galaxy pairs
        # Each item in per_cluster_data is a list of arrays.
        # Flattening that list of arrays into a single array
        # gives the entire galaxy sample for that cluster
        my_counts = np.array([sum(len(x[0]) for x in d) for d in per_cluster_data])
        # This is now the number of cluster-galaxy pairs found by this
        # process. We also want the sum from all the processes
        my_total_count = my_counts.sum()
        if self.comm is None:
            total_count = my_total_count
        else:
            total_count = int(self.comm.allreduce(my_total_count))
        
        # The root process saves all the data. First it setps up the output
        # file here.
        if self.rank == 0:
            print("Overall pair count = ", total_count)
            # Create space for the catalog
            catalog_group = outfile.create_group(f"catalog{suffix}")
            catalog_group.create_dataset("cluster_id", shape=(ncluster,), dtype=np.int64)
            catalog_group.create_dataset("cluster_theta_max_arcmin", shape=(ncluster,), dtype=np.float64)
            # where the pairs of each cluster are in the index below
            catalog_group.create_dataset("cluster_sample_start", shape=(ncluster,), dtype=np.int64)
            catalog_group.create_dataset("cluster_sample_count", shape=(ncluster,), dtype=np.int64)
            # and for the index into that catalog
            index_group = outfile.create_group(f"index{suffix}")
            index_group.create_dataset("cluster_index", shape=(total_count,), dtype=np.int64)
            index_group.create_dataset("source_index", shape=(total_count,), dtype=np.int64)
            index_group.create_dataset("cluster_id", shape=(total_count,), dtype=np.int64)
            index_group.create_dataset("distance_arcmin", shape=(total_count,), dtype=np.float64)


        # Now we loop through each cluster and collect all the galaxies
        # behind it from all the different processes. Each different
        # process received a different chunk of the shear catalog,
        # so we need to pull together all the shears relevant to each
        # cluster
        start = 0
        for i, c in enumerate(clusters):

            if (self.rank == 0) and (i%100 == 0):
                print(f"Collecting data for cluster {i}")

            if len(per_cluster_data[i]) == 0:
                # If there is no background data for this cluster
                # for this particular processor then we just send
                # an empty array
                indices = np.zeros(0, dtype=int)
                distances = np.zeros(0)
            else:
                # Each process flattens the list of all the galaxies for this cluster
                indices = np.concatenate([d[0] for d in per_cluster_data[i]])
                distances = np.concatenate([d[1] for d in per_cluster_data[i]])


            # If we are running in parallel then collect together the values from
            # all the processes
            if self.comm is not None:
                indices, distances = self.collect(indices, distances)

            # Only the root process does the writing, so the others just
            # go to the next set of clusters.
            if self.rank != 0:
                continue

            # Sort so the indices are montonic increasing
            if indices.size != 0:
                srt = indices.argsort()
                indices = indices[srt]
                distances = distances[srt]
            
            # And finally write out all the data from the root process.
            n = indices.size
            # First we write out the information about the clusters,
            # and the index into the other index.
            print(f"Found {n} total galaxies in catalog for cluster {c['id']}")
            catalog_group["cluster_id"][i] = c["id"]
            catalog_group["cluster_theta_max_arcmin"][i] = np.degrees(cluster_theta_max[i]) * 60
            catalog_group["cluster_sample_start"][i] = start
            catalog_group["cluster_sample_count"][i] = n

            # Now we write out the per-cluster shear catalog information
            index_group["cluster_index"][start:start + n] = i
            index_group["source_index"][start:start + n] = indices
            index_group["cluster_id"][start:start + n] = catalog_group["cluster_id"][i]
            index_group["distance_arcmin"][start:start + n] = np.degrees(distances) * 60

            start += n

    def find_galaxies_near_each_cluster(self, galaxy_data, cluster_data, tree, max_theta_max, index, cluster_theta_max=None):
        # Get the location of the galaxies in this chunk of data,
        # in the form that the tree requires. Only the rows of the
        # chunk in index are searched.
        X = np.radians([galaxy_data["dec"][index], galaxy_data["ra"][index]]).T


        # First we will get all the objects that
        # are near each cluster using the maximum search radius for any cluster.
        # Then in a moment we will cut down to the ones that are
        # within the radius for each specific cluster.
        nearby_clusters, cluster_distances1 = tree.query_radius(X, max_theta_max, return_distance=True)

        # Now we invert our tree information. We currently have the list of
        # all the clusters near each galaxy. Now we invert it to get the list
        # of galaxies near this cluster. This strange pattern is because then
        # we only have to make the tree object (which does fast searches for
        # nearby objects once, for the cluster information.
        # We do it with flat arrays of all the (galaxy, cluster) pairs, which
        # take much less memory than lists when the search radius is large.
        n_near = np.array([c.size for c in nearby_clusters])
        if n_near.sum() == 0:
            return {}, {}
        cluster_i = np.concatenate(nearby_clusters)
        distance = np.concatenate(cluster_distances1)
        gal_i = np.repeat(index, n_near)
        del nearby_clusters, cluster_distances1

        # Cut down to the radius of each cluster
        if cluster_theta_max is not None:
            close = distance < cluster_theta_max[cluster_i]
            cluster_i, distance, gal_i = cluster_i[close], distance[close], gal_i[close]

        # Group the pairs by cluster, keeping the order of the galaxies
        order = np.argsort(cluster_i, kind="stable")
        cluster_i, distance, gal_i = cluster_i[order], distance[order], gal_i[order]
        found, starts = np.unique(cluster_i, return_index=True)
        ends = np.append(starts[1:], cluster_i.size)
        nearby_galaxies = {c: gal_i[a:b] for c, a, b in zip(found, starts, ends)}
        nearby_galaxy_distances = {c: distance[a:b] for c, a, b in zip(found, starts, ends)}

        return nearby_galaxies, nearby_galaxy_distances


    def collect(self, indices, distances):
        # total number of background objects for this cluster, over all the processes

        counts = np.array(self.comm.allgather(indices.size))
        total = counts.sum()

        # Early exit if nothing here
        if total == 0:
            indices = np.zeros(0, dtype=int)
            distances = np.zeros(0)
            return indices, distances

        # This collects together all the results from different processes for this cluster
        if self.rank == 0:
            all_indices = np.empty(total, dtype=indices.dtype)
            all_distances = np.empty(total, dtype=distances.dtype)
            self.comm.Gatherv(sendbuf=distances, recvbuf=(all_distances, counts))
            self.comm.Gatherv(sendbuf=indices, recvbuf=(all_indices, counts))
            indices = all_indices
            distances = all_distances
        else:
            self.comm.Gatherv(sendbuf=distances, recvbuf=(None, counts))
            self.comm.Gatherv(sendbuf=indices, recvbuf=(None, counts))

        return indices, distances


    def compute_theta_max(self, z):
        """
        Convert the maximum radius into a maximum angle at the redshift
        of each cluster.
        """
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


    def load_cluster_catalog(self):   # TO TEST
        from astropy.table import Table
        with self.open_input("cluster_catalog") as f:
            g = f["clusters/"]
            cl_list = self.load_cluster_list(group=g)
            
        return cl_list
    
    
    def load_cluster_list(self, group=None):
        from astropy.table import Table
        ra = group["ra"][:]
        dec = group["dec"][:]
        redshift = group["redshift"][:]
        rich = group["richness"][:]
        ids = group["cluster_id"][:]        
        
        return Table({"ra": ra, "dec": dec, "redshift": redshift, "richness": rich, "id": ids})
          




    def select_on_redshift(self, data, gal_index, cluster_z, delta_z):
        """Selects which galaxies pass the redshift selection and return info used

        Returns
        -------
        z_good : np.ndarray
            Booloeans to filter gal_index for sources that pass the cut
        """
        redshift_cut_criterion = self.config["redshift_cut_criterion"]
        if redshift_cut_criterion == "pdf":

            pdf_z = data["pdf_z"]
            pdf = data["pdf_pz"][gal_index]
            # If we cut based on the PDF then we need the probability
            # that the galaxy z is behind the cluster to be greater
            # than a cut criterion.
            pdf_frac = pdf[:, pdf_z > cluster_z + delta_z].sum(axis=1) / pdf.sum(axis=1)
            z_good = pdf_frac > self.config["redshift_cut_criterion_pdf_fraction"]

        elif redshift_cut_criterion in ["ztrue", "zmode", "zmean", "zmedian", "catalog_redshift"]:

            zgal = data["redshift"][gal_index]
            z_good = zgal > cluster_z + delta_z

        elif redshift_cut_criterion == "tomo_bins":

            tomo_edges = data["tomo_bin_edges"]
            source_tomo_inds = data["bin"][gal_index]

            # find minimum index of the tomo bins to be used
            min_tomo_ind = (tomo_edges[:,0] <= cluster_z + delta_z).sum() - 1
            # if cluster in last bin, change min_tomo_ind to include last bin
            if tomo_edges[-1, 0] < cluster_z + delta_z < tomo_edges[-1, 1]:
                min_tomo_ind -= 1
            z_good = (source_tomo_inds>min_tomo_ind)*(source_tomo_inds>=0)

        else:
            raise NotImplementedError("Not implemented other z cuts")
        return z_good

#def get_cluster_shear_catalogs_index_from_cluster_id(cluster_file, cluster_id):
#    outfile = HDFFile(cluster_file, "r").file
#    cond = outfile['catalog/cluster_id'][:] == cluster_id
#    return np.where(cond)[0][0]
