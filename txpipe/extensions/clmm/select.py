import os
import gc
import numpy as np
from ...base_stage import PipelineStage
from ...data_types import ShearCatalog, HDFFile, PhotozPDFFile, FiducialCosmology, TomographyCatalog, ShearCatalog
from ...utils.calibrators import Calibrator
from ...utils import DynamicSplitter
from collections import defaultdict
import yaml
import ceci
import itertools


class CLClusterBinningRedshiftRichness(PipelineStage):
    name = "CLClusterBinningRedshiftRichness"
    parallel = False
    inputs = [("cluster_catalog", HDFFile)]
    outputs = [("cluster_catalog_tomography", HDFFile)]
    config_options = {
        "zedge": [0.2, 0.4, 0.6, 0.8, 1.0],
        "richedge": [5., 10., 20.],
        "initial_size": 100_000,
        "chunk_rows": 100_000,
    }
    def run(self):        
        initial_size = self.config["initial_size"]
        chunk_rows = self.config["chunk_rows"]
        
        zedge = np.array(self.config['zedge'])
        richedge = np.array(self.config['richedge'])

        nz = len(zedge - 1) 
        nr = len(richedge - 1)

        # add infinities to either end to catch objects that spill out
        zedge = np.concatenate([[-np.inf], zedge, [np.inf]])
        richedge = np.concatenate([[-np.inf], richedge, [np.inf]])

        # all pairs of z bin, richness bin indices
        bins = list(itertools.product(range(nz), range(nr)))
        bin_names = {f"zbin_{i}_richbin_{j}":initial_size for i,j in bins}

        # Columns we want to save for each object
        cols = ['cluster_id', 'dec', 'ra', 'redshift', 'redshift_err', 'richness', 'richness_err', 'scaleval']


        f = self.open_output("cluster_catalog_tomography")
        g = f.create_group("cluster_bin")
        g.attrs['nr'] = nr
        g.attrs['nz'] = nz
        splitter = DynamicSplitter(g, "bin", cols, bin_names)

        # Make an iterator that will read a chunk of data at a time
        it = self.iterate_hdf("cluster_catalog", "clusters", cols, chunk_rows)

        # Loop through the chunks of data; each time `data` will be a
        # dictionary of column names -> numpy arrays
        for _, _, data in it:
            n = len(data["redshift"])

            # Figure out which bin each halo it in, if any, starts at 0
            zbin = np.digitize(data['redshift'], zedge) - 1
            richbin = np.digitize(data["richness"], richedge) - 1

            # Find which bin each object is in, or None
            for zi in range(1, nz):
                for ri in range(1, nr):
                    w = np.where((zbin == zi) & (richbin == ri))
                    # if there are no objects in this bin in this chunk,
                    # then we skip the rest
                    if w[0].size == 0:
                        continue

                    # Otherwise we extract the bit of this chunk of
                    # data that is in this bin and have our splitter
                    # object write it out.
                    d = {name:col[w] for name, col in data.items()}
                    bin_name = f"zbin_{zi-1}_richbin_{ri-1}" #TO CHANGE ?
                    splitter.write_bin(d, bin_name)

        # Truncate arrays to correct size
        splitter.finish()

        # Save metadata
        for (i, j), name in zip(bins, bin_names):
            metadata = splitter.subgroups[name].attrs
            metadata['rich_min'] = richedge[j]
            metadata['rich_max'] = richedge[j+1]
            metadata['z_min'] = zedge[i]
            metadata['z_max'] = zedge[i+1]

        f.close()
        

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
        import clmm.cosmology.ccl

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

        # Store the neighbours for each cluster
        per_cluster_data = [list() for i in range(ncluster)]

        with self.open_input("fiducial_cosmology", wrapper=True) as f:
            ccl_cosmo = f.to_ccl()
            clmm_cosmo = clmm.cosmology.ccl.CCLCosmology()
            clmm_cosmo.set_be_cosmo(ccl_cosmo)

        # Buffer in redshift behind each object        
        delta_z = self.config["delta_z"]

        # Loop through the source catalog.
        for s, e, data in self.iterate_source_catalog():
            print(f"Process {self.rank} processing chunk {s:,} - {e:,}")
                
            nearby_gals, nearby_gal_dists = self.find_galaxies_near_each_cluster(data, clusters, tree, max_theta_max)

            # Now we have all the galaxies near each cluster
            # We need to cut down to a specific radius for this
            # cluster
            for cluster_index, gal_index in nearby_gals.items():
                # gal_index is an index into this chunk of galaxy data
                # pointing to all the galaxies near this cluster
                gal_index = np.array(gal_index)
                distance = np.array(nearby_gal_dists[cluster_index])

                if gal_index.size == 0:
                    continue

                # Cut down to galaxies close enough to this cluster
                dist_good = distance < cluster_theta_max[cluster_index]
                gal_index = gal_index[dist_good]
                distance = distance[dist_good]

                if gal_index.size == 0:
                    continue

                cluster_z = clusters[cluster_index]["redshift"]
                # # Cut down to clusters that are in front of this galaxy
                zgal = data["redshift"][gal_index]
                z_good = zgal > cluster_z + delta_z
                gal_index = gal_index[z_good]
                distance = distance[z_good]

                if gal_index.size == 0:
                    continue

                # Compute weights
                weights = self.compute_weights(clmm_cosmo, data, gal_index, cluster_z)

                # we want to save the index into the overall shear catalog,
                # not just into this chunk of data
                global_index = data["original_index"][gal_index]
                per_cluster_data[cluster_index].append((global_index, distance, weights))

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
            outfile = self.open_output("cluster_shear_catalogs")
            # Create space for the catalog
            catalog_group = outfile.create_group("catalog")
            catalog_group.create_dataset("cluster_sample_start", shape=(ncluster,), dtype=np.int32)
            catalog_group.create_dataset("cluster_sample_count", shape=(ncluster,), dtype=np.int32)
            catalog_group.create_dataset("cluster_id", shape=(ncluster,), dtype=np.int64)
            catalog_group.create_dataset("cluster_theta_max_arcmin", shape=(ncluster,), dtype=np.float64)
            # and for the index into that catalog
            index_group = outfile.create_group("index")
            index_group.create_dataset("cluster_index", shape=(total_count,), dtype=np.int64)
            index_group.create_dataset("source_index", shape=(total_count,), dtype=np.int64)
            index_group.create_dataset("weight", shape=(total_count,), dtype=np.float64)
            index_group.create_dataset("distance_arcmin", shape=(total_count,), dtype=np.float64)


        # Now we loop through each cluster and collect all the galaxies
        # behind it from all the different processes.
        start = 0
        for i, c in enumerate(clusters):

            if (self.rank == 0) and (i%100 == 0):
                print(f"Collecting data for cluster {i}")

            if len(per_cluster_data[i]) == 0:
                indices = np.zeros(0, dtype=int)
                weights = np.zeros(0)
                distances = np.zeros(0)
            else:
                # Each process flattens the list of all the galaxies for this cluster
                indices = np.concatenate([d[0] for d in per_cluster_data[i]])
                distances = np.concatenate([d[1] for d in per_cluster_data[i]])
                weights = np.concatenate([d[2] for d in per_cluster_data[i]])


            # If we are running in parallel then collect together the values from
            # all the processes
            if self.comm is not None:
                indices, weights, distances = self.collect(indices, weights, distances)

            # Only the root process does the writing, so the others just
            # go to the next set of clusters.
            if self.rank != 0:
                continue

            # Sort so the indices are montonic increasing
            if indices.size != 0:
                srt = indices.argsort()
                indices = indices[srt]
                weights = weights[srt]
                distances = distances[srt]
            
            # And finally write out all the data from the root process.
            n = indices.size
            print(f"Found {n} total galaxies in catalog for cluster {c['id']}")
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

    def find_galaxies_near_each_cluster(self, galaxy_data, cluster_data, tree, max_theta_max):
        # Get the location of the galaxies in this chunk of data,
        # in the form that the tree requires
        X = np.radians([galaxy_data["dec"], galaxy_data["ra"]]).T


        # First we will get all the objects that
        # are near each cluster using the maximum search radius for any cluster.
        # Then in a moment we will cut down to the ones that are
        # within the radius for each specific cluster.
        nearby_clusters, cluster_distances1 = tree.query_radius(X, max_theta_max, return_distance=True)
        nearby_galaxies = defaultdict(list)
        nearby_galaxy_distances = defaultdict(list)

        # Now we invert our tree information. We currently have the list of
        # all the clusters near each galaxy. Now we invert it to get the list
        # of galaxies near this cluster. This strange pattern is because then
        # we only have to make the tree object (which does fast searches for
        # nearby objects once, for the cluster information.
        for gal_i, (cluster_indices, cluster_distances) in enumerate(zip(nearby_clusters, cluster_distances1)):
            for (cluster_i, cluster_distance) in zip(cluster_indices, cluster_distances):
                nearby_galaxies[cluster_i].append(gal_i)
                nearby_galaxy_distances[cluster_i].append(cluster_distance)

        return nearby_galaxies, nearby_galaxy_distances


    def collect(self, indices, weights, distances):
        # total number of background objects for t

        counts = np.array(self.comm.allgather(indices.size))
        total = counts.sum()

        # Early exit if nothing here
        if total == 0:
            indices = np.zeros(0, dtype=int)
            weights = np.zeros(0)
            distances = np.zeros(0)
            return indices, weights, distances

        # This collects together all the results from different processes for this cluster
        if self.rank == 0:
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
            pdf_pz = data["pdf_pz"][index]
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
            shape_component1=data["g1"][index],
            shape_component2=data["g2"][index],
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

            # give the shear columns a unified name, whether
            # they are metacal, metadetect, etc., also rename
            # zmean or zmode to "redshift"
            for old, new in rename.items():
                data[new] = data.pop(old)

            # Apply the shear calibration to this sample.
            # Optionally subtract the mean (of the whole WL sample,
            # not the local mean)
            data["g1"], data["g2"] = shear_cal.apply(data["g1"],
                                                     data["g2"],
                                                     subtract_mean=subtract_mean
            )

            # If we are in PDF mode then we need this extra info
            if redshift_criterion == "pdf":
                data["pdf_z"] = pdf_z
                # also rename this for clarity
                data["pdf_pz"] = data.pop("yvals")

            # Give this chunk of data to the main run function
            yield s, e, data



class CombinedClusterCatalog:
    def __init__(self, shear_catalog, shear_tomography_catalog, cluster_catalog, cluster_shear_catalogs, photoz_pdfs):
        _, self.calibrator = Calibrator.load(shear_tomography_catalog)
        self.shear_cat = ShearCatalog(shear_catalog, "r")
        self.pz_cat = PhotozPDFFile(photoz_pdfs,"r").file
        self.cluster_catalog = HDFFile(cluster_catalog, "r").file
        self.cluster_shear_catalogs = HDFFile(cluster_shear_catalogs, "r").file
        self.cluster_cat_cols = list(self.cluster_catalog['clusters'].keys())
        self.ncluster = self.cluster_shear_catalogs['catalog/cluster_id'].size
        self.pz_criterion = "z" + self.cluster_shear_catalogs['provenance'].attrs['config/redshift_criterion']
        self.pz_col = self.pz_cat[f'ancil/{self.pz_criterion}']
    @classmethod
    def from_pipeline_file(cls, pipeline_file, run_dir='.'):
        pipe_config = ceci.Pipeline.build_config(
            pipeline_file,
            dry_run=True
        )

        pipeline = ceci.Pipeline.create(pipe_config)

        outputs = {}
        for stage in pipeline.stages:
            outputs.update(stage.find_outputs(pipe_config["output_dir"]))


        # make a list of files we need
        tags = [
            "shear_catalog",
            "cluster_catalog",
            "cluster_shear_catalogs",
            "shear_tomography_catalog",
            "photoz_pdfs",
        ]

        paths = pipeline.overall_inputs.copy()
        for stage in pipeline.stages:
            paths.update(stage.find_outputs(pipe_config["output_dir"]))

        files = {}
        for tag in tags:
            if tag not in paths:
                raise ValueError(f"This pipeline did not generate or ingest {tag} needed for cluster WL")
            path = paths[tag]
            if not os.path.exists(path):
                raise ValueError(f"File {path} does not exist - pipeline may not have run")
            files[tag] = path

        return cls(**files)


    def get_cluster_info(self, cluster_index):
        return {k: self.cluster_catalog[f'clusters/{k}'][cluster_index] for k in self.cluster_cat_cols}


    def get_background_catalog_indexing(self, cluster_index):
        cat_group = self.cluster_shear_catalogs['catalog']
        index_group = self.cluster_shear_catalogs['index']

        start = cat_group['cluster_sample_start'][cluster_index]
        n = cat_group['cluster_sample_count'][cluster_index]
        end = start + n

        index = index_group['source_index'][start:end]
        weight = index_group['weight'][start:end]
        distance = index_group['distance_arcmin'][start:end]

        return index, distance, weight

    def get_background_shear_catalog(self, cluster_index):
        import clmm
        index, distance, weight = self.get_background_catalog_indexing(cluster_index)
        cat_names, rename = self.shear_cat.get_primary_catalog_names()

        cat = {}
        for col_name in cat_names:
            cat[col_name] = self.shear_cat.file[f'shear/{col_name}'][index]
        
        # rename so no matter what kind of shear catalog you
        # have it's always the same names here
        for old, new in rename.items():
            cat[new] = cat.pop(old)

        # Calibrate g1 and g2
        g1 = cat.pop("g1")
        g2 = cat.pop("g2")
        cat["e1"], cat["e2"] = self.calibrator.apply(g1, g2, subtract_mean=True)

        # Add some more columns and rename some others
        cat["weight_clmm"] = weight
        cat["distance_arcmin"] = distance
        cat["weight_original"] = cat.pop("weight")

        cat[self.pz_criterion] = self.pz_col[index]


        return clmm.GCData(data=cat)

