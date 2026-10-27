import os
import gc
import numpy as np
from ...base_stage import PipelineStage
from ...data_types import ShearCatalog, HDFFile, PhotozPDFFile, FiducialCosmology, TomographyCatalog, ShearCatalog
from ...shear_calibration import Calibrator
from collections import defaultdict
import yaml
import ceci
import warnings

class CLClusterShearCalibrator(PipelineStage):
    """
    Calibrate the background shears of each cluster stack (richness-redshift bin)
    with the response of the sources of that stack, using the TXPipe calculator of
    the shear catalog type (metadetect or anacal).

    Work in progress: the stacks are built (all the sources of the clusters of a bin,
    in the CLMM radial bins), but the calibration is not applied yet, so the g1, g2
    saved are the ones of cluster_shear_catalogs (calibrated=False).

    Only the sources inside [r_min, r_max] are saved, with their radial bin. The output
    has the same catalog/index layout as cluster_shear_catalogs.
    """
    name = "CLClusterShearCalibrator"
    inputs = [
        ("cluster_catalog_tomography", HDFFile),
        ("cluster_shear_catalogs", HDFFile),
        ("fiducial_cosmology", FiducialCosmology),
        ("shear_tomography_catalog", TomographyCatalog),
    ]

    outputs = [
        ("cluster_shear_catalogs_calibrated", HDFFile),
    ]

    config_options = {
        "r_min" : 0.2, #in Mpc or arcmin
        "r_max" : 3.0, #in Mpc or arcmin
        "nbins" : 5, # number of bins
        "bin_units": "mpc", # options are mpc or arcmin
    }

    def run(self):
        import astropy
        import h5py
        import clmm
        import clmm.cosmology.ccl
        bin_min = self.config["r_min"]
        bin_max = self.config["r_max"]
        self.distance_bins = clmm.dataops.make_bins(bin_min, bin_max, nbins=self.config["nbins"], method="evenlog10width")
        print(f"The bins are: {self.distance_bins}")

        with self.open_input("fiducial_cosmology", wrapper=True) as f:
            ccl_cosmo = f.to_ccl()
            self.clmm_cosmo = clmm.cosmology.ccl.CCLCosmology()
            self.clmm_cosmo.set_be_cosmo(ccl_cosmo)

        # Catalog type, delta_gamma and selection cuts used by the source selector
        with self.open_input("shear_tomography_catalog") as f:
            self.catalog_type = f["tomography"].attrs["catalog_type"]
            self.selector_config = self.load_selector_config(f)
        print(f"Calibrating a {self.catalog_type} catalog")
        print(self.selector_config)
        with self.open_input("cluster_catalog_tomography") as f:
            bin_names = list(f["cluster_bin"].keys())

        stacks = []
        for bin_name in bin_names:
            clusters, sources = self.load_cluster_bin_sources(bin_name)
            if len(sources) == 0:
                print(f"{bin_name}: {len(clusters)} clusters, no sources")
                continue
            # Only the sources that enter the profile will be calibrated
            sources = sources[sources["radial_bin"] >= 0]
            if len(sources) == 0:
                print(f"{bin_name}: {len(clusters)} clusters, no sources in [r_min, r_max]")
                continue
            counts = np.bincount(sources["radial_bin"], minlength=self.config["nbins"])
            print(f"{bin_name}: {len(clusters)} clusters, {len(sources)} sources, per radial bin {counts}")

            # TODO: calibrate the sources of the stack here
            stacks.append((bin_name, clusters, sources))

        self.write_calibrated_catalog(stacks)


    def load_selector_config(self, f):
        """
        Config of the source selector that made the shear tomography catalog (f, open),
        read back from its provenance, where TXPipe saves each option as a string
        attribute "config/<key>".

        To override the selector config (a catalog that did not come from a TXPipe
        source selector, or different cuts for the clusters), change this function.
        """
        # str(None) is saved as "None", which yaml reads back as a string
        return {k[len("config/"):]: yaml.safe_load(v) for k, v in f["provenance"].attrs.items() if k.startswith("config/")}

    def load_cluster_bin_sources(self, bin_name):
        """
        Collect the clusters of one richness-redshift bin and all their background sources,
        and put each cluster-source pair in the CLMM radial bins (self.distance_bins).

        Returns two astropy tables:
        - clusters: one row per cluster of the bin, with its number of sources
        - sources: one row per cluster-source pair, with the cluster_id, the distance in
          bin_units and radial_bin (-1 outside [r_min, r_max]).
          source_index points to the row of the shear catalog.

        g1, g2, tangential_comp and cross_comp come from CLClusterShearCatalogs, where they
        are calibrated with the global response of the source selector.
        """
        import clmm
        from astropy.table import Table, vstack

        bin_units = self.config["bin_units"].lower()
        nbins = len(self.distance_bins) - 1
        cols = ["source_index", "distance_arcmin", "tangential_comp", "cross_comp", "weight", "g1", "g2"]

        with self.open_input("cluster_catalog_tomography") as f:
            g = f["cluster_bin"][bin_name]
            clusters = Table({col: g[col][:] for col in ["cluster_id", "ra", "dec", "redshift", "richness"]})
            clusters.meta.update(dict(g.attrs))

        n_sources = np.zeros(len(clusters), dtype=int)
        sources = []
        with self.open_input("cluster_shear_catalogs") as f:
            profile_type = f["index"].attrs["profile_type"]
            coordinate_system = f["provenance"].attrs["config/coordinate_system"]
            row = {int(cid): i for i, cid in enumerate(f["catalog/cluster_id"][:])}
            starts = f["catalog/cluster_sample_start"][:]
            counts = f["catalog/cluster_sample_count"][:]
            index = f["index"]

            for i, cl in enumerate(clusters):
                j = row[int(cl["cluster_id"])]
                start, n = starts[j], counts[j]
                n_sources[i] = n
                # Clusters outside the shear footprint have no sources
                if n == 0:
                    continue

                cat = Table({col: index[col][start:start + n] for col in cols})
                cat["cluster_id"] = np.full(n, cl["cluster_id"])
                if bin_units == "mpc":
                    cat["distance"] = clmm.utils.convert_units(np.asarray(cat["distance_arcmin"]), "arcmin", "Mpc", cl["redshift"], self.clmm_cosmo)
                else:
                    cat["distance"] = cat["distance_arcmin"]

                # bin i is [edge_i, edge_i+1), sources outside the edges get -1
                radial_bin = np.digitize(cat["distance"], self.distance_bins) - 1
                radial_bin[(radial_bin >= nbins)] = -1
                cat["radial_bin"] = radial_bin
                sources.append(cat)

        clusters["n_sources"] = n_sources
        sources = vstack(sources) if sources else Table()
        sources.meta.update({"bin_name": bin_name, "profile_type": profile_type,
                             "coordinate_system": coordinate_system, "bin_units": bin_units,
                             "distance_bins": self.distance_bins})

        return clusters, sources


    def write_calibrated_catalog(self, stacks):
        """
        Same catalog/index layout as cluster_shear_catalogs, with only the clusters of the
        stacks and their sources inside [r_min, r_max], plus the radial bin and the stack
        of each pair. Only g1, g2 are saved: the weights and the tangential and cross
        components are computed later from them.
        The calibration is not applied yet, so g1, g2 are the ones of cluster_shear_catalogs.
        """
        import h5py
        cols = ["source_index", "cluster_id", "distance_arcmin", "g1", "g2", "radial_bin"]
        ncluster = sum(len(clusters) for _, clusters, _ in stacks)
        npair = sum(len(sources) for _, _, sources in stacks)

        f = self.open_output("cluster_shear_catalogs_calibrated")
        catalog = f.create_group("catalog")
        catalog.create_dataset("cluster_id", (ncluster,), dtype=np.int64)
        catalog.create_dataset("cluster_sample_start", (ncluster,), dtype=np.int64)
        catalog.create_dataset("cluster_sample_count", (ncluster,), dtype=np.int64)
        catalog.create_dataset("cluster_bin", (ncluster,), dtype=h5py.string_dtype())

        index = f.create_group("index")
        for col in cols:
            dtype = np.int64 if col in ["source_index", "cluster_id", "radial_bin"] else np.float64
            index.create_dataset(col, (npair,), dtype=dtype)

        if stacks:
            meta = stacks[0][2].meta
            for key in ["profile_type", "coordinate_system", "bin_units"]:
                index.attrs[key] = meta[key]
        index.attrs["distance_bins"] = self.distance_bins
        index.attrs["catalog_type"] = self.catalog_type
        index.attrs["calibrated"] = False

        i = 0
        start = 0
        for bin_name, clusters, sources in stacks:
            # sources are ordered by cluster, as in cluster_shear_catalogs
            for cl in clusters:
                n = int(np.sum(sources["cluster_id"] == cl["cluster_id"]))
                catalog["cluster_id"][i] = cl["cluster_id"]
                catalog["cluster_sample_start"][i] = start
                catalog["cluster_sample_count"][i] = n
                catalog["cluster_bin"][i] = bin_name
                start += n
                i += 1

            s = start - len(sources)
            for col in cols:
                index[col][s:start] = np.asarray(sources[col])

        f.close()

