import numpy as np
from ...base_stage import PipelineStage
from ...data_types import ShearCatalog, HDFFile, FiducialCosmology, TomographyCatalog, BinAssignments, CatalogSelection
import yaml
import warnings


def tangential_response(R):
    """
    Response of the tangential shear. Per source R_T = cos²2φ R11 + sin²2φ R22, and the
    off-diagonal terms average out (Zhou et al. 2026, arXiv:2606.21141, Eq. 22; McClintock
    et al. 2019, arXiv:1805.00039, Eqs. 4-5). Averaged over sources spread around the
    clusters, R_T = (R11 + R22) / 2. AnaCal gives a scalar R, which is already R_T.
    """
    R = np.atleast_1d(R)
    if R.size == 1:
        return float(R[0])
    if R.shape == (2, 2):
        return 0.5 * (R[0, 0] + R[1, 1])
    # one response per shear component
    return float(np.mean(R))


class CLClusterShearCalibrator(PipelineStage):
    """
    Response of the background sources of each cluster stack (richness-redshift bin),
    in each radial bin or for the whole stack, using the TXPipe calculator of the
    shear catalog type (metadetect, scalar_metadetect or anacal).

    Only the response is computed and saved: the tangential response R_T and the full R
    of each stack, and the radial bins they were computed in. CLClusterEnsembleProfiles
    divides the tangential profile of each radial bin by its R_T, as in Zhou et al. 2026
    (arXiv:2606.21141, Eq. 23), where the response changes with radius because of the
    magnification of the sources.
    """
    name = "CLClusterShearCalibrator"
    inputs = [
        ("cluster_catalog_tomography", HDFFile),
        ("cluster_shear_catalogs", HDFFile),
        ("fiducial_cosmology", FiducialCosmology),
        ("shear_catalog", ShearCatalog),
        ("shear_tomography_catalog", TomographyCatalog),
        ("shear_bin_assignments", BinAssignments),
        ("shear_selection", CatalogSelection),
    ]

    outputs = [
        ("cluster_shear_response", HDFFile),
    ]

    config_options = {
        "r_min" : 0.2, #in Mpc or arcmin
        "r_max" : 3.0, #in Mpc or arcmin
        "nbins" : 5, # number of bins
        "bin_units": "mpc", # options are mpc or arcmin
        "chunk_rows": 100_000, # rows read at once from the shear catalogs
        "radial_calibration": True, # one response per radial bin of each stack, or one per stack
    }

    def run(self):
        import clmm
        import clmm.cosmology.ccl
        bin_min = self.config["r_min"]
        bin_max = self.config["r_max"]
        self.distance_bins = clmm.dataops.make_bins(bin_min, bin_max, nbins=self.config["nbins"], method="evenlog10width")
        print(f"The bins are: {self.distance_bins}")

        # To convert the distances to Mpc
        with self.open_input("fiducial_cosmology", wrapper=True) as f:
            ccl_cosmo = f.to_ccl()
            self.clmm_cosmo = clmm.cosmology.ccl.CCLCosmology()
            self.clmm_cosmo.set_be_cosmo(ccl_cosmo)

        # Catalog type, and delta_gamma of the calibration stage
        with self.open_input("shear_tomography_catalog") as f:
            self.catalog_type = f["tomography"].attrs["catalog_type"]
            self.selector_config = self.load_selector_config(f)
        print(f"Calibrating a {self.catalog_type} catalog")
        with self.open_input("cluster_catalog_tomography") as f:
            bin_names = list(f["cluster_bin"].keys())

        stacks = []
        for bin_name in bin_names:
            clusters, sources = self.load_cluster_bin_sources(bin_name)
            if len(sources) == 0:
                print(f"{bin_name}: {len(clusters)} clusters, no sources")
                continue
            # Only the sources that enter the profile are used
            sources = sources[sources["radial_bin"] >= 0]
            if len(sources) == 0:
                print(f"{bin_name}: {len(clusters)} clusters, no sources in [r_min, r_max]")
                continue
            counts = np.bincount(sources["radial_bin"], minlength=self.config["nbins"])
            print(f"{bin_name}: {len(clusters)} clusters, {len(sources)} sources, per radial bin {counts}")

            calibrators = self.compute_calibrators(sources)
            for k, calibrator in enumerate(calibrators):
                label = f"radial bin {k}" if self.config["radial_calibration"] else "stack"
                if calibrator is None:
                    print(f"{bin_name}, {label}: no sources")
                else:
                    print(f"{bin_name}, {label}: R_T = {tangential_response(calibrator.R):.4f}, R = {np.ravel(calibrator.R)}")
            stacks.append((bin_name, clusters, sources, calibrators))

        self.write_response(stacks)


    def load_selector_config(self, f):
        """
        Config of the stage that made the shear tomography catalog.
        Read back from its provenance, where TXPipe saves each option as a string
        attribute "config/<key>".

        To override it (a catalog that did not come from TXPipe), change this function.
        """
        # str(None) is saved as "None", which yaml reads back as a string
        return {k[len("config/"):]: yaml.safe_load(v) for k, v in f["provenance"].attrs.items() if k.startswith("config/")}

    def load_cluster_bin_sources(self, bin_name, variant=None):
        """
        Collect the clusters of one richness-redshift bin and all their background sources,
        and put each cluster-source pair in the CLMM radial bins (self.distance_bins).

        Returns two astropy tables:
        - clusters: one row per cluster of the bin
        - sources: one row per cluster-source pair, with the cluster_id, the distance in
          bin_units and radial_bin (-1 outside [r_min, r_max]).
          source_index points to the row of the shear catalog.

        variant: None for the pairs of the main (ns) catalog, in the catalog and index groups,
        or a metadetect variant (1p, 1m, ...) for the pairs found in the catalog of that
        variant, in the catalog_<variant> and index_<variant> groups.
        """
        import clmm
        from astropy.table import Table, vstack

        bin_units = self.config["bin_units"].lower()
        nbins = len(self.distance_bins) - 1
        suffix = "" if variant is None else f"_{variant}"
        cols = ["source_index", "distance_arcmin"]

        with self.open_input("cluster_catalog_tomography") as f:
            g = f["cluster_bin"][bin_name]
            clusters = Table({col: g[col][:] for col in ["cluster_id", "redshift"]})
            clusters.meta.update(dict(g.attrs))

        sources = []
        with self.open_input("cluster_shear_catalogs") as f:
            row = {int(cid): i for i, cid in enumerate(f[f"catalog{suffix}/cluster_id"][:])}
            starts = f[f"catalog{suffix}/cluster_sample_start"][:]
            counts = f[f"catalog{suffix}/cluster_sample_count"][:]
            index = f[f"index{suffix}"]

            for cl in clusters:
                j = row[int(cl["cluster_id"])]
                start, n = starts[j], counts[j]
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

        sources = vstack(sources) if sources else Table()
        sources.meta.update({"bin_name": bin_name, "bin_units": bin_units})

        return clusters, sources


    def compute_calibrators(self, sources):
        """
        Calibrators of the sources of one stack: one per radial bin with radial_calibration
        (None for a radial bin with no sources), otherwise a list with the one of the stack.
        """
        if self.config["radial_calibration"]:
            radial_bins = range(len(self.distance_bins) - 1)
        else:
            radial_bins = [None]

        calibrators = []
        for k in radial_bins:
            sel = np.ones(len(sources), dtype=bool) if k is None else np.asarray(sources["radial_bin"] == k)
            if not sel.any():
                calibrators.append(None)
                continue
            calibrators.append(self.compute_stack_calibrator(sources[sel], radial_bin=k))
        return calibrators


    def compute_stack_calibrator(self, sources, radial_bin=None):
        """
        Response of the sources of one stack, with the TXPipe calculator of the catalog type.
        Each cluster-source pair counts once, so a galaxy behind two clusters counts twice.
        sources can be the sources of one radial bin of the stack, given by radial_bin,
        so that the sheared metadetect variants are cut to the same radial bin.

        Returns the calibrator.
        """
        from ...calibration_utils import (MetaDetectCalculator, ScalarMetaDetectCalculator, AnaCalCalculator,
                                          META_VARIANTS, SCALAR_META_VARIANTS)

        delta_gamma = self.selector_config["delta_gamma"]
        def read_rows(dataset):
            return self.read_source_rows(dataset, sources["source_index"])

        # As in TXComputeShearCalibrationBase.select_any_bin: a source of each variant is
        # used if it has a tomographic bin and passes the cuts of the selection stage
        def select_any_bin(d):
            return (d["bin"] >= 0) & d["selection"]

        if self.catalog_type in ["metadetect", "scalar_metadetect"]:
            if self.catalog_type == "metadetect":
                variants = META_VARIANTS
                calculator = MetaDetectCalculator(select_any_bin, delta_gamma)
            else:
                variants = SCALAR_META_VARIANTS
                calculator = ScalarMetaDetectCalculator(select_any_bin, delta_gamma)

            # Each variant is averaged over its own sources, which can be different
            # objects from ns, as in TXComputeShearCalibrationMetadetect
            data = {}
            with self.open_input("shear_catalog") as f, \
                 self.open_input("shear_bin_assignments") as b, \
                 self.open_input("shear_selection") as s:
                for v in variants:
                    source_index = self.variant_source_index(sources, v, f, radial_bin)
                    for col in ["g1", "g2", "weight"]:
                        data[f"{v}/{col}"] = self.read_source_rows(f[f"shear/{v}/{col}"], source_index)
                    data[f"{v}/bin"] = self.read_source_rows(b[f"{v}/bin"], source_index)
                    data[f"{v}/selection"] = self.read_source_rows(s[f"shear/{v}/selection"], source_index)

            calculator.add_data(data)

        elif self.catalog_type == "anacal":
            # The columns read by TXComputeShearCalibrationAnacal. The sheared variants
            # are the _1p, _1m, _2p, _2m columns of the same rows.
            cols = ["e1", "e2", "weight", "wsel", "e1_raw", "e2_raw",
                    "de1_dg1", "de2_dg2", "weight_dg1", "weight_dg2"]
            with self.open_input("shear_catalog") as f:
                data = {col: read_rows(f[f"shear/{col}"]) for col in cols}
            with self.open_input("shear_bin_assignments") as f:
                for suf in ["", "_1p", "_1m", "_2p", "_2m"]:
                    data[f"bin{suf}"] = read_rows(f[f"bin{suf}"])
            with self.open_input("shear_selection") as f:
                for suf in ["", "_1p", "_1m", "_2p", "_2m"]:
                    data[f"selection{suf}"] = read_rows(f[f"shear/selection{suf}"])

            calculator = AnaCalCalculator(select_any_bin, delta_gamma)
            calculator.add_data(data)

        else:
            raise ValueError(f"Cluster calibration not implemented for {self.catalog_type} catalogs")

        return calculator.collect().calibrator


    def variant_source_index(self, sources, variant, shear_file, radial_bin=None):
        """
        Rows of the shear catalog of a metadetect variant for the sources of one stack.

        ns uses the pairs of the stack. A sheared variant uses its own pairs when
        CLClusterShearCatalogs saved them (index_<variant> group), in the same radial
        bin (radial_bin), or in [r_min, r_max] when radial_bin is None. Otherwise, as a
        placeholder, it reuses the ns rows, which is only right when the variants have
        the same objects in the same rows (simulations).
        """
        if variant == "ns":
            return sources["source_index"]

        with self.open_input("cluster_shear_catalogs") as f:
            has_variant_pairs = f"index_{variant}" in f

        if has_variant_pairs:
            _, variant_sources = self.load_cluster_bin_sources(sources.meta["bin_name"], variant=variant)
            if len(variant_sources) == 0:
                return np.zeros(0, dtype=np.int64)
            if radial_bin is None:
                keep = variant_sources["radial_bin"] >= 0
            else:
                keep = variant_sources["radial_bin"] == radial_bin
            return variant_sources["source_index"][keep]

        # Placeholder until CLClusterShearCatalogs saves the pairs of each variant
        if shear_file[f"shear/{variant}/g1"].size != shear_file["shear/ns/g1"].size:
            raise ValueError(f"The {variant} catalog has different objects from ns and "
                             f"cluster_shear_catalogs has no index_{variant} pairs")
        warnings.warn(f"No index_{variant} pairs in cluster_shear_catalogs: using the ns rows "
                      f"for {variant}, which assumes the variants have the same objects")
        return sources["source_index"]


    def read_source_rows(self, dataset, source_index):
        """
        Read the rows source_index (unsorted, with repeats) of an HDF5 dataset.
        Only the chunks of chunk_rows rows that contain sources are read, so the
        memory does not depend on how spread the sources are in the catalog.
        """
        chunk_rows = self.config["chunk_rows"]
        # h5py reads faster as slices, so read each chunk with sources and pick
        idx, inverse = np.unique(source_index, return_inverse=True)
        out = np.empty(idx.size, dtype=dataset.dtype)
        chunks, first = np.unique(idx // chunk_rows, return_index=True)
        last = np.append(first[1:], idx.size)
        for c, i, j in zip(chunks, first, last):
            start = c * chunk_rows
            out[i:j] = dataset[start:start + chunk_rows][idx[i:j] - start]
        return out[inverse]


    def write_response(self, stacks):
        """
        Response of each stack in the group calibration/<stack>:
        - R_T, the tangential response, and the full R: with radial_calibration, one row
          per radial bin (NaN where the radial bin has no sources), otherwise those of
          the stack
        - attributes: the numbers of clusters and sources, and the bin edges of the stack
        The radial bins and the options used are attributes of the calibration group,
        so that the next stage can check they are the same as its own.
        """
        f = self.open_output("cluster_shear_response")
        calibration = f.create_group("calibration")
        calibration.attrs["catalog_type"] = self.catalog_type
        calibration.attrs["radial_calibration"] = self.config["radial_calibration"]
        calibration.attrs["distance_bins"] = self.distance_bins
        calibration.attrs["bin_units"] = self.config["bin_units"].lower()

        for bin_name, clusters, sources, calibrators in stacks:
            ref = next(c for c in calibrators if c is not None)
            R = [np.full(np.atleast_1d(ref.R).shape, np.nan) if c is None else np.atleast_1d(c.R) for c in calibrators]
            R_T = [np.nan if c is None else tangential_response(c.R) for c in calibrators]
            if not self.config["radial_calibration"]:
                R, R_T = R[0], R_T[0]
            g = calibration.create_group(bin_name)
            g.create_dataset("R_T", data=np.array(R_T))
            g.create_dataset("R", data=np.array(R))
            g.attrs["n_clusters"] = len(clusters)
            g.attrs["n_sources"] = len(sources)
            g.attrs["n_sources_per_radial_bin"] = np.bincount(sources["radial_bin"], minlength=len(self.distance_bins) - 1)
            g.attrs.update(clusters.meta)

        f.close()
