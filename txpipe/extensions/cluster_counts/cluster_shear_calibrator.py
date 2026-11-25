import numpy as np
from ...base_stage import PipelineStage
from ...data_types import ShearCatalog, HDFFile, FiducialCosmology, BinAssignments, CatalogSelection
from ...utils import read_shear_catalog_type
import yaml
import warnings


class CLClusterShearCalibrator(PipelineStage):
    """
    Response of the background sources of each cluster stack (richness-redshift bin),
    in each radial bin or for the whole stack, using the TXPipe calculator of the
    shear catalog type (metadetect, scalar_metadetect or anacal).

    Only the response is computed and saved: the full R of each stack, its bootstrap error,
    and the radial bins they were computed in. This stage is optional:
    CLClusterEnsembleProfiles with shear_calibration "cluster" calibrates the
    shears of the sources of each stack with its R (per radial bin with radial_calibration,
    as in Zhou et al. 2026, arXiv:2606.21141, Eq. 23, where the response changes with radius
    because of the magnification of the sources).

    The response is the plain mean over the pairs (uniform weights).
    """
    name = "CLClusterShearCalibrator"
    # The response is computed in one process
    parallel = False
    inputs = [
        ("cluster_catalog_tomography", HDFFile),
        ("cluster_shear_selection", HDFFile),
        ("fiducial_cosmology", FiducialCosmology),
        ("shear_catalog", ShearCatalog),
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
        "radial_calibration": False, # one response per radial bin of each stack, or one per stack
        "n_bootstrap": 100, # resamplings of the clusters for the error of R, 0 for none
        "bootstrap_seed": 42,
        "delta_gamma": float, # the shear step of the sheared versions, as in TXComputeShearCalibration*
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

        self.catalog_type = read_shear_catalog_type(self)
        self.delta_gamma = self.config["delta_gamma"]
        print(f"Calibrating a {self.catalog_type} catalog")
        # The sources of the pairs are those that pass the selection and, with the
        # require_tomo_bin option of CLClusterSourceSelect, are in a tomographic bin
        with self.open_input("cluster_shear_selection") as f:
            self.require_tomo_bin = yaml.safe_load(str(f["provenance"].attrs["config/require_tomo_bin"]))
        with self.open_input("cluster_catalog_tomography") as f:
            bin_names = list(f["cluster_bin"].keys())

        stacks = []
        self.stacks_without_sources = []
        for bin_name in bin_names:
            clusters, sources = self.load_cluster_bin_sources(bin_name)
            # Only the sources that enter the profile are used
            if len(sources) > 0:
                sources = sources[sources["radial_bin"] >= 0]
            if len(sources) == 0:
                print(f"{bin_name}: {len(clusters)} clusters, no sources in [r_min, r_max]")
                self.stacks_without_sources.append(bin_name)
                stacks.append((bin_name, clusters, sources, None, None))
                continue
            counts = np.bincount(sources["radial_bin"], minlength=self.config["nbins"])
            print(f"{bin_name}: {len(clusters)} clusters, {len(sources)} sources, per radial bin {counts}")

            calibrators, errors = self.compute_calibrators(sources, clusters["cluster_id"])
            for k, (calibrator, err) in enumerate(zip(calibrators, errors)):
                label = f"radial bin {k}" if self.config["radial_calibration"] else "stack"
                if calibrator is None:
                    print(f"{bin_name}, {label}: no sources")
                else:
                    print(f"{bin_name}, {label}: R = {np.ravel(calibrator.R)} +- {None if err is None else np.ravel(err)}")
            stacks.append((bin_name, clusters, sources, calibrators, errors))

        self.write_response(stacks)


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
        with self.open_input("cluster_shear_selection") as f:
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

                # bin i is [edge_i, edge_i+1), and the last one includes r_max, as in CLMM.
                # Sources outside the edges get -1
                radial_bin = np.digitize(cat["distance"], self.distance_bins) - 1
                radial_bin[np.asarray(cat["distance"]) == self.distance_bins[-1]] = nbins - 1
                radial_bin[(radial_bin >= nbins)] = -1
                cat["radial_bin"] = radial_bin
                sources.append(cat)

        sources = vstack(sources) if sources else Table()
        sources.meta.update({"bin_name": bin_name, "bin_units": bin_units})

        return clusters, sources


    def compute_calibrators(self, sources, cluster_ids):
        """
        Calibrators of the sources of one stack: one per radial bin with radial_calibration
        (None for a radial bin with no sources), otherwise a list with the one of the stack.
        Also returns the bootstrap error of R of each (None when there is none).
        cluster_ids are the clusters of the stack, which the bootstrap resamples.
        """
        if self.config["radial_calibration"]:
            radial_bins = range(len(self.distance_bins) - 1)
        else:
            radial_bins = [None]

        calibrators = []
        errors = []
        for k in radial_bins:
            sel = np.ones(len(sources), dtype=bool) if k is None else np.asarray(sources["radial_bin"] == k)
            if not sel.any():
                calibrators.append(None)
                errors.append(None)
                continue
            calibrator, err = self.compute_stack_calibrator(sources[sel], cluster_ids, radial_bin=k)
            calibrators.append(calibrator)
            errors.append(err)
        return calibrators, errors


    def compute_stack_calibrator(self, sources, cluster_ids, radial_bin=None):
        """
        Response of the sources of one stack, with the TXPipe calculator of the catalog type.
        Each cluster-source pair counts once, so a galaxy behind two clusters counts twice.
        sources can be the sources of one radial bin of the stack, given by radial_bin,
        so that the sheared metadetect variants are cut to the same radial bin.

        Returns the calibrator and the bootstrap error of its R.
        """
        data, row_clusters, make_calculator = self.read_calibration_data(sources, radial_bin)
        calculator = make_calculator()
        calculator.add_data(data)
        calibrator = calculator.collect().calibrator
        return calibrator, self.bootstrap_error(make_calculator, data, row_clusters, cluster_ids)


    def bootstrap_error(self, make_calculator, data, row_clusters, cluster_ids):
        """
        Error of each element of R from n_bootstrap resamplings of the clusters of the stack with
        replacement. A cluster drawn n times brings all its pairs n times, so the scatter
        between clusters and the sources shared by several clusters are included.
        row_clusters gives the cluster of each row of each array of data.
        Each stack (and radial bin) starts from bootstrap_seed, so its error does not depend
        on the other stacks. There is no error (None) with fewer than two clusters with sources.
        """
        n_bootstrap = self.config["n_bootstrap"]
        if n_bootstrap == 0 or np.unique(np.concatenate(list(row_clusters.values()))).size < 2:
            return None
        rng = np.random.default_rng(self.config["bootstrap_seed"])
        cluster_ids = np.sort(np.asarray(cluster_ids))
        position = {key: np.searchsorted(cluster_ids, c) for key, c in row_clusters.items()}
        R = []
        for b in range(n_bootstrap):
            draws = np.bincount(rng.integers(len(cluster_ids), size=len(cluster_ids)),
                                minlength=len(cluster_ids))
            resampled = {key: np.repeat(values, draws[position[key]]) for key, values in data.items()}
            calculator = make_calculator()
            # A resampling can leave a variant with no sources
            with np.errstate(divide="ignore", invalid="ignore"):
                calculator.add_data(resampled)
                R.append(np.atleast_1d(calculator.collect().calibrator.R))
        return np.nanstd(R, axis=0, ddof=1)


    def read_calibration_data(self, sources, radial_bin=None):
        """
        Columns used by the calculator of the catalog type for the sources of one stack.

        Returns the data for the calculator, the cluster of each row of each of its
        arrays, and a function that makes a new calculator.
        """
        from ...calibration.calibration_utils import (MetaDetectCalculator, ScalarMetaDetectCalculator, AnaCalCalculator,
                                          META_VARIANTS, SCALAR_META_VARIANTS)

        delta_gamma = self.delta_gamma
        def read_rows(dataset):
            return self.read_source_rows(dataset, sources["source_index"])

        # The same sources as the pairs, for each variant: those that pass the cuts of the
        # selection stage and, with require_tomo_bin, have a tomographic bin, as in
        # TXComputeShearCalibrationBase.select_any_bin
        require_tomo_bin = self.require_tomo_bin
        def select_any_bin(d):
            if require_tomo_bin:
                return (d["bin"] >= 0) & d["selection"]
            return d["selection"]

        data = {}
        row_clusters = {}
        if self.catalog_type in ["metadetect", "scalar_metadetect"]:
            if self.catalog_type == "metadetect":
                variants = META_VARIANTS
                calculator_class = MetaDetectCalculator
            else:
                variants = SCALAR_META_VARIANTS
                calculator_class = ScalarMetaDetectCalculator

            # Each variant is averaged over its own sources, which can be different
            # objects from ns, as in TXComputeShearCalibrationMetadetect
            with self.open_input("shear_catalog") as f, \
                 self.open_input("shear_bin_assignments") as b, \
                 self.open_input("shear_selection") as s:
                for v in variants:
                    source_index, cluster_id = self.variant_source_index(sources, v, f, radial_bin)
                    for col in ["g1", "g2"]:
                        data[f"{v}/{col}"] = self.read_source_rows(f[f"shear/{v}/{col}"], source_index)
                    # uniform weights, as AnaCal: the catalog weights are not used in the profile
                    data[f"{v}/weight"] = np.ones(len(source_index))
                    data[f"{v}/bin"] = self.read_source_rows(b[f"{v}/bin"], source_index)
                    data[f"{v}/selection"] = self.read_source_rows(s[f"shear/{v}/selection"], source_index)
                    for col in ["g1", "g2", "weight", "bin", "selection"]:
                        row_clusters[f"{v}/{col}"] = cluster_id

        elif self.catalog_type == "anacal":
            # The columns read by TXComputeShearCalibrationAnacal. The sheared variants
            # are the _1p, _1m, _2p, _2m columns of the same rows.
            cols = ["e1", "e2", "weight", "wsel", "e1_raw", "e2_raw",
                    "de1_dg1", "de2_dg2", "weight_dg1", "weight_dg2"]
            with self.open_input("shear_catalog") as f:
                data.update({col: read_rows(f[f"shear/{col}"]) for col in cols})
            with self.open_input("shear_bin_assignments") as f:
                for suf in ["", "_1p", "_1m", "_2p", "_2m"]:
                    data[f"bin{suf}"] = read_rows(f[f"bin{suf}"])
            with self.open_input("shear_selection") as f:
                for suf in ["", "_1p", "_1m", "_2p", "_2m"]:
                    data[f"selection{suf}"] = read_rows(f[f"shear/selection{suf}"])
            # All the columns are rows of the same pairs
            cluster_id = np.asarray(sources["cluster_id"])
            row_clusters = {key: cluster_id for key in data}
            calculator_class = AnaCalCalculator

        else:
            raise ValueError(f"Cluster calibration not implemented for {self.catalog_type} catalogs")

        def make_calculator():
            return calculator_class(select_any_bin, delta_gamma)

        return data, row_clusters, make_calculator


    def variant_source_index(self, sources, variant, shear_file, radial_bin=None):
        """
        Rows of the shear catalog of a metadetect variant for the sources of one stack.

        ns uses the pairs of the stack. A sheared variant uses its own pairs when
        CLClusterSourceSelect saved them (index_<variant> group), in the same radial
        bin (radial_bin), or in [r_min, r_max] when radial_bin is None. Otherwise, as a
        placeholder, it reuses the ns rows, which is only right when the variants have
        the same objects in the same rows (simulations).

        Returns the rows and the cluster of each row.
        """
        if variant == "ns":
            return sources["source_index"], np.asarray(sources["cluster_id"])

        with self.open_input("cluster_shear_selection") as f:
            has_variant_pairs = f"index_{variant}" in f

        if has_variant_pairs:
            _, variant_sources = self.load_cluster_bin_sources(sources.meta["bin_name"], variant=variant)
            if len(variant_sources) == 0:
                return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
            if radial_bin is None:
                keep = variant_sources["radial_bin"] >= 0
            else:
                keep = variant_sources["radial_bin"] == radial_bin
            return variant_sources["source_index"][keep], np.asarray(variant_sources["cluster_id"][keep])

        # Placeholder until CLClusterSourceSelect saves the pairs of each variant
        if shear_file[f"shear/{variant}/g1"].size != shear_file["shear/ns/g1"].size:
            raise ValueError(f"The {variant} catalog has different objects from ns and "
                             f"cluster_shear_selection has no index_{variant} pairs")
        warnings.warn(f"No index_{variant} pairs in cluster_shear_selection: using the ns rows "
                      f"for {variant}, which assumes the variants have the same objects")
        return sources["source_index"], np.asarray(sources["cluster_id"])


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
        - R and its bootstrap error R_err (NaN without bootstrap): with radial_calibration,
          one row per radial bin (NaN where the radial bin has no sources), otherwise those
          of the stack
        - cluster_id: the clusters of the stack, so that the next stage can check its binning
        - attributes: the numbers of clusters and sources, and the bin edges of the stack
        A stack without sources has no R and R_err.
        The radial bins and the options used are attributes of the calibration group,
        so that the next stage can check they are the same as its own.
        """
        f = self.open_output("cluster_shear_response")
        calibration = f.create_group("calibration")
        calibration.attrs["catalog_type"] = self.catalog_type
        calibration.attrs["radial_calibration"] = self.config["radial_calibration"]
        calibration.attrs["distance_bins"] = self.distance_bins
        calibration.attrs["bin_units"] = self.config["bin_units"].lower()
        calibration.attrs["n_bootstrap"] = self.config["n_bootstrap"]
        # the stacks without sources in the radial bins have no response
        calibration.attrs["stacks_without_sources"] = np.array(self.stacks_without_sources, dtype="S")

        for bin_name, clusters, sources, calibrators, errors in stacks:
            g = calibration.create_group(bin_name)
            g.create_dataset("cluster_id", data=np.asarray(clusters["cluster_id"]))
            g.attrs["n_clusters"] = len(clusters)
            g.attrs["n_sources"] = len(sources)
            radial_bins = sources["radial_bin"] if len(sources) > 0 else np.zeros(0, dtype=int)
            g.attrs["n_sources_per_radial_bin"] = np.bincount(radial_bins, minlength=len(self.distance_bins) - 1)
            g.attrs.update(clusters.meta)
            if calibrators is None:
                continue
            ref = next(c for c in calibrators if c is not None)
            nan = np.full(np.atleast_1d(ref.R).shape, np.nan)
            R = [nan if c is None else np.atleast_1d(c.R) for c in calibrators]
            R_err = [nan if e is None else e for e in errors]
            if not self.config["radial_calibration"]:
                R, R_err = R[0], R_err[0]
            g.create_dataset("R", data=np.array(R))
            g.create_dataset("R_err", data=np.array(R_err))

        f.close()
