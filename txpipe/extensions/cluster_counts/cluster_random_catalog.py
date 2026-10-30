"""
Turn the binned cluster randoms into a TXPipe binned random catalog.

CLClusterRandoms writes randoms in the cluster_catalog layout, and
CLClusterBinningRedshiftRichness (aliased) bins them into
cluster_bin/bin_zbin_{i}_richbin_{j}. TXDeltaSigma instead reads
randoms/bin_{k}/{ra,dec,z}, so this stage rewrites the randoms in that layout.
Random bin k is taken from the same (zbin, richbin) as lens bin k in
binned_lens_catalog, so the two always line up.
"""

import numpy as np
from ...base_stage import PipelineStage
from ...data_types import HDFFile, BinnedCatalog, RandomsCatalog


class CLClusterRandomsBinned(PipelineStage):
    """Write the binned cluster randoms as binned_random_catalog and random_cats for TXDeltaSigma."""

    name = "CLClusterRandomsBinned"
    parallel = False
    inputs = [
        ("cluster_random_catalog_tomography", HDFFile),
        ("binned_lens_catalog", BinnedCatalog),
    ]
    outputs = [
        ("binned_random_catalog", RandomsCatalog),
        ("random_cats", RandomsCatalog),
    ]
    config_options = {}

    def run(self):
        # The lens_k -> (zbin, richbin) mapping and bin edges, as written by CLClusterLensCatalog
        lens_bins = []
        with self.open_input("binned_lens_catalog") as f:
            lens = f["lens"]
            for k in range(lens.attrs["nbin_lens"]):
                b = lens[f"bin_{k}"]
                lens_bins.append((dict(b.attrs), b["ra"].size))

        # Randoms for the same (zbin, richbin) groups, in lens bin order
        bins = []
        with self.open_input("cluster_random_catalog_tomography") as f:
            g = f["cluster_bin"]
            for k, (attrs, n_lens) in enumerate(lens_bins):
                name = f"bin_zbin_{attrs['zbin']}_richbin_{attrs['richbin']}"
                b = g[name]
                if b["ra"].size == 0:
                    raise ValueError(f"No randoms in {name}, but lens bin {k} has {n_lens} clusters")
                data = {
                    "ra": b["ra"][:].astype(np.float64),
                    "dec": b["dec"][:].astype(np.float64),
                    "z": b["redshift"][:].astype(np.float64),
                }
                print(f"random bin {k} = {name}: {data['ra'].size} randoms, "
                      f"{data['ra'].size / n_lens:.1f} per cluster")
                bins.append(data)

        with self.open_output("binned_random_catalog") as f:
            randoms = f.create_group("randoms")
            randoms.attrs["nbin"] = len(bins)
            for k, ((attrs, _), data) in enumerate(zip(lens_bins, bins)):
                b = randoms.create_group(f"bin_{k}")
                for col, values in data.items():
                    b.create_dataset(col, data=values)
                # Same link back to the cluster bin as lens/bin_k
                b.attrs.update(attrs)

        # All bins together, with the bin index, as TXRandomCat writes it;
        # TXDeltaSigma only reads this for an "all" lens bin
        with self.open_output("random_cats") as f:
            randoms = f.create_group("randoms")
            for col in ["ra", "dec", "z"]:
                randoms.create_dataset(col, data=np.concatenate([data[col] for data in bins]))
            randoms.create_dataset(
                "bin", data=np.concatenate([np.full(data["ra"].size, k, dtype=np.int16) for k, data in enumerate(bins)])
            )
