"""
Turn the binned cluster catalog into a TXPipe binned lens catalog.

CLClusterBinningRedshiftRichness writes clusters to
cluster_bin/bin_zbin_{i}_richbin_{j}. TXTwoPoint (do_shear_pos) instead reads
lens/bin_{k}/{ra,dec,weight} and a lens n(z), so this stage rewrites the same
clusters in that layout, with each non-empty (zbin, richbin) pair as one lens bin.
"""

import numpy as np
from ...base_stage import PipelineStage
from ...data_types import HDFFile, BinnedCatalog, QPNOfZFile


class CLClusterLensCatalog(PipelineStage):
    """Write the binned cluster catalog as a binned_lens_catalog and lens n(z) for TXTwoPoint."""

    name = "CLClusterLensCatalog"
    parallel = False
    inputs = [("cluster_catalog_tomography", HDFFile)]
    outputs = [
        ("binned_lens_catalog", BinnedCatalog),
        ("lens_photoz_stack", QPNOfZFile),
    ]
    config_options = {}

    def run(self):
        import qp

        # Read the non-empty (zbin, richbin) groups, in the order CLClusterBinningRedshiftRichness made them.
        # Empty bins are left out: they would give an all-zero (NaN once normalised) n(z).
        bins = []
        with self.open_input("cluster_catalog_tomography") as f:
            g = f["cluster_bin"]
            for zi in range(g.attrs["nz"]):
                for ri in range(g.attrs["nr"]):
                    b = g[f"bin_zbin_{zi}_richbin_{ri}"]
                    if b["ra"].size == 0:
                        print(f"Skipping empty cluster bin zbin_{zi}_richbin_{ri}")
                        continue
                    data = {col: b[col][:] for col in ["ra", "dec", "redshift"]}
                    bins.append((zi, ri, dict(b.attrs), data))

        if not bins:
            raise ValueError("No clusters in any redshift/richness bin")

        # TXPipe evaluates n(z) on 0 < z < 3, so the histogram uses the same range, dz = 0.01
        z_edges = np.linspace(0.0, 3.0, 301)
        nz_hist = np.zeros((len(bins) + 1, z_edges.size - 1))

        with self.open_output("binned_lens_catalog") as f:
            lens = f.create_group("lens")
            lens.attrs["nbin_lens"] = len(bins)

            for k, (zi, ri, edges, data) in enumerate(bins):
                n = data["ra"].size
                print(f"lens bin {k} = zbin_{zi}_richbin_{ri}: {n} clusters")
                b = lens.create_group(f"bin_{k}")
                b.create_dataset("ra", data=data["ra"])
                b.create_dataset("dec", data=data["dec"])
                b.create_dataset("z", data=data["redshift"])
                # TXTwoPoint reads the weight column; all clusters count equally
                b.create_dataset("weight", data=np.ones(n))

                # Keep the link back to the cluster bin, so lens_k in the sacc file can be identified
                b.attrs["zbin"] = zi
                b.attrs["richbin"] = ri
                for key in ["z_min", "z_max", "rich_min", "rich_max"]:
                    b.attrs[key] = edges[key]

                nz_hist[k], _ = np.histogram(data["redshift"], bins=z_edges)

        # Lens n(z): one histogram per bin, plus the sum as the last (2D) entry
        nz_hist[-1] = nz_hist[:-1].sum(axis=0)
        stack = qp.Ensemble(qp.hist, data={"bins": z_edges, "pdfs": nz_hist})
        with self.open_output("lens_photoz_stack", wrapper=True) as f:
            f.write_ensemble(stack)
