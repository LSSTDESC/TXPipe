from ..base_stage import PipelineStage
from ..data_types import HDFFile, FitsFile, QPNOfZFile
import numpy as np


class TXIngestRedmagic(PipelineStage):
    """
    Ingest a redmagic catalog

    This starts with the FITS file format, but may be outdated.
    """
    name = "TXIngestRedmagic"
    parallel = False
    inputs = [
        ("redmagic_catalog", FitsFile),
    ]

    outputs = [
        ("lens_catalog", HDFFile),
        ("lens_tomography_catalog_unweighted", HDFFile),
        ("lens_photoz_stack", QPNOfZFile),
    ]

    config_options = {
        "lens_zbin_edges": [float],
        "chunk_rows": 100_000,
        "zmin": 0.0,
        "zmax": 3.0,
        "dz": 0.01,
        "bands": "grizy",
    }

    def run(self):
        import qp
        # Count number of objects
        f = self.open_input("redmagic_catalog")
        n = f[1].get_nrows()
        f.close()

        chunk_rows = self.config["chunk_rows"]
        bands = self.config["bands"]
        zbin_edges = self.config["lens_zbin_edges"]
        nbin_lens = len(zbin_edges) - 1

        cat = self.open_output("lens_catalog")
        tomo = self.open_output("lens_tomography_catalog_unweighted")

        # redshift grid
        zmin = self.config["zmin"]
        zmax = self.config["zmax"]
        dz = self.config["dz"]
        z_grid = np.arange(zmin, zmax, dz)
        nz_grid = np.zeros((nbin_lens + 1, z_grid.size))
        nz = len(z_grid)

        # Create space in outputs
        g = cat.create_group("lens")
        g.attrs["bands"] = bands
        g.create_dataset("ra", (n,), dtype=np.float64)
        g.create_dataset("dec", (n,), dtype=np.float64)
        g.create_dataset("chisq", (n,), dtype=np.float64)
        g.create_dataset("redshift", (n,), dtype=np.float64)
        for b in bands:
            g.create_dataset(f"mag_{b}", (n,), dtype=np.float64)
            g.create_dataset(f"mag_err_{b}", (n,), dtype=np.float64)
        g.attrs["bands"]  = bands

        h = tomo.create_group("tomography")
        h.create_dataset("bin", (n,), dtype=np.int32)
        h.create_dataset("lens_weight", (n,), dtype=np.float64)
        h.create_dataset("counts", (nbin_lens,), dtype="i")
        h.create_dataset("counts_2d", (1,), dtype="i")
        h.attrs["nbin"] = nbin_lens
        h.attrs[f"zbin_edges"] = zbin_edges

        # we keep track of the counts per-bin also
        counts = np.zeros(nbin_lens, dtype=np.int64)
        counts_2d = 0

        # all cols that might be useful
        cols = ["ra", "dec", "zredmagic", "mag", "mag_err", "chisq", "zspec"]

        for (s, e, data) in self.iterate_fits("redmagic_catalog", 1, cols, chunk_rows):
            n = data["ra"].size
            z = data["zredmagic"]
            z_true = data["zspec"]
            # Unit weight still
            weight = np.repeat(1.0, n)

            # work out the redshift bin for each object, if any.
            # do any other selection here
            zbin = np.digitize(z, zbin_edges) - 1
            zbin[zbin == nbin_lens] = -1

            # can select on any other criterion here, e.g.
            # mag or chisq.  This is an example
            sel = data["chisq"] < 10
            # deselect these objects
            zbin[~sel] = -1

            # Build up the count of the n(z) histograms per-bin
            z_grid_index = np.floor((z_true - zmin) / dz).astype(int)
            for i, (i_z, b) in enumerate(zip(z_grid_index, zbin)):
                if b >= 0:
                    nz_grid[b,i_z] += weight[i]

            # Build up the counts
            any_bin = zbin >= 0
            counts += np.bincount(zbin[any_bin], minlength=nbin_lens)
            counts_2d += any_bin.sum()

            # save data
            g["ra"][s:e] = data["ra"]
            g["dec"][s:e] = data["dec"]
            g["chisq"][s:e] = data["chisq"]
            g["redshift"][s:e] = data["zredmagic"]

            # including mags
            for i, b in enumerate(bands):
                g[f"mag_{b}"][s:e] = data["mag"][:, i]
                g[f"mag_err_{b}"][s:e] = data["mag_err"][:, i]

            h["bin"][s:e] = zbin
            h["lens_weight"][s:e] = 1.0

        # this is an overall count
        h["counts"][:] = counts
        h["counts_2d"][:] = counts_2d

        # Generate and save the 2D n(z) histogram also, just
        # by summing up all the individual values.
        nz_grid[-1] = nz_grid[:-1].sum(axis=0)

        stack_object = qp.Ensemble(qp.hist, data={"bins":z_grid, "pdfs":nz_grid[:, :-1]})
        with self.open_output("lens_photoz_stack", wrapper=True) as stack:
            stack.write_ensemble(stack_object)

