from ..base_stage import PipelineStage
from ..data_types import HDFFile, FitsFile, QPNOfZFile, RandomsCatalog, FiducialCosmology
import numpy as np
from ceci.config import StageParameter

class TXIngestDESIRandoms(PipelineStage):
    """
    Ingest a DESI random catalog from FITS and write random_cats,
    binned_random_catalog, and a subsampled binned_random_catalog_sub.

    Output format matches TXRandomCat: group "randoms" with datasets
    ra, dec, z, comoving_distance, bin at the top level, and per-bin
    subgroups bin_0 ... bin_N in the binned outputs.
    """

    name = "TXIngestDESIRandoms"
    parallel = False
    inputs = [
        ("desi_random_catalog", FitsFile),
        ("fiducial_cosmology", FiducialCosmology),
    ]
    outputs = [
        ("random_cats", RandomsCatalog),
        ("binned_random_catalog", RandomsCatalog),
        ("binned_random_catalog_sub", RandomsCatalog),
    ]
    config_options = {
        "lens_zbin_edges": StageParameter(list, [float], msg="Edges of lens redshift bins."),
        "chunk_rows": StageParameter(int, 100_000, msg="Number of rows to process in each chunk."),
        "sample_rate": StageParameter(float, 0.5, msg="Fraction of randoms retained in the sub-catalog."),
        "z_col": StageParameter(str, "z", msg="Redshift column name in the FITS random catalog."),
        "ra_col": StageParameter(str, "ra", msg="Right ascension column name in the FITS random catalog."),
        "dec_col": StageParameter(str, "dec", msg="Declination column name in the FITS random catalog."),
    }

    def run(self):
        import pyccl

        with self.open_input("fiducial_cosmology", wrapper=True) as f:
            cosmo = f.to_ccl()

        zbin_edges = self.config["lens_zbin_edges"]
        nbin = len(zbin_edges) - 1
        chunk_rows = self.config["chunk_rows"]
        z_col = self.config["z_col"]
        ra_col = self.config["ra_col"]
        dec_col = self.config["dec_col"]

        # Count total rows
        f = self.open_input("desi_random_catalog")
        n_total = f[1].get_nrows()
        f.close()
        print(f"Total randoms: {n_total}")

        # First pass: count objects per redshift bin
        bin_counts = np.zeros(nbin, dtype=np.int64)
        for s, e, data in self.iterate_fits("desi_random_catalog", 1, [z_col], chunk_rows):
            zbin = np.digitize(data[z_col], zbin_edges) - 1
            zbin[zbin >= nbin] = -1
            valid = zbin >= 0
            bin_counts += np.bincount(zbin[valid], minlength=nbin)
        for i, c in enumerate(bin_counts):
            print(f"  Bin {i}: {c} randoms")

        # Create output files and datasets
        random_cats = self.open_output("random_cats")
        binned_output = self.open_output("binned_random_catalog")

        g = random_cats.create_group("randoms")
        g.create_dataset("ra", (n_total,), dtype=np.float64)
        g.create_dataset("dec", (n_total,), dtype=np.float64)
        g.create_dataset("z", (n_total,), dtype=np.float64)
        g.create_dataset("comoving_distance", (n_total,), dtype=np.float64)
        g.create_dataset("bin", (n_total,), dtype=np.int16)

        gb = binned_output.create_group("randoms")
        gb.attrs["nbin"] = nbin
        subgroups = []
        for i in range(nbin):
            sg = gb.create_group(f"bin_{i}")
            sg.create_dataset("ra", (bin_counts[i],), dtype=np.float64)
            sg.create_dataset("dec", (bin_counts[i],), dtype=np.float64)
            sg.create_dataset("z", (bin_counts[i],), dtype=np.float64)
            sg.create_dataset("comoving_distance", (bin_counts[i],), dtype=np.float64)
            subgroups.append(sg)

        # Second pass: fill data
        bin_cursors = np.zeros(nbin, dtype=np.int64)
        cols = [ra_col, dec_col, z_col]
        for s, e, data in self.iterate_fits("desi_random_catalog", 1, cols, chunk_rows):
            ra = data[ra_col]
            dec = data[dec_col]
            z = data[z_col]
            zbin = np.digitize(z, zbin_edges) - 1
            zbin[zbin >= nbin] = -1
            chi = pyccl.comoving_radial_distance(cosmo, 1.0 / (1.0 + z))

            g["ra"][s:e] = ra
            g["dec"][s:e] = dec
            g["z"][s:e] = z
            g["comoving_distance"][s:e] = chi
            g["bin"][s:e] = zbin

            for i in range(nbin):
                sel = zbin == i
                n_sel = int(sel.sum())
                if n_sel > 0:
                    c = bin_cursors[i]
                    subgroups[i]["ra"][c:c + n_sel] = ra[sel]
                    subgroups[i]["dec"][c:c + n_sel] = dec[sel]
                    subgroups[i]["z"][c:c + n_sel] = z[sel]
                    subgroups[i]["comoving_distance"][c:c + n_sel] = chi[sel]
                    bin_cursors[i] += n_sel

        self.subsample_randoms(binned_output, nbin)

        random_cats.close()
        binned_output.close()

    def subsample_randoms(self, binned_output, nbin):
        """Randomly subsample the binned random catalog and write binned_random_catalog_sub."""
        sample_rate = self.config["sample_rate"]
        print(f"Sub-sampling randoms at rate {sample_rate}")

        binned_output_sub = self.open_output("binned_random_catalog_sub")
        gb_sub = binned_output_sub.create_group("randoms")
        gb_sub.attrs["nbin"] = nbin

        for j in range(nbin):
            ra = binned_output[f"randoms/bin_{j}/ra"][:]
            dec = binned_output[f"randoms/bin_{j}/dec"][:]
            z = binned_output[f"randoms/bin_{j}/z"][:]
            chi = binned_output[f"randoms/bin_{j}/comoving_distance"][:]

            ntotal = len(ra)
            nsub = int(sample_rate * ntotal)
            idx = np.random.choice(ntotal, size=nsub, replace=False)

            sg = gb_sub.create_group(f"bin_{j}")
            sg.create_dataset("ra", data=ra[idx])
            sg.create_dataset("dec", data=dec[idx])
            sg.create_dataset("z", data=z[idx])
            sg.create_dataset("comoving_distance", data=chi[idx])

        binned_output_sub.close()


class TXIngestDESI(PipelineStage):
    """
    Ingest a DESI catalog (mock or real)

    This starts with the FITS file format.
    """

    name = "TXIngestDESI"
    parallel = False

    inputs = [
    ("desi_catalog_selected", FitsFile),  # from stage 1
]

    outputs = [
        ("lens_catalog", HDFFile),
        ("binned_lens_catalog", HDFFile),
        ("lens_tomography_catalog_unweighted", HDFFile),
        ("lens_tomography_catalog", HDFFile),
        ("lens_photoz_stack", QPNOfZFile),
    ]

    # TODO: CHANGE CONFIG OPTIONS
    config_options = {
        "lens_zbin_edges": StageParameter(list, [float], msg="Edges of lens redshift bins."),
        "chunk_rows": StageParameter(int, 100_000, msg="Number of rows to process in each chunk."),
        "zmin": StageParameter(float, 0.0, msg="Minimum redshift for binning."),
        "zmax": StageParameter(float, 3.0, msg="Maximum redshift for binning."),
        "dz": StageParameter(float, 0.01, msg="Redshift bin width."),
        #"bands": StageParameter(str, "grizy", msg="Bands to use for DESI selection."),
        "mock": StageParameter(bool, False, msg="Whether to use a mock catalog."),
    }


    def run(self):
        import qp

        # Count number of objects
        f = self.open_input("desi_catalog_selected")
        n = f[1].get_nrows()
        f.close()

        chunk_rows = self.config["chunk_rows"]
        # bands = self.config["bands"]
        zbin_edges = self.config["lens_zbin_edges"]
        nbin_lens = len(zbin_edges) - 1

        cat = self.open_output("lens_catalog")
        cat_binned = self.open_output("binned_lens_catalog")
        tomo_uw = self.open_output("lens_tomography_catalog_unweighted")
        tomo_w = self.open_output("lens_tomography_catalog")

        # redshift grid
        zmin = self.config["zmin"]
        zmax = self.config["zmax"]
        dz = self.config["dz"]
        z_grid = np.arange(zmin, zmax, dz)
        nz_grid = np.zeros((nbin_lens + 1, z_grid.size))
        nz = len(z_grid)

        # Create space in outputs
        g = cat.create_group("lens")
        # g.attrs["bands"] = bands
        g.create_dataset("ra", (n,), dtype=np.float64)
        g.create_dataset("dec", (n,), dtype=np.float64)
        g.create_dataset("z", (n,), dtype=np.float64)
        # for b in bands:
        #     g.create_dataset(f"mag_{b}", (n,), dtype=np.float64)
        #     g.create_dataset(f"mag_err_{b}", (n,), dtype=np.float64)
        # g.attrs["bands"] = bands

        gb = cat_binned.create_group("lens")
        gb.attrs["nbin_lens"] = nbin_lens
        # for bn in range(nbin_lens):
        #     gb_ = gb.create_group(f"bin_{bn}")
        #     gb_.create_dataset("ra", (n,), dtype=np.float64)
        #     gb_.create_dataset("dec", (n,), dtype=np.float64)
        #     gb_.create_dataset("z", (n,), dtype=np.float64)
        #     gb_.create_dataset("w_sys", (n,), dtype=np.float64)

        h_uw = tomo_uw.create_group("tomography")
        h_uw.create_dataset("bin", (n,), dtype=np.int32)
        h_uw.create_dataset("lens_weight", (n,), dtype=np.float64)
        h_uw.attrs["nbin"] = nbin_lens
        h_uw.attrs["zbin_edges"] = zbin_edges
        h_counts_uw = tomo_uw.create_group("counts")
        h_counts_uw.create_dataset("counts", (nbin_lens,), dtype="i")
        h_counts_uw.create_dataset("counts_2d", (1,), dtype="i")

        h_w = tomo_w.create_group("tomography")
        h_w.create_dataset("bin", (n,), dtype=np.int32)
        h_w.create_dataset("lens_weight", (n,), dtype=np.float64)
        h_w.attrs["nbin"] = nbin_lens
        h_w.attrs["zbin_edges"] = zbin_edges
        h_counts_w = tomo_w.create_group("counts")
        h_counts_w.create_dataset("counts", (nbin_lens,), dtype="i")
        h_counts_w.create_dataset("counts_2d", (1,), dtype="i")

        # we keep track of the counts per-bin also
        counts = np.zeros(nbin_lens, dtype=np.int64)
        counts_2d = 0

        # all cols that might be useful
        cols = ["ra", "dec", "redshift", "weight"]

        for s, e, data in self.iterate_fits("desi_catalog_selected", 1, cols, chunk_rows):
            n = data["ra"].size
            z = data["redshift"]
            # Unit weight still
            weight = np.repeat(1.0, n)

            # work out the redshift bin for each object, if any.
            # do any other selection here
            zbin = np.digitize(z, zbin_edges) - 1
            zbin[zbin == nbin_lens] = -1

            # # can select on any other criterion here, e.g.
            # # mag or chisq.  This is an example
            # sel = data["chisq"] < 10
            # # deselect these objects
            # zbin[~sel] = -1

            # Build up the count of the n(z) histograms per-bin
            z_grid_index = np.floor((z - zmin) / dz).astype(int)
            for i, (i_z, b) in enumerate(zip(z_grid_index, zbin)):
                if b >= 0:
                    nz_grid[b, i_z] += weight[i]

            # Build up the counts
            any_bin = zbin >= 0
            counts += np.bincount(zbin[any_bin], minlength=nbin_lens)
            counts_2d += any_bin.sum()

            # save data to tomography catalog
            g["ra"][s:e] = data["ra"]
            g["dec"][s:e] = data["dec"]
            g["z"][s:e] = data["redshift"]

            # # including mags
            # for i, b in enumerate(bands):
            #     g[f"mag_{b}"][s:e] = data["mag"][:, i]
            #     g[f"mag_err_{b}"][s:e] = data["mag_err"][:, i]

            h_uw["bin"][s:e] = zbin
            h_uw["lens_weight"][s:e] = 1.0

            h_w["bin"][s:e] = zbin
            if self.config["mock"]:
                h_w["lens_weight"][s:e] = 1.0
            else:
                h_w["lens_weight"][s:e] = data["weight"]

        # save data to binned lens catalog
        for bn in range(nbin_lens):
            sel = h_w["bin"][:] == bn

            gb_ = gb.create_group(f"bin_{bn}")
            gb_.create_dataset("ra", data=g["ra"][:][sel], dtype=np.float64)
            gb_.create_dataset("dec", data=g["dec"][:][sel], dtype=np.float64)
            gb_.create_dataset("z", data=g["z"][:][sel], dtype=np.float64)
            gb_.create_dataset("w_sys", data=h_w["lens_weight"][:][sel], dtype=np.float64)

        # this is an overall count
        h_counts_uw["counts"][:] = counts
        h_counts_uw["counts_2d"][:] = counts_2d
        h_counts_w["counts"][:] = counts
        h_counts_w["counts_2d"][:] = counts_2d

        # Generate and save the 2D n(z) histogram also, just
        # by summing up all the individual values.
        nz_grid[-1] = nz_grid[:-1].sum(axis=0)

        stack_object = qp.Ensemble(qp.hist, data={"bins": z_grid, "pdfs": nz_grid[:, :-1]})
        with self.open_output("lens_photoz_stack", wrapper=True) as stack:
            stack.write_ensemble(stack_object)