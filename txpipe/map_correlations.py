from .base_stage import PipelineStage
from .data_types import MapsFile, FileCollection
import pathlib
import glob
import numpy as np
from ceci.config import StageParameter


class TXMapCorrelations(PipelineStage):
    """
    Plot shear, density, and convergence correlations with survey property maps

    The Supreme code generates survey property maps; this stage makes
    plots of the correlations with those maps with a simple linear fit.

    Since the Supreme maps are loaded from a directory, outside the pipeline,
    we don't know in advance what plots will be generated, so the formal output
    is a directory.
    """

    name = "TXMapCorrelations"
    parallel = False
    inputs = [
        ("lens_maps", MapsFile),
        ("convergence_maps", MapsFile),
        ("source_maps", MapsFile),
        ("mask", MapsFile),
    ]

    outputs = [
        ("map_systematic_correlations", FileCollection),
    ]

    config_options = {
        "supreme_path_root": StageParameter(
            str, "/global/cscratch1/sd/erykoff/dc2_dr6/supreme/supreme_dc2_dr6d_v2", msg="Root path for supreme files."
        ),
        "nbin": StageParameter(int, 20, msg="Number of percentile bins to use in the map property."),
        "outlier_fraction": StageParameter(float, 0.05, msg="Fraction of outliers to exclude."),
    }

    def read_healsparse(self, map_path, nside):
        import healsparse

        # Convert to correct res healsparse map
        m = healsparse.HealSparseMap.read(map_path, degrade_nside=nside)
        return m

    def run(self):
        import healsparse
        import matplotlib.pyplot
        import scipy.stats
        import healpy

        data_maps = {}
        nside = 0

        with self.open_input("lens_maps", wrapper=True) as map_file:
            ngal = map_file.read_map("weighted_ngal_2D")
            nside = map_file.read_map_info("weighted_ngal_2D")["nside"]

        with self.open_input("source_maps", wrapper=True) as map_file:
            source_g1 = map_file.read_map("g1_2D")
            source_g2 = map_file.read_map("g2_2D")

        with self.open_input("convergence_maps", wrapper=True) as map_file:
            kappa = map_file.read_map("kappa_E_2D")

        with self.open_input("mask", wrapper=True) as map_file:
            mask = map_file.read_map("mask")

        # In python (unlike in e.g. C) you can chain equality tests
        # like this (or indeed inequalities). Cool right?
        if not (kappa.nside_sparse == ngal.nside_sparse == mask.nside_sparse):
            raise ValueError("Maps are different sizes")

        output_dir = self.open_output("map_systematic_correlations", wrapper=True)

        root = self.config["supreme_path_root"]
        sys_maps = glob.glob(f"{root}*.hs")
        nsys = len(sys_maps)
        print(f"Found {nsys} total systematic maps")

        outputs = []
        for i, map_path in enumerate(sys_maps):
            # strip root, .hs, and underscores to get friendly name
            sys_name = map_path[len(root) : -3].strip("_")

            # get actual data for this map
            sys_map = self.read_healsparse(map_path, nside)

            # Correlate with g1, g2, ngal, kappa
            print(f"Correlating systematic {i + 1}/{nsys} {sys_name}")
            corr = self.correlate(sys_map, source_g1, mask)
            outfile = self.save(sys_name, "g1", corr, output_dir)
            outputs.append(outfile)

            corr = self.correlate(sys_map, source_g2, mask)
            outfile = self.save(sys_name, "g2", corr, output_dir)
            outputs.append(outfile)

            corr = self.correlate(sys_map, ngal, mask)
            outfile = self.save(sys_name, "number_density", corr, output_dir)
            outputs.append(outfile)

            corr = self.correlate(sys_map, kappa, mask)
            outfile = self.save(sys_name, "convergence", corr, output_dir)
            outputs.append(outfile)

        output_dir.write_listing(outputs)

    def correlate(self, sys_map, data_map, mask):
        """
        Compute a binned correlation between a systematic map and a data map.

        Parameters
        ----------
        sys_map : healsparse.HealSparseMap
            Systematic map
        data_map : healsparse.HealSparseMap
            Data map whose mean value is computed in bins of the systematic.
        mask : healsparse.HealSparseMap
            fractional mask map

        Returns
        -------
        x : np.ndarray
            Mean systematic value in each bin.
        y : np.ndarray
            Mean data value in each bin.
        yerr : np.ndarray
            Uncertainty on the mean data value in each bin, computed as
            sqrt(var / N).
        """
        import scipy.stats
        import healpy

        N = self.config["nbin"]
        f = 0.5 * self.config["outlier_fraction"]

        # clean the data
        valid_pix = np.intersect1d(sys_map.valid_pixels, data_map.valid_pixels)
        valid_pix = np.intersect1d(valid_pix, mask.valid_pixels)
        finite = np.isfinite(sys_map[valid_pix]) & np.isfinite(data_map[valid_pix])
        valid_pix = valid_pix[finite]

        sys_map = sys_map[valid_pix]
        data_map = data_map[valid_pix]

        # Choose bin edges and put pixels in them.
        percentiles = np.linspace(f, 1 - f, N + 1)
        bin_edges = scipy.stats.mstats.mquantiles(sys_map, percentiles)

        # Remove outliers in the systematic
        clip = (sys_map > bin_edges[0]) & (sys_map < bin_edges[-1])
        sys_map = sys_map[clip]
        data_map = data_map[clip]

        # Find the bin each pixel lands in
        bins = np.digitize(sys_map, bin_edges) - 1

        # Get the count in each bin in the
        # systematic parameter
        counts = np.bincount(bins)

        # and the mean of the systematic value
        # itself, and of the y and y^2 values
        x = np.bincount(bins, weights=sys_map) / counts
        y = np.bincount(bins, weights=data_map) / counts
        y2 = np.bincount(bins, weights=data_map**2) / counts

        # the var on the mean = var / count
        yerr = np.sqrt((y2 - y**2) / counts)

        # return things we want to plot
        return x, y, yerr

    def save(self, sys_name, data_name, corr, output_dir):
        import matplotlib.pyplot as plt

        x, y, yerr = corr

        # Make plot
        plt.figure(figsize=(8, 6))
        plt.errorbar(x, y, yerr, fmt=".")
        plt.xlabel(sys_name)
        plt.ylabel(data_name)

        # Save plot
        base = f"{sys_name}_{data_name}.png"
        filename = output_dir.path_for_file(base)
        plt.savefig(filename)
        plt.close()

        return base
