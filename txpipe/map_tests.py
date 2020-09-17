from .base_stage import PipelineStage
from .data_types import MapsFile, FileCollection
import pathlib
import glob
import numpy as np

class TXMapCorrelations(PipelineStage):
    name = "TXMapCorrelations"
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
        "supreme_path_root": "/global/cscratch1/sd/erykoff/dc2_dr6/supreme/supreme_dc2_dr6d_v2",
        "nbin": 20,
        "outlier_fraction": 0.05,
    }

    def read_healsparse(self, map_path, nside):
        import healsparse
        import healpy

        print(f"Reading systematics healsparse map {map_path}")
        # Convert to correct res healpix map
        m = healsparse.HealSparseMap.read(map_path)
        m = m.generate_healpix_map(nside=nside)
        m = healpy.ud_grade(m, nside, order_in="nest", order_out="ring")
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
            source_w  = map_file.read_map("lensing_weight_2D")

        with self.open_input("convergence_maps", wrapper=True) as map_file:
            kappa = map_file.read_map("kappa_E_2D")

        with self.open_input("mask", wrapper=True) as map_file:
            mask = map_file.read_map("mask")

        if not (kappa.size == ngal.size == source_w.size == mask.size):
            raise ValueError("Maps are different sizes")

        output_dir = self.open_output("map_systematic_correlations", wrapper=True)


        root = self.config["supreme_path_root"]
        sys_maps = glob.glob(f"{root}*.hs")
        nsys = len(sys_maps)
        print(f"Found {nsys} total systematic maps")

        outputs = []
        for map_path in sys_maps:
            # strip root, .hs, and underscores to get friendly name
            sys_name = map_path[len(root):-3].strip("_")

            # get actual data for this map
            sys_map = self.read_healsparse(map_path, nside)

            if 'e1' in sys_name:
                print(f"Correlating {sys_name} with g1")
                corr = self.correlate(sys_map, source_g1, source_w)
                outfile = self.save(sys_name, 'g1', corr, output_dir)
                outputs.append(outfile)

            elif 'e2' in sys_name:
                print(f"Correlating {sys_name} with g2")
                corr = self.correlate(sys_map, source_g2, source_w)
                outfile = self.save(sys_name, 'g2', corr, output_dir)
                outputs.append(outfile)
            else:
                print(f"Correlating {sys_name} with convergance and number_density")
                # correlate with kappa and and ngal maps, each with
                # the appropriate weight
                corr = self.correlate(sys_map, ngal, mask)
                outfile = self.save(sys_name, 'number_density', corr, output_dir)
                outputs.append(outfile)

                corr = self.correlate(sys_map, kappa, source_w)
                outfile = self.save(sys_name, 'convergence', corr, output_dir)
                outputs.append(outfile)


        output_dir.write_listing(outputs)

    def correlate(self, sys_map, data_map, weight_map):
        import scipy.stats
        import healpy

        N = self.config["nbin"]
        f = 0.5 * self.config["outlier_fraction"]

        # clean the data
        finite = (
            np.isfinite(sys_map + data_map)
            & (sys_map != healpy.UNSEEN)
            & (data_map != healpy.UNSEEN)
            & (weight_map != healpy.UNSEEN)
        )

        sys_map = sys_map[finite]
        data_map = data_map[finite]
        weight_map = weight_map[finite]

        # Choose bin edges and put pixels in them.
        # Ignore the weights in the percentiles for now
        percentiles = np.linspace(f, 1 - f, N + 1)
        bin_edges = scipy.stats.mstats.mquantiles(sys_map, percentiles)

        # Remove outliers in the systematic
        clip = (sys_map > bin_edges[0]) & (sys_map < bin_edges[-1])
        sys_map = sys_map[clip]
        data_map = data_map[clip]
        weight_map = weight_map[clip]

        # Find the bin each pixel lands in
        bins = np.digitize(sys_map, bin_edges) - 1

        # Get the weighted count in each bin in the
        # systematic parameter
        counts = np.bincount(bins, weights=weight_map)

        # and the weighted mean of the systematic value
        # itself, and of the y and y^2 values
        x = np.bincount(bins, weights=sys_map * weight_map) / counts
        y = np.bincount(bins, weights=data_map * weight_map) / counts
        y2 = np.bincount(bins, weights=data_map ** 2 * weight_map) / counts

        # the error on the mean = var / count
        yerr = np.sqrt((y ** 2 - y2 ** 2) / counts)

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
        filename = output_dir.path_for(base)
        plt.savefig(filename)
        plt.close()

        return base
