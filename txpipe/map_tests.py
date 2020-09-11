from .base_stage import PipelineStage
from .data_types import MapsFile, FileCollection
import pathlib
import glob
import numpy as np

class TXMapCorrelations(PipelineStage):
    name = "TXMapCorrelations"
    inputs = [
        ("density_maps", MapsFile),
        ("convergence_maps", MapsFile),
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
        for map_input in ["density_maps", "convergence_maps"]:
            n = 0
            with self.open_input(map_input, wrapper=True) as map_file:
                for name in map_file.list_maps():
                    # load map and pix info
                    ns = map_file.read_map_info(name)["nside"]
                    if nside == 0:
                        nside = ns
                    else:
                        if nside != ns:
                            raise ValueError("Cannot correlate maps of varying nside")
                    data_maps[name] = map_file.read_map(name)
                    n += 1
            print(f"Found {n} maps in {map_input}")

        ndata = len(data_maps)
        print(f"=> Found {ndata} total maps")
        output_dir = self.open_output("map_systematic_correlations", wrapper=True)

        # use the name of the last map we read to find the nside
        # of them all
        

        root = self.config["supreme_path_root"]
        sys_maps = glob.glob(f"{root}*.hs")
        nsys = len(sys_maps)
        print(f"Found {nsys} total systematic maps")
        print(f"Generating {nsys * ndata} total correlations")
        for map_path in sys_maps:
            # strip root, .hs, and underscores to get friendly name
            sys_name = map_path[len(root):-3].strip("_")

            # get actual data for this map
            sys_map = self.read_healsparse(map_path, nside)

            # Compute the correlation with each of our data
            # maps and save
            for data_name, data_map in data_maps.items():
                print(f"\nCorrelating {sys_name} x {data_name}")
                corr = self.correlate(sys_map, data_map)
                self.save(sys_name, data_name, corr, output_dir)

    def correlate(self, sys_map, data_map):
        import scipy.stats
        import healpy

        N = self.config["nbin"]
        f = 0.5 * self.config["outlier_fraction"]
        # clean data
        finite = (
            np.isfinite(sys_map + data_map)
            & (sys_map != healpy.UNSEEN)
            & (data_map != healpy.UNSEEN)
        )
        sys_map = sys_map[finite]
        data_map = data_map[finite]
        print(f"Using {sys_map.size} finite pixels")

        # Choose bin edges and put pixels in them
        percentiles = np.linspace(f, 1 - f, N + 1)
        bin_edges = scipy.stats.mstats.mquantiles(sys_map, percentiles)
        clip = (sys_map > bin_edges[0]) & (sys_map < bin_edges[-1])
        sys_map = sys_map[clip]
        data_map = data_map[clip]
        bins = np.digitize(sys_map, bin_edges) - 1

        # Get stats on those pixels
        counts = np.bincount(bins)
        x = np.bincount(bins, weights=sys_map) / counts
        y = np.bincount(bins, weights=data_map) / counts
        y2 = np.bincount(bins, weights=data_map ** 2) / counts
        yerr = np.sqrt((y ** 2 - y2 ** 2) / counts)
        print(counts)
        # return things we want to plit
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
        filename = output_dir.path_for(f"{sys_name}_{data_name}.png")
        plt.savefig(filename)
        plt.close()
