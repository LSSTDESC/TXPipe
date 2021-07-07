import numpy as np
from ...base_stage import PipelineStage
from ...data_types import HDFFile, MapsFile
from ...utils import choose_pixelization


class CMRandoms(PipelineStage):
    name = "CMRandoms"
    inputs = [
        ("cluster_mag_halo_catalog", HDFFile),
        ("cluster_mag_footprint", MapsFile),
    ]
    outputs = [("random_cats", HDFFile)]
    config_options = {"density": 30.0}  #  per sq arcmin

    def run(self):
        import scipy.stats
        from ...randoms import random_points_in_quadrilateral
        import healpy

        # open the footprint catalog and read in the things we need
        # from it - the list of hit pixels, and the map scheme
        with self.open_input("cluster_mag_footprint", wrapper=True) as f:
            hit_map = f.read_map("footprint")
            info = f.read_map_info("footprint")
            nside = info["nside"]
            scheme = choose_pixelization(**info)

        if self.rank == 0:
            print(f"Generating randoms at nside = {nside}")

        #  Hit pixels and the coordinates of their vertices
        pix = np.where(hit_map > 0)[0]
        vertices = scheme.vertices(pix)

        # Randomly select the number of objects per pixel, using a poisson distribution with
        #  the specified mean density, for now
        area = scheme.pixel_area(degrees=True) * 60.0 * 60.0
        density = np.repeat(self.config["density"], pix.size)
        counts = scipy.stats.poisson.rvs(density * area, 1)

        # Use the same counts for all the processors
        if self.comm is not None:
            counts = self.comm.bcast(counts)

        # total number of objects to be generated over all the pixels
        total_count = counts.sum()

        # The starting index of each pixel in the array, so we can parallelize
        starts = np.concatenate([[0], np.cumsum(counts)])

        # Open the output data and create the necessary columns
        output_file = self.open_output("random_cats", parallel=True)
        group = output_file.create_group("randoms")
        ra_out = group.create_dataset("ra", (total_count,), dtype=np.float64)
        dec_out = group.create_dataset("dec", (total_count,), dtype=np.float64)

        # Each processor now does a subset of the pixels, generating and saving
        #  points in each
        for i, vertex in self.split_tasks_by_rank(enumerate(vertices)):
            if (i % (10_000 * self.size)) == self.rank:
                print(
                    f"Rank {self.rank} done {i//self.size:,} of its {counts.size//self.size:,} pixels"
                )
            # Generate random points in this pixel
            p1, p2, p3, p4 = vertex.T
            N = counts[i]
            P = random_points_in_quadrilateral(p1, p2, p3, p4, N)

            # This is not healpy-specific so we just use it as a convenience function
            ra, dec = healpy.vec2ang(P, lonlat=True)

            # Write output
            s = starts[i]
            e = starts[i + 1]
            ra_out[s:e] = ra
            dec_out[s:e] = dec

        output_file.close()


