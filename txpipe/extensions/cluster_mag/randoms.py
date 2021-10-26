import sys
import numpy as np
from ...base_stage import PipelineStage
from ...data_types import HDFFile, MapsFile
from ...utils import choose_pixelization
from .buffer import Buffer

class CMRandoms(PipelineStage):
    name = "CMRandoms"
    inputs = [
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
            self.comm.Bcast(counts)

        # total number of objects to be generated over all the pixels
        total_count = counts.sum()
        if self.rank == 0:
            print(f"Generating a total of {total_count:,} randoms")

        # The starting index of each pixel in the array, so we can parallelize
        starts = np.concatenate([[0], np.cumsum(counts)])

        # Open the output data and create the necessary columns
        output_file = self.open_output("random_cats", parallel=True)
        group = output_file.create_group("randoms")
        ra_out = group.create_dataset("ra", (total_count,), dtype=np.float64)
        dec_out = group.create_dataset("dec", (total_count,), dtype=np.float64)

        # Each processor now does a subset of the pixels, generating and saving
        #  points in each
        my_indices = np.array_split(np.arange(len(vertices)), self.size)[self.rank]
        my_vertices = vertices[my_indices]
        my_start = starts[my_indices[0]]

        ra_buf = Buffer(1_000_000, ra_out, my_start)
        dec_buf = Buffer(1_000_000, dec_out, my_start)
        for j, (i, vertex) in enumerate(zip(my_indices, my_vertices)):
            if j % 10_000 == 0:
                print(
                    f"Rank {self.rank} done {j:,} of its {my_indices.size:,} pixels"
                )
                sys.stdout.flush()
            # Generate random points in this pixel
            p1, p2, p3, p4 = vertex.T
            N = counts[i]
            P = random_points_in_quadrilateral(p1, p2, p3, p4, N)

            # This is not healpy-specific so we just use it as a convenience function
            ra, dec = healpy.vec2ang(P, lonlat=True)

            # Buffer output for writing. We are writing contiguous chunks of data
            # so shouldn't need output sizes
            ra_buf.append(ra, verbose=True)
            dec_buf.append(dec)
        # Write any residual stuff
        ra_buf.write(verbose=True)
        dec_buf.write()
        del ra_buf, dec_buf

        output_file.close()


