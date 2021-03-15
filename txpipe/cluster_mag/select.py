import numpy as np
import itertools
from ..base_stage import PipelineStage
from ..data_types import HDFFile, MapsFile
from ..utils import DynamicSplitter

class CMSelectHalos(PipelineStage):
    name = "CMSelectHalos"
    parallel = False
    inputs = [("cluster_mag_halo_catalog", HDFFile)]
    outputs = [("cluster_mag_halo_tomography", HDFFile)]
    config_options = {
        "zedge": [0.2, 0.4, 0.6, 0.8, 1.0, 1.2],
        "medge": [20, 30, 45, 70, 120, 220]

    }
    def run(self):
        initial_size = 100_000
        chunk_rows = 100_000
        
        zedge = np.array(self.config['zedge'])
        # where does this number 45 come from?
        medge = np.array(self.config['medge']) * (1e14 / 45)

        nz = len(zedge) - 1
        nm = len(medge) - 1

        # add infinities to either end to catch objects that spill out
        zedge = np.concatenate([[-np.inf], zedge, [np.inf]])
        medge = np.concatenate([[-np.inf], medge, [np.inf]])

        # all pairs of z bin, m bin indices
        bins = list(itertools.product(range(nz), range(nm)))
        bin_names = {f"{i}_{j}":initial_size for i,j in bins}

        # my_bins = [i, pair for pair in self.split_tasks_by_rank(enumerate(bins))]
        cols = ["halo_mass", "redshift", "ra", "dec"]
        it = self.iterate_hdf("cluster_mag_halo_catalog", "halos", cols, chunk_rows)

        f = self.open_output("cluster_mag_halo_tomography")
        g = f.create_group("tomography")
        g.attrs['nm'] = nm
        g.attrs['nz'] = nz
        splitter = DynamicSplitter(g, "bin", cols, bin_names)

        for _, _, data in it:
            n = len(data["redshift"])
            zbin = np.digitize(data['redshift'], zedge)
            mbin = np.digitize(data['halo_mass'], medge)

            # Find which bin each object is in, or None
            for zi in range(1, nz + 1):
                for mi in range(1, nm + 1):
                    w = np.where((zbin == zi) & (mbin == mi))
                    if w[0].size == 0:
                        continue
                    d = {name:col[w] for name, col in data.items()}
                    splitter.write_bin(d, f"{zi - 1}_{mi - 1}")

        # Truncate arrays to correct size
        splitter.finish()

        # Save metadata
        for (i, j), name in zip(bins, bin_names):
            metadata = splitter.subgroups[name].attrs
            metadata['mass_min'] = medge[i+1]
            metadata['mass_max'] = medge[i+2]
            metadata['z_min'] = zedge[i+1]
            metadata['z_max'] = zedge[i+2]

        f.close()



class CMBackgroundSelector(PipelineStage):
    name = "CMBackgroundSelector"
    parallel = False
    inputs = [("photometry_catalog", HDFFile)]
    outputs = [("cluster_mag_background", HDFFile), ("cluster_mag_footprint", MapsFile)]

    # Default configuration settings
    config_options = {
        "ra_range": [50.0, 73.1],
        "dec_range": [-45.0, -27.0],
        "mag_cut": 26.0,
        "zmin": 1.5,
        "nside": 2048,
        "initial_size": 100_000,
        "chunk_rows": 100_000,
    }

    def run(self):
        import healpy

        #  Count the max number of objects we will look at in total
        with self.open_input("photometry_catalog") as f:
            N = f["photometry/ra"].size

        #  Open and set up the columns in the output
        f = self.open_output("cluster_mag_background")
        g = f.create_group("sample")
        sz = self.config["initial_size"]
        ra = g.create_dataset("ra", (sz,), maxshape=(None,))
        dec = g.create_dataset("dec", (sz,), maxshape=(None,))

        # Get values from the user configutation.
        # These can be set on the command line, in the config file,
        #  or use the default values above
        ra_min, ra_max = self.config["ra_range"]
        dec_min, dec_max = self.config["dec_range"]
        mag_cut = self.config["mag_cut"]
        zmin = self.config["zmin"]
        nside = self.config["nside"]
        chunk_rows = self.config["chunk_rows"]

        # We will keep track of a hit map to help us build the
        #  random catalog later.  Make it zero now; every time a pixel
        #  hits it we will set a value to one
        npix = healpy.nside2npix(nside)
        hit_map = np.zeros(npix)

        # Prepare an iterator that will loop through the data
        it = self.iterate_hdf(
            "photometry_catalog",
            "photometry",
            ["ra", "dec", "mag_i", "redshift_true"],
            chunk_rows,
        )

        s = 0
        # Loop through the data.  The indices s1, e1 refer to the full catalog
        # start/end, that we are selecting from. The indices s and e refer to
        #  the data we have selected
        for s1, e1, data in it:
            #  make selection
            sel = (
                (data["ra"] > ra_min)
                & (data["ra"] < ra_max)
                & (data["dec"] > dec_min)
                & (data["dec"] < dec_max)
                & (data["mag_i"] < mag_cut)
                & (data["redshift_true"] > 1.5)
            )

            #  Pull out the chunk of data we would like to select
            ra_sel = data["ra"][sel]
            dec_sel = data["dec"][sel]

            # Mark any hit pixels
            pix = healpy.ang2pix(nside, ra_sel, dec_sel, lonlat=True)
            hit_map[pix] = 1

            #  Number of selected objects
            n = ra_sel.size

            # Print out our progress
            frac = n / (e1 - s1)
            print(
                f"Read data chunk {s1:,} - {e1:,} and selected {n:,} objects ({frac:.1%})"
            )

            e = s + n
            if e > sz:
                print(f"Resizing output to {sz}")
                sz = int(1.5 * e)
                ra.resize((sz,))
                dec.resize((sz,))

            # write output.
            ra[s:e] = ra_sel
            dec[s:e] = dec_sel

            # update start for next point
            s = e

        # Chop off any unused space
        print(f"Final catalog size {e:,}")
        ra.resize((e,))
        dec.resize((e,))
        f.close()

        #  Save the footprint map.  Select all the non-zero pixels
        pix = np.where(hit_map > 0)[0]
        # We just want a binary map for now, but can upgrade this to
        #  a depth map later
        val = np.ones(pix.size, dtype=np.int8)
        metadata = {"pixelization": "healpix", "nside": nside}

        with self.open_output("cluster_mag_footprint", wrapper=True) as f:
            f.write_map("footprint", pix, val, metadata)


