from txpipe import PipelineStage
from ceci.config import StageParameter
from txpipe.data_types import HDFFile, ParquetFile, FitsFile, QPNOfZFile, DataFile, MapsFile, ShearCatalog
import numpy as np
import tempfile
import tarfile

d2r = np.pi / 180
sq_deg_on_sky = 360**2 / np.pi


# Euclid uses microJanskys with a reference magnitude of 23.9
# different to the LSST convention so we re-do these files
# http://st-dm.pages.euclid-sgs.uk/data-product-doc/dm10/merdpd/merphotometrycookbook.html
REF_MAG = 23.9


def microjansky_to_mag_ab(flux):
    return -2.5 * np.log10(flux) + REF_MAG

def microjansky_err_to_mag_ab(flux, flux_err):
    return 2.5 / np.log(10) * (flux_err / flux)

def parquet_batch_to_astropy_table(batch):
    from astropy.table import Table
    data = {}
    for name, col in zip(batch.schema.names, batch.columns):
        data[name] = np.array(col)
    return Table(data)


class TXIngestEuclidRR2(PipelineStage):
    """
    Ingest the Euclid RR2 shear catalog from Parquet format into a shear catalog
    and mask
    """
    name = "TXIngestEuclidRR2"
    inputs = [
        ("euclid_rr2_parquet", ParquetFile),
        ("euclid_rr2_masks", DataFile),

    ]
    outputs = [
        ("shear_catalog", HDFFile),
        ("mask", MapsFile),
    ]
    config_options = {
        "shear": StageParameter(
            str,
            default="lensmc",
            msg="Which shear data set to use from the catalog, lensmc or metacal",
        )
    }

    def run(self):
        self.ingest_shear()

    def ingest_shear(self):
        import pyarrow.parquet as pq
        from txpipe.utils.splitters import DynamicSplitter
        from txpipe.utils.calibration_tools import MockCalculator
        from parallel_statistics import ParallelMeanVariance, ParallelSum

        bands = [
            "vis_unif",
            "y_unif",
            "j_unif",
            "h_unif",
            "g_ext_decam_unif",
            "r_ext_decam_unif",
            "i_ext_decam_unif",
            "z_ext_decam_unif",
        ]

        input_file, iterator = self.iterate_shear(bands)

        with self.open_output("shear_catalog") as full_output_file:

            full_output_group = full_output_file.create_group("shear")
            
            start = 0
            end = 0
            output_start = 0
            for k, d in enumerate(iterator):

                # Convert the Parquet chunk to a saner object,
                # an astropy table
                d = parquet_batch_to_astropy_table(d)

                # Report progress
                end = start + len(d)
                print(f"Processing chunk {k} rows {start:,} - {end:,}")
                
                full_output = self.process_data(d, bands)
                full_sel = self.select(d)
                    

                selected_full_data = {k:v[full_sel] for k,v in full_output.items()}
                output_end = output_start + selected_full_data['galaxy_id'].shape[0]
                    

                # Write full catalog data
                if k == 0:
                    for col in full_output.keys():
                        full_output_group.create_dataset(col, data=selected_full_data[col], maxshape=(None,))
                        sz = selected_full_data[col].shape[0]
                else:
                    # Double the column sizes if needed
                    if output_end > sz:
                        new_sz = output_end * 2
                        print("Resizing full output columns from", sz, "to", new_sz)
                        for col in selected_full_data.keys():
                            full_output_group[col].resize((new_sz,))
                        sz = new_sz

                    # Dump everything to the full output file
                    print("Writing full output columns from ", output_start, "to", output_end)
                    for name, col in selected_full_data.items():
                        full_output_group[name][output_start:output_end] = col

                start = end
                output_start = output_end

            
            # cut off any unused space in the full output file
            for col in full_output.keys():
                full_output_group[col].resize((end,))

        input_file.close()


    def iterate_shear(self, bands):
        cat = self.config["shear"]

        input_file = self.open_input("euclid_rr2_parquet")

        # The band names in LSST are all single letters, but I don't
        # think we assume that anywhere so here we keep the full names.
        # We will rename things later.
        cols = [p for p in input_file.schema.names if p.startswith(f"she_{cat}")]
        cols += [p for p in input_file.schema.names if "flag" in p]
        cols += ['object_id', 'tom_bin_id', "vis_det", "eff_cov_flag"]

        # Define the bands and columns we want to read. The listing
        # is here: https://cosmohub.pic.es/catalogs/349
        # We will ingest all the she_* columns in case useful, though some will be renamed

        # Everything is stored as fluxes in microJanskys in this file.
        for b in bands:
            cols.append("flux_" + b)
            cols.append("fluxerr_" + b)

        return input_file, input_file.iter_batches(columns=cols)

    def select(self, data):
        n = len(data)
        kind = self.config["shear"]

        d1 = data[f"she_{kind}_weight"] > 0
        d2 = data["vis_det"] == 1
        d3 = data["eff_cov_flag"] > 0
        d12 = d1 & d2
        d123 = d12 & d3
        n1 = d1.sum()
        n2 = d2.sum()
        n3 = d3.sum()
        n12 = d12.sum()
        n123 = d123.sum()
        print(f"Shear weight cut selects {n1/n:.1%} of data")
        print(f"Vis det cut selects {n2/n:.1%} of data or {n12/n1:.1%} of data after previous cuts, leading to total selected fraction {n12/n:.1%}")
        print(f"Eff cov cut selects {n3/n:.1%} of data or {n123/n12:.1%} of data after previous cuts, leading to total selected fraction {n123/n:.1%}")
        return d123


    def process_data(self, batch, bands):
        kind = self.config["shear"]
        renames = {
            "object_id": "galaxy_id",
            f"she_{kind}_ra": "ra",
            f"she_{kind}_dec": "dec",
            f"she_{kind}_e1_corrected": "g1",
            f"she_{kind}_e2_corrected": "g2",
            f"she_{kind}_weight": "weight",
            f"she_{kind}_c1": "c1",
            f"she_{kind}_c2": "c2",
            f"she_{kind}_m11": "m11",
            f"she_{kind}_m12": "m12",
            f"she_{kind}_m21": "m21",
            f"she_{kind}_m22": "m22",
            f"she_{kind}_psf_r2": "psf_T_mean",   
        }

        out = {}
        for in_name, out_name in renames.items():
            out[out_name] = batch[in_name]

        # Copy all the non-renamed ones to the output
        for col in batch.colnames:
            if col not in renames:
                out[col] = batch[col]

        for band in bands:
            flux = batch["flux_" + band]
            flux_err = batch["fluxerr_" + band]
            mag = microjansky_to_mag_ab(flux)
            mag_err = microjansky_err_to_mag_ab(flux, flux_err)
            out[f"mag_{band}"] = mag
            out[f"mag_err_{band}"] = mag_err

        return out





class TXIngestEuclidRR2Binned(TXIngestEuclidRR2):
    """
    Ingest Euclid RR2 shear catalog from Parquet format and produce binned_shear_catalog.

    Expects to find a file 'rr2.parquet' in the data directory.
    Produces a file 'binned_shear_catalog.hdf5' in the output directory.
    """

    name = "TXIngestEuclidRR2Binned"
    inputs = [
        ("euclid_rr2_parquet", ParquetFile),
        ("euclid_rr2_shear_photoz_stack", FitsFile),
        ("euclid_rr2_randoms", DataFile),
        ("euclid_rr2_masks", DataFile),

    ]
    outputs = [
        ("binned_shear_catalog", HDFFile),
        ("shear_photoz_stack", QPNOfZFile),
        ("binned_random_catalog", HDFFile),
        ("random_cats", HDFFile),
        ("tracer_metadata", HDFFile),
        ("mask", MapsFile),
    ]
    config_options = {
        "nbin": StageParameter(
            int,
            default=6,
            msg="Number of tomographic bins - fixed in RR2 pre-selection",
        ),
        "z_spacing": StageParameter(
            float,
            default=0.002,
            msg="Redshift spacing for the photo-z n(z) stacks",
        ),
        "shear": StageParameter(
            str,
            default="lensmc",
            msg="Which shear data set to use from the catalog, lensmc or metacal",
        )
    }


    def run(self):
        import pyarrow.parquet as pq
        import h5py
        import astropy
        import qp

        self.ingest_photoz_stack()
        area_sq_deg = self.ingest_mask()
        self.ingest_binned_random_catalog()
        self.ingest_shear(area_sq_deg)



    def ingest_binned_random_catalog(self):
        import fitsio
        # only the ra and dec are actually referenced for this
        # for the main workflow, although there is a redshift and
        # fiducial distance column too
        input_gzip_file = self.get_input("euclid_rr2_randoms")
        nbin = self.config["nbin"]

        # unzip the file to a temporary directory
        with tempfile.TemporaryDirectory() as tmpdirname:
            with tarfile.open(input_gzip_file, "r:gz") as tar:
                tar.extractall(path=tmpdirname)
            print("Unzipped random catalog to temporary directory", tmpdirname)

            ras = []
            decs = []
            # Now we can copy all this to the output file
            with self.open_output("binned_random_catalog", wrapper=False) as binned_output:
                binned_output.attrs["nbin_source"] = nbin
                # If the data files stay in the same format but get
                # much larger we might need to do this in a streaming
                # way rather than all in memory at once.
                # The input bins are 1-based.
                for i in range(1, nbin + 1):
                    input_file = f"{tmpdirname}/random_tombin{i}.fits"
                    with fitsio.FITS(input_file) as f:
                        ra = f[1].read_column("RIGHT_ASCENSION")
                        dec = f[1].read_column("DECLINATION")
                    print(f"Read {len(ra)} from random bin {i}")
                    # Write to output file - using 0-based bin numbering
                    group = binned_output.create_group(f"bin_{i-1}")
                    group.create_dataset("ra", data=ra)
                    group.create_dataset("dec", data=dec)
                    ras.append(ra)
                    decs.append(dec)

        with self.open_output("random_cats", wrapper=False) as full_output:
            group = full_output.create_group("randoms")
            size = sum(len(r) for r in ras)
            group.create_dataset("bin", dtype=np.int16, shape=(size,))
            group.create_dataset("ra", dtype=np.float64, shape=(size,))
            group.create_dataset("dec", dtype=np.float64, shape=(size,))
            start = 0
            for i in range(nbin):
                end = start + len(ras[i])
                group["bin"][start:end] = i
                group["ra"][start:end] = ras[i]
                group["dec"][start:end] = decs[i]
                start = end



    def ingest_photoz_stack(self):
        import qp

        # Read from the input fits file using fitsio
        with self.open_input("euclid_rr2_shear_photoz_stack") as f:
            data = f[1].read()

        # The z spacing is 0.002 based on matching the mean z values from the file.
        # This leads to a very spiky n(z)
        dz = self.config["z_spacing"]

        # The nbin here should match the nbin in the configuration
        nbin = len(data)
        assert nbin == self.config["nbin"]

        # Space for the output pdfs etc.
        nsample = len(data[0]['N_Z'])
        z = np.arange(nsample) * dz
        pdfs = np.zeros((nbin+1, nsample))
        total_nz = 0

        for i, tomo_bin in enumerate(data):
            nz = tomo_bin['N_Z']

            # Accumulate total n(z). The normalization of the individual
            # tomographic bins is the number of objects in it, so we can
            # do this and then normalize at the end.
            total_nz += nz

            # But we normalize the individual n(z) to unity.
            norm = nz.sum() * dz
            nz /= norm

            pdfs[i] = nz

        # Normalize and set the overall non-tomographic n(z)
        total_nz /= total_nz.sum() * dz
        pdfs[-1] = total_nz

        # for now we just propagate this high resolution n(z)
        # into the output file. Later we may want to downsample it.
        f.close()

        # Create qp ensemble and write to output file
        q = qp.Ensemble(qp.interp, data={"xvals":z, "yvals":pdfs})
        with self.open_output("shear_photoz_stack", "w") as f:
            f.write_ensemble(q)

    def ingest_shear(self, area_sq_deg):
        import pyarrow.parquet as pq
        from txpipe.utils.splitters import DynamicSplitter
        from txpipe.utils.calibration_tools import MockCalculator
        from parallel_statistics import ParallelMeanVariance, ParallelSum


        bands = [
            "vis_unif",
            "y_unif",
            "j_unif",
            "h_unif",
            "g_ext_decam_unif",
            "r_ext_decam_unif",
            "i_ext_decam_unif",
            "z_ext_decam_unif",
        ]

        nbin = self.config["nbin"]

        input_file, iterator = self.iterate_shear(bands)


        # Initial bin sizes - not particularly important, the columns
        # will be auto-resized by the splitter as needed
        # The input data file is 1-based, but TXPipe expects
        # 0-based bin numbering later on.
        bin_sizes = {b: 100_000 for b in range(nbin)}
        bin_sizes["all"] = 100_000 * nbin

        # We will create this once we have collected the first chunk
        # of data so we don't have to specify twice what the columns are.
        splitter = None


        g1_stats = ParallelMeanVariance(nbin)
        g2_stats = ParallelMeanVariance(nbin)
        weight_stats = ParallelSum(nbin)
        weight2_stats = ParallelSum(nbin)
        counts = np.zeros(nbin, dtype=int)

        with self.open_output("binned_shear_catalog") as output_file:
            output_group = output_file.create_group("shear")
            output_group.attrs["nbin_source"] = nbin
            
            start = 0
            end = 0
            for k, d in enumerate(iterator):

                # Convert the Parquet chunk to a saner object,
                # an astropy table
                d = parquet_batch_to_astropy_table(d)

                # Report progress
                end = start + len(d)
                print(f"Processing chunk {k} rows {start:,} - {end:,}")
                start = end
                

                # Loop through the tomographic bins using the 1-based
                # indexing from the input file
                for i in range(1, nbin + 1):
                    # Use the pre-defined tomographic bin ids
                    # given in RR2
                    sel = self.select(d, i)
                    selected_data = d[sel]
                    
                    # Do all the renaming and flux to mag conversion
                    # to get things in TXPipe format
                    output_data = self.process_data(selected_data, bands)

                    # We can only create the splitter once we know
                    # what all the columns will be, so do that on the first iteration
                    if splitter is None:
                        columns = list(output_data.keys())
                        splitter = DynamicSplitter(
                            output_group, "bin", columns, bin_sizes
                        )
                    # Write the data for this tomographic bin using
                    # the zero-based bin index we want in the output
                    splitter.write_bin(output_data, i - 1)
                    # The "all" bin gets everything
                    splitter.write_bin(output_data, "all")

                    # Collect statistics for metadata - we need the means and variances
                    # of the shear and the effective number density
                    g1_stats.add_data(i - 1, output_data["g1"], output_data["weight"])
                    g2_stats.add_data(i - 1, output_data["g2"], output_data["weight"])
                    weight_stats.add_data(i - 1, output_data["weight"])
                    weight2_stats.add_data(i - 1, output_data["weight"]**2)
                    counts[i - 1] += len(selected_data)
                

            splitter.finish()

        input_file.close()

        self.save_metadata(g1_stats, g2_stats, weight_stats, weight2_stats, counts, area_sq_deg)

    def save_metadata(self, g1_stats, g2_stats, weight_stats, weight2_stats, counts, area_sq_deg):
        _, g1_means, g1_vars = g1_stats.collect()
        _, g2_means, g2_vars = g2_stats.collect()
        _, weight_sum = weight_stats.collect()
        _, weight2_sum = weight2_stats.collect()
        neff = weight_sum **2 / weight2_sum
    
        sigma_e = np.sqrt(0.5 * (g1_vars + g2_vars))

        print("Saving tracer metadata:")
        print("Counts per bin:", counts)
        print("N_eff per bin:", neff)
        print("Sigma_e per bin:", sigma_e)
        print("Mean e1 per bin:", g1_means)
        print("Mean e2 per bin:", g2_means)


        with self.open_output("tracer_metadata", wrapper=False) as meta_file:
            group = meta_file.create_group("tracers")
            group.attrs["area"] = area_sq_deg
            group.attrs["area_unit"] = "deg^2"
            group.create_dataset("sigma_e", data=sigma_e)
            group.create_dataset("N_eff", data=neff)
            group.create_dataset("mean_e1", data=g1_means)
            group.create_dataset("mean_e2", data=g2_means)
            group.create_dataset("counts", data=counts)
            
    def process_data(self, batch, bands):
        out = super().process_data(batch, bands)
        # remove all the she_ columns that we don't need
        out = {k:v for k,v in out.items() if not k.startswith("she_")}
        return out

    def select(self, data, tomo_bin):
        d0 = super().select(data)
        n = len(data)
        n0 = d0.sum()

        d1 = data[f"tom_bin_id"] == tomo_bin
        n1 = d1.sum()

        sel = d1 & d0
        nf = sel.sum()
        f = nf / n0
        print(f"Tomo cut for bin {tomo_bin} selects {n1/n:.1%} of data, or {f:.1%} of data selected by previous cuts, for total selected fraction {nf/n:.1%}")
        print("")
        return sel

    def ingest_mask(self):
        import healpy
        mask_gzip_file = self.get_input("euclid_rr2_masks")
        nbin = self.config["nbin"]

        # unzip the file to a temporary directory
        masks = []
        with tempfile.TemporaryDirectory() as tmpdirname:
            with tarfile.open(mask_gzip_file, "r:gz") as tar:
                tar.extractall(path=tmpdirname)

            print("Unzipped masks to temporary directory", tmpdirname)
            for i in range(1, nbin + 1):
                mask = healpy.read_map(f"{tmpdirname}/visibility_tombinid_{i}.fits")
                npix = len(mask)
                nside = healpy.npix2nside(npix)
                pix = np.where(mask > 0)[0]
                values = mask[pix]
                masks.append((pix, values))
                print("Read mask for bin", i, "with", len(pix), "pixels with coverage")
        

        mask_union = np.zeros(healpy.nside2npix(nside))
        for pix, _ in masks:
            print("Mask has", len(pix), "pixels with coverage")
            mask_union[pix] = 1

        area_sq_deg = healpy.nside2pixarea(nside, degrees=True) * np.sum(mask_union)
        print("Overall sky area = ", area_sq_deg, "sq deg")

        metadata = {"pixelization": "healpix", "nside": nside, "nest": False}
        with self.open_output("mask", wrapper=True) as f:
            f.file.create_group("maps")
            f.file["maps"].attrs["nbin"] = nbin
            for i, (pix, mask) in enumerate(masks):
                name = f"mask_{i}"
                f.write_map(name, pix, mask, metadata)

            metadata["area_unit"] = "deg^2"
            metadata["area"] = area_sq_deg
            metadata["f_sky"] = area_sq_deg / sq_deg_on_sky

            mask_union_pix = np.where(mask_union > 0)[0]
            f.write_map("mask", mask_union_pix, mask_union[mask_union_pix], metadata)

        
        return area_sq_deg

if __name__ == "__main__":
    PipelineStage.main()
