from .base_stage import PipelineStage
from .data_types import (
    YamlFile,
    TomographyCatalog,
    HDFFile,
    TextFile,
    FiducialCosmology,
    PhotometryCatalog,
    FitsFile,
)
from .utils import LensNumberDensityStats, Splitter, rename_iterated
from .binning import build_tomographic_classifier, apply_classifier
import numpy as np
import warnings


class TXBaseLensSelector(PipelineStage):
    """
    Base class for lens object selection, using the BOSS Target Selection.

    Subclasses of this pipeline stage select objects to be used
    as the lens sample for the galaxy clustering and
    shear-position calibrations.

    The cut used here is simplistic and should be replaced.
    """

    name = "TXBaseLensSelector"

    outputs = [
        ("lens_tomography_catalog_unweighted", TomographyCatalog),
    ]

    config_options = {
        "verbose": False,
        "chunk_rows": 10000,
        "lens_zbin_edges": [float],
        # Mag cuts
        # Default photometry cuts based on the BOSS Galaxy Target Selection:
        # http://www.sdss3.org/dr9/algorithms/boss_galaxy_ts.php
        "cperp_cut": 0.2,
        "r_cpar_cut": 13.5,
        "r_lo_cut": 16.0,
        "r_hi_cut": 19.6,
        "i_lo_cut": 17.5,
        "i_hi_cut": 19.9,
        "r_i_cut": 2.0,
        "random_seed": 42,
        "selection_type": "boss",
        "maglim_band": "i",
        "maglim_limit": 24.1,

    }

    def run(self):
        """
        Run the analysis for this stage.

         - Collect the list of columns to read
         - Create iterators to read chunks of those columns
         - Loop through chunks:
            - select objects for each bin
            - write them out
            - accumulate selection bias values
         - Average the selection biases
         - Write out biases and close the output
        """
        import astropy.table
        import sklearn.ensemble

        if self.name == "TXBaseLensSelector":
            raise ValueError("Do not run TXBaseLensSelector - run a sub-class")

        # Suppress some warnings from numpy that are not relevant
        original_warning_settings = np.seterr(all="ignore")

        # The output file we will put the tomographic
        # information into
        output_file = self.setup_output()

        iterator = self.data_iterator()

        selector = self.prepare_selector()

        # We will collect the selection biases for each bin
        # as a matrix.  We will collect together the different
        # matrices for each chunk and do a weighted average at the end.
        nbin_lens = len(self.config["lens_zbin_edges"]) - 1

        number_density_stats = LensNumberDensityStats(nbin_lens, self.comm)

        # Loop through the input data, processing it chunk by chunk
        for (start, end, phot_data) in iterator:
            print(f"Process {self.rank} running selection for rows {start:,}-{end:,}")

            pz_data = self.apply_redshift_cut(phot_data, selector)

            # Select lens bin objects
            lens_gals = self.select_lens(phot_data)

            # Combine this selection with size and snr cuts to produce a source selection
            # and calculate the shear bias it would generate
            tomo_bin, counts = self.calculate_tomography(pz_data, phot_data, lens_gals)

            # Save the tomography for this chunk
            self.write_tomography(output_file, start, end, tomo_bin)

            # Accumulate information on the number counts and the selection biases.
            # These will be brought together at the end.
            number_density_stats.add_data(tomo_bin)

        # Do the selection bias averaging and output that too.
        self.write_global_values(output_file, number_density_stats)

        # Save and complete
        output_file.close()

        # Restore the original warning settings in case we are being called from a library
        np.seterr(**original_warning_settings)

    def prepare_selector(self):
        return None

    def apply_redshift_cut(self, phot_data, _):

        pz_data = {}
        nbin = len(self.config["lens_zbin_edges"]) - 1

        z = phot_data[f"z"]
        ntot = len(z)

        zbin = np.repeat(-1, ntot)
        nsel = 0
        nfinite = np.isfinite(z).sum()
        zmean = np.nanmean(z)
        for zi in range(nbin):
            mask_zbin = (z >= self.config["lens_zbin_edges"][zi]) & (
                z < self.config["lens_zbin_edges"][zi + 1]
            )
            nsel += mask_zbin.sum()
            zbin[mask_zbin] = zi
        print(f"Rank {self.rank} found {nsel} / {ntot} galaxies within the selection redshift range, in any bin. {nfinite} galaxies had finite z. Mean z of these was {zmean}")

        pz_data[f"zbin"] = zbin

        return pz_data

    def setup_output(self):
        """
        Set up the output data file.

        Creates the data sets and groups to put module output
        in the tomography_catalog output file.
        """
        n = self.open_input("photometry_catalog")["photometry/ra"].size
        nbin_lens = len(self.config["lens_zbin_edges"]) - 1

        output = self.open_output("lens_tomography_catalog_unweighted", parallel=True, wrapper=True)
        outfile = output.file
        group = outfile.create_group("tomography")
        output.write_zbins(self.config["lens_zbin_edges"])
        group.create_dataset("bin", (n,), dtype="i")
        group.create_dataset("lens_weight", (n,), dtype="f")
        group.create_dataset("counts", (nbin_lens,), dtype="i")
        group.create_dataset("counts_2d", (1,), dtype="i")

        group.attrs["nbin"] = nbin_lens
        group.attrs[f"zbin_edges"] = self.config["lens_zbin_edges"]

        return outfile

    def write_tomography(self, outfile, start, end, lens_bin):
        """
        Write out a chunk of tomography and response.

        Parameters
        ----------
        outfile: h5py.File

        start: int
            The index into the output this chunk starts at

        end: int
            The index into the output this chunk ends at

        tomo_bin: array of shape (nrow,)
            The bin index for each output object

        R: array of shape (nrow,2,2)
            Multiplicative bias calibration factor for each object
        """

        group = outfile["tomography"]
        group["bin"][start:end] = lens_bin
        group["lens_weight"][start:end] = 1.0

    def write_global_values(self, outfile, number_density_stats):
        """
        Write out overall selection biases

        Parameters
        ----------
        outfile: h5py.File
        """
        lens_counts, lens_counts_2d = number_density_stats.collect()

        if self.rank == 0:
            group = outfile["tomography"]
            group["counts"][:] = lens_counts
            group["counts_2d"][:] = lens_counts_2d

    def select_lens(self, phot_data):
        t = self.config["selection_type"]
        if t == "boss":
            s = self.select_lens_boss(phot_data)
        elif t == "maglim":
            s= self.select_lens_maglim(phot_data)
        else:
            raise ValueError(f"Unknown lens selection type {t} - expected boss or maglim")
        ntot = s.size
        nsel = s.sum()
        print(f"Rank {self.rank} selected {nsel} objects out of {ntot} as potential lenses with method {t}")
        return s

    def select_lens_boss(self, phot_data):
        """Photometry cuts based on the BOSS Galaxy Target Selection:
        http://www.sdss3.org/dr9/algorithms/boss_galaxy_ts.php
        """
        mag_i = phot_data["mag_i"]
        mag_r = phot_data["mag_r"]
        mag_g = phot_data["mag_g"]

        # Mag cuts
        cperp_cut_val = self.config["cperp_cut"]
        r_cpar_cut_val = self.config["r_cpar_cut"]
        r_lo_cut_val = self.config["r_lo_cut"]
        r_hi_cut_val = self.config["r_hi_cut"]
        i_lo_cut_val = self.config["i_lo_cut"]
        i_hi_cut_val = self.config["i_hi_cut"]
        r_i_cut_val = self.config["r_i_cut"]

        n = len(mag_i)
        # HDF does not support bools, so we will prepare a binary array
        # where 0 is a lens and 1 is not
        lens_gals = np.repeat(0, n)

        cpar = 0.7 * (mag_g - mag_r) + 1.2 * ((mag_r - mag_i) - 0.18)
        cperp = (mag_r - mag_i) - ((mag_g - mag_r) / 4.0) - 0.18
        dperp = (mag_r - mag_i) - ((mag_g - mag_r) / 8.0)

        # LOWZ
        cperp_cut = np.abs(cperp) < cperp_cut_val  # 0.2
        r_cpar_cut = mag_r < r_cpar_cut_val + cpar / 0.3
        r_lo_cut = mag_r > r_lo_cut_val  # 16.0
        r_hi_cut = mag_r < r_hi_cut_val  # 19.6

        lowz_cut = (cperp_cut) & (r_cpar_cut) & (r_lo_cut) & (r_hi_cut)

        # CMASS
        i_lo_cut = mag_i > i_lo_cut_val  # 17.5
        i_hi_cut = mag_i < i_hi_cut_val  # 19.9
        r_i_cut = (mag_r - mag_i) < r_i_cut_val  # 2.0
        # dperp_cut = dperp > 0.55 # this cut did not return any sources...

        cmass_cut = (i_lo_cut) & (i_hi_cut) & (r_i_cut)

        # If a galaxy is a lens under either LOWZ or CMASS give it a zero
        lens_mask = lowz_cut | cmass_cut
        lens_gals[lens_mask] = 1

        return lens_gals
    
    def select_lens_maglim(self, phot_data):
        band = self.config["maglim_band"]
        limit = self.config["maglim_limit"]
        mag_i = phot_data[f"mag_{band}"]
        s = (mag_i < limit).astype(np.int8)
        return s

    def calculate_tomography(self, pz_data, phot_data, lens_gals):

        nbin = len(self.config["lens_zbin_edges"]) - 1
        n = len(phot_data["mag_i"])

        # The main output data - the tomographic
        # bin index for each object, or -1 for no bin.
        tomo_bin = np.repeat(-1, n)

        # We also keep count of total count of objects in each bin
        counts = np.zeros(nbin, dtype=int)

        for i in range(nbin):
            sel_00 = (pz_data["zbin"] == i) & (lens_gals == 1)
            tomo_bin[sel_00] = i
            counts[i] = sel_00.sum()

        return tomo_bin, counts


class TXTruthLensSelector(TXBaseLensSelector):
    """
    Select lens objects based on true redshifts and BOSS criteria

    This is useful for testing with idealised lens bins.
    """

    name = "TXTruthLensSelector"

    inputs = [
        ("photometry_catalog", PhotometryCatalog),
    ]

    def data_iterator(self):
        print(f"We are cheating and using the true redshift.")
        chunk_rows = self.config["chunk_rows"]
        phot_cols = ["mag_i", "mag_r", "mag_g", "redshift_true"]
        # Input data.  These are iterators - they lazily load chunks
        # of the data one by one later when we do the for loop.
        # This code can be run in parallel, and different processes will
        # each get different chunks of the data
        for s, e, data in self.iterate_hdf(
            "photometry_catalog", "photometry", phot_cols, chunk_rows
        ):
            data["z"] = data["redshift_true"]
            yield s, e, data


class TXMeanLensSelector(TXBaseLensSelector):
    """
    Select lens objects based on mean redshifts and BOSS criteria

    This requires PDFs to have been estimated earlier.
    """

    name = "TXMeanLensSelector"
    inputs = [
        ("photometry_catalog", PhotometryCatalog),
        ("lens_photoz_pdfs", HDFFile),
    ]

    def data_iterator(self):
        chunk_rows = self.config["chunk_rows"]
        phot_cols = ["mag_i", "mag_r", "mag_g"]
        z_cols = ["zmean"]
        rename = {"zmean": "z"}

        it = self.combined_iterators(
            chunk_rows,
            "photometry_catalog",
            "photometry",
            phot_cols,
            "lens_photoz_pdfs",
            "ancil",
            z_cols,
        )

        return rename_iterated(it, rename)


class TXModeLensSelector(TXBaseLensSelector):
    """
    Select lens objects based on best-fit redshifts and BOSS criteria

    This requires PDFs to have been estimated earlier.
    """

    name = "TXModeLensSelector"
    inputs = [
        ("photometry_catalog", PhotometryCatalog),
        ("lens_photoz_pdfs", HDFFile),
    ]

    def data_iterator(self):
        chunk_rows = self.config["chunk_rows"]
        phot_cols = ["mag_i", "mag_r", "mag_g"]
        z_cols = ["zmode"]
        rename = {"zmode": "z"}

        it = self.combined_iterators(
            chunk_rows,
            "photometry_catalog",
            "photometry",
            phot_cols,
            "lens_photoz_pdfs",
            "ancil",
            z_cols,
        )

        return rename_iterated(it, rename)


class TXRandomForestLensSelector(TXBaseLensSelector):
    name = "TXRandomForestLensSelector"
    inputs = [
        ("photometry_catalog", PhotometryCatalog),
        ("calibration_table", TextFile),
    ]
    config_options = TXBaseLensSelector.config_options.copy().update({
        "verbose": False,
        "bands": "ugrizy",
        "chunk_rows": 10000,
        "lens_zbin_edges": [float],
        "random_seed": 42,
    })

    def data_iterator(self):
        chunk_rows = self.config["chunk_rows"]
        phot_cols = ["mag_u", "mag_g", "mag_r", "mag_i", "mag_z", "mag_y"]

        for s, e, data in self.iterate_hdf(
            "photometry_catalog", "photometry", phot_cols, chunk_rows
        ):
            yield s, e, data

    def prepare_selector(self):
        return build_tomographic_classifier(
            self.config["bands"],
            self.get_input("calibration_table"),
            self.config["lens_zbin_edges"],
            self.config["random_seed"],
            self.comm,
        )

    def apply_redshift_cut(self, phot_data, selector):
        classifier, features = selector
        shear_catalog_type = "not applicable"
        bands = self.config["bands"]
        pz_data = apply_classifier(
            classifier, features, bands, shear_catalog_type, phot_data
        )
        return pz_data



class TXLensCatalogSplitter(PipelineStage):
    """
    Split a lens catalog file into a new file with separate bins

    Splitting up like this helps reduce memory usage in TreeCorr later
    """

    name = "TXLensCatalogSplitter"

    inputs = [
        ("lens_tomography_catalog_unweighted", TomographyCatalog),
        ("photometry_catalog", PhotometryCatalog),
        ("fiducial_cosmology", FiducialCosmology),
        ("lens_photoz_pdfs", HDFFile),
    ]

    outputs = [
        ("binned_lens_catalog_unweighted", HDFFile),
    ]

    config_options = {
        "initial_size": 100_000,
        "chunk_rows": 100_000,
        "extra_cols": [""],
        "redshift_column": "zmean",
    }

    def get_lens_tomo_name(self): #can overwrite this in a weighted subclass
        return "lens_tomography_catalog_unweighted"
    def get_binned_lens_name(self): #can overwrite this in a weighted subclass
        return "binned_lens_catalog_unweighted"

    def run(self):

        with self.open_input(self.get_lens_tomo_name()) as f:
            nbin = f["tomography"].attrs["nbin"]
            counts = f["tomography/counts"][:]
            count2d = f["tomography/counts_2d"][:]

        # We also copy over the magnitudes and their errors to tbe new
        # per-bin catalogs. This makes it easier to run photo-z with RAIL
        # on each of the bins. So here we add those columns to our list
        with self.open_input("photometry_catalog", wrapper=True) as f:
            bands = f.get_bands()

        if self.rank == 0:
            print(f"Copying photometry bands {bands} to sub-catalogs")

        band_cols = [f"mag_{b}" for b in bands]
        band_cols += [f"mag_err_{b}" for b in bands]
        self.config["extra_cols"] += band_cols

        # For some reason we briefly ended up with None or empty columns
        # in the configuration.
        extra_cols = [c for c in self.config["extra_cols"] if c]

        # Regular columns.
        cols = ["ra", "dec", "weight", "comoving_distance"]

        # Object we use to make the separate lens bins catalog
        cat_output = self.open_output(self.get_binned_lens_name(), parallel=True)
        cat_group = cat_output.create_group("lens")
        cat_group.attrs["nbin"] = len(counts)
        cat_group.attrs["nbin_lens"] = len(counts)

        bins = {b: c for b, c in enumerate(counts)}
        bins["all"] = count2d
        dtypes = {"id": "i8", "flags": "i8"}
        splitter = Splitter(cat_group, "bin", cols + extra_cols, bins, dtypes=dtypes)

        my_bins = list(self.split_tasks_by_rank(bins))
        if my_bins:
            my_bins_text = ", ".join(str(x) for x in my_bins)
            print(f"Process {self.rank} collating bins: [{my_bins_text}]")
        else:
            print(f"Note: Process {self.rank} will not do anything.")

        for s, e, data in self.data_iterator():
            if self.rank == 0:
                print(f"Process 0 binning data in range {s:,} - {e:,}")

            data["weight"] = data["lens_weight"]
            for b in my_bins:
                if b == "all":
                    w = np.where(data["bin"] >= 0)
                else:
                    w = np.where(data["bin"] == b)
                d = {name: col[w] for name, col in data.items()}
                splitter.write_bin(d, b)

        splitter.finish(my_bins)
        cat_output.close()

    def data_iterator(self):
        z_col = self.config["redshift_column"]
        extra_cols = [c for c in self.config["extra_cols"] if c]

        it = self.combined_iterators(
            self.config["chunk_rows"],
            # first file
            self.get_lens_tomo_name(),
            "tomography",
            ["bin", "lens_weight"],
            # second file
            "photometry_catalog",
            "photometry",
            ["ra", "dec"] + extra_cols,
            "lens_photoz_pdfs",
            "ancil",
            [z_col],
            parallel=False,
        )

        return self.add_redshifts(it)

    def add_redshifts(self, iterator):
        import pyccl
        z_col = self.config["redshift_column"]
        # This iterates through chunks of the input catalogs, but for each
        # chunk it also intercepts and adds the radial distance. So this function
        # returns an iterator, just like combined_iterators does.
        with self.open_input("fiducial_cosmology", wrapper=True) as f:
            cosmo = f.to_ccl()

        for s, e, data in iterator:
            z = data[z_col]
            a = 1.0 / (1 + z)
            d = pyccl.comoving_radial_distance(cosmo, a)
            data["comoving_distance"] = d
            yield s, e, data


class TXTruthLensCatalogSplitter(TXLensCatalogSplitter):
    """
    Split a lens catalog file into a new file with separate bins with true redshifts.

    The redshifts are used to calculate the comoving distances.
    """
    name = "TXTruthLensCatalogSplitter"
    inputs = [
            ("lens_tomography_catalog_unweighted", TomographyCatalog),
            ("photometry_catalog", PhotometryCatalog),
            ("fiducial_cosmology", FiducialCosmology),
        ]
    config_options = TXLensCatalogSplitter.config_options.copy()
    config_options["redshift_column"] = "redshift_true"


    def data_iterator(self):
        z_col = self.config["redshift_column"]
        extra_cols = [c for c in self.config["extra_cols"] if c]

        it = self.combined_iterators(
            self.config["chunk_rows"],
            # first file
            self.get_lens_tomo_name(),
            "tomography",
            ["bin", "lens_weight"],
            # second file
            "photometry_catalog",
            "photometry",
            ["ra", "dec", z_col] + extra_cols,
            parallel=False,
        )

        return self.add_redshifts(it)


class TXExternalLensCatalogSplitter(TXLensCatalogSplitter):
    """
    Split an external lens catalog into bins

    Implemented as a subclass of TXLensCatalogSplitter, and
    changes only file names.
    """
    config_options = TXLensCatalogSplitter.config_options.copy()
    config_options["redshift_column"] = "redshift"

    name = "TXExternalLensCatalogSplitter"
    inputs = [
        ("lens_tomography_catalog_unweighted", TomographyCatalog),
        ("lens_catalog", HDFFile),
        ("fiducial_cosmology", FiducialCosmology),
    ]

    def data_iterator(self):
        z_col = self.config["redshift_column"]
        extra_cols = [
            c for c in self.config["extra_cols"] if c and c != "comoving_distance"
        ]
        iterator = self.combined_iterators(
            self.config["chunk_rows"],
            # first file
            self.get_lens_tomo_name(),
            "tomography",
            ["bin", "lens_weight"],
            # second file
            "lens_catalog",
            "lens",
            ["ra", "dec", z_col] + extra_cols,
            parallel=False,
        )

        return self.add_redshifts(iterator)

class TXTruthLensCatalogSplitterWeighted(TXTruthLensCatalogSplitter):
    """
    Split a lens catalog file into a new file with separate bins with true redshifts.

    The redshifts are used to calculate the comoving distances.
    """
    name = "TXTruthLensCatalogSplitterWeighted"
    inputs = [
            ("lens_tomography_catalog", TomographyCatalog),
            ("photometry_catalog", PhotometryCatalog),
            ("fiducial_cosmology", FiducialCosmology),
        ]
    outputs = [
        ("binned_lens_catalog", HDFFile),
    ]
    def get_lens_tomo_name(self): #can overwrite this in a weighted subclass
        return "lens_tomography_catalog"
    def get_binned_lens_name(self): #can overwrite this in a weighted subclass
        return "binned_lens_catalog"
    
    def data_iterator(self):
        z_col = self.config["redshift_column"]
        extra_cols = [
            c for c in self.config["extra_cols"] if c and c != "comoving_distance"
        ]
        iterator = self.combined_iterators(
            self.config["chunk_rows"],
            # first file
            self.get_lens_tomo_name(),
            "tomography",
            ["bin", "lens_weight"],
            # second file
            "photometry_catalog",
            "photometry",
            ["ra", "dec", z_col] + extra_cols,
            parallel=False,
        )
        
        return self.add_redshifts(iterator)


if __name__ == "__main__":
    PipelineStage.main()
