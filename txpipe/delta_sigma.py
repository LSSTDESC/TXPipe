from .twopoint import TXTwoPoint, TREECORR_CONFIG, SHEAR_POS
from .data_types import SACCFile, ShearCatalog, HDFFile, QPNOfZFile, TextFile
import numpy as np
from ceci.config import StageParameter
import os


class TXDeltaSigma(TXTwoPoint):
    """Delta Sigma two-point correlation function class.

    This class handles the computation and storage of the Delta Sigma two-point correlation function.
    """

    name = "TXDeltaSigma"

    inputs = [
        ("binned_shear_catalog", ShearCatalog),
        ("binned_lens_catalog", HDFFile),
        ("binned_random_catalog", HDFFile),
        ("shear_photoz_stack", QPNOfZFile),
        ("lens_photoz_stack", QPNOfZFile),
        ("tracer_metadata", HDFFile),
    ]
    outputs = [("delta_sigma", SACCFile)]

    config_options = TREECORR_CONFIG | {
        "source_bins": StageParameter(list, [-1], msg="List of source bins to use (-1 means all)"),
        "lens_bins": StageParameter(list, [-1], msg="List of lens bins to use (-1 means all)"),
        "var_method": StageParameter(str, "jackknife", msg="Method for computing variance (jackknife, sample, etc.)"),
        "use_randoms": StageParameter(bool, True, msg="Whether to use random catalogs"),
        "low_mem": StageParameter(bool, False, msg="Whether to use low memory mode"),
        "patch_dir": StageParameter(str, "./cache/delta-sigma-patches", msg="Directory for storing patch files"),
        "chunk_rows": StageParameter(int, 100_000, msg="Number of rows to process in each chunk"),
        "share_patch_files": StageParameter(bool, False, msg="Whether to share patch files across processes"),
        "metric": StageParameter(str, "Euclidean", msg="Distance metric to use (Euclidean, Arc, etc.)"),
        "gaussian_sims_factor": StageParameter(
            list,
            default=[1.0],
            msg="Factor by which to decrease lens density to account for increased density contrast.",
        ),
        "use_subsampled_randoms": StageParameter(bool, True, msg="Use subsampled randoms file for RR calculation"),
        "redshift_slice_size": StageParameter(float, 0.01, msg="Size of redshift slices for re-weighting")
    }



    def read_nbin(self):
        """
        For Delta Sigma we are splitting both the source and lens bins into
        much smaller slices so that we can re-weight and combine later. We
        first use the parent class methods to read the actual number of tomographic
        bins, before splitting each of those up into tomographic slices.

        So the total number of source bins will be:
            nbin_source_total = nslice * nbin_source_tomography
        and similarly for the lens bins.
        """
        if self.config["source_bins"] == [-1] and self.config["lens_bins"] == [-1]:
            source_list, lens_list = self._read_nbin_from_tomography()
        else:
            source_list, lens_list = self._read_nbin_from_config()

        zmin = np.inf
        zmax = -np.inf
        with self.open_input("shear_photoz_stack", wrapper=True) as f:
            for i in source_list:
                z, Nz = f.get_bin_n_of_z(i)
                zmin = min(zmin, z[np.where(Nz > 0)].min())
                zmax = max(zmax, z[np.where(Nz > 0)].max())
        with self.open_input("lens_photoz_stack", wrapper=True) as f:
            for i in lens_list:
                z, Nz = f.get_bin_n_of_z(i)
                zmin = min(zmin, z[np.where(Nz > 0)].min())
                zmax = max(zmax, z[np.where(Nz > 0)].max())

        # This is the splitting used for the narrow redshift slices
        self.zmin = zmin
        self.zmax = zmax
        self.zbins = np.arange(zmin, zmax + self.config["redshift_slice_size"]*0.1, self.config["redshift_slice_size"])
        self.nslice = len(self.zbins) - 1
        nbin_source_total = self.nslice * len(source_list)
        nbin_lens_total = self.nslice * len(lens_list)
        print(lens_list, source_list, nbin_lens_total, nbin_source_total)
        return np.arange(nbin_source_total), np.arange(nbin_lens_total)

    def index_to_tomograpic_bin_and_slice(self, index):
        """
        Given a bin index (for either source or lens), return the
        tomographic bin and slice index.

        Parameters
        ----------
        index : int
            The overall bin index.
        nbin : int
            The number of tomographic bins (not including slices).

        Returns
        -------
        tomo_bin : int
            The tomographic bin index.
        slice_index : int
            The slice index within the tomographic bin.
        """
        tomo_bin = index // self.nslice
        slice_index = index % self.nslice
        return tomo_bin, slice_index


    def prepare_patches(self, calcs, meta):
        # We are going to slightly abuse the parent class prepare_patches
        # here. The individual redshift slices are too small to usefully use
        # patches, so a "patch" will actually be the full catalog for a given
        # slice-bin-kind combination (where kind=lens/source, bin=tomo-bin).
        source_bins = set()
        lens_bins = set()
        for i, j, k in calcs:
            assert k == SHEAR_POS, "Delta Sigma only supports shear-position correlations"
            source_bins.add(i)
            lens_bins.add(j)

        self.split_cat_by_redshift("source")
        self.split_cat_by_redshift("lens")
        self.split_cat_by_redshift("randoms")

    def get_slice_filename(self, kind, tomo_bin, slice_index):
        output_dir = self.config["patch_dir"]
        filename = f"{output_dir}/{kind}_{tomo_bin}_slice_{slice_index}.hdf5"
        return filename

    def split_cat_by_redshift(self, kind):
        import h5py

        # Open the correct file depending on the kind of thing we are splitting
        if kind == "source":
            input_cat = self.open_input("binned_shear_catalog")
            group = input_cat["shear"]
        elif kind == "lens":
            input_cat = self.open_input("binned_lens_catalog")
            group = input_cat["lens"]
        elif kind == "randoms":
            input_cat = self.open_input("binned_random_catalog")
            group = input_cat["randoms"]
        else:
            raise ValueError(f"Unknown kind {kind} for splitting catalog by redshift")

        nbin = group.attrs["nbin"]
        # The maximum bin is the 2D "all" bin.
        bins = [i for i in range(nbin) if i in bins_to_use]
        print(bins_to_use)

        # Split the tomographic bins by redshift slices
        for i in self.split_tasks_by_rank(bins):
            bin_name = f"bin_{i}"
            subgroup = group[bin_name]
            z = subgroup["z"][:]
            
            # If we are in low memory mode, we read columns on demand
            # otherwise we read everything into memory at the start
            if self.config["low_mem"]:
                get_col = lambda k: subgroup[k][:]
            else:
                cache = {}
                for k in subgroup.keys():
                    cache[k] = subgroup[k][:]
                get_col = lambda k: cache[k]

            # Loop through the narrow redshift slices
            for slice_index in range(self.nslice):
                zmin = self.zbins[slice_index]
                zmax = self.zbins[slice_index + 1]

                # select the part of the catalog in this redshift slice
                mask = (z >= zmin) & (z < zmax)
                out_path = self.get_slice_filename(kind, bin_name, slice_index)
                with h5py.File(out_path, "w") as f_out:
                    for key in subgroup.keys():
                        data = get_col(key)
                        f_out.create_dataset(key, data=data[mask])

        input_cat.close()

    def touch_patches(self, cat):
        pass

    def select_calculations(self, source_list, lens_list):
        calcs = []

        # For shear-position we use all pairs
        k = SHEAR_POS
        for i in source_list:
            for j in lens_list:
                calcs.append((i, j, k))
        
        return calcs


    def get_shear_catalog(self, i):
        import treecorr
        tomo_bin, slice_index = self.index_to_tomograpic_bin_and_slice(i)

        filename = self.get_slice_filename("source", tomo_bin, slice_index)
        # Load and calibrate the appropriate bin data
        cat = treecorr.Catalog(
            filename,
            g1_col="g1",
            g2_col="g2",
            ra_col="ra",
            dec_col="dec",
            w_col="weight",
            ra_units="degree",
            dec_units="degree",
            flip_g1=self.config["flip_g1"],
            flip_g2=self.config["flip_g2"],
        )

        return cat

    def get_lens_catalog(self, i):
        import treecorr
        tomo_bin, slice_index = self.index_to_tomograpic_bin_and_slice(i)

        filename = self.get_slice_filename("lens", tomo_bin, slice_index)
        # Load and calibrate the appropriate bin data
        cat = treecorr.Catalog(
            filename,
            ra_col="ra",
            dec_col="dec",
            w_col="weight",
            ra_units="degree",
            dec_units="degree",
        )

        return cat


    def get_random_catalog(self, i):
        import treecorr
        tomo_bin, slice_index = self.index_to_tomograpic_bin_and_slice(i)

        filename = self.get_slice_filename("randoms", tomo_bin, slice_index)
        # Load and calibrate the appropriate bin data
        cat = treecorr.Catalog(
            filename,
            ra_col="ra",
            dec_col="dec",
            w_col="weight",
            ra_units="degree",
            dec_units="degree",
        )

        return cat


    def write_output(self, source_list, lens_list, meta, results):
        print("I honestly did not think we'd get this far...")
        breakpoint()
