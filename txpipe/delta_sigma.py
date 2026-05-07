from .twopoint import TXTwoPoint, TREECORR_CONFIG, SHEAR_POS
from .base_stage import PipelineStage
from .data_types import SACCFile, ShearCatalog, HDFFile, QPNOfZFile, FiducialCosmology, TextFile, PNGFile
import numpy as np
from ceci.config import StageParameter
import os


class TXDeltaSigma(TXTwoPoint):
    """Compute Delta-Sigma, the excess surface density around lenses.
    This version uses the dSigma code.
    """

    name = "TXDeltaSigma"

    inputs = [
        ("binned_shear_catalog", ShearCatalog),
        ("binned_lens_catalog", HDFFile),
        # we use both the binned randoms for the case where we split
        # the lens catalog tomographically and the full version for
        # when we do the 2D stack of all the lenses together
        ("binned_random_catalog", HDFFile),
        ("random_cats", HDFFile),
        ("shear_photoz_stack", QPNOfZFile),
        ("lens_photoz_stack", QPNOfZFile),
        # ("tracer_metadata", HDFFile),
        ("fiducial_cosmology", FiducialCosmology),
    ]
    outputs = [("delta_sigma", SACCFile)]

    config_options = {
        "source_bins": StageParameter(list, [-1], msg="List of source bins to use (-1 means all)"),
        "lens_bins": StageParameter(list, [-1], msg="List of lens bins to use (-1 means all)"),
        "r_min": StageParameter(float, 0.1, msg="Minimum radius to use in Mpc"),
        "r_max": StageParameter(float, 10.0, msg="Maximum radius to use in Mpc"),
        "nbins": StageParameter(int, 10, msg="Number of radial bins"),
        "photoz": StageParameter(bool, False, msg="Whether or not objects have point photo-z estimate"),
        "lens_source_sep": StageParameter(float, 0.1, msg="Minimum redshift separation between lens and source bins to use for dSigma measurement"),
        "lower_bin_edge": StageParameter(list, [0.0, 0.358, 0.631, 0.872], msg="Lower edge of the source redshift bins to use for dSigma measurement, only used if photoz=True"),
        "source_cat_w_col": StageParameter(str, "weight", msg="Source catalog weight column name."),
        "lens_cat_w_col": StageParameter(str, "weight", msg="Lens catalog weight column name."),
    }

    def run(self):
        import dsigma
        import sacc

        bin_pairs = self.get_bin_pairs()
        source_bin = 3
        lens_bin = 0

        with self.open_input("fiducial_cosmology", wrapper=True) as cosmo_file:
            cosmo = cosmo_file.to_astropy()
        source_n_of_z = self.load_redshift_distribution("shear_photoz_stack")
        print(source_n_of_z['z'])

        # The lens n_of_z is only used when saving at the end
        lens_n_of_z = self.load_redshift_distribution("lens_photoz_stack")

        bins = np.logspace(np.log10(self.config["r_min"]), np.log10(self.config["r_max"]), self.config["nbins"])

        results = []
        last_source_bin = None

        # suppress numpy division warnings. Real scientists divide by zero
        # all the time.
        numpy_error_settings = np.seterr(divide="ignore", invalid="ignore")

        for source_bin, lens_bin in self.split_tasks_by_rank(bin_pairs):
            # Load the new source bin if changed
            if source_bin != last_source_bin:
                last_source_bin = source_bin
                source_table = self.load_source_table(source_bin)

            # Always reload the lens bins because we will add some
            # columns to them in-place in a minute.
            lens_table = self.load_lens_table(lens_bin)
            randoms_table = self.load_random_table(lens_bin)

            if self.config["photoz"]:
                # if we do not have point photo-z estimates for the sources then we use the lower edge of the tomographic bins to make z column
                source_table['z'] = np.array(self.config["lower_bin_edge"])[source_table['z_bin']]


            # Add columns to two tables in-place to do most of the pre-computation work.
            # We should look at using the n_jobs multiprocessing option here
            # but I don't know if it will play well with MPI on NERSC that we are
            # using to split the bin pairs across ranks
            print(
                f"Computing excess surface density for source = {source_bin}, lens = {lens_bin}, "
                f"with {len(source_table)} sources, {len(lens_table)} lenses, {len(randoms_table)} randoms"
            )
            print(np.amax(source_n_of_z['z'][source_n_of_z['n'][:, 0] > 0]))
            dsigma.precompute.precompute(lens_table, source_table, table_n=source_n_of_z, bins=bins, cosmology=cosmo, lens_source_cut=self.config["lens_source_sep"])
            dsigma.precompute.precompute(randoms_table, source_table, table_n=source_n_of_z, bins=bins, cosmology=cosmo, lens_source_cut=self.config["lens_source_sep"])

            # stack to get the excess surface density Delta(Sigma)
            result = dsigma.stacking.excess_surface_density(
                lens_table, table_r=randoms_table, random_subtraction=True, boost_correction=True, return_table=True
            )
            results.append((source_bin, lens_bin, result))

        # restore numpy settings
        np.seterr(**numpy_error_settings)

        # Collate results from different ranks and save to SACC file
        self.save_results(results, source_n_of_z, lens_n_of_z)

    def get_bin_pairs(self):
        """
        Determine the list of (source, lens) bin pairs to measure.

        Returns
        -------
        bin_pairs: list of tuples
            A list of (source_bin, lens_bin) pairs to measure.
        """

        # Get the number of bins of each type from the input files.
        with self.open_input("binned_shear_catalog") as shear_file:
            nbin_source = shear_file["shear"].attrs["nbin_source"]
        with self.open_input("binned_lens_catalog") as lens_file:
            nbin_lens = lens_file["lens"].attrs["nbin_lens"]

        # User can override the list of bins to do in the options.
        # but if they do not specify then we use everything
        source_bins = self.config["source_bins"]
        lens_bins = self.config["lens_bins"]

        if source_bins == [-1]:
            source_bins = list(range(nbin_source))
            source_bins.append("all")

        if lens_bins == [-1]:
            lens_bins = list(range(nbin_lens))
            lens_bins.append("all")

        # Collect all bin pairs together.
        # This is everything to be done by all processes.
        bin_pairs = []
        for s in source_bins:
            for l in lens_bins:
                bin_pairs.append((s, l))
        return bin_pairs

    def load_table(self, group, names):
        """
        Helper function to read tables from HDF5 into the astropy
        format that dSigma expects, with the right column names.

        Parameters
        ----------
        group: h5py.Group
            The group in the HDF5 file where the data is stored
        names: dict
            A mapping from the column names that dSigma expects to the column names in TXPipe.
        """
        from astropy.table import Table

        table = Table()
        for dsigma_name, txpipe_name in names.items():
            table[dsigma_name] = group[txpipe_name][:]
        return table

    def load_source_table(self, bin_index):
        """
        Load the lens table for a given bin index
        as an astropy table with the columns we need for dSigma.

        Parameters
        ----------
        bin_index: int or str
            The index of the source bin to load, or "all" for the 2D stack of all the objects
        """
        names = {
            "ra": "ra",
            "dec": "dec",
            "z": "z",
            "w": self.config["source_cat_w_col"],
            "e_1": "g1",
            "e_2": "g2",
        }
        with self.open_input("binned_shear_catalog") as shear_file:
            group = shear_file[f"shear/bin_{bin_index}"]
            if self.config["photoz"]:
                del names["z"]
            table = self.load_table(group, names)
            nbin = shear_file["shear"].attrs["nbin_source"]

        if bin_index == "all":
            table["z_bin"] = np.repeat(nbin, len(table))
        else:
            table["z_bin"] = np.repeat(bin_index, len(table))

        return table

    def load_redshift_distribution(self, file_tag):
        """
        Load the redshift distribution for a given source bin index as a tuple of (z, n(z)).

        dSigma wants all the redshift distributions for all the source bins together in a single table,
        so we load them all and check they are consistent.

        Parameters
        ----------
        file_tag: str
            The tag for the input file containing the redshift distribution

        Returns
        -------
        table: astropy.table.Table
            A table with columns 'z' and 'n'
        """
        from astropy.table import Table

        zs = []
        ns = []
        with self.open_input(file_tag, wrapper=True) as f:
            nbin = f.get_nbin()
            for i in range(nbin):
                z, n_of_z = f.get_bin_n_of_z(i)
                zs.append(z)
                ns.append(n_of_z)
            z, n_of_z = f.get_2d_n_of_z()
            zs.append(z)
            ns.append(n_of_z)

        # check all the z arrays are the same
        for i in range(1, len(zs)):
            if not np.allclose(zs[i], zs[0]):
                raise ValueError("Z arrays for different bins are not the same, cannot use for dSigma")
        # stack the n_of_z arrays into a 2D array of shape (nz, nbin)
        n_of_z = np.stack(ns, axis=0).T  # shape (nz, nbin)

        table = Table({"z": z, "n": n_of_z})
        return table

    def load_lens_table(self, bin_index):
        """
        Load the lens table for a given bin index
        as an astropy table with the columns we need for dSigma.

        Parameters
        ----------
        bin_index: int or str
            The index of the lens bin to load, or "all" for the 2D stack of all the objects

        Returns
        -------
        table: astropy.table.Table
            A table in the format that dSigma expects for lens samples
        """
        names = {
            "ra": "ra",
            "dec": "dec",
            "z": "z",
            "w_sys": self.config["lens_cat_w_col"],
        }
        with self.open_input("binned_lens_catalog") as lens_file:
            group = lens_file[f"lens/bin_{bin_index}"]
            table = self.load_table(group, names)

        return table

    def load_random_table(self, bin_index):
        """
        Load the randoms table for a given bin index
        as an astropy table with the columns we need for dSigma.

        Parameters
        ----------
        bin_index: int or str
            The index of the randoms bin to load, or "all" for the 2D stack of all the objects

        Returns
        -------
        table: astropy.table.Table
            A table in the format that dSigma expects for lens samples
        """
        names = {
            "ra": "ra",
            "dec": "dec",
            "z": "z",
        }

        if bin_index == "all":
            with self.open_input("random_cats") as random_file:
                group = random_file["randoms"]
                table = self.load_table(group, names)
        else:
            with self.open_input("binned_random_catalog") as random_file:
                group = random_file[f"randoms/bin_{bin_index}"]
                table = self.load_table(group, names)

        # randoms should be constructed to have unit weights
        for col in list(table.colnames):
            table[col] = table[col].astype(np.float64)
        table["w_sys"] = np.ones(len(table))

        return table

    def save_results(self, results, shear_photoz_stack, lens_photoz_stack):
        import sacc

        if self.comm is not None:
            # Gather results from all ranks to the root rank
            results = self.comm.gather(results, root=0)

        # Only the root process saves the output file.
        if self.rank != 0:
            return

        # sort by the bin pairs to put the results in a deterministic order
        results = sorted(results, key=lambda x: (str(x[0]), str(x[1])))

        s = sacc.Sacc()

        # Create tracers for the source sample
        # as SACC objects
        z = shear_photoz_stack["z"]
        Nz = shear_photoz_stack["n"]
        nbin_source = Nz.shape[1]
        for i in range(nbin_source):
            if i == nbin_source - 1:
                i = "all"
            s.add_tracer("NZ", f"source_{i}", z, Nz)

        # Create tracers for the lens sample
        # as SACC objects
        z = lens_photoz_stack["z"]
        Nz = lens_photoz_stack["n"]
        nbin_lens = Nz.shape[1]
        for i in range(nbin_lens):
            if i == nbin_lens - 1:
                i = "all"

            s.add_tracer("NZ", f"lens_{i}", z, Nz)

        # for each bin pair's results, add all the
        # measurements in the output data table
        for source_bin, lens_bin, result in results:
            tracer1 = f"source_{source_bin}"
            tracer2 = f"lens_{lens_bin}"

            for row in result:
                # Add the data point and all the various tags
                s.add_data_point(
                    "galaxy_shearDensity_deltasigma",
                    (tracer1, tracer2),
                    row["ds"],  # Final corrected Delta(Sigma) measurement
                    rp=row["rp"],  # radius of the bin
                    rp_min=row["rp_min"],  # minimum radius of the bin
                    rp_max=row["rp_max"],  # maximum radius of the bin
                    raw_value=row["ds_raw"],  # the raw Delta(Sigma) measurement before boost correction
                    boost=row["b"],  # the boost factor that was applied to get the final Delta(Sigma)
                    random_subtraction=row["ds_r"],  # the contribution from the randoms that was subtracted off
                    n_pairs=row["n_pairs"],  # the number of pairs in the bin
                )

        # Add provenance and potentially other metadata stuff.
        provenance = self.gather_provenance()
        other_metadata = {"nbin_source": nbin_source - 1, "nbin_lens": nbin_lens - 1}

        SACCFile.add_metadata(s, provenance, other_metadata)
        output_filename = self.get_output("delta_sigma")
        # switch to HDF5 backend for sacc since metadata
        # handlind is better
        print("Saving results to ", output_filename)
        s.save_hdf5(output_filename, overwrite=True)


class TXDeltaSigmaPlots(PipelineStage):
    """Make plots of Delta Sigma results."""

    name = "TXDeltaSigmaPlots"
    inputs = [
        ("delta_sigma", SACCFile),
        ("fiducial_cosmology", FiducialCosmology),
    ]
    outputs = [
        ("delta_sigma_plot", PNGFile),
    ]
    config_options = {}

    def run(self):
        import sacc
        import matplotlib.pyplot as plt

        sacc_data = sacc.Sacc.load_hdf5(self.get_input("delta_sigma"))

        # Plot in theta coordinates
        nbin_source = sacc_data.metadata["nbin_source"]
        nbin_lens = sacc_data.metadata["nbin_lens"]

        # Plot in r coordinates
        nbin_source = sacc_data.metadata["nbin_source"]
        nbin_lens = sacc_data.metadata["nbin_lens"]
        with self.open_output("delta_sigma_plot", wrapper=True, figsize=(5 * nbin_lens, 4 * nbin_source)) as fig:
            axes = fig.file.subplots(nbin_source, nbin_lens, squeeze=False)
            for s in range(nbin_source):
                for l in range(nbin_lens):
                    axes[s, l].set_title(f"Source {s}, Lens {l}")
                    axes[s, l].set_xlabel("Radius [Mpc]")
                    axes[s, l].set_ylabel(r"$R \cdot \Delta \Sigma [M_\odot / pc^2]$")
                    axes[s, l].grid()
                    x = sacc_data.get_tag("rp", tracers=(f"source_{s}", f"lens_{l}"))
                    y = sacc_data.get_mean(tracers=(f"source_{s}", f"lens_{l}"))
                    axes[s, l].plot(x, y * np.array(x))
            plt.subplots_adjust(hspace=0.3, wspace=0.3)
