from .base_stage import PipelineStage
from ceci.config import StageParameter
from .data_types import HDFFile, PickleFile, ShearCatalog, BinAssignments, TomographyCatalog
from .binning import read_training_data, build_tomographic_classifier, apply_classifier
from .calibration_utils import band_variants
import numpy as np

class TXTrainRandomForestTomography(PipelineStage):
    """Train a random forest classifier to assign source galaxies to tomographic bins based on their photometry.

    The trained classified is saved as a pickle file to be used by the selector stage(s).
    This was originally merged in with the selector stage, but it can become slow in situations where the spectroscopic 
    training sample is large, so it has been split out into a separate stage to assist with debugging and re-running
    the selection stages.

    At some point this might be replaced by RAIL stages.
    """
    name = "TXTrainRandomForestTomography"
    inputs = [
        ("spectroscopic_catalog", HDFFile),
    ]
    outputs = [
        ("shear_tomography_classifier", PickleFile),
    ]
    config_options = {
        "bands": StageParameter(list, ["r", "i", "z"], msg="Bands from the catalog to use for selection"),
        "spec_mag_column_format": StageParameter(str, "photometry/{band}", msg="Format string for spectroscopic magnitude columns"),
        "spec_redshift_column": StageParameter(str, "photometry/redshift", msg="Column name for spectroscopic redshifts"),
        "source_zbin_edges": StageParameter(list, required=True, msg="Redshift bin edges for source tomography"),
        "random_seed": StageParameter(int, 42, msg="Random seed for reproducibility"),
    }

    def run(self):
        bands = self.config['bands']

        # Load the data that we need
        with self.open_input("spectroscopic_catalog") as spec_file:
            training_data = read_training_data(
                spec_file,
                bands,
                self.config["spec_mag_column_format"],
                self.config["spec_redshift_column"],
            )

        # Build the actual classifier.
        classifier, features = build_tomographic_classifier(
            bands,
            training_data,
            self.config["source_zbin_edges"],
            self.config["random_seed"],
            self.comm,
        )

        # Save the classifier.
        with self.open_output("shear_tomography_classifier", wrapper=True) as outfile:
            pickle_data = {
                "classifier": classifier,
                "features": features,
                "bands": bands,
                "source_zbin_edges": self.config["source_zbin_edges"],
            }
            outfile.write(pickle_data)


class TXComputeSourceTomography(PipelineStage):
    """Base class for tomography.
    """
    name = "TXComputeSourceTomography"
    inputs = [
        ("shear_catalog", ShearCatalog),
    ]
    outputs = [
        ("shear_bin_assignments", BinAssignments)
    ]
    config_options = {
        "chunk_rows": StageParameter(int, 10_000, msg="Rows to read at once")
    }
    
    def run(self):
        # do some setup
        info = self.prepare_tomography()
        input_columns = self.choose_columns(info)
        chunk_rows = self.config['chunk_rows']

        # open the input and output files (the latter in parallel)
        f_in = self.open_input("shear_catalog", wrapper=True)
        f_out = self.open_output("shear_bin_assignments", parallel=True)

        self.save_metadata(f_out, info)

        prefixes = f_in.get_classifier_prefixes()
        suffices = f_in.get_classifier_suffixes()
        output_cols = f_in.get_tomography_bin_columns()
        
        for (pre, suf, out_col_name) in zip(prefixes, suffices, output_cols):
            n = f_in.file[f"shear/{pre}g1{suf}"].size
            out_col = f_out.create_dataset(out_col_name, shape=(n,), dtype=np.int16)

            # Determine the input column names for this group. Some names
            # that we might care about get qualified for metacal, such as
            # g1, mags, etc. If we have a redshift_true column that probably
            # won't be qualified. So we check them here.
            cols = []
            with self.open_input("shear_catalog") as f:
                for col in input_columns:
                    var_col = f"{pre}{col}{suf}"
                    if var_col in f:
                        cols.append(var_col)
                    else:
                        cols.append(f"{pre}{col}")

            for (s, e, data) in self.iterate_hdf(
                "shear_catalog",
                "shear",
                cols,
                chunk_rows,
                ):
                data = {
                    name.removeprefix(pre).removesuffix(suf): col
                    for name, col in data.items()
                }
                zbin = self.compute_tomography(data, info)
                out_col[s:e] = zbin

    def prepare_tomography(self):
        return {}

    def choose_columns(self):
        return []

    def save_metadata(self, output_file, info):
        pass


class TXApplyRandomForestSourceTomography(TXComputeSourceTomography):
    name = "TXApplyRandomForestSourceTomography"
    inputs = [
        ("shear_catalog", ShearCatalog),
        ("shear_tomography_classifier", PickleFile),
    ]
    config_options = {}

    def prepare_tomography(self):
        # Read the tomography classifier from file
        with self.open_input("shear_tomography_classifier", wrapper=True) as infile:
            pickle_data = infile.read()

        # This is all that is actually needed in this class
        info = {
            "classifier": pickle_data["classifier"],
            "features": pickle_data["features"],
            "bands": pickle_data["bands"],
            "bin_edges": pickle_data["source_zbin_edges"],
        }
        return info

    def choose_columns(self, info):
        bands = info["bands"]
        # we always use the simple catalog type here because we are selecting
        # the group name higher up
        return band_variants(bands, "mag", "mag_err", shear_catalog_type="simple")


    def compute_tomography(self, data, info):
        # Need to update apply_classifier to
        # not try to do every 
        return apply_classifier(
            info["classifier"],
            info["features"],
            info["bands"],
            data,
        )

    def save_metadata(self, output_file, info):
        output_file.attrs["bands"] = info["bands"]
        output_file.attrs["bin_edges"] = info["bin_edges"]
        output_file.attrs["nbin_source"] = len(info["bin_edges"]) - 1


class TXSimpleRedshiftSourceTomography(TXComputeSourceTomography):
    name = "TXSimpleRedshiftSourceTomography"
    inputs = [
        ("shear_catalog", ShearCatalog),
    ]
    config_options = {
        "redshift_col": StageParameter(str, "redshift_true", msg="Column name to use for simple redshift cut"),
        "source_zbin_edges": StageParameter(list, required=True, msg="Redshift bin edges for source tomography"),
    }

    def compute_tomography(self, data, info):
        col_name = self.config["redshift_col"]
        z = data[col_name]
        bin_edges = self.config["source_zbin_edges"]
        nbin = len(bin_edges) - 1
        tomo_bin = np.repeat(np.int16(-1), z.size)
        for i in range(nbin):
            zmin = bin_edges[i]
            zmax = bin_edges[i + 1]
            sel = (z >= zmin) & (z < zmax)
            tomo_bin[sel] = i
        return tomo_bin

    def choose_columns(self, info):
        return [self.config["redshift_col"]]

    def save_metadata(self, output_file, info):
        output_file.attrs["redshift_col"] = self.config["redshift_col"]
        output_file.attrs["bin_edges"] = self.config["source_zbin_edges"]
        output_file.attrs["nbin_source"] = len(self.config["source_zbin_edges"]) - 1
