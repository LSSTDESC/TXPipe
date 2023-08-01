from .base_stage import PipelineStage
from .data_types import PhotozPDFFile, TomographyCatalog, HDFFile, PNGFile, NOfZFile, ShearCatalog, QPFile
from .utils.mpi_utils import in_place_reduce
from .utils import rename_iterated
import numpy as np
import warnings


class TXPhotozStack(PipelineStage):
    """
    Naive stacker using QP
    
    """
    name = "TXPhotozStack"
    inputs = [
        ("photoz_pdfs", QPFile),
        ("tomography_catalog", TomographyCatalog),
        ('weights_catalog', HDFFile),
    ]
    outputs = [
        ("photoz_stack", QPFile),
    ]
    config_options = {
        "chunk_rows": 5000,
        "tomo_name": "source",
        "weight_col": "shear/00/weight",
        "zmax": 0.0, # zmax and nz to use if these are not specified in the PDFs input file
        "nz": 0,
    }

    def run(self):
        import qp
        qp_filename = self.get_input("photoz_pdfs")
        tomo_cat = self.open_input("tomography_catalog")

        if self.get_input("weights_catalog") == "none":
            weight_cat = None
        else:
            weight_cat = self.open_input("weights_catalog")
            weight_col = self.config["weight_col"]

        tomo_name = self.config["tomo_name"]
        chunk_rows = self.config["chunk_rows"]

        with self.open_input("tomography_catalog", wrapper=True) as f:
            nbin = f.read_nbin(tomo_name)


        with self.open_input("photoz_pdfs", wrapper=True) as f:
            z = f.get_z()
            pdf_type = f.get_qp_pdf_type()
            nz = z.size if pdf_type == "interp" else z.size - 1
            if pdf_type == "hist":
                nz = z.size - 1
            elif pdf_type == "interp":
                nz = z.size
            else:
                raise ValueError(f"TXPipe cannot yet use QP PDF type {pdf_type}")


        pdfs = np.zeros((nbin, nz))
        total_weight = np.zeros(nbin)
        with self.open_input("photoz_pdfs", wrapper=True) as f:
            for start, end, qp_chunk in f.iterate(chunk_rows, rank=self.rank, size=self.size):
                print(f"Rank {self.rank} stacking PDFs {start} to {end}")
                bins = tomo_cat[f"tomography/{tomo_name}_bin"][start:end]
                if weight_cat is None:
                    weights = None
                else:
                    weights = weight_cat[weight_col][start:end]

                for i in range(nbin):
                    sel = bins == i
                    qp_bin = qp_chunk[sel]

                    # This is a bit of a hack - we want to leave the QP
                    # object in whichever form it naturally came, so we
                    # query it internally and stack either the PDFs or yvals.
                    if pdf_type == "hist":
                        pdfs_chunk = qp_bin._gen_obj.pdfs
                    elif pdf_type == "interp":
                        pdfs_chunk = qp_bin._gen_obj.yvals
                    else:
                        raise ValueError(f"TXPipe cannot yet use QP PDF type {pdf_type}")

                    if weights is None:
                        pdfs[i] += pdfs_chunk.sum(axis=0)
                        total_weight[i] += qp_bin.npdf
                    else:
                        # This is not yet tested!
                        pdfs[i] += weights[sel] @ pdfs_chunk
                        total_weight[i] += weights[sel].sum()
        
        # Collect the results from all processors
        in_place_reduce(pdfs, self.comm)
        in_place_reduce(total_weight, self.comm)
        
        # only root saves output
        if self.rank == 0:
            print("pdfs", pdfs)
            pdfs /= total_weight[:, None]

            # Create a qp object for the n(z) information that we have.
            if pdf_type == "interp":
                q = qp.Ensemble(qp.interp, data={"xvals":z, "yvals":pdfs})
            elif pdf_type == "hist":
                q = qp.Ensemble(qp.hist, data={"bins":z, "pdfs":pdfs.T})
            else:
                raise ValueError(f"TXPipe cannot yet use QP PDF type {pdf_type}")

            q.write_to(self.get_output("photoz_stack"))





class TXPhotozPlots(PipelineStage):
    """
    Make n(z) plots of source and lens galaxies
    """
    parallel = False
    name = "TXPhotozPlots"
    inputs = [("photoz_stack", QPFile)]
    outputs = [
        ("nz_plot", PNGFile),
    ]

    config_options = {
        "label": ""
    }

    def run(self):
        import matplotlib
        matplotlib.use("agg")
        import matplotlib.pyplot as plt

        with self.open_input("photoz_stack", wrapper=True) as f:
            ensemble = f.ensemble()

        label = self.config["label"]
        if label:
            label = f"{label} n(z)"
        else:
            label = "n(z)"

        with self.open_output("nz_plot", wrapper=True) as fig:
            ax = fig.file.gca()
            for i in range(ensemble.npdf):
                ensemble.plot(i, axes=ax, label=f"Bin {i}")
            plt.legend(frameon=False)
            plt.title(label)
            plt.xlim(xmin=0)
            plt.xlabel("Redshift")


class TXTrueNumberDensity(PipelineStage):
    """
    Make an ideal true source n(z) using true redshifts

    Fake an n(z) by histogramming the true redshift values for each object.
    """
    name = "TXTrueNumberDensity"
    inputs = [("tomography_catalog", HDFFile), ("catalog", HDFFile), ("weights_catalog", HDFFile)]
    outputs = [
        ("photoz_stack", QPFile),
    ]
    config_options = {
        "chunk_rows": 5000,  # number of rows to read at once
        "zmax": float,
        "nz": int,
        "tomo_group": "source",
    }

    def run(self):

        # Create the stack objects
        with self.open_input("tomograph_catalog", wrapper=True) as f:
            nbin = f.read_nbin(tomo_name)


        histograms = np.zeros((nbin + 1, nz))
        total_weight = np.zeros((nbin + 1))
        

        # So we just do a single loop through the pair of files.
        for (s, e, data) in self.data_iterator():
            # Feed back on our progress
            print(f"Process {self.rank} read data chunk {s:,} - {e:,}")
            # Add data to the stacks
            self.stack_data("source", data, outputs)
        # Save the stacks
        self.write_outputs("shear_photoz_stack", outputs)



class TXSourceTrueNumberDensity(PipelineStage):
    """
    Make an ideal true source n(z) using true redshifts

    Fake an n(z) by histogramming the true redshift values for each object.
    Uses the same method as its parent but loads the data
    differently and uses the truth redshift as a delta function PDF
    """

    name = "TXSourceTrueNumberDensity"
    inputs = [
        ("shear_catalog", ShearCatalog),
        ("shear_tomography_catalog", TomographyCatalog),
    ]
    outputs = [
        ("shear_photoz_stack", NOfZFile),
    ]
    config_options = {
        "chunk_rows": 5000,  # number of rows to read at once
        "zmax": float,
        "nz": int,
    }

    def data_iterator(self):
        with self.open_input("shear_catalog", wrapper=True) as f:
            group = f.get_primary_catalog_group()
        return self.combined_iterators(
            self.config["chunk_rows"],
            "shear_catalog",  # tag of input file to iterate through
            group,  # data group within file to look at
            ["redshift_true"],  # column(s) to read
            "shear_tomography_catalog",  # tag of input file to iterate through
            "tomography",  # data group within file to look at
            ["source_bin"],  # column(s) to read
        )

    def stack_data(self, name, data, outputs):
        stack, stack2d = outputs
        stack.add_delta_function(data[f"{name}_bin"], data["redshift_true"])
        bin2d = data[f"{name}_bin"].clip(-1, 0)
        stack2d.add_delta_function(bin2d, data["redshift_true"])

    def get_metadata(self):
        # Check we are running on a photo file with redshift_true
        with self.open_input("shear_catalog", wrapper=True) as f:
            group = f.get_primary_catalog_group()
            has_z = "redshift_true" in f.file[group].keys()
            if not has_z:
                msg = (
                    "The shear_catalog file you supplied does not have a redshift_true column. "
                    "If you're running on sims you need to make sure to ingest that column from GCR. "
                    "If you're running on real data then sadly this isn't going to work. "
                    "Use a different stacking stage."
                )
                raise ValueError(msg)

        zmax = self.config["zmax"]
        nz = self.config["nz"]
        z = np.linspace(0, zmax, nz)

        shear_tomo_file = self.open_input("shear_tomography_catalog")
        nbin_source = shear_tomo_file["tomography"].attrs["nbin"]
        shear_tomo_file.close()

        return z, nbin_source


class TXLensTrueNumberDensity(TXSourceTrueNumberDensity):
    """
    Make an ideal true lens n(z) using true redshifts

    This inherits from two parent classes, which can be confusing.
    """
    name = "TXLensTrueNumberDensity"
    inputs = [
        ("photometry_catalog", HDFFile),
        ("lens_tomography_catalog", TomographyCatalog),
    ]
    outputs = [
        ("lens_photoz_stack", NOfZFile),
    ]
    config_options = {
        "chunk_rows": 5000,  # number of rows to read at once
        "zmax": float,
        "nz": int,
    }

    def data_iterator(self):
        # This collects together matching inputs from the different
        # input files and returns an iterator to them which yields
        # start, end, data
        return self.combined_iterators(
            self.config["chunk_rows"],
            "photometry_catalog",  # tag of input file to iterate through
            "photometry",  # data group within file to look at
            ["redshift_true"],  # column(s) to read
            "lens_tomography_catalog",  # tag of input file to iterate through
            "tomography",  # data group within file to look at
            ["lens_bin"],  # column(s) to read
        )

    def get_metadata(self):
        zmax = self.config["zmax"]
        nz = self.config["nz"]
        z = np.linspace(0, zmax, nz)
        # Save again but for the number of bins in the tomography catalog
        with self.open_input("lens_tomography_catalog") as tomo_file:
            nbin_lens = tomo_file["tomography"].attrs["nbin_lens"]

        return z, nbin_lens
