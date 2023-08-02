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
                        pdfs_chunk = qp_bin.gen_obj.pdfs
                    elif pdf_type == "interp":
                        pdfs_chunk = qp_bin.gen_obj.yvals
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






class TXTruePhotozStack(PipelineStage):
    """
    Make an ideal true source n(z) using true redshifts

    Fake an n(z) by histogramming the true redshift values for each object.
    """
    name = "TXTruePhotozStack"
    inputs = [("tomography_catalog", TomographyCatalog), ("catalog", HDFFile), ("weights_catalog", HDFFile)]
    outputs = [
        ("photoz_stack", QPFile),
    ]
    config_options = {
        "chunk_rows": 5000,  # number of rows to read at once
        "zmax": float,
        "nz": int,
        "weight_col": "weight",
        "redshift_group": str,
    }

    def run(self):
        import qp

        # Create the stack objects
        nz = self.config["nz"]
        zmax = self.config["zmax"]
        z = np.linspace(0, zmax, nz + 1)
        with self.open_input("tomography_catalog", wrapper=True) as f:
            nbin = f.read_nbin()

        if self.get_input("weights_catalog") == "none":
            weight_col = None
        else:
            weight_col = self.config["weight_col"]

        histograms = np.zeros((nbin + 1, nz))
        total_weight = np.zeros((nbin + 1))

        # So we just do a single loop through the pair of files.
        for (s, e, data) in self.data_iterator(weight_col):
            # Feed back on our progress
            print(f"Process {self.rank} read data chunk {s:,} - {e:,}")
            # Add data to the stacks
            self.stack_data(data, histograms, total_weight, zmax, weight_col)

        # Sum the stacks
        in_place_reduce(histograms, self.comm)
        in_place_reduce(total_weight, self.comm)

        # Normalize the histograms
        histograms /= total_weight[:, None]


        # only root saves output
        if self.rank == 0:
            # Create a qp object for the n(z) information that we have.
            q = qp.Ensemble(qp.hist, data={"bins":z, "pdfs":histograms})
            
            with self.open_output("photoz_stack", wrapper=True) as f:
                    f.write_ensemble(q)


    def data_iterator(self, weight_col):
        # This collects together matching inputs from the different
        # input files and returns an iterator to them which yields
        # start, end, data
        redshift_group = self.config["redshift_group"]

        # basic arguments to the iterator function
        input_spec = [
            "tomography_catalog", "tomography", ["bin"],
            "catalog", redshift_group, ["redshift_true"],
        ]

        # If we have weights, add them to the iterator
        if weight_col is not None:
            input_spec.extend(["weights_catalog", "/", weight_col])

        return self.combined_iterators(self.config["chunk_rows"], *input_spec)
    
    def stack_data(self, data, histograms, total_weight, zmax, weight_col):
        # Stack the data from a single chunk into the histograms
        # and total weight arrays
        z = data["redshift_true"]
        bin = data["bin"]

        if weight_col:
            weights = data[weight_col]
        else:
            weights = np.ones_like(z)

        # Sizes for the histogram
        nbin = total_weight.shape[0] - 1
        nz = histograms.shape[1]

        for i in range(nbin):
            sel = (bin == i)
            h = np.histogram(z[sel], bins=nz, range=(0, zmax), weights=weights[sel])[0]
            w = weights[sel].sum()
            histograms[i] += h
            total_weight[i] += w

            # The 2D bin
            histograms[nbin] += h
            total_weight[nbin] += w
        
        
        

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
        "label": "",
        "zmax": 3.0,
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
            ax.set_xlim(0, self.config["zmax"])
            for i in range(ensemble.npdf - 1):
                ensemble.plot(i, axes=ax, label=f"Bin {i}")

            ensemble.plot(ensemble.npdf - 1, axes=ax, label=f"Total", norm=ensemble.npdf - 1, linestyle='--')

            plt.legend()
            plt.title(label)
            plt.xlim(xmin=0)
            plt.xlabel("z")
            plt.ylabel("n(z)")

