import numpy as np

def iterate_source_catalog(stage, redshift_type, read_selection=False, read_tomo_bin=False):
    """
    Iterate through the shear catalog, loading the locations
    of them and whether they are assigned to any tomographic bin.
    No row is removed, so row k of a chunk starting at s is row s + k
    of the catalog.

    The options of redshift_type require the existence of:
    * redshift_type == "tomo_bins":
        stage.open_input("shear_tomography_catalog")
    * redshift_type == "ztrue":
        stage.open_input("shear_catalog", wrapper=True)
    * redshift_type in ("zmode", "zmean", "zmedian", "pdf"):
        stage.open_input("source_photoz_pdfs")
    * redshift_type == "catalog_redshift":
        the column catalog_redshift_column of the shear catalog (e.g. mean_z for AnaCal)
    For the other options the source_photoz_pdfs input is not read and can be "none".

    With read_selection, each chunk also has "selection" (whether the object passes
    the cuts in shear_selection), and with read_tomo_bin it has "bin" (its bin in
    shear_tomography_catalog, -1 if it is in none).
    """
    rows = stage.config["chunk_rows"]
    use_true_shear = stage.config["use_true_shear"]

    # where and what to read from the shear catalog
    with stage.open_input("shear_catalog", wrapper=True) as f:
        shear_group = "shear"
        shear_cols, rename = f.get_primary_catalog_names(true_shear=use_true_shear)
        prefix = f.get_primary_catalog_prefix()


    ##########################
    # redshift cols management
    ##########################

    # where and what to read rom the PZ catalog. This is in a QP
    # format where the mode and mean are stored in a file called
    # "ancil". The columns are called zmode and zmean.

    # to aggregate groups of data to be read not in shear_catalog
    z_info = []
    additional_data_info = {}

    if redshift_type in ("zmode", "zmean", "zmedian", "pdf") and stage.get_input("source_photoz_pdfs").lower() == "none":
        raise ValueError(f"redshift_type={redshift_type} needs the source_photoz_pdfs input, which is none")

    if redshift_type is None:
        print("redshift_type is None, no reshift information is being read.")
    elif redshift_type == "tomo_bins":
        # where and what to read from the tomography catalog.
        # We just want the values from the source bin. We will use
        # any selected object, so we just ask for bin >= 0.
        z_info += ["shear_tomography_catalog", "tomography", ["bin"]]

        with stage.open_input("shear_tomography_catalog") as f:
            tomo_attrs = f["tomography"].attrs
            additional_data_info["tomo_bin_edges"] = np.array(
                [(tomo_attrs[f"zmin_{i}"], tomo_attrs[f"zmax_{i}"]) for i in range(tomo_attrs["nbin"])]
            )
    # The truth redshifts are keps in the shear catalog not
    # the p(z) catalog. If we ask for that column we need to load it
    # here.

    elif redshift_type == "ztrue":
        with stage.open_input("shear_catalog", wrapper=True) as f:
            col = f.get_true_redshift_column()
            shear_cols.append(col)
            rename[col] = "redshift"

    # for PDF
    elif redshift_type == "pdf":
        z_info += ["source_photoz_pdfs", "data", ["yvals"]]
        rename["yvals"] = "pdf_pz"

        # we will also need the z axis values in this case
        with stage.open_input("source_photoz_pdfs") as f:
            # this data seems to be 1D in my QP file.
            # but that's the kind of thing they might change.
            additional_data_info["pdf_z"] = np.squeeze(f["/meta/xvals"][0])

    elif redshift_type in ("zmode", "zmean", "zmedian"):
        z_info += ["source_photoz_pdfs", "ancil", [redshift_type]]
        rename[redshift_type] = "redshift"

    # a redshift that comes with the shear catalog
    elif redshift_type == "catalog_redshift":
        col = f"{prefix}{stage.config['catalog_redshift_column']}"
        shear_cols.append(col)
        rename[col] = "redshift"
    else:
        raise ValueError(f"redshift_type={redshift_type} not valid!")

    if read_tomo_bin and redshift_type != "tomo_bins":
        z_info += ["shear_tomography_catalog", "tomography", ["bin"]]

    # The selection file has the groups of the shear catalog,
    # e.g. ns/selection for metadetect
    if read_selection:
        z_info += ["shear_selection", "shear", [f"{prefix}selection"]]
        rename[f"{prefix}selection"] = "selection"


    # Loop through all these input files simultaneously.
    for s, e, data in stage.combined_iterators(
            rows,
            "shear_catalog",
            shear_group,
            shear_cols,
            *z_info,
            parallel=True
    ):
        # For each chunk of data we also want to store the original
        # index.  This comes in useful later because we will cut
        # down here to just objects in the WL sample.
        data["original_index"] = np.arange(s, e, dtype=int)
        # give the shear columns a unified name, whether
        # they are metacal, metadetect, etc., also rename
        # zmean or zmode to "redshift"
        for old, new in rename.items():
            data[new] = data.pop(old)

        # add extra info that does not depend on the rows
        data.update(additional_data_info)

        # Give this chunk of data to the main run function
        yield s, e, data
