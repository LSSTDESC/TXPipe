import numpy as np

def iterate_source_catalog(stage, redshift_type, read_selection=False, read_tomo_bin=False, version=None):
    """
    Iterate through the shear catalog, loading the locations
    of them and whether they are assigned to any tomographic bin.
    No row is removed, so row k of a chunk starting at s is row s + k
    of the catalog.

    The options of redshift_type require the existence of:
    * redshift_type == "tomo_bins":
        stage.open_input("shear_bin_assignments")
    * redshift_type == "ztrue":
        stage.open_input("shear_catalog", wrapper=True)
    * redshift_type in ("zmode", "zmean", "zmedian", "pdf"):
        stage.open_input("source_photoz_pdfs")
    * redshift_type == "catalog_redshift":
        the column catalog_redshift_column of the shear catalog (e.g. mean_z for AnaCal)
    For the other options the source_photoz_pdfs input is not read and can be "none".

    With read_selection, each chunk also has "selection" (whether the object passes
    the cuts in shear_selection), and with read_tomo_bin it has "bin" (its bin in
    shear_bin_assignments, -1 if it is in none). The bin does not include the cuts of the
    selection: the sample of the tomographic bins is the objects with selection and bin >= 0.

    version is None for the main catalog (ns for metadetect), or a sheared version of it
    (see CLClusterSourceSelect.sheared_versions), with the prefix and suffix of its columns
    and its bin column in shear_bin_assignments. For a version only the positions are read
    from the shear catalog, with its own selection, its own bin (from shear_bin_assignments,
    -1 if it is in none) and its own catalog_redshift column. The photo-z of
    source_photoz_pdfs and the true redshift are those of the rows of the main catalog.
    """
    rows = stage.config["chunk_rows"]
    use_true_shear = stage.config["use_true_shear"]

    # where and what to read from the shear catalog
    with stage.open_input("shear_catalog", wrapper=True) as f:
        shear_group = "shear"
        shear_cols, rename = f.get_primary_catalog_names(true_shear=use_true_shear)
        prefix = f.get_primary_catalog_prefix()
        # the bin column of the main catalog in shear_bin_assignments, e.g. ns/bin for metadetect
        main_bin = f.get_classifier_output_columns()[0]
    suffix = ""
    # where to read the tomographic bin
    tomo_bin_info = ["shear_bin_assignments", "/", [main_bin]]
    tomo_bin_rename = {main_bin: "bin"}
    if version is not None:
        # a sheared version: its own positions, cuts, bin and redshift
        prefix, suffix = version["prefix"], version["suffix"]
        shear_cols = [f"{prefix}ra", f"{prefix}dec"]
        rename = {col: col[len(prefix):] for col in shear_cols}
        tomo_bin_info = ["shear_bin_assignments", "/", [version["bin"]]]
        tomo_bin_rename = {version["bin"]: "bin"}


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
        # The edges of the tomographic bins, (zmin, zmax) of each bin.
        # The bin of each object is read below
        with stage.open_input("shear_bin_assignments") as f:
            edges = f.attrs["bin_edges"]
            additional_data_info["tomo_bin_edges"] = np.column_stack([edges[:-1], edges[1:]])
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
        col = f"{prefix}{stage.config['catalog_redshift_column']}{suffix}"
        shear_cols.append(col)
        rename[col] = "redshift"
    else:
        raise ValueError(f"redshift_type={redshift_type} not valid!")

    if read_tomo_bin or redshift_type == "tomo_bins":
        z_info += tomo_bin_info
        rename.update(tomo_bin_rename)

    # The selection file has the groups of the shear catalog,
    # e.g. ns/selection for metadetect, and the suffixes of its versions, e.g. selection_1p
    if read_selection:
        z_info += ["shear_selection", "shear", [f"{prefix}selection{suffix}"]]
        rename[f"{prefix}selection{suffix}"] = "selection"


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
