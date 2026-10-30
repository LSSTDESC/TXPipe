import numpy as np

def iterate_source_catalog(self, redshift_type):
    """
    Iterate through the shear catalog, loading the locations
    of them and whether they are assigned to any tomographic bin.

    The options of redshift_type require the existence of:
    * redshift_type == "tomo_bins":
        self.open_input("shear_tomography_catalog")
    * redshift_type == "ztrue":
        self.open_input("shear_catalog", wrapper=True)
    * redshift_type in ("zmode", "zmean", "median", "pdf"):
        self.open_input("source_photoz_pdfs")
    """
    rows = self.config["chunk_rows"]
    use_true_shear = self.config["use_true_shear"]

    # where and what to read from the shear catalog
    with self.open_input("shear_catalog", wrapper=True) as f:
        shear_group = "shear"
        shear_cols, rename = f.get_primary_catalog_names(true_shear=use_true_shear)

    # load the shear calibration information
    # for the moment, load the average overall shear calibrator,
    # that applies to the collective 2D bin. This won't be high
    # accuracy, but I'm not clear right now exactly what the right
    # model is yet. Here this calibrator is just used for the
    # weight information calculation, which needs g1 and g2 estimates.
    # The same calibration should be used later for the actual shears.
    if self.rank == 0:
        print("Using single 2D shear calibration!")


    ##########################
    # redshift cols management
    ##########################

    # where and what to read rom the PZ catalog. This is in a QP
    # format where the mode and mean are stored in a file called
    # "ancil". The columns are called zmode and zmean.

    # to aggregate groups of redshift data to be read not in shear_catalog
    z_info = []
    additional_data_info = {}

    if redshift_type is None:
        print("redshift_type is None, no reshift information is being read.")
    elif redshift_type == "tomo_bins":
        # where and what to read from the tomography catalog.
        # We just want the values from the source bin. We will use
        # any selected object, so we just ask for bin >= 0.
        z_info += ["shear_tomography_catalog", "tomography", ["bin"]]

        with self.open_input("shear_tomography_catalog") as f:
            tomo_attrs = f["tomography"].attrs
            additional_data_info["tomo_bin_edges"] = np.array(
                [(tomo_attrs[f"zmin_{i}"], tomo_attrs[f"zmax_{i}"]) for i in range(tomo_attrs["nbin"])]
            )
    # The truth redshifts are keps in the shear catalog not
    # the p(z) catalog. If we ask for that column we need to load it
    # here.

    elif redshift_type == "ztrue":
        with self.open_input("shear_catalog", wrapper=True) as f:
            col = f.get_true_redshift_column()
            shear_cols.append(col)
            rename[col] = "redshift"

    # for PDF
    elif redshift_type == "pdf":
        z_info += ["source_photoz_pdfs", "data", ["yvals"]]
        rename["yvals"] = "pdf_pz"

        # we will also need the z axis values in this case
        with self.open_input("source_photoz_pdfs") as f:
            # this data seems to be 1D in my QP file.
            # but that's the kind of thing they might change.
            additional_data_info["pdf_z"] = np.squeeze(f["/meta/xvals"][0])

    else:
        redshift_type_ok = False
        # check which point source estimate is added
        for z_type in ("zmode", "zmean", "zmedian"):
            if z_type == redshift_type:
                z_info += ["source_photoz_pdfs", "ancil", [z_type]]
                rename[z_type] = "redshift"
                redshift_type_ok = True
                break
        if not redshift_type_ok:
            raise ValueError(f"redshift_type={redshift_type} not valid!")


    # Loop through all these input files simultaneously.
    for s, e, data in self.combined_iterators(
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

        ##################################
        # This is Eduardo's problem now ;)
        ##################################
        # Apply the shear calibration to this sample.
        # Optionally subtract the mean (of the whole WL sample, not the local mean)
        #_, shear_cal = Calibrator.load(self.get_input("shear_tomography_catalog"), null=use_true_shear)
        #data["g1"], data["g2"] = shear_cal.apply(data["g1"],
        #                                         data["g2"],
        #                                         subtract_mean=subtract_mean
        #)


        # Give this chunk of data to the main run function
        yield s, e, data
